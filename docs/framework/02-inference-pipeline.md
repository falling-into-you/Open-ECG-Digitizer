# 02 — 推理 Pipeline 架构

## 2.1 入口：`src/digitize.py`

推理通过命令行启动：

```bash
python3 -m src.digitize --config src/config/inference_wrapper.yml
# 支持配置覆盖：
python3 -m src.digitize --config src/config/inference_wrapper.yml DATA.output_path=my_output
```

### 执行流程

```python
# 1. 解析命令行参数 (argparse)
#    --config: YAML 配置路径
#    overrides: KEY=VALUE 形式的配置覆盖

# 2. 加载配置
config_path = find_config_path(args.config)  # 在 . 和 src/config/ 中搜索
cfg = get_cfg(config_path)                    # yacs CfgNode

# 3. 动态实例化 InferenceWrapper
inference_wrapper_class = import_class_from_path(config.MODEL.class_path)
inference_wrapper = inference_wrapper_class(**config.MODEL.KWARGS)

# 4. 遍历图像文件夹
for file_path in get_candidate_file_paths(config):
    process_one_file(file_path, config, inference_wrapper, save_mode)

# 5. process_one_file:
#    - decode_and_prepare_image() → [1, 3, H, W] 张量
#    - inference_wrapper(image) → got_values dict
#    - save_outputs() → CSV + PNG + 元数据
```

### 动态类加载机制

项目通过 `import_class_from_path()` 实现完全动态的模块加载：

```python
# src/utils.py
def import_class_from_path(path: str) -> Any:
    module = importlib.import_module(".".join(path.split(".")[:-1]))
    return getattr(module, path.split(".")[-1])

# 示例: "src.model.inference_wrapper.InferenceWrapper"
#  → import src.model.inference_wrapper
#  → getattr(module, "InferenceWrapper")
```

这意味着 Pipeline 的每个组件都可以通过修改 YAML 配置来替换，无需修改代码。

---

## 2.2 InferenceWrapper — 主编排器

**文件**: `src/model/inference_wrapper.py`
**类**: `InferenceWrapper(torch.nn.Module)`

### 初始化

```
InferenceWrapper.__init__(config, device, ...)
    │
    ├── _load_signal_extractor()      → SignalExtractor
    ├── _load_perspective_detector()   → PerspectiveDetector
    ├── _load_segmentation_model()     → UNet (+ 加载权重 + .eval())
    ├── _load_cropper()                → Cropper
    ├── _load_pixel_size_finder()      → PixelSizeFinder
    ├── _load_dewarper()               → Dewarper
    └── _load_layout_identifier()      → LeadIdentifier (+ 小 UNet + 布局 YAML)
```

### forward() 完整流程（8 个阶段）

```
┌───────────────────────────────────────────────────────────┐
│  输入: image [1, 3, H, W]                                 │
└─────────────────────┬─────────────────────────────────────┘
                      │
     ┌────────────────┴────────────────┐
     │  阶段 1: 预处理与重采样          │
     │  min_max_normalize()             │
     │  _resample_image()               │
     │  · 最大边缩放到 3000px           │
     │  · 最小边不低于 512px            │
     │  · 支持 rotate_on_resample       │
     └────────────────┬────────────────┘
                      │
     ┌────────────────┴────────────────┐
     │  阶段 2: UNet 语义分割           │
     │  _get_feature_maps(image)        │
     │  · logits = segmentation_model() │
     │  · softmax → 4 类概率图          │
     │  · 提取 signal/grid/text 3 个    │
     │  · process_sparse_prob():        │
     │    去均值 → clamp(min=0) → 归一化│
     └────────────────┬────────────────┘
                      │ signal_prob, grid_prob, text_prob
     ┌────────────────┴────────────────┐
     │  阶段 3: 透视检测                │
     │  perspective_detector(grid_prob) │
     │  · 二值化 (98% 分位阈值)         │
     │  · 两遍 Hough 变换              │
     │  → alignment_params dict        │
     │    {rho_min/max,                 │
     │     theta_min/max_horizontal,    │
     │     theta_min/max_vertical}      │
     └────────────────┬────────────────┘
                      │
     ┌────────────────┴────────────────┐
     │  阶段 4: 裁剪与透视校正          │
     │  cropper(signal_prob, params)    │
     │  → source_points [4, 2]         │
     │                                  │
     │  apply_perspective() × 4 次:     │
     │  · aligned_image                 │
     │  · aligned_signal_prob           │
     │  · aligned_grid_prob             │
     │  · aligned_text_prob             │
     │                                  │
     │  + _rotate_on_resample() 可选    │
     │  + _crop_y() 去除空白边缘        │
     └────────────────┬────────────────┘
                      │
     ┌────────────────┴────────────────┐
     │  阶段 5: 网格尺寸估计            │
     │  pixel_size_finder(grid_prob)    │
     │  → mm_per_pixel_x, mm_per_pixel_y│
     │  → avg_pixel_per_mm (两方向均值) │
     └────────────────┬────────────────┘
                      │
     ┌────────────────┴────────────────┐
     │  阶段 6: 去弯曲 (可选)           │
     │  if apply_dewarping:             │
     │    dewarper.fit(grid, px/mm)     │
     │    dewarper.transform(signal)    │
     │  默认关闭 (apply_dewarping=false)│
     └────────────────┬────────────────┘
                      │
     ┌────────────────┴────────────────┐
     │  阶段 7: 信号提取                │
     │  signal_extractor(signal_prob)   │
     │  → signals [N_lines, W]          │
     │  · 迭代提取 (最多 4 次)          │
     │  · 线段合并 (匈牙利算法)         │
     └────────────────┬────────────────┘
                      │
     ┌────────────────┴────────────────┐
     │  阶段 8: 导联识别与归一化        │
     │  identifier(signals, text_prob,  │
     │             avg_px/mm)           │
     │  · 小 UNet 检测导联标签位置      │
     │  · 与布局模板匹配               │
     │  · canonicalize_lines():         │
     │    重排为标准 12 导联顺序        │
     │  · normalize():                  │
     │    pixels → µV, 插值到 5000 点  │
     └────────────────┬────────────────┘
                      │
     ┌────────────────┴────────────────┐
     │  输出 dict:                      │
     │  · layout_name: str              │
     │  · input_image: Tensor           │
     │  · aligned: {image, signal/grid/ │
     │              text prob}          │
     │  · signal: {raw_lines,           │
     │    canonical_lines [12, 5000],   │
     │    layout_matching_cost}         │
     │  · pixel_spacing_mm: {x, y}     │
     │  · source_points: [4, 2]        │
     └─────────────────────────────────┘
```

---

## 2.3 输出保存

`save_outputs()` 根据 `save_mode` 保存不同内容：

| save_mode | 输出内容 |
|-----------|---------|
| `"all"` | CSV + PNG + 元数据 |
| `"timeseries_only"` | 仅 CSV |
| `"png_only"` | 仅 PNG |

### CSV 格式

- 文件名: `{stem}_timeseries_canonical.csv`
- 列: `I,II,III,aVR,aVL,aVF,V1,V2,V3,V4,V5,V6`
- 行: 5000 个采样点（可配置 `target_num_samples`）
- 单位: **微伏 (µV)**

### PNG 可视化

四宫格图 (2×2)：
1. 左上: 原始图像 + 裁剪四边形红点
2. 右上: 透视校正后图像 + 网格叠加
3. 左下: 信号概率图
4. 右下: 提取的 12 导联波形

### 元数据

- 文件: `digitization_metadata.csv`（追加模式）
- 列: `file_path, matching_cost, is_flipped, lead_layout`

---

## 2.4 性能计时

设置 `enable_timing: true` 后，Pipeline 会打印各阶段耗时：

```
 Timing results:
    Initial resampling    0.12 s
    Segmentation          0.45 s
    Perspective detection 0.08 s
    Cropping              0.03 s
    Feature map resampling 0.15 s
    Pixel size search     0.22 s
    Dewarping             0.00 s
    Signal extraction     0.31 s
Total time: 1.36 s
```
