# 07 — 推理阶段后处理详解

本文档聚焦于 **U-Net 语义分割之后** 的所有后处理步骤。U-Net 输出 4 通道概率图后，Pipeline 通过 6 个后处理阶段将像素级概率转换为最终的 12 导联微伏时间序列。

---

## 7.1 后处理总览

```
U-Net 输出: softmax 4 通道概率图
    │
    ├── signal_prob (信号线)
    ├── grid_prob   (网格线)
    └── text_prob   (文本标注)
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  后处理阶段 1: 稀疏概率处理 (process_sparse_prob)            │
│  后处理阶段 2: 透视检测 (PerspectiveDetector)                │
│  后处理阶段 3: ROI 裁切与透视校正 (Cropper)                  │
│  后处理阶段 4: 网格间距估计 (PixelSizeFinder)                │
│  后处理阶段 5: TPS 去畸变 (Dewarper, 默认关闭)              │
│  后处理阶段 6: 信号提取 (SignalExtractor)                    │
│  后处理阶段 7: 导联识别与归一化 (LeadIdentifier)             │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
最终输出: canonical_lines [12, 5000] (µV)
```

---

## 7.2 阶段 1: 稀疏概率处理

**位置**: `src/model/inference_wrapper.py:284-288` (`process_sparse_prob`)

U-Net softmax 输出的概率图通常是稀疏的（大部分区域概率很低），直接使用会导致后续处理的噪声问题。此步骤对 signal_prob、grid_prob、text_prob 三个概率图分别做相同处理：

```python
def process_sparse_prob(self, signal_prob: Tensor) -> Tensor:
    signal_prob = signal_prob - signal_prob.mean()   # 1. 减去全局均值
    signal_prob = torch.clamp(signal_prob, min=0)    # 2. 裁剪负值为 0
    signal_prob = signal_prob / (signal_prob.max() + 1e-9)  # 3. 归一化到 [0, 1]
    return signal_prob
```

**设计意图**:
- **去均值**: 消除低概率背景噪声，只保留高于平均水平的区域
- **裁剪负值**: 概率不应为负数
- **归一化**: 统一尺度，使后续模块不受绝对概率值影响

---

## 7.3 阶段 2: 透视检测

**位置**: `src/model/perspective_detector.py`
**输入**: `grid_prob [1, 1, H, W]`
**输出**: `alignment_params dict` — 包含网格线的角度和距离参数

### 算法: 两遍 Hough 变换

此阶段利用 ECG 网格纸的规则网格线来检测拍摄时的透视畸变角度。

#### 步骤 1: 二值化
```python
threshold = quantile(image, 0.98)  # 取 98% 分位数作为阈值
binary_image = (image > threshold).float()
```
仅保留概率最高的 2% 像素，对应最明显的网格线位置。

#### 步骤 2: Hough 变换投票

对二值图中每个非零像素 `(x, y)`，在所有候选角度 θ 下计算 ρ：
```
ρ = x·cos(θ) + y·sin(θ)
```
累加器 `accumulator[ρ, θ]` 计数每对 (ρ, θ) 的投票数。

**性能优化**: 若非零像素超过 100,000 个，随机抽取子集避免内存爆炸。

#### 步骤 3: 两遍搜索策略

**第一遍 (粗搜索)**:
- 水平线角度: θ ∈ [-π/4, π/4]，125 个等分采样
- 垂直线角度: θ ∈ [π/4, 3π/4]，125 个等分采样
- 应用 Hann 窗加权（`pow(0.25)` 软化），增强中心频率、抑制边缘
- 计算方差矩阵 `variance[W, W]`，找到方差最大的 (θ_top, θ_bottom) 对
  - 高方差 = 周期性强 = 网格线方向

**第二遍 (精搜索)**:
- 在第一遍结果附近 ±1° 范围内重复搜索
- 得到最终的精确角度参数

#### 输出参数
```python
{
    "rho_min": Tensor,              # 距离参数最小值
    "rho_max": Tensor,              # 距离参数最大值
    "theta_min_vertical": Tensor,   # 垂直网格线最小角度
    "theta_max_vertical": Tensor,   # 垂直网格线最大角度
    "theta_min_horizontal": Tensor, # 水平网格线最小角度
    "theta_max_horizontal": Tensor, # 水平网格线最大角度
}
```

---

## 7.4 阶段 3: ROI 裁切与透视校正

**位置**: `src/model/cropper.py` + `inference_wrapper.py:172-222`
**输入**: `signal_prob`, `alignment_params` (来自阶段 2)
**输出**: 对齐后的 `aligned_image`, `aligned_signal_prob`, `aligned_grid_prob`, `aligned_text_prob`

### 3a. 计算源点 (Cropper.forward)

**配置参数**:
| 参数 | 默认值 | 含义 |
|------|--------|------|
| `granularity` | 80 | 将图像沿 Hough 线方向分成多少条带 |
| `percentiles` | [0.02, 0.98] | 信号累积分布的裁剪百分位 |
| `alpha` | 0.85 | 透视变换平滑因子 |

**流程**:
1. 在 Hough 空间中均匀采样 ρ 和 θ，划分条带
2. 计算每个条带的信号概率累积和
3. 按百分位数 [0.02, 0.98] 确定上下/左右边界
4. 求 4 对 Hough 线的交点 → 四边形角点 `source_points [4, 2]`
   ```
   交点公式:
   线1: x·sin(θ₁) + y·cos(θ₁) = ρ₁
   线2: x·sin(θ₂) + y·cos(θ₂) = ρ₂
   
   det = cos(θ₁)·sin(θ₂) - cos(θ₂)·sin(θ₁)
   x = (ρ₁·sin(θ₂) - ρ₂·sin(θ₁)) / det
   y = (ρ₂·cos(θ₁) - ρ₁·cos(θ₂)) / det
   ```
5. 角点顺序: [top-left, top-right, bottom-right, bottom-left]

### 3b. 透视变换 (Cropper.apply_perspective)

对 image、signal_prob、grid_prob、text_prob 四个张量分别应用同一透视变换：
- 计算目标矩形，保持宽高比
- `alpha=0.85` 意味着目标点不完全对准图像边缘，而是向中心收缩 15%，避免边缘伪影
- 使用 `torchvision.transforms.functional.perspective()` 执行变换

### 3c. 可选旋转 (_rotate_on_resample)

若 `rotate_on_resample=true` 且对齐后图像 H > W（竖屏状态），自动逆时针旋转 90°:
```python
if aligned_image.shape[2] > aligned_image.shape[3]:
    aligned_image = torch.rot90(aligned_image, k=3, dims=(2, 3))  # 270° = 逆时针 90°
```

### 3d. Y 方向裁切 (_crop_y)

移除对齐后图像上下无信号的空白区域：
- 将 signal_prob 和 grid_prob 相加，按行求和
- 减去均值并裁剪负值
- 找到第一个和最后一个非零行 → 裁切边界

---

## 7.5 阶段 4: 网格间距估计

**位置**: `src/model/pixel_size_finder.py`
**输入**: `aligned_grid_prob [1, 1, H, W]`
**输出**: `mm_per_pixel_x`, `mm_per_pixel_y`, `avg_pixel_per_mm`

### 算法: 自相关 + 多级缩放网格搜索

ECG 标准纸张的网格间距为 **5mm (大格) / 1mm (小格)**。该模块利用这一先验，通过自相关找到网格的像素周期。

#### 步骤 1: 一维化
```python
col_sum = grid_prob.sum(dim=-1)  # 按列求和 → 一维信号 [H]
col_sum = col_sum - col_sum.mean()  # 去直流分量
```

#### 步骤 2: 自相关
```python
autocorrelation = np.correlate(col_sum, col_sum, mode="full")
```
自相关在周期等于网格间距处出现峰值。

#### 步骤 3: 缩放网格搜索

**配置参数**:
| 参数 | 默认值 | 含义 |
|------|--------|------|
| `mm_between_grid_lines` | 5 | ECG 标准大格间距 (mm) |
| `samples` | 1000 | 每轮搜索的候选数 |
| `min_number_of_grid_lines` | 30 | 最少期望网格线数 |
| `max_number_of_grid_lines` | 70 | 最多期望网格线数 |
| `max_zoom` | 10 | 缩放迭代次数 |
| `zoom_factor` | 10.0 | 每次缩小搜索范围的倍率 |
| `lower_grid_line_factor` | 0.3 | 小格线 (1mm) 的权重 |

**搜索流程 (每个方向独立)**:
```
初始搜索范围: [图像长度/max_grid_lines, 图像长度/min_grid_lines]

for zoom in range(10):
    对 1000 个均匀候选间距 pxl:
        1. 按 pxl 间距生成理想网格模板:
           - 每 pxl 像素一条大格线 (权重=1.0)
           - 每 pxl/5 像素一条小格线 (权重=0.3)
        2. 距离衰减加权 (远处权重低)
        3. 得分 = Σ(autocorrelation × template)
        4. 取得分最高的间距
    
    缩小范围: best ± (max-min)/zoom_factor
```

#### 步骤 4: 计算 mm/pixel
```python
mm_per_pixel = mm_between_grid_lines / best_pixel_distance  # = 5 / pxl
```

X、Y 方向分别独立计算，最终求平均:
```python
avg_pixel_per_mm = (1/mm_per_pixel_x + 1/mm_per_pixel_y) / 2
```

---

## 7.6 阶段 5: TPS 去畸变（默认关闭）

**位置**: `src/model/dewarper.py`
**配置**: `apply_dewarping: false` (默认不启用)
**输入**: `aligned_grid_prob`, `avg_pixel_per_mm`
**输出**: 去畸变后的 `aligned_signal_prob`

适用于弯曲或折叠的 ECG 纸张。算法分 4 步：

### 步骤 1: 球谐卷积检测网格交点

生成 4 重对称球谐核 `cos(4φ) × exp(-r²/2σ²)`，核大小 = `10 × pixels_per_mm`，对网格概率图卷积，然后 `peak_local_max()` 提取局部最大值作为网格交点候选。

### 步骤 2: KNN 过滤 + 图构建

- KNN(k=5) 为每个交点找 4 个邻居
- **方向一致性过滤**: 4 个邻居的向量和应接近 0（正交十字分布）
- **幅度过滤**: 向量和的范数不应过大
- 构建图，取最大连通分量

### 步骤 3: 梯度优化

将网格节点位置作为可训练参数，Adam 优化 1000 步:
```
Loss = sqrt(mean((邻居最大距离 - target_grid_size)² + 邻居最小距离²))
```
目标: 让邻居间距等于 `5 × pixels_per_mm`，且方向正交。

### 步骤 4: TPS 拟合与变换

- 从优化前/后的位置中采样最多 75 个控制点
- 拟合薄板样条 (TPS)
- 用 `grid_sample()` 对 signal_prob 应用去畸变

---

## 7.7 阶段 6: 信号提取

**位置**: `src/model/signal_extractor.py`
**输入**: `aligned_signal_prob [H, W]` (squeeze 后的 2D 概率图)
**输出**: `signals [N_lines, W]` — 每行代表一条信号线的 y 坐标轨迹

### 6a. 迭代提取 (最多 4 轮)

每轮包含:

1. **连通组件标记**: `skimage.measure.label()` 对二值化（阈值=0.1）后的概率图标记连通区域

2. **分条带重标记**: 将图像水平分成 4 条带，分别重新标记，防止来自不同行的信号被误连为同一区域

3. **面积过滤**: 概率和 < `threshold_sum=10` 的小区域被丢弃

4. **加权质心提取**: 对每个连通区域，用概率加权求每列的 y 坐标质心:
   ```python
   line[col] = Σ(y × prob[y, col]) / Σ(prob[y, col])
   ```

5. **线分类**: 检查线在 mask 内的比例，≥ 95% 为有效线，否则为拒绝线

6. **迭代优化**: 对拒绝区域进行水平路径追踪分离重叠信号，更新概率图后重新提取

### 6b. 线段合并 (匈牙利算法)

提取出的线段可能是同一条信号的断裂片段，需要合并:

1. **代价矩阵构建**:
   ```
   cost(i→j) = (|Δx| + |Δy|) × (1 + 30 × |高度差|)
   ```
   - Δx: 线段 i 的右端点到线段 j 的左端点的水平距离
   - Δy: 端点 y 坐标差
   - 高度差惩罚: 不同行高度的线段匹配代价高 30 倍
   - **环绕处理**: `min(|Δx|, W - |Δx|) × λ`，支持 rhythm strip 跨边界合并

2. **匈牙利算法**: `scipy.optimize.linear_sum_assignment` 找最优匹配

3. **连通分量合并**: 构建匹配图，同一连通分量的线段合并为一条

4. **长度过滤**: 有效点数 < W/5 的线段被丢弃

### 6c. 自动行数检测

```python
# softmax → 概率分布
# 统计每列的 "零→非零" 过渡次数
# 取众数作为预期行数 (num_peaks)
```

---

## 7.8 阶段 7: 导联识别与归一化

**位置**: `src/model/lead_identifier.py`
**输入**: `signals [N, W]`, `aligned_text_prob [1, 1, H, W]`, `avg_pixel_per_mm`
**输出**: `canonical_lines [12, 5000]` (µV)

这是后处理的最后也是最复杂的阶段，将原始信号线重排为标准 12 导联并转换为物理单位。

### 7a. 合并非重叠线

相邻行若在 x 方向无重叠，则合并为一行（处理某些布局中同行多导联不重叠的情况）。

### 7b. 像素到微伏转换 (normalize)

```python
# 1. 去均值
lines -= lines.nanmean(dim=1, keepdim=True)

# 2. 单位转换: 像素 → µV
# ECG 标准: 0.1 mV/mm = 100 µV/mm
scale = (mv_per_mm / avg_pixel_per_mm) * 1000  # µV/pixel
lines *= scale

# 3. 翻转极性 (ECG 信号正方向向上，图像 y 轴向下)
lines = -lines

# 4. 裁剪无效区域
# 要求每列至少 3 条线有有效值 (非 NaN)
# 找到首尾有效列，裁剪

# 5. 重采样到 5000 点
lines = F.interpolate(lines, size=5000, mode="linear")
```

### 7c. 导联文本检测

使用一个小型专用 U-Net (1 通道输入 → 13 通道输出) 检测文本概率图中的导联标签:

```python
logits = lead_unet(text_prob)   # [1, 13, H, W]
probs = softmax(logits, dim=1)[:, :12]  # 取前 12 个导联类别
probs[:, 0] = 0  # 忽略 "I" 导联 (因为字符简单，误检率高)
probs[probs < 0.8] = 0  # 阈值过滤
```

提取每个检测到的导联标签的质心坐标 `(name, x_com, y_com)`。

### 7d. 布局模板匹配

遍历所有预定义布局模板（定义在 `src/config/lead_layouts_all.yml`），找最佳匹配:

```python
for layout_name, desc in layouts.items():
    for flip in [False, True]:  # 可选翻转检测
        # 1. 生成模板网格位置 (归一化到 [0,1])
        # 2. 最小二乘估计 缩放s + 平移t:
        #    G = s × P + t
        # 3. 计算残差:
        #    residual = Σ|检测点 - 模板点| + 缺失导联惩罚 (0.5/个)
        # 4. 保留代价最小的布局
```

还包含 **Cabrera 检测**: 用伪逆矩阵验证 Cabrera 导联排列关系，`cos_sim > 0.992` 则限制搜索范围为 Cabrera 布局。

### 7e. 标准化排列 (_canonicalize_lines)

按匹配到的布局定义，将信号线重排为标准 12 导联顺序:
```python
LEAD_CHANNEL_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF",
                       "V1", "V2", "V3", "V4", "V5", "V6"]
```

- **翻转处理**: 若检测为翻转布局，旋转 180° 并反转幅度
- **负号处理**: 某些布局中导联名前缀 `-` 表示极性反转
- **Rhythm 导联**: 通过余弦相似度 + 匈牙利算法匹配到对应标准导联
  - 常见 rhythm 导联: II, V1, V5 (加权提升匹配概率)

---

## 7.9 最终输出

### InferenceWrapper 返回值

```python
{
    "layout_name": str,                    # 匹配的布局名称
    "input_image": Tensor[1, 3, H, W],    # 预处理后的输入图像

    "aligned": {
        "image": Tensor[1, 3, H', W'],       # 透视校正后的图像
        "signal_prob": Tensor[1, 1, H', W'], # 对齐后的信号概率图
        "grid_prob": Tensor[1, 1, H', W'],   # 对齐后的网格概率图
        "text_prob": Tensor[1, 1, H', W'],   # 对齐后的文本概率图
    },

    "signal": {
        "raw_lines": Tensor[N, W'],           # 原始提取的信号线 (像素坐标)
        "canonical_lines": Tensor[12, 5000],  # 标准 12 导联信号 (µV)
        "lines": Tensor[N, W'],               # 归一化后的信号线 (µV)
        "layout_matching_cost": float,         # 布局匹配代价
        "layout_is_flipped": str,              # 是否翻转 ("True"/"False")
    },

    "pixel_spacing_mm": {
        "x": float,                  # x 方向 mm/pixel
        "y": float,                  # y 方向 mm/pixel
        "average_pixel_per_mm": float,  # 平均 pixel/mm
    },

    "source_points": Tensor[4, 2],    # 透视变换的四角源点坐标
}
```

### 文件输出 (由 digitize.py 处理)

| 文件 | 格式 | 内容 |
|------|------|------|
| `*_timeseries_canonical.csv` | CSV | 12 列 (I~V6) × 5000 行，单位 µV |
| `*.png` | PNG | 四宫格可视化 (原图+校正+概率图+波形) |
| `digitization_metadata.csv` | CSV | 匹配代价、翻转状态、布局名称 |

---

## 7.10 关键数值参数汇总

| 阶段 | 参数 | 值 | 说明 |
|------|------|------|------|
| 稀疏处理 | — | — | 减均值 → clamp(0) → 归一化 |
| 透视检测 | `num_thetas` | 250 | Hough 角度采样总数 (水平125+垂直125) |
| 裁切 | `granularity` | 80 | 条带划分数 |
| 裁切 | `percentiles` | [0.02, 0.98] | 信号边界百分位 |
| 裁切 | `alpha` | 0.85 | 目标点内缩比例 |
| 网格估计 | `min/max_grid_lines` | 30 / 70 | 期望网格线范围 |
| 网格估计 | `lower_grid_line_factor` | 0.3 | 小格线权重 |
| 网格估计 | `max_zoom` | 10 | 搜索迭代数 |
| 去畸变 | `abs_peak_threshold` | 0.1 | 交点检测阈值 |
| 去畸变 | `max_num_warp_points` | 75 | TPS 控制点上限 |
| 去畸变 | `optimizer_steps` | 1000 | 梯度优化步数 |
| 信号提取 | `label_thresh` | 0.1 | 二值化阈值 |
| 信号提取 | `threshold_line_in_mask` | 0.95 | 线有效性判定比例 |
| 信号提取 | `max_iterations` | 4 | 迭代提取最大轮数 |
| 信号提取 | `split_num_stripes` | 4 | 分条带数 |
| 导联识别 | `target_num_samples` | 5000 | 输出采样点数 |
| 导联识别 | `mv_per_mm` | 0.1 | ECG 标准 0.1 mV/mm |
| 导联识别 | `threshold` | 0.8 | 导联文本检测阈值 |
| 导联识别 | `possibly_flipped` | false | 是否检测翻转布局 |
