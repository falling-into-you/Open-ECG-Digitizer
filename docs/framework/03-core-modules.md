# 03 — 核心模块详解

本文档详细介绍 Pipeline 中 7 个核心模块的设计与实现。

---

## 3.1 UNet 语义分割 (`model/unet.py`)

### 架构概述

自定义 U-Net 网络，用于将 ECG 图像像素级分类为 4 类。

```
输入: [B, 3, H, W]  (RGB)
输出: [B, 4, H, W]  (logits)

4 类语义:
  0 = grid       (网格线)
  1 = text_bg    (文字/背景标注区)
  2 = signal     (ECG 信号线)
  3 = background (纯背景)
```

### 网络结构

```
编码器 (8 级):
  dims = [32, 64, 128, 256, 320, 320, 320, 320]
  每级 depth=2 层 Conv-InstanceNorm-LeakyReLU
  下采样: stride-2 卷积 (非池化)
  跳跃连接: 1×1 卷积 (残差相加，非拼接)

解码器 (7 级):
  双线性插值上采样 → concat(skip) → 2 层解码块

最终层:
  Conv2d(32 → 4, kernel=3, padding=1)
```

### 关键设计决策

| 设计 | 选择 | 原因 |
|------|------|------|
| 归一化 | InstanceNorm (affine=True) | 比 BatchNorm 更适合变化大的图像 |
| 激活 | LeakyReLU(0.01) | 避免 dying ReLU |
| 下采样 | stride-2 卷积 | 比池化保留更多信息 |
| 跳跃连接 | 1×1 卷积残差相加 | 比 concat 更参数高效 |
| padding | replicate | 减少边缘效应 |

### 代码核心

```python
# 编码器前向传播
for encoder, skip, down in zip(self.encoders, self.encoder_skips, self.encoder_downscaling):
    x = encoder(x) + skip(x)  # 残差跳跃连接
    skips.append(x)
    x = down(x)               # stride-2 下采样

# 解码器前向传播
for i, (decoder, skip) in enumerate(zip(self.decoders, self.decoder_skips)):
    x = self._upsample(x, skips[i + 1])                    # 双线性插值
    x = decoder(torch.cat([x, skips[i + 1]], dim=1))        # concat + 解码
```

---

## 3.2 透视检测 (`model/perspective_detector.py`)

### 功能

检测 ECG 网格纸的透视畸变参数，支持最大约 45° 的旋转校正。

### 算法流程

```
grid_prob → binarize(98%分位阈值) → binary_image

┌─── 第一遍 Hough 变换 (粗搜索) ────────────────────────┐
│  θ ∈ [-π/4, π/4] ∪ [π/4, 3π/4]  (各 125 个采样)       │
│  Hann 窗加权消除边缘效应                                │
│  计算累加器方差矩阵 variance[W, W]                      │
│  → 找到方差最大的 θ_top, θ_bottom (水平+垂直)           │
└───────────────────────────────┬──────────────────────┘
                                │
┌───────────────────────────────┴──────────────────────┐
│  第二遍 Hough 变换 (精搜索)                            │
│  在第一遍结果附近 ±eps (默认 1°) 精细搜索               │
│  → 最终 θ 和 ρ 参数                                   │
└───────────────────────────────┬──────────────────────┘
                                │
                                ▼
         输出: {rho_min, rho_max,
                theta_min/max_horizontal,
                theta_min/max_vertical}
```

### 关键算法: Hough 变换

```python
# 极坐标参数方程: x·cos(θ) + y·sin(θ) = ρ
rhos_vals = x_idxs * cos_thetas + y_idxs * sin_thetas

# 累加器: 每个 (ρ, θ) 对的投票数
accumulator.index_add_(0, idxs_flat, ones)

# 方差最大化: 在累加器上计算对角线方差
# 对所有 (θ_top, θ_bottom) 组合，沿从 top 到 bottom 的线采样
# 方差最大的组合即为最优角度
```

### 性能优化

- 随机采样：若非零像素 > 100,000，随机抽取子集
- Hann 窗加权：`pow(0.25)` 减少频谱泄漏
- 两遍策略：粗搜索 250 个角度 → 精搜索缩小范围

---

## 3.3 裁剪与透视校正 (`model/cropper.py`)

### 功能

基于信号概率图确定感兴趣区域的四边形，然后应用透视变换。

### 算法流程

```
signal_prob + alignment_params
    │
    ├── 1. 初始化参数
    │   · ρ: linspace(rho_max, rho_min, granularity=50)
    │   · θ_h, θ_v: 从 params 插值
    │
    ├── 2. 水平边界检测
    │   · 用 Hough 线将图像划分为 50 个水平条带
    │   · 计算每个条带的信号概率累积和
    │   · 百分位数 [0.02, 0.98] 确定上下边界
    │
    ├── 3. 垂直边界检测 (同理)
    │
    ├── 4. 计算四角源点
    │   · 4 对 Hough 线的交点
    │   · 返回 [top-left, top-right, bottom-right, bottom-left]
    │
    └── 5. 透视变换
        · 计算目标矩形 (保持宽高比)
        · torchvision.transforms.functional.perspective()
        · alpha 参数控制校正强度 (默认 0.85)
```

### 关键公式: Hough 线交点

```
线 1: x·sin(θ₁) + y·cos(θ₁) = ρ₁
线 2: x·sin(θ₂) + y·cos(θ₂) = ρ₂

解:
x = (ρ₁·sin(θ₂) - ρ₂·sin(θ₁)) / det
y = (ρ₂·cos(θ₁) - ρ₁·cos(θ₂)) / det
det = cos(θ₁)·sin(θ₂) - cos(θ₂)·sin(θ₁)
```

---

## 3.4 网格尺寸估计 (`model/pixel_size_finder.py`)

### 功能

从对齐后的网格概率图，估算 x 和 y 方向的 mm/pixel 比例。

### 算法: 自相关 + 多级缩放网格搜索

```
aligned_grid_prob
    │
    ├── 对 x 方向: 按列求和 → 自相关
    ├── 对 y 方向: 按行求和 → 自相关 (考虑纵横比)
    │
    └── 多级搜索 (每个方向独立):
        for zoom in range(10):
            · 在 [min_dist, max_dist] 范围内均匀采样 1000 个候选间距
            · 对每个候选间距，生成模拟网格模板:
              - 主线 (每 5mm): 权重 1.0
              - 副线 (每 1mm): 权重 0.3~0.5
            · 计算模板与自相关的点积得分
            · 取得分最高的间距
            · 缩小搜索范围: 当前最优 ± range/zoom_factor
        │
        └── 返回: mm_between_grid_lines / best_pixels = mm/pixel
```

### 参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `mm_between_grid_lines` | 5 | 标准 ECG 网格间距 (5mm) |
| `samples` | 1000 | 每次搜索的采样数 |
| `min_number_of_grid_lines` | 15~30 | 最少网格线数 |
| `max_number_of_grid_lines` | 70~120 | 最多网格线数 |
| `max_zoom` | 10 | 缩放迭代次数 |
| `lower_grid_line_factor` | 0.3~0.5 | 副线权重 |

---

## 3.5 信号提取 (`model/signal_extractor.py`)

### 功能

从对齐后的信号概率图中提取 ECG 波形轨迹。

### 算法: 迭代提取 + 线段合并

```
aligned_signal_prob [H, W]
    │
    ├── 阶段 A: 迭代提取 (最多 4 次)
    │   for it in range(max_iterations):
    │     ├── 1. 连通区域标记 (skimage.label)
    │     ├── 2. 分条带重新标记 (4条带，避免跨条连接)
    │     ├── 3. 对每个区域:
    │     │   ├── 加权质心法提取线条轨迹
    │     │   └── 分类: 线在 mask 内比例 ≥ 95% → 有效
    │     ├── 4. 有效线 → good[], 拒绝线 → rejected[]
    │     └── 5. 对拒绝区域: 水平路径追踪 → 分离重叠信号 → 更新 fmap
    │
    ├── 阶段 B: 线段匹配与合并
    │   ├── 1. 提取端点坐标 (xmin, xmax, ymin, ymax)
    │   ├── 2. 构建代价矩阵:
    │   │   · delta_x (含环绕处理) + delta_y + 高度差惩罚
    │   ├── 3. 匈牙利算法最优匹配
    │   ├── 4. 构建匹配图 → 连通分量
    │   └── 5. 合并同一连通分量的线段
    │
    └── 阶段 C: 自动峰值计数
        · softmax → 概率分布
        · 统计每列的零→非零过渡数
        · 众数作为预期行数
        · 过滤过短线段 (< W/5)
```

### 线段分类标准

```python
def _classify_line(self, line, mask):
    # 线条的位置在 mask 内的比例 ≥ 95% 为有效
    return in_mask[cols].float().mean() >= 0.95
```

---

## 3.6 导联识别 (`model/lead_identifier.py`)

### 功能

识别 ECG 的导联布局，将提取的信号线重排为标准 12 导联顺序。

### 算法: 双 UNet + 模板匹配

```
signals [N_lines, W] + text_prob [1, 1, H, W]
    │
    ├── 1. 合并非重叠线 (_merge_nonoverlapping_lines)
    ├── 2. 归一化: pixels → µV
    │   · lines -= nanmean; lines *= (mv_per_mm / avg_px_per_mm) * 1000
    │   · 裁剪无效区域 (NaN 过多的列)
    │   · 插值到 target_num_samples (默认 5000)
    │
    ├── 3. 导联标签检测
    │   · 小 UNet(1ch→13ch) 处理 text_prob
    │   · softmax → 取前 12 通道 (忽略 "I" 导联，误检率高)
    │   · 阈值过滤 (> 0.8)
    │   · 提取每个导联的质心坐标 (x_com, y_com)
    │
    ├── 4. 布局模板匹配 (_match_layout)
    │   for each layout in layouts:
    │     for flip in [False, True]:  # 翻转检测
    │       · 生成模板网格位置 (归一化 0~1)
    │       · 缩放+平移对齐检测到的点与模板点
    │       · 计算残差代价 (含缺失惩罚 0.5)
    │       · 保留代价最小的布局
    │
    ├── 5. Cabrera 检测 (可选)
    │   · 伪逆矩阵验证 Cabrera 关系
    │   · cos_sim > 0.992 → 限制为 Cabrera 布局
    │
    └── 6. 标准化输出 (_canonicalize_lines)
        · 按布局定义重排为 [I, II, III, aVR, aVL, aVF, V1-V6]
        · 处理翻转: flip(lines, dims=[0,1]); max_val - lines
        · Rhythm 导联: 余弦相似度 + 匈牙利算法匹配
        · 常见 rhythm: II, V1, V5 (加权提升匹配概率)
```

### 输出格式

```python
{
    "layout": "standard_3x4_with_r1",   # 布局名称
    "flip": False,                       # 是否翻转
    "cost": 0.123,                       # 匹配代价
    "canonical_lines": Tensor[12, 5000], # 标准 12 导联
    "lines": Tensor[N, W],              # 原始线段
    "rows_in_layout": 4,                # 行数
    "n_detected": 8,                    # 检测到的导联数
}
```

---

## 3.7 去弯曲 — 实验性 (`model/dewarper.py`)

### 功能

处理弯曲或折叠的 ECG 纸张（默认关闭）。

### 算法: 球谐卷积 → KNN图 → 梯度优化 → TPS

```
grid_prob [H, W] + pixels_per_mm
    │
    ├── 1. 球谐卷积核
    │   · 4 重对称 (kernel_m=4) 的方向性核
    │   · 高斯包络限制范围
    │   · 尺寸 = 10 * pixels_per_mm
    │
    ├── 2. 卷积 + 峰值检测
    │   · Conv2d(grid_prob, kernel) → 响应图
    │   · peak_local_max() → 网格交叉点候选
    │
    ├── 3. KNN 过滤 + 图构建
    │   · KNN(k=5) 构建邻域
    │   · 方向性过滤: cos_sim 产物 ≥ 0.95
    │   · 幅度过滤: 向量和范数 < 0.95
    │   · 取最大连通分量
    │
    ├── 4. 梯度优化 (1000 步)
    │   · 目标: 相邻节点距离 = target_grid_size
    │   · Adam(lr=1.0, decay=0.999)
    │   · Loss = sqrt(mean((max_diff - target)² + min_diff²))
    │
    └── 5. TPS 拟合
        · 原始位置 → 优化后位置
        · ThinPlateSpline.fit()
        · grid_sample() 变形
```

### 注意事项

- **默认关闭**: 配置中 `apply_dewarping: false`
- 适用于弯曲/折叠纸张，不建议用于平整纸张
- 最多使用 75 个控制点（性能考虑）
