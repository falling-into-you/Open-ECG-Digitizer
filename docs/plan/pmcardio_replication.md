# 复现 PMcardio 后处理流程 — 实施计划

**目标**：逐步修改后处理阶段，复现 PMcardio 的效果优势。每一步用量化指标（PCC / RMSE / SNR）衡量变化，用可视化图片直观对比。

**测试图片**：`ecg_images/40792771-0.png` (2200 x 1700, RGBA)

---

## 第一阶段：环境搭建与基准

### 1.1 创建新推理入口

- 复制 `src/digitize.py` → `src/digitize_pmcardio.py`
- 结构保持一致，后续逐步替换内部后处理模块
- 配置文件：`src/config/inference_wrapper_pmcardio.yml`（复制自 `inference_wrapper.yml`）
- 输出目录：`sandbox/inference_output_pmcardio/`

### 1.2 测试脚本

`shells/infer_dig.sh` 更新为：
1. 运行原版 → 输出到 `sandbox/inference_output_baseline/`
2. 运行改进版 → 输出到 `sandbox/inference_output_pmcardio/`
3. 运行对比脚本 → 输出到 `sandbox/comparison_results/`

### 1.3 量化对比模块

创建 `src/metrics/comparison.py`，实现：

| 指标 | 公式 | 含义 |
|------|------|------|
| Pearson 相关系数 (PCC) | `corr(baseline, improved)` | 波形形状相似度，越接近 1 越好 |
| 均方根误差 (RMSE) | `sqrt(mean((baseline - improved)^2))` | 绝对偏差，单位 µV，越低越好 |
| 信噪比 (SNR) | `10 * log10(signal_power / noise_power)` | 信号清晰度，单位 dB，越高越好 |

输出：
- 逐导联指标 CSV
- 并排波形对比 PNG（12 导联，上下两行分别是原版和改进版）
- 终端打印 summary 表格（方便我看到量化结果）

### 1.4 基准测试

运行一次原版流程，记录：
- 12 导联 canonical_lines CSV
- 四宫格可视化 PNG
- 各阶段耗时

---

## 第二阶段：逐步改进

**策略**：每个改进独立修改、独立测试、观察效果后决定是否保留。

### 改进 2.1：基线估计（高优先级）

**现状**：`lead_identifier.py:normalize()` 中用 `nanmean` 去基线
**问题**：均值对 QRS 高尖峰敏感，基线可能被拉偏
**改进**：
- 用 `nanmedian` 替代 `nanmean`（鲁棒估计，抗尖峰干扰）
- 或用滑动窗口中位数实现局部基线估计
**修改文件**：`src/digitize_pmcardio.py` 中对应的 normalize 流程
**观测**：各导联基线偏移量变化、PCC/RMSE 变化

### 改进 2.2：导联列分割（中优先级）

**现状**：`signal_extractor.py` 用连通组件标记 + 分 4 条带重标记
**问题**：条带数固定为 4，对非标准布局可能切割不当
**改进**：
- 先沿 Y 轴对概率图求和得到行投影，用峰值检测确定导联行位置
- 根据检测到的行数动态划分条带，而非固定 4 条
- 增加相邻条带间的重叠区域处理
**修改文件**：`src/digitize_pmcardio.py` 中的信号提取流程
**观测**：导联列边界处误连接是否减少

### 改进 2.3：端点连接与路径延拓（中优先级）

**现状**：匈牙利算法匹配线段，代价函数 = 距离 x 高度差惩罚
**问题**：缺少沿轨迹方向的外推，断裂处可能匹配到错误线段
**改进**：
- 在匈牙利匹配前，对每条线段的左右端点计算局部斜率（取最近 N 个有效点的线性拟合）
- 代价函数中增加**方向一致性惩罚**：端点斜率差异越大，代价越高
- 对短 gap（< 阈值像素），先尝试线性插值补全
**修改文件**：`src/digitize_pmcardio.py` 中的线段合并逻辑
**观测**：断裂导联数量、合并后连续性

### 改进 2.4：TPS 去畸变（中-低优先级）

**现状**：`dewarper.py` 已实现但默认关闭（`apply_dewarping: false`）
**改进**：
- 在 `inference_wrapper_pmcardio.yml` 中设为 `true`
- 观察效果，若不理想则调整参数（`abs_peak_threshold`、`max_num_warp_points`、`optimizer_steps`）
**修改文件**：`src/config/inference_wrapper_pmcardio.yml`
**观测**：网格对齐效果、信号提取精度

### 改进 2.5：重叠导联处理（低优先级）

**现状**：迭代提取最多 4 轮 + 水平路径追踪分离
**问题**：V3-V5 高振幅导联重叠仍难以正确分离
**改进**：
- 在重叠区域，引入概率图加权的多候选路径评分
- 对交叠区域，用基线约束和轨迹连续性联合判断归属
- 增加迭代轮数或改进拒绝区域的处理策略
**修改文件**：`src/digitize_pmcardio.py` 中的迭代提取逻辑
**观测**：V3-V5 的 PCC/RMSE/SNR 变化

---

## 第三阶段：工作流程

```
对于每个改进:
  1. 修改 src/digitize_pmcardio.py（或配置文件）
  2. 运行: bash shells/infer_dig.sh
  3. 查看终端输出的量化指标表格
  4. 查看 sandbox/comparison_results/ 下的对比 PNG
  5. 决定: 保留 / 回滚 / 继续调整
```

---

## 关键文件清单

| 文件 | 角色 |
|------|------|
| `src/digitize.py` | 原版入口（不动） |
| `src/digitize_pmcardio.py` | **新建** — 改进版入口 |
| `src/config/inference_wrapper_pmcardio.yml` | **新建** — 改进版配置 |
| `src/metrics/comparison.py` | **新建** — 量化对比工具 |
| `shells/infer_dig.sh` | **修改** — 同时运行两版并对比 |
| `ecg_images/40792771-0.png` | 测试图片 |
| `sandbox/inference_output_baseline/` | 原版输出 |
| `sandbox/inference_output_pmcardio/` | 改进版输出 |
| `sandbox/comparison_results/` | 对比结果 |
