# Open ECG Digitizer — 项目架构文档

> 版本: v1.9.2 | Python 3.12+ | 论文: npj Digital Medicine (arXiv: 2510.19590)

## 文档索引

| 文档 | 内容 |
|------|------|
| [README.md](./README.md) | 本文件 — 项目概览与快速导航 |
| [01-project-overview.md](./01-project-overview.md) | 项目简介、目录结构、依赖关系 |
| [02-inference-pipeline.md](./02-inference-pipeline.md) | 推理 Pipeline 完整流程与数据流 |
| [03-core-modules.md](./03-core-modules.md) | 7 个核心模块的详细设计与实现 |
| [04-training-system.md](./04-training-system.md) | 训练系统、损失函数、数据增强 |
| [05-configuration.md](./05-configuration.md) | 配置系统、YAML 文件说明、导联布局 |
| [06-data-flow.md](./06-data-flow.md) | 端到端数据流、I/O 格式、评估指标 |
| [07-post-processing.md](./07-post-processing.md) | 推理阶段后处理详解 (U-Net 之后的 7 个处理步骤) |

## 一句话概述

Open ECG Digitizer 是一个**高度可配置的 12 导联心电图数字化工具**，能够从扫描图像或手机拍摄的照片中，通过语义分割 → 透视校正 → 网格估计 → 信号提取 → 导联识别的多阶段 Pipeline，提取原始时间序列数据（输出单位：µV）。

## 架构全景图

```
┌─────────────────────────────────────────────────────────────────┐
│                        ECG 图像输入                              │
│                  (扫描 / 手机照片 / JPEG/PNG)                    │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   InferenceWrapper (编排器)                      │
│  ┌──────────┐  ┌──────────────┐  ┌────────┐  ┌──────────────┐  │
│  │  UNet    │→│ Perspective  │→│ Cropper │→│ PixelSize    │  │
│  │ 语义分割  │  │  Detector    │  │ 裁剪校正 │  │  Finder      │  │
│  └──────────┘  └──────────────┘  └────────┘  └──────────────┘  │
│                                                                  │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │ Dewarper │→│   Signal     │→│    Lead Identifier       │  │
│  │ (可选)    │  │  Extractor   │  │  (UNet + 模板匹配)       │  │
│  └──────────┘  └──────────────┘  └──────────────────────────┘  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                        输出结果                                  │
│         CSV 时间序列 (12导联, µV) + PNG 可视化 + 元数据           │
└─────────────────────────────────────────────────────────────────┘
```

## 三个入口点

```bash
# 推理：从图像提取 ECG 时间序列
python3 -m src.digitize --config src/config/inference_wrapper.yml

# 训练：训练/微调分割模型
python3 -m src.train --config src/config/unet.yml

# 评估：对比数字化结果与 Ground Truth
python3 -m src.evaluate --config src/config/evaluate.yml
```
