# 01 — 项目概览

## 1.1 项目简介

**Open ECG Digitizer** 是一个开源的 12 导联心电图 (ECG) 数字化工具，核心功能是从纸质 ECG 的扫描图像或手机拍摄的照片中，提取原始时间序列数据。

### 核心特性

- **输入**: 扫描图 (.png/.jpg/.jpeg) 或手机照片
- **输出**: 12 导联时间序列 CSV（单位：µV，默认 5000 采样点）+ PNG 可视化 + 元数据
- **透视校正**: 自动检测并校正拍照产生的透视畸变（最大约 45°）
- **任意导联子集**: 支持 12 标准导联的任意子集
- **高度可配置**: 通过 YAML 配置文件控制全部参数，支持命令行覆盖

### 技术栈

- **语言**: Python 3.12+
- **深度学习框架**: PyTorch + torchvision
- **配置管理**: yacs (YAML Ain't Markup Language)
- **超参搜索**: Ray Tune
- **图像处理**: scikit-image, OpenCV, PIL
- **科学计算**: NumPy, SciPy, scikit-learn

---

## 1.2 目录结构

```
Open-ECG-Digitizer/
│
├── src/                              # ====== 核心源码 ======
│   ├── __init__.py                   # 包初始化（空文件）
│   ├── digitize.py                   # 推理入口点 ★
│   ├── train.py                      # 训练入口点 ★
│   ├── evaluate.py                   # 评估入口点 ★
│   ├── utils.py                      # 工具类与函数
│   │
│   ├── config/                       # ------ 配置文件 ------
│   │   ├── __init__.py
│   │   ├── default.py                # yacs CfgNode 基类
│   │   ├── unet.yml                  # 分割 UNet 训练配置
│   │   ├── lead_name_unet.yml        # 导联名称 UNet 训练配置
│   │   ├── inference_wrapper.yml     # 通用推理配置
│   │   ├── inference_wrapper_ahus_testset.yml
│   │   ├── inference_wrapper_george-moody-2024.yml
│   │   ├── lead_layouts_all.yml      # 完整导联布局模板（14种）
│   │   ├── lead_layouts_reduced.yml  # 精简布局模板（2种）
│   │   ├── lead_layouts_george-moody-2024.yml
│   │   ├── evaluate.yml              # 评估配置
│   │   └── photo_transform.yml       # 数据增强配置
│   │
│   ├── model/                        # ------ 核心 Pipeline 模块 ------
│   │   ├── __init__.py
│   │   ├── inference_wrapper.py      # 主编排器 ★ (InferenceWrapper)
│   │   ├── unet.py                   # U-Net 语义分割网络
│   │   ├── perspective_detector.py   # Hough 变换透视检测
│   │   ├── cropper.py                # 裁剪 + 透视校正
│   │   ├── pixel_size_finder.py      # 网格尺寸估计 (mm/pixel)
│   │   ├── dewarper.py               # TPS 去弯曲（实验性）
│   │   ├── signal_extractor.py       # 信号轨迹提取
│   │   └── lead_identifier.py        # 导联布局识别
│   │
│   ├── dataset/                      # ------ 数据集 ------
│   │   ├── __init__.py
│   │   ├── ecg_scan.py               # ECGScanDataset (扫描图+mask)
│   │   └── synthetic_lead_text.py    # SyntheticLeadTextDataset (合成文字)
│   │
│   ├── loss/
│   │   └── loss.py                   # DiceFocalLoss 损失函数
│   │
│   ├── optimizer/
│   │   ├── __init__.py
│   │   └── adammuon.py               # AdamMuon 混合优化器
│   │
│   ├── transform/
│   │   ├── vision.py                 # 22 种数据增强变换
│   │   └── overlay_images/           # 文字叠加素材 (2张 PNG)
│   │
│   ├── scripts/                      # ------ 辅助脚本 ------
│   │   ├── split_transform_ecg_dataset.py  # 数据集分割与变换
│   │   ├── visualize_transforms.py         # 增强效果可视化
│   │   └── redact/                         # 数字脱敏
│   │       ├── apply_digital_redaction.py
│   │       └── detect_redacted_regions.ipynb
│   │
│   └── report/                       # ------ 论文图表生成 ------
│       ├── binarization_example.py
│       ├── perspective_example_figure.py
│       └── perspective_example_images.py
│
├── weights/                          # ====== 预训练权重 (Git LFS) ======
│   ├── unet_weights_07072025.pt              # 主分割 UNet
│   └── lead_name_unet_weights_07072025.pt    # 导联名称 UNet
│
├── test/                             # ====== 测试 ======
│   ├── test_config.py                # 集成测试（完整训练流程 1 epoch）
│   ├── test_utils.py                 # CosineToConstantLR 单元测试
│   └── test_data/
│       ├── config/unet.yml           # 测试用极小模型配置
│       └── data/ecg_data/            # 3 张测试 ECG 图 + 对应 mask
│
├── assets/                           # ====== 文档资源 ======
│   ├── visual_abstract-img0.png
│   ├── visual_abstract-img1.png
│   ├── visual_abstract-img2.png
│   └── pipeline-overview.svg
│
├── .github/workflows/                # ====== CI/CD ======
│   ├── test.yml                      # 自动化测试
│   ├── lint.yml                      # 代码格式检查
│   ├── type.yml                      # mypy 类型检查
│   ├── commit_lint.yml               # 提交消息规范
│   ├── release.yml                   # 语义化发布
│   └── release_noop.yml              # 发布 dry-run
│
├── setup.py                          # 包元数据 (version=1.9.2)
├── pyproject.toml                    # 工具配置 (black, isort, mypy, semantic-release)
├── requirements.txt                  # 24 个依赖
├── format_and_check.sh               # 格式化+检查脚本
├── README.md                         # 项目说明
└── CHANGELOG.md                      # 版本历史 (v0.0.0 → v1.9.2)
```

---

## 1.3 依赖关系

### 核心依赖 (requirements.txt)

| 类别 | 包 | 用途 |
|------|-----|------|
| **深度学习** | `torch`, `torchvision`, `torchaudio` | 模型推理与训练 |
| **科学计算** | `numpy`, `scipy`, `scikit-image`, `scikit-learn` | 数值计算、图像处理、KNN |
| **图像处理** | `pillow`, `opencv-python` | 图像 I/O、文字渲染 |
| **配置** | `yacs` | YAML 配置管理 |
| **超参搜索** | `ray[data,train,tune]` | 分布式超参数搜索 |
| **可视化** | `matplotlib`, `tensorboard`, `plotly` | 图表、训练监控 |
| **几何变换** | `torch-tps` | 薄板样条 (Thin Plate Spline) |
| **图算法** | `networkx` | 图的连通分量 |
| **NLP** | `transformers` | (未来扩展预留) |
| **交互** | `tqdm`, `ipython`, `ipywidgets` | 进度条、Notebook |
| **类型检查** | `types-tqdm`, `types-setuptools`, `types-networkx` | mypy 类型存根 |

### 模块间依赖关系

```
digitize.py ──→ config/default.py (配置加载)
     │     ──→ utils.py (import_class_from_path, find_config_path)
     │     ──→ model/inference_wrapper.py (动态实例化)
     │
     ▼
inference_wrapper.py ──→ model/unet.py
                     ──→ model/perspective_detector.py
                     ──→ model/cropper.py
                     ──→ model/pixel_size_finder.py
                     ──→ model/dewarper.py
                     ──→ model/signal_extractor.py
                     ──→ model/lead_identifier.py

train.py ──→ config/default.py
         ──→ utils.py (get_data_loaders, load_model, EarlyStopper, CosineToConstantLR)
         ──→ dataset/ecg_scan.py | dataset/synthetic_lead_text.py
         ──→ loss/loss.py
         ──→ optimizer/adammuon.py
         ──→ transform/vision.py (通过配置动态加载)
```

---

## 1.4 版本演进摘要

| 版本 | 日期 | 里程碑 |
|------|------|--------|
| v0.1~v0.5 | 2024-12 | 基础架构、Hough 变换、数据增强 |
| v0.6~v0.8 | 2024-12 | 透视检测、变换支持 |
| v0.9~v1.0 | 2025-01 | 缓存、多类别模型、PNG mask 支持 |
| v1.1~v1.2 | 2025-01 | 裁剪模块、mm/pixel 估计 |
| v1.3 | 2025-06 | Python 3.12、4 类分割、Muon 优化器 |
| v1.4 | 2025-06 | 数字脱敏 (redaction) |
| v1.5 | 2025-07 | 导联识别、信号提取 |
| v1.6 | 2025-07 | 推理 wrapper、CSV 输出、指标计算 |
| v1.7 | 2025-09 | George Moody 2024 竞赛配置 |
| v1.8 | 2025-10 | 元数据保存、可变阈值裁剪 |
| v1.9 | 2025-10 | 配置覆盖支持、文档改进 |

---

## 1.5 预训练权重

通过 **Git LFS** 管理，存放在 `weights/` 目录：

| 文件 | 模型 | 输入→输出 | 特征维度 |
|------|------|-----------|----------|
| `unet_weights_07072025.pt` | 主分割 UNet | 3ch(RGB) → 4ch(grid/text/signal/bg) | `[32,64,128,256,320,320,320,320]` |
| `lead_name_unet_weights_07072025.pt` | 导联名 UNet | 1ch(灰度) → 13ch(12导联+bg) | `[32,64,128,256,256]` |
