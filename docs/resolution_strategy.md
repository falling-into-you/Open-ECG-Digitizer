# U-Net 输入分辨率策略

本文档描述 U-Net 在训练和推理阶段的输入分辨率处理策略。

## 训练阶段

训练时采用**固定分辨率**策略，通过 `RandomCrop` 将原始图像裁剪为 **1024 x 1024** 的正方形区域。

### 配置

配置文件：`src/config/unet.yml`

```yaml
# 训练集
DATASET:
  TRAIN:
    TRANSFORM:
      - class_path: 'src.transform.vision.RandomCrop'
        KWARGS:
          size: [1024, 1024]

  # 验证集使用相同分辨率
  VAL:
    TRANSFORM:
      - class_path: 'src.transform.vision.RandomCrop'
        KWARGS:
          size: [1024, 1024]

  # 测试集不做 transform
  TEST:
    TRANSFORM: ~
```

### 处理流程

1. 从磁盘加载原始 PNG 图像（任意分辨率）
2. `RandomCrop` 在原图上随机裁剪 1024x1024 区域（`src/transform/vision.py:285-304`）
3. 后续数据增强（翻转、JPEG 压缩、缩放等）保持 1024x1024 不变
4. 以 batch_size=12 输入 U-Net

### 关键参数

| 参数 | 值 |
|------|-----|
| 输入分辨率 | 1024 x 1024 |
| 输入通道数 | 3 (RGB) |
| 输出通道数 | 4 |
| 批大小 | 12 |

---

## 推理阶段

推理时采用**动态缩放**策略，保持原始长宽比，仅在图像过大或过小时按比例调整。

### 配置

配置文件：`src/config/inference_wrapper.yml`

```yaml
MODEL:
  KWARGS:
    resample_size: 3000          # 长边上限
    rotate_on_resample: true     # 高>宽时自动旋转
    # minimum_image_size 默认为 512
```

### 缩放规则

由 `InferenceWrapper._resample_image()` 实现（`src/model/inference_wrapper.py:254-282`）：

| 条件 | 处理方式 |
|------|---------|
| 最小边 < 512 | 按比例放大，使最小边 = 512 |
| 长边 > 3000 | 按比例缩小，使长边 = 3000 |
| 512 <= 尺寸 <= 3000 | 保持原分辨率，不做缩放 |
| `resample_size` 为 tuple | 直接缩放到指定尺寸 |
| `resample_size` 为 None | 不做任何缩放 |

### 处理流程

1. 加载原始图像
2. 如果 `rotate_on_resample=true` 且图像高度 > 宽度，旋转 90 度
3. 根据上述规则动态缩放
4. 以 batch_size=1 输入 U-Net
5. 将输出结果缩放回原始分辨率

---

## 训练 vs 推理对比

| | 训练 | 推理 |
|---|------|------|
| 分辨率 | 固定 1024 x 1024 | 动态（512 ~ 3000） |
| 裁剪/缩放方式 | RandomCrop 随机裁剪 | 按比例 resize |
| 长宽比 | 1:1 正方形 | 保留原始比例 |
| 批大小 | 12 | 1 |
| 配置文件 | `src/config/unet.yml` | `src/config/inference_wrapper.yml` |
