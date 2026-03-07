# 快速入门指南

## 项目概览

这个项目实现了一个完整的端到端Vision Transformer (ViT) 模型，专注于图像分类任务。这是我关于ViT的博客文章：[ViT](https://cheny1ming.github.io/Blogs/post.html?id=vit)

### 核心文件
- [image_encoder.py](image_encoder.py) - 图像编码器实现
- [ViT.py](ViT.py) - 端到端ViT主模型

### 工具文件
- [train.py](train.py) - 完整的训练脚本
- [data_utils.py](data_utils.py) - 数据处理工具
- [test_model.py](test_model.py) - 模型测试脚本
- [examples.py](examples.py) - 使用示例

## 5分钟快速开始

### 1. 安装依赖

```bash
pip install torch torchvision numpy Pillow tqdm
```

或者使用requirements.txt：
```bash
pip install -r requirements.txt
```

### 2. 测试模型

运行测试脚本验证模型是否正常工作：

```bash
python test_model.py
```

### 3. 查看示例

运行示例代码了解如何使用模型：

```bash
python examples.py
```

### 4. 开始训练

使用CIFAR-10数据集训练模型：

```bash
python train.py
```

## 基本使用

### 图像分类

```python
from ViT import VisionTransformer, ViTConfig
import torch

# 创建模型
config = ViTConfig.vit_base()
config.num_classes = 1000

model = VisionTransformer(
    img_size=config.img_size,
    patch_size=config.patch_size,
    num_classes=config.num_classes,
    embed_dim=config.embed_dim,
    depth=config.depth,
    num_heads=config.num_heads
)

# 预测
image = torch.randn(1, 3, 224, 224)
output = model(image)
```

### 特征提取

```python
from ViT import VisionTransformer
import torch

model = VisionTransformer(
    img_size=224,
    patch_size=16,
    num_classes=1000
)

# 提取特征
image = torch.randn(4, 3, 224, 224)
features = model.forward_features(image)
```

## 模型配置

### 预定义配置

```python
from ViT import ViTConfig

# ViT-Tiny (~8M parameters)
config = ViTConfig.vit_tiny()

# ViT-Small (~22M parameters)
config = ViTConfig.vit_small()

# ViT-Base (~86M parameters)
config = ViTConfig.vit_base()

# ViT-Large (~307M parameters)
config = ViTConfig.vit_large()

# ViT-Huge (~632M parameters)
config = ViTConfig.vit_huge()
```

### 自定义配置

```python
config = ViTConfig(
    img_size=224,
    patch_size=16,
    num_classes=1000,
    embed_dim=768,
    depth=12,
    num_heads=12
)
```

## 训练流程

### 1. 准备数据

```python
from data_utils import ImageDataset, DataModule, get_image_transforms

# 创建数据集
transform = get_image_transforms(is_training=True)
dataset = ImageDataset(
    image_paths=['image1.jpg', 'image2.jpg'],
    labels=[0, 1],
    transform=transform
)

# 创建数据加载器
data_module = DataModule(batch_size=32)
train_loader, val_loader = data_module.get_image_dataloaders(
    train_dataset=dataset,
    val_dataset=dataset
)
```

### 2. 创建训练器

```python
from train import Trainer
from ViT import VisionTransformer, ViTConfig

config = ViTConfig.vit_base()
config.num_classes = 10

model = VisionTransformer(
    img_size=config.img_size,
    patch_size=config.patch_size,
    num_classes=config.num_classes,
    embed_dim=config.embed_dim,
    depth=config.depth,
    num_heads=config.num_heads
)

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    lr=3e-4,
    max_epochs=100
)
```

### 3. 开始训练

```python
trainer.train()
```

### 4. 保存和加载模型

```python
# 保存模型
trainer.save_model('my_model.pth')

# 加载模型
trainer.load_model('my_model.pth')
```

## 关键特性

### 1. 多种模型尺寸
- ViT-Tiny: 快速原型设计
- ViT-Small: 轻量级部署
- ViT-Base: 标准性能
- ViT-Large: 高性能
- ViT-Huge: 最佳性能

### 2. 完整的训练功能
- 学习率调度（Cosine Annealing + Warmup）
- 梯度裁剪
- 验证和检查点保存
- 进度跟踪

### 3. 数据处理
- 图像数据增强
- 灵活的数据集接口
- 高效的数据加载

### 4. 特征提取
- 支持特征提取用于迁移学习
- 可选的分类头
- 灵活的模型配置

## 常见问题

### Q: 如何使用自己的数据集？

A: 使用 `ImageDataset` 类，提供图像路径和标签列表：
```python
dataset = ImageDataset(
    image_paths=['path/to/img1.jpg', 'path/to/img2.jpg'],
    labels=[0, 1],
    transform=transform
)
```

### Q: 如何调整模型大小？

A: 使用预定义配置或自定义配置：
```python
# 使用预定义配置
config = ViTConfig.vit_base()

# 或自定义配置
config = ViTConfig(embed_dim=512, depth=8, num_heads=8)
```

### Q: 如何进行迁移学习？

A: 冻结编码器，只训练分类头：
```python
# 冻结编码器
for param in model.image_encoder.parameters():
    param.requires_grad = False

# 只训练分类头
for param in model.image_encoder.head.parameters():
    param.requires_grad = True
```

### Q: 如何使用不同的输入尺寸？

A: 在创建模型时指定 `img_size` 参数：
```python
model = VisionTransformer(
    img_size=384,  # 使用384x384输入
    patch_size=16,
    num_classes=1000
)
```

## 下一步

1. 查看 [README.md](README.md) 了解详细文档
2. 运行 [examples.py](examples.py) 查看更多示例
3. 运行 [test_model.py](test_model.py) 测试模型
4. 使用 [train.py](train.py) 开始训练

## 技术支持

如有问题，请查看相关文件或提交Issue。
