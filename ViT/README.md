# Vision Transformer (ViT) 实现

这是一个从零开始实现的端到端Vision Transformer (ViT) 模型，专注于图像分类任务。这是我关于ViT的博客文章：[ViT](https://cheny1ming.github.io/Blogs/post.html?id=vit)

## 项目结构

```
ViT/
├── image_encoder.py      # 图像编码器实现
├── ViT.py               # 主模型文件
├── train.py             # 训练脚本
├── data_utils.py        # 数据处理工具
├── requirements.txt     # 依赖包
├── test_model.py        # 测试脚本
├── examples.py          # 使用示例
└── README.md           # 项目说明
```

## 主要特性

### 1. 图像编码器 ([image_encoder.py](image_encoder.py))
- **Patch Embedding**: 将图像分割成patches并转换为embeddings
- **Multi-Head Self-Attention**: 多头自注意力机制
- **Transformer Blocks**: 标准的Transformer编码器块
- **Positional Encoding**: 位置编码
- **CLS Token**: 用于分类的类别token

### 2. ViT模型 ([ViT.py](ViT.py))
- **完整的图像分类**: 端到端的图像分类pipeline
- **预定义配置**: ViT-Tiny、ViT-Small、ViT-Base、ViT-Large、ViT-Huge
- **特征提取**: 支持特征提取用于迁移学习
- **灵活配置**: 支持自定义输入尺寸和模型架构

### 3. 训练工具 ([train.py](train.py))
- **完整的训练循环**: 支持训练、验证、保存检查点
- **学习率调度**: Cosine Annealing + Warmup
- **梯度裁剪**: 防止梯度爆炸
- **进度跟踪**: 使用tqdm显示训练进度

### 4. 数据处理 ([data_utils.py](data_utils.py))
- **图像数据集**: 支持自定义图像数据
- **数据增强**: 自动应用数据增强
- **数据加载器**: 高效的数据加载

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 图像分类

```python
from ViT import VisionTransformer, ViTConfig

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

# 前向传播
import torch
image = torch.randn(1, 3, 224, 224)
output = model(image)
print(output.shape)  # torch.Size([1, 1000])
```

### 2. 特征提取

```python
from ViT import VisionTransformer

model = VisionTransformer(
    img_size=224,
    patch_size=16,
    num_classes=1000
)

# 提取特征
image = torch.randn(1, 3, 224, 224)
features = model.forward_features(image)
print(features.shape)  # torch.Size([1, 768])
```

更多详细配置请见QUICKSTART.md

## 训练模型

### 使用自定义数据集

```python
from data_utils import ImageDataset, DataModule, get_image_transforms
from train import Trainer
from ViT import VisionTransformer, ViTConfig

# 准备数据
transform = get_image_transforms(is_training=True)
train_dataset = ImageDataset(
    image_paths=["path/to/images"],
    labels=[0, 1, 2],
    transform=transform
)

# 创建数据加载器
data_module = DataModule(batch_size=32)
train_loader, val_loader = data_module.get_image_dataloaders(
    train_dataset=train_dataset,
    val_dataset=train_dataset
)

# 创建模型
config = ViTConfig.vit_base()
config.num_classes = 3
model = VisionTransformer(
    img_size=config.img_size,
    patch_size=config.patch_size,
    num_classes=config.num_classes,
    embed_dim=config.embed_dim,
    depth=config.depth,
    num_heads=config.num_heads
)

# 训练
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    lr=3e-4,
    max_epochs=100
)
trainer.train()
```

### 使用CIFAR-10数据集

```bash
python train.py
```

## 模型配置

### ViT-Tiny
```python
config = ViTConfig.vit_tiny()
# embed_dim=192, depth=12, num_heads=3
```

### ViT-Small
```python
config = ViTConfig.vit_small()
# embed_dim=384, depth=12, num_heads=6
```

### ViT-Base
```python
config = ViTConfig.vit_base()
# embed_dim=768, depth=12, num_heads=12
```

### ViT-Large
```python
config = ViTConfig.vit_large()
# embed_dim=1024, depth=24, num_heads=16
```

### ViT-Huge
```python
config = ViTConfig.vit_huge()
# embed_dim=1280, depth=32, num_heads=16
```

## 测试模型

测试模型功能：

```bash
python test_model.py
```

## 查看示例

运行使用示例：

```bash
python examples.py
```

## 关键技术细节

### 1. Patch Embedding
- 使用卷积层将图像分割成patches
- 每个patch大小为 16x16 像素
- 对于224x224图像，得到196个patches + 1个CLS token = 197个tokens

### 2. Multi-Head Attention
- 支持12个注意力头（Base模型）
- 每个头的维度为 embed_dim / num_heads
- 使用缩放点积注意力

### 3. Transformer Block
- Pre-LN结构（Layer Normalization在attention/MLP之前）
- GELU激活函数
- MLP扩展比例为4.0

### 4. 位置编码
- 可学习的位置嵌入
- 长度为 n_patches + 1 (包含CLS token)

## 性能优化建议

1. **使用混合精度训练**: 添加 `torch.cuda.amp` 支持
2. **梯度累积**: 当batch size较小时使用
3. **分布式训练**: 使用 `DistributedDataParallel`
4. **模型并行**: 对于Large和Huge模型
5. **数据增强**: 使用更强的图像增强策略

## 扩展功能

### 添加新的数据增强
在 [data_utils.py](data_utils.py) 的 `get_image_transforms` 函数中添加新的增强策略。

### 自定义数据集
继承 `ImageDataset` 类并实现自己的数据加载逻辑。

### 修改损失函数
在 [train.py](train.py) 的 `Trainer` 类中修改 `self.criterion`。

## 参考文献

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) - ViT原始论文
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer论文

## 模型参数对比

| 模型 | Layers | Hidden Size | MLP Size | Heads | 参数量 |
|------|--------|-------------|----------|-------|--------|
| ViT-Tiny | 12 | 192 | 768 | 3 | ~8M |
| ViT-Small | 12 | 384 | 1536 | 6 | ~22M |
| ViT-Base | 12 | 768 | 3072 | 12 | ~86M |
| ViT-Large | 24 | 1024 | 4096 | 16 | ~307M |
| ViT-Huge | 32 | 1280 | 5120 | 16 | ~632M |

## 许可证

MIT License

## 贡献

欢迎提交问题和拉取请求！

## 联系方式

如有问题，请提交Issue或联系作者。
