"""
Vision Transformer 使用示例
展示如何在实际场景中使用ViT模型进行图像分类
"""

import torch
from ViT import VisionTransformer, ViTConfig
from data_utils import ImageDataset, DataModule, get_image_transforms
from PIL import Image
import numpy as np


# ============================================
# 示例 1: 基础图像分类
# ============================================

def example_basic_classification():
    """
    基础图像分类示例
    """
    print("="*60)
    print("Example 1: Basic Image Classification")
    print("="*60)

    # 1. 创建模型
    print("\n1. Creating model...")
    config = ViTConfig.vit_base()
    config.num_classes = 10  # 假设有10个类别

    model = VisionTransformer(
        img_size=config.img_size,
        patch_size=config.patch_size,
        num_classes=config.num_classes,
        embed_dim=config.embed_dim,
        depth=config.depth,
        num_heads=config.num_heads
    )

    model.eval()  # 设置为评估模式

    # 2. 准备图像
    print("2. Preparing image...")
    transform = get_image_transforms(img_size=224, is_training=False)

    # 示例：创建一个随机图像（实际使用时替换为真实图像）
    # image = Image.open("path/to/your/image.jpg")
    # image_tensor = transform(image).unsqueeze(0)  # 添加batch维度

    # 这里使用随机张量作为示例
    image_tensor = torch.randn(1, 3, 224, 224)
    print(f"   Image tensor shape: {image_tensor.shape}")

    # 3. 预测
    print("3. Making prediction...")
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1)
        confidence = probabilities[0, predicted_class].item()

    print(f"   Predicted class: {predicted_class.item()}")
    print(f"   Confidence: {confidence:.4f}")
    print(f"   All probabilities: {probabilities[0].tolist()}")


# ============================================
# 示例 2: 使用预定义配置
# ============================================

def example_predefined_configs():
    """
    使用预定义模型配置
    """
    print("\n" + "="*60)
    print("Example 2: Using Predefined Configurations")
    print("="*60)

    configs = {
        'ViT-Tiny': ViTConfig.vit_tiny(),
        'ViT-Small': ViTConfig.vit_small(),
        'ViT-Base': ViTConfig.vit_base(),
        'ViT-Large': ViTConfig.vit_large()
    }

    for name, config in configs.items():
        print(f"\n{name}:")
        config.num_classes = 1000

        model = VisionTransformer(
            img_size=config.img_size,
            patch_size=config.patch_size,
            num_classes=config.num_classes,
            embed_dim=config.embed_dim,
            depth=config.depth,
            num_heads=config.num_heads
        )

        params = sum(p.numel() for p in model.parameters())
        print(f"   Parameters: {params:,}")
        print(f"   Embedding dim: {config.embed_dim}")
        print(f"   Depth: {config.depth}")
        print(f"   Heads: {config.num_heads}")


# ============================================
# 示例 3: 特征提取
# ============================================

def example_feature_extraction():
    """
    特征提取示例 - 用于迁移学习
    """
    print("\n" + "="*60)
    print("Example 3: Feature Extraction for Transfer Learning")
    print("="*60)

    # 1. 创建模型
    print("\n1. Creating model...")
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

    model.eval()

    # 2. 提取特征
    print("2. Extracting features...")
    images = torch.randn(4, 3, 224, 224)  # batch_size=4

    with torch.no_grad():
        features = model.forward_features(images)

    print(f"   Input images shape: {images.shape}")
    print(f"   Extracted features shape: {features.shape}")
    print(f"   Feature dimension: {features.shape[-1]}")

    # 3. 这些特征可以用于其他任务
    print("\n3. Using features for transfer learning...")
    # 例如：添加一个简单的分类器
    num_new_classes = 5
    classifier = torch.nn.Linear(config.embed_dim, num_new_classes)

    # 训练这个新的分类器（原始ViT参数可以冻结）
    new_predictions = classifier(features)
    print(f"   New predictions shape: {new_predictions.shape}")


# ============================================
# 示例 4: 批量推理
# ============================================

def example_batch_inference():
    """
    批量推理示例
    """
    print("\n" + "="*60)
    print("Example 4: Batch Inference")
    print("="*60)

    # 1. 创建模型
    print("\n1. Creating model...")
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        num_classes=10
    )

    model.eval()

    # 2. 批量处理
    print("2. Processing batch of images...")
    batch_size = 8
    images = torch.randn(batch_size, 3, 224, 224)

    with torch.no_grad():
        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=-1)
        predictions = torch.argmax(probabilities, dim=-1)

    print(f"   Batch size: {batch_size}")
    print(f"   Output shape: {outputs.shape}")
    print(f"   Predictions: {predictions.tolist()}")


# ============================================
# 示例 5: 自定义数据集训练准备
# ============================================

def example_custom_dataset():
    """
    自定义数据集准备示例
    """
    print("\n" + "="*60)
    print("Example 5: Custom Dataset Preparation")
    print("="*60)

    # 1. 创建数据变换
    print("\n1. Creating data transforms...")
    train_transform = get_image_transforms(img_size=224, is_training=True)
    val_transform = get_image_transforms(img_size=224, is_training=False)

    # 2. 创建数据集
    print("2. Creating datasets...")
    # 示例：替换为你的实际图像路径
    image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
    labels = [0, 1, 2]

    train_dataset = ImageDataset(
        image_paths=image_paths,
        labels=labels,
        transform=train_transform
    )

    print(f"   Training dataset size: {len(train_dataset)}")

    # 3. 创建数据加载器
    print("3. Creating data loaders...")
    data_module = DataModule(batch_size=2, num_workers=0)
    train_loader, _ = data_module.get_image_dataloaders(
        train_dataset=train_dataset
    )

    print(f"   Number of batches: {len(train_loader)}")

    # 4. 测试数据加载
    print("4. Testing data loading...")
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"   Batch {batch_idx + 1}:")
        print(f"      Images shape: {images.shape}")
        print(f"      Labels shape: {labels.shape}")
        if batch_idx >= 1:  # 只显示前两个batch
            break


# ============================================
# 示例 6: 不同输入尺寸
# ============================================

def example_different_input_sizes():
    """
    不同输入尺寸示例
    """
    print("\n" + "="*60)
    print("Example 6: Different Input Sizes")
    print("="*60)

    input_sizes = [224, 384, 512]

    for size in input_sizes:
        print(f"\nInput size: {size}x{size}")

        model = VisionTransformer(
            img_size=size,
            patch_size=16,
            num_classes=1000
        )

        model.eval()

        # 测试前向传播
        image = torch.randn(1, 3, size, size)
        with torch.no_grad():
            output = model(image)

        params = sum(p.numel() for p in model.parameters())
        print(f"   Input shape: {image.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Parameters: {params:,}")


# ============================================
# 主函数
# ============================================

def main():
    """
    运行所有示例
    """
    print("\n" + "="*60)
    print("Vision Transformer Usage Examples")
    print("="*60)

    # 运行示例
    example_basic_classification()
    example_predefined_configs()
    example_feature_extraction()
    example_batch_inference()
    example_custom_dataset()
    example_different_input_sizes()

    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)


if __name__ == "__main__":
    main()
