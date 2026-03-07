import torch
import torch.nn as nn
from image_encoder import ImageEncoder


class VisionTransformer(nn.Module):
    """
    端到端的Vision Transformer模型
    专注于图像分类任务
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.1
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # 初始化图像编码器
        self.image_encoder = ImageEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )

    def forward(self, x, return_features=False):
        """
        前向传播
        x: (batch_size, channels, height, width) - 输入图像
        return_features: 如果为True，返回特征而不是分类结果
        """
        return self.image_encoder(x, return_features=return_features)

    def forward_features(self, x):
        """
        返回特征表示，不进行分类
        用于迁移学习等任务
        """
        return self.forward(x, return_features=True)


class ViTConfig:
    """
    ViT模型配置类
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.1
    ):
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout

    @classmethod
    def vit_tiny(cls):
        """ViT-Tiny配置"""
        return cls(
            img_size=224,
            patch_size=16,
            embed_dim=192,
            depth=12,
            num_heads=3,
            mlp_ratio=4.0
        )

    @classmethod
    def vit_small(cls):
        """ViT-Small配置"""
        return cls(
            img_size=224,
            patch_size=16,
            embed_dim=384,
            depth=12,
            num_heads=6,
            mlp_ratio=4.0
        )

    @classmethod
    def vit_base(cls):
        """ViT-Base配置"""
        return cls(
            img_size=224,
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0
        )

    @classmethod
    def vit_large(cls):
        """ViT-Large配置"""
        return cls(
            img_size=224,
            patch_size=16,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            mlp_ratio=4.0
        )

    @classmethod
    def vit_huge(cls):
        """ViT-Huge配置"""
        return cls(
            img_size=224,
            patch_size=14,
            embed_dim=1280,
            depth=32,
            num_heads=16,
            mlp_ratio=4.0
        )


def create_model(config):
    """
    根据配置创建模型
    """
    return VisionTransformer(
        img_size=config.img_size,
        patch_size=config.patch_size,
        in_channels=config.in_channels,
        num_classes=config.num_classes,
        embed_dim=config.embed_dim,
        depth=config.depth,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        dropout=config.dropout
    )


# 测试代码
if __name__ == "__main__":
    print("="*50)
    print("Testing Vision Transformer Model")
    print("="*50)

    # 测试不同大小的模型
    configs = {
        'ViT-Tiny': ViTConfig.vit_tiny(),
        'ViT-Small': ViTConfig.vit_small(),
        'ViT-Base': ViTConfig.vit_base(),
        'ViT-Large': ViTConfig.vit_large()
    }

    for name, config in configs.items():
        print(f"\n{name}:")
        config.num_classes = 1000

        model = create_model(config)

        # 测试前向传播
        image_input = torch.randn(2, 3, config.img_size, config.img_size)
        output = model(image_input)
        features = model.forward_features(image_input)

        print(f"   Input shape: {image_input.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Features shape: {features.shape}")
        print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\n" + "="*50)
    print("All tests completed successfully!")
    print("="*50)
