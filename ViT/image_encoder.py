import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """
    将图像分割成patch并转换为embeddings
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        # 使用卷积层进行patch提取
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: (batch_size, channels, height, width)
        x = self.proj(x)  # (batch_size, embed_dim, n_patches^0.5, n_patches^0.5)
        x = x.flatten(2)  # (batch_size, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (batch_size, n_patches, embed_dim)
        return x


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Q, K, V投影
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        # 生成Q, K, V
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (batch_size, num_heads, seq_len, seq_len)

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # 应用注意力到V
        x = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        x = self.proj(x)
        x = self.dropout(x)

        return x


class MLP(nn.Module):
    """
    前馈神经网络
    """
    def __init__(self, embed_dim, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer编码器块
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)

    def forward(self, x, mask=None):
        # 多头注意力 + 残差连接
        x = x + self.attn(self.norm1(x), mask)
        # MLP + 残差连接
        x = x + self.mlp(self.norm2(x))
        return x


class ImageEncoder(nn.Module):
    """
    图像编码器 - Vision Transformer的图像处理部分
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
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # Patch Embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)

        # 位置编码
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim)
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)

        # 分类头
        self.head = nn.Linear(embed_dim, num_classes)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        # 初始化位置编码和CLS token
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x, return_features=False):
        # x: (batch_size, channels, height, width)
        batch_size = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (batch_size, n_patches, embed_dim)

        # 添加CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # 添加位置编码
        x = x + self.pos_embed

        # 通过Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Layer normalization
        x = self.norm(x)

        # 提取CLS token的输出
        cls_output = x[:, 0]  # (batch_size, embed_dim)

        if return_features:
            return cls_output

        # 分类
        logits = self.head(cls_output)
        return logits

    def forward_features(self, x):
        """返回特征表示，不进行分类"""
        return self.forward(x, return_features=True)


# 测试代码
if __name__ == "__main__":
    # 创建模型
    model = ImageEncoder(
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.1
    )

    # 测试前向传播
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    features = model.forward_features(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
