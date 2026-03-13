"""
Vision Transformer (ViT) Image Encoder for CLIP
Based on "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class PatchEmbedding(nn.Module):
    """Convert image into patches and embed them."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 32,
        in_channels: int = 3,
        embed_dim: int = 768
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        # Use convolution to create patches
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, channels, height, width)
        Returns:
            patches: (batch_size, n_patches, embed_dim)
        """
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)  # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""

    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        assert self.head_dim * n_heads == embed_dim, "embed_dim must be divisible by n_heads"

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, embed_dim)
            mask: (batch_size, seq_len, seq_len) or None
        Returns:
            out: (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.shape

        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, n_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (batch, n_heads, seq_len, seq_len)

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        out = self.proj(out)
        return out


class MLP(nn.Module):
    """Feed-forward network."""

    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, n_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) for image encoding.

    Args:
        img_size: Input image size
        patch_size: Size of each patch
        in_channels: Number of input channels
        embed_dim: Transformer dimension
        depth: Number of transformer blocks
        n_heads: Number of attention heads
        mlp_ratio: Ratio of MLP hidden dim to embed dim
        dropout: Dropout probability
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 32,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        n_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.n_patches = self.patch_embed.n_patches

        # Learnable class token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using truncated normal and zero."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, channels, height, width)
        Returns:
            features: (batch_size, embed_dim) - class token output
        """
        batch_size = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)

        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, n_patches + 1, embed_dim)

        # Add position embedding
        x = x + self.pos_embed

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Layer norm and extract class token
        x = self.norm(x)
        return x[:, 0]  # Return class token


class ImageEncoder(nn.Module):
    """
    Image encoder for CLIP.

    Uses Vision Transformer to extract image features,
    then projects them to a shared embedding space.

    Args:
        embed_dim: Output embedding dimension
        img_size: Input image size
        patch_size: Size of each patch
        in_channels: Number of input channels
        vision_embed_dim: ViT internal dimension
        depth: Number of transformer blocks
        n_heads: Number of attention heads
        mlp_ratio: Ratio of MLP hidden dim to embed dim
        dropout: Dropout probability
    """

    def __init__(
        self,
        embed_dim: int = 512,
        img_size: int = 224,
        patch_size: int = 32,
        in_channels: int = 3,
        vision_embed_dim: int = 768,
        depth: int = 12,
        n_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0
    ):
        super().__init__()

        self.vit = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=vision_embed_dim,
            depth=depth,
            n_heads=n_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )

        # Projection to shared embedding space
        self.projection = nn.Sequential(
            nn.LayerNorm(vision_embed_dim),
            nn.Linear(vision_embed_dim, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, channels, height, width)
        Returns:
            features: (batch_size, embed_dim) - L2 normalized
        """
        features = self.vit(x)
        features = self.projection(features)
        # L2 normalize for cosine similarity
        features = F.normalize(features, dim=-1)
        return features


if __name__ == "__main__":
    # Test the image encoder
    batch_size = 4
    img = torch.randn(batch_size, 3, 224, 224)

    # Create encoder
    encoder = ImageEncoder(
        embed_dim=512,
        img_size=224,
        patch_size=32,
        vision_embed_dim=768,
        depth=6,  # Smaller for testing
        n_heads=8
    )

    # Forward pass
    with torch.no_grad():
        features = encoder(img)

    print(f"Input shape: {img.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Output norm (should be ~1.0): {features.norm(dim=-1).mean().item():.4f}")
