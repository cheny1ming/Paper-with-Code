"""
Transformer Text Encoder for CLIP
Based on "Attention Is All You Need"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


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
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, embed_dim)
            attn_mask: (seq_len, seq_len) or (batch_size, n_heads, seq_len, seq_len)
            key_padding_mask: (batch_size, seq_len)
        Returns:
            out: (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.shape

        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, n_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (batch, n_heads, seq_len, seq_len)

        # Apply padding mask if provided
        if key_padding_mask is not None:
            # key_padding_mask: (batch_size, seq_len) -> (batch, 1, 1, seq_len)
            padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(padding_mask, float('-inf'))

        # Apply attention mask if provided
        if attn_mask is not None:
            if attn_mask.ndim == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            attn = attn.masked_fill(attn_mask == 0, float('-inf'))

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
    """Transformer encoder block with causal mask support."""

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

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attn_mask, key_padding_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class TextTransformer(nn.Module):
    """
    Transformer for text encoding.

    Args:
        vocab_size: Size of vocabulary
        max_seq_len: Maximum sequence length
        embed_dim: Transformer dimension
        depth: Number of transformer blocks
        n_heads: Number of attention heads
        mlp_ratio: Ratio of MLP hidden dim to embed dim
        dropout: Dropout probability
    """

    def __init__(
        self,
        vocab_size: int = 49408,
        max_seq_len: int = 77,
        embed_dim: int = 512,
        depth: int = 12,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0
    ):
        super().__init__()

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        # Position embedding
        self.positional_embedding = nn.Parameter(torch.zeros(max_seq_len, embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.ln_final = nn.LayerNorm(embed_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

    def forward(
        self,
        text: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            text: (batch_size, seq_len)
            key_padding_mask: (batch_size, seq_len) - True for padding tokens
        Returns:
            features: (batch_size, embed_dim) - EOS token representation
        """
        batch_size, seq_len = text.shape

        # Token embeddings
        x = self.token_embedding(text)  # (B, seq_len, embed_dim)

        # Add positional embedding
        x = x + self.positional_embedding[:seq_len, :]

        # Create causal attention mask (lower triangular)
        attn_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

        # Layer normalization
        x = self.ln_final(x)

        # Take features from the EOS token (last non-padded token)
        if key_padding_mask is not None:
            # Find the last non-padding token for each sequence
            seq_lengths = (~key_padding_mask).sum(dim=1) - 1  # (batch_size,)
            batch_indices = torch.arange(batch_size, device=x.device)
            features = x[batch_indices, seq_lengths]
        else:
            # If no padding mask, use the last token (EOS)
            features = x[torch.arange(batch_size, device=x.device), text.argmax(dim=-1)]

        return features


class TextEncoder(nn.Module):
    """
    Text encoder for CLIP.

    Uses Transformer to encode text,
    then projects to a shared embedding space.

    Args:
        embed_dim: Output embedding dimension
        vocab_size: Size of vocabulary
        max_seq_len: Maximum sequence length
        text_embed_dim: Transformer internal dimension
        depth: Number of transformer blocks
        n_heads: Number of attention heads
        mlp_ratio: Ratio of MLP hidden dim to embed dim
        dropout: Dropout probability
    """

    def __init__(
        self,
        embed_dim: int = 512,
        vocab_size: int = 49408,
        max_seq_len: int = 77,
        text_embed_dim: int = 512,
        depth: int = 12,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0
    ):
        super().__init__()

        self.transformer = TextTransformer(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            embed_dim=text_embed_dim,
            depth=depth,
            n_heads=n_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )

        # Projection to shared embedding space
        self.projection = nn.Sequential(
            nn.LayerNorm(text_embed_dim),
            nn.Linear(text_embed_dim, embed_dim)
        )

    def forward(
        self,
        text: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            text: (batch_size, seq_len)
            key_padding_mask: (batch_size, seq_len)
        Returns:
            features: (batch_size, embed_dim) - L2 normalized
        """
        features = self.transformer(text, key_padding_mask)
        features = self.projection(features)
        # L2 normalize for cosine similarity
        features = F.normalize(features, dim=-1)
        return features


if __name__ == "__main__":
    # Test the text encoder
    batch_size = 4
    seq_len = 77
    vocab_size = 49408

    # Create dummy tokens with padding
    text = torch.randint(0, vocab_size, (batch_size, seq_len))
    # Create padding mask (True for padding)
    padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    padding_mask[:, 60:] = True  # Last 17 tokens are padding

    # Create encoder
    encoder = TextEncoder(
        embed_dim=512,
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        text_embed_dim=512,
        depth=6,  # Smaller for testing
        n_heads=8
    )

    # Forward pass
    with torch.no_grad():
        features = encoder(text, padding_mask)

    print(f"Input shape: {text.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Output norm (should be ~1.0): {features.norm(dim=-1).mean().item():.4f}")
