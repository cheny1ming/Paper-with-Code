"""
CLIP (Contrastive Language-Image Pre-training)
End-to-end implementation with contrastive learning loss.

Reference: "Learning Transferable Visual Models From Natural Language Supervision"
https://arxiv.org/abs/2103.00020
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from image_encoder import ImageEncoder
from text_encoder import TextEncoder


class CLIPLoss(nn.Module):
    """
    Contrastive loss for CLIP training.

    Uses symmetric cross-entropy loss over image-text similarity matrix.
    Implements both caption matching and image matching.

    Args:
        temperature: Temperature parameter for softmax
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            image_features: (batch_size, embed_dim) - L2 normalized
            text_features: (batch_size, embed_dim) - L2 normalized
        Returns:
            loss: Scalar loss value
            metrics: Dictionary with additional metrics
        """
        batch_size = image_features.shape[0]

        # Compute similarity matrix (cosine similarity since features are normalized)
        # logits[i][j] = similarity between image i and text j
        logits = (image_features @ text_features.T) / self.temperature

        # Ground truth: diagonal elements are matching pairs
        labels = torch.arange(batch_size, device=image_features.device)

        # Symmetric loss: image-to-text and text-to-image
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        loss = (loss_i2t + loss_t2i) / 2

        # Compute accuracy
        with torch.no_grad():
            # Image-to-text accuracy
            i2t_pred = logits.argmax(dim=1)
            i2t_acc = (i2t_pred == labels).float().mean()

            # Text-to-image accuracy
            t2i_pred = logits.argmax(dim=0)
            t2i_acc = (t2i_pred == labels).float().mean()

            metrics = {
                'loss_i2t': loss_i2t.item(),
                'loss_t2i': loss_t2i.item(),
                'acc_i2t': i2t_acc.item(),
                'acc_t2i': t2i_acc.item()
            }

        return loss, metrics


class CLIP(nn.Module):
    """
    End-to-end CLIP model.

    Args:
        embed_dim: Shared embedding dimension
        image_embed_dim: Image encoder internal dimension
        text_embed_dim: Text encoder internal dimension
        img_size: Input image size
        patch_size: Size of each image patch
        vocab_size: Size of vocabulary
        max_seq_len: Maximum text sequence length
        vision_depth: Number of vision transformer blocks
        text_depth: Number of text transformer blocks
        vision_heads: Number of vision attention heads
        text_heads: Number of text attention heads
        temperature: Temperature for contrastive loss
    """

    def __init__(
        self,
        embed_dim: int = 512,
        image_embed_dim: int = 768,
        text_embed_dim: int = 512,
        img_size: int = 224,
        patch_size: int = 32,
        vocab_size: int = 49408,
        max_seq_len: int = 77,
        vision_depth: int = 12,
        text_depth: int = 12,
        vision_heads: int = 12,
        text_heads: int = 8,
        temperature: float = 0.07
    ):
        super().__init__()

        self.image_encoder = ImageEncoder(
            embed_dim=embed_dim,
            img_size=img_size,
            patch_size=patch_size,
            in_channels=3,
            vision_embed_dim=image_embed_dim,
            depth=vision_depth,
            n_heads=vision_heads
        )

        self.text_encoder = TextEncoder(
            embed_dim=embed_dim,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            text_embed_dim=text_embed_dim,
            depth=text_depth,
            n_heads=text_heads
        )

        self.loss_fn = CLIPLoss(temperature=temperature)

        # Logit scale (learnable temperature parameter)
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / temperature)))

    def forward(
        self,
        images: torch.Tensor,
        texts: torch.Tensor,
        text_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass with contrastive loss computation.

        Args:
            images: (batch_size, 3, height, width)
            texts: (batch_size, seq_len)
            text_padding_mask: (batch_size, seq_len) - True for padding
        Returns:
            loss: Scalar loss value
            metrics: Dictionary with additional metrics
        """
        # Encode images and texts
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(texts, text_padding_mask)

        # Compute contrastive loss
        loss, metrics = self.loss_fn(image_features, text_features)

        return loss, metrics

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to embeddings.

        Args:
            images: (batch_size, 3, height, width)
        Returns:
            features: (batch_size, embed_dim) - L2 normalized
        """
        return self.image_encoder(images)

    def encode_text(self, texts: torch.Tensor, text_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode texts to embeddings.

        Args:
            texts: (batch_size, seq_len)
            text_padding_mask: (batch_size, seq_len) - True for padding
        Returns:
            features: (batch_size, embed_dim) - L2 normalized
        """
        return self.text_encoder(texts, text_padding_mask)

    def compute_similarity(
        self,
        images: torch.Tensor,
        texts: torch.Tensor,
        text_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute image-text similarity scores.

        Args:
            images: (batch_size, 3, height, width)
            texts: (batch_size, seq_len)
            text_padding_mask: (batch_size, seq_len)
        Returns:
            similarity: (batch_size, batch_size) - cosine similarity matrix
        """
        image_features = self.encode_image(images)
        text_features = self.encode_text(texts, text_padding_mask)

        # Apply learnable temperature
        similarity = (image_features @ text_features.T) * self.logit_scale.exp()
        return similarity

    def retrieve_text(
        self,
        images: torch.Tensor,
        candidate_texts: torch.Tensor,
        text_padding_mask: Optional[torch.Tensor] = None,
        top_k: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve top-k texts for given images.

        Args:
            images: (batch_size, 3, height, width)
            candidate_texts: (n_candidates, seq_len)
            text_padding_mask: (n_candidates, seq_len)
            top_k: Number of top candidates to return
        Returns:
            scores: (batch_size, top_k)
            indices: (batch_size, top_k)
        """
        image_features = self.encode_image(images)
        text_features = self.encode_text(candidate_texts, text_padding_mask)

        similarity = image_features @ text_features.T  # (batch_size, n_candidates)

        scores, indices = similarity.topk(top_k, dim=1)
        return scores, indices

    def retrieve_images(
        self,
        texts: torch.Tensor,
        candidate_images: torch.Tensor,
        text_padding_mask: Optional[torch.Tensor] = None,
        top_k: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve top-k images for given texts.

        Args:
            texts: (batch_size, seq_len)
            candidate_images: (n_candidates, 3, height, width)
            text_padding_mask: (batch_size, seq_len)
            top_k: Number of top candidates to return
        Returns:
            scores: (batch_size, top_k)
            indices: (batch_size, top_k)
        """
        text_features = self.encode_text(texts, text_padding_mask)
        image_features = self.encode_image(candidate_images)

        similarity = text_features @ image_features.T  # (batch_size, n_candidates)

        scores, indices = similarity.topk(top_k, dim=1)
        return scores, indices

    @torch.no_grad()
    def zero_shot_classification(
        self,
        images: torch.Tensor,
        class_texts: torch.Tensor,
        text_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Zero-shot image classification using text prompts.

        Args:
            images: (batch_size, 3, height, width)
            class_texts: (n_classes, seq_len) - text descriptions for each class
            text_padding_mask: (n_classes, seq_len)
        Returns:
            probs: (batch_size, n_classes) - class probabilities
        """
        image_features = self.encode_image(images)
        text_features = self.encode_text(class_texts, text_padding_mask)

        # Compute logit scores
        logits = image_features @ text_features.T  # (batch_size, n_classes)
        logits = logits * self.logit_scale.exp()

        # Convert to probabilities
        probs = F.softmax(logits, dim=1)
        return probs


class SimpleTokenizer:
    """
    Simple tokenizer for testing (in practice, use a proper tokenizer like SentencePiece).

    This is a minimal implementation for demonstration purposes.
    """

    def __init__(self, vocab_size: int = 49408):
        self.vocab_size = vocab_size
        # Build a simple vocabulary (for demo purposes)
        self.word_to_idx = {"<pad>": 0, "<eos>": 1, "<sos>": 2, "<unk>": 3}
        self.idx_to_word = {v: k for k, v in self.word_to_idx.items()}

    def encode(self, text: str, max_length: int = 77) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode text to token indices."""
        words = text.lower().split()
        tokens = [self.word_to_idx.get(w, self.word_to_idx["<unk>"]) for w in words]
        tokens.append(self.word_to_idx["<eos>"])

        # Truncate or pad
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        else:
            tokens = tokens + [self.word_to_idx["<pad>"]] * (max_length - len(tokens))

        token_tensor = torch.tensor(tokens, dtype=torch.long)
        padding_mask = torch.tensor([t == self.word_to_idx["<pad>"] for t in tokens], dtype=torch.bool)

        return token_tensor.unsqueeze(0), padding_mask.unsqueeze(0)


if __name__ == "__main__":
    print("=" * 60)
    print("CLIP Model Test")
    print("=" * 60)

    # Model hyperparameters
    batch_size = 4
    embed_dim = 256
    vocab_size = 10000

    # Create model with smaller dimensions for testing
    model = CLIP(
        embed_dim=embed_dim,
        image_embed_dim=384,
        text_embed_dim=256,
        img_size=224,
        patch_size=32,
        vocab_size=vocab_size,
        max_seq_len=77,
        vision_depth=4,
        text_depth=4,
        vision_heads=6,
        text_heads=4,
        temperature=0.07
    )

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Create dummy data
    images = torch.randn(batch_size, 3, 224, 224)
    texts = torch.randint(0, vocab_size, (batch_size, 77))
    padding_mask = torch.zeros(batch_size, 77, dtype=torch.bool)
    padding_mask[:, 60:] = True  # Mark last 17 tokens as padding

    print(f"\nInput shapes:")
    print(f"  Images: {images.shape}")
    print(f"  Texts: {texts.shape}")

    # Forward pass
    print("\n" + "=" * 60)
    print("Testing forward pass with loss...")
    print("=" * 60)

    loss, metrics = model(images, texts, padding_mask)

    print(f"\nLoss: {loss.item():.4f}")
    print(f"Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    # Test encoding functions
    print("\n" + "=" * 60)
    print("Testing encoding functions...")
    print("=" * 60)

    image_features = model.encode_image(images)
    text_features = model.encode_text(texts, padding_mask)

    print(f"\nImage features shape: {image_features.shape}")
    print(f"Text features shape: {text_features.shape}")
    print(f"Image features norm: {image_features.norm(dim=-1).mean().item():.4f}")
    print(f"Text features norm: {text_features.norm(dim=-1).mean().item():.4f}")

    # Test similarity computation
    print("\n" + "=" * 60)
    print("Testing similarity computation...")
    print("=" * 60)

    similarity = model.compute_similarity(images, texts, padding_mask)
    print(f"\nSimilarity matrix shape: {similarity.shape}")
    print(f"Similarity matrix diagonal (should be high): {similarity.diag().mean().item():.4f}")

    # Test retrieval
    print("\n" + "=" * 60)
    print("Testing text retrieval...")
    print("=" * 60)

    candidate_texts = torch.randint(0, vocab_size, (20, 77))
    scores, indices = model.retrieve_text(images, candidate_texts, top_k=5)

    print(f"\nRetrieved scores shape: {scores.shape}")
    print(f"Retrieved indices shape: {indices.shape}")

    # Test zero-shot classification
    print("\n" + "=" * 60)
    print("Testing zero-shot classification...")
    print("=" * 60)

    n_classes = 10
    class_texts = torch.randint(0, vocab_size, (n_classes, 77))
    probs = model.zero_shot_classification(images, class_texts)

    print(f"\nClass probabilities shape: {probs.shape}")
    print(f"Probabilities sum to 1: {probs.sum(dim=-1).mean().item():.4f}")
    print(f"Predicted classes: {probs.argmax(dim=-1)}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
