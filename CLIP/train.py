"""
CLIP Training Example
Demonstrates how to train the CLIP model end-to-end.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from CLIP import CLIP


class DummyCLIPDataset(Dataset):
    """
    Dummy dataset for demonstration.
    In practice, use a real dataset like LAION, CC3M, or COCO.
    """

    def __init__(self, size=1000, img_size=224, vocab_size=49408, max_seq_len=77):
        self.size = size
        self.img_size = img_size
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Random image
        image = torch.randn(3, self.img_size, self.img_size)

        # Random text tokens
        # In practice, tokenize text with a proper tokenizer
        seq_len = np.random.randint(20, self.max_seq_len)
        text = torch.randint(0, self.vocab_size, (seq_len,))

        # Pad to max_seq_len
        padding_len = self.max_seq_len - seq_len
        text = torch.cat([text, torch.zeros(padding_len, dtype=torch.long)])

        # Create padding mask (True for padding tokens, which have value 0)
        padding_mask = torch.cat([
            torch.zeros(seq_len, dtype=torch.bool),
            torch.ones(padding_len, dtype=torch.bool)
        ])

        return {
            'image': image,
            'text': text,
            'padding_mask': padding_mask
        }


def collate_fn(batch):
    """Custom collate function for batching."""
    images = torch.stack([item['image'] for item in batch])
    texts = torch.stack([item['text'] for item in batch])
    padding_masks = torch.stack([item['padding_mask'] for item in batch])
    return {
        'images': images,
        'texts': texts,
        'padding_masks': padding_masks
    }


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_acc_i2t = 0
    total_acc_t2i = 0
    num_batches = 0

    for batch in dataloader:
        images = batch['images'].to(device)
        texts = batch['texts'].to(device)
        padding_masks = batch['padding_masks'].to(device)

        # Forward pass
        loss, metrics = model(images, texts, padding_masks)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        total_loss += loss.item()
        total_acc_i2t += metrics['acc_i2t']
        total_acc_t2i += metrics['acc_t2i']
        num_batches += 1

    return {
        'loss': total_loss / num_batches,
        'acc_i2t': total_acc_i2t / num_batches,
        'acc_t2i': total_acc_t2i / num_batches
    }


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    total_acc_i2t = 0
    total_acc_t2i = 0
    num_batches = 0

    for batch in dataloader:
        images = batch['images'].to(device)
        texts = batch['texts'].to(device)
        padding_masks = batch['padding_masks'].to(device)

        loss, metrics = model(images, texts, padding_masks)

        total_loss += loss.item()
        total_acc_i2t += metrics['acc_i2t']
        total_acc_t2i += metrics['acc_t2i']
        num_batches += 1

    return {
        'loss': total_loss / num_batches,
        'acc_i2t': total_acc_i2t / num_batches,
        'acc_t2i': total_acc_t2i / num_batches
    }


def main():
    """Main training loop."""

    # Configuration
    config = {
        'embed_dim': 256,
        'image_embed_dim': 384,
        'text_embed_dim': 256,
        'img_size': 224,
        'patch_size': 32,
        'vocab_size': 10000,
        'max_seq_len': 77,
        'vision_depth': 4,
        'text_depth': 4,
        'vision_heads': 6,
        'text_heads': 4,
        'temperature': 0.07,
        'batch_size': 16,
        'num_epochs': 10,
        'lr': 1e-4,
        'weight_decay': 0.01,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    print("=" * 60)
    print("CLIP Training Example")
    print("=" * 60)
    print(f"\nDevice: {config['device']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Epochs: {config['num_epochs']}")
    print(f"Learning rate: {config['lr']}")

    # Create model
    model = CLIP(
        embed_dim=config['embed_dim'],
        image_embed_dim=config['image_embed_dim'],
        text_embed_dim=config['text_embed_dim'],
        img_size=config['img_size'],
        patch_size=config['patch_size'],
        vocab_size=config['vocab_size'],
        max_seq_len=config['max_seq_len'],
        vision_depth=config['vision_depth'],
        text_depth=config['text_depth'],
        vision_heads=config['vision_heads'],
        text_heads=config['text_heads'],
        temperature=config['temperature']
    ).to(config['device'])

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")

    # Create datasets and dataloaders
    train_dataset = DummyCLIPDataset(size=1000, vocab_size=config['vocab_size'])
    val_dataset = DummyCLIPDataset(size=200, vocab_size=config['vocab_size'])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    print(f"\nTrain dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )

    # Optional: Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['num_epochs']
    )

    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    best_val_loss = float('inf')

    for epoch in range(1, config['num_epochs'] + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, config['device'])

        # Evaluate
        val_metrics = evaluate(model, val_loader, config['device'])

        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Print metrics
        print(f"\nEpoch {epoch}/{config['num_epochs']} (LR: {current_lr:.6f})")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
              f"Acc I2T: {train_metrics['acc_i2t']:.4f}, "
              f"Acc T2I: {train_metrics['acc_t2i']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
              f"Acc I2T: {val_metrics['acc_i2t']:.4f}, "
              f"Acc T2I: {val_metrics['acc_t2i']:.4f}")

        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'config': config
            }, 'clip_best.pth')
            print(f"  -> Saved best model (val_loss: {best_val_loss:.4f})")

    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)

    # Test inference
    print("\nTesting inference...")
    model.eval()

    # Create sample batch
    sample_images = torch.randn(2, 3, 224, 224).to(config['device'])
    sample_texts = torch.randint(0, config['vocab_size'], (2, 77)).to(config['device'])

    # Zero-shot classification demo
    class_prompts = torch.randint(0, config['vocab_size'], (5, 77)).to(config['device'])
    probs = model.zero_shot_classification(sample_images, class_prompts)

    print(f"\nImage-text similarity shape: {model.compute_similarity(sample_images, sample_texts).shape}")
    print(f"Zero-shot classification probs shape: {probs.shape}")
    print(f"Predicted classes: {probs.argmax(dim=-1).cpu().numpy()}")


if __name__ == "__main__":
    main()
