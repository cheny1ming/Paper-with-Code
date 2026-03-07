"""
Vision Transformer 训练脚本
支持图像分类训练
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from ViT import VisionTransformer, ViTConfig


class Trainer:
    """
    ViT模型训练器
    """
    def __init__(
        self,
        model: VisionTransformer,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        lr: float = 3e-4,
        weight_decay: float = 0.05,
        warmup_epochs: int = 5,
        max_epochs: int = 100,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        save_dir: str = './checkpoints'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.device = device
        self.save_dir = save_dir

        # 损失函数
        self.criterion = nn.CrossEntropyLoss()

        # 优化器（使用AdamW）
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max_epochs,
            eta_min=lr * 0.01
        )

        # 训练统计
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_accs = []

    def warmup_lr(self, current_step: int, total_steps: int):
        """学习率warmup"""
        return self.lr * (current_step + 1) / total_steps

    def train_epoch(self, epoch: int):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        # 设置进度条
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.max_epochs}')

        for batch_idx, (images, labels) in enumerate(pbar):
            # 将数据移到设备
            images = images.to(self.device)
            labels = labels.to(self.device)

            # 前向传播
            outputs = self.model(images)

            # 计算损失
            loss = self.criterion(outputs, labels)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # 统计
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 更新进度条
            pbar.set_postfix({
                'loss': f'{total_loss/(batch_idx+1):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

        # 计算平均损失和准确率
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    def validate(self):
        """验证模型"""
        if self.val_loader is None:
            return None, None

        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc='Validating'):
                # 将数据移到设备
                images = images.to(self.device)
                labels = labels.to(self.device)

                # 前向传播
                outputs = self.model(images)

                # 计算损失
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                # 统计
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        # 计算平均损失和准确率
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    def train(self):
        """完整训练流程"""
        print(f"Starting training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

        start_time = time.time()

        for epoch in range(self.max_epochs):
            self.current_epoch = epoch

            # 学习率warmup
            if epoch < self.warmup_epochs:
                warmup_lr = self.warmup_lr(epoch, self.warmup_epochs)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = warmup_lr

            # 训练一个epoch
            train_loss, train_acc = self.train_epoch(epoch)

            # 验证
            val_loss, val_acc = self.validate()

            # 更新学习率
            if epoch >= self.warmup_epochs:
                self.scheduler.step()

            # 记录统计信息
            self.train_losses.append(train_loss)
            if val_loss is not None:
                self.val_losses.append(val_loss)
                self.val_accs.append(val_acc)

            # 打印统计信息
            print(f'\nEpoch {epoch+1}/{self.max_epochs}:')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            if val_loss is not None:
                print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'Learning Rate: {self.optimizer.param_groups[0]["lr"]:.6f}')

            # 保存最佳模型
            if val_acc is not None and val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_model('best_model.pth')
                print(f'Saved best model with val acc: {val_acc:.2f}%')

            # 定期保存模型
            if (epoch + 1) % 10 == 0:
                self.save_model(f'checkpoint_epoch_{epoch+1}.pth')

        total_time = time.time() - start_time
        print(f'\nTraining completed in {total_time/3600:.2f} hours')
        print(f'Best validation accuracy: {self.best_val_acc:.2f}%')

    def save_model(self, filename: str):
        """保存模型"""
        import os
        os.makedirs(self.save_dir, exist_ok=True)

        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accs': self.val_accs
        }

        torch.save(checkpoint, f'{self.save_dir}/{filename}')

    def load_model(self, filename: str):
        """加载模型"""
        checkpoint = torch.load(f'{self.save_dir}/{filename}', map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint['best_val_acc']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.val_accs = checkpoint['val_accs']

        print(f'Loaded model from epoch {self.current_epoch}')


if __name__ == "__main__":
    """
    示例训练代码
    """
    from torchvision import datasets, transforms

    # 数据预处理
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 创建数据集（这里使用CIFAR-10作为示例）
    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform_train
    )

    val_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform_val
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 创建模型
    config = ViTConfig.vit_base()
    config.num_classes = 10  # CIFAR-10有10个类别

    model = VisionTransformer(
        img_size=config.img_size,
        patch_size=config.patch_size,
        in_channels=config.in_channels,
        num_classes=config.num_classes,
        embed_dim=config.embed_dim,
        depth=config.depth,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio
    )

    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=3e-4,
        weight_decay=0.05,
        warmup_epochs=5,
        max_epochs=100,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # 开始训练
    trainer.train()
