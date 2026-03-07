"""
Vision Transformer 数据处理工具
包含图像数据处理类
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
import os
from typing import List, Dict, Optional, Tuple


class ImageDataset(Dataset):
    """
    图像数据集
    """
    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform=None,
        img_size: int = 224
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.img_size = img_size

        # 默认变换
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 加载图像
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        # 应用变换
        if self.transform:
            image = self.transform(image)

        # 获取标签
        label = self.labels[idx]

        return image, label


class DataModule:
    """
    数据模块 - 管理数据加载
    """
    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def get_image_dataloaders(
        self,
        train_dataset: ImageDataset,
        val_dataset: Optional[ImageDataset] = None
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        """获取图像数据加载器"""
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory
            )

        return train_loader, val_loader


def get_image_transforms(img_size: int = 224, is_training: bool = True):
    """获取图像变换"""
    if is_training:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


# 示例使用
if __name__ == "__main__":
    """
    示例：如何使用数据处理工具
    """
    # 示例1：创建图像数据集
    print("Example 1: Creating Image Dataset")
    image_paths = ["image1.jpg", "image2.jpg"]  # 替换为实际路径
    labels = [0, 1]

    transform = get_image_transforms(img_size=224, is_training=True)
    image_dataset = ImageDataset(
        image_paths=image_paths,
        labels=labels,
        transform=transform
    )

    print(f"Image dataset size: {len(image_dataset)}")

    # 示例2：创建数据加载器
    print("\nExample 2: Creating Data Loaders")
    data_module = DataModule(batch_size=2, num_workers=0)

    train_loader, val_loader = data_module.get_image_dataloaders(
        train_dataset=image_dataset,
        val_dataset=image_dataset
    )

    print(f"Train loader batches: {len(train_loader)}")
    if val_loader:
        print(f"Val loader batches: {len(val_loader)}")
