"""Dataset & DataLoader cho bộ ảnh văn hóa Việt Nam."""
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.utils.seed import seed_worker

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif")


class NeuroSymbolicDataset(Dataset):
    """Dataset ảnh, chia theo split train/val/test bằng cấu trúc thư mục con."""

    def __init__(self, data_dir: str, split: str, transform: Optional[Callable] = None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.split_dir = self.data_dir / split
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")

        self.class_to_idx = self._find_classes()
        self.classes = list(self.class_to_idx.keys())
        self.samples: List[Tuple[Path, int]] = []
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = self.split_dir / class_name
            if not class_dir.is_dir():
                continue
            for img_path in class_dir.glob("*"):
                if img_path.suffix.lower() in VALID_EXTENSIONS:
                    self.samples.append((img_path, class_idx))

        self.transform = transform or self.get_transforms(split)

    def _find_classes(self) -> dict:
        classes = sorted(d.name for d in self.split_dir.iterdir() if d.is_dir())
        return {cls: idx for idx, cls in enumerate(classes)}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    @staticmethod
    def get_transforms(split: str) -> transforms.Compose:
        if split == "train":
            return transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(degrees=15),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ]
            )
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )


def create_dataloaders(data_dir: str, batch_size: int = 32, num_workers: int = 4, seed: int = 42):
    """Tạo dataloaders cho train/val/test với seeding đầy đủ (reproducibility)."""
    dataloaders = {}
    for split in ["train", "val", "test"]:
        dataset = NeuroSymbolicDataset(
            data_dir=data_dir, split=split, transform=NeuroSymbolicDataset.get_transforms(split)
        )
        g = torch.Generator()
        g.manual_seed(seed)
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            drop_last=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            generator=g,
            worker_init_fn=seed_worker,
        )
    return dataloaders, dataloaders["train"], dataloaders["val"], dataloaders["test"]
