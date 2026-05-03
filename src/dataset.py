"""Dataset loading, augmentation, and train/val/test splitting for Pokemon classification."""

import os
import random
import numpy as np
from pathlib import Path
from typing import Tuple, List

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from PIL import Image


def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_train_transform() -> transforms.Compose:
    """Return augmentation transform for training set."""
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_val_transform() -> transforms.Compose:
    """Return deterministic transform for validation and test sets."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


class TransformSubset(Dataset):
    """Wraps a Subset and applies a given transform, overriding the parent dataset's transform."""

    def __init__(self, subset: Subset, transform: transforms.Compose) -> None:
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img, label = self.subset[idx]
        if self.transform:
            # subset returns PIL image when parent dataset has transform=None
            img = self.transform(img)
        return img, label


def load_datasets(
    data_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[Dataset, Dataset, Dataset, List[str]]:
    """
    Load PokemonData and split into train/val/test using stratified split.

    Returns:
        train_dataset, val_dataset, test_dataset, class_names
    """
    set_seed(seed)

    # Load all images without transform so subsets can apply their own
    full_dataset = ImageFolder(root=data_dir, transform=None)
    class_names = full_dataset.classes
    targets = full_dataset.targets

    indices = list(range(len(full_dataset)))
    test_ratio = 1.0 - train_ratio - val_ratio

    # Stratified train / (val+test) split
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=(val_ratio + test_ratio),
        stratify=targets,
        random_state=seed,
    )

    temp_targets = [targets[i] for i in temp_idx]
    relative_test = test_ratio / (val_ratio + test_ratio)

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=relative_test,
        stratify=temp_targets,
        random_state=seed,
    )

    train_dataset = TransformSubset(Subset(full_dataset, train_idx), get_train_transform())
    val_dataset   = TransformSubset(Subset(full_dataset, val_idx),   get_val_transform())
    test_dataset  = TransformSubset(Subset(full_dataset, test_idx),  get_val_transform())

    return train_dataset, val_dataset, test_dataset, class_names


def get_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Build DataLoaders for train, val, and test splits.

    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    train_ds, val_ds, test_ds, class_names = load_datasets(data_dir, seed=seed)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader, class_names


def load_class_names(data_dir: str = "data/PokemonData") -> List[str]:
    """Return sorted list of class names from the dataset directory."""
    return sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])
