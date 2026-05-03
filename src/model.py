"""Model definitions with selectable backbone and fine-tuning scope."""

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
from torchvision import models
import timm
from typing import Optional


def build_model(
    backbone: str,
    num_classes: int = 150,
    pretrained: bool = True,
    finetune_scope: str = "full",
) -> nn.Module:
    """
    Build a classification model with the specified backbone and fine-tuning scope.

    Args:
        backbone: One of 'resnet50', 'efficientnet_b0', 'vit_b16'.
        num_classes: Number of output classes.
        pretrained: If True, load ImageNet weights; otherwise random init.
        finetune_scope: 'head_only', 'last2_blocks', or 'full'.

    Returns:
        Configured PyTorch model.
    """
    model = _load_backbone(backbone, num_classes, pretrained)
    _apply_finetune_scope(model, backbone, finetune_scope)
    return model


def _load_backbone(backbone: str, num_classes: int, pretrained: bool) -> nn.Module:
    """Load backbone and replace the classification head."""
    weights_arg = "IMAGENET1K_V1" if pretrained else None

    if backbone == "resnet50":
        model = models.resnet50(weights=weights_arg)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights_arg)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif backbone == "vit_b16":
        # Use timm for ViT-B/16 with standard ImageNet pretrain
        model = timm.create_model(
            "vit_base_patch16_224",
            pretrained=pretrained,
            num_classes=num_classes,
        )

    else:
        raise ValueError(f"Unknown backbone: {backbone}. Choose from resnet50, efficientnet_b0, vit_b16.")

    return model


def _apply_finetune_scope(model: nn.Module, backbone: str, scope: str) -> None:
    """Freeze parameters according to the fine-tuning scope (in-place)."""
    if scope == "full":
        for param in model.parameters():
            param.requires_grad = True
        return

    if scope == "head_only":
        # Freeze everything first
        for param in model.parameters():
            param.requires_grad = False

        if backbone == "resnet50":
            for param in model.fc.parameters():
                param.requires_grad = True
        elif backbone == "efficientnet_b0":
            for param in model.classifier.parameters():
                param.requires_grad = True
        elif backbone == "vit_b16":
            for param in model.head.parameters():
                param.requires_grad = True

    elif scope == "last2_blocks":
        # Freeze everything, then unfreeze last 2 blocks + head
        for param in model.parameters():
            param.requires_grad = False

        if backbone == "resnet50":
            for param in model.layer4.parameters():
                param.requires_grad = True
            for param in model.fc.parameters():
                param.requires_grad = True

        elif backbone == "efficientnet_b0":
            # EfficientNet-B0 has 9 MBConv blocks; unfreeze last 2
            blocks = list(model.features.children())
            for block in blocks[-2:]:
                for param in block.parameters():
                    param.requires_grad = True
            for param in model.classifier.parameters():
                param.requires_grad = True

        elif backbone == "vit_b16":
            # ViT has 12 transformer blocks; unfreeze last 2
            for block in model.blocks[-2:]:
                for param in block.parameters():
                    param.requires_grad = True
            for param in model.head.parameters():
                param.requires_grad = True

    else:
        raise ValueError(f"Unknown finetune_scope: {scope}. Choose from head_only, last2_blocks, full.")


def count_trainable_params(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
