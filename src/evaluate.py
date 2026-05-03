"""Evaluation: metrics, confusion matrix, and error sample visualization."""

import json
import os
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from tqdm import tqdm
from PIL import Image


@torch.no_grad()
def get_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference on the full loader.

    Returns:
        all_labels, all_preds, all_probs  (all numpy arrays)
    """
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    for images, labels in tqdm(loader, desc="  Eval "):
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)

        all_labels.extend(labels.numpy())
        all_preds.extend(preds)
        all_probs.extend(probs)

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def top5_accuracy(labels: np.ndarray, probs: np.ndarray) -> float:
    """Compute top-5 accuracy from probability matrix."""
    top5_preds = np.argsort(probs, axis=1)[:, -5:]
    correct = sum(labels[i] in top5_preds[i] for i in range(len(labels)))
    return correct / len(labels)


def compute_metrics(
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
) -> Dict[str, float]:
    """Compute and return all evaluation metrics as a dict."""
    return {
        "test_accuracy":         float(accuracy_score(labels, preds)),
        "test_precision_macro":  float(precision_score(labels, preds, average="macro", zero_division=0)),
        "test_recall_macro":     float(recall_score(labels, preds, average="macro", zero_division=0)),
        "test_f1_macro":         float(f1_score(labels, preds, average="macro", zero_division=0)),
        "top5_accuracy":         float(top5_accuracy(labels, probs)),
    }


def save_confusion_matrix(
    labels: np.ndarray,
    preds: np.ndarray,
    class_names: List[str],
    save_path: str,
) -> None:
    """Save 150×150 confusion matrix as a PNG (log-scaled for visibility)."""
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(20, 18))
    sns.heatmap(
        np.log1p(cm),
        ax=ax,
        cmap="Blues",
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"label": "log(count + 1)"},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (log scale)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_error_samples(
    loader: DataLoader,
    model: nn.Module,
    class_names: List[str],
    device: torch.device,
    save_path: str,
    n_samples: int = 5,
) -> None:
    """Save a grid of n_samples misclassified images with true/predicted labels."""
    model.eval()
    errors: List[Tuple] = []  # (img_tensor, true_label, pred_label)

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            preds = outputs.argmax(dim=1).cpu()
            wrong = (preds != labels).nonzero(as_tuple=True)[0]
            for idx in wrong:
                if len(errors) >= n_samples:
                    break
                errors.append((images[idx], labels[idx].item(), preds[idx].item()))
            if len(errors) >= n_samples:
                break

    if not errors:
        return

    fig, axes = plt.subplots(1, len(errors), figsize=(4 * len(errors), 4))
    if len(errors) == 1:
        axes = [axes]

    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    for ax, (img_t, true_lbl, pred_lbl) in zip(axes, errors):
        img_np = img_t.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np * std + mean, 0, 1)
        ax.imshow(img_np)
        ax.set_title(
            f"True: {class_names[true_lbl]}\nPred: {class_names[pred_lbl]}",
            fontsize=8,
            color="red",
        )
        ax.axis("off")

    plt.suptitle("Misclassified Samples", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    class_names: List[str],
    save_dir: str,
    device: torch.device,
) -> Dict[str, float]:
    """
    Full evaluation pipeline: metrics + confusion matrix + error samples.

    Returns:
        metrics dict (also saved to metrics.json)
    """
    os.makedirs(save_dir, exist_ok=True)

    labels, preds, probs = get_predictions(model, test_loader, device)
    metrics = compute_metrics(labels, preds, probs)

    print("  Test metrics:")
    for k, v in metrics.items():
        print(f"    {k}: {v:.4f}")

    # Save metrics
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Confusion matrix
    save_confusion_matrix(labels, preds, class_names,
                          os.path.join(save_dir, "confusion_matrix.png"))

    # Error samples
    save_error_samples(test_loader, model, class_names, device,
                       os.path.join(save_dir, "error_samples.png"))

    return metrics
