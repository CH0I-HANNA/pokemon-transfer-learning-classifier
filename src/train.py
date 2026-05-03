"""Training loop with early stopping, scheduler, and learning curve export."""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one training epoch and return average loss."""
    model.train()
    total_loss = 0.0
    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate on validation set. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss, correct = 0.0, 0
    for images, labels in tqdm(loader, desc="  Val  ", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        total_loss += criterion(outputs, labels).item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


def save_learning_curve(history: Dict[str, List], save_path: str) -> None:
    """Save training/validation loss and accuracy curves as a PNG."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["train_loss"], label="Train Loss")
    ax1.plot(epochs, history["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curve")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, history["val_acc"], color="green", label="Val Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Validation Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_checkpoint(
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    history: Dict[str, List],
    best_val_loss: float,
    patience_counter: int,
    save_dir: str,
    exp_id: str,
    completed: bool = False,
) -> None:
    """Save full training state so training can be resumed from this epoch."""
    path = os.path.join(save_dir, f"{exp_id}_checkpoint_epoch{epoch:03d}.pth")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "history": history,
            "best_val_loss": best_val_loss,
            "patience_counter": patience_counter,
            "completed": completed,
        },
        path,
    )
    # Keep only the latest checkpoint to save disk space
    for old in Path(save_dir).glob(f"{exp_id}_checkpoint_epoch*.pth"):
        if old != Path(path):
            old.unlink()


def load_latest_checkpoint(
    save_dir: str,
    exp_id: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
) -> Tuple[int, Dict[str, List], float, int]:
    """
    Load the latest checkpoint for this experiment if one exists.

    Returns:
        (start_epoch, history, best_val_loss, patience_counter)
        start_epoch is 1 when no checkpoint is found.
    """
    checkpoints = sorted(Path(save_dir).glob(f"{exp_id}_checkpoint_epoch*.pth"))
    if not checkpoints:
        return 1, {"train_loss": [], "val_loss": [], "val_acc": []}, float("inf"), 0

    ckpt = torch.load(checkpoints[-1], map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    # If training already finished (normal end or early stopping), signal skip
    if ckpt.get("completed", False):
        print(f"[{exp_id}] Training already completed at epoch {ckpt['epoch']}. Skipping.")
        return ckpt["epoch"] + 99999, ckpt["history"], ckpt["best_val_loss"], ckpt["patience_counter"]

    start_epoch = ckpt["epoch"] + 1
    print(f"Resumed from checkpoint: epoch {ckpt['epoch']} ({checkpoints[-1].name})")
    return start_epoch, ckpt["history"], ckpt["best_val_loss"], ckpt["patience_counter"]


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict,
    save_dir: str,
    device: torch.device,
    resume: bool = True,
) -> Dict[str, List]:
    """
    Full training loop with early stopping, scheduler, and per-epoch checkpointing.

    Args:
        model: Model to train.
        train_loader / val_loader: DataLoaders.
        config: Dict with keys: epochs, lr, weight_decay, patience, exp_id.
        save_dir: Directory to save best model and learning curve.
        device: torch.device.
        resume: If True, automatically resume from the latest checkpoint when found.

    Returns:
        history dict with train_loss, val_loss, val_acc per epoch.
    """
    os.makedirs(save_dir, exist_ok=True)

    epochs       = config.get("epochs", 30)
    lr           = config.get("lr", 1e-4)
    weight_decay = config.get("weight_decay", 1e-4)
    patience     = config.get("patience", 5)
    exp_id       = config.get("exp_id", "EXP")

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    best_model_path = os.path.join(save_dir, f"{exp_id}_best.pth")

    # ── Resume from checkpoint if available ──────────────────────────────
    if resume:
        start_epoch, history, best_val_loss, patience_counter = load_latest_checkpoint(
            save_dir, exp_id, model, optimizer, scheduler, device
        )
    else:
        start_epoch, history, best_val_loss, patience_counter = (
            1, {"train_loss": [], "val_loss": [], "val_acc": []}, float("inf"), 0
        )

    if start_epoch > epochs:
        print("Training already complete (all epochs done).")
        return history

    for epoch in range(start_epoch, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1

        # Save checkpoint every epoch (overwrites previous to save disk)
        save_checkpoint(
            epoch, model, optimizer, scheduler,
            history, best_val_loss, patience_counter,
            save_dir, exp_id,
        )

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            # Mark as completed so resume skips this experiment
            save_checkpoint(
                epoch, model, optimizer, scheduler,
                history, best_val_loss, patience_counter,
                save_dir, exp_id, completed=True,
            )
            break

    else:
        # Normal end (all epochs finished) — mark completed
        save_checkpoint(
            epochs, model, optimizer, scheduler,
            history, best_val_loss, patience_counter,
            save_dir, exp_id, completed=True,
        )

    # Load best weights before returning
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Save learning curve and history
    save_learning_curve(history, os.path.join(save_dir, "learning_curve.png"))
    with open(os.path.join(save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    return history
