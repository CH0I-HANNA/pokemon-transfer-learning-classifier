"""Experiment definitions, batch runner, and result comparison."""

import argparse
import csv
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List

import torch
import matplotlib.pyplot as plt

from dataset import get_dataloaders, set_seed
from model import build_model, count_trainable_params
from train import train
from evaluate import evaluate


# ---------------------------------------------------------------------------
# Experiment configurations
# ---------------------------------------------------------------------------
EXPERIMENTS: List[Dict] = [
    {
        "exp_id":        "EXP-01",
        "backbone":      "resnet50",
        "finetune_scope":"head_only",
        "pretrained":    True,
        "epochs":        30,
        "lr":            1e-4,
        "weight_decay":  1e-4,
        "patience":      5,
        "batch_size":    64,
    },
    {
        "exp_id":        "EXP-02",
        "backbone":      "resnet50",
        "finetune_scope":"full",
        "pretrained":    True,
        "epochs":        30,
        "lr":            1e-4,
        "weight_decay":  1e-4,
        "patience":      5,
        "batch_size":    64,
    },
    {
        "exp_id":        "EXP-03",
        "backbone":      "efficientnet_b0",
        "finetune_scope":"last2_blocks",
        "pretrained":    True,
        "epochs":        30,
        "lr":            1e-4,
        "weight_decay":  1e-4,
        "patience":      5,
        "batch_size":    64,
    },
    {
        "exp_id":        "EXP-04",
        "backbone":      "vit_b16",
        "finetune_scope":"full",
        "pretrained":    True,
        "epochs":        30,
        "lr":            1e-4,
        "weight_decay":  1e-4,
        "patience":      5,
        "batch_size":    32,   # ViT is memory-heavy
    },
    {
        "exp_id":        "EXP-05",
        "backbone":      "resnet50",
        "finetune_scope":"full",
        "pretrained":    False,
        "epochs":        30,
        "lr":            1e-4,
        "weight_decay":  1e-4,
        "patience":      5,
        "batch_size":    64,
    },
]


def run_experiment(
    config: Dict,
    data_dir: str,
    results_root: str,
    models_dir: str,
    device: torch.device,
    num_workers: int = 4,
    num_classes: int = 150,
    seed: int = 42,
) -> Dict:
    """
    Run a single experiment end-to-end: train → evaluate → return metrics.

    Saves model checkpoint to models/<exp_id>_<backbone>.pth.
    Saves results to results/<exp_id>/.
    """
    exp_id = config["exp_id"]
    save_dir = os.path.join(results_root, exp_id)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # Skip if this experiment already completed successfully
    metrics_path = os.path.join(save_dir, "metrics.json")
    if os.path.exists(metrics_path):
        print(f"\n[{exp_id}] Already completed — loading saved metrics.")
        with open(metrics_path) as f:
            return json.load(f)

    set_seed(seed)

    # Log config
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*60}")
    print(f"[{exp_id}] backbone={config['backbone']}  "
          f"scope={config['finetune_scope']}  pretrained={config['pretrained']}")
    print(f"{'='*60}")

    # Data
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_dir=data_dir,
        batch_size=config["batch_size"],
        num_workers=num_workers,
        seed=seed,
    )

    # Model
    model = build_model(
        backbone=config["backbone"],
        num_classes=num_classes,
        pretrained=config["pretrained"],
        finetune_scope=config["finetune_scope"],
    ).to(device)

    print(f"  Trainable params: {count_trainable_params(model):,}")

    # Train
    train(model, train_loader, val_loader, config, save_dir, device)

    # Copy best checkpoint to models/
    best_ckpt = os.path.join(save_dir, f"{exp_id}_best.pth")
    final_ckpt = os.path.join(models_dir, f"{exp_id}_{config['backbone']}.pth")
    if os.path.exists(best_ckpt):
        shutil.copy(best_ckpt, final_ckpt)

    # Evaluate
    metrics = evaluate(model, test_loader, class_names, save_dir, device)
    metrics.update({
        "exp_id":         exp_id,
        "backbone":       config["backbone"],
        "finetune_scope": config["finetune_scope"],
        "pretrained":     config["pretrained"],
    })
    return metrics


def build_comparison(all_metrics: List[Dict], results_root: str) -> None:
    """Save comparison_table.csv and comparison_chart.png from all experiment results."""
    csv_path = os.path.join(results_root, "comparison_table.csv")
    fieldnames = [
        "exp_id", "backbone", "finetune_scope", "pretrained",
        "test_accuracy", "test_precision_macro", "test_recall_macro",
        "test_f1_macro", "top5_accuracy",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_metrics)
    print(f"\nSaved comparison table → {csv_path}")

    # Bar chart
    exp_ids  = [m["exp_id"] for m in all_metrics]
    acc      = [m.get("test_accuracy", 0) for m in all_metrics]
    f1       = [m.get("test_f1_macro", 0) for m in all_metrics]
    top5     = [m.get("top5_accuracy", 0) for m in all_metrics]

    x = range(len(exp_ids))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar([i - width for i in x], acc,  width, label="Top-1 Accuracy")
    ax.bar([i          for i in x], f1,   width, label="F1 (macro)")
    ax.bar([i + width for i in x], top5, width, label="Top-5 Accuracy")

    ax.set_xticks(list(x))
    ax.set_xticklabels(exp_ids)
    ax.set_ylabel("Score")
    ax.set_title("Experiment Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    chart_path = os.path.join(results_root, "comparison_chart.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"Saved comparison chart → {chart_path}")


def main(args: argparse.Namespace) -> None:
    """Entry point: run all (or selected) experiments, then build comparison."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Filter experiments if --exp_ids specified
    experiments = EXPERIMENTS
    if args.exp_ids:
        experiments = [e for e in EXPERIMENTS if e["exp_id"] in args.exp_ids]

    all_metrics: List[Dict] = []

    for config in experiments:
        metrics = run_experiment(
            config=config,
            data_dir=args.data_dir,
            results_root=args.results_dir,
            models_dir=args.models_dir,
            device=device,
            num_workers=args.num_workers,
            num_classes=args.num_classes,
            seed=args.seed,
        )
        all_metrics.append(metrics)

    if len(all_metrics) > 1:
        build_comparison(all_metrics, args.results_dir)

    print("\nAll experiments complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Pokemon classifier experiments")
    parser.add_argument("--data_dir",    default="data/PokemonData")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--models_dir",  default="models")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=150)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument(
        "--exp_ids", nargs="*",
        help="Run specific experiments only, e.g. --exp_ids EXP-01 EXP-02"
    )
    main(parser.parse_args())
