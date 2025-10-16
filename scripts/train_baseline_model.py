#!/usr/bin/env python3
"""Train CNN/ViT baselines on CAMELS maps."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset
import umap
import seaborn as sns
import matplotlib.pyplot as plt

from camels_aion.baseline_data import CamelsMapDataset
from camels_aion.baseline_models import build_model
from camels_aion.config import CAMELS_FIELDS

PARAMETER_NAMES = ["Omega_m", "sigma8", "A_SN1", "A_SN2", "A_AGN1", "A_AGN2"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["cnn", "vit"], default="cnn")
    parser.add_argument("--base-path", type=Path, required=True)
    parser.add_argument("--suite", type=str, default="IllustrisTNG")
    parser.add_argument("--set", dest="set_name", type=str, default="LH")
    parser.add_argument("--redshift", type=float, default=0.0)
    parser.add_argument("--fields", nargs="*", default=CAMELS_FIELDS)
    parser.add_argument("--normalization-stats", type=Path, default=None)
    parser.add_argument("--normalization-clip", type=float, default=1.5)
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    return parser.parse_args()


def split_indices(num_samples: int, train_frac: float, val_frac: float, seed: int):
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(num_samples, generator=g)
    train_end = int(train_frac * num_samples)
    val_end = train_end + int(val_frac * num_samples)
    return perm[:train_end], perm[train_end:val_end], perm[val_end:]


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    preds, targets, features = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs, feats = model(images, return_features=True)
            preds.append(outputs.cpu())
            features.append(feats.cpu())
            targets.append(labels.cpu())
    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)
    features = torch.cat(features, dim=0)
    mse = torch.mean((preds - targets) ** 2, dim=0)
    mae = torch.mean(torch.abs(preds - targets), dim=0)
    metrics = {
        "mse": [float(v) for v in mse],
        "mae": [float(v) for v in mae],
    }
    return metrics, preds, targets, features


def make_dataloaders(dataset: CamelsMapDataset, train_idx, val_idx, test_idx, batch_size, num_workers):
    train_loader = DataLoader(
        Subset(dataset, train_idx.tolist()),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def save_predictions(output_dir: Path, preds: torch.Tensor, targets: torch.Tensor, suffix: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df = pd.DataFrame(
        torch.cat([targets, preds], dim=1).numpy(),
        columns=[f"target_{name}" for name in PARAMETER_NAMES]
        + [f"pred_{name}" for name in PARAMETER_NAMES],
    )
    csv_path = output_dir / f"{suffix}_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    df.to_csv(output_dir / f"{suffix}_latest.csv", index=False)
    return csv_path


def plot_regression(output_dir: Path, preds: torch.Tensor, targets: torch.Tensor, suffix: str):
    df = pd.DataFrame(
        torch.cat([targets, preds], dim=1).numpy(),
        columns=[f"target_{name}" for name in PARAMETER_NAMES]
        + [f"pred_{name}" for name in PARAMETER_NAMES],
    )
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for ax, name in zip(axes, PARAMETER_NAMES):
        sns.scatterplot(
            data=df,
            x=f"target_{name}",
            y=f"pred_{name}",
            ax=ax,
            s=15,
            alpha=0.6,
        )
        min_val = df[[f"target_{name}", f"pred_{name}"]].min().min()
        max_val = df[[f"target_{name}", f"pred_{name}"]].max().max()
        ax.plot([min_val, max_val], [min_val, max_val], "--", color="black")
        ax.set_title(name)
        ax.set_xlabel("Target")
        ax.set_ylabel("Prediction")
    plt.tight_layout()
    plt.savefig(output_dir / f"regression_{suffix}.png", dpi=200)
    plt.close(fig)


def plot_umap(output_dir: Path, features: torch.Tensor, targets: torch.Tensor, suffix: str):
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.2, metric="cosine", random_state=42)
    embedding_2d = reducer.fit_transform(features.numpy())
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for ax, name, idx in zip(axes, PARAMETER_NAMES, range(len(PARAMETER_NAMES))):
        sc = ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=targets[:, idx].numpy(), cmap="viridis", s=10, alpha=0.7)
        ax.set_title(name)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        fig.colorbar(sc, ax=ax, label=name, shrink=0.9)
    plt.tight_layout()
    plt.savefig(output_dir / f"umap_{suffix}.png", dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    args.fields = list(args.fields)

    dataset = CamelsMapDataset(
        fields=args.fields,
        suite=args.suite,
        set_name=args.set_name,
        redshift=args.redshift,
        base_path=args.base_path,
        normalization_stats=args.normalization_stats,
        normalization_clip=args.normalization_clip,
    )

    train_idx, val_idx, test_idx = split_indices(len(dataset), args.train_frac, args.val_frac, args.seed)
    train_loader, val_loader, test_loader = make_dataloaders(
        dataset,
        train_idx,
        val_idx,
        test_idx,
        args.batch_size,
        args.num_workers,
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, len(args.fields), len(PARAMETER_NAMES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    best_state = None
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        val_metrics, _, _, _ = evaluate(model, val_loader, device)
        val_loss = sum(val_metrics["mse"]) / len(PARAMETER_NAMES)
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": best_val,
                "metadata": {
                    "model_type": args.model,
                    "fields": args.fields,
                    "suite": args.suite,
                    "set": args.set_name,
                    "redshift": args.redshift,
                    "normalization_stats": str(args.normalization_stats) if args.normalization_stats else None,
                    "normalization_clip": args.normalization_clip,
                    "train_frac": args.train_frac,
                    "val_frac": args.val_frac,
                },
            }

    if best_state is None:
        raise RuntimeError("Training failed to produce a checkpoint.")

    torch.save(best_state, args.output_dir / "best_model.pt")
    model.load_state_dict(best_state["model_state"])

    metrics, preds, targets, features = evaluate(model, test_loader, device)
    with open(args.output_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump({"test": metrics, "metadata": best_state["metadata"]}, fh, indent=2)

    csv_path = save_predictions(args.output_dir, preds, targets, "test_predictions")
    print(f"Saved predictions to {csv_path}")
    plot_regression(args.output_dir, preds, targets, "test")
    plot_umap(args.output_dir, features, targets, "test")

    np.savez(
        args.output_dir / "test_features_latest.npz",
        features=features.numpy(),
        targets=targets.numpy(),
        predictions=preds.numpy(),
        indices=test_idx.numpy(),
    )


if __name__ == "__main__":
    main()
