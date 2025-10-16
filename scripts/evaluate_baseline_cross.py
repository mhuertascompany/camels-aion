#!/usr/bin/env python3
"""Evaluate a trained baseline CNN/ViT on a different CAMELS suite."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import umap
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from camels_aion.config import CAMELS_FIELDS

from camels_aion.baseline_data import CamelsMapDataset
from camels_aion.baseline_models import build_model

PARAMETER_NAMES = ["Omega_m", "sigma8", "A_SN1", "A_SN2", "A_AGN1", "A_AGN2"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Checkpoint produced by train_baseline_model.py.")
    parser.add_argument("--base-path", type=Path, required=True)
    parser.add_argument("--suite", type=str, default="SIMBA")
    parser.add_argument("--set", dest="set_name", type=str, default="LH")
    parser.add_argument("--redshift", type=float, default=0.0)
    parser.add_argument("--fields", nargs="*", default=None)
    parser.add_argument("--normalization-stats", type=Path, default=None)
    parser.add_argument("--normalization-clip", type=float, default=1.5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    preds, targets, features = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs, feats = model(images, return_features=True)
            preds.append(outputs.cpu())
            targets.append(labels.cpu())
            features.append(feats.cpu())
    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)
    features = torch.cat(features, dim=0)
    mse = torch.mean((preds - targets) ** 2, dim=0)
    mae = torch.mean(torch.abs(preds - targets), dim=0)
    metrics = {"mse": [float(v) for v in mse], "mae": [float(v) for v in mae]}
    return metrics, preds, targets, features


def save_predictions(output_dir: Path, preds: torch.Tensor, targets: torch.Tensor, prefix: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df = pd.DataFrame(
        torch.cat([targets, preds], dim=1).numpy(),
        columns=[f"target_{name}" for name in PARAMETER_NAMES]
        + [f"pred_{name}" for name in PARAMETER_NAMES],
    )
    csv_path = output_dir / f"{prefix}_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    df.to_csv(output_dir / f"{prefix}_latest.csv", index=False)


def plot_regression(output_dir: Path, preds: torch.Tensor, targets: torch.Tensor, prefix: str):
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
    plt.savefig(output_dir / f"regression_{prefix}.png", dpi=200)
    plt.close(fig)


def plot_umap(output_dir: Path, features: torch.Tensor, targets: torch.Tensor, prefix: str):
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
    plt.savefig(output_dir / f"umap_{prefix}.png", dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("metadata", {})
    model_type = metadata.get("model_type")
    fields = metadata.get("fields") or args.fields or CAMELS_FIELDS
    normalization_stats = args.normalization_stats or metadata.get("normalization_stats")
    normalization_clip = args.normalization_clip if args.normalization_stats else metadata.get("normalization_clip", 1.5)

    dataset = CamelsMapDataset(
        fields=fields,
        suite=args.suite,
        set_name=args.set_name,
        redshift=args.redshift,
        base_path=args.base_path,
        normalization_stats=normalization_stats,
        normalization_clip=normalization_clip,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_model(model_type, len(fields), len(PARAMETER_NAMES)).to(device)
    model.load_state_dict(checkpoint["model_state"])

    metrics, preds, targets, features = evaluate(model, loader, device)
    with open(args.output_dir / "cross_metrics.json", "w", encoding="utf-8") as fh:
        json.dump({"metrics": metrics, "metadata": metadata}, fh, indent=2)

    save_predictions(args.output_dir, preds, targets, "cross_predictions")
    plot_regression(args.output_dir, preds, targets, "cross")
    plot_umap(args.output_dir, features, targets, "cross")

    np.savez(
        args.output_dir / "cross_features_latest.npz",
        features=features.numpy(),
        targets=targets.numpy(),
        predictions=preds.numpy(),
    )


if __name__ == "__main__":
    main()
