#!/usr/bin/env python3
"""Train a regression head on AION embeddings to predict Omega_m only."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch import nn
from torch.utils.data import TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import umap

from camels_aion.regression_head import RegressionModel, TokenPooler
from train_parameter_head import load_embeddings, split_indices, make_loaders, compute_feature_stats, evaluate

PARAMETER_NAME = "Omega_m"
PARAMETER_INDEX = 0
PARAMETER_NAMES = [PARAMETER_NAME]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Manifest JSON listing embedding shards.")
    parser.add_argument("--shard-dir", type=Path, required=True, help="Directory containing embedding shards.")
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden size for the regression head (set to 0 for linear).")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of hidden layers in the regression head.")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout probability inside the regression head.")
    parser.add_argument("--pool-type", type=str, choices=["mean", "meanmax", "attention"], default="meanmax", help="Pooling strategy for token embeddings.")
    parser.add_argument("--pool-heads", type=int, default=4, help="Number of attention heads when using attention pooling.")
    parser.add_argument("--pool-dropout", type=float, default=0.1, help="Dropout applied inside the pooling module.")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    #parser.add_argument("--label-standardize", action="store_true", help="Standardize labels using training set mean/std.")
    parser.add_argument("--freeze-pool-epochs", type=int, default=10, help="Number of initial epochs to freeze pooling parameters.")
    parser.add_argument("--scheduler", type=str, choices=["none", "cosine"], default="cosine", help="Learning rate scheduler to use.")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Warmup epochs when using a scheduler.")
    return parser.parse_args()


def save_predictions(output_dir: Path, preds: torch.Tensor, targets: torch.Tensor, suffix: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df = pd.DataFrame(
        torch.cat([targets, preds], dim=1).numpy(),
        columns=[f"target_{PARAMETER_NAME}", f"pred_{PARAMETER_NAME}"],
    )
    csv_path = output_dir / f"{suffix}_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    df.to_csv(output_dir / f"{suffix}_latest.csv", index=False)
    return csv_path


def plot_regression(output_dir: Path, preds: torch.Tensor, targets: torch.Tensor, suffix: str):
    df = pd.DataFrame(
        torch.cat([targets, preds], dim=1).numpy(),
        columns=[f"target_{PARAMETER_NAME}", f"pred_{PARAMETER_NAME}"],
    )
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.scatterplot(
        data=df,
        x=f"target_{PARAMETER_NAME}",
        y=f"pred_{PARAMETER_NAME}",
        ax=ax,
        s=15,
        alpha=0.6,
    )
    min_val = df.min().min()
    max_val = df.max().max()
    ax.plot([min_val, max_val], [min_val, max_val], "--", color="black")
    ax.set_title(PARAMETER_NAME)
    ax.set_xlabel("Target")
    ax.set_ylabel("Prediction")
    plt.tight_layout()
    plt.savefig(output_dir / f"regression_{suffix}.png", dpi=200)
    plt.close(fig)


def plot_umap(output_dir: Path, features: torch.Tensor, targets: torch.Tensor, suffix: str):
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.2, metric="cosine", random_state=42)
    embedding_2d = reducer.fit_transform(features.numpy())
    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=targets[:, 0].numpy(),
        cmap="viridis",
        s=10,
        alpha=0.7,
    )
    ax.set_title(PARAMETER_NAME)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    fig.colorbar(sc, ax=ax, label=PARAMETER_NAME, shrink=0.9)
    plt.tight_layout()
    plt.savefig(output_dir / f"umap_{suffix}.png", dpi=200)
    plt.close(fig)


def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for param in module.parameters():
        param.requires_grad = requires_grad


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str,
    total_epochs: int,
    warmup_epochs: int,
):
    if scheduler_name == "cosine":
        warmup_epochs = max(0, min(warmup_epochs, total_epochs))
        schedules = []
        milestones = []
        if warmup_epochs > 0:
            schedules.append(LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs))
            milestones.append(warmup_epochs)
        cosine_epochs = max(1, total_epochs - warmup_epochs)
        schedules.append(CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=0.0))
        if len(schedules) == 1:
            return schedules[0]
        return SequentialLR(optimizer, schedules, milestones=milestones)
    return None


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    with open(args.manifest, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    shard_paths = [args.shard_dir / name for name in manifest["shards"]]

    embeddings, labels, _ = load_embeddings(shard_paths)
    omega_labels = labels[:, [PARAMETER_INDEX]]

    train_idx, val_idx, test_idx = split_indices(embeddings.shape[0], args.train_frac, args.val_frac, args.seed)

    pool = TokenPooler(
        pool_type=args.pool_type,
        embed_dim=embeddings.shape[-1],
        num_heads=args.pool_heads,
        dropout=args.pool_dropout,
    )

    if args.device.lower().startswith("cuda") and torch.cuda.is_available():
        stats_device = torch.device(args.device)
    else:
        stats_device = torch.device("cpu")
    feat_mean, feat_std = compute_feature_stats(
        pool,
        embeddings,
        train_idx,
        chunk_size=max(args.batch_size, 256),
        device=stats_device,
    )

    label_mean = torch.zeros(len(PARAMETER_NAMES))
    label_std = torch.ones(len(PARAMETER_NAMES))
    if args.label_standardize:
        train_labels = omega_labels[train_idx]
        label_mean = train_labels.mean(dim=0)
        label_std = train_labels.std(dim=0, unbiased=False).clamp_min(1e-6)
        omega_labels = (omega_labels - label_mean) / label_std

    hidden_dims: list[int] = []
    if args.hidden_dim and args.num_layers > 0:
        hidden_dims = [args.hidden_dim] * args.num_layers

    model = RegressionModel(
        input_dim=embeddings.shape[-1],
        num_outputs=len(PARAMETER_NAMES),
        hidden_dims=hidden_dims,
        dropout=args.dropout,
        pool=pool,
        pool_type=args.pool_type,
        pool_heads=args.pool_heads,
        pool_dropout=args.pool_dropout,
        feature_mean=feat_mean,
        feature_std=feat_std,
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    dataset = TensorDataset(embeddings, omega_labels)
    train_loader, val_loader, test_loader = make_loaders(
        dataset, train_idx, val_idx, test_idx, args.batch_size
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, args.scheduler, args.epochs, args.warmup_epochs)

    freeze_epochs = max(0, args.freeze_pool_epochs)
    if freeze_epochs > 0:
        set_requires_grad(model.pool, False)

    best_state = None
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        if freeze_epochs > 0 and epoch == freeze_epochs + 1:
            set_requires_grad(model.pool, True)
        model.train()
        running_loss = 0.0
        for batch_embeddings, batch_labels in train_loader:
            batch_embeddings = batch_embeddings.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            preds = model(batch_embeddings)
            loss = criterion(preds, batch_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_embeddings.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        val_metrics, _, _, _ = evaluate(model, val_loader, device)
        val_loss = sum(val_metrics["mse"]) / len(PARAMETER_NAMES)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:03d} | lr={current_lr:.6e} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": best_val,
                "metadata": {
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                    "dropout": args.dropout,
                    "feature_mean": feat_mean.tolist(),
                    "feature_std": feat_std.tolist(),
                    "label_mean": label_mean.tolist(),
                    "label_std": label_std.tolist(),
                    "train_frac": args.train_frac,
                    "val_frac": args.val_frac,
                    "parameter": PARAMETER_NAME,
                    "pool_frozen_epochs": freeze_epochs,
                    "scheduler": args.scheduler,
                    "warmup_epochs": args.warmup_epochs,
                },
            }
        if scheduler is not None:
            scheduler.step()

    if best_state is None:
        raise RuntimeError("Training failed to produce a valid checkpoint.")

    torch.save(best_state, args.output_dir / "best_model.pt")
    model.load_state_dict(best_state["model_state"])

    test_metrics, preds, targets, features = evaluate(model, test_loader, device)
    with open(args.output_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump({"test": test_metrics, "metadata": best_state["metadata"]}, fh, indent=2)

    csv_path = save_predictions(args.output_dir, preds, targets, "test_predictions")
    print(f"Saved predictions to {csv_path}")
    plot_regression(args.output_dir, preds, targets, "test")
    plot_umap(args.output_dir, features, targets, "test")

    npz_path = args.output_dir / "test_features_latest.npz"
    np.savez(
        npz_path,
        features=features.numpy(),
        targets=targets.numpy(),
        predictions=preds.numpy(),
        indices=test_idx.numpy(),
    )
    print(f"Saved features to {npz_path}")


if __name__ == "__main__":
    main()
