#!/usr/bin/env python3
"""Train a regression head on AION embeddings to predict CAMELS parameters."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, Subset
import umap

from camels_aion.regression_head import RegressionModel, TokenPooler

PARAMETER_NAMES = ["Omega_m", "sigma8", "A_SN1", "A_SN2", "A_AGN1", "A_AGN2"]


def load_embeddings(shard_paths: Iterable[Path]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    embeddings, labels, indices_list = [], [], []
    offset = 0
    for shard_path in shard_paths:
        payload = torch.load(shard_path, weights_only=False)
        emb = payload["embeddings"].float()
        embeddings.append(emb)
        labels.append(payload["labels"].float())
        if "indices" in payload:
            idx = payload["indices"].long()
        else:
            idx = torch.arange(emb.shape[0]) + offset
        indices_list.append(idx)
        offset += emb.shape[0]
    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0)
    indices = torch.cat(indices_list, dim=0) if indices_list else torch.arange(embeddings.shape[0])
    return embeddings, labels, indices


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, list[float]]:
    mse = torch.mean((pred - target) ** 2, dim=0)
    mae = torch.mean(torch.abs(pred - target), dim=0)
    return {
        "mse": [float(value) for value in mse],
        "mae": [float(value) for value in mae],
    }


def evaluate(
    model: RegressionModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, list[float]], torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    preds, targets, features = [], [], []
    with torch.no_grad():
        for batch_embeddings, batch_labels in loader:
            batch_embeddings = batch_embeddings.to(device)
            batch_labels = batch_labels.to(device)
            outputs, feats = model(batch_embeddings, return_features=True)
            preds.append(outputs.cpu())
            targets.append(batch_labels.cpu())
            features.append(feats.cpu())
    predictions = torch.cat(preds, dim=0)
    target = torch.cat(targets, dim=0)
    feats = torch.cat(features, dim=0)
    metrics = compute_metrics(predictions, target)
    return metrics, predictions, target, feats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Manifest JSON listing embedding shards.")
    parser.add_argument("--shard-dir", type=Path, required=True, help="Directory containing embedding shards.")
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden size for the regression head (set to 0 for linear).")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of hidden layers in the regression head.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability inside the regression head.")
    parser.add_argument("--pool-type", type=str, choices=["mean", "meanmax", "attention"], default="attention", help="Pooling strategy for token embeddings.")
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
    parser.add_argument("--label-standardize", action="store_true", help="Standardize labels using training set mean/std.")
    return parser.parse_args()


def split_indices(num_samples: int, train_frac: float, val_frac: float, seed: int):
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(num_samples, generator=g)
    train_end = int(train_frac * num_samples)
    val_end = train_end + int(val_frac * num_samples)
    return perm[:train_end], perm[train_end:val_end], perm[val_end:]


def make_loaders(dataset: TensorDataset, train_idx, val_idx, test_idx, batch_size: int):
    train_loader = DataLoader(
        Subset(dataset, train_idx.tolist()),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx.tolist()),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx.tolist()),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
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

    with open(args.manifest, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    shard_paths = [args.shard_dir / name for name in manifest["shards"]]

    embeddings, labels, _ = load_embeddings(shard_paths)

    train_idx, val_idx, test_idx = split_indices(embeddings.shape[0], args.train_frac, args.val_frac, args.seed)

    pool = TokenPooler(
        pool_type=args.pool_type,
        embed_dim=embeddings.shape[-1],
        num_heads=args.pool_heads,
        dropout=args.pool_dropout,
    )

    pooled_train = pool(embeddings[train_idx])
    feat_mean = pooled_train.mean(dim=0)
    feat_std = pooled_train.std(dim=0, unbiased=False).clamp_min(1e-6)

    label_mean = torch.zeros(len(PARAMETER_NAMES))
    label_std = torch.ones(len(PARAMETER_NAMES))
    if args.label_standardize:
        train_labels = labels[train_idx]
        label_mean = train_labels.mean(dim=0)
        label_std = train_labels.std(dim=0, unbiased=False).clamp_min(1e-6)
        labels = (labels - label_mean) / label_std

    hidden_dims = []
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

    dataset = TensorDataset(embeddings, labels)
    train_loader, val_loader, test_loader = make_loaders(dataset, train_idx, val_idx, test_idx, args.batch_size)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
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
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

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
                },
            }

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
    np.savez(npz_path, features=features.numpy(), targets=targets.numpy(), predictions=preds.numpy(), indices=test_idx.numpy())
    print(f"Saved features to {npz_path}")


if __name__ == "__main__":
    main()
