#!/usr/bin/env python3
"""Train a regression head on AION embeddings to predict CAMELS parameters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PARAMETER_NAMES = ["Omega_m", "sigma8", "A_SN1", "A_SN2", "A_AGN1", "A_AGN2"]


def load_embeddings(shard_paths: Iterable[Path]) -> tuple[torch.Tensor, torch.Tensor]:
    embeddings = []
    labels = []
    for shard_path in shard_paths:
        payload = torch.load(shard_path, weights_only=False)
        embeddings.append(payload["embeddings"])
        labels.append(payload["labels"])
    return torch.cat(embeddings, dim=0), torch.cat(labels, dim=0)


def build_model(input_dim: int, hidden_dim: int | None = None) -> nn.Module:
    if hidden_dim is None:
        return nn.Linear(input_dim, len(PARAMETER_NAMES))
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, len(PARAMETER_NAMES)),
    )


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, list[float]]:
    mse = torch.mean((pred - target) ** 2, dim=0)
    mae = torch.mean(torch.abs(pred - target), dim=0)
    return {
        "mse": [float(value) for value in mse],
        "mae": [float(value) for value in mae],
    }


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, list[float]]:
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch_embeddings, batch_labels in loader:
            batch_embeddings = batch_embeddings.to(device)
            batch_labels = batch_labels.to(device)
            preds.append(model(batch_embeddings))
            targets.append(batch_labels)
    predictions = torch.cat(preds, dim=0)
    target = torch.cat(targets, dim=0)
    return compute_metrics(predictions, target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Manifest JSON listing embedding shards.",
    )
    parser.add_argument(
        "--shard-dir",
        type=Path,
        required=True,
        help="Directory containing embedding shards.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=None, help="Hidden size for MLP head.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    return parser.parse_args()


def split_indices(num_samples: int, train_frac: float, val_frac: float, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(num_samples, generator=g)
    train_end = int(train_frac * num_samples)
    val_end = train_end + int(val_frac * num_samples)
    return perm[:train_end], perm[train_end:val_end], perm[val_end:]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.manifest, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    shard_paths = [args.shard_dir / name for name in manifest["shards"]]

    embeddings, labels = load_embeddings(shard_paths)

    train_idx, val_idx, test_idx = split_indices(
        embeddings.shape[0],
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
    )

    device = torch.device(args.device)
    model = build_model(embeddings.shape[-1], hidden_dim=args.hidden_dim).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def make_loader(idx: torch.Tensor, shuffle: bool = False) -> DataLoader:
        dataset = TensorDataset(embeddings[idx], labels[idx])
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle)

    train_loader = make_loader(train_idx, shuffle=True)
    val_loader = make_loader(val_idx)
    test_loader = make_loader(test_idx)

    best_val_loss = float("inf")
    best_state = None

    metadata = {
        "hidden_dim": args.hidden_dim,
        "input_dim": embeddings.shape[-1],
        "parameter_names": PARAMETER_NAMES,
    }

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
        epoch_loss = running_loss / len(train_loader.dataset)

        val_metrics = evaluate(model, val_loader, device)
        val_loss = sum(val_metrics["mse"]) / len(PARAMETER_NAMES)

        print(f"Epoch {epoch:03d} | train_loss={epoch_loss:.6f} | val_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": best_val_loss,
                "metadata": metadata,
            }

    if best_state is not None:
        torch.save(best_state, args.output_dir / "best_model.pt")

    test_metrics = evaluate(model, test_loader, device)
    summary = {name: {"mse": mse, "mae": mae} for name, mse, mae in zip(PARAMETER_NAMES, test_metrics["mse"], test_metrics["mae"])}
    with open(args.output_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump({"test": summary, "metadata": metadata}, fh, indent=2)


if __name__ == "__main__":
    main()
