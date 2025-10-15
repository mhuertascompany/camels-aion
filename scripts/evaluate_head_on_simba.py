#!/usr/bin/env python3
"""Evaluate a trained head on SIMBA CAMELS embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from train_parameter_head import load_embeddings, evaluate, PARAMETER_NAMES, build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Checkpoint produced by train_parameter_head.py")
    parser.add_argument("--manifest", type=Path, required=True, help="Manifest JSON listing SIMBA shards.")
    parser.add_argument("--shard-dir", type=Path, required=True, help="Directory containing SIMBA embedding shards.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=Path, required=True, help="Where to write evaluation metrics JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.manifest, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    shard_paths = [args.shard_dir / name for name in manifest["shards"]]

    embeddings, labels = load_embeddings(shard_paths)

    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("metadata", {})
    hidden_dim = metadata.get("hidden_dim")
    model_state = checkpoint["model_state"]

    model = build_model(embeddings.shape[-1], hidden_dim=hidden_dim)
    model.load_state_dict(model_state)
    model = model.to(args.device)

    dataset = TensorDataset(embeddings, labels)
    loader = DataLoader(dataset, batch_size=args.batch_size)

    metrics, predictions, targets = evaluate(
        model, loader, torch.device(args.device), return_outputs=True
    )
    summary = {name: {"mse": mse, "mae": mae} for name, mse, mae in zip(PARAMETER_NAMES, metrics["mse"], metrics["mae"])}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump({"simba": summary}, fh, indent=2)

    import pandas as pd

    df = pd.DataFrame(
        torch.cat([targets.cpu(), predictions.cpu()], dim=1).numpy(),
        columns=[f"target_{name}" for name in PARAMETER_NAMES]
        + [f"pred_{name}" for name in PARAMETER_NAMES],
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = args.output.parent / f"{args.output.stem}_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    latest_path = args.output.parent / f"{args.output.stem}_latest.csv"
    df.to_csv(latest_path, index=False)


if __name__ == "__main__":
    main()
