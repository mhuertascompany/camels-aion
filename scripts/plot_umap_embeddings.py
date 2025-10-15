#!/usr/bin/env python3
"""Generate UMAP plots of CAMELS embeddings colour-coded by target parameters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import umap

PARAMETER_NAMES = ["Omega_m", "sigma8", "A_SN1", "A_SN2", "A_AGN1", "A_AGN2"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", type=Path, required=True, help="Directory containing embedding shards.")
    parser.add_argument("--manifest", type=Path, required=True, help="Manifest JSON listing the shard files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where plots will be saved.")
    parser.add_argument("--max-points", type=int, default=5000, help="Maximum number of points to visualise.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_embeddings(manifest_path: Path, shard_dir: Path, max_points: int) -> tuple[torch.Tensor, torch.Tensor]:
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    shard_paths = [shard_dir / name for name in manifest["shards"]]

    embeddings = []
    labels = []
    total = 0
    for shard in shard_paths:
        payload = torch.load(shard, weights_only=False)
        emb = payload["embeddings"].float()
        if emb.ndim == 3:
            emb = emb.mean(dim=1)
        lab = payload["labels"].float()

        embeddings.append(emb)
        labels.append(lab)
        total += emb.shape[0]
        if total >= max_points:
            break

    embeddings = torch.cat(embeddings, dim=0)[:max_points]
    labels = torch.cat(labels, dim=0)[:max_points]
    return embeddings, labels


def plot_umap(embedding_2d: np.ndarray, labels: torch.Tensor, output_dir: Path) -> None:
    sns.set_context("talk")
    output_dir.mkdir(parents=True, exist_ok=True)

    n_cols = 3
    n_rows = int(np.ceil(len(PARAMETER_NAMES) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows), squeeze=False)

    for ax, name in zip(axes.flatten(), PARAMETER_NAMES):
        idx = PARAMETER_NAMES.index(name)
        sc = ax.scatter(
            embedding_2d[:, 0],
            embedding_2d[:, 1],
            c=labels[:, idx].numpy(),
            cmap="viridis",
            s=10,
            linewidths=0,
        )
        ax.set_title(name)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        fig.colorbar(sc, ax=ax, label=name)

    for ax in axes.flatten()[len(PARAMETER_NAMES):]:
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(output_dir / "umap_parameters.png", dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    embeddings, labels = load_embeddings(args.manifest, args.shard_dir, args.max_points)

    reducer = umap.UMAP(n_neighbors=30, min_dist=0.2, metric="cosine", random_state=args.seed)
    embedding_2d = reducer.fit_transform(embeddings.numpy())

    plot_umap(embedding_2d, labels, args.output_dir)


if __name__ == "__main__":
    main()
