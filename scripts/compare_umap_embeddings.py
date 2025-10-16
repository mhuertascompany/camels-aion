#!/usr/bin/env python3
"""Compare UMAP projections of CAMELS embeddings from two suites."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import umap

PARAMETER_NAMES = ["Omega_m", "sigma8", "A_SN1", "A_SN2", "A_AGN1", "A_AGN2"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref-manifest", type=Path, required=True, help="Manifest JSON for the reference suite (UMAP is fit on this set).")
    parser.add_argument("--ref-shard-dir", type=Path, required=True, help="Directory containing reference embeddings.")
    parser.add_argument("--target-manifest", type=Path, required=True, help="Manifest JSON for the target suite to be projected.")
    parser.add_argument("--target-shard-dir", type=Path, required=True, help="Directory containing target embeddings.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ref-label", type=str, default="IllustrisTNG")
    parser.add_argument("--target-label", type=str, default="SIMBA")
    parser.add_argument("--max-points", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--color-parameter", type=str, default="Omega_m", help="Optional parameter name used for per-dataset colour plots.")
    return parser.parse_args()


def load_embeddings(manifest: Path, shard_dir: Path, max_points: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    with open(manifest, "r", encoding="utf-8") as fh:
        manifest_data = json.load(fh)
    shard_paths = [shard_dir / name for name in manifest_data["shards"]]

    embeddings = []
    labels = []
    total = 0
    rng = np.random.default_rng(seed)

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

    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0)

    if embeddings.shape[0] > max_points:
        idx = rng.choice(embeddings.shape[0], size=max_points, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]

    return embeddings, labels


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_context("talk")

    ref_emb, ref_labels = load_embeddings(args.ref_manifest, args.ref_shard_dir, args.max_points, args.seed)
    tgt_emb, tgt_labels = load_embeddings(args.target_manifest, args.target_shard_dir, args.max_points, args.seed)

    reducer = umap.UMAP(n_neighbors=30, min_dist=0.2, metric="cosine", random_state=args.seed)
    ref_2d = reducer.fit_transform(ref_emb.numpy())
    tgt_2d = reducer.transform(tgt_emb.numpy())

    df = pd.DataFrame(
        np.vstack([ref_2d, tgt_2d]),
        columns=["umap1", "umap2"],
    )
    df["dataset"] = (
        [args.ref_label] * ref_2d.shape[0] + [args.target_label] * tgt_2d.shape[0]
    )

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="umap1", y="umap2", hue="dataset", s=20, alpha=0.6)
    plt.title("UMAP projection by dataset")
    plt.tight_layout()
    plt.savefig(args.output_dir / "umap_dataset_comparison.png", dpi=200)
    plt.close()

    if args.color_parameter:
        if args.color_parameter not in PARAMETER_NAMES:
            raise ValueError(f"Unknown parameter {args.color_parameter}; choose from {PARAMETER_NAMES}")
        idx = PARAMETER_NAMES.index(args.color_parameter)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
        scat = axes[0].scatter(ref_2d[:, 0], ref_2d[:, 1], c=ref_labels[:, idx].numpy(), cmap="viridis", s=15, alpha=0.7)
        axes[0].set_title(f"{args.ref_label} coloured by {args.color_parameter}")
        axes[0].set_xlabel("UMAP-1")
        axes[0].set_ylabel("UMAP-2")

        axes[1].scatter(tgt_2d[:, 0], tgt_2d[:, 1], c=tgt_labels[:, idx].numpy(), cmap="viridis", s=15, alpha=0.7)
        axes[1].set_title(f"{args.target_label} coloured by {args.color_parameter}")
        axes[1].set_xlabel("UMAP-1")

        fig.colorbar(scat, ax=axes, label=args.color_parameter, shrink=0.9)
        plt.tight_layout()
        plt.savefig(args.output_dir / f"umap_{args.color_parameter}_comparison.png", dpi=200)
        plt.close()


if __name__ == "__main__":
    main()
