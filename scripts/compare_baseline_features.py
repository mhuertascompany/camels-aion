#!/usr/bin/env python3
"""Compare baseline features from two suites using a shared UMAP projection."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import umap
import matplotlib.pyplot as plt

PARAMETER_NAMES = ["Omega_m", "sigma8", "A_SN1", "A_SN2", "A_AGN1", "A_AGN2"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref-features", type=Path, required=True, help="NPZ file with features/targets (reference suite).")
    parser.add_argument("--target-features", type=Path, required=True, help="NPZ file with features/targets (target suite).")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ref-label", type=str, default="IllustrisTNG")
    parser.add_argument("--target-label", type=str, default="SIMBA")
    parser.add_argument("--color-parameter", type=str, default="Omega_m")
    parser.add_argument("--max-points", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_npz(path: Path, max_points: int | None, seed: int):
    data = np.load(path)
    features = data["features"]
    targets = data.get("targets")
    if targets is None:
        raise ValueError(f"{path} does not contain 'targets'")
    if max_points is not None and features.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(features.shape[0], size=max_points, replace=False)
        features = features[idx]
        targets = targets[idx]
    return features, targets


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_context("talk")

    ref_features, ref_targets = load_npz(args.ref_features, args.max_points, args.seed)
    tgt_features, tgt_targets = load_npz(args.target_features, args.max_points, args.seed)

    reducer = umap.UMAP(n_neighbors=30, min_dist=0.2, metric="cosine", random_state=args.seed)
    ref_2d = reducer.fit_transform(ref_features)
    tgt_2d = reducer.transform(tgt_features)

    df = pd.DataFrame(
        np.vstack([ref_2d, tgt_2d]),
        columns=["umap1", "umap2"],
    )
    df["dataset"] = [args.ref_label] * ref_2d.shape[0] + [args.target_label] * tgt_2d.shape[0]

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="umap1", y="umap2", hue="dataset", s=20, alpha=0.6)
    plt.title("Baseline features UMAP (dataset comparison)")
    plt.tight_layout()
    plt.savefig(args.output_dir / "baseline_umap_dataset.png", dpi=200)
    plt.close()

    if args.color_parameter:
        if args.color_parameter not in PARAMETER_NAMES:
            raise ValueError(f"Unknown parameter {args.color_parameter}; choose from {PARAMETER_NAMES}")
        idx = PARAMETER_NAMES.index(args.color_parameter)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
        sc = axes[0].scatter(ref_2d[:, 0], ref_2d[:, 1], c=ref_targets[:, idx], cmap="viridis", s=15, alpha=0.7)
        axes[0].set_title(f"{args.ref_label} coloured by {args.color_parameter}")
        axes[0].set_xlabel("UMAP-1")
        axes[0].set_ylabel("UMAP-2")

        axes[1].scatter(tgt_2d[:, 0], tgt_2d[:, 1], c=tgt_targets[:, idx], cmap="viridis", s=15, alpha=0.7)
        axes[1].set_title(f"{args.target_label} coloured by {args.color_parameter}")
        axes[1].set_xlabel("UMAP-1")
        fig.colorbar(sc, ax=axes, label=args.color_parameter, shrink=0.9)
        plt.tight_layout()
        plt.savefig(args.output_dir / f"baseline_umap_{args.color_parameter}.png", dpi=200)
        plt.close()


if __name__ == "__main__":
    main()
