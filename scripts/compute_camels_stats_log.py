#!/usr/bin/env python3
"""Compute log-scaled normalization statistics for CAMELS maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from camels_aion.data import CamelsIllustrisDataset

DEFAULT_FIELDS = ("Mstar", "Mgas", "T", "Z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-path", type=Path, required=True)
    parser.add_argument("--suite", type=str, default="IllustrisTNG")
    parser.add_argument("--set", dest="set_name", type=str, default="LH")
    parser.add_argument("--redshift", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fields", nargs="*", default=DEFAULT_FIELDS)
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--pixels-per-map", type=int, default=4096)
    parser.add_argument("--clip", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def compute_stats(dataset: CamelsIllustrisDataset, sample_size: int, pixels_per_map: int, clip: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    num_samples = len(dataset)
    sample_size = min(sample_size, num_samples)

    stats: dict[str, dict] = {
        "metadata": {
            "suite": dataset.suite,
            "set": dataset.set_name,
            "redshift": dataset.redshift,
            "clip": clip,
            "sample_size": sample_size,
            "pixels_per_map": pixels_per_map,
            "transform": "log1p",
        }
    }

    idx = rng.choice(num_samples, size=sample_size, replace=False)

    for field in dataset.fields:
        maps = np.array(dataset._maps[field][idx], dtype=np.float32, copy=False)
        b, h, w = maps.shape
        total_pixels = h * w
        flat = maps.reshape(b, total_pixels)
        if pixels_per_map < total_pixels:
            pix_idx = rng.choice(total_pixels, size=pixels_per_map, replace=False)
            flat = flat[:, pix_idx]
        values = flat.reshape(-1)

        scale = np.median(values)
        eps = 1e-6
        denom = scale if abs(scale) > eps else eps
        transformed = np.log1p(values / denom)
        low = float(np.percentile(transformed, 1.0))
        high = float(np.percentile(transformed, 99.0))

        stats[field] = {
            "transform": "log1p",
            "scale": float(denom),
            "low": low,
            "high": high,
            "clip": clip,
            "eps": eps,
            "pixels_per_map": pixels_per_map,
        }

    return stats


def main() -> None:
    args = parse_args()
    dataset = CamelsIllustrisDataset(
        fields=args.fields,
        suite=args.suite,
        set_name=args.set_name,
        redshift=args.redshift,
        base_path=args.base_path,
        apply_normalization=False,
    )

    stats = compute_stats(dataset, args.sample_size, args.pixels_per_map, args.clip, args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    print(f"Saved log-scale stats to {args.output}")


if __name__ == "__main__":
    main()
