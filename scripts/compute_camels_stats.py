#!/usr/bin/env python3
"""Estimate per-field statistics for CAMELS maps to support normalization."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from camels_aion.data import CamelsIllustrisDataset

DEFAULT_FIELDS = ("Mstar", "Mgas", "T", "Z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-path", type=Path, required=True, help="Root directory containing CAMELS maps.")
    parser.add_argument("--suite", type=str, default="IllustrisTNG")
    parser.add_argument("--set", dest="set_name", type=str, default="LH")
    parser.add_argument("--redshift", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True, help="Path to the JSON file to write stats to.")
    parser.add_argument("--fields", nargs="*", default=DEFAULT_FIELDS, help="Fields to include (defaults to Mstar, Mgas, T, Z).")
    parser.add_argument("--sample-size", type=int, default=2000, help="Number of maps sampled per field.")
    parser.add_argument("--pixels-per-map", type=int, default=4096, help="Number of pixels sampled per map when estimating stats.")
    parser.add_argument("--clip", type=float, default=1.5, help="Default clip value used during normalization.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def compute_stats(
    dataset: CamelsIllustrisDataset,
    sample_size: int,
    clip: float,
    seed: int,
    pixels_per_map: int,
) -> dict:
    rng = np.random.default_rng(seed)
    num_samples = len(dataset)
    sample_size = min(sample_size, num_samples)

    stats = {
        "metadata": {
        "suite": dataset.suite,
        "set": dataset.set_name,
        "redshift": dataset.redshift,
        "clip": clip,
        "sample_size": sample_size,
        "pixels_per_map": pixels_per_map,
    }
    }

    indices = rng.choice(num_samples, size=sample_size, replace=False)

    for field in dataset.fields:
        maps = dataset._maps[field][indices]
        maps = np.array(maps, dtype=np.float32, copy=False)
        b, h, w = maps.shape
        total_pixels = h * w
        flattened = maps.reshape(b, total_pixels)
        if pixels_per_map < total_pixels:
            pixel_idx = rng.choice(total_pixels, size=pixels_per_map, replace=False)
            flattened = flattened[:, pixel_idx]
        maps = flattened.reshape(-1)

        scale = float(np.median(maps))
        eps = 1e-6
        denom = scale if abs(scale) > eps else eps
        transformed = np.arcsinh(maps / denom)
        low = float(np.percentile(transformed, 1.0))
        high = float(np.percentile(transformed, 99.0))

        stats[field] = {
            "transform": "arcsinh",
            "scale": scale if abs(scale) > eps else eps,
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
        mmap=True,
        apply_normalization=False,
    )

    stats = compute_stats(dataset, args.sample_size, args.clip, args.seed, args.pixels_per_map)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    print(f"Saved statistics to {args.output}")


if __name__ == "__main__":
    main()
