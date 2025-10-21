#!/usr/bin/env python3
"""Export CAMELS maps into shards tailored for codec training.

The script mirrors AION's expected codec inputs by writing torch tensor
payloads containing flux maps and accompanying metadata. Each shard bundles:
    - `flux`: Float tensor of shape (B, C, H, W) in the order defined by
      `CAMELS_CODEC_BANDS`.
    - `indices`: Original map indices within the CAMELS archive.
    - `labels`: Corresponding cosmological/astrophysical parameters.
    - `metadata`: Split name, field list, normalization information.

Usage example:
    python codec_training/scripts/prepare_camels_codec_data.py \
        --output-dir codec_training/data/illustris_codec \
        --suite IllustrisTNG --set LH --redshift 0.0 \
        --normalization-stats /path/to/stats.json \
        --train-frac 0.9 --val-frac 0.1 \
        --batch-size 64 --seed 42

The resulting directory contains `train/` and `val/` subfolders along with
manifest JSON files listing shard names for each split.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import torch
from tqdm.auto import tqdm

from camels_aion.config import CAMELS_CODEC_BANDS, CAMELS_FIELDS
from camels_aion.data import CamelsIllustrisDataset, CamelsMapSample


@dataclass(frozen=True)
class SplitSpec:
    name: str
    indices: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-path", type=Path, default=None, help="Override CAMELS base dataset location.")
    parser.add_argument("--suite", type=str, default="IllustrisTNG", help="Simulation suite (IllustrisTNG, SIMBA, ...).")
    parser.add_argument("--set", dest="set_name", type=str, default="LH", help="Simulation subset (LH, CV, 1P, ...).")
    parser.add_argument("--redshift", type=float, default=0.0, help="Redshift slice to load.")
    parser.add_argument(
        "--fields",
        type=str,
        nargs="+",
        default=list(CAMELS_FIELDS),
        help="Ordered list of map fields to stack. Defaults to canonical four-channel layout.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Root directory where shards and manifests are written.")
    parser.add_argument("--batch-size", type=int, default=64, help="Number of maps per shard.")
    parser.add_argument("--normalization-stats", type=Path, default=None, help="JSON file with per-field normalization settings.")
    parser.add_argument("--normalization-clip", type=float, default=1.5, help="Clip value after normalization prior to codec mapping.")
    parser.add_argument("--train-frac", type=float, default=0.9, help="Fraction of samples assigned to the training split.")
    parser.add_argument("--val-frac", type=float, default=0.1, help="Fraction of samples assigned to the validation split.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed controlling the split permutation.")
    parser.add_argument("--manifest-name", type=str, default="manifest.json", help="Filename for per-split shard manifests.")
    return parser.parse_args()


def validate_splits(train_frac: float, val_frac: float) -> None:
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train-frac must lie strictly between 0 and 1.")
    if not 0.0 < val_frac < 1.0:
        raise ValueError("val-frac must lie strictly between 0 and 1.")
    if train_frac + val_frac > 1.0 + 1e-6:
        raise ValueError("train-frac + val-frac must not exceed 1.0.")


def generate_splits(num_samples: int, train_frac: float, val_frac: float, seed: int) -> list[SplitSpec]:
    validate_splits(train_frac, val_frac)
    generator = torch.Generator()
    generator.manual_seed(seed)
    permutation = torch.randperm(num_samples, generator=generator)

    train_end = int(math.floor(train_frac * num_samples))
    val_end = train_end + int(math.floor(val_frac * num_samples))

    train_idx = permutation[:train_end]
    val_idx = permutation[train_end:val_end]
    holdout_idx = permutation[val_end:]

    splits: list[SplitSpec] = [
        SplitSpec(name="train", indices=train_idx),
        SplitSpec(name="val", indices=val_idx),
    ]
    if holdout_idx.numel() > 0:
        splits.append(SplitSpec(name="holdout", indices=holdout_idx))
    return splits


def iter_subset(dataset: CamelsIllustrisDataset, indices: torch.Tensor, batch_size: int) -> Iterator[CamelsMapSample]:
    if indices.numel() == 0:
        return iter(())
    sorted_indices = indices.tolist()
    sorted_indices.sort()
    return dataset.iter_batches(batch_size=batch_size, indices=sorted_indices)


def write_shard(output_dir: Path, prefix: str, sample: CamelsMapSample, split: str, bands: Sequence[str]) -> Path:
    start_idx = sample.indices[0]
    end_idx = sample.indices[-1]
    shard_name = f"{prefix}_{split}_{start_idx:05d}-{end_idx:05d}.pt"
    shard_path = output_dir / shard_name
    payload = {
        "flux": torch.from_numpy(sample.images).float(),
        "indices": torch.tensor(sample.indices, dtype=torch.long),
        "labels": torch.from_numpy(sample.labels).float(),
        "metadata": {
            "split": split,
            "bands": list(bands),
        },
    }
    torch.save(payload, shard_path)
    return shard_path


def save_manifest(directory: Path, manifest_name: str, shard_paths: Sequence[Path]) -> None:
    manifest_path = directory / manifest_name
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump({"shards": [path.name for path in shard_paths]}, fh, indent=2)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = CamelsIllustrisDataset(
        fields=args.fields,
        suite=args.suite,
        set_name=args.set_name,
        redshift=args.redshift,
        base_path=args.base_path,
        normalization_stats=args.normalization_stats,
        normalization_clip=args.normalization_clip,
    )

    splits = generate_splits(len(dataset), args.train_frac, args.val_frac, args.seed)
    overall_progress = tqdm(total=len(dataset), desc="Preparing shards")

    for split in splits:
        split_dir = args.output_dir / split.name
        split_dir.mkdir(parents=True, exist_ok=True)
        shards: list[Path] = []
        subset_iter = iter_subset(dataset, split.indices, args.batch_size)

        for sample in subset_iter:
            shard_path = write_shard(split_dir, prefix=f"{args.suite}_{args.set_name}_z{args.redshift:0.2f}", sample=sample, split=split.name, bands=CAMELS_CODEC_BANDS)
            shards.append(shard_path)
            overall_progress.update(len(sample.indices))

        save_manifest(split_dir, args.manifest_name, shards)

    overall_progress.close()

    summary = {
        "suite": args.suite,
        "set": args.set_name,
        "redshift": args.redshift,
        "fields": args.fields,
        "normalization_stats": str(args.normalization_stats) if args.normalization_stats else None,
        "normalization_clip": args.normalization_clip,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "seed": args.seed,
        "total_samples": len(dataset),
    }
    with open(args.output_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
