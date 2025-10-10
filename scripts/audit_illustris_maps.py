#!/usr/bin/env python3
"""Quick sanity checks for IllustrisTNG LH z=0 CAMELS maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from camels_aion.data import CamelsIllustrisDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-path",
        type=Path,
        default=None,
        help="Override CAMELS base path (defaults to config setting).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=2048,
        help="Number of samples used to estimate statistics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = CamelsIllustrisDataset(base_path=args.base_path)

    print(f"Total samples: {len(dataset)}")
    channels, height, width = dataset.image_shape
    print(f"Image shape  : {channels} × {height} × {width}")

    stats = dataset.compute_channel_stats(sample_size=args.sample_size)
    print("Channel statistics (approximate):")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
