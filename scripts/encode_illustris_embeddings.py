#!/usr/bin/env python3
"""Encode IllustrisTNG LH CAMELS maps into AION embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from camels_aion.data import CamelsIllustrisDataset
from camels_aion.encoding import CamelsAionEncoder, EncoderConfig, EmbeddingWriter


def parse_indices(start: int | None, end: int | None, total: int) -> Iterable[int]:
    if start is None and end is None:
        return range(total)
    start = start or 0
    end = end if end is not None else total
    return range(start, min(end, total))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-path",
        type=Path,
        default=None,
        help="Override CAMELS base path (defaults to config setting).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where embedding shards will be written.",
    )
    parser.add_argument("--suite", type=str, default="IllustrisTNG", help="Simulation suite, e.g., IllustrisTNG or SIMBA.")
    parser.add_argument("--set", dest="set_name", type=str, default="LH", help="Simulation set (CV, LH, 1P, ...).")
    parser.add_argument("--redshift", type=float, default=0.0, help="Redshift slice to load.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--codec-device", type=str, default="cpu")
    parser.add_argument(
        "--num-encoder-tokens",
        type=int,
        default=600,
        help="Maximum number of encoder tokens provided to AION.",
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Disable mixed-precision encoding even on CUDA.",
    )
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--model-dir", type=Path, default=None, help="Path to a local AION snapshot (offline mode).")
    parser.add_argument("--model-name", type=str, default="polymathic-ai/aion-base", help="Hugging Face repo id to use when downloading the model.")
    parser.add_argument("--codec-repo", type=str, default=None, help="Repo id or local path for codec weights (defaults to model path/repo).")
    parser.add_argument("--normalization-stats", type=Path, default=None, help="Optional JSON file with per-field normalization statistics.")
    parser.add_argument("--normalization-clip", type=float, default=1.5, help="Clip value applied after normalization before mapping into codec range.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset = CamelsIllustrisDataset(
        base_path=args.base_path,
        suite=args.suite,
        set_name=args.set_name,
        redshift=args.redshift,
        normalization_stats=args.normalization_stats,
        normalization_clip=args.normalization_clip,
    )
    indices = parse_indices(args.start_index, args.end_index, len(dataset))

    encoder = CamelsAionEncoder(
        config=EncoderConfig(
            device=args.device,
            batch_size=args.batch_size,
            num_encoder_tokens=args.num_encoder_tokens,
            fp16=not args.fp32,
            codec_device=args.codec_device,
        ),
        model_path=args.model_dir,
        model_name=args.model_name,
        codec_repo=args.codec_repo,
    )

    prefix = args.prefix or f"{args.suite}_{args.set_name}_z{args.redshift:0.2f}".replace(".", "p")
    writer = EmbeddingWriter(output_dir=args.output_dir, prefix=prefix)
    encoder.encode_dataset(dataset, writer=writer, batch_size=args.batch_size, indices=indices)


if __name__ == "__main__":
    main()
