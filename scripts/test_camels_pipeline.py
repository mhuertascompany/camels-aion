#!/usr/bin/env python3
"""Quick end-to-end smoke test for the CAMELS ↔︎ AION pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from camels_aion.config import CAMELS_FIELDS
from camels_aion.data import CamelsIllustrisDataset
from camels_aion.encoding import CamelsAionEncoder, EncoderConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-path", type=Path, default=None, help="Override CAMELS base directory.")
    parser.add_argument("--suite", type=str, default="IllustrisTNG", help="Simulation suite (e.g., IllustrisTNG, SIMBA).")
    parser.add_argument("--set", dest="set_name", type=str, default="LH", help="Simulation set identifier.")
    parser.add_argument("--redshift", type=float, default=0.0, help="Redshift slice to load.")
    parser.add_argument("--sample-count", type=int, default=8, help="Number of maps to load and encode for the smoke test.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for encoding.")
    parser.add_argument("--device", type=str, default="auto", help="Device for AION model (cuda/cpu/auto).")
    parser.add_argument("--codec-device", type=str, default="cpu", help="Device on which codecs operate.")
    parser.add_argument("--num-encoder-tokens", type=int, default=600, help="Maximum encoder tokens for AION.")
    parser.add_argument("--model-dir", type=Path, default=None, help="Path to a local AION snapshot (offline mode).")
    parser.add_argument("--model-name", type=str, default="polymathic-ai/aion-base", help="Hugging Face repo or identifier when downloading.")
    parser.add_argument("--codec-repo", type=str, default=None, help="Codec repo id or local path (defaults to model path/repo).")
    parser.add_argument("--fp32", action="store_true", help="Disable mixed precision during encoding.")
    parser.add_argument("--skip-encoder", action="store_true", help="Only load CAMELS maps without running AION.")
    parser.add_argument("--stats-samples", type=int, default=512, help="Number of samples for channel statistics.")
    parser.add_argument("--normalization-stats", type=Path, default=None, help="Optional JSON stats file used to normalize CAMELS maps before tokenization.")
    parser.add_argument("--normalization-clip", type=float, default=1.5, help="Clip value for normalized data.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_raw = CamelsIllustrisDataset(
        fields=CAMELS_FIELDS,
        suite=args.suite,
        set_name=args.set_name,
        redshift=args.redshift,
        base_path=args.base_path,
        apply_normalization=False,
    )

    print("== CAMELS Dataset ==")
    print(f"Total samples : {len(dataset_raw)}")
    channels, height, width = dataset_raw.image_shape
    print(f"Image shape   : {channels} × {height} × {width}")

    stats = dataset_raw.compute_channel_stats(sample_size=args.stats_samples)
    print("Per-channel stats (approximate):")
    for field, summary in stats.items():
        print(
            f"  {field:>5s} | mean={summary['mean']:.4e} std={summary['std']:.4e} "
            f"min={summary['min']:.4e} max={summary['max']:.4e}"
        )

    dataset = CamelsIllustrisDataset(
        fields=CAMELS_FIELDS,
        suite=args.suite,
        set_name=args.set_name,
        redshift=args.redshift,
        base_path=args.base_path,
        normalization_stats=args.normalization_stats,
        normalization_clip=args.normalization_clip,
    )

    sample_count = min(args.sample_count, len(dataset))
    sample_indices = list(range(sample_count))
    sample = next(dataset.iter_batches(batch_size=args.batch_size, indices=sample_indices))

    print("\nSample batch:")
    print(f"  indices      : {sample.indices}")
    print(f"  images shape : {sample.images.shape}")
    print(f"  labels shape : {sample.labels.shape}")
    print(f"  first label  : {sample.labels[0]}")

    if args.skip_encoder:
        print("\nSkipping AION encoding as requested.")
        return

    encoder_config = EncoderConfig(
        device=args.device,
        batch_size=args.batch_size,
        num_encoder_tokens=args.num_encoder_tokens,
        fp16=not args.fp32,
        codec_device=args.codec_device,
    )

    encoder = CamelsAionEncoder(
        config=encoder_config,
        model_path=args.model_dir,
        model_name=args.model_name,
        codec_repo=args.codec_repo
        if args.codec_repo is not None
        else (args.model_dir if args.model_dir is not None else args.model_name),
    )

    embeddings = encoder.encode_sample(sample)

    print("\n== AION Encoding ==")
    print(f"Embeddings shape : {tuple(embeddings.shape)}")
    print(f"Embedding dtype  : {embeddings.dtype}")
    print(
        "Embedding stats  : "
        f"mean={embeddings.mean().item():.4e}, std={embeddings.std().item():.4e}"
    )


if __name__ == "__main__":
    main()
