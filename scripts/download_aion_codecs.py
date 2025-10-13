#!/usr/bin/env python3
"""Download/prime codec weights required for CAMELS ↔︎ AION encoding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from aion.codecs.config import HF_REPO_ID, MODALITY_CODEC_MAPPING
from aion.modalities import LegacySurveyImage

from camels_aion.config import CAMELS_FIELDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=str,
        default=HF_REPO_ID,
        help="Hugging Face repo id or local path containing codec weights (default: polymathic-ai/aion-base)",
    )
    parser.add_argument(
        "--codec-device",
        type=str,
        default="cpu",
        help="Device on which to instantiate codecs during priming.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=128,
        help="Side length (pixels) of the dummy image used for priming.",
    )
    return parser.parse_args()


def _load_codec(repo: str | Path, codec_device: str):
    codec_cls = MODALITY_CODEC_MAPPING[LegacySurveyImage]

    if Path(repo).exists():
        repo_path = Path(repo)
        config_path = repo_path / "codecs" / LegacySurveyImage.name / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Could not find codec config at {config_path}")
    else:
        config_path = hf_hub_download(repo, f"codecs/{LegacySurveyImage.name}/config.json")
        repo_path = repo

    with open(config_path, "r", encoding="utf-8") as fh:
        config = json.load(fh)

    codec = codec_cls.from_pretrained(repo_path, modality=LegacySurveyImage, config=config)
    codec = codec.to(codec_device)
    return codec


def main() -> None:
    args = parse_args()

    print(f"== Priming LegacySurvey image codec from {args.repo} ==")
    codec = _load_codec(args.repo, args.codec_device)

    flux = torch.zeros(
        1,
        len(CAMELS_FIELDS),
        args.image_size,
        args.image_size,
        dtype=torch.float32,
        device=args.codec_device,
    )
    image = LegacySurveyImage(flux=flux, bands=list(CAMELS_FIELDS))

    with torch.no_grad():
        tokens = codec.encode(image)

    print(f"  tokens shape: {tuple(tokens.shape)}, dtype={tokens.dtype}")

    print("Codec weights downloaded and cached (see HF_HOME or local repo directory).")


if __name__ == "__main__":
    main()
