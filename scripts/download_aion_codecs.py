#!/usr/bin/env python3
"""Download/prime codec weights required for CAMELS ↔︎ AION encoding."""

from __future__ import annotations

import argparse
import inspect
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
        help="Hugging Face repo id or local snapshot path containing codec weights.",
    )
    parser.add_argument(
        "--codec-device",
        type=str,
        default="cpu",
        help="Device to instantiate the codec on while priming.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=128,
        help="Edge length (pixels) of a dummy map used to run the codec once.",
    )
    return parser.parse_args()


def load_config(repo: str | Path) -> dict[str, float | int | list[int]]:
    if Path(repo).exists():
        config_path = Path(repo) / "codecs" / LegacySurveyImage.name / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Could not find codec config at {config_path}. "
                "Ensure the local snapshot contains the `codecs/` folder."
            )
    else:
        config_path = Path(
            hf_hub_download(repo, f"codecs/{LegacySurveyImage.name}/config.json")
        )
    with open(config_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def instantiate_codec(
    repo: str | Path, device: str, config: dict[str, object]
):
    codec_cls = MODALITY_CODEC_MAPPING[LegacySurveyImage]

    init_params = inspect.signature(codec_cls.__init__).parameters
    init_kwargs = {
        name: config[name]
        for name in init_params
        if name != "self" and name in config
    }

    repo_ref = str(repo) if Path(repo).exists() else repo
    codec = codec_cls.from_pretrained(
        repo_ref,
        modality=LegacySurveyImage,
        **init_kwargs,
    )
    return codec.to(device).eval()


def main() -> None:
    args = parse_args()

    print(f"== Priming LegacySurvey image codec from {args.repo} ==")
    config = load_config(args.repo)
    codec = instantiate_codec(args.repo, args.codec_device, config)

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
    print(f"  token tensor shape: {tuple(tokens.shape)} dtype: {tokens.dtype}")

    print("Codec weights are now cached under HF_HOME/local snapshot.")


if __name__ == "__main__":
    main()
