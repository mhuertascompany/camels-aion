#!/usr/bin/env python3
"""Download/prime codec weights required for CAMELS ↔︎ AION encoding."""

from __future__ import annotations

import argparse

import torch

from aion.codecs import CodecManager
from aion.modalities import LegacySurveyImage

from camels_aion.config import CAMELS_FIELDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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


def main() -> None:
    args = parse_args()

    print("== Priming LegacySurvey image codec ==")
    codec_manager = CodecManager(device=args.codec_device)

    flux = torch.zeros(
        1,
        len(CAMELS_FIELDS),
        args.image_size,
        args.image_size,
        dtype=torch.float32,
    )
    image = LegacySurveyImage(flux=flux, bands=list(CAMELS_FIELDS))
    tokens = codec_manager.encode(image)

    for key, value in tokens.items():
        print(f"  cached {key}: shape={tuple(value.shape)}, dtype={value.dtype}")

    print("Codec weights downloaded and cached (see HF_HOME).")


if __name__ == "__main__":
    main()
