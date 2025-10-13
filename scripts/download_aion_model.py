#!/usr/bin/env python3
"""Download the AION model snapshot to a local directory."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        type=str,
        default="polymathic-ai/aion-base",
        help="Hugging Face repository ID to download.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Specific revision (tag/commit) to pin; defaults to latest main.",
    )
    default_dest = Path(os.environ.get("AION_MODEL_DIR", os.environ.get("WORK", str(Path.home())))) / "models" / "aion"
    parser.add_argument(
        "--dest",
        type=Path,
        default=default_dest,
        help="Destination directory where the snapshot will be stored.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading `{args.repo_id}` to `{args.dest}`...")
    snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=args.dest,
        local_dir_use_symlinks=False,
    )
    print("Done.")


if __name__ == "__main__":
    main()
