"""Local-aware codec manager that supports offline operation."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download

from aion.codecs.manager import CodecManager
from aion.codecs.config import MODALITY_CODEC_MAPPING, CodecType
from aion.modalities import Modality


class LocalCodecManager(CodecManager):
    """Codec manager that can load codecs from a local snapshot or HF repo."""

    def __init__(self, repo: str | Path, device: str | torch.device = "cpu") -> None:
        super().__init__(device=device)
        self.repo = Path(repo) if Path(repo).exists() else repo

    @staticmethod
    def _config_path(repo: str | Path, modality: type[Modality]) -> Path:
        base = Path(repo)
        return base / "codecs" / modality.name / "config.json"

    @staticmethod
    def _load_config(path: Path) -> dict[str, Any]:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    @lru_cache
    def _load_codec(self, modality_type: type[Modality]):
        if modality_type not in MODALITY_CODEC_MAPPING:
            raise ValueError(f"No codec mapping found for modality {modality_type}")
        codec_class: CodecType = MODALITY_CODEC_MAPPING[modality_type]

        if isinstance(self.repo, Path):
            config_path = self._config_path(self.repo, modality_type)
            config = self._load_config(config_path)
            repo_ref = str(self.repo)
        else:
            config_path = hf_hub_download(
                self.repo, f"codecs/{modality_type.name}/config.json"
            )
            config = self._load_config(Path(config_path))
            repo_ref = self.repo

        codec = codec_class.from_pretrained(
            repo_ref, modality=modality_type, config=config
        )
        codec = codec.eval()
        return codec
