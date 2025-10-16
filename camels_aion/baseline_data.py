"""Dataset utilities for baseline CNN/ViT training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import (
    CAMELS_BASE_PATH,
    CAMELS_FIELDS,
    CAMELS_REDSHIFT,
    CAMELS_SET,
    CAMELS_SUITE,
)
from .data import load_map_file, load_param_table


class NormalizationHelper:
    """Apply per-field transforms defined in a stats JSON."""

    def __init__(self, stats: Mapping[str, Mapping], default_clip: float = 1.5):
        self.metadata = stats.get("metadata", {})
        self.stats = {
            field: value
            for field, value in stats.items()
            if field != "metadata"
        }
        self.default_clip = default_clip

    @classmethod
    def from_path(cls, path: Path, default_clip: float = 1.5) -> "NormalizationHelper":
        with open(path, "r", encoding="utf-8") as fh:
            stats = json.load(fh)
        return cls(stats, default_clip=default_clip)

    def normalize(self, field: str, data: np.ndarray) -> np.ndarray:
        stats = self.stats.get(field)
        if stats is None:
            mean = data.mean()
            std = data.std() + 1e-6
            return ((data - mean) / std).astype(np.float32)

        transform = stats.get("transform", "arcsinh")
        scale = float(stats.get("scale", 1.0))
        eps = float(stats.get("eps", 1e-6))
        denom = scale if abs(scale) > eps else eps

        if transform == "log1p":
            transformed = np.log1p(data / denom)
        elif transform == "none":
            transformed = data / denom
        else:
            transformed = np.arcsinh(data / denom)

        low = stats.get("low")
        high = stats.get("high")
        if low is None or high is None or abs(high - low) < 1e-6:
            mean = transformed.mean()
            std = transformed.std() + 1e-6
            normalized = (transformed - mean) / std
        else:
            normalized = (transformed - low) / (high - low + 1e-6) * 2 - 1

        clip = float(stats.get("clip", self.default_clip))
        normalized = np.clip(normalized, -clip, clip)
        return normalized.astype(np.float32)


class CamelsMapDataset(Dataset):
    """PyTorch dataset yielding normalized CAMELS maps and parameter labels."""

    def __init__(
        self,
        fields: Sequence[str] | None = None,
        suite: str = CAMELS_SUITE,
        set_name: str = CAMELS_SET,
        redshift: float = CAMELS_REDSHIFT,
        base_path: Path | None = None,
        normalization_stats: Path | Mapping | None = None,
        normalization_clip: float = 1.5,
    ) -> None:
        self.fields = tuple(fields) if fields is not None else tuple(CAMELS_FIELDS)
        self.suite = suite
        self.set_name = set_name
        self.redshift = redshift
        self.base_path = Path(base_path) if base_path else CAMELS_BASE_PATH

        self._maps = {
            field: load_map_file(
                field,
                suite=suite,
                set_name=set_name,
                redshift=redshift,
                base_path=self.base_path,
                mmap=True,
            )
            for field in self.fields
        }
        lengths = {field: arr.shape[0] for field, arr in self._maps.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Inconsistent map counts: {lengths}")
        self.num_samples = next(iter(lengths.values()))

        self.params = load_param_table(
            suite=suite,
            set_name=set_name,
            base_path=self.base_path,
        )

        if normalization_stats is None:
            self.normalizer = None
        elif isinstance(normalization_stats, Mapping):
            self.normalizer = NormalizationHelper(normalization_stats, default_clip=normalization_clip)
        else:
            self.normalizer = NormalizationHelper.from_path(Path(normalization_stats), default_clip=normalization_clip)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        channels = []
        for field in self.fields:
            data = np.array(self._maps[field][index], copy=False, dtype=np.float32)
            if self.normalizer is not None:
                data = self.normalizer.normalize(field, data)
            else:
                mean = data.mean()
                std = data.std() + 1e-6
                data = (data - mean) / std
            channels.append(data)
        image = np.stack(channels, axis=0).astype(np.float32)
        label = self.params[index // 15].astype(np.float32)
        return torch.from_numpy(image), torch.from_numpy(label)


def create_dataset(
    suite: str,
    set_name: str,
    redshift: float,
    base_path: Path | None,
    fields: Sequence[str] | None,
    normalization_stats: Path | Mapping | None,
    normalization_clip: float,
) -> CamelsMapDataset:
    return CamelsMapDataset(
        fields=fields,
        suite=suite,
        set_name=set_name,
        redshift=redshift,
        base_path=base_path,
        normalization_stats=normalization_stats,
        normalization_clip=normalization_clip,
    )
