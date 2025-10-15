"""Data loading helpers for CAMELS 2D maps."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
from aion.codecs.preprocessing.image import Clamp

from .config import (
    CAMELS_BASE_PATH,
    CAMELS_CODEC_BANDS,
    CAMELS_FIELDS,
    CAMELS_REDSHIFT,
    CAMELS_SET,
    CAMELS_SUITE,
    MAP_FILENAME_TEMPLATE,
    PARAM_FILENAME_TEMPLATE,
)


def _resolve_file(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "None of the expected CAMELS paths exist. Checked:\n"
        + "\n".join(str(candidate) for candidate in candidates)
    )


def build_map_path(
    field: str,
    suite: str = CAMELS_SUITE,
    set_name: str = CAMELS_SET,
    redshift: float = CAMELS_REDSHIFT,
    base_path: Path | None = None,
) -> Path:
    """Return full path to a CAMELS 2D map file."""
    base = Path(base_path) if base_path else CAMELS_BASE_PATH
    filename = MAP_FILENAME_TEMPLATE.format(
        field=field, suite=suite, set=set_name, redshift=redshift
    )
    candidates = [
        base / suite / set_name / f"z={redshift:0.2f}" / filename,
        base / suite / set_name / filename,
        base / suite / f"z={redshift:0.2f}" / filename,
        base / suite / filename,
        base / filename,
    ]
    return _resolve_file(candidates)


def build_param_path(
    suite: str = CAMELS_SUITE,
    set_name: str = CAMELS_SET,
    base_path: Path | None = None,
) -> Path:
    """Return full path to CAMELS parameter table."""
    base = Path(base_path) if base_path else CAMELS_BASE_PATH
    filename = PARAM_FILENAME_TEMPLATE.format(set=set_name, suite=suite)
    candidates = [
        base / suite / set_name / filename,
        base / suite / filename,
        base / filename,
    ]
    return _resolve_file(candidates)


def load_map_file(
    field: str,
    suite: str = CAMELS_SUITE,
    set_name: str = CAMELS_SET,
    redshift: float = CAMELS_REDSHIFT,
    base_path: Path | None = None,
    mmap: bool = True,
) -> np.ndarray:
    """Load a CAMELS 2D map numpy file.

    Args:
        field: Field name (e.g. ``"Mgas"``).
        mmap: Use memory mapping to avoid reading entire array at once.
    """
    path = build_map_path(field, suite=suite, set_name=set_name, redshift=redshift, base_path=base_path)
    mmap_mode = "r" if mmap else None
    return np.load(path, mmap_mode=mmap_mode)


def load_param_table(
    suite: str = CAMELS_SUITE,
    set_name: str = CAMELS_SET,
    base_path: Path | None = None,
) -> np.ndarray:
    """Load CAMELS parameter table."""
    path = build_param_path(suite=suite, set_name=set_name, base_path=base_path)
    return np.loadtxt(path)


@dataclass
class CamelsMapSample:
    """Container for a batch of CAMELS maps and labels."""

    indices: list[int]
    images: np.ndarray  # Shape (B, C, H, W)
    labels: np.ndarray  # Shape (B, num_params)


class CamelsIllustrisDataset:
    """Iterable dataset for IllustrisTNG LH CAMELS maps."""

    def __init__(
        self,
        fields: Sequence[str] = CAMELS_FIELDS,
        suite: str = CAMELS_SUITE,
        set_name: str = CAMELS_SET,
        redshift: float = CAMELS_REDSHIFT,
        base_path: Path | None = None,
        dtype: np.dtype = np.float32,
        mmap: bool = True,
        normalization_stats: dict | str | Path | None = None,
        normalization_clip: float = 1.5,
        apply_normalization: bool = True,
    ) -> None:
        self.fields = tuple(fields)
        self.suite = suite
        self.set_name = set_name
        self.redshift = redshift
        self.base_path = Path(base_path) if base_path else CAMELS_BASE_PATH
        self.dtype = dtype
        self.normalization_clip = normalization_clip

        self._maps = {
            field: load_map_file(
                field,
                suite=suite,
                set_name=set_name,
                redshift=redshift,
                base_path=self.base_path,
                mmap=mmap,
            )
            for field in self.fields
        }
        lengths = {field: arr.shape[0] for field, arr in self._maps.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Inconsistent map counts across fields: {lengths}")
        self.num_samples = next(iter(lengths.values()))

        self.params = load_param_table(suite=suite, set_name=set_name, base_path=self.base_path)

        if normalization_stats is not None:
            self.normalization_stats = self._load_normalization_stats(normalization_stats)
        else:
            self.normalization_stats = None
        self.apply_normalization = apply_normalization and self.normalization_stats is not None
        if self.apply_normalization:
            self._clamp = Clamp()
            self._clamp_bounds = {
                field: self._clamp.clamp_dict[band]
                for field, band in zip(self.fields, CAMELS_CODEC_BANDS)
            }
        else:
            self._clamp = None
            self._clamp_bounds = {}

    def __len__(self) -> int:
        return self.num_samples

    @property
    def image_shape(self) -> tuple[int, int, int]:
        sample_field = self.fields[0]
        _, height, width = self._maps[sample_field].shape
        return (len(self.fields), height, width)

    def iter_batches(self, batch_size: int, indices: Iterable[int] | None = None) -> Iterator[CamelsMapSample]:
        """Yield batches of stacked images and corresponding labels."""
        if indices is None:
            indices = range(self.num_samples)
        indices = list(indices)
        num_batches = math.ceil(len(indices) / batch_size)

        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(indices))
            batch_indices = indices[start:end]

            # Stack channels for each field
            stacked = []
            for field in self.fields:
                maps = np.array(self._maps[field][batch_indices], copy=False)
                if self.apply_normalization:
                    maps = self._normalize_field(field, maps)
                else:
                    maps = maps.astype(self.dtype, copy=False)
                stacked.append(np.expand_dims(maps, axis=1))
            images = np.concatenate(stacked, axis=1)
            # Each parameter row corresponds to 15 maps -> map index // 15
            label_indices = np.array(batch_indices) // 15
            labels = self.params[label_indices].astype(np.float32, copy=False)
            yield CamelsMapSample(indices=batch_indices, images=images, labels=labels)

    def compute_channel_stats(self, sample_size: int = 1024) -> dict[str, dict[str, float]]:
        """Estimate per-channel mean and std for normalization."""
        sample_size = min(sample_size, self.num_samples)
        idx = np.linspace(0, self.num_samples - 1, sample_size, dtype=int)
        stats: dict[str, dict[str, float]] = {}
        for field in self.fields:
            data = self._maps[field][idx]
            stats[field] = {
                "mean": float(np.mean(data)),
                "std": float(np.std(data)),
                "min": float(np.min(data)),
                "max": float(np.max(data)),
            }
        return stats

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_normalization_stats(self, spec: dict | str | Path) -> dict:
        if isinstance(spec, dict):
            stats = spec
        else:
            path = Path(spec)
            with open(path, "r", encoding="utf-8") as fh:
                stats = json.load(fh)
        return stats

    def _normalize_field(self, field: str, data: np.ndarray) -> np.ndarray:
        stats = self.normalization_stats.get(field)
        if stats is None:
            return data.astype(self.dtype, copy=False)
        if field not in self._clamp_bounds:
            return data.astype(self.dtype, copy=False)

        transform = stats.get("transform", "arcsinh")
        scale = float(stats.get("scale", 1.0))
        eps = float(stats.get("eps", 1e-8))
        denom = scale if abs(scale) > eps else eps

        if transform == "log1p":
            transformed = np.log1p(data / denom)
        elif transform == "none":
            transformed = data / denom
        else:
            transformed = np.arcsinh(data / denom)

        low = float(stats.get("low", np.percentile(transformed, 1)))
        high = float(stats.get("high", np.percentile(transformed, 99)))
        if high - low <= 1e-6:
            normalized = transformed - low
        else:
            normalized = (transformed - low) / (high - low) * 2 - 1

        clip = float(stats.get("clip", self.normalization_clip))
        normalized = np.clip(normalized, -clip, clip)

        limit = float(self._clamp_bounds[field])
        mapped = normalized * (limit / clip)
        return mapped.astype(self.dtype, copy=False)
