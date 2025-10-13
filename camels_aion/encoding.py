"""Embedding utilities wrapping AION for CAMELS maps."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import torch
from tqdm.auto import tqdm

from aion import AION
from aion.modalities import LegacySurveyImage

from .config import CAMELS_FIELDS, CAMELS_CODEC_BANDS
from .data import CamelsIllustrisDataset, CamelsMapSample
from .codec_manager import LocalCodecManager


@dataclass
class EncoderConfig:
    """Configuration block for the encoder."""

    device: str = "auto"
    batch_size: int = 32
    num_encoder_tokens: int = 600
    fp16: bool = True
    codec_device: str = "cpu"


class EmbeddingWriter:
    """Persist embeddings and metadata to disk per batch."""

    def __init__(self, output_dir: Path, prefix: str = "illustris_lh") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self._shards: list[Path] = []

    def write(self, sample: CamelsMapSample, embeddings: torch.Tensor) -> Path:
        """Write a batch of embeddings + labels to a shard file."""
        start_idx = sample.indices[0]
        end_idx = sample.indices[-1]
        shard_name = f"{self.prefix}_{start_idx:05d}-{end_idx:05d}.pt"
        shard_path = self.output_dir / shard_name
        payload = {
            "indices": torch.tensor(sample.indices, dtype=torch.long),
            "embeddings": embeddings.cpu(),
            "labels": torch.from_numpy(sample.labels),
        }
        torch.save(payload, shard_path)
        self._shards.append(shard_path)
        return shard_path

    def save_manifest(self) -> Path:
        manifest_path = self.output_dir / f"{self.prefix}_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(
                {"shards": [path.name for path in self._shards]},
                fh,
                indent=2,
            )
        return manifest_path


class CamelsAionEncoder:
    """Utility for encoding CAMELS maps using AION."""

    @staticmethod
    def _resolve_device(spec: str | torch.device) -> torch.device:
        if isinstance(spec, torch.device):
            return spec
        if isinstance(spec, str):
            if spec.lower() == "auto":
                return torch.device("cuda" if torch.cuda.is_available() else "cpu")
            return torch.device(spec)
        raise ValueError(f"Unsupported device specification: {spec}")

    def __init__(
        self,
        model: Optional[AION] = None,
        codec_manager: Optional[LocalCodecManager] = None,
        config: EncoderConfig | None = None,
        model_path: str | Path | None = None,
        model_name: str = "polymathic-ai/aion-base",
        codec_repo: str | Path | None = None,
    ) -> None:
        self.config = config or EncoderConfig()
        self.device = self._resolve_device(self.config.device)
        if model is not None:
            self.model = model
        else:
            if model_path is not None:
                self.model = AION.from_pretrained(str(model_path))
            else:
                resolved_name = model_name if "/" in model_name else f"polymathic-ai/{model_name}"
                self.model = AION.from_pretrained(resolved_name)
        try:
            self.model = self.model.to(self.device)
        except RuntimeError as exc:
            if self.device.type == "cuda" and "cuda" in str(exc).lower():
                print("[INFO] CUDA unavailable, falling back to CPU for AION model.")
                self.device = torch.device("cpu")
                self.model = self.model.to(self.device)
            else:
                raise
        codec_device = self._resolve_device(self.config.codec_device)
        if codec_repo is not None:
            repo_ref = codec_repo
        elif model_path is not None:
            repo_ref = model_path
        else:
            repo_ref = model_name
        if isinstance(repo_ref, str) and "/" not in repo_ref and not Path(repo_ref).exists():
            repo_ref = f"polymathic-ai/{repo_ref}"
        if isinstance(repo_ref, str) and Path(repo_ref).exists():
            repo_ref = Path(repo_ref)
        self.codec_manager = codec_manager or LocalCodecManager(repo=repo_ref, device=codec_device)
        self.codec_device = codec_device
        self.fields = CAMELS_FIELDS

    def encode_sample(self, sample: CamelsMapSample) -> torch.Tensor:
        """Encode a batch of CAMELS maps into AION embeddings."""
        flux = torch.from_numpy(sample.images).to(self.codec_device, dtype=torch.float32)

        modality = LegacySurveyImage(flux=flux, bands=list(CAMELS_CODEC_BANDS))
        tokens = self.codec_manager.encode(modality)

        tokens = {key: tensor.to(self.device) for key, tensor in tokens.items()}

        use_amp = self.config.fp16 and self.device.type == "cuda"

        with torch.no_grad():
            if use_amp:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    embeddings = self.model.encode(
                        tokens, num_encoder_tokens=self.config.num_encoder_tokens
                    )
            else:
                embeddings = self.model.encode(
                    tokens, num_encoder_tokens=self.config.num_encoder_tokens
                )
        return embeddings.detach()

    def encode_dataset(
        self,
        dataset: CamelsIllustrisDataset,
        writer: EmbeddingWriter,
        batch_size: int | None = None,
        indices: Iterable[int] | None = None,
        show_progress: bool = True,
    ) -> list[Path]:
        """Iterate dataset, encode, and write embedding shards."""
        batch_size = batch_size or self.config.batch_size

        if indices is None:
            index_list = list(range(len(dataset)))
        else:
            index_list = list(indices)

        progress = tqdm(total=len(index_list), disable=not show_progress, desc="Encoding")
        shards: list[Path] = []
        for sample in dataset.iter_batches(batch_size=batch_size, indices=index_list):
            embeddings = self.encode_sample(sample)
            shard_path = writer.write(sample, embeddings)
            shards.append(shard_path)
            progress.update(len(sample.indices))
        progress.close()
        writer.save_manifest()
        return shards
