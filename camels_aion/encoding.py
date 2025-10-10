"""Embedding utilities wrapping AION for CAMELS maps."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import torch
from tqdm.auto import tqdm

from aion import AION
from aion.codecs import CodecManager
from aion.modalities import LegacySurveyImage

from .config import CAMELS_FIELDS
from .data import CamelsIllustrisDataset, CamelsMapSample


@dataclass
class EncoderConfig:
    """Configuration block for the encoder."""

    device: str = "cuda"
    batch_size: int = 32
    num_encoder_tokens: int = 600
    fp16: bool = True


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

    def __init__(
        self,
        model: Optional[AION] = None,
        codec_manager: Optional[CodecManager] = None,
        config: EncoderConfig | None = None,
    ) -> None:
        self.config = config or EncoderConfig()
        self.device = torch.device(self.config.device)
        self.model = model or AION.from_pretrained("aion-base")
        self.model = self.model.to(self.device)
        self.codec_manager = codec_manager or CodecManager(device=self.device)
        self.fields = CAMELS_FIELDS

    def encode_sample(self, sample: CamelsMapSample) -> torch.Tensor:
        """Encode a batch of CAMELS maps into AION embeddings."""
        flux = torch.from_numpy(sample.images).to(self.device, dtype=torch.float32)
        if self.config.fp16 and self.device.type == "cuda":
            flux = flux.half()

        modality = LegacySurveyImage(
            flux=flux,
            bands=list(self.fields),
        )
        tokens = self.codec_manager.encode(modality)

        tokens = {key: tensor.to(self.device) for key, tensor in tokens.items()}

        if self.config.fp16 and self.device.type == "cuda":
            autocast_ctx = torch.cuda.amp.autocast
            autocast_kwargs = {"enabled": True, "dtype": torch.float16}
        else:
            autocast_ctx = torch.autocast
            autocast_kwargs = {"device_type": self.device.type, "enabled": False}

        with autocast_ctx(**autocast_kwargs):
            embeddings = self.model.encode(
                tokens,
                num_encoder_tokens=self.config.num_encoder_tokens,
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
