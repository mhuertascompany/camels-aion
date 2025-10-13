"""Utilities for working with CAMELS maps and AION embeddings."""

from .config import CAMELS_BASE_PATH, CAMELS_FIELDS, CAMELS_SUITE, CAMELS_SET, CAMELS_CODEC_BANDS
from .data import load_param_table, load_map_file, CamelsMapSample, CamelsIllustrisDataset
from .encoding import CamelsAionEncoder, EmbeddingWriter
from .codec_manager import LocalCodecManager

__all__ = [
    "CAMELS_BASE_PATH",
    "CAMELS_FIELDS",
    "CAMELS_SUITE",
    "CAMELS_SET",
    "CAMELS_CODEC_BANDS",
    "load_param_table",
    "load_map_file",
    "CamelsMapSample",
    "CamelsIllustrisDataset",
    "CamelsAionEncoder",
    "EmbeddingWriter",
    "LocalCodecManager",
]
