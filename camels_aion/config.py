"""Configuration constants for CAMELS ↔︎ AION experiments."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

# Base directory on Jean-Zay where CAMELS 2D maps are stored.
# The path can be overridden via the CAMELS_BASE_PATH environment variable.
CAMELS_BASE_PATH: Final[Path] = Path(
    os.environ.get(
        "CAMELS_BASE_PATH", "/lustre/fsmisc/dataset/CAMELS_Multifield_Dataset/2D_maps/data"
    )
)

# Default suite / set / redshift used in the initial pipeline.
CAMELS_SUITE: Final[str] = "IllustrisTNG"
CAMELS_SET: Final[str] = "LH"
CAMELS_REDSHIFT: Final[float] = 0.0

# Fields we will combine into a 4-channel image: order matters because it
# defines channel layout when feeding LegacySurveyImage.
CAMELS_FIELDS: Final[tuple[str, ...]] = ("Mstar", "Mgas", "T", "Z")

# File naming template for CAMELS 2D map numpy files.
MAP_FILENAME_TEMPLATE: Final[str] = "Maps_{field}_{suite}_{set}_z={redshift:0.2f}.npy"

# Filename for the parameter table distributed with CAMELS maps.
PARAM_FILENAME_TEMPLATE: Final[str] = "params_{set}_{suite}.txt"

# Chunk size used when streaming maps from disk; can be overridden at runtime.
DEFAULT_CHUNK_SIZE: Final[int] = 256
