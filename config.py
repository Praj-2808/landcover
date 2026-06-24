"""
config.py
Central configuration for the Land Cover Classification and Change Detection app.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = BASE_DIR / "data"
CACHE_DIR: Path = DATA_DIR / "cache"
IMAGERY_DIR: Path = DATA_DIR / "imagery"
CLASSIFICATION_DIR: Path = DATA_DIR / "classifications"
MODEL_DIR: Path = BASE_DIR / "models" / "trained"
OUTPUT_DIR: Path = BASE_DIR / "outputs"
LOG_DIR: Path = BASE_DIR / "logs"

for _d in (DATA_DIR, CACHE_DIR, IMAGERY_DIR, CLASSIFICATION_DIR, MODEL_DIR, OUTPUT_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Land cover classes
# ---------------------------------------------------------------------------
# Class codes used throughout the pipeline (raster pixel values)
LAND_COVER_CLASSES: Dict[int, str] = {
    0: "Water",
    1: "Vegetation",
    2: "Urban/Built-up",
    3: "Agriculture",
    4: "Bare Land",
}

# Display colors (hex) for maps and charts
LAND_COVER_COLORS: Dict[int, str] = {
    0: "#1f77b4",  # Water - blue
    1: "#2ca02c",  # Vegetation - green
    2: "#d62728",  # Urban - red
    3: "#bcbd22",  # Agriculture - olive/yellow
    4: "#8c564b",  # Bare Land - brown
}

CLASS_NAME_TO_CODE: Dict[str, int] = {v: k for k, v in LAND_COVER_CLASSES.items()}

# ---------------------------------------------------------------------------
# Satellite data source configuration
# ---------------------------------------------------------------------------
# STAC endpoint (Microsoft Planetary Computer) - free, no auth required
STAC_API_URL: str = "https://planetarycomputer.microsoft.com/api/stac/v1"
PLANETARY_COMPUTER_SAS_URL: str = "https://planetarycomputer.microsoft.com/api/sas/v1/sign"

# Collections
SENTINEL2_COLLECTION: str = "sentinel-2-l2a"
LANDSAT_COLLECTION: str = "landsat-c2-l2"

# Year threshold: Sentinel-2 data is reliably available from 2015 onward.
# For years before this, use Landsat.
SENTINEL2_MIN_YEAR: int = 2016

# Cloud cover search parameters
INITIAL_CLOUD_COVER_THRESHOLD: float = 20.0
CLOUD_COVER_STEP: float = 10.0
MAX_CLOUD_COVER_THRESHOLD: float = 80.0

# Bounding box buffer around city center (in degrees, ~ city-scale AOI)
DEFAULT_AOI_BUFFER_DEG: float = 0.05  # ~5.5 km

# Target resolution for analysis (meters per pixel)
TARGET_RESOLUTION_M: int = 30  # Common to both Sentinel-2 (resampled) and Landsat

# ---------------------------------------------------------------------------
# Band mapping per sensor (mapped to a common naming scheme)
# ---------------------------------------------------------------------------
# Common band roles: blue, green, red, nir, swir1, swir2
SENTINEL2_BAND_MAP: Dict[str, str] = {
    "blue": "B02",
    "green": "B03",
    "red": "B04",
    "nir": "B08",
    "swir1": "B11",
    "swir2": "B12",
}

LANDSAT_BAND_MAP: Dict[str, str] = {
    "blue": "blue",
    "green": "green",
    "red": "red",
    "nir": "nir08",
    "swir1": "swir16",
    "swir2": "swir22",
}

COMMON_BANDS: List[str] = ["blue", "green", "red", "nir", "swir1", "swir2"]

# ---------------------------------------------------------------------------
# Machine Learning configuration
# ---------------------------------------------------------------------------
FEATURE_NAMES: List[str] = [
    "blue", "green", "red", "nir", "swir1", "swir2",
    "ndvi", "ndwi", "ndbi", "savi", "evi",
]

RANDOM_STATE: int = 42
N_SYNTHETIC_SAMPLES_PER_CLASS: int = 1500

RF_PARAMS: Dict = {
    "n_estimators": 150,
    "max_depth": 18,
    "min_samples_leaf": 2,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}

XGB_PARAMS: Dict = {
    "n_estimators": 150,
    "max_depth": 8,
    "learning_rate": 0.1,
    "objective": "multi:softmax",
    "num_class": len(LAND_COVER_CLASSES),
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "eval_metric": "mlogloss",
}

LIGHTGBM_PARAMS: Dict = {
    "n_estimators": 150,
    "max_depth": 8,
    "learning_rate": 0.1,
    "objective": "multiclass",
    "num_class": len(LAND_COVER_CLASSES),
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": -1,
}

DEFAULT_MODEL_TYPE: str = "random_forest"

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
GEOCODER_USER_AGENT: str = "landcover_classification_app"

# Quick comparison offsets (years)
QUICK_COMPARISON_OFFSETS: List[int] = [5, 10, 15, 20]

# Logging
LOG_LEVEL: str = os.environ.get("LANDCOVER_LOG_LEVEL", "INFO")
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
