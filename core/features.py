"""
core/features.py
Feature engineering: computes spectral indices (NDVI, NDWI, NDBI, SAVI, EVI)
from raw band arrays and assembles the full feature stack used by the
classification models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from config import FEATURE_NAMES
from utils.logger import get_logger

logger = get_logger(__name__)

_EPS = 1e-6


@dataclass
class FeatureStack:
    """Container for the per-pixel feature stack."""

    arrays: Dict[str, np.ndarray]  # feature name -> 2D array (H, W)
    height: int
    width: int

    def to_array(self) -> np.ndarray:
        """Return features stacked as (H, W, F) in FEATURE_NAMES order."""
        return np.stack([self.arrays[name] for name in FEATURE_NAMES], axis=-1)

    def to_pixel_matrix(self) -> np.ndarray:
        """Return features reshaped to (H*W, F) for ML model input."""
        return self.to_array().reshape(-1, len(FEATURE_NAMES))


class FeatureEngineer:
    """Computes spectral indices and assembles feature stacks from band data."""

    @staticmethod
    def _normalize_bands(bands: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Normalize reflectance bands to a 0-1 range.

        Sentinel-2 and Landsat Collection 2 surface reflectance values are
        typically scaled integers (e.g. 0-10000 for Sentinel-2, with a scale
        factor for Landsat). This rescales to approximate reflectance.
        """
        normalized: Dict[str, np.ndarray] = {}
        for name, arr in bands.items():
            arr = arr.astype(np.float32)
            max_val = np.nanpercentile(arr, 99.5)
            if max_val <= 0 or not np.isfinite(max_val):
                max_val = 1.0
            # Heuristic: large integer-scaled reflectance values
            if max_val > 2.0:
                scale = 10000.0 if max_val > 100 else max_val
                norm = np.clip(arr / scale, 0.0, 1.0)
            else:
                norm = np.clip(arr, 0.0, 1.0)
            normalized[name] = norm
        return normalized

    def compute_features(self, bands: Dict[str, np.ndarray]) -> FeatureStack:
        """
        Compute the full feature stack (normalized bands + spectral indices)
        from raw satellite band arrays.

        Args:
            bands: Dict mapping common band names ('blue', 'green', 'red',
                'nir', 'swir1', 'swir2') to 2D numpy arrays of equal shape.

        Returns:
            A FeatureStack containing all bands and computed indices.
        """
        logger.info("Computing spectral indices and feature stack.")
        norm = self._normalize_bands(bands)

        blue = norm["blue"]
        green = norm["green"]
        red = norm["red"]
        nir = norm["nir"]
        swir1 = norm["swir1"]

        ndvi = self._ndvi(nir, red)
        ndwi = self._ndwi(green, nir)
        ndbi = self._ndbi(swir1, nir)
        savi = self._savi(nir, red)
        evi = self._evi(nir, red, blue)

        arrays: Dict[str, np.ndarray] = {
            "blue": blue,
            "green": green,
            "red": red,
            "nir": nir,
            "swir1": swir1,
            "swir2": norm.get("swir2", np.zeros_like(blue)),
            "ndvi": ndvi,
            "ndwi": ndwi,
            "ndbi": ndbi,
            "savi": savi,
            "evi": evi,
        }

        height, width = blue.shape
        logger.info("Feature stack assembled: %d features, shape (%d, %d).",
                    len(FEATURE_NAMES), height, width)
        return FeatureStack(arrays=arrays, height=height, width=width)

    # ------------------------------------------------------------------
    # Index formulas
    # ------------------------------------------------------------------
    @staticmethod
    def _ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
        """Normalized Difference Vegetation Index."""
        return (nir - red) / (nir + red + _EPS)

    @staticmethod
    def _ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """Normalized Difference Water Index (McFeeters)."""
        return (green - nir) / (green + nir + _EPS)

    @staticmethod
    def _ndbi(swir1: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """Normalized Difference Built-up Index."""
        return (swir1 - nir) / (swir1 + nir + _EPS)

    @staticmethod
    def _savi(nir: np.ndarray, red: np.ndarray, L: float = 0.5) -> np.ndarray:
        """Soil-Adjusted Vegetation Index."""
        return ((nir - red) / (nir + red + L + _EPS)) * (1 + L)

    @staticmethod
    def _evi(
        nir: np.ndarray, red: np.ndarray, blue: np.ndarray,
        G: float = 2.5, C1: float = 6.0, C2: float = 7.5, L: float = 1.0,
    ) -> np.ndarray:
        """Enhanced Vegetation Index."""
        denom = nir + C1 * red - C2 * blue + L + _EPS
        return G * (nir - red) / denom
