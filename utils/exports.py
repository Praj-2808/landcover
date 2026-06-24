"""
utils/exports.py
Helpers for exporting analysis outputs: classified rasters (GeoTIFF),
change rasters, CSV statistics, and transition matrices.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine

from config import OUTPUT_DIR
from utils.logger import get_logger

logger = get_logger(__name__)


def export_classification_geotiff(
    classification_map: np.ndarray, transform: Optional[Affine], crs: Optional[str],
    filename: str,
) -> Path:
    """
    Export a classification map as a single-band GeoTIFF.

    Args:
        classification_map: 2D int array of class codes.
        transform: Affine geotransform (if None, an identity transform is used).
        crs: Coordinate reference system string (e.g. 'EPSG:4326'). If None,
            EPSG:4326 is used as a fallback.
        filename: Output filename (without directory), e.g. 'mumbai_2025_classification.tif'.

    Returns:
        Path to the written GeoTIFF.
    """
    out_path = Path(OUTPUT_DIR) / filename
    transform = transform or Affine.identity()
    crs = crs or "EPSG:4326"

    data = classification_map.astype(np.uint8)
    with rasterio.open(
        out_path, "w", driver="GTiff",
        height=data.shape[0], width=data.shape[1],
        count=1, dtype=data.dtype,
        crs=crs, transform=transform,
        compress="lzw",
    ) as dst:
        dst.write(data, 1)

    logger.info("Exported classification GeoTIFF to %s", out_path)
    return out_path


def export_change_geotiff(
    change_map: np.ndarray, transform: Optional[Affine], crs: Optional[str], filename: str,
) -> Path:
    """Export a binary change map as a single-band GeoTIFF."""
    return export_classification_geotiff(change_map, transform, crs, filename)


def export_dataframe_csv(df: pd.DataFrame, filename: str) -> Path:
    """
    Export a DataFrame to CSV in the outputs directory.

    Args:
        df: DataFrame to export.
        filename: Output filename (without directory), e.g. 'stats.csv'.

    Returns:
        Path to the written CSV file.
    """
    out_path = Path(OUTPUT_DIR) / filename
    df.to_csv(out_path, index=True)
    logger.info("Exported CSV to %s", out_path)
    return out_path


def dataframe_to_csv_bytes(df: pd.DataFrame, index: bool = True) -> bytes:
    """Return CSV bytes for a DataFrame, suitable for st.download_button."""
    return df.to_csv(index=index).encode("utf-8")


def geotiff_to_bytes(path: Path) -> bytes:
    """Read a file from disk and return its raw bytes (for download buttons)."""
    with open(path, "rb") as f:
        return f.read()
