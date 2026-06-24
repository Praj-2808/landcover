"""
core/imagery.py
Satellite imagery acquisition using the Microsoft Planetary Computer STAC API.

Implements intelligent image selection:
  - Automatically chooses Sentinel-2 (recent years) or Landsat (historical years).
  - Searches Jan 1 - Dec 31 of the target year.
  - Applies a cloud-cover threshold, increasing it gradually if no scene is found.
  - Selects the scene with lowest cloud cover / best coverage / best quality.
  - Caches downloaded band arrays to disk to avoid re-downloading.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import planetary_computer
import pystac_client
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

from config import (
    COMMON_BANDS,
    IMAGERY_DIR,
    INITIAL_CLOUD_COVER_THRESHOLD,
    CLOUD_COVER_STEP,
    MAX_CLOUD_COVER_THRESHOLD,
    LANDSAT_BAND_MAP,
    LANDSAT_COLLECTION,
    SENTINEL2_BAND_MAP,
    SENTINEL2_COLLECTION,
    SENTINEL2_MIN_YEAR,
    STAC_API_URL,
    TARGET_RESOLUTION_M,
)
from core.aoi import AreaOfInterest
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SceneMetadata:
    """Metadata describing the selected satellite scene."""

    collection: str
    item_id: str
    datetime: str
    cloud_cover: float
    platform: str
    resolution_m: int

    def to_dict(self) -> dict:
        return {
            "collection": self.collection,
            "item_id": self.item_id,
            "datetime": self.datetime,
            "cloud_cover": self.cloud_cover,
            "platform": self.platform,
            "resolution_m": self.resolution_m,
        }


@dataclass
class ImageryResult:
    """Container for acquired imagery: stacked band array + metadata."""

    bands: Dict[str, np.ndarray]  # band name -> 2D array
    transform: rasterio.Affine
    crs: str
    width: int
    height: int
    scene: SceneMetadata

    def stack(self, band_order: Optional[List[str]] = None) -> np.ndarray:
        """Return bands stacked into a (H, W, B) array in the given order."""
        order = band_order or COMMON_BANDS
        return np.stack([self.bands[b] for b in order], axis=-1)


class ImageryAcquisition:
    """
    Handles searching, selecting, and downloading satellite imagery for a
    given AOI and year, using the Planetary Computer STAC API.
    """

    def __init__(self) -> None:
        self._catalog = pystac_client.Client.open(
            STAC_API_URL, modifier=planetary_computer.sign_inplace
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_imagery_for_year(self, aoi: AreaOfInterest, year: int) -> ImageryResult:
        """
        Retrieve the best available satellite imagery for the given AOI and year.

        Automatically selects Sentinel-2 for years >= SENTINEL2_MIN_YEAR and
        Landsat Collection 2 for earlier years. Uses local caching to avoid
        re-downloading previously fetched scenes.

        Args:
            aoi: The resolved Area of Interest.
            year: The target year (1-365 day window searched).

        Returns:
            An ImageryResult containing band arrays and scene metadata.

        Raises:
            RuntimeError: If no suitable imagery can be found even at the
                maximum cloud cover threshold.
        """
        collection = (
            SENTINEL2_COLLECTION if year >= SENTINEL2_MIN_YEAR else LANDSAT_COLLECTION
        )
        band_map = SENTINEL2_BAND_MAP if collection == SENTINEL2_COLLECTION else LANDSAT_BAND_MAP

        cache_path = self._cache_path(aoi, year, collection)
        if cache_path.exists():
            logger.info("Loading cached imagery for %s %d from %s", aoi.name, year, cache_path)
            return self._load_from_cache(cache_path)

        item, cloud_cover = self._search_best_item(aoi, year, collection)
        logger.info(
            "Selected %s item '%s' for %s %d (cloud cover %.2f%%)",
            collection, item.id, aoi.name, year, cloud_cover,
        )

        bands, transform, crs, width, height = self._download_bands(item, aoi, band_map)

        scene = SceneMetadata(
            collection=collection,
            item_id=item.id,
            datetime=str(item.datetime),
            cloud_cover=cloud_cover,
            platform=item.properties.get("platform", "unknown"),
            resolution_m=TARGET_RESOLUTION_M,
        )

        result = ImageryResult(
            bands=bands, transform=transform, crs=crs, width=width, height=height, scene=scene
        )
        self._save_to_cache(cache_path, result)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _cache_path(self, aoi: AreaOfInterest, year: int, collection: str) -> Path:
        key = f"{aoi.name}_{aoi.bbox}_{year}_{collection}"
        digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:16]
        return Path(IMAGERY_DIR) / f"{aoi.name.replace(' ', '_')}_{year}_{collection}_{digest}.npz"

    def _save_to_cache(self, path: Path, result: ImageryResult) -> None:
        try:
            np.savez_compressed(
                path,
                **{f"band_{k}": v for k, v in result.bands.items()},
                transform=np.array(result.transform.to_gdal()),
                crs=np.array(result.crs.encode("utf-8")),
                width=result.width,
                height=result.height,
                scene=np.array(json.dumps(result.scene.to_dict()).encode("utf-8")),
            )
            logger.info("Cached imagery to %s", path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to cache imagery: %s", exc)

    def _load_from_cache(self, path: Path) -> ImageryResult:
        data = np.load(path, allow_pickle=False)
        bands = {
            b: data[f"band_{b}"] for b in COMMON_BANDS if f"band_{b}" in data
        }
        transform = rasterio.Affine.from_gdal(*data["transform"].tolist())
        crs = bytes(data["crs"]).decode("utf-8")
        width = int(data["width"])
        height = int(data["height"])
        scene_dict = json.loads(bytes(data["scene"]).decode("utf-8"))
        scene = SceneMetadata(**scene_dict)
        return ImageryResult(
            bands=bands, transform=transform, crs=crs, width=width, height=height, scene=scene
        )

    def _search_best_item(
        self, aoi: AreaOfInterest, year: int, collection: str
    ):
        """
        Search the STAC catalog for the best item, gradually relaxing the
        cloud cover threshold until a match is found.

        Returns:
            Tuple of (selected STAC item, cloud cover percentage).
        """
        date_range = f"{year}-01-01/{year}-12-31"
        threshold = INITIAL_CLOUD_COVER_THRESHOLD

        while threshold <= MAX_CLOUD_COVER_THRESHOLD:
            logger.info(
                "Searching %s for %s in %s (cloud cover < %.0f%%)",
                collection, aoi.name, date_range, threshold,
            )
            search = self._catalog.search(
                collections=[collection],
                bbox=list(aoi.bbox),
                datetime=date_range,
                query={"eo:cloud_cover": {"lt": threshold}},
            )
            items = list(search.items())

            if collection == LANDSAT_COLLECTION:
                # Restrict to surface reflectance Landsat sensors with required bands
                items = [
                    it for it in items
                    if it.properties.get("platform", "").lower().startswith("landsat")
                ]

            if items:
                best = min(items, key=lambda it: it.properties.get("eo:cloud_cover", 100.0))
                cloud_cover = float(best.properties.get("eo:cloud_cover", 100.0))
                logger.info(
                    "Found %d candidate(s); best cloud cover = %.2f%%",
                    len(items), cloud_cover,
                )
                return best, cloud_cover

            logger.warning(
                "No %s scenes found for %s in %d with cloud cover < %.0f%%. Relaxing threshold.",
                collection, aoi.name, year, threshold,
            )
            threshold += CLOUD_COVER_STEP

        raise RuntimeError(
            f"No suitable {collection} imagery found for {aoi.name} in {year} "
            f"even at {MAX_CLOUD_COVER_THRESHOLD}% cloud cover threshold."
        )

    def _download_bands(
        self, item, aoi: AreaOfInterest, band_map: Dict[str, str]
    ) -> Tuple[Dict[str, np.ndarray], rasterio.Affine, str, int, int]:
        """
        Download and read each required band, clipped to the AOI bounding box,
        resampled to TARGET_RESOLUTION_M.

        Returns:
            Tuple of (band arrays dict, transform, crs string, width, height).
        """
        bands: Dict[str, np.ndarray] = {}
        ref_transform: Optional[rasterio.Affine] = None
        ref_crs: Optional[str] = None
        ref_shape: Optional[Tuple[int, int]] = None

        for common_name, asset_key in band_map.items():
            if asset_key not in item.assets:
                logger.warning(
                    "Band '%s' (asset '%s') not found in item %s; skipping.",
                    common_name, asset_key, item.id,
                )
                continue

            asset = item.assets[asset_key]
            href = asset.href

            with rasterio.open(href) as src:
                # Transform AOI bbox into the raster's CRS
                bbox_in_crs = transform_bounds("EPSG:4326", src.crs, *aoi.bbox)
                window = from_bounds(*bbox_in_crs, transform=src.transform)

                # Compute output shape based on target resolution
                src_res = src.res[0]
                scale = src_res / TARGET_RESOLUTION_M if src_res > 0 else 1.0
                out_height = max(1, int(window.height * scale)) if scale < 1 else max(1, int(window.height))
                out_width = max(1, int(window.width * scale)) if scale < 1 else max(1, int(window.width))

                # Ensure consistent shape across bands of differing native resolution
                if ref_shape is None:
                    # Target ~ resolution based on first band processed
                    target_h = max(1, int(window.height * src_res / TARGET_RESOLUTION_M))
                    target_w = max(1, int(window.width * src_res / TARGET_RESOLUTION_M))
                    ref_shape = (target_h, target_w)

                data = src.read(
                    1,
                    window=window,
                    out_shape=ref_shape,
                    resampling=Resampling.bilinear,
                    boundless=True,
                    fill_value=0,
                ).astype(np.float32)

                if ref_transform is None:
                    win_transform = src.window_transform(window)
                    scale_x = window.width / ref_shape[1]
                    scale_y = window.height / ref_shape[0]
                    ref_transform = win_transform * rasterio.Affine.scale(scale_x, scale_y)
                    ref_crs = str(src.crs)

                bands[common_name] = data

        if not bands:
            raise RuntimeError(f"No bands could be downloaded for item {item.id}")

        h, w = next(iter(bands.values())).shape
        return bands, ref_transform, ref_crs, w, h
