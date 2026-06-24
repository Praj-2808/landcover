"""
core/aoi.py
Area of Interest (AOI) resolution: convert a city name (and optional country)
into a bounding box / boundary suitable for satellite imagery queries.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple

from geopy.geocoders import Nominatim
from geopy.exc import GeocoderServiceError, GeocoderTimedOut

from config import CACHE_DIR, DEFAULT_AOI_BUFFER_DEG, GEOCODER_USER_AGENT
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AreaOfInterest:
    """Represents a resolved Area of Interest."""

    name: str
    country: Optional[str]
    latitude: float
    longitude: float
    # bounding box: (min_lon, min_lat, max_lon, max_lat)
    bbox: Tuple[float, float, float, float]
    display_name: str

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def geojson_bbox(self) -> dict:
        """Return the bounding box as a GeoJSON Polygon geometry."""
        min_lon, min_lat, max_lon, max_lat = self.bbox
        return {
            "type": "Polygon",
            "coordinates": [[
                [min_lon, min_lat],
                [max_lon, min_lat],
                [max_lon, max_lat],
                [min_lon, max_lat],
                [min_lon, min_lat],
            ]],
        }


class AOIResolver:
    """
    Resolves a city name (and optional country) into an AreaOfInterest
    using OpenStreetMap Nominatim geocoding, with local file-based caching
    to avoid repeated lookups.
    """

    def __init__(self, buffer_deg: float = DEFAULT_AOI_BUFFER_DEG) -> None:
        self.buffer_deg = buffer_deg
        self._geolocator = Nominatim(user_agent=GEOCODER_USER_AGENT, timeout=10)
        self._cache_file = Path(CACHE_DIR) / "aoi_cache.json"
        self._cache = self._load_cache()

    def _load_cache(self) -> dict:
        if self._cache_file.exists():
            try:
                with open(self._cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to load AOI cache: %s", exc)
        return {}

    def _save_cache(self) -> None:
        try:
            with open(self._cache_file, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=2)
        except OSError as exc:
            logger.warning("Failed to save AOI cache: %s", exc)

    def resolve(
        self,
        city_name: str,
        country: Optional[str] = None,
        buffer_deg: Optional[float] = None,
        max_retries: int = 3,
    ) -> AreaOfInterest:
        """
        Resolve a city (and optional country) to an AreaOfInterest.

        Args:
            city_name: Name of the city, e.g. "Mumbai".
            country: Optional country name, e.g. "India".
            buffer_deg: Optional override for the AOI buffer in degrees.
            max_retries: Number of geocoding retries on transient failure.

        Returns:
            An AreaOfInterest instance.

        Raises:
            ValueError: If the location cannot be geocoded.
        """
        buffer_deg = buffer_deg if buffer_deg is not None else self.buffer_deg
        query = f"{city_name}, {country}" if country else city_name
        cache_key = f"{query.lower().strip()}|{buffer_deg}"

        if cache_key in self._cache:
            logger.info("AOI cache hit for '%s'", query)
            cached = self._cache[cache_key]
            return AreaOfInterest(
                name=cached["name"],
                country=cached["country"],
                latitude=cached["latitude"],
                longitude=cached["longitude"],
                bbox=tuple(cached["bbox"]),
                display_name=cached["display_name"],
            )

        last_error: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                logger.info("Geocoding '%s' (attempt %d/%d)", query, attempt, max_retries)
                location = self._geolocator.geocode(query, exactly_one=True)
                if location is None:
                    raise ValueError(f"Could not geocode location: '{query}'")

                lat, lon = location.latitude, location.longitude
                bbox = (
                    lon - buffer_deg,
                    lat - buffer_deg,
                    lon + buffer_deg,
                    lat + buffer_deg,
                )

                aoi = AreaOfInterest(
                    name=city_name,
                    country=country,
                    latitude=lat,
                    longitude=lon,
                    bbox=bbox,
                    display_name=location.address,
                )

                self._cache[cache_key] = aoi.to_dict()
                self._cache[cache_key]["bbox"] = list(aoi.bbox)
                self._save_cache()

                logger.info(
                    "Resolved AOI for '%s': lat=%.4f, lon=%.4f, bbox=%s",
                    query, lat, lon, bbox,
                )
                return aoi

            except (GeocoderServiceError, GeocoderTimedOut) as exc:
                last_error = exc
                logger.warning("Geocoding attempt %d failed: %s", attempt, exc)
                time.sleep(1.5 * attempt)

        raise ValueError(
            f"Failed to resolve AOI for '{query}' after {max_retries} attempts: {last_error}"
        )
