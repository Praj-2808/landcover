"""
core/change_detection.py
Post-classification change detection: builds transition matrices,
computes area gained/lost per class, and derives urban expansion and
vegetation loss analyses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from config import LAND_COVER_CLASSES, CLASS_NAME_TO_CODE, TARGET_RESOLUTION_M
from utils.logger import get_logger

logger = get_logger(__name__)

PIXEL_AREA_KM2 = (TARGET_RESOLUTION_M ** 2) / 1_000_000.0


@dataclass
class ChangeResult:
    """Container for change detection outputs."""

    transition_matrix_km2: pd.DataFrame   # rows=from class, cols=to class, values=km2
    transition_matrix_pct: pd.DataFrame   # values = % of total AOI area
    area_by_class_from: pd.Series         # km2 per class in "from" year
    area_by_class_to: pd.Series           # km2 per class in "to" year
    net_change_km2: pd.Series             # to - from per class
    net_change_pct: pd.Series             # percentage change relative to "from" area
    change_map: np.ndarray                # 2D array: 1 where class changed, 0 otherwise
    from_to_table: pd.DataFrame           # long-format From/To/Area/Percent table


class ChangeDetector:
    """Performs post-classification change detection between two classification maps."""

    def __init__(self) -> None:
        self.class_names = [LAND_COVER_CLASSES[k] for k in sorted(LAND_COVER_CLASSES.keys())]
        self.class_codes = sorted(LAND_COVER_CLASSES.keys())

    def detect_change(
        self, map_from: np.ndarray, map_to: np.ndarray
    ) -> ChangeResult:
        """
        Compare two classification maps (same shape) and compute all
        change-detection products.

        Args:
            map_from: 2D int array of class codes for the earlier year.
            map_to: 2D int array of class codes for the later year.

        Returns:
            A ChangeResult with transition git statusmatrices, area statistics, and
            a binary change map.
        """
        if map_from.shape != map_to.shape: 
            logger.warning("Shape mismatch: %s vs %s. Cropping to common size.", map_from.shape, map_to.shape,)
            h = min(map_from.shape[0], map_to.shape[0])
            w = min(map_from.shape[1], map_to.shape[1])
            map_from = map_from[:h, :w]
            map_to = map_to[:h, :w]
        
        logger.info("Running change detection on maps of shape %s", map_from.shape)

        flat_from = map_from.reshape(-1)
        flat_to = map_to.reshape(-1)

        # Transition matrix in pixel counts
        n_classes = len(self.class_codes)
        counts = np.zeros((n_classes, n_classes), dtype=np.int64)
        for i, c_from in enumerate(self.class_codes):
            from_mask = flat_from == c_from
            if not np.any(from_mask):
                continue
            to_vals = flat_to[from_mask]
            for j, c_to in enumerate(self.class_codes):
                counts[i, j] = np.sum(to_vals == c_to)

        transition_km2 = counts.astype(np.float64) * PIXEL_AREA_KM2
        transition_df = pd.DataFrame(
            transition_km2, index=self.class_names, columns=self.class_names
        )

        total_area_km2 = flat_from.size * PIXEL_AREA_KM2
        transition_pct_df = (transition_df / total_area_km2) * 100.0

        area_from = transition_df.sum(axis=1)
        area_from.name = "area_from_km2"
        area_to = transition_df.sum(axis=0)
        area_to.name = "area_to_km2"

        net_change = area_to - area_from
        net_change.name = "net_change_km2"

        net_change_pct = (net_change / area_from.replace(0, np.nan)) * 100.0
        net_change_pct.name = "net_change_pct"
        net_change_pct = net_change_pct.fillna(0.0)

        change_map = (map_from != map_to).astype(np.int8)

        from_to_table = self._build_from_to_table(transition_df, total_area_km2)

        logger.info(
            "Change detection complete. %.2f%% of AOI changed class.",
            100.0 * np.mean(change_map),
        )

        return ChangeResult(
            transition_matrix_km2=transition_df,
            transition_matrix_pct=transition_pct_df,
            area_by_class_from=area_from,
            area_by_class_to=area_to,
            net_change_km2=net_change,
            net_change_pct=net_change_pct,
            change_map=change_map,
            from_to_table=from_to_table,
        )

    def _build_from_to_table(
        self, transition_df: pd.DataFrame, total_area_km2: float
    ) -> pd.DataFrame:
        """Build a long-format From/To/Area/Percent table, excluding the diagonal (no change)."""
        records: List[dict] = []
        for from_class in transition_df.index:
            for to_class in transition_df.columns:
                if from_class == to_class:
                    continue
                area = transition_df.loc[from_class, to_class]
                if area <= 0:
                    continue
                records.append({
                    "From Class": from_class,
                    "To Class": to_class,
                    "Area Changed (km²)": round(float(area), 4),
                    "Percentage Changed (%)": round(float(area / total_area_km2 * 100.0), 4),
                })
        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values("Area Changed (km²)", ascending=False).reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Specialized analyses
    # ------------------------------------------------------------------
    def urban_expansion_analysis(self, result: ChangeResult) -> Dict:
        """
        Summarize urban growth: new urban area, growth percentage, and
        top source classes for urban expansion (growth "hotspots" by class).

        Args:
            result: Output of detect_change().

        Returns:
            Dict with 'new_urban_area_km2', 'urban_growth_pct', and
            'growth_sources' (DataFrame of contributing from-classes).
        """
        urban = "Urban/Built-up"
        area_from = result.area_by_class_from.get(urban, 0.0)
        area_to = result.area_by_class_to.get(urban, 0.0)

        new_urban = result.transition_matrix_km2.drop(index=urban, errors="ignore")[urban] \
            if urban in result.transition_matrix_km2.columns else pd.Series(dtype=float)
        new_urban = new_urban[new_urban > 0].sort_values(ascending=False)

        growth_pct = ((area_to - area_from) / area_from * 100.0) if area_from > 0 else 0.0

        sources_df = new_urban.reset_index()
        sources_df.columns = ["Source Class", "New Urban Area (km²)"]

        return {
            "previous_urban_area_km2": round(float(area_from), 4),
            "new_total_urban_area_km2": round(float(area_to), 4),
            "new_urban_area_km2": round(float(new_urban.sum()), 4),
            "urban_growth_pct": round(float(growth_pct), 4),
            "growth_sources": sources_df,
        }

    def vegetation_loss_analysis(self, result: ChangeResult) -> Dict:
        """
        Summarize vegetation and agriculture loss, including total green
        cover reduction.

        Args:
            result: Output of detect_change().

        Returns:
            Dict with forest (vegetation) loss, agriculture loss, and
            combined green cover reduction statistics.
        """
        veg = "Vegetation"
        agri = "Agriculture"

        veg_from = result.area_by_class_from.get(veg, 0.0)
        veg_to = result.area_by_class_to.get(veg, 0.0)
        veg_loss = veg_from - veg_to

        agri_from = result.area_by_class_from.get(agri, 0.0)
        agri_to = result.area_by_class_to.get(agri, 0.0)
        agri_loss = agri_from - agri_to

        green_from = veg_from + agri_from
        green_to = veg_to + agri_to
        green_loss = green_from - green_to
        green_loss_pct = (green_loss / green_from * 100.0) if green_from > 0 else 0.0

        veg_to_urban = result.transition_matrix_km2.loc[veg, "Urban/Built-up"] \
            if "Urban/Built-up" in result.transition_matrix_km2.columns else 0.0
        agri_to_urban = result.transition_matrix_km2.loc[agri, "Urban/Built-up"] \
            if "Urban/Built-up" in result.transition_matrix_km2.columns else 0.0

        return {
            "vegetation_area_loss_km2": round(float(veg_loss), 4),
            "vegetation_loss_pct": round(float(veg_loss / veg_from * 100.0) if veg_from > 0 else 0.0, 4),
            "agriculture_area_loss_km2": round(float(agri_loss), 4),
            "agriculture_loss_pct": round(float(agri_loss / agri_from * 100.0) if agri_from > 0 else 0.0, 4),
            "total_green_cover_loss_km2": round(float(green_loss), 4),
            "green_cover_loss_pct": round(float(green_loss_pct), 4),
            "vegetation_converted_to_urban_km2": round(float(veg_to_urban), 4),
            "agriculture_converted_to_urban_km2": round(float(agri_to_urban), 4),
        }

    def class_area_statistics(self, classification_map: np.ndarray) -> pd.DataFrame:
        """
        Compute area and percentage statistics for a single classification map.

        Args:
            classification_map: 2D int array of class codes.

        Returns:
            DataFrame with columns: Class, Pixel Count, Area (km²), Percentage (%).
        """
        flat = classification_map.reshape(-1)
        total = flat.size
        records = []
        for code in self.class_codes:
            count = int(np.sum(flat == code))
            records.append({
                "Class": LAND_COVER_CLASSES[code],
                "Pixel Count": count,
                "Area (km²)": round(count * PIXEL_AREA_KM2, 4),
                "Percentage (%)": round(count / total * 100.0, 4) if total > 0 else 0.0,
            })
        return pd.DataFrame(records)
