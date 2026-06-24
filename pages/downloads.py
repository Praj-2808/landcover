"""
pages/downloads.py
Downloads page: centralized export of classified rasters, change
rasters, CSV statistics, and change matrices from all completed analyses.
"""

from __future__ import annotations

from rasterio.transform import Affine

from core.change_detection import ChangeDetector
from utils.exports import dataframe_to_csv_bytes, export_classification_geotiff, export_change_geotiff, geotiff_to_bytes
from utils.logger import get_logger
from utils.state import get_value

import streamlit as st

logger = get_logger(__name__)


def _aoi_transform(aoi, shape) -> Affine:
    """Build a simple affine geotransform spanning the AOI bbox for the given raster shape."""
    min_lon, min_lat, max_lon, max_lat = aoi.bbox
    h, w = shape
    return Affine(
        (max_lon - min_lon) / w, 0, min_lon,
        0, -(max_lat - min_lat) / h, max_lat,
    )


def render() -> None:
    """Render the Downloads page."""
    st.title("⬇️ Downloads")
    st.markdown("Export classified rasters, change rasters, statistics, and matrices from your analyses.")

    aoi = get_value("aoi")
    single_result = get_value("single_year_result")
    comparison = get_value("comparison_result")
    trend_results = get_value("trend_results")
    trend_years = get_value("trend_years")

    if aoi is None:
        st.info("No analyses available yet. Run an analysis on another page first.")
        return

    detector = ChangeDetector()

    # Single-year outputs
    st.subheader("Single-Year Outputs")
    if single_result is None:
        st.write("No single-year analysis available.")
    else:
        stats_df = detector.class_area_statistics(single_result.classification_map)
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                f"Statistics CSV ({single_result.year})",
                data=dataframe_to_csv_bytes(stats_df, index=False),
                file_name=f"{aoi.name}_{single_result.year}_stats.csv",
                mime="text/csv",
                key="dl_single_csv",
            )
        with col2:
            if st.button(f"Generate Classified Raster ({single_result.year})", key="dl_single_tif_btn"):
                transform = _aoi_transform(aoi, single_result.classification_map.shape)
                filename = f"{aoi.name}_{single_result.year}_classification.tif"
                path = export_classification_geotiff(single_result.classification_map, transform, "EPSG:4326", filename)
                st.download_button(
                    "Download GeoTIFF", data=geotiff_to_bytes(path), file_name=filename,
                    mime="image/tiff", key="dl_single_tif",
                )

    st.divider()

    # Comparison outputs
    st.subheader("Multi-Year Comparison Outputs")
    if comparison is None:
        st.write("No multi-year comparison available.")
    else:
        result_from = comparison["result_from"]
        result_to = comparison["result_to"]
        change = comparison["change"]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button(
                "Transition Matrix (CSV)",
                data=dataframe_to_csv_bytes(change.transition_matrix_km2),
                file_name=f"{aoi.name}_{result_from.year}_{result_to.year}_transition_matrix.csv",
                mime="text/csv",
                key="dl_cmp_matrix",
            )
        with col2:
            st.download_button(
                "Change Details (CSV)",
                data=dataframe_to_csv_bytes(change.from_to_table, index=False),
                file_name=f"{aoi.name}_{result_from.year}_{result_to.year}_change_details.csv",
                mime="text/csv",
                key="dl_cmp_details",
            )
        with col3:
            if st.button("Generate Change Raster (GeoTIFF)", key="dl_change_tif_btn"):
                transform = _aoi_transform(aoi, change.change_map.shape)
                filename = f"{aoi.name}_{result_from.year}_{result_to.year}_change.tif"
                path = export_change_geotiff(change.change_map, transform, "EPSG:4326", filename)
                st.download_button(
                    "Download Change GeoTIFF", data=geotiff_to_bytes(path), file_name=filename,
                    mime="image/tiff", key="dl_change_tif",
                )

        col4, col5 = st.columns(2)
        with col4:
            if st.button(f"Generate Classified Raster ({result_from.year})", key="dl_from_tif_btn"):
                transform = _aoi_transform(aoi, result_from.classification_map.shape)
                filename = f"{aoi.name}_{result_from.year}_classification.tif"
                path = export_classification_geotiff(result_from.classification_map, transform, "EPSG:4326", filename)
                st.download_button(
                    "Download GeoTIFF", data=geotiff_to_bytes(path), file_name=filename,
                    mime="image/tiff", key="dl_from_tif",
                )
        with col5:
            if st.button(f"Generate Classified Raster ({result_to.year})", key="dl_to_tif_btn"):
                transform = _aoi_transform(aoi, result_to.classification_map.shape)
                filename = f"{aoi.name}_{result_to.year}_classification.tif"
                path = export_classification_geotiff(result_to.classification_map, transform, "EPSG:4326", filename)
                st.download_button(
                    "Download GeoTIFF", data=geotiff_to_bytes(path), file_name=filename,
                    mime="image/tiff", key="dl_to_tif",
                )

    st.divider()

    # Trend outputs
    st.subheader("Trend Analysis Outputs")
    if not trend_results or not trend_years:
        st.write("No trend analysis available.")
    else:
        import pandas as pd
        records = []
        for year in trend_years:
            stats_df = trend_results[year]["stats"]
            row = {"Year": year}
            for _, r in stats_df.iterrows():
                row[r["Class"]] = r["Area (km²)"]
            records.append(row)
        trend_df = pd.DataFrame(records)
        st.download_button(
            "Trend Data (CSV)",
            data=dataframe_to_csv_bytes(trend_df, index=False),
            file_name=f"{aoi.name}_trend_{trend_years[0]}_{trend_years[-1]}.csv",
            mime="text/csv",
            key="dl_trend_csv",
        )
