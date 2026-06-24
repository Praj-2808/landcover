"""
pages/home.py
Home page: project overview and quick navigation guidance.
"""

from __future__ import annotations

import streamlit as st

from config import LAND_COVER_CLASSES, LAND_COVER_COLORS, QUICK_COMPARISON_OFFSETS


def render() -> None:
    """Render the Home page."""
    st.title("🛰️ Land Cover Classification & Change Detection")

    st.markdown(
        """
        This application provides an end-to-end workflow for analyzing land
        cover and detecting change over time for any city in the world,
        using free satellite imagery (Sentinel-2 / Landsat via the
        Microsoft Planetary Computer) and machine learning classification.
        """
    )

    st.subheader("How it works")
    st.markdown(
        """
        1. **Enter a location** — type a city name (and optionally a country).
        2. **Automatic AOI detection** — the app geocodes the city and defines
           an Area of Interest (AOI) bounding box around it.
        3. **Imagery acquisition** — the app searches for the best available,
           lowest-cloud-cover satellite scene for each requested year
           (Sentinel-2 for recent years, Landsat for historical years).
        4. **Feature engineering** — spectral indices (NDVI, NDWI, NDBI,
           SAVI, EVI) are computed automatically.
        5. **Classification** — a Random Forest / XGBoost / LightGBM model is
           trained on automatically generated training samples and used to
           classify every pixel into one of five land-cover classes.
        6. **Change detection** — for multi-year analyses, the app compares
           classification maps to produce transition matrices, urban
           expansion stats, and vegetation loss analyses.
        7. **Visualization & download** — interactive maps and charts, with
           CSV / GeoTIFF export.
        """
    )

    st.subheader("Land Cover Classes")
    cols = st.columns(len(LAND_COVER_CLASSES))
    for col, (code, name) in zip(cols, LAND_COVER_CLASSES.items()):
        with col:
            st.markdown(
                f"<div style='background:{LAND_COVER_COLORS[code]};"
                f"border-radius:6px;padding:10px;text-align:center;"
                f"color:white;font-weight:600;'>{name}</div>",
                unsafe_allow_html=True,
            )

    st.subheader("Available Pages")
    st.markdown(
        """
        - **Single-Year Analysis** — generate a land-cover map and statistics for one year.
        - **Multi-Year Comparison** — compare land cover between two years (including quick
          presets: current vs 5 / 10 / 15 / 20 years ago).
        - **Trend Analysis** — analyze land-cover trends across multiple years.
        - **Change Detection** — detailed change analytics, transition matrices, and
          urban/vegetation analyses.
        - **Interactive Maps** — explore classification and change layers on interactive maps.
        - **Downloads** — export classified rasters, change rasters, and CSV statistics.
        """
    )

    st.info(
        f"Quick comparison presets available: "
        + ", ".join(f"Current vs {y} years ago" for y in QUICK_COMPARISON_OFFSETS)
    )

    st.caption(
        "Data sources: Sentinel-2 L2A and Landsat Collection 2 Level-2, "
        "via the Microsoft Planetary Computer STAC API (free, no authentication required)."
    )
