"""
pages/single_year.py
Single-Year Analysis page: resolve AOI, generate land-cover map and
statistics for one year.
"""

from __future__ import annotations

import streamlit as st
from streamlit_folium import st_folium

from config import LAND_COVER_CLASSES, MODEL_DIR
from core.pipeline import LandCoverPipeline
from utils.exports import dataframe_to_csv_bytes, export_classification_geotiff, geotiff_to_bytes
from utils.logger import get_logger
from utils.visualization import build_classification_map, pie_chart_land_cover
from utils.state import get_value, set_value

logger = get_logger(__name__)


def render() -> None:
    """Render the Single-Year Analysis page."""
    st.title("📍 Single-Year Land Cover Analysis")
    st.markdown("Enter a city, select a year, and generate a land-cover classification.")

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        city_name = st.text_input("City Name", value=get_value("city_name", ""), placeholder="e.g. Mumbai")
    with col2:
        country = st.text_input("Country (optional)", value=get_value("country", ""), placeholder="e.g. India")
    with col3:
        year = st.number_input("Analysis Year", min_value=1990, max_value=2025, value=2025, step=1)

    model_type = st.selectbox(
        "Classification Model",
        options=["random_forest", "xgboost", "lightgbm"],
        index=0,
        help="Random Forest is the primary model. XGBoost and LightGBM are available for comparison.",
    )

    run = st.button("Run Analysis", type="primary")

    if run:
        if not city_name.strip():
            st.error("Please enter a city name.")
            return

        set_value("city_name", city_name)
        set_value("country", country)

        pipeline = LandCoverPipeline(model_type=model_type)

        with st.spinner(f"Resolving location for '{city_name}'..."):
            try:
                aoi = pipeline.resolve_aoi(city_name, country or None)
            except ValueError as exc:
                st.error(str(exc))
                return

        set_value("aoi", aoi)
        st.success(f"AOI resolved: {aoi.display_name} (lat={aoi.latitude:.4f}, lon={aoi.longitude:.4f})")

        with st.spinner(f"Acquiring imagery and classifying land cover for {year}... this may take a minute."):
            try:
                result = pipeline.analyze_year(aoi, int(year), model_type=model_type)
            except RuntimeError as exc:
                st.error(f"Failed to acquire imagery: {exc}")
                return
            except Exception as exc:  # noqa: BLE001
                logger.exception("Unexpected error during analysis")
                st.error(f"Unexpected error: {exc}")
                return

        set_value("single_year_result", result)
        st.success("Analysis complete.")

    result = get_value("single_year_result")
    aoi = get_value("aoi")

    if result is None or aoi is None:
        st.info("Run an analysis above to see results.")
        return

    st.divider()
    st.subheader(f"Results: {aoi.name} — {result.year}")

    # Scene metadata
    with st.expander("Selected Satellite Scene Metadata", expanded=False):
        st.json(result.scene.to_dict())

    # Model evaluation
    if result.evaluation is not None:
        with st.expander("Model Evaluation Metrics", expanded=False):
            st.metric("Accuracy", f"{result.evaluation.accuracy * 100:.2f}%")
            st.metric("Macro F1-score", f"{result.evaluation.f1_macro:.4f}")
            st.json(result.evaluation.report)

    # Land cover map
    st.markdown("### 🗺️ Land Cover Map")
    fmap = build_classification_map(aoi, result.classification_map, title=f"Land Cover {result.year}")
    st_folium(fmap, width=900, height=500, key="single_year_map")

    # Statistics
    from core.change_detection import ChangeDetector
    detector = ChangeDetector()
    stats_df = detector.class_area_statistics(result.classification_map)

    st.markdown("### 📊 Land Cover Statistics")
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
    with col_b:
        fig = pie_chart_land_cover(stats_df, title=f"Land Cover Share — {result.year}")
        st.plotly_chart(fig, use_container_width=True)

    # Downloads
    st.markdown("### ⬇️ Download Outputs")
    csv_bytes = dataframe_to_csv_bytes(stats_df, index=False)
    st.download_button(
        "Download Statistics (CSV)", data=csv_bytes,
        file_name=f"{aoi.name}_{result.year}_landcover_stats.csv", mime="text/csv",
    )

    if st.button("Generate Classified Raster (GeoTIFF)"):
        try:
            from rasterio.transform import Affine
            # Reconstruct a simple geotransform spanning the AOI bbox
            min_lon, min_lat, max_lon, max_lat = aoi.bbox
            h, w = result.classification_map.shape
            transform = Affine(
                (max_lon - min_lon) / w, 0, min_lon,
                0, -(max_lat - min_lat) / h, max_lat,
            )
            filename = f"{aoi.name}_{result.year}_classification.tif"
            path = export_classification_geotiff(result.classification_map, transform, "EPSG:4326", filename)
            tif_bytes = geotiff_to_bytes(path)
            st.download_button(
                "Download Classified Raster (GeoTIFF)", data=tif_bytes,
                file_name=filename, mime="image/tiff",
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to export GeoTIFF")
            st.error(f"Failed to export raster: {exc}")
