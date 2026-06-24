"""
pages/multi_year.py
Multi-Year Comparison page: compare land cover between two years, with
quick "Current vs N years ago" presets.
"""

from __future__ import annotations

import streamlit as st
from streamlit_folium import st_folium

from config import QUICK_COMPARISON_OFFSETS
from core.pipeline import LandCoverPipeline
from utils.exports import dataframe_to_csv_bytes
from utils.logger import get_logger
from utils.visualization import (
    bar_chart_area_comparison,
    build_change_map,
    build_classification_map,
    sankey_transitions,
)
from utils.state import get_value, set_value

logger = get_logger(__name__)


def render() -> None:
    """Render the Multi-Year Comparison page."""
    st.title("🔁 Multi-Year Land Cover Comparison")
    st.markdown("Compare land cover between two years for the same location.")

    col1, col2 = st.columns([2, 2])
    with col1:
        city_name = st.text_input("City Name", value=get_value("city_name", ""), placeholder="e.g. Mumbai", key="my_city")
    with col2:
        country = st.text_input("Country (optional)", value=get_value("country", ""), placeholder="e.g. India", key="my_country")

    st.markdown("#### Select Comparison Mode")
    mode = st.radio(
        "Comparison Mode",
        options=["Manual Year Selection", "Automatic Time Difference"],
        horizontal=True,
    )

    if mode == "Manual Year Selection":
        col_a, col_b = st.columns(2)
        with col_a:
            year_to = st.number_input("Current / Recent Year", min_value=1991, max_value=2025, value=2025, step=1)
        with col_b:
            year_from = st.number_input("Earlier Year", min_value=1990, max_value=2024, value=2020, step=1)
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            current_year = st.number_input("Current Year", min_value=1995, max_value=2025, value=2025, step=1)
        with col_b:
            offset = st.selectbox(
                "Comparison Window",
                options=QUICK_COMPARISON_OFFSETS,
                format_func=lambda x: f"Current vs {x} years ago",
            )
        year_from, year_to = current_year - offset, current_year
        st.caption(f"This will compare **{year_from}** vs **{year_to}**.")

    model_type = st.selectbox(
        "Classification Model", options=["random_forest", "xgboost", "lightgbm"], index=0,
    )

    run = st.button("Run Comparison", type="primary")

    if run:
        if not city_name.strip():
            st.error("Please enter a city name.")
            return
        if year_from >= year_to:
            st.error("The earlier year must be strictly before the later year.")
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

        with st.spinner(f"Analyzing {year_from} and {year_to}... this may take a few minutes."):
            try:
                comparison = pipeline.compare_years(aoi, int(year_from), int(year_to), model_type=model_type)
            except RuntimeError as exc:
                st.error(f"Failed to acquire imagery: {exc}")
                return
            except Exception as exc:  # noqa: BLE001
                logger.exception("Unexpected error during comparison")
                st.error(f"Unexpected error: {exc}")
                return

        set_value("comparison_result", comparison)
        st.success("Comparison complete.")

    comparison = get_value("comparison_result")
    aoi = get_value("aoi")

    if comparison is None or aoi is None:
        st.info("Run a comparison above to see results.")
        return

    result_from = comparison["result_from"]
    result_to = comparison["result_to"]
    change = comparison["change"]

    st.divider()
    st.subheader(f"Results: {aoi.name} — {result_from.year} vs {result_to.year}")

    # Side-by-side maps
    st.markdown("### 🗺️ Land Cover Maps")
    map_col1, map_col2 = st.columns(2)
    with map_col1:
        st.markdown(f"**{result_from.year}**")
        fmap_from = build_classification_map(aoi, result_from.classification_map, title=f"Land Cover {result_from.year}")
        st_folium(fmap_from, width=440, height=400, key="map_from")
    with map_col2:
        st.markdown(f"**{result_to.year}**")
        fmap_to = build_classification_map(aoi, result_to.classification_map, title=f"Land Cover {result_to.year}")
        st_folium(fmap_to, width=440, height=400, key="map_to")

    # Change map
    st.markdown("### 🔥 Change Map")
    change_fmap = build_change_map(aoi, change.change_map, title="Areas of Change")
    st_folium(change_fmap, width=900, height=450, key="change_map")
    pct_changed = float(change.change_map.mean() * 100)
    st.metric("Area Changed", f"{pct_changed:.2f}% of AOI")

    # Transition matrix
    st.markdown("### 🔢 Transition Matrix (km²)")
    st.dataframe(change.transition_matrix_km2.round(3), use_container_width=True)

    # From/To table
    st.markdown("### 📋 Change Details (From → To)")
    st.dataframe(change.from_to_table, use_container_width=True, hide_index=True)

    # Area gained/lost
    st.markdown("### 📈 Area Gained / Lost")
    area_summary = change.net_change_km2.to_frame("Net Change (km²)")
    area_summary["Net Change (%)"] = change.net_change_pct.round(2)
    st.dataframe(area_summary, use_container_width=True)

    # Bar chart comparison
    st.markdown("### 📊 Area Comparison")
    bar_fig = bar_chart_area_comparison(
        change.area_by_class_from, change.area_by_class_to, result_from.year, result_to.year,
    )
    st.plotly_chart(bar_fig, use_container_width=True)

    # Sankey
    st.markdown("### 🌊 Land Cover Transitions (Sankey)")
    sankey_fig = sankey_transitions(change.from_to_table, result_from.year, result_to.year)
    st.plotly_chart(sankey_fig, use_container_width=True)

    # Downloads
    st.markdown("### ⬇️ Download Outputs")
    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        st.download_button(
            "Transition Matrix (CSV)",
            data=dataframe_to_csv_bytes(change.transition_matrix_km2),
            file_name=f"{aoi.name}_{result_from.year}_{result_to.year}_transition_matrix.csv",
            mime="text/csv",
        )
    with col_d2:
        st.download_button(
            "Change Details (CSV)",
            data=dataframe_to_csv_bytes(change.from_to_table, index=False),
            file_name=f"{aoi.name}_{result_from.year}_{result_to.year}_change_details.csv",
            mime="text/csv",
        )
    with col_d3:
        st.download_button(
            "Net Change Summary (CSV)",
            data=dataframe_to_csv_bytes(area_summary),
            file_name=f"{aoi.name}_{result_from.year}_{result_to.year}_net_change.csv",
            mime="text/csv",
        )
