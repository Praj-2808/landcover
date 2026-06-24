"""
pages/change_detection_page.py
Change Detection page: detailed change analytics including urban
expansion and vegetation loss analyses, built on the most recent
Multi-Year Comparison result.
"""

from __future__ import annotations

import streamlit as st

from core.change_detection import ChangeDetector
from utils.exports import dataframe_to_csv_bytes
from utils.visualization import change_heatmap
from utils.state import get_value


def render() -> None:
    """Render the Change Detection page."""
    st.title("🧭 Change Detection Analytics")
    st.markdown(
        "Detailed change analytics based on the most recent **Multi-Year Comparison** run. "
        "Run a comparison on the *Multi-Year Comparison* page first."
    )

    comparison = get_value("comparison_result")
    aoi = get_value("aoi")

    if comparison is None or aoi is None:
        st.info("No comparison results available. Please run a Multi-Year Comparison first.")
        return

    result_from = comparison["result_from"]
    result_to = comparison["result_to"]
    change = comparison["change"]

    st.subheader(f"{aoi.name}: {result_from.year} → {result_to.year}")

    detector = ChangeDetector()

    # Urban expansion analysis
    st.markdown("### 🏙️ Urban Expansion Analysis")
    urban = detector.urban_expansion_analysis(change)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Previous Urban Area", f"{urban['previous_urban_area_km2']:.2f} km²")
    with col2:
        st.metric("New Urban Area", f"{urban['new_total_urban_area_km2']:.2f} km²")
    with col3:
        st.metric(
            "Urban Growth",
            f"{urban['new_urban_area_km2']:.2f} km²",
            f"{urban['urban_growth_pct']:.2f}%",
        )

    st.markdown("**Growth Hotspots (Source Classes Converted to Urban)**")
    if not urban["growth_sources"].empty:
        st.dataframe(urban["growth_sources"], use_container_width=True, hide_index=True)
    else:
        st.write("No new urban area detected in this period.")

    # Vegetation loss analysis
    st.markdown("### 🌳 Vegetation Loss Analysis")
    veg = detector.vegetation_loss_analysis(change)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Vegetation Loss",
            f"{veg['vegetation_area_loss_km2']:.2f} km²",
            f"{-veg['vegetation_loss_pct']:.2f}%",
        )
    with col2:
        st.metric(
            "Agriculture Loss",
            f"{veg['agriculture_area_loss_km2']:.2f} km²",
            f"{-veg['agriculture_loss_pct']:.2f}%",
        )
    with col3:
        st.metric(
            "Total Green Cover Loss",
            f"{veg['total_green_cover_loss_km2']:.2f} km²",
            f"{-veg['green_cover_loss_pct']:.2f}%",
        )

    st.markdown("**Conversion to Urban Land**")
    st.write(
        f"- Vegetation → Urban: **{veg['vegetation_converted_to_urban_km2']:.2f} km²**\n"
        f"- Agriculture → Urban: **{veg['agriculture_converted_to_urban_km2']:.2f} km²**"
    )

    # Change intensity heatmap
    st.markdown("### 🔥 Change Intensity Heatmap")
    fig = change_heatmap(change.transition_matrix_km2, title=f"Transition Intensity ({result_from.year} → {result_to.year})")
    st.plotly_chart(fig, use_container_width=True)

    # Full from/to table
    st.markdown("### 📋 Full Change Matrix")
    st.dataframe(change.from_to_table, use_container_width=True, hide_index=True)

    # Downloads
    st.markdown("### ⬇️ Download Analytics")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        if not urban["growth_sources"].empty:
            st.download_button(
                "Urban Growth Sources (CSV)",
                data=dataframe_to_csv_bytes(urban["growth_sources"], index=False),
                file_name=f"{aoi.name}_{result_from.year}_{result_to.year}_urban_growth_sources.csv",
                mime="text/csv",
            )
    with col_d2:
        st.download_button(
            "Change Matrix (CSV)",
            data=dataframe_to_csv_bytes(change.from_to_table, index=False),
            file_name=f"{aoi.name}_{result_from.year}_{result_to.year}_change_matrix.csv",
            mime="text/csv",
        )
