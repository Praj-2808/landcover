"""
pages/interactive_maps.py
Interactive Maps page: explore classification and change layers for the
most recent single-year or comparison results.
"""

from __future__ import annotations

import streamlit as st
from streamlit_folium import st_folium

from utils.visualization import build_change_map, build_classification_map
from utils.state import get_value


def render() -> None:
    """Render the Interactive Maps page."""
    st.title("🗺️ Interactive Maps")
    st.markdown(
        "Explore land-cover and change layers from your most recent analyses. "
        "Use the layer control (top-right of each map) to toggle the AOI boundary and overlays."
    )

    aoi = get_value("aoi")
    if aoi is None:
        st.info("No analysis available yet. Run a Single-Year or Multi-Year analysis first.")
        return

    single_result = get_value("single_year_result")
    comparison = get_value("comparison_result")

    tabs = st.tabs(["Single-Year Map", "Comparison Maps", "Change Map"])

    with tabs[0]:
        if single_result is None:
            st.info("No single-year analysis available. Run one on the Single-Year Analysis page.")
        else:
            st.markdown(f"**{aoi.name} — {single_result.year}**")
            fmap = build_classification_map(
                aoi, single_result.classification_map, title=f"Land Cover {single_result.year}"
            )
            st_folium(fmap, width=900, height=550, key="interactive_single_map")

    with tabs[1]:
        if comparison is None:
            st.info("No comparison available. Run one on the Multi-Year Comparison page.")
        else:
            result_from = comparison["result_from"]
            result_to = comparison["result_to"]
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**{result_from.year}**")
                fmap_from = build_classification_map(
                    aoi, result_from.classification_map, title=f"Land Cover {result_from.year}"
                )
                st_folium(fmap_from, width=440, height=500, key="interactive_map_from")
            with col2:
                st.markdown(f"**{result_to.year}**")
                fmap_to = build_classification_map(
                    aoi, result_to.classification_map, title=f"Land Cover {result_to.year}"
                )
                st_folium(fmap_to, width=440, height=500, key="interactive_map_to")

    with tabs[2]:
        if comparison is None:
            st.info("No comparison available. Run one on the Multi-Year Comparison page.")
        else:
            change = comparison["change"]
            result_from = comparison["result_from"]
            result_to = comparison["result_to"]
            st.markdown(f"**Change: {result_from.year} → {result_to.year}**")
            change_fmap = build_change_map(aoi, change.change_map, title="Areas of Change")
            st_folium(change_fmap, width=900, height=550, key="interactive_change_map")
