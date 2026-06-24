"""
app.py
Main entry point for the Land Cover Classification and Change Detection
Streamlit application. Provides sidebar navigation across all pages.
"""

from __future__ import annotations

import streamlit as st

from pages import (
    change_detection_page,
    downloads,
    home,
    interactive_maps,
    multi_year,
    single_year,
    trend_analysis,
)
from utils.logger import get_logger
from utils.state import init_session_state

logger = get_logger(__name__)

PAGES = {
    "Home": home,
    "Single-Year Analysis": single_year,
    "Multi-Year Comparison": multi_year,
    "Trend Analysis": trend_analysis,
    "Change Detection": change_detection_page,
    "Interactive Maps": interactive_maps,
    "Downloads": downloads,
}


def main() -> None:
    """Configure and run the Streamlit application."""
    st.set_page_config(
        page_title="Land Cover Classification & Change Detection",
        page_icon="🛰️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    hide_streamlit_style = """
<style>
[data-testid="stSidebarNav"] {
    display: none !important;
}
</style>
"""
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    init_session_state()

    st.sidebar.title("🛰️ Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()), label_visibility="collapsed")

    st.sidebar.divider()
    st.sidebar.markdown(
        "**Data Sources**\n\n"
        "- Sentinel-2 L2A (2016+)\n"
        "- Landsat Collection 2 (pre-2016)\n\n"
        "via Microsoft Planetary Computer"
    )

    try:
        PAGES[selection].render()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unhandled error while rendering page '%s'", selection)
        st.error(f"An unexpected error occurred: {exc}")


if __name__ == "__main__":
    main()
