"""
utils/state.py
Centralized Streamlit session-state initialization and accessors.
"""

from __future__ import annotations

from typing import Any, Optional

import streamlit as st

from config import DEFAULT_MODEL_TYPE


def init_session_state() -> None:
    """Initialize all required keys in st.session_state if not already present."""
    defaults = {
        "pipeline_model_type": DEFAULT_MODEL_TYPE,
        "aoi": None,
        "city_name": "",
        "country": "",
        "single_year_result": None,
        "comparison_result": None,
        "trend_results": None,
        "trend_years": [],
        "last_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def set_value(key: str, value: Any) -> None:
    """Set a value in session state."""
    st.session_state[key] = value


def get_value(key: str, default: Optional[Any] = None) -> Any:
    """Get a value from session state, with a default fallback."""
    return st.session_state.get(key, default)
