"""
pages/trend_analysis.py
Trend Analysis page: analyze land-cover trends across multiple years
(e.g. urban growth, vegetation/water/agriculture trends).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.change_detection import ChangeDetector
from core.pipeline import LandCoverPipeline
from utils.exports import dataframe_to_csv_bytes
from utils.logger import get_logger
from utils.visualization import time_series_chart
from utils.state import get_value, set_value

logger = get_logger(__name__)


def render() -> None:
    """Render the Trend Analysis page."""
    st.title("📈 Multi-Year Trend Analysis")
    st.markdown(
        "Analyze land-cover trends across multiple years for a city — "
        "track urban growth, vegetation, water bodies, and agriculture over time."
    )

    col1, col2 = st.columns([2, 2])
    with col1:
        city_name = st.text_input("City Name", value=get_value("city_name", ""), placeholder="e.g. Mumbai", key="trend_city")
    with col2:
        country = st.text_input("Country (optional)", value=get_value("country", ""), placeholder="e.g. India", key="trend_country")

    st.markdown("#### Select Years")
    default_years = "2005, 2010, 2015, 2020, 2025"
    years_input = st.text_input(
        "Comma-separated years for trend analysis", value=default_years,
        help="Example: 2005, 2010, 2015, 2020, 2025",
    )

    model_type = st.selectbox(
        "Classification Model", options=["random_forest", "xgboost", "lightgbm"], index=0,
    )

    run = st.button("Run Trend Analysis", type="primary")

    if run:
        if not city_name.strip():
            st.error("Please enter a city name.")
            return

        try:
            years = sorted({int(y.strip()) for y in years_input.split(",") if y.strip()})
        except ValueError:
            st.error("Please enter valid comma-separated years (e.g. 2005, 2010, 2015).")
            return

        if len(years) < 2:
            st.error("Please provide at least two years for trend analysis.")
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

        results = {}
        progress = st.progress(0.0, text="Starting trend analysis...")
        detector = ChangeDetector()

        for i, year in enumerate(years):
            progress.progress(i / len(years), text=f"Analyzing {year} ({i + 1}/{len(years)})...")
            try:
                result = pipeline.analyze_year(aoi, year, model_type=model_type)
            except RuntimeError as exc:
                st.warning(f"Skipping {year}: {exc}")
                continue
            except Exception as exc:  # noqa: BLE001
                logger.exception("Unexpected error analyzing year %d", year)
                st.warning(f"Skipping {year}: unexpected error ({exc})")
                continue

            results[year] = {
                "result": result,
                "stats": detector.class_area_statistics(result.classification_map),
            }

        progress.progress(1.0, text="Done.")

        if len(results) < 2:
            st.error("Not enough successful years to build a trend (need at least 2).")
            return

        set_value("trend_results", results)
        set_value("trend_years", sorted(results.keys()))
        st.success(f"Trend analysis complete for {len(results)} year(s).")

    results = get_value("trend_results")
    years = get_value("trend_years")
    aoi = get_value("aoi")

    if not results or not years or aoi is None:
        st.info("Run a trend analysis above to see results.")
        return

    st.divider()
    st.subheader(f"Trend Results: {aoi.name}")

    # Build trend dataframe: rows=years, columns=classes, values=area km2
    records = []
    for year in years:
        stats_df = results[year]["stats"]
        row = {"Year": year}
        for _, r in stats_df.iterrows():
            row[r["Class"]] = r["Area (km²)"]
        records.append(row)
    trend_df = pd.DataFrame(records)

    st.markdown("### 📊 Land Cover Area by Year (km²)")
    st.dataframe(trend_df, use_container_width=True, hide_index=True)

    st.markdown("### 📈 Time-Series Trend")
    fig = time_series_chart(trend_df, title=f"Land Cover Trend — {aoi.name}")
    st.plotly_chart(fig, use_container_width=True)

    # Per-class trend summaries
    st.markdown("### 🔍 Class-Specific Trends")
    classes_of_interest = ["Urban/Built-up", "Vegetation", "Water", "Agriculture"]
    summary_records = []
    first_year, last_year = years[0], years[-1]
    for cls in classes_of_interest:
        if cls not in trend_df.columns:
            continue
        start_val = trend_df.loc[trend_df["Year"] == first_year, cls].values[0]
        end_val = trend_df.loc[trend_df["Year"] == last_year, cls].values[0]
        change_val = end_val - start_val
        change_pct = (change_val / start_val * 100.0) if start_val > 0 else 0.0
        summary_records.append({
            "Class": cls,
            f"Area {first_year} (km²)": round(float(start_val), 3),
            f"Area {last_year} (km²)": round(float(end_val), 3),
            "Net Change (km²)": round(float(change_val), 3),
            "Net Change (%)": round(float(change_pct), 2),
        })
    summary_df = pd.DataFrame(summary_records)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # Highlight urban growth and vegetation/water/agriculture trends
    col1, col2 = st.columns(2)
    with col1:
        urban_row = summary_df[summary_df["Class"] == "Urban/Built-up"]
        if not urban_row.empty:
            st.metric(
                "Urban Growth",
                f"{urban_row.iloc[0]['Net Change (km²)']:.2f} km²",
                f"{urban_row.iloc[0]['Net Change (%)']:.2f}%",
            )
    with col2:
        veg_row = summary_df[summary_df["Class"] == "Vegetation"]
        if not veg_row.empty:
            st.metric(
                "Vegetation Change",
                f"{veg_row.iloc[0]['Net Change (km²)']:.2f} km²",
                f"{veg_row.iloc[0]['Net Change (%)']:.2f}%",
            )

    # Transition statistics between consecutive years
    st.markdown("### 🔄 Land Cover Transition Statistics (Consecutive Years)")
    detector = ChangeDetector()
    for i in range(len(years) - 1):
        y_from, y_to = years[i], years[i + 1]
        change = detector.detect_change(
            results[y_from]["result"].classification_map,
            results[y_to]["result"].classification_map,
        )
        with st.expander(f"{y_from} → {y_to}"):
            st.dataframe(change.from_to_table, use_container_width=True, hide_index=True)

    # Download
    st.markdown("### ⬇️ Download Trend Data")
    st.download_button(
        "Download Trend Data (CSV)",
        data=dataframe_to_csv_bytes(trend_df, index=False),
        file_name=f"{aoi.name}_trend_{years[0]}_{years[-1]}.csv",
        mime="text/csv",
    )
