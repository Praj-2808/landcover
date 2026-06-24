"""
utils/visualization.py
Visualization helpers: Folium maps for classification/change layers, and
Plotly charts (pie, bar, sankey, time-series, heatmap).
"""

from __future__ import annotations

import base64
import io
from typing import Dict, List, Optional

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

from config import LAND_COVER_CLASSES, LAND_COVER_COLORS
from core.aoi import AreaOfInterest
from utils.logger import get_logger

logger = get_logger(__name__)


def _class_map_to_rgba(classification_map: np.ndarray, alpha: int = 160) -> Image.Image:
    """Convert a class-code 2D array into an RGBA PIL Image using LAND_COVER_COLORS."""
    h, w = classification_map.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for code, hex_color in LAND_COVER_COLORS.items():
        mask = classification_map == code
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        rgba[mask] = (r, g, b, alpha)
    return Image.fromarray(rgba, mode="RGBA")


def _image_to_data_url(img: Image.Image) -> str:
    """Encode a PIL image as a base64 PNG data URL."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def build_classification_map(
    aoi: AreaOfInterest, classification_map: np.ndarray, title: str = "Land Cover",
) -> folium.Map:
    """
    Build a Folium map showing the AOI boundary and a classification raster
    overlay with a legend.

    Args:
        aoi: The AreaOfInterest (provides bbox and center coordinates).
        classification_map: 2D int array of class codes.
        title: Title used for the overlay layer name.

    Returns:
        A folium.Map instance ready for display (e.g. via streamlit-folium).
    """
    m = folium.Map(
        location=[aoi.latitude, aoi.longitude],
        zoom_start=11,
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
    )

    min_lon, min_lat, max_lon, max_lat = aoi.bbox

    img = _class_map_to_rgba(classification_map)
    data_url = _image_to_data_url(img)

    folium.raster_layers.ImageOverlay(
        image=data_url,
        bounds=[[min_lat, min_lon], [max_lat, max_lon]],
        opacity=0.75,
        name=title,
    ).add_to(m)

    folium.GeoJson(
        aoi.geojson_bbox,
        name="AOI Boundary",
        style_function=lambda x: {
            "fillOpacity": 0, "color": "yellow", "weight": 2,
        },
    ).add_to(m)

    folium.LayerControl().add_to(m)
    _add_legend(m)
    return m


def build_change_map(
    aoi: AreaOfInterest, change_map: np.ndarray, title: str = "Change Map",
) -> folium.Map:
    """
    Build a Folium map highlighting pixels that changed class between two years.

    Args:
        aoi: The AreaOfInterest.
        change_map: 2D binary array (1 = changed, 0 = unchanged).
        title: Overlay layer name.

    Returns:
        A folium.Map instance.
    """
    m = folium.Map(
        location=[aoi.latitude, aoi.longitude],
        zoom_start=11,
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
    )

    min_lon, min_lat, max_lon, max_lat = aoi.bbox

    h, w = change_map.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[change_map == 1] = (255, 0, 0, 180)  # red = changed
    img = Image.fromarray(rgba, mode="RGBA")
    data_url = _image_to_data_url(img)

    folium.raster_layers.ImageOverlay(
        image=data_url,
        bounds=[[min_lat, min_lon], [max_lat, max_lon]],
        opacity=0.75,
        name=title,
    ).add_to(m)

    folium.GeoJson(
        aoi.geojson_bbox,
        name="AOI Boundary",
        style_function=lambda x: {
            "fillOpacity": 0, "color": "yellow", "weight": 2,
        },
    ).add_to(m)

    folium.LayerControl().add_to(m)
    return m


def _add_legend(m: folium.Map) -> None:
    """Add an HTML legend for land-cover classes to a Folium map."""
    items = "".join(
        f'<div style="display:flex;align-items:center;margin-bottom:4px;">'
        f'<span style="display:inline-block;width:14px;height:14px;'
        f'background:{LAND_COVER_COLORS[code]};margin-right:6px;'
        f'border:1px solid #060101; "></span>{name}</div>'
        for code, name in LAND_COVER_CLASSES.items()
    )
    legend_html = f"""
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999;
                background: white; color:black; padding: 10px 14px; border-radius: 6px;
                box-shadow: 0 0 6px rgba(0,0,0,0.4); font-size: 13px;">
        <b>Land Cover Classes</b><br>{items}
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))


# ---------------------------------------------------------------------------
# Plotly chart builders
# ---------------------------------------------------------------------------
def pie_chart_land_cover(stats_df: pd.DataFrame, title: str = "Land Cover Distribution") -> go.Figure:
    """Build a pie chart of land-cover class shares from a class-statistics DataFrame."""
    colors = [LAND_COVER_COLORS[code] for code in sorted(LAND_COVER_CLASSES.keys())]
    fig = px.pie(
        stats_df, names="Class", values="Percentage (%)", title=title,
        color="Class",
        color_discrete_map={LAND_COVER_CLASSES[k]: v for k, v in LAND_COVER_COLORS.items()},
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return fig


def bar_chart_area_comparison(
    area_from: pd.Series, area_to: pd.Series, year_from: int, year_to: int,
    title: str = "Area Comparison",
) -> go.Figure:
    """Build a grouped bar chart comparing class areas between two years."""
    df = pd.DataFrame({
        "Class": area_from.index,
        str(year_from): area_from.values,
        str(year_to): area_to.values,
    })
    df_melt = df.melt(id_vars="Class", var_name="Year", value_name="Area (km²)")
    fig = px.bar(
        df_melt, x="Class", y="Area (km²)", color="Year", barmode="group", title=title,
    )
    return fig


def sankey_transitions(from_to_table: pd.DataFrame, year_from: int, year_to: int) -> go.Figure:
    """
    Build a Sankey diagram of land-cover transitions between two years.

    Args:
        from_to_table: Long-format DataFrame with 'From Class', 'To Class',
            and 'Area Changed (km²)' columns.
        year_from: Earlier year (used for node labeling).
        year_to: Later year (used for node labeling).

    Returns:
        A Plotly Sankey figure.
    """
    if from_to_table.empty:
        fig = go.Figure()
        fig.update_layout(title="No land-cover transitions detected")
        return fig

    from_labels = [f"{c} ({year_from})" for c in from_to_table["From Class"].unique()]
    to_labels = [f"{c} ({year_to})" for c in from_to_table["To Class"].unique()]
    all_labels = list(dict.fromkeys(from_labels + to_labels))
    label_index = {label: i for i, label in enumerate(all_labels)}

    sources = [label_index[f"{r['From Class']} ({year_from})"] for _, r in from_to_table.iterrows()]
    targets = [label_index[f"{r['To Class']} ({year_to})"] for _, r in from_to_table.iterrows()]
    values = from_to_table["Area Changed (km²)"].tolist()

    node_colors = []
    for label in all_labels:
        class_name = label.rsplit(" (", 1)[0]
        code = next((k for k, v in LAND_COVER_CLASSES.items() if v == class_name), None)
        node_colors.append(LAND_COVER_COLORS.get(code, "#888888"))

    fig = go.Figure(go.Sankey(
        node=dict(label=all_labels, color=node_colors, pad=15, thickness=18),
        link=dict(source=sources, target=targets, value=values),
    ))
    fig.update_layout(title_text=f"Land Cover Transitions: {year_from} → {year_to}", font_size=12)
    return fig


def time_series_chart(
    trend_df: pd.DataFrame, title: str = "Land Cover Trend Over Time",
) -> go.Figure:
    """
    Build a multi-line time-series chart of class areas across years.

    Args:
        trend_df: DataFrame indexed or columned by 'Year', with one column
            per land-cover class containing area values (km²).

    Returns:
        A Plotly line chart figure.
    """
    df_melt = trend_df.melt(id_vars="Year", var_name="Class", value_name="Area (km²)")
    color_map = {LAND_COVER_CLASSES[k]: v for k, v in LAND_COVER_COLORS.items()}
    fig = px.line(
        df_melt, x="Year", y="Area (km²)", color="Class", markers=True, title=title,
        color_discrete_map=color_map,
    )
    return fig


def change_heatmap(transition_matrix: pd.DataFrame, title: str = "Change Intensity Heatmap") -> go.Figure:
    """Build a heatmap of the transition matrix (from-class rows, to-class columns)."""
    fig = px.imshow(
        transition_matrix.values,
        x=transition_matrix.columns,
        y=transition_matrix.index,
        text_auto=".2f",
        color_continuous_scale="YlOrRd",
        labels=dict(x="To Class", y="From Class", color="Area (km²)"),
        title=title,
    )
    return fig
