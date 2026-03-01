"""
web_utils.py
============
Helper functions for the Streamlit web map application.

Handles:
- Coordinate conversion from local UTM to WGS84 (lat/lon)
- GeoJSON generation for Plotly Choroplethmapbox
- Color scale construction for categorical and continuous maps
"""

import numpy as np
import geopandas as gpd
from shapely.affinity import translate as shp_translate
from shapely.geometry import mapping


# Chicago UTM Zone 16N base — shifts local (0,0)–(20000,20000) onto the map.
# Box spans easting 430k–450k (≈ -87.80° to -87.60°), entirely west of lakefront.
CHICAGO_OFFSET = (430_000, 4_623_000)   # (easting, northing)
MAP_CENTER     = {"lat": 41.85, "lon": -87.73}
MAP_ZOOM_FULL  = 11
MAP_ZOOM_CELL  = 13


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def localize_gdf(gdf: gpd.GeoDataFrame, offset: tuple = CHICAGO_OFFSET) -> gpd.GeoDataFrame:
    """
    Shift a local-coordinate GDF into valid Chicago UTM coords, then reproject
    to WGS84 (EPSG:4326) for web-map use.

    Parameters
    ----------
    gdf    : GeoDataFrame with local coordinates (0–20000 range, EPSG:32616)
    offset : (dx, dy) to add to all coordinates
    """
    out = gdf.copy()
    out.geometry = out.geometry.apply(
        lambda g: shp_translate(g, xoff=offset[0], yoff=offset[1])
    )
    out = out.set_crs("EPSG:32616", allow_override=True)
    return out.to_crs("EPSG:4326")


def local_xy_to_wgs84(x: float, y: float, offset: tuple = CHICAGO_OFFSET) -> tuple[float, float]:
    """Convert a single (x, y) local point to (lon, lat)."""
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:32616", "EPSG:4326", always_xy=True)
    lon, lat = tr.transform(x + offset[0], y + offset[1])
    return float(lon), float(lat)


# ---------------------------------------------------------------------------
# GeoJSON helpers
# ---------------------------------------------------------------------------

def cells_to_geojson(cell_gdf_wgs84: gpd.GeoDataFrame) -> dict:
    """
    Convert a grid-cell GeoDataFrame (WGS84) to a GeoJSON FeatureCollection
    suitable for Plotly Choroplethmapbox.

    Each feature's 'id' is the string form of 'flat_idx'.
    """
    features = []
    for _, row in cell_gdf_wgs84.iterrows():
        features.append(
            {
                "type": "Feature",
                "id": str(int(row["flat_idx"])),
                "properties": {"flat_idx": int(row["flat_idx"])},
                "geometry": mapping(row.geometry),
            }
        )
    return {"type": "FeatureCollection", "features": features}


def grid_values_to_series(
    grid_2d: np.ndarray, stc
) -> tuple[list, list]:
    """
    Flatten a (nrows, ncols) grid into parallel (flat_idx_list, values_list)
    for Choroplethmapbox.
    """
    flat = grid_2d.ravel()
    idxs = [str(i) for i in range(len(flat))]
    return idxs, flat.tolist()


# ---------------------------------------------------------------------------
# Colour scales
# ---------------------------------------------------------------------------

CATEGORY_COLORS = {
    "NEW_HOTSPOT":           "#FF0000",
    "CONSECUTIVE_HOTSPOT":   "#FF6600",
    "INTENSIFYING_HOTSPOT":  "#FFAA00",
    "PERSISTENT_HOTSPOT":    "#FFD700",
    "DIMINISHING_HOTSPOT":   "#AAAAFF",
    "SPORADIC_HOTSPOT":      "#FFCCCC",
    "OSCILLATING_HOTSPOT":   "#FF00FF",
    "HISTORICAL_HOTSPOT":    "#BBBBBB",
    "NEW_COLDSPOT":          "#0000FF",
    "CONSECUTIVE_COLDSPOT":  "#0066FF",
    "INTENSIFYING_COLDSPOT": "#00AAFF",
    "PERSISTENT_COLDSPOT":   "#00CCFF",
    "DIMINISHING_COLDSPOT":  "#CCCCFF",
    "SPORADIC_COLDSPOT":     "#CCDDFF",
    "OSCILLATING_COLDSPOT":  "#AA00FF",
    "HISTORICAL_COLDSPOT":   "#DDDDFF",
    "NO_PATTERN":            "#F5F5F5",
}


def category_colorscale(categories: list[str]) -> list[list]:
    """
    Build a Plotly discrete colorscale from category labels mapped to integers.
    Returns (colorscale_list, encoded_values, tickvals, ticktext).
    """
    unique_cats = sorted(set(categories))
    cat_to_int  = {c: i for i, c in enumerate(unique_cats)}
    n = len(unique_cats)

    colorscale = []
    for i, cat in enumerate(unique_cats):
        lo = i / n
        hi = (i + 1) / n
        col = CATEGORY_COLORS.get(cat, "#CCCCCC")
        colorscale.append([lo, col])
        colorscale.append([hi, col])

    encoded = [float(cat_to_int[c]) for c in categories]
    tickvals = [float(i) + 0.5 for i in range(n)]
    ticktext = [c.replace("_", " ").title() for c in unique_cats]

    return colorscale, encoded, tickvals, ticktext


# ---------------------------------------------------------------------------
# Common Plotly layout
# ---------------------------------------------------------------------------

def base_mapbox_layout(
    center: dict = MAP_CENTER,
    zoom: float = MAP_ZOOM_FULL,
    height: int = 600,
) -> dict:
    return dict(
        mapbox=dict(
            style="carto-positron",
            center=center,
            zoom=zoom,
        ),
        margin=dict(r=0, t=0, l=0, b=0),
        height=height,
        legend=dict(
            yanchor="top", y=0.99, xanchor="left", x=0.01,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#cccccc",
            borderwidth=1,
        ),
    )
