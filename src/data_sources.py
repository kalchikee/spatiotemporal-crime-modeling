"""
data_sources.py
===============
Real Open Data connectors for crime incident records.

Supported sources
-----------------
NYC Open Data   — NYPD Complaint Data (Socrata API, no token needed)
Chicago         — Chicago Data Portal (Socrata API)
Local CSV/GPKG  — Pass your own file path

All loaders return a GeoDataFrame with a consistent schema:
    geometry    : Point (WGS84, EPSG:4326)
    datetime    : pd.Timestamp
    crime_type  : str
    x, y        : projected coordinates (EPSG:32616 for NYC / EPSG:32616 for Chicago)

Usage
-----
    from src.data_sources import load_nyc_crimes, load_chicago_crimes

    gdf = load_nyc_crimes(limit=20_000, start_date="2023-01-01")
    gdf = load_chicago_crimes(limit=20_000, start_date="2023-01-01")
"""

import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _to_gdf(df: pd.DataFrame, lon_col: str, lat_col: str, crs_out: str = "EPSG:32616") -> gpd.GeoDataFrame:
    """Drop NaN coordinates and build a projected GeoDataFrame."""
    df = df.dropna(subset=[lon_col, lat_col]).copy()
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df = df.dropna(subset=[lon_col, lat_col])

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326",
    ).to_crs(crs_out)

    gdf["x"] = gdf.geometry.x
    gdf["y"] = gdf.geometry.y
    return gdf


def _socrata_get(base_url: str, resource_id: str, params: dict) -> pd.DataFrame:
    """Fetch records from a Socrata endpoint (no auth token required)."""
    import requests

    url = f"{base_url}/resource/{resource_id}.json"
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    if not data:
        raise ValueError(f"No data returned from {url}. Check filters.")
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# NYC Open Data — NYPD Complaint Data
# ---------------------------------------------------------------------------

NYC_BASE_URL   = "https://data.cityofnewyork.us"
NYC_RESOURCE   = "qgea-i56i"   # NYPD Complaint Data Historic

NYC_OFFENSE_MAP = {
    "FELONY ASSAULT":       "assault",
    "ROBBERY":              "robbery",
    "BURGLARY":             "burglary",
    "PETIT LARCENY":        "theft",
    "GRAND LARCENY":        "theft",
    "THEFT OF SERVICES":    "theft",
    "CRIMINAL MISCHIEF":    "vandalism",
    "HARRASSMENT 2":        "harassment",
}


def load_nyc_crimes(
    limit: int = 20_000,
    start_date: str = "2023-01-01",
    end_date: str | None = None,
    crime_types: list[str] | None = None,
) -> gpd.GeoDataFrame:
    """
    Download NYPD crime complaint records from NYC Open Data.

    Parameters
    ----------
    limit       : max records to download
    start_date  : earliest complaint date (YYYY-MM-DD)
    end_date    : latest complaint date (YYYY-MM-DD). Default: today.
    crime_types : filter to specific types ('theft','assault','burglary','robbery')
                  If None, all types are returned.

    Returns
    -------
    GeoDataFrame (EPSG:32618 — UTM Zone 18N for NYC)
    """
    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

    where_clause = (
        f"cmplnt_fr_dt >= '{start_date}T00:00:00.000' "
        f"AND cmplnt_fr_dt <= '{end_date}T23:59:59.999' "
        f"AND latitude IS NOT NULL"
    )

    params = {
        "$limit":  limit,
        "$where":  where_clause,
        "$select": "cmplnt_fr_dt,ofns_desc,latitude,longitude,boro_nm,law_cat_cd",
        "$order":  "cmplnt_fr_dt DESC",
    }

    print(f"[NYC] Fetching up to {limit:,} records from NYC Open Data …")
    try:
        df = _socrata_get(NYC_BASE_URL, NYC_RESOURCE, params)
    except Exception as e:
        warnings.warn(f"[NYC] Download failed: {e}. Returning empty GDF.")
        return gpd.GeoDataFrame()

    df = df.rename(columns={
        "cmplnt_fr_dt": "datetime",
        "ofns_desc":    "offense",
        "boro_nm":      "borough",
        "law_cat_cd":   "severity",
    })
    df["datetime"]    = pd.to_datetime(df["datetime"], errors="coerce")
    df["crime_type"]  = df["offense"].map(NYC_OFFENSE_MAP).fillna("other")
    df["hotspot_id"]  = -1

    gdf = _to_gdf(df, "longitude", "latitude", crs_out="EPSG:32618")

    if crime_types:
        gdf = gdf[gdf["crime_type"].isin(crime_types)]

    gdf = gdf.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    print(f"[NYC] Loaded {len(gdf):,} records ({gdf['datetime'].min().date()} → {gdf['datetime'].max().date()})")
    return gdf[["geometry", "datetime", "crime_type", "hotspot_id", "x", "y"]]


# ---------------------------------------------------------------------------
# Chicago Open Data
# ---------------------------------------------------------------------------

CHICAGO_BASE_URL = "https://data.cityofchicago.org"
CHICAGO_RESOURCE = "ijzp-q8t2"   # Crimes — 2001 to present

CHICAGO_OFFENSE_MAP = {
    "THEFT":          "theft",
    "BATTERY":        "assault",
    "ASSAULT":        "assault",
    "BURGLARY":       "burglary",
    "ROBBERY":        "robbery",
    "MOTOR VEHICLE THEFT": "theft",
    "CRIMINAL DAMAGE": "vandalism",
}


def load_chicago_crimes(
    limit: int = 20_000,
    start_date: str = "2023-01-01",
    end_date: str | None = None,
) -> gpd.GeoDataFrame:
    """
    Download crime records from the Chicago Data Portal.

    Returns
    -------
    GeoDataFrame (EPSG:32616 — UTM Zone 16N for Chicago)
    """
    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

    where_clause = (
        f"date >= '{start_date}T00:00:00.000' "
        f"AND date <= '{end_date}T23:59:59.999' "
        f"AND latitude IS NOT NULL"
    )

    params = {
        "$limit":  limit,
        "$where":  where_clause,
        "$select": "date,primary_type,latitude,longitude,community_area",
        "$order":  "date DESC",
    }

    print(f"[Chicago] Fetching up to {limit:,} records from Chicago Data Portal …")
    try:
        df = _socrata_get(CHICAGO_BASE_URL, CHICAGO_RESOURCE, params)
    except Exception as e:
        warnings.warn(f"[Chicago] Download failed: {e}. Returning empty GDF.")
        return gpd.GeoDataFrame()

    df = df.rename(columns={
        "date":         "datetime",
        "primary_type": "offense",
    })
    df["datetime"]   = pd.to_datetime(df["datetime"], errors="coerce")
    df["crime_type"] = df["offense"].map(CHICAGO_OFFENSE_MAP).fillna("other")
    df["hotspot_id"] = -1

    gdf = _to_gdf(df, "longitude", "latitude", crs_out="EPSG:32616")
    gdf = gdf.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    print(f"[Chicago] Loaded {len(gdf):,} records")
    return gdf[["geometry", "datetime", "crime_type", "hotspot_id", "x", "y"]]


# ---------------------------------------------------------------------------
# Local file loader
# ---------------------------------------------------------------------------

def load_from_file(
    filepath: str,
    datetime_col: str = "datetime",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    crime_type_col: str = "crime_type",
    crs_out: str = "EPSG:32616",
) -> gpd.GeoDataFrame:
    """
    Load crime data from a local CSV or GeoPackage.

    For GeoPackage (.gpkg): read directly with geopandas.
    For CSV: expects lat/lon columns in WGS84.
    """
    if filepath.lower().endswith(".gpkg"):
        gdf = gpd.read_file(filepath)
        if gdf.crs != crs_out:
            gdf = gdf.to_crs(crs_out)
    else:
        df = pd.read_csv(filepath)
        df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
        gdf = _to_gdf(df, lon_col, lat_col, crs_out=crs_out)

    gdf = gdf.rename(columns={datetime_col: "datetime", crime_type_col: "crime_type"})
    if "hotspot_id" not in gdf.columns:
        gdf["hotspot_id"] = -1
    if "x" not in gdf.columns:
        gdf["x"] = gdf.geometry.x
        gdf["y"] = gdf.geometry.y

    gdf = gdf.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    print(f"[File] Loaded {len(gdf):,} records from {filepath}")
    return gdf
