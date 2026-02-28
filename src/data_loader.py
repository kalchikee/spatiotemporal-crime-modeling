"""
data_loader.py
==============
Synthetic crime data generation with realistic spatiotemporal patterns.

Generates a GeoDataFrame of crime events with:
- 5 spatial hotspot clusters that evolve over time (emerging, persistent, diminishing)
- Seasonal variation (summer peak)
- Time-of-day variation (night peak)
- Multiple crime types
- Background random noise
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


class SyntheticCrimeGenerator:
    """
    Generate synthetic crime events with evolving spatiotemporal hotspots.

    Parameters
    ----------
    n_events : int
        Approximate total number of crime events over the full period.
    start_date : str
        First date of crime records (YYYY-MM-DD).
    end_date : str
        Last date of crime records (YYYY-MM-DD).
    bbox : tuple
        (minx, miny, maxx, maxy) bounding box in projected meters.
    crs : str
        Coordinate reference system (projected, meters).
    seed : int
        Random seed for reproducibility.
    """

    # Hotspot definitions: center (m), sigma (m), base intensity, monthly trend, start month
    HOTSPOT_CONFIGS = [
        # Persistent high-crime area — stable
        dict(center=(5000, 5000),   sigma=1500, intensity=0.28, trend=0.000,  start=0,  label="Persistent NW"),
        # Large southern hotspot — gradually diminishing
        dict(center=(15000, 15000), sigma=1200, intensity=0.24, trend=-0.008, start=0,  label="Diminishing SE"),
        # Central hotspot — emerging partway through
        dict(center=(10000, 8000),  sigma=1000, intensity=0.20, trend=0.018,  start=6,  label="Emerging Central"),
        # Small stable cluster in NE
        dict(center=(4000, 16000),  sigma=800,  intensity=0.12, trend=0.000,  start=0,  label="Persistent NE"),
        # SW cluster that fades out
        dict(center=(17000, 5000),  sigma=1000, intensity=0.10, trend=-0.022, start=0,  label="Diminishing SW"),
    ]

    CRIME_TYPES  = ["theft", "assault", "burglary", "robbery"]
    CRIME_WEIGHTS = [0.40,    0.25,      0.20,        0.15]

    def __init__(
        self,
        n_events: int = 12_000,
        start_date: str = "2022-01-01",
        end_date: str = "2023-12-31",
        bbox: tuple = (0, 0, 20_000, 20_000),
        crs: str = "EPSG:32616",
        seed: int = 42,
    ):
        self.n_events = n_events
        self.start = pd.Timestamp(start_date)
        self.end = pd.Timestamp(end_date)
        self.bbox = bbox
        self.crs = crs
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self) -> gpd.GeoDataFrame:
        """Return a GeoDataFrame of synthetic crime events."""
        months = self._month_periods()
        n_months = len(months)
        base_per_month = self.n_events / n_months

        records = []
        for m_idx, (m_start, m_end) in enumerate(months):
            seasonal = self._seasonal_factor(m_start.month)
            n_total = int(base_per_month * seasonal)

            # Hotspot events
            for hs in self.HOTSPOT_CONFIGS:
                if m_idx < hs["start"]:
                    continue
                eff = hs["intensity"] * (1 + hs["trend"] * (m_idx - hs["start"]))
                eff = float(np.clip(eff, 0, 1))
                n_hs = int(n_total * eff)
                records.extend(self._sample_hotspot(hs, n_hs, m_start, m_end, m_idx))

            # Background noise (30% of total)
            n_noise = max(0, int(n_total * 0.30))
            records.extend(self._sample_noise(n_noise, m_start, m_end))

        df = pd.DataFrame(records)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)

        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["x"], df["y"]),
            crs=self.crs,
        )
        return gdf

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _month_periods(self):
        """List of (period_start, period_end) for every calendar month."""
        periods = []
        current = self.start.to_period("M")
        end_period = self.end.to_period("M")
        while current <= end_period:
            periods.append((current.to_timestamp(), current.to_timestamp("M")))
            current += 1
        return periods

    @staticmethod
    def _seasonal_factor(month: int) -> float:
        """Crime peaks in summer (month 7-8), dips in winter."""
        return 1.0 + 0.35 * np.sin(np.pi * (month - 1) / 6.0 - np.pi / 2)

    def _time_of_day_probs(self) -> np.ndarray:
        """Probability for each hour; peaks 20-02, low 06-09."""
        w = np.ones(24)
        w[20:24] = 3.0
        w[0:3]   = 2.5
        w[6:9]   = 0.5
        return w / w.sum()

    def _sample_hotspot(self, hs, n, m_start, m_end, m_idx):
        if n <= 0:
            return []
        x = self.rng.normal(hs["center"][0], hs["sigma"], n)
        y = self.rng.normal(hs["center"][1], hs["sigma"], n)
        x = np.clip(x, self.bbox[0], self.bbox[2])
        y = np.clip(y, self.bbox[1], self.bbox[3])
        return self._make_records(x, y, n, m_start, m_end, m_idx)

    def _sample_noise(self, n, m_start, m_end):
        if n <= 0:
            return []
        x = self.rng.uniform(self.bbox[0], self.bbox[2], n)
        y = self.rng.uniform(self.bbox[1], self.bbox[3], n)
        return self._make_records(x, y, n, m_start, m_end, hotspot_id=-1)

    def _make_records(self, x, y, n, m_start, m_end, hotspot_id=None):
        days_span = max(1, (m_end - m_start).days)
        day_offsets = self.rng.integers(0, days_span, n)
        hours = self.rng.choice(24, n, p=self._time_of_day_probs())
        crime_types = self.rng.choice(self.CRIME_TYPES, n, p=self.CRIME_WEIGHTS)
        records = []
        for i in range(n):
            dt = m_start + pd.Timedelta(days=int(day_offsets[i]), hours=int(hours[i]))
            records.append(
                dict(x=x[i], y=y[i], datetime=dt, crime_type=crime_types[i], hotspot_id=hotspot_id)
            )
        return records


def load_or_generate(
    filepath: str | None = None,
    **kwargs,
) -> gpd.GeoDataFrame:
    """
    Load crime data from a GeoPackage/CSV, or generate synthetic data.

    If `filepath` is None or the file doesn't exist, synthetic data is generated
    and optionally saved to `filepath`.

    Minimum required columns for real data:
        geometry (Point), datetime (parseable), crime_type (str)
    """
    if filepath is not None:
        try:
            if filepath.endswith(".gpkg"):
                gdf = gpd.read_file(filepath)
            else:
                df = pd.read_csv(filepath, parse_dates=["datetime"])
                gdf = gpd.GeoDataFrame(
                    df,
                    geometry=gpd.points_from_xy(df["x"], df["y"]),
                    crs=kwargs.get("crs", "EPSG:32616"),
                )
            print(f"[data_loader] Loaded {len(gdf):,} records from {filepath}")
            return gdf
        except FileNotFoundError:
            print(f"[data_loader] {filepath} not found — generating synthetic data.")

    gen = SyntheticCrimeGenerator(**kwargs)
    gdf = gen.generate()
    print(f"[data_loader] Generated {len(gdf):,} synthetic crime events.")

    if filepath is not None:
        if filepath.endswith(".gpkg"):
            gdf.to_file(filepath, driver="GPKG")
        else:
            gdf.drop(columns="geometry").to_csv(filepath, index=False)
        print(f"[data_loader] Saved to {filepath}")

    return gdf
