"""
space_time_cube.py
==================
Space-Time Cube (STC) — the core 3D data structure.

The study area is divided into a regular grid of square cells.
Crime events are binned into (row, col, time_bin) voxels, producing
a 3D count array:  cube[t, row, col] = n_crimes

Key attributes after calling .fit():
    cube        : np.ndarray, shape (T, nrows, ncols)  — raw counts
    cell_gdf    : GeoDataFrame of grid-cell polygons with row/col index
    time_labels : list[pd.Period] — one period label per time slice
    T, nrows, ncols : int dimensions
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box as shapely_box


class SpaceTimeCube:
    """
    Regular-grid, calendar-month space-time cube.

    Parameters
    ----------
    cell_size : float
        Side length of each square grid cell (same units as GDF coordinates).
    bbox : tuple | None
        (minx, miny, maxx, maxy).  If None, inferred from the data.
    freq : str
        Pandas period frequency for time bins.  Default ``"M"`` (monthly).
    """

    def __init__(
        self,
        cell_size: float = 500.0,
        bbox: tuple | None = None,
        freq: str = "M",
    ):
        self.cell_size = cell_size
        self.bbox = bbox
        self.freq = freq

        # Set after fit()
        self.cube: np.ndarray | None = None
        self.cell_gdf: gpd.GeoDataFrame | None = None
        self.time_labels: list | None = None
        self.T = self.nrows = self.ncols = 0
        self._minx = self._miny = 0.0

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, gdf: gpd.GeoDataFrame, datetime_col: str = "datetime") -> "SpaceTimeCube":
        """
        Bin crime events into the space-time cube.

        Parameters
        ----------
        gdf : GeoDataFrame
            Point crime events with a datetime column.
        datetime_col : str
            Name of the timestamp column.
        """
        gdf = gdf.copy()
        gdf[datetime_col] = pd.to_datetime(gdf[datetime_col])

        # Bounding box
        if self.bbox is None:
            self._minx, self._miny, maxx, maxy = gdf.total_bounds
        else:
            self._minx, self._miny, maxx, maxy = self.bbox

        cs = self.cell_size
        self.ncols = int(np.ceil((maxx - self._minx) / cs))
        self.nrows = int(np.ceil((maxy - self._miny) / cs))

        # Assign row/col to each event
        xs = gdf.geometry.x.values
        ys = gdf.geometry.y.values
        cols = np.floor((xs - self._minx) / cs).astype(int).clip(0, self.ncols - 1)
        rows = np.floor((ys - self._miny) / cs).astype(int).clip(0, self.nrows - 1)
        gdf["_col"] = cols
        gdf["_row"] = rows

        # Time bins
        periods = gdf[datetime_col].dt.to_period(self.freq)
        gdf["_period"] = periods
        all_periods = pd.period_range(periods.min(), periods.max(), freq=self.freq)
        self.time_labels = list(all_periods)
        self.T = len(all_periods)
        period_to_idx = {p: i for i, p in enumerate(all_periods)}
        gdf["_t"] = gdf["_period"].map(period_to_idx)

        # Build 3D cube
        self.cube = np.zeros((self.T, self.nrows, self.ncols), dtype=np.float32)
        grp = gdf.groupby(["_t", "_row", "_col"]).size()
        for (t, r, c), cnt in grp.items():
            self.cube[t, r, c] = cnt

        # Build grid cell GeoDataFrame
        self.cell_gdf = self._build_cell_gdf(gdf.crs)

        print(
            f"[SpaceTimeCube] Built {self.T} × {self.nrows} × {self.ncols} cube "
            f"({self.T * self.nrows * self.ncols:,} voxels, "
            f"cell_size={cs:.0f}m, total events={int(self.cube.sum()):,})"
        )
        return self

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_time_slice(self, t: int) -> np.ndarray:
        """Return the 2-D crime-count grid for time step *t*."""
        return self.cube[t]

    def get_cell_timeseries(self, row: int, col: int) -> np.ndarray:
        """Return the 1-D monthly count time series for cell (row, col)."""
        return self.cube[:, row, col]

    def flat_index(self, row: int, col: int) -> int:
        """Row-major flat index into a time-slice vector."""
        return row * self.ncols + col

    def flat_to_rowcol(self, flat: int) -> tuple[int, int]:
        """Convert flat index back to (row, col)."""
        return divmod(flat, self.ncols)

    def n_cells(self) -> int:
        return self.nrows * self.ncols

    # ------------------------------------------------------------------
    # Derived statistics
    # ------------------------------------------------------------------

    def total_counts(self) -> np.ndarray:
        """Sum over time — (nrows, ncols) array."""
        return self.cube.sum(axis=0)

    def mean_counts(self) -> np.ndarray:
        """Mean over time — (nrows, ncols) array."""
        return self.cube.mean(axis=0)

    def active_cells(self, min_total: int = 3) -> list[tuple[int, int]]:
        """Cells with at least *min_total* events across all time steps."""
        totals = self.total_counts()
        rows, cols = np.where(totals >= min_total)
        return list(zip(rows.tolist(), cols.tolist()))

    def flat_slice(self, t: int) -> np.ndarray:
        """Return time slice t as a 1-D vector (row-major)."""
        return self.cube[t].ravel()

    def panel_df(self) -> pd.DataFrame:
        """
        Return a long-format DataFrame with columns:
            t, period, row, col, flat_idx, count
        Useful for regression models.
        """
        records = []
        for t, period in enumerate(self.time_labels):
            for r in range(self.nrows):
                for c in range(self.ncols):
                    records.append(
                        dict(
                            t=t,
                            period=period,
                            row=r,
                            col=c,
                            flat_idx=self.flat_index(r, c),
                            count=float(self.cube[t, r, c]),
                        )
                    )
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Grid geometry
    # ------------------------------------------------------------------

    def _build_cell_gdf(self, crs) -> gpd.GeoDataFrame:
        cs = self.cell_size
        records = []
        for r in range(self.nrows):
            for c in range(self.ncols):
                minx = self._minx + c * cs
                miny = self._miny + r * cs
                geom = shapely_box(minx, miny, minx + cs, miny + cs)
                records.append(dict(row=r, col=c, flat_idx=self.flat_index(r, c), geometry=geom))
        return gpd.GeoDataFrame(records, crs=crs)

    def cell_center(self, row: int, col: int) -> tuple[float, float]:
        cs = self.cell_size
        cx = self._minx + (col + 0.5) * cs
        cy = self._miny + (row + 0.5) * cs
        return cx, cy

    def cell_centers_array(self) -> np.ndarray:
        """(nrows*ncols, 2) array of (x, y) cell centers, row-major."""
        centers = []
        for r in range(self.nrows):
            for c in range(self.ncols):
                centers.append(self.cell_center(r, c))
        return np.array(centers)
