"""
hotspot_analysis.py
===================
Getis-Ord Gi* statistic over time + Emerging Hotspot Analysis.

Pipeline
--------
1. For each time step t: compute local Gi* Z-score for every grid cell.
2. Stack Z-scores into a (T, nrows, ncols) array.
3. For each cell, apply Mann-Kendall trend test to the Z-score sequence.
4. Classify each cell into one of 9 ESRI-style emerging-hotspot categories.

Hotspot categories
------------------
NEW            Last time step is significant hotspot; none before
CONSECUTIVE    ≥2 consecutive sig. hotspot steps ending at the last step
INTENSIFYING   ≥90% of steps are sig. hotspot AND significant upward MK trend
PERSISTENT     ≥90% of steps are sig. hotspot AND no significant MK trend
DIMINISHING    ≥90% of steps are sig. hotspot AND significant downward MK trend
SPORADIC       <90% of steps are sig. hotspot (on-off pattern)
OSCILLATING    Significant hotspot at some steps, significant coldspot at others
HISTORICAL     Was sig. hotspot at some point but not in most recent 2 steps
NO_PATTERN     Never a significant hotspot

Cold-spot equivalents (prefix COLD_) follow the same logic for coldspots.
"""

import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import libpysal.weights as lps_weights
from scipy import stats

from .space_time_cube import SpaceTimeCube


# ---------------------------------------------------------------------------
# Mann-Kendall (pure-numpy, no external dependency required)
# ---------------------------------------------------------------------------

def mann_kendall(x: np.ndarray) -> dict:
    """
    Two-sided Mann-Kendall trend test.

    Returns
    -------
    dict with keys: trend ('increasing'|'decreasing'|'no trend'),
                    p_value, z, s, significant (bool at alpha=0.05)
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 4:
        return dict(trend="no trend", p_value=1.0, z=0.0, s=0, significant=False)

    s = 0
    for k in range(n - 1):
        s += np.sign(x[k + 1:] - x[k]).sum()

    # Variance (ignoring ties for simplicity)
    var_s = n * (n - 1) * (2 * n + 5) / 18.0

    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0

    p = 2 * (1 - stats.norm.cdf(abs(z)))
    if z > 0:
        trend = "increasing"
    elif z < 0:
        trend = "decreasing"
    else:
        trend = "no trend"

    return dict(trend=trend, p_value=float(p), z=float(z), s=int(s), significant=p < 0.05)


# ---------------------------------------------------------------------------
# Gi* computation
# ---------------------------------------------------------------------------

def _build_queen_weights(nrows: int, ncols: int) -> lps_weights.W:
    """Queen-contiguity weights for a regular nrows×ncols lattice."""
    return lps_weights.lat2W(nrows, ncols, rook=False)


def compute_gi_star_slice(values_flat: np.ndarray, w: lps_weights.W) -> np.ndarray:
    """
    Compute Getis-Ord Gi* Z-scores for a single spatial slice.

    Parameters
    ----------
    values_flat : 1-D array of length n_cells (row-major order)
    w           : libpysal spatial weights (row-standardized or binary)

    Returns
    -------
    z_scores : 1-D array of Gi* Z-scores
    """
    n = len(values_flat)
    x = values_flat.astype(float)

    x_bar = x.mean()
    s = x.std()

    if s == 0:
        return np.zeros(n)

    # Gi* formula (using row-standardized neighbour sums including self)
    # Z_i = (sum_j w_ij * x_j  -  x_bar * W_i) / (s * sqrt((n*W2i - W1i^2)/(n-1)))
    # where W_i = sum_j w_ij  and  W2i = sum_j w_ij^2
    # With star=True we include self in the sum.

    z = np.zeros(n)
    for i in range(n):
        # neighbours including self
        nbrs = list(w.neighbors[i]) + [i]
        nbr_w = [w.weights[i][list(w.neighbors[i]).index(j)] if j != i else 1.0 for j in nbrs]

        W1 = sum(nbr_w)
        W2 = sum(wi ** 2 for wi in nbr_w)
        num = sum(wi * x[j] for wi, j in zip(nbr_w, nbrs)) - x_bar * W1
        denom = s * np.sqrt((n * W2 - W1 ** 2) / (n - 1))
        z[i] = num / denom if denom != 0 else 0.0

    return z


# ---------------------------------------------------------------------------
# Emerging Hotspot Analyzer
# ---------------------------------------------------------------------------

class EmergingHotspotAnalyzer:
    """
    Run Gi* over every time slice of a SpaceTimeCube and classify emerging
    hotspot / coldspot patterns.

    Parameters
    ----------
    stc : SpaceTimeCube
        A fitted SpaceTimeCube instance.
    alpha : float
        Significance threshold for Gi* Z-scores (default 0.05 → |Z| > 1.96).
    persistent_pct : float
        Fraction of time steps required to call a cell "persistent/intensifying/
        diminishing" (default 0.90).
    """

    _HOTSPOT_CATS = [
        "NEW_HOTSPOT", "CONSECUTIVE_HOTSPOT", "INTENSIFYING_HOTSPOT",
        "PERSISTENT_HOTSPOT", "DIMINISHING_HOTSPOT", "SPORADIC_HOTSPOT",
        "OSCILLATING_HOTSPOT", "HISTORICAL_HOTSPOT",
    ]
    _COLDSPOT_CATS = [c.replace("HOTSPOT", "COLDSPOT") for c in _HOTSPOT_CATS]

    def __init__(
        self,
        stc: SpaceTimeCube,
        alpha: float = 0.05,
        persistent_pct: float = 0.90,
    ):
        self.stc = stc
        self.alpha = alpha
        self.z_thresh = abs(stats.norm.ppf(alpha / 2))   # ~1.96
        self.persistent_pct = persistent_pct

        # Filled by .fit()
        self.z_cube: np.ndarray | None = None     # (T, nrows, ncols)
        self.mk_results: list | None = None        # length n_cells
        self.category_grid: np.ndarray | None = None   # (nrows, ncols) str

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, verbose: bool = True) -> "EmergingHotspotAnalyzer":
        """Compute Gi* for all time steps and classify every cell."""
        stc = self.stc
        w = _build_queen_weights(stc.nrows, stc.ncols)

        # 1. Gi* for every time step
        z_cube = np.zeros((stc.T, stc.nrows, stc.ncols), dtype=np.float32)
        for t in range(stc.T):
            flat = stc.flat_slice(t)
            z_flat = compute_gi_star_slice(flat, w)
            z_cube[t] = z_flat.reshape(stc.nrows, stc.ncols)
            if verbose and (t % 6 == 0 or t == stc.T - 1):
                print(f"  [Gi*] time step {t+1}/{stc.T}")

        self.z_cube = z_cube

        # 2. Mann-Kendall per cell and classification
        n_cells = stc.n_cells()
        self.mk_results = []
        category_flat = np.full(n_cells, "NO_PATTERN", dtype=object)

        for flat_i in range(n_cells):
            r, c = stc.flat_to_rowcol(flat_i)
            z_series = z_cube[:, r, c].astype(float)
            mk = mann_kendall(z_series)
            self.mk_results.append(mk)
            category_flat[flat_i] = self._classify(z_series, mk)

        self.category_grid = category_flat.reshape(stc.nrows, stc.ncols)
        return self

    # ------------------------------------------------------------------
    # Classification logic
    # ------------------------------------------------------------------

    def _classify(self, z_series: np.ndarray, mk: dict) -> str:
        T = len(z_series)
        thresh = self.z_thresh
        sig_hot  = z_series >  thresh   # significant hotspot at each step
        sig_cold = z_series < -thresh   # significant coldspot at each step

        hot_any   = sig_hot.any()
        cold_any  = sig_cold.any()
        hot_pct   = sig_hot.mean()

        last_hot  = sig_hot[-1]
        last_cold = sig_cold[-1]

        # ---------- HOTSPOT branch ----------
        if hot_any:
            # Oscillating: both hotspot and coldspot at different steps
            if cold_any:
                return "OSCILLATING_HOTSPOT"

            # High-coverage hotspot (≥ persistent_pct of all steps)
            if hot_pct >= self.persistent_pct:
                if mk["significant"] and mk["trend"] == "increasing":
                    return "INTENSIFYING_HOTSPOT"
                if mk["significant"] and mk["trend"] == "decreasing":
                    return "DIMINISHING_HOTSPOT"
                return "PERSISTENT_HOTSPOT"

            # Not in most recent 2 steps
            if not sig_hot[-1] and not sig_hot[min(-2, -T)]:
                return "HISTORICAL_HOTSPOT"

            # New hotspot: only the last step is significant
            if last_hot and not sig_hot[:-1].any():
                return "NEW_HOTSPOT"

            # Consecutive: last 2+ steps are significant hotspots
            if last_hot and T >= 2 and sig_hot[-2]:
                return "CONSECUTIVE_HOTSPOT"

            return "SPORADIC_HOTSPOT"

        # ---------- COLDSPOT branch ----------
        if cold_any:
            cold_pct = sig_cold.mean()
            if hot_any:
                return "OSCILLATING_COLDSPOT"
            if cold_pct >= self.persistent_pct:
                if mk["significant"] and mk["trend"] == "decreasing":
                    return "INTENSIFYING_COLDSPOT"
                if mk["significant"] and mk["trend"] == "increasing":
                    return "DIMINISHING_COLDSPOT"
                return "PERSISTENT_COLDSPOT"
            if not sig_cold[-1] and not sig_cold[min(-2, -T)]:
                return "HISTORICAL_COLDSPOT"
            if last_cold and not sig_cold[:-1].any():
                return "NEW_COLDSPOT"
            if last_cold and T >= 2 and sig_cold[-2]:
                return "CONSECUTIVE_COLDSPOT"
            return "SPORADIC_COLDSPOT"

        return "NO_PATTERN"

    # ------------------------------------------------------------------
    # Export results
    # ------------------------------------------------------------------

    def to_geodataframe(self) -> gpd.GeoDataFrame:
        """
        Return a GeoDataFrame of grid cells with Gi* summary statistics
        and emerging-hotspot category labels.
        """
        stc = self.stc
        gdf = stc.cell_gdf.copy()
        n = len(gdf)

        # Add total count and mean Gi* Z-score
        totals = stc.total_counts().ravel()
        mean_z = self.z_cube.mean(axis=0).ravel()
        last_z = self.z_cube[-1].ravel()
        mk_pval = [r["p_value"] for r in self.mk_results]
        mk_trend = [r["trend"] for r in self.mk_results]

        gdf["total_count"] = totals[gdf["flat_idx"].values]
        gdf["mean_gi_z"]   = mean_z[gdf["flat_idx"].values]
        gdf["last_gi_z"]   = last_z[gdf["flat_idx"].values]
        gdf["mk_p_value"]  = [mk_pval[i] for i in gdf["flat_idx"].values]
        gdf["mk_trend"]    = [mk_trend[i] for i in gdf["flat_idx"].values]
        gdf["category"]    = [self.category_grid.ravel()[i] for i in gdf["flat_idx"].values]

        return gdf

    def summary(self) -> pd.Series:
        """Count of cells per emerging-hotspot category."""
        cats, cnts = np.unique(self.category_grid, return_counts=True)
        return pd.Series(cnts, index=cats, name="cell_count").sort_values(ascending=False)

    def get_z_slice(self, t: int) -> np.ndarray:
        """Gi* Z-score grid for time step t."""
        return self.z_cube[t]
