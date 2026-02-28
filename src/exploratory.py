"""
exploratory.py
==============
Phase 2 — Exploratory Spatial Data Analysis

Methods
-------
KDE (Kernel Density Estimation)
    Gaussian KDE on the study grid.  Returns a (nrows, ncols) density surface.

Local Moran's I (LISA)
    For each cell: local indicator of spatial association.
    Quadrant labels: HH=1, LH=2, LL=3, HL=4  (High-High, Low-High, etc.)
    Only cells where p < alpha are considered significant.

Ripley's K / L function
    Measures clustering at multiple spatial scales.
    L(d) > 0  → clustered beyond random at distance d
    L(d) < 0  → dispersed

Stationarity tests
    Augmented Dickey-Fuller (ADF): H0 = unit root (non-stationary)
    KPSS:                          H0 = stationary
    A series is stationary when ADF rejects AND KPSS fails to reject.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import libpysal.weights as lps_weights
from sklearn.neighbors import BallTree, KernelDensity
from scipy import stats


# ---------------------------------------------------------------------------
# KDE
# ---------------------------------------------------------------------------

def compute_kde_on_grid(gdf: gpd.GeoDataFrame, stc, bandwidth: float | None = None) -> np.ndarray:
    """
    Fit a Gaussian KDE to crime events and evaluate on the STC grid centroids.

    Parameters
    ----------
    gdf       : GeoDataFrame of crime points (local projected CRS)
    stc       : SpaceTimeCube
    bandwidth : kernel bandwidth in same units as coordinates (metres).
                Default: Silverman's rule.

    Returns
    -------
    kde_grid : np.ndarray, shape (nrows, ncols) — density values
    """
    coords = np.column_stack([gdf.geometry.x.values, gdf.geometry.y.values])

    if bandwidth is None:
        # Silverman's rule of thumb
        n, d = coords.shape
        bandwidth = np.std(coords) * (n * (d + 2) / 4) ** (-1 / (d + 4))

    kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian", metric="euclidean")
    kde.fit(coords)

    # Evaluate at cell centres
    centres = stc.cell_centers_array()   # (n_cells, 2)
    log_dens = kde.score_samples(centres)
    density  = np.exp(log_dens)

    return density.reshape(stc.nrows, stc.ncols)


# ---------------------------------------------------------------------------
# Local Moran's I (LISA)
# ---------------------------------------------------------------------------

def compute_lisa(
    values_flat: np.ndarray,
    w: lps_weights.W,
    alpha: float = 0.05,
    permutations: int = 199,
) -> dict:
    """
    Compute Local Moran's I for a single spatial slice.

    Returns
    -------
    dict with keys:
        Is       : local statistic array (n_cells,)
        p_sim    : pseudo p-values from permutation test
        quadrant : 1=HH, 2=LH, 3=LL, 4=HL, 0=not significant
        labels   : string label per cell
    """
    try:
        from esda.moran import Moran_Local
        ml = Moran_Local(values_flat, w, permutations=permutations, seed=42)
        Is    = ml.Is
        p_sim = ml.p_sim
        q     = ml.q.copy().astype(int)   # 1=HH, 2=LH, 3=LL, 4=HL
        q[p_sim >= alpha] = 0              # 0 = not significant
    except Exception:
        # Fallback: manual local Moran's
        Is, p_sim, q = _manual_lisa(values_flat, w, alpha)

    label_map = {0: "Not Significant", 1: "HH (Hotspot)", 2: "LH (Spatial Outlier)",
                 3: "LL (Coldspot)", 4: "HL (Spatial Outlier)"}
    labels = [label_map.get(int(qi), "Not Significant") for qi in q]

    return dict(Is=Is, p_sim=p_sim, quadrant=q, labels=labels)


def _manual_lisa(values_flat, w, alpha):
    """Pure-numpy Local Moran's I fallback."""
    n = len(values_flat)
    x = values_flat.astype(float)
    z = x - x.mean()
    m2 = (z ** 2).mean()

    Is = np.zeros(n)
    for i in range(n):
        nbr_sum = sum(w.weights[i][k] * z[w.neighbors[i][k]]
                      for k in range(len(w.neighbors[i])))
        Is[i] = (z[i] / m2) * nbr_sum

    # Approximate p-values via normal approximation (rough)
    E_Is = -1.0 / (n - 1)
    var_Is = 0.25  # approximate; use permutations for accuracy
    z_scores = (Is - E_Is) / np.sqrt(var_Is)
    p_sim = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))

    # Quadrant assignment
    z = x - x.mean()
    wz = np.array([
        sum(w.weights[i][k] * z[w.neighbors[i][k]] for k in range(len(w.neighbors[i])))
        for i in range(n)
    ])
    q = np.zeros(n, dtype=int)
    q[(z > 0) & (wz > 0)] = 1   # HH
    q[(z < 0) & (wz > 0)] = 2   # LH
    q[(z < 0) & (wz < 0)] = 3   # LL
    q[(z > 0) & (wz < 0)] = 4   # HL
    q[p_sim >= alpha] = 0

    return Is, p_sim, q


def compute_lisa_all_steps(stc, w: lps_weights.W, alpha: float = 0.05) -> list[dict]:
    """Run LISA for every time step in the STC. Returns a list of result dicts."""
    results = []
    for t in range(stc.T):
        flat = stc.flat_slice(t)
        results.append(compute_lisa(flat, w, alpha=alpha))
    return results


# ---------------------------------------------------------------------------
# Ripley's K / L function
# ---------------------------------------------------------------------------

def ripley_kl(
    gdf: gpd.GeoDataFrame,
    distances: np.ndarray | None = None,
    bbox: tuple | None = None,
    n_simulations: int = 39,
    seed: int = 42,
) -> dict:
    """
    Compute Ripley's K(d) and L(d) = sqrt(K(d)/π) - d functions
    with Monte Carlo confidence envelope.

    Parameters
    ----------
    gdf           : crime event GeoDataFrame (projected CRS)
    distances     : array of distance thresholds (metres). Default: 20 values.
    bbox          : (minx, miny, maxx, maxy). Inferred from data if None.
    n_simulations : number of CSR simulations for envelope (0 = no envelope)

    Returns
    -------
    dict: distances, K, L, K_lower, K_upper, L_lower, L_upper
    """
    coords = np.column_stack([gdf.geometry.x.values, gdf.geometry.y.values])
    n = len(coords)

    if bbox is None:
        minx, miny, maxx, maxy = gdf.total_bounds
    else:
        minx, miny, maxx, maxy = bbox
    area = (maxx - minx) * (maxy - miny)

    if distances is None:
        max_d = min(maxx - minx, maxy - miny) / 4
        distances = np.linspace(100, max_d, 20)

    tree = BallTree(coords, metric="euclidean")
    K_obs = _compute_K(tree, coords, distances, n, area)
    L_obs = np.sqrt(K_obs / np.pi) - distances

    # Monte Carlo envelope
    rng = np.random.default_rng(seed)
    K_sims = []
    for _ in range(n_simulations):
        rand_x = rng.uniform(minx, maxx, n)
        rand_y = rng.uniform(miny, maxy, n)
        rand_coords = np.column_stack([rand_x, rand_y])
        rand_tree = BallTree(rand_coords, metric="euclidean")
        K_sims.append(_compute_K(rand_tree, rand_coords, distances, n, area))

    K_sims = np.array(K_sims)
    K_lower = K_sims.min(axis=0)
    K_upper = K_sims.max(axis=0)
    L_lower = np.sqrt(K_lower / np.pi) - distances
    L_upper = np.sqrt(K_upper / np.pi) - distances

    return dict(
        distances=distances,
        K=K_obs, L=L_obs,
        K_lower=K_lower, K_upper=K_upper,
        L_lower=L_lower, L_upper=L_upper,
    )


def _compute_K(tree: BallTree, coords: np.ndarray, distances: np.ndarray, n: int, area: float) -> np.ndarray:
    K = np.zeros(len(distances))
    for idx, d in enumerate(distances):
        counts = tree.query_radius(coords, r=d, count_only=True) - 1  # exclude self
        K[idx] = (area / (n * (n - 1))) * counts.sum()
    return K


# ---------------------------------------------------------------------------
# Stationarity tests
# ---------------------------------------------------------------------------

def test_stationarity(series: np.ndarray, series_name: str = "series") -> dict:
    """
    Augmented Dickey-Fuller (ADF) and KPSS stationarity tests.

    A series is considered stationary when:
        - ADF  p-value  < 0.05  (rejects unit-root H0)
        - KPSS p-value  > 0.05  (fails to reject stationarity H0)

    Returns
    -------
    dict with test statistics, p-values and plain-English verdict
    """
    from statsmodels.tsa.stattools import adfuller, kpss

    s = np.asarray(series, dtype=float)
    if len(s) < 8:
        return dict(verdict="Insufficient data", series=series_name)

    # ADF
    adf_stat, adf_p, adf_lags, _, adf_crit, _ = adfuller(s, autolag="AIC")
    adf_stationary = adf_p < 0.05

    # KPSS
    try:
        kpss_stat, kpss_p, kpss_lags, kpss_crit = kpss(s, regression="c", nlags="auto")
        kpss_stationary = kpss_p > 0.05
    except Exception:
        kpss_stat, kpss_p, kpss_stationary = float("nan"), float("nan"), False

    if adf_stationary and kpss_stationary:
        verdict = "Stationary"
    elif not adf_stationary and not kpss_stationary:
        verdict = "Non-stationary (unit root)"
    elif adf_stationary and not kpss_stationary:
        verdict = "Trend-stationary"
    else:
        verdict = "Difference-stationary (possible structural break)"

    return dict(
        series=series_name,
        adf_statistic=round(float(adf_stat), 4),
        adf_p_value=round(float(adf_p), 4),
        adf_stationary=adf_stationary,
        kpss_statistic=round(float(kpss_stat), 4),
        kpss_p_value=round(float(kpss_p), 4),
        kpss_stationary=kpss_stationary,
        verdict=verdict,
    )


def stationarity_report(stc, min_total: int = 5) -> pd.DataFrame:
    """
    Run stationarity tests on all active cells in the STC.
    Returns a DataFrame sorted by verdict.
    """
    active = stc.active_cells(min_total=min_total)
    records = []
    for r, c in active:
        series = stc.get_cell_timeseries(r, c)
        result = test_stationarity(series, series_name=f"({r},{c})")
        result["row"] = r
        result["col"] = c
        records.append(result)
    return pd.DataFrame(records)
