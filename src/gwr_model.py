"""
gwr_model.py
============
Geographically Weighted Regression (GWR)

GWR relaxes the assumption of spatial stationarity in OLS by allowing
regression coefficients to vary over space.  For each location i:

    y_i = β_0(u_i, v_i) + β_1(u_i, v_i) x_{i1} + … + ε_i

where the β_k(u_i, v_i) are estimated using a local weighted least-squares
with a spatial kernel that down-weights observations far from i.

Implementation uses a pure-NumPy Gaussian kernel GWR (no external gwr/mgwr
dependency required — falls back to `mgwr` if available for bandwidth
cross-validation).

Outputs
-------
params      : (n_cells, n_features+1) array of local coefficients
              column 0 = intercept, columns 1…p = predictor coefficients
local_r2    : (n_cells,) local pseudo-R² values
residuals   : (n_cells,) local OLS residuals
bandwidth   : optimal bandwidth (metres) selected by GCV or provided
"""

import warnings
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Gaussian kernel
# ---------------------------------------------------------------------------

def _gaussian_kernel(distances: np.ndarray, bandwidth: float) -> np.ndarray:
    """Gaussian spatial kernel weights."""
    return np.exp(-0.5 * (distances / bandwidth) ** 2)


def _bisquare_kernel(distances: np.ndarray, bandwidth: float) -> np.ndarray:
    """Bi-square kernel (compact support — zero beyond bandwidth)."""
    u = distances / bandwidth
    w = (1 - u ** 2) ** 2
    w[u >= 1] = 0
    return w


# ---------------------------------------------------------------------------
# Bandwidth selection via Generalised Cross-Validation (GCV)
# ---------------------------------------------------------------------------

def select_bandwidth_gcv(
    coords: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    bandwidths: np.ndarray | None = None,
    kernel: str = "gaussian",
) -> float:
    """
    Select optimal fixed bandwidth by minimising GCV score.

    Parameters
    ----------
    coords      : (n, 2) array of (x, y) locations
    y           : (n,) dependent variable
    X           : (n, p) feature matrix (WITHOUT intercept column)
    bandwidths  : candidate bandwidths to try. Default: 10 values from
                  5th to 50th percentile of pairwise distances.
    kernel      : 'gaussian' or 'bisquare'

    Returns
    -------
    float : optimal bandwidth
    """
    from scipy.spatial.distance import pdist
    dists = pdist(coords)

    if bandwidths is None:
        p5  = np.percentile(dists, 5)
        p50 = np.percentile(dists, 50)
        bandwidths = np.linspace(p5, p50, 12)

    kern_fn = _gaussian_kernel if kernel == "gaussian" else _bisquare_kernel
    n = len(y)
    Xb = np.column_stack([np.ones(n), X])

    best_bw  = bandwidths[0]
    best_gcv = np.inf

    from scipy.spatial.distance import cdist
    D = cdist(coords, coords)

    for bw in bandwidths:
        gcv = _gcv_score(D, y, Xb, bw, kern_fn, n)
        if gcv < best_gcv:
            best_gcv = gcv
            best_bw  = bw

    return float(best_bw)


def _gcv_score(D, y, Xb, bw, kern_fn, n):
    """GCV score for a given bandwidth."""
    residuals = np.zeros(n)
    hat_diag  = np.zeros(n)

    for i in range(n):
        w = kern_fn(D[i], bw)
        W = np.diag(w)
        XtW = Xb.T @ W
        try:
            H_ii = Xb[i] @ np.linalg.solve(XtW @ Xb + 1e-10 * np.eye(Xb.shape[1]), XtW[:, i])
        except np.linalg.LinAlgError:
            return np.inf
        beta = np.linalg.lstsq(XtW @ Xb, XtW @ y, rcond=None)[0]
        residuals[i] = y[i] - Xb[i] @ beta
        hat_diag[i]  = H_ii

    tr_H = hat_diag.sum()
    gcv  = (residuals ** 2).mean() / (1 - tr_H / n) ** 2
    return float(gcv)


# ---------------------------------------------------------------------------
# Main GWR class
# ---------------------------------------------------------------------------

class GWRModel:
    """
    Geographically Weighted Regression with Gaussian kernel.

    Parameters
    ----------
    bandwidth : float | None
        Fixed kernel bandwidth (metres).  If None, selected via GCV.
    kernel : str
        'gaussian' (default) or 'bisquare'.
    auto_bandwidth : bool
        If True and bandwidth is None, run GCV selection. Otherwise use
        a heuristic (25th percentile of pairwise distances).
    feature_names : list[str] | None
        Names for the predictor columns (for output labelling).
    """

    def __init__(
        self,
        bandwidth: float | None = None,
        kernel: str = "gaussian",
        auto_bandwidth: bool = True,
        feature_names: list[str] | None = None,
    ):
        self.bandwidth       = bandwidth
        self.kernel          = kernel
        self.auto_bandwidth  = auto_bandwidth
        self.feature_names   = feature_names

        # Fitted attributes
        self.params_: np.ndarray | None = None    # (n, p+1)
        self.local_r2_: np.ndarray | None = None  # (n,)
        self.residuals_: np.ndarray | None = None # (n,)
        self.fitted_: np.ndarray | None = None    # (n,)
        self._coords = None
        self._y      = None
        self._Xb     = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, coords: np.ndarray, y: np.ndarray, X: np.ndarray) -> "GWRModel":
        """
        Fit GWR.

        Parameters
        ----------
        coords : (n, 2) array — spatial locations (metres)
        y      : (n,)   array — dependent variable
        X      : (n, p) array — predictors (without intercept)
        """
        from scipy.spatial.distance import cdist

        n = len(y)
        Xb = np.column_stack([np.ones(n), X])   # add intercept
        D  = cdist(coords, coords)

        # Bandwidth selection
        if self.bandwidth is None:
            if self.auto_bandwidth:
                print("[GWR] Selecting bandwidth via GCV …")
                self.bandwidth = select_bandwidth_gcv(coords, y, X, kernel=self.kernel)
                print(f"[GWR] Optimal bandwidth = {self.bandwidth:.0f} m")
            else:
                from scipy.spatial.distance import pdist
                self.bandwidth = float(np.percentile(pdist(coords), 25))

        kern_fn = _gaussian_kernel if self.kernel == "gaussian" else _bisquare_kernel

        # Local regression for each location
        params    = np.zeros((n, Xb.shape[1]))
        local_r2  = np.zeros(n)
        residuals = np.zeros(n)
        fitted    = np.zeros(n)

        for i in range(n):
            w = kern_fn(D[i], self.bandwidth)
            W = np.diag(w)
            XtW = Xb.T @ W
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                beta = np.linalg.lstsq(XtW @ Xb, XtW @ y, rcond=None)[0]
            params[i]    = beta
            y_hat_i      = Xb[i] @ beta
            fitted[i]    = y_hat_i
            residuals[i] = y[i] - y_hat_i

        # Local R²: weighted RSS vs TSS
        for i in range(n):
            w = kern_fn(D[i], self.bandwidth)
            y_hat = Xb @ params[i]
            y_w   = y * w
            y_hat_w = y_hat * w
            ss_res = (w * (y - y_hat) ** 2).sum()
            ss_tot = (w * (y - y_w.sum() / w.sum()) ** 2).sum()
            local_r2[i] = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

        self.params_    = params
        self.local_r2_  = np.clip(local_r2, 0, 1)
        self.residuals_ = residuals
        self.fitted_    = fitted
        self._coords    = coords
        self._y         = y
        self._Xb        = Xb

        global_r2 = 1 - (residuals ** 2).sum() / ((y - y.mean()) ** 2).sum()
        print(f"[GWR] Fit complete | bandwidth={self.bandwidth:.0f}m | "
              f"global R²={global_r2:.4f} | n={n}")
        return self

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------

    def coefficient_surface(self, feature_idx: int) -> np.ndarray:
        """
        Return the local coefficient for predictor `feature_idx` as a flat array.
        feature_idx=0 → intercept, 1 → first predictor, etc.
        """
        return self.params_[:, feature_idx]

    def summary_df(self, stc) -> pd.DataFrame:
        """
        Return a DataFrame of local coefficients, R², and residuals,
        indexed by (row, col).
        """
        n = stc.nrows * stc.ncols
        names = ["intercept"] + (self.feature_names or [f"β{i}" for i in range(1, self.params_.shape[1])])
        df = pd.DataFrame(self.params_, columns=names[:self.params_.shape[1]])
        df["local_r2"]  = self.local_r2_
        df["residual"]  = self.residuals_
        df["flat_idx"]  = range(n)
        df["row"]       = [stc.flat_to_rowcol(i)[0] for i in range(n)]
        df["col"]       = [stc.flat_to_rowcol(i)[1] for i in range(n)]
        return df

    def to_grid(self, feature_idx: int, stc) -> np.ndarray:
        """Reshape coefficient surface to (nrows, ncols)."""
        return self.coefficient_surface(feature_idx).reshape(stc.nrows, stc.ncols)


# ---------------------------------------------------------------------------
# Convenience: build GWR features from STC
# ---------------------------------------------------------------------------

def build_gwr_features(stc, t: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build GWR training data for time step t (requires t >= 1).

    Returns
    -------
    coords : (n_cells, 2) — cell centre coordinates
    y      : (n_cells,)   — crime counts at t
    X      : (n_cells, 3) — [temporal_lag, dist_center, seasonal_sin]
    """
    n = stc.n_cells()
    coords = stc.cell_centers_array()   # (n, 2)
    y = stc.cube[t].ravel().astype(float)

    # Feature 1: temporal lag
    y_lag = stc.cube[max(t - 1, 0)].ravel().astype(float)

    # Feature 2: distance to city centre (normalised)
    cx = (stc.bbox[0] + stc.bbox[2]) / 2 if hasattr(stc, "bbox") else coords[:, 0].mean()
    cy = (stc.bbox[1] + stc.bbox[3]) / 2 if hasattr(stc, "bbox") else coords[:, 1].mean()
    # Fall back to mean centre if bbox not set
    if not hasattr(stc, "bbox") or stc.bbox is None:
        cx, cy = coords.mean(axis=0)
    dists = np.sqrt((coords[:, 0] - cx) ** 2 + (coords[:, 1] - cy) ** 2)
    dists_norm = dists / dists.max()

    # Feature 3: seasonal sine
    month = t % 12
    sin_s = np.full(n, np.sin(2 * np.pi * month / 12))

    X = np.column_stack([y_lag, dists_norm, sin_s])
    return coords, y, X


def fit_gwr_on_stc(stc, t: int = -1, bandwidth: float | None = None) -> GWRModel:
    """
    Convenience wrapper: fit GWR on the space-time cube at time step t.

    Parameters
    ----------
    stc       : SpaceTimeCube
    t         : time step index (default -1 = last step)
    bandwidth : spatial kernel bandwidth; if None, selected via GCV
    """
    if t < 0:
        t = stc.T + t
    t = max(t, 1)  # need at least one lag

    coords, y, X = build_gwr_features(stc, t)

    model = GWRModel(
        bandwidth=bandwidth,
        auto_bandwidth=(bandwidth is None),
        feature_names=["temporal_lag", "dist_center", "seasonal_sin"],
    )
    model.fit(coords, y, X)

    # Store bbox reference for to_grid
    if not hasattr(stc, "bbox") or stc.bbox is None:
        stc.bbox = (stc._minx, stc._miny,
                    stc._minx + stc.ncols * stc.cell_size,
                    stc._miny + stc.nrows * stc.cell_size)

    return model
