"""
spatial_lag.py
==============
Spatial regression models for crime count prediction.

Models
------
OLS Baseline
    y_t = X_t β + ε
    No spatial structure; acts as a benchmark.

Spatial Lag Model (SLM)  — also called SAR (Simultaneous Autoregressive)
    y_t = ρ W y_t + X_t β + ε
    ρ : spatial autoregressive coefficient
    W : row-standardised spatial weights matrix

Spatial Error Model (SEM)
    y_t = X_t β + u_t,   u_t = λ W u_t + ε_t
    λ : spatial autocorrelation of the error term

Panel Spatial Lag
    y_{it} = ρ W y_{it} + α_t + X_{it} β + ε_{it}
    Stacks all time steps; includes time fixed effects.

Features (X)
-----------
    - temporal_lag : crime count at the same cell, previous month
    - spatial_lag  : spatially weighted crime count of neighbours (W y_{t-1})
    - trend        : numeric time index
    - sin/cos seasonality : captures annual cycle

Requires: libpysal, spreg
"""

import warnings
import numpy as np
import pandas as pd
import libpysal.weights as lps_weights
from scipy import stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_queen_weights(nrows: int, ncols: int) -> lps_weights.W:
    w = lps_weights.lat2W(nrows, ncols, rook=False)
    w.transform = "r"  # row-standardise
    return w


def _make_features(
    cube: np.ndarray,
    t: int,
    w: lps_weights.W,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build feature matrix X and response y for time step t.

    Returns X (n_cells, n_features) and y (n_cells,).
    Requires t >= 1 (temporal lag exists).
    """
    T, nrows, ncols = cube.shape
    n = nrows * ncols

    y = cube[t].ravel().astype(float)

    # Temporal lag (previous month counts)
    y_lag1 = cube[t - 1].ravel().astype(float)

    # Spatial lag of previous month (W y_{t-1})
    sp_lag = np.array([
        sum(w.weights[i][k] * y_lag1[w.neighbors[i][k]] for k in range(len(w.neighbors[i])))
        for i in range(n)
    ])

    # Time-trend component
    trend = np.full(n, float(t))

    # Seasonal features (monthly — period = 12)
    month_idx = t % 12
    sin_season = np.full(n, np.sin(2 * np.pi * month_idx / 12))
    cos_season = np.full(n, np.cos(2 * np.pi * month_idx / 12))

    X = np.column_stack([y_lag1, sp_lag, trend, sin_season, cos_season])
    return X, y


# ---------------------------------------------------------------------------
# OLS baseline
# ---------------------------------------------------------------------------

class OLSBaseline:
    """
    Ordinary least squares regression (no spatial structure).
    Useful as a benchmark for spatial models.
    """

    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self.r2_ = None
        self._w = None
        self._nrows = self._ncols = 0

    def fit(self, stc, train_t_end: int) -> "OLSBaseline":
        from .space_time_cube import SpaceTimeCube
        cube = stc.cube
        T, nrows, ncols = cube.shape
        self._nrows, self._ncols = nrows, ncols
        self._w = _build_queen_weights(nrows, ncols)

        X_all, y_all = [], []
        for t in range(1, train_t_end):
            X, y = _make_features(cube, t, self._w)
            X_all.append(X)
            y_all.append(y)

        X_all = np.vstack(X_all)
        y_all = np.concatenate(y_all)

        # Add intercept
        Xb = np.column_stack([np.ones(len(X_all)), X_all])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            beta, _, _, _ = np.linalg.lstsq(Xb, y_all, rcond=None)

        self.intercept_ = beta[0]
        self.coef_ = beta[1:]

        y_pred = Xb @ beta
        ss_res = ((y_all - y_pred) ** 2).sum()
        ss_tot = ((y_all - y_all.mean()) ** 2).sum()
        self.r2_ = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        print(f"[OLS] R² = {self.r2_:.4f}  |  coef = {np.round(self.coef_, 4)}")
        return self

    def predict(self, stc, t: int) -> np.ndarray:
        X, _ = _make_features(stc.cube, t, self._w)
        Xb = np.column_stack([np.ones(len(X)), X])
        pred = Xb @ np.concatenate([[self.intercept_], self.coef_])
        return np.clip(pred, 0, None).reshape(self._nrows, self._ncols)


# ---------------------------------------------------------------------------
# Spatial Lag Model (SAR)
# ---------------------------------------------------------------------------

class SpatialLagModel:
    """
    Spatial Lag Model estimated by 2SLS / Generalised Method of Moments
    via `spreg.GM_Lag`.

    y = ρ Wy + Xβ + ε

    The spatial multiplier (I - ρW)^{-1} propagates local shocks across
    the lattice, capturing the contagion / displacement of crime.
    """

    def __init__(self):
        self._model = None
        self._w = None
        self._nrows = self._ncols = 0
        self.rho_ = None
        self.coef_ = None
        self.r2_ = None

    def fit(self, stc, train_t_end: int) -> "SpatialLagModel":
        try:
            from spreg import GM_Lag
        except ImportError:
            raise ImportError("spreg is required: pip install spreg")

        cube = stc.cube
        T, nrows, ncols = cube.shape
        self._nrows, self._ncols = nrows, ncols
        self._w = _build_queen_weights(nrows, ncols)

        X_all, y_all = [], []
        for t in range(1, train_t_end):
            X, y = _make_features(cube, t, self._w)
            X_all.append(X)
            y_all.append(y)

        X_all = np.vstack(X_all)
        y_all = np.concatenate(y_all).reshape(-1, 1)

        # Stack spatial weights for the panel (block-diagonal)
        n_cells = nrows * ncols
        n_times = train_t_end - 1
        w_panel = lps_weights.block_weights([self._w] * n_times, silence_warnings=True)
        w_panel.transform = "r"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = GM_Lag(y_all, X_all, w=w_panel, name_y="crime", name_x=["y_lag1", "sp_lag", "trend", "sin", "cos"])

        self._model = model
        self.rho_ = float(model.betas[-1])
        self.coef_ = model.betas[:-1].ravel()
        self.r2_ = float(model.pr2)

        print(
            f"[SpatialLag] ρ = {self.rho_:.4f}  |  "
            f"Pseudo-R² = {self.r2_:.4f}  |  "
            f"coef = {np.round(self.coef_, 4)}"
        )
        return self

    def predict(self, stc, t: int) -> np.ndarray:
        """OLS-style reduced-form prediction (approximation)."""
        X, _ = _make_features(stc.cube, t, self._w)
        Xb = np.column_stack([np.ones(len(X)), X])
        # Direct effect approximation: ŷ ≈ (I - ρ W)^{-1} Xβ  ≈ Xβ / (1 - ρ)
        beta = self.coef_   # includes intercept as first element
        pred = Xb @ beta
        if abs(1 - self.rho_) > 1e-6:
            pred = pred / (1 - self.rho_)
        return np.clip(pred, 0, None).reshape(self._nrows, self._ncols)


# ---------------------------------------------------------------------------
# Spatial Error Model (SEM)
# ---------------------------------------------------------------------------

class SpatialErrorModel:
    """
    Spatial Error Model via `spreg.GM_Error`.

    y = Xβ + u,    u = λ Wu + ε

    Accounts for unobserved spatial spillovers in the error term without
    introducing a direct lag of the dependent variable.
    """

    def __init__(self):
        self._model = None
        self._w = None
        self._nrows = self._ncols = 0
        self.lambda_ = None
        self.coef_ = None
        self.r2_ = None

    def fit(self, stc, train_t_end: int) -> "SpatialErrorModel":
        try:
            from spreg import GM_Error
        except ImportError:
            raise ImportError("spreg is required: pip install spreg")

        cube = stc.cube
        T, nrows, ncols = cube.shape
        self._nrows, self._ncols = nrows, ncols
        self._w = _build_queen_weights(nrows, ncols)

        X_all, y_all = [], []
        for t in range(1, train_t_end):
            X, y = _make_features(cube, t, self._w)
            X_all.append(X)
            y_all.append(y)

        X_all = np.vstack(X_all)
        y_all = np.concatenate(y_all).reshape(-1, 1)

        n_times = train_t_end - 1
        w_panel = lps_weights.block_weights([self._w] * n_times, silence_warnings=True)
        w_panel.transform = "r"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = GM_Error(y_all, X_all, w=w_panel)

        self._model = model
        self.lambda_ = float(model.betas[-1])
        self.coef_ = model.betas[:-1].ravel()
        self.r2_ = float(model.pr2)

        print(
            f"[SpatialError] λ = {self.lambda_:.4f}  |  "
            f"Pseudo-R² = {self.r2_:.4f}  |  "
            f"coef = {np.round(self.coef_, 4)}"
        )
        return self

    def predict(self, stc, t: int) -> np.ndarray:
        X, _ = _make_features(stc.cube, t, self._w)
        Xb = np.column_stack([np.ones(len(X)), X])
        pred = Xb @ self.coef_
        return np.clip(pred, 0, None).reshape(self._nrows, self._ncols)


# ---------------------------------------------------------------------------
# Moran's I (global spatial autocorrelation)
# ---------------------------------------------------------------------------

def morans_i(values: np.ndarray, w: lps_weights.W) -> dict:
    """
    Compute global Moran's I for a spatial variable.

    Returns
    -------
    dict: I, E[I], Var[I], z, p_value
    """
    n = len(values)
    z = values - values.mean()
    wz = np.array([
        sum(w.weights[i][k] * z[w.neighbors[i][k]] for k in range(len(w.neighbors[i])))
        for i in range(n)
    ])
    W = sum(sum(row) for row in w.weights.values())
    I = (n / W) * (z @ wz) / (z @ z)

    E_I = -1 / (n - 1)
    # Variance under normality assumption
    S1 = 0.5 * sum(
        (w.weights[i][k] + w.weights[j][w.neighbors[j].index(i)]) ** 2
        if i in w.neighbors[j] else 0
        for i in range(n)
        for k, j in enumerate(w.neighbors[i])
    )
    S2 = sum(
        (sum(w.weights[i]) + sum(w.weights[j][w.neighbors[j].index(i)]
                                  if i in w.neighbors[j] else 0
                                  for j in range(n))) ** 2
        for i in range(n)
    )
    var_I = (n * ((n**2 - 3*n + 3)*S1 - n*S2 + 3*W**2)
             - (z**4).sum() / (z**2).sum()**2 * ((n**2 - n)*S1 - 2*n*S2 + 6*W**2)
             ) / ((n-1)*(n-2)*(n-3)*W**2) - E_I**2

    var_I = max(var_I, 1e-12)  # guard against near-zero
    z_score = (I - E_I) / np.sqrt(var_I)
    p_val = 2 * (1 - stats.norm.cdf(abs(z_score)))

    return dict(I=float(I), E_I=float(E_I), z=float(z_score), p_value=float(p_val))
