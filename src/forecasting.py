"""
forecasting.py
==============
Per-cell time-series forecasting of crime counts.

Strategy
--------
For each grid cell we fit a separate univariate model to its monthly count
time series and forecast the next N steps.

Model selection per cell:
    - ARIMA(1, 1, 1) via statsmodels  — cells with ≥ min_obs observations
    - Prophet (Facebook)              — optional; richer seasonality decomposition
    - Simple Exponential Smoothing    — fallback for sparse cells
    - Mean imputation                 — cells with no historical activity

The per-cell forecasts are assembled back into a spatial grid, giving a
predicted crime-count raster for each future time step.

Stationarity
------------
ADF test is run on each cell time series.  Non-stationary cells are
first-differenced before fitting ARIMA.

Output
------
    forecast_cube  : np.ndarray, shape (n_steps, nrows, ncols)  — point forecast
    lower_cube     : np.ndarray, shape (n_steps, nrows, ncols)  — 95% lower CI
    upper_cube     : np.ndarray, shape (n_steps, nrows, ncols)  — 95% upper CI
"""

import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

from .space_time_cube import SpaceTimeCube


# ---------------------------------------------------------------------------
# Individual cell forecasters
# ---------------------------------------------------------------------------

def _check_stationarity(series: np.ndarray) -> bool:
    """Return True if series is stationary (ADF p < 0.05)."""
    try:
        from statsmodels.tsa.stattools import adfuller
        _, p_value, *_ = adfuller(series, autolag="AIC")
        return p_value < 0.05
    except Exception:
        return True  # assume stationary on failure


def _forecast_arima(series: np.ndarray, n_steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    ARIMA(1,d,1) forecast with 95% confidence interval.
    d is determined by ADF stationarity test.
    Returns (point_forecast, lower_95, upper_95).
    """
    from statsmodels.tsa.arima.model import ARIMA
    d = 0 if _check_stationarity(series) else 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model  = ARIMA(series, order=(1, d, 1))
        result = model.fit()
        fc_obj = result.get_forecast(steps=n_steps)
        fc     = fc_obj.predicted_mean
        ci     = fc_obj.conf_int(alpha=0.05)
    lower = np.clip(ci[:, 0] if ci.ndim == 2 else ci.iloc[:, 0].values, 0, None)
    upper = np.clip(ci[:, 1] if ci.ndim == 2 else ci.iloc[:, 1].values, 0, None)
    return np.clip(fc, 0, None), lower, upper


def _forecast_prophet(series: np.ndarray, n_steps: int, freq: str = "MS") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prophet forecast with uncertainty intervals.
    Returns (point_forecast, lower_95, upper_95).
    """
    from prophet import Prophet
    dates = pd.date_range(end=pd.Timestamp.today(), periods=len(series), freq=freq)
    df = pd.DataFrame({"ds": dates, "y": series.astype(float)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                    daily_seasonality=False, interval_width=0.95,
                    seasonality_mode="additive")
        m.fit(df)
        future = m.make_future_dataframe(periods=n_steps, freq=freq)
        forecast = m.predict(future).tail(n_steps)
    fc    = np.clip(forecast["yhat"].values,       0, None)
    lower = np.clip(forecast["yhat_lower"].values, 0, None)
    upper = np.clip(forecast["yhat_upper"].values, 0, None)
    return fc, lower, upper


def _forecast_ses(series: np.ndarray, n_steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simple Exponential Smoothing forecast (with approximate CI via residual std)."""
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model  = SimpleExpSmoothing(series, initialization_method="estimated")
        result = model.fit(optimized=True)
        fc     = result.forecast(n_steps)
    residual_std = max(result.resid.std(), 0.0)
    z95 = 1.96
    lower = np.clip(fc - z95 * residual_std, 0, None)
    upper = fc + z95 * residual_std
    return np.clip(fc, 0, None), lower, upper


def _forecast_mean(series: np.ndarray, n_steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Naive mean forecast with simple CI."""
    m   = max(series.mean(), 0.0)
    std = series.std()
    fc  = np.full(n_steps, m)
    return fc, np.clip(fc - 1.96 * std, 0, None), fc + 1.96 * std


# ---------------------------------------------------------------------------
# Spatial ensemble smoother
# ---------------------------------------------------------------------------

def spatial_smooth(grid: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Apply Gaussian smoothing to a 2-D forecast grid to borrow strength
    from neighbouring cells.
    """
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(grid, sigma=sigma)


# ---------------------------------------------------------------------------
# Main forecaster
# ---------------------------------------------------------------------------

class SpaceTimeForecaster:
    """
    Fit per-cell time-series models on a training window and forecast
    future crime counts with 95% confidence intervals.

    Parameters
    ----------
    stc : SpaceTimeCube
        Fitted SpaceTimeCube.
    train_t_end : int
        Last time-step index (exclusive) to include in training.
        E.g. train_t_end=20 uses steps 0–19 for training.
    min_obs_arima : int
        Minimum number of non-zero observations required to use ARIMA.
    smooth_sigma : float
        Gaussian smoothing sigma applied to the assembled forecast grid.
        0 = no smoothing.
    use_prophet : bool
        If True and prophet is installed, use Prophet for cells with
        enough data.  Falls back to ARIMA if prophet is not available.
    """

    def __init__(
        self,
        stc: SpaceTimeCube,
        train_t_end: int | None = None,
        min_obs_arima: int = 8,
        smooth_sigma: float = 0.8,
        use_prophet: bool = False,
    ):
        self.stc = stc
        self.train_t_end   = train_t_end if train_t_end is not None else stc.T
        self.min_obs_arima = min_obs_arima
        self.smooth_sigma  = smooth_sigma
        self.use_prophet   = use_prophet

        self.forecast_cube: np.ndarray | None = None  # (n_steps, nrows, ncols)
        self.lower_cube:    np.ndarray | None = None  # 95% lower CI
        self.upper_cube:    np.ndarray | None = None  # 95% upper CI
        self.n_steps: int = 0
        self._arima_count   = 0
        self._prophet_count = 0
        self._ses_count     = 0
        self._mean_count    = 0
        self._prophet_avail: bool | None = None

    # ------------------------------------------------------------------
    # Fit & Predict
    # ------------------------------------------------------------------

    def fit_predict(self, n_steps: int = 1, verbose: bool = True) -> np.ndarray:
        """
        Train on cube[:train_t_end] and forecast the next *n_steps* time steps.

        Returns
        -------
        forecast_cube : np.ndarray, shape (n_steps, nrows, ncols) — point forecast
        Also sets self.lower_cube and self.upper_cube (95% CI).
        """
        stc = self.stc
        self.n_steps = n_steps
        train_cube   = stc.cube[:self.train_t_end]

        # Check Prophet availability once
        if self.use_prophet and self._prophet_avail is None:
            try:
                import prophet  # noqa
                self._prophet_avail = True
            except ImportError:
                self._prophet_avail = False
                if verbose:
                    print("[Forecaster] prophet not installed — falling back to ARIMA")

        fc_flat    = np.zeros((n_steps, stc.nrows * stc.ncols), dtype=np.float32)
        lower_flat = np.zeros_like(fc_flat)
        upper_flat = np.zeros_like(fc_flat)

        cells    = list(range(stc.n_cells()))
        iterator = tqdm(cells, desc="Forecasting cells", disable=not verbose)

        for flat_i in iterator:
            r, c   = stc.flat_to_rowcol(flat_i)
            series = train_cube[:, r, c].astype(float)
            fc, lo, hi = self._fit_cell(series, n_steps)
            fc_flat[:, flat_i]    = fc
            lower_flat[:, flat_i] = lo
            upper_flat[:, flat_i] = hi

        fc_cube    = fc_flat.reshape(n_steps, stc.nrows, stc.ncols)
        lower_cube = lower_flat.reshape(n_steps, stc.nrows, stc.ncols)
        upper_cube = upper_flat.reshape(n_steps, stc.nrows, stc.ncols)

        if self.smooth_sigma > 0:
            for s in range(n_steps):
                fc_cube[s]    = spatial_smooth(fc_cube[s],    sigma=self.smooth_sigma)
                lower_cube[s] = spatial_smooth(lower_cube[s], sigma=self.smooth_sigma)
                upper_cube[s] = spatial_smooth(upper_cube[s], sigma=self.smooth_sigma)

        self.forecast_cube = fc_cube
        self.lower_cube    = lower_cube
        self.upper_cube    = upper_cube

        if verbose:
            print(
                f"[Forecaster] {self._arima_count} ARIMA | "
                f"{self._prophet_count} Prophet | "
                f"{self._ses_count} SES | {self._mean_count} mean cells"
            )

        return fc_cube

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fit_cell(self, series: np.ndarray, n_steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (point_forecast, lower_95, upper_95) for one cell."""
        n_nonzero = int((series > 0).sum())
        n_total   = series.sum()

        if n_total == 0:
            self._mean_count += 1
            zeros = np.zeros(n_steps)
            return zeros, zeros, zeros

        # Prophet (richest model — only for cells with plenty of data)
        if self.use_prophet and self._prophet_avail and n_nonzero >= 12 and len(series) >= 18:
            try:
                fc, lo, hi = _forecast_prophet(series, n_steps)
                self._prophet_count += 1
                return fc, lo, hi
            except Exception:
                pass

        # ARIMA with stationarity check
        if n_nonzero >= self.min_obs_arima and len(series) >= 12:
            try:
                fc, lo, hi = _forecast_arima(series, n_steps)
                self._arima_count += 1
                return fc, lo, hi
            except Exception:
                pass

        # Simple Exponential Smoothing
        if n_nonzero >= 3:
            try:
                fc, lo, hi = _forecast_ses(series, n_steps)
                self._ses_count += 1
                return fc, lo, hi
            except Exception:
                pass

        self._mean_count += 1
        return _forecast_mean(series, n_steps)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def risk_surface(self, step: int = 0) -> np.ndarray:
        """
        Return a normalised risk surface for forecast step *step*.
        Values in [0, 1] where 1 = highest predicted crime density.
        """
        g = self.forecast_cube[step]
        mn, mx = g.min(), g.max()
        if mx == mn:
            return np.zeros_like(g)
        return (g - mn) / (mx - mn)

    def hotspot_mask(self, step: int = 0, percentile: float = 80.0) -> np.ndarray:
        """
        Boolean mask of predicted hotspot cells (top *percentile*% of risk).
        """
        surface = self.risk_surface(step)
        threshold = np.percentile(surface, percentile)
        return surface >= threshold

    def future_period_labels(self) -> list:
        """Period labels for the forecasted steps."""
        last_period = self.stc.time_labels[self.train_t_end - 1]
        labels = []
        p = last_period
        for _ in range(self.n_steps):
            p = p + 1
            labels.append(p)
        return labels
