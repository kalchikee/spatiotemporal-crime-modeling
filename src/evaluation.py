"""
evaluation.py
=============
Predictive accuracy metrics for spatiotemporal crime forecasting.

Metrics
-------
Hit Rate (HR)
    Fraction of actual crimes that fell within predicted hotspot cells.
    HR = crimes_in_hotspot / total_crimes

Predictive Accuracy Index (PAI)
    Adjusts hit rate for the spatial extent of the hotspot area.
    PAI = HR / (area_hotspot / area_total)
    PAI > 1 → prediction outperforms random selection;
    PAI = 5 → hotspot cells capture crimes at 5× the rate of chance.

Area-to-Coverage Index (ACI)  — inverse of PAI normalised view
    ACI = (area_hotspot / area_total) / HR
    Lower is better.

Prediction Efficiency Index (PEI)
    PEI = (HR - area_fraction) / (1 - area_fraction)
    Ranges [-1, 1]; PEI = 1 means perfect prediction at minimal area.

RMSE / MAE
    Per-cell count prediction errors, compared against observed counts.

Moran's I on Residuals
    Checks whether prediction errors are spatially autocorrelated.

ROC Curve + AUC
    Binary classification: predicted hotspot cell vs actual hotspot cell.
    AUC = 1.0 is perfect; AUC = 0.5 is random.

Confusion Matrix
    TP/FP/FN/TN for the binary hotspot prediction task at a chosen threshold.

Precision-Recall Curve
    Precision = TP / (TP + FP)
    Recall    = TP / (TP + FN)
    Particularly useful when hotspot cells are rare (class imbalance).
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report,
)


class PredictiveAccuracyEvaluator:
    """
    Evaluate predictive hotspot accuracy against observed crime events.

    Parameters
    ----------
    stc : SpaceTimeCube
        The fitted space-time cube.
    forecast_grid : np.ndarray, shape (nrows, ncols)
        Predicted crime counts (or risk scores) for the evaluation period.
    actual_events : GeoDataFrame
        Point events that occurred during the evaluation period.
    """

    def __init__(
        self,
        stc,
        forecast_grid: np.ndarray,
        actual_events: gpd.GeoDataFrame,
    ):
        self.stc = stc
        self.forecast_grid = forecast_grid.copy()
        self.actual_events = actual_events.copy()

        # Bin actual events into the same grid
        self._actual_grid = self._bin_actual_events()

    # ------------------------------------------------------------------
    # Core metrics
    # ------------------------------------------------------------------

    def hit_rate(self, percentile: float = 80.0) -> float:
        """Fraction of actual crimes captured in top-*percentile* risk cells."""
        mask = self._hotspot_mask(percentile)
        actual = self._actual_grid
        total = actual.sum()
        if total == 0:
            return 0.0
        captured = actual[mask].sum()
        return float(captured / total)

    def pai(self, percentile: float = 80.0) -> float:
        """Predictive Accuracy Index at *percentile* threshold."""
        mask = self._hotspot_mask(percentile)
        hr = self.hit_rate(percentile)
        area_frac = mask.sum() / mask.size
        if area_frac == 0:
            return 0.0
        return float(hr / area_frac)

    def pei(self, percentile: float = 80.0) -> float:
        """Prediction Efficiency Index (PEI)."""
        mask = self._hotspot_mask(percentile)
        hr = self.hit_rate(percentile)
        area_frac = mask.sum() / mask.size
        denom = 1 - area_frac
        if denom == 0:
            return 0.0
        return float((hr - area_frac) / denom)

    def rmse(self) -> float:
        """Root Mean Square Error of per-cell count predictions."""
        diff = self.forecast_grid - self._actual_grid
        return float(np.sqrt((diff ** 2).mean()))

    def mae(self) -> float:
        """Mean Absolute Error of per-cell count predictions."""
        diff = self.forecast_grid - self._actual_grid
        return float(np.abs(diff).mean())

    def pearson_r(self) -> float:
        """Pearson correlation between predicted and actual cell counts."""
        pred = self.forecast_grid.ravel()
        actual = self._actual_grid.ravel()
        if pred.std() == 0 or actual.std() == 0:
            return 0.0
        return float(np.corrcoef(pred, actual)[0, 1])

    # ------------------------------------------------------------------
    # Sweep across thresholds
    # ------------------------------------------------------------------

    def pai_curve(
        self, percentiles: np.ndarray | None = None
    ) -> pd.DataFrame:
        """
        Compute PAI and Hit Rate across a range of area thresholds.
        Returns a DataFrame with columns: percentile, area_fraction, hit_rate, pai.
        """
        if percentiles is None:
            percentiles = np.arange(50, 100, 5)
        records = []
        for p in percentiles:
            mask = self._hotspot_mask(p)
            af = mask.sum() / mask.size
            hr = self.hit_rate(p)
            pai_val = hr / af if af > 0 else 0.0
            records.append(dict(percentile=p, area_fraction=af, hit_rate=hr, pai=pai_val))
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Full report
    # ------------------------------------------------------------------

    def report(self, percentile: float = 80.0) -> pd.Series:
        """Print and return a summary Series of all metrics."""
        metrics = {
            "hit_rate":   self.hit_rate(percentile),
            "pai":        self.pai(percentile),
            "pei":        self.pei(percentile),
            "rmse":       self.rmse(),
            "mae":        self.mae(),
            "pearson_r":  self.pearson_r(),
            "area_frac":  self._hotspot_mask(percentile).sum() / self._hotspot_mask(percentile).size,
            "n_actual":   int(self._actual_grid.sum()),
            "threshold_pct": percentile,
        }

        print("\n" + "=" * 55)
        print("  PREDICTIVE ACCURACY REPORT")
        print("=" * 55)
        print(f"  Hotspot threshold    : top {100-percentile:.0f}% of cells")
        print(f"  Area fraction        : {metrics['area_frac']:.3f} ({metrics['area_frac']*100:.1f}%)")
        print(f"  Hit Rate             : {metrics['hit_rate']:.3f}")
        print(f"  PAI                  : {metrics['pai']:.2f}  (>1 is better than random)")
        print(f"  PEI                  : {metrics['pei']:.3f}  (range -1 to 1)")
        print(f"  RMSE (per cell)      : {metrics['rmse']:.3f}")
        print(f"  MAE  (per cell)      : {metrics['mae']:.3f}")
        print(f"  Pearson r (pred/act) : {metrics['pearson_r']:.3f}")
        print(f"  Total actual events  : {metrics['n_actual']:,}")
        print("=" * 55 + "\n")

        return pd.Series(metrics)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _hotspot_mask(self, percentile: float) -> np.ndarray:
        """Boolean mask: True for cells in the top (100-percentile)% of risk."""
        threshold = np.percentile(self.forecast_grid, percentile)
        return self.forecast_grid >= threshold

    def _bin_actual_events(self) -> np.ndarray:
        """Bin actual_events into the same grid as the space-time cube."""
        stc = self.stc
        gdf = self.actual_events
        cs = stc.cell_size

        xs = gdf.geometry.x.values
        ys = gdf.geometry.y.values
        cols = np.floor((xs - stc._minx) / cs).astype(int).clip(0, stc.ncols - 1)
        rows = np.floor((ys - stc._miny) / cs).astype(int).clip(0, stc.nrows - 1)

        grid = np.zeros((stc.nrows, stc.ncols), dtype=float)
        for r, c in zip(rows, cols):
            grid[r, c] += 1
        return grid

    # ------------------------------------------------------------------
    # Compare models
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # ROC, Precision-Recall, Confusion Matrix
    # ------------------------------------------------------------------

    def roc_data(self, count_threshold: int = 1) -> dict:
        """
        ROC curve for binary hotspot classification.

        A cell is a true positive if actual count > count_threshold.
        The predicted score is the continuous forecast value.

        Returns
        -------
        dict: fpr, tpr, thresholds, auc_score
        """
        y_true  = (self._actual_grid.ravel() > count_threshold).astype(int)
        y_score = self.forecast_grid.ravel()
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        auc_score = auc(fpr, tpr)
        return dict(fpr=fpr, tpr=tpr, thresholds=thresholds, auc=float(auc_score))

    def precision_recall_data(self, count_threshold: int = 1) -> dict:
        """
        Precision-Recall curve.

        Returns
        -------
        dict: precision, recall, thresholds, average_precision
        """
        y_true  = (self._actual_grid.ravel() > count_threshold).astype(int)
        y_score = self.forecast_grid.ravel()
        prec, rec, thresholds = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        return dict(precision=prec, recall=rec, thresholds=thresholds, average_precision=float(ap))

    def confusion_matrix_data(
        self,
        percentile: float = 80.0,
        count_threshold: int = 1,
    ) -> dict:
        """
        Confusion matrix at a given forecast percentile threshold.

        Returns
        -------
        dict: cm (2×2 array), TP, FP, FN, TN, precision, recall, f1, accuracy
        """
        pred_hot  = self._hotspot_mask(percentile).ravel().astype(int)
        actual_hot = (self._actual_grid.ravel() > count_threshold).astype(int)
        cm = confusion_matrix(actual_hot, pred_hot)
        # cm[actual, predicted] — rows=actual, cols=predicted
        TN, FP, FN, TP = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        precision  = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall     = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1         = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy   = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
        return dict(
            cm=cm, TP=int(TP), FP=int(FP), FN=int(FN), TN=int(TN),
            precision=float(precision), recall=float(recall),
            f1=float(f1), accuracy=float(accuracy),
        )

    def full_validation_report(self, percentile: float = 80.0) -> dict:
        """Return all validation results in one dict."""
        basic   = {k: float(v) if isinstance(v, (np.floating, float)) else v
                   for k, v in self.report(percentile).items()}
        roc     = self.roc_data()
        pr      = self.precision_recall_data()
        cm      = self.confusion_matrix_data(percentile)
        return dict(basic=basic, roc=roc, pr=pr, cm=cm)

    # ------------------------------------------------------------------
    # Compare models
    # ------------------------------------------------------------------

    @staticmethod
    def compare_models(evaluators: dict, percentile: float = 80.0) -> pd.DataFrame:
        """
        Compare multiple models side-by-side.

        Parameters
        ----------
        evaluators : dict
            {model_name: PredictiveAccuracyEvaluator}

        Returns
        -------
        pd.DataFrame with model names as index and metrics as columns.
        """
        rows = {}
        for name, ev in evaluators.items():
            rows[name] = {
                "Hit Rate":  ev.hit_rate(percentile),
                "PAI":       ev.pai(percentile),
                "PEI":       ev.pei(percentile),
                "RMSE":      ev.rmse(),
                "MAE":       ev.mae(),
                "Pearson r": ev.pearson_r(),
            }
        df = pd.DataFrame(rows).T
        print("\nModel Comparison:")
        print(df.round(4).to_string())
        return df
