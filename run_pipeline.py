"""
run_pipeline.py
===============
Full Spatiotemporal Crime Modeling Pipeline
============================================

Stages
------
1.  Generate / load synthetic crime data
2.  Build Space-Time Cube (STC)
3.  Getis-Ord Gi* — local spatial autocorrelation over time
4.  Emerging Hotspot Analysis — Mann-Kendall + ESRI classification
5.  ST-DBSCAN — spatiotemporal clustering of raw events
6.  Time-series forecasting per cell (ARIMA / SES)
7.  Spatial Lag Model (SAR) — crime count regression with spatial dependence
8.  Predictive Accuracy Evaluation — PAI, Hit Rate, RMSE
9.  Save all figures to outputs/figures/

Usage
-----
    python run_pipeline.py

Configuration is set via the CONFIG dict below.
"""

import os
import sys
import warnings
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for saving figures
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── Project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from src.data_loader      import SyntheticCrimeGenerator
from src.space_time_cube  import SpaceTimeCube
from src.hotspot_analysis import EmergingHotspotAnalyzer
from src.clustering       import STDBSCAN
from src.forecasting      import SpaceTimeForecaster
from src.spatial_lag      import OLSBaseline, SpatialLagModel, morans_i
from src.evaluation       import PredictiveAccuracyEvaluator
from src import visualization as viz
import libpysal.weights as lps_weights

# ── Configuration ────────────────────────────────────────────────────────────
CONFIG = dict(
    # Data
    n_events       = 12_000,
    start_date     = "2022-01-01",
    end_date       = "2023-12-31",
    bbox           = (0, 0, 20_000, 20_000),
    crs            = "EPSG:32616",
    seed           = 42,

    # Space-Time Cube
    cell_size      = 500,       # metres per cell side
    freq           = "ME",      # monthly bins  (pandas period frequency)

    # Hotspot analysis
    gi_alpha       = 0.05,
    persistent_pct = 0.90,

    # ST-DBSCAN
    eps_spatial    = 1_500,     # metres
    eps_temporal   = 45 * 24 * 3600,  # 45 days in seconds
    min_samples    = 8,

    # Forecasting
    train_months   = 20,        # use first 20 months for training
    n_forecast     = 4,         # forecast next 4 months
    smooth_sigma   = 0.8,

    # Evaluation
    hotspot_pct    = 80,        # top 20% of cells = hotspot zone

    # Output
    fig_dir        = "outputs/figures",
)

FIG = CONFIG["fig_dir"]
os.makedirs(FIG, exist_ok=True)
os.makedirs("outputs/predictions", exist_ok=True)


# ── Helpers ──────────────────────────────────────────────────────────────────

def section(title: str):
    bar = "=" * 60
    print(f"\n{bar}\n  {title}\n{bar}")


def save(fig, name: str):
    path = os.path.join(FIG, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ============================================================================
# STAGE 1 — Data generation
# ============================================================================

section("STAGE 1 — Generating Synthetic Crime Data")
t0 = time.time()

gen = SyntheticCrimeGenerator(
    n_events   = CONFIG["n_events"],
    start_date = CONFIG["start_date"],
    end_date   = CONFIG["end_date"],
    bbox       = CONFIG["bbox"],
    crs        = CONFIG["crs"],
    seed       = CONFIG["seed"],
)
crime_gdf = gen.generate()

print(f"  Events        : {len(crime_gdf):,}")
print(f"  Date range    : {crime_gdf['datetime'].min().date()} → {crime_gdf['datetime'].max().date()}")
print(f"  Crime types   : {crime_gdf['crime_type'].value_counts().to_dict()}")
print(f"  Time          : {time.time()-t0:.1f}s")

fig, _ = viz.plot_crime_map(crime_gdf, title="Synthetic Crime Events — Hexbin Density")
save(fig, "01_crime_density_map.png")


# ============================================================================
# STAGE 2 — Space-Time Cube
# ============================================================================

section("STAGE 2 — Building Space-Time Cube")
t0 = time.time()

stc = SpaceTimeCube(cell_size=CONFIG["cell_size"], bbox=CONFIG["bbox"], freq=CONFIG["freq"])
stc.fit(crime_gdf)

print(f"  Dimensions    : {stc.T} × {stc.nrows} × {stc.ncols}")
print(f"  Total events  : {int(stc.cube.sum()):,}")
print(f"  Max per cell  : {int(stc.cube.max())}")
print(f"  Time          : {time.time()-t0:.1f}s")

fig = viz.plot_space_time_cube(stc, n_cols=6)
save(fig, "02_space_time_cube.png")


# ============================================================================
# STAGE 3 — Getis-Ord Gi* over time
# ============================================================================

section("STAGE 3 — Getis-Ord Gi* Spatial Autocorrelation")
t0 = time.time()

eha = EmergingHotspotAnalyzer(stc, alpha=CONFIG["gi_alpha"], persistent_pct=CONFIG["persistent_pct"])
eha.fit(verbose=True)

print(f"  Time          : {time.time()-t0:.1f}s")

# Global Moran's I for the last time step
w = lps_weights.lat2W(stc.nrows, stc.ncols, rook=False)
last_counts = stc.flat_slice(-1)
mi = morans_i(last_counts, w)
print(f"\n  Moran's I (last step): I={mi['I']:.4f}  Z={mi['z']:.2f}  p={mi['p_value']:.4f}")
if mi["p_value"] < 0.05:
    print("  → Significant positive spatial autocorrelation (crime clusters in space)")

fig = viz.plot_gi_star(eha.z_cube, stc.time_labels, n_cols=6)
save(fig, "03_gi_star_timeseries.png")


# ============================================================================
# STAGE 4 — Emerging Hotspot Analysis
# ============================================================================

section("STAGE 4 — Emerging Hotspot Analysis")

category_gdf = eha.to_geodataframe()
summary = eha.summary()

print("\n  Category distribution:")
print(summary.to_string())

# Save category GDF
category_gdf.drop(columns="geometry").to_csv("outputs/predictions/emerging_hotspot_categories.csv", index=False)
print("\n  Saved category table → outputs/predictions/emerging_hotspot_categories.csv")

fig, _ = viz.plot_emerging_hotspots(category_gdf)
save(fig, "04_emerging_hotspots.png")

fig = viz.plot_category_distribution(summary)
save(fig, "04b_category_distribution.png")


# ============================================================================
# STAGE 5 — ST-DBSCAN Clustering
# ============================================================================

section("STAGE 5 — ST-DBSCAN Spatiotemporal Clustering")
t0 = time.time()

# Work on a random sample for speed (full dataset can be slow)
sample_size = min(5_000, len(crime_gdf))
sample_gdf = crime_gdf.sample(sample_size, random_state=CONFIG["seed"]).reset_index(drop=True)

stdbscan = STDBSCAN(
    eps_spatial  = CONFIG["eps_spatial"],
    eps_temporal = CONFIG["eps_temporal"],
    min_samples  = CONFIG["min_samples"],
)
clustered_gdf = stdbscan.fit_transform(sample_gdf)
cluster_summary = stdbscan.cluster_summary(sample_gdf)

print(f"\n  Cluster summary (top 10 by size):")
print(cluster_summary.sort_values("n_events", ascending=False).head(10).to_string(index=False))
print(f"\n  Time          : {time.time()-t0:.1f}s")

cluster_summary.to_csv("outputs/predictions/stdbscan_clusters.csv", index=False)

fig = viz.plot_stdbscan_clusters(clustered_gdf)
save(fig, "05_stdbscan_clusters.png")


# ============================================================================
# STAGE 6 — Time-Series Forecasting per Cell
# ============================================================================

section("STAGE 6 — ARIMA / SES Forecasting per Cell")
t0 = time.time()

train_t_end = CONFIG["train_months"]
n_fc        = CONFIG["n_forecast"]

forecaster = SpaceTimeForecaster(
    stc          = stc,
    train_t_end  = train_t_end,
    smooth_sigma = CONFIG["smooth_sigma"],
)
fc_cube = forecaster.fit_predict(n_steps=n_fc, verbose=True)

print(f"  Forecast period: {forecaster.future_period_labels()}")
print(f"  Time           : {time.time()-t0:.1f}s")

# Aggregate forecast over all forecast steps
fc_total = fc_cube.sum(axis=0)
np.save("outputs/predictions/forecast_grid.npy", fc_total)

# Actual events in the test period (steps train_t_end … stc.T-1)
test_cube = stc.cube[train_t_end:]
actual_test_grid = test_cube.sum(axis=0)

# Hotspot mask
hotspot_mask = forecaster.hotspot_mask(step=0, percentile=CONFIG["hotspot_pct"])

fig = viz.plot_forecast_surface(
    forecast_grid = fc_cube[0],
    actual_grid   = stc.get_time_slice(train_t_end) if train_t_end < stc.T else None,
    hotspot_mask  = hotspot_mask,
    period_label  = str(forecaster.future_period_labels()[0]),
)
save(fig, "06_forecast_surface.png")

# Temporal profile for top 3 active cells
active = stc.active_cells(min_total=10)
if active:
    totals = stc.total_counts()
    top3 = sorted(active, key=lambda rc: totals[rc[0], rc[1]], reverse=True)[:3]
    fig = viz.plot_temporal_profile(stc, top3, forecast_cube=fc_cube, train_t_end=train_t_end)
    save(fig, "06b_temporal_profile.png")


# ============================================================================
# STAGE 7 — Spatial Lag Model
# ============================================================================

section("STAGE 7 — Spatial Regression Models")
t0 = time.time()

# OLS baseline
ols = OLSBaseline()
try:
    ols.fit(stc, train_t_end)
    ols_pred = sum(ols.predict(stc, t) for t in range(train_t_end, min(train_t_end + n_fc, stc.T)))
except Exception as e:
    print(f"  OLS warning: {e}")
    ols_pred = None

# Spatial Lag Model
slm = None
try:
    slm = SpatialLagModel()
    slm.fit(stc, train_t_end)
    slm_pred = sum(slm.predict(stc, t) for t in range(train_t_end, min(train_t_end + n_fc, stc.T)))
    print(f"  ρ (spatial autoregressive) = {slm.rho_:.4f}")
    if abs(slm.rho_) > 0.1:
        print("  → Strong spatial dependence detected in crime counts")
except Exception as e:
    print(f"  SpatialLag skipped (spreg may not be installed): {e}")
    slm_pred = None

print(f"  Time          : {time.time()-t0:.1f}s")


# ============================================================================
# STAGE 8 — Predictive Accuracy Evaluation
# ============================================================================

section("STAGE 8 — Predictive Accuracy Evaluation")

# Actual events during test period
test_events = crime_gdf[
    crime_gdf["datetime"] >= stc.time_labels[train_t_end].to_timestamp()
].copy() if train_t_end < stc.T else crime_gdf.copy()

evaluators = {}

# ARIMA forecast evaluation
ev_arima = PredictiveAccuracyEvaluator(stc, fc_total, test_events)
arima_report = ev_arima.report(percentile=CONFIG["hotspot_pct"])
evaluators["ARIMA"] = ev_arima

# OLS evaluation
if ols_pred is not None:
    ev_ols = PredictiveAccuracyEvaluator(stc, ols_pred, test_events)
    _ = ev_ols.report(percentile=CONFIG["hotspot_pct"])
    evaluators["OLS"] = ev_ols

# Spatial Lag evaluation
if slm_pred is not None:
    ev_slm = PredictiveAccuracyEvaluator(stc, slm_pred, test_events)
    _ = ev_slm.report(percentile=CONFIG["hotspot_pct"])
    evaluators["Spatial Lag"] = ev_slm

# Naive baseline: uniform random prediction
uniform_pred = np.ones_like(fc_total) * fc_total.mean()
ev_naive = PredictiveAccuracyEvaluator(stc, uniform_pred, test_events)
evaluators["Naive (Uniform)"] = ev_naive

# PAI curve for ARIMA
pai_df = ev_arima.pai_curve(np.arange(50, 100, 5))
pai_df.to_csv("outputs/predictions/pai_curve.csv", index=False)
fig = viz.plot_pai_curve(pai_df, model_name="ARIMA Per-Cell Forecast")
save(fig, "07_pai_curve.png")

# Model comparison
if len(evaluators) > 1:
    comparison_df = PredictiveAccuracyEvaluator.compare_models(evaluators, CONFIG["hotspot_pct"])
    comparison_df.to_csv("outputs/predictions/model_comparison.csv")
    fig = viz.plot_model_comparison(comparison_df)
    save(fig, "08_model_comparison.png")


# ============================================================================
# Final summary
# ============================================================================

section("PIPELINE COMPLETE")
print(f"\n  All figures saved to : {FIG}/")
print(f"  Predictions saved to : outputs/predictions/")
print("\n  Key findings:")
print(f"    Space-time cube      : {stc.T}×{stc.nrows}×{stc.ncols} ({stc.T*stc.nrows*stc.ncols:,} voxels)")
print(f"    ST-DBSCAN clusters   : {stdbscan.n_clusters_}")
print(f"    ARIMA PAI @ top 20%  : {arima_report['pai']:.2f}")
print(f"    ARIMA Hit Rate       : {arima_report['hit_rate']*100:.1f}%")
print(f"    Moran's I (last step): {mi['I']:.4f}  (p={mi['p_value']:.4f})")
if slm is not None:
    print(f"    Spatial lag ρ        : {slm.rho_:.4f}")
print("\n  Run completed successfully.\n")
