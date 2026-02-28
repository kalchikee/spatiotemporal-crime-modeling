# Spatiotemporal Crime Modeling
### Space-Time Cube · Predictive Hotspot Analysis · Interactive Web Map

A full spatiotemporal crime analysis pipeline built in Python — from raw event data through statistical modeling to an interactive Streamlit web map.  Covers all 8 phases of advanced spatial crime analysis.

---

## Live Dashboard Preview

10 switchable map layers · 7 analysis tabs · Time slider · Animated Gi* · Real Open Data support

**Layers:** Crime Density · KDE · Space-Time Cube · Getis-Ord Gi* · Emerging Hotspots · LISA Clusters · ST-DBSCAN · GWR Coefficients · Forecast Risk · Forecast Uncertainty

**Tabs:** Time Series · Exploratory Analysis · Hotspot Categories · GWR · Full Validation · Model Comparison · Data & Config

---

## Methods Implemented

### Phase 1 — Data
- Synthetic crime generator (5 spatiotemporal hotspots with configurable trends)
- **NYC Open Data** (NYPD Complaint API — live download)
- **Chicago Open Data** (Crime incidents API — live download)
- Local CSV / GeoPackage loader

### Phase 2 — Exploratory Spatial Analysis
- **Kernel Density Estimation (KDE)** — Gaussian KDE on regular grid
- **Global Moran's I** — tests for overall spatial autocorrelation
- **Local Moran's I (LISA)** — HH / LH / LL / HL cluster classification per cell
- **Ripley's K / L function** — clustering at multiple distance scales with Monte Carlo CSR envelope
- **ADF + KPSS stationarity tests** — per cell time series

### Phase 3 — Space-Time Cube
- Regular grid (configurable cell size, monthly bins)
- **Emerging Hotspot Analysis** — 9 ESRI-style categories (New, Consecutive, Intensifying, Persistent, Diminishing, Sporadic, Oscillating, Historical, No Pattern)
- **Mann-Kendall trend test** — per cell Z-score time series

### Phase 4 — Temporal Modeling
- **ARIMA** — stationarity-aware (ADF test selects differencing order d automatically)
- **Prophet** (optional) — Facebook's time-series model with yearly seasonality
- **Simple Exponential Smoothing** — fallback for sparse cells
- **95% Confidence Intervals** on all forecasts

### Phase 5 — Spatial Regression
- **OLS Baseline** — no spatial structure, benchmark
- **Spatial Lag Model (SAR)** — y = ρWy + Xβ + ε, fitted via GM 2SLS
- **Spatial Error Model (SEM)** — u = λWu + ε, captures residual spatial correlation
- **GWR (Geographically Weighted Regression)** — spatially varying coefficients via Gaussian kernel, GCV bandwidth selection

### Phase 6 — Predictive Risk Surface
- Per-cell forecast assembled into spatial risk grid
- Gaussian spatial smoothing
- Uncertainty map (95% CI width)

### Phase 7 — Validation
- **Hit Rate** — % of actual crimes captured in predicted hotspot zone
- **PAI (Predictive Accuracy Index)** — hit rate adjusted for area coverage
- **PEI (Prediction Efficiency Index)**
- **ROC Curve + AUC**
- **Confusion Matrix** — TP / FP / FN / TN
- **Precision-Recall Curve**
- **RMSE / MAE** — per cell count errors
- Multi-model comparison table

### Phase 8 — Interactive Web Map
- Plotly Mapbox (carto-positron, no API token required)
- Time slider for Space-Time Cube, Gi*, and LISA layers
- Animated Gi* (play/pause button, frame slider)
- Dark theme dashboard

---

## Project Structure

```
├── app.py                  # Streamlit web map (main entry point)
├── run_pipeline.py         # Batch pipeline → PNG outputs
├── requirements.txt
├── src/
│   ├── data_loader.py      # Synthetic crime data generator
│   ├── data_sources.py     # NYC Open Data + Chicago API connectors
│   ├── space_time_cube.py  # 3D grid (time × rows × cols)
│   ├── exploratory.py      # KDE, LISA, Ripley's K, stationarity tests
│   ├── hotspot_analysis.py # Getis-Ord Gi* + Mann-Kendall + ESRI categories
│   ├── clustering.py       # ST-DBSCAN spatiotemporal clustering
│   ├── forecasting.py      # ARIMA / Prophet / SES with CI bands
│   ├── gwr_model.py        # GWR — spatially varying coefficients
│   ├── spatial_lag.py      # OLS, SAR, SEM, Global Moran's I
│   ├── evaluation.py       # PAI, Hit Rate, ROC, Confusion Matrix, PR
│   ├── visualization.py    # Static matplotlib plots (for batch pipeline)
│   └── web_utils.py        # Coordinate conversion, GeoJSON, Plotly helpers
├── outputs/
│   ├── figures/            # PNG plots from run_pipeline.py
│   └── predictions/        # Forecast grids, CSVs
└── data/                   # Place real data files here
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch interactive web map
streamlit run app.py

# 3. (Optional) Run batch pipeline — saves PNGs to outputs/figures/
python run_pipeline.py

# 4. (Optional) Prophet time-series support
pip install prophet
# Then check "Use Prophet" in the sidebar
```

Open **http://localhost:8501** in your browser.

---

## Real Data

Select **NYC Open Data** or **Chicago Open Data** in the sidebar to download live crime records. No API key required — uses public Socrata endpoints.

To use your own CSV:
```python
from src.data_sources import load_from_file
gdf = load_from_file("your_crimes.csv", lat_col="latitude", lon_col="longitude")
```

Minimum required columns: `latitude`, `longitude`, `datetime` (or equivalent).

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `geopandas`, `shapely`, `pyproj` | Spatial data handling |
| `libpysal`, `esda` | Spatial weights, LISA, Gi* |
| `spreg` | Spatial lag / error regression |
| `statsmodels` | ARIMA, SES, ADF/KPSS |
| `scikit-learn` | KDE, BallTree, evaluation metrics |
| `streamlit` | Web dashboard |
| `plotly` | Interactive maps and charts |
| `requests` | Open Data API calls |

---

## Key Concepts

**PAI > 1** means your model identifies hotspots at a higher crime rate than random selection. A PAI of 5 means the predicted zone captures crimes at 5× the rate of chance.

**Spatial lag ρ** quantifies crime contagion — how much crime in one cell is driven by crime in neighbouring cells.

**GWR** reveals where your global regression assumptions break down — cells with low local R² suggest unmeasured local factors (policing intensity, social conditions) that a global model misses.
