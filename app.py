"""
app.py — Spatiotemporal Crime Intelligence Dashboard  (Complete Edition)
=========================================================================
All 8 phases: KDE · Global+Local Moran's I · Ripley's K · Space-Time Cube ·
Getis-Ord Gi* · Emerging Hotspots · ST-DBSCAN · ARIMA · Prophet · GWR ·
OLS/SAR/SEM · ROC · Confusion Matrix · PAI · Time-slider Animation

Run:  streamlit run app.py
"""

import os, sys, warnings, json
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import libpysal.weights as lps_weights

from src.data_loader      import SyntheticCrimeGenerator
from src.data_sources     import load_nyc_crimes, load_chicago_crimes
from src.space_time_cube  import SpaceTimeCube
from src.hotspot_analysis import EmergingHotspotAnalyzer
from src.exploratory      import (compute_kde_on_grid, compute_lisa_all_steps,
                                   ripley_kl, test_stationarity, stationarity_report)
from src.clustering       import STDBSCAN
from src.forecasting      import SpaceTimeForecaster
from src.gwr_model        import fit_gwr_on_stc
from src.spatial_lag      import OLSBaseline, SpatialLagModel, SpatialErrorModel, morans_i
from src.evaluation       import PredictiveAccuracyEvaluator
from src.web_utils        import (localize_gdf, cells_to_geojson, grid_values_to_series,
                                   category_colorscale, base_mapbox_layout,
                                   CATEGORY_COLORS, MAP_CENTER)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Crime Intelligence", page_icon="🗺️",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
.main .block-container{padding-top:.7rem;padding-bottom:.3rem}
section[data-testid="stSidebar"]{background:#0d1117}
section[data-testid="stSidebar"] *{color:#c9d1d9!important}
section[data-testid="stSidebar"] h1,section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3{color:#58a6ff!important}
[data-testid="metric-container"]{background:#161b22;border:1px solid #30363d;
  border-radius:8px;padding:.5rem .7rem}
[data-testid="metric-container"] label{color:#58a6ff!important;font-size:.78rem}
[data-testid="stMetricValue"]{color:#f0f6fc!important}
hr{border-color:#21262d;margin:.5rem 0}
</style>""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
TRAIN_MONTHS = 12   # 1 year of training (was 20) — fewer ARIMA fits
N_FORECAST   = 2    # 2 months ahead (was 4)
CELL_SIZE    = 2_000   # 2 km cells → 10×10=100 cells — 4× faster than 1km
BBOX         = (0, 0, 20_000, 20_000)
N_EVENTS     = 4_000   # reduced for faster KDE / ST-DBSCAN on free tier
MAP_H        = 565

LAYERS = [
    "🔴  Crime Density",
    "🔥  KDE Density",
    "📦  Space-Time Cube",
    "⚡  Getis-Ord Gi*",
    "🏷️  Emerging Hotspots",
    "🗺️  LISA Clusters",
    "🔵  ST-DBSCAN Clusters",
    "📐  GWR Coefficients",
    "🔮  Forecast Risk",
    "⚠️  Forecast Uncertainty",
]


# ════════════════════════════════════════════════════════════════════════════════
# CACHED COMPUTATIONS
# ════════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _gen_synthetic():
    return SyntheticCrimeGenerator(n_events=N_EVENTS, bbox=BBOX, seed=42).generate()

@st.cache_data(show_spinner=False)
def _nyc(limit, start):   return load_nyc_crimes(limit=limit, start_date=start)
@st.cache_data(show_spinner=False)
def _chi(limit, start):   return load_chicago_crimes(limit=limit, start_date=start)

@st.cache_data(show_spinner=False)
def _stc(_gdf, use_bbox):
    stc = SpaceTimeCube(cell_size=CELL_SIZE, bbox=BBOX if use_bbox else None, freq="M")
    stc.fit(_gdf)
    return stc

@st.cache_data(show_spinner=False)
def _eha(_stc):
    e = EmergingHotspotAnalyzer(_stc, alpha=0.05); e.fit(verbose=False); return e

@st.cache_data(show_spinner=False)
def _kde(_gdf, _stc):    return compute_kde_on_grid(_gdf, _stc)

@st.cache_data(show_spinner=False)
def _lisa_all(_stc):
    w = lps_weights.lat2W(_stc.nrows, _stc.ncols, rook=False)
    return compute_lisa_all_steps(_stc, w, alpha=0.05)

@st.cache_data(show_spinner=False)
def _stdb(_gdf):
    s = _gdf.sample(min(3000, len(_gdf)), random_state=42).reset_index(drop=True)
    return STDBSCAN(eps_spatial=2000, eps_temporal=45*24*3600, min_samples=8).fit_transform(s)

@st.cache_data(show_spinner=False)
def _forecast(_stc, use_prophet):
    fc = SpaceTimeForecaster(_stc, train_t_end=TRAIN_MONTHS, smooth_sigma=0.8, use_prophet=use_prophet)
    cube = fc.fit_predict(n_steps=N_FORECAST, verbose=False)
    return fc, cube

@st.cache_data(show_spinner=False)
def _ols(_stc):
    try:
        m = OLSBaseline(); m.fit(_stc, TRAIN_MONTHS)
        p = sum(m.predict(_stc, t) for t in range(TRAIN_MONTHS, min(TRAIN_MONTHS+N_FORECAST, _stc.T)))
        return m, p
    except Exception: return None, None

@st.cache_data(show_spinner=False)
def _slm(_stc):
    try:
        m = SpatialLagModel(); m.fit(_stc, TRAIN_MONTHS)
        p = sum(m.predict(_stc, t) for t in range(TRAIN_MONTHS, min(TRAIN_MONTHS+N_FORECAST, _stc.T)))
        return m, p
    except Exception: return None, None

@st.cache_data(show_spinner=False)
def _sem(_stc):
    try:
        m = SpatialErrorModel(); m.fit(_stc, TRAIN_MONTHS)
        p = sum(m.predict(_stc, t) for t in range(TRAIN_MONTHS, min(TRAIN_MONTHS+N_FORECAST, _stc.T)))
        return m, p
    except Exception: return None, None

@st.cache_data(show_spinner=False)
def _gwr(_stc):
    try: return fit_gwr_on_stc(_stc, t=-1, bandwidth=None)
    except Exception: return None

@st.cache_data(show_spinner=False)
def _ripley(_gdf, bbox):
    s = _gdf.sample(min(800, len(_gdf)), random_state=42)
    return ripley_kl(s, bbox=bbox, n_simulations=9)

@st.cache_data(show_spinner=False)
def _stat_df(_stc): return stationarity_report(_stc, min_total=5)

@st.cache_data(show_spinner=False)
def _cells_wgs84(_stc):
    c = localize_gdf(_stc.cell_gdf); return c, cells_to_geojson(c)

@st.cache_data(show_spinner=False)
def _gdf_wgs84(_gdf):  return localize_gdf(_gdf)

@st.cache_data(show_spinner=False)
def _morans(_stc):
    w = lps_weights.lat2W(_stc.nrows, _stc.ncols, rook=False)
    return morans_i(_stc.flat_slice(-1), w)

@st.cache_data(show_spinner=False)
def _animated_gi(z_list, labels, geojson_str):
    geojson = json.loads(geojson_str)
    z = np.array(z_list)
    vlim = max(abs(z.min()), abs(z.max()), 2.0)
    T, nR, nC = z.shape
    idxs = [str(i) for i in range(nR*nC)]
    frames = [go.Frame(
        data=[go.Choroplethmapbox(geojson=geojson, locations=idxs,
                                   z=z[t].ravel().tolist(), colorscale="RdBu_r",
                                   zmin=-vlim, zmax=vlim,
                                   marker_opacity=0.75, marker_line_width=0)],
        name=labels[t]) for t in range(T)]
    fig = go.Figure(data=frames[0].data, frames=frames,
        layout=go.Layout(
            **base_mapbox_layout(height=MAP_H),
            updatemenus=[dict(type="buttons", showactive=False, y=1.08, x=0.5, xanchor="center",
                buttons=[dict(label="▶ Play", method="animate",
                              args=[None, dict(frame=dict(duration=700, redraw=True), fromcurrent=True)]),
                         dict(label="⏸ Pause", method="animate",
                              args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False))])])],
            sliders=[dict(steps=[dict(method="animate",
                                     args=[[f.name], dict(mode="immediate",frame=dict(duration=700,redraw=True))],
                                     label=f.name) for f in frames],
                         x=0.05, len=0.9, y=0, yanchor="top",
                         currentvalue=dict(prefix="Month: ", visible=True))]))
    return fig


# ════════════════════════════════════════════════════════════════════════════════
# MAP BUILDERS
# ════════════════════════════════════════════════════════════════════════════════

def _choro(geojson, vals, colorscale, zmin, zmax, hover, name, opacity=0.76, **cbkw):
    idxs = [str(i) for i in range(len(vals))]
    return go.Choroplethmapbox(
        geojson=geojson, locations=idxs, z=list(vals),
        colorscale=colorscale, zmin=zmin, zmax=zmax,
        marker_opacity=opacity, marker_line_width=0,
        colorbar=dict(thickness=12, **cbkw),
        hovertemplate="%{customdata}<extra></extra>",
        customdata=list(hover), name=name)

def _lay(**kw): return base_mapbox_layout(height=MAP_H, **kw)

def _dk(**kw): return dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117",
                            font_color="#c9d1d9", margin=dict(t=38,b=36,l=8,r=8), **kw)

def fig_density(gdf_w):
    fig = go.Figure(go.Densitymapbox(lat=gdf_w.geometry.y, lon=gdf_w.geometry.x,
        radius=16, colorscale="YlOrRd", colorbar=dict(title="Density", thickness=12), name="Events"))
    fig.update_layout(**_lay()); return fig

def fig_kde(kde, stc, gj):
    _, vals = grid_values_to_series(kde, stc)
    hover = [f"KDE: {v:.5f}" for v in vals]
    fig = go.Figure(_choro(gj, vals, "plasma", 0, max(vals) or 1, hover, "KDE", title="KDE\nDensity"))
    fig.update_layout(**_lay()); return fig

def fig_stc(stc, t, gj):
    _, vals = grid_values_to_series(stc.get_time_slice(t), stc)
    hover = [f"{stc.time_labels[t]}  Count: {int(v)}" for v in vals]
    fig = go.Figure(_choro(gj, vals, "YlOrRd", 0, float(stc.cube.max()) or 1, hover, "Count", title="Count"))
    fig.update_layout(**_lay()); return fig

def fig_gi(eha, t, gj):
    zf   = eha.get_z_slice(t).ravel()
    vlim = max(abs(zf.min()), abs(zf.max()), 2.5)
    hover = [f"{eha.stc.time_labels[t]}  Z={v:.2f}  {'🔴' if v>1.96 else '🔵' if v<-1.96 else '⬜'}" for v in zf]
    fig = go.Figure(_choro(gj, zf, "RdBu_r", -vlim, vlim, hover, "Gi* Z",
                           title="Gi* Z", tickvals=[-2,-1.96,0,1.96,2],
                           ticktext=["−2","−1.96","0","+1.96","+2"]))
    fig.update_layout(**_lay()); return fig

def fig_emerging(eha, gj):
    cats = eha.category_grid.ravel()
    cs, enc, tvs, tts = category_colorscale(list(cats))
    hover = [c.replace("_"," ").title() for c in cats]
    fig = go.Figure(go.Choroplethmapbox(
        geojson=gj, locations=[str(i) for i in range(len(enc))], z=enc,
        colorscale=cs, zmin=0, zmax=max(enc)+1, marker_opacity=0.82,
        marker_line_width=0.4, marker_line_color="rgba(0,0,0,0.12)",
        colorbar=dict(title="Pattern", thickness=14, tickvals=tvs, ticktext=tts, tickfont=dict(size=8)),
        hovertemplate="%{customdata}<extra></extra>", customdata=hover, name="Category"))
    fig.update_layout(**_lay()); return fig

def fig_lisa(lisa_all, t, gj):
    res = lisa_all[t]
    q   = [float(x) for x in res["quadrant"]]
    qc  = {0:"#e8e8e8",1:"#e53935",2:"#fb8c00",3:"#1e88e5",4:"#8e24aa"}
    n   = int(max(q))+1
    cs  = []
    for i in range(n):
        cs += [[i/n, qc.get(i,"#aaa")],[(i+1)/n, qc.get(i,"#aaa")]]
    hover = [f"LISA: {res['labels'][i]}<br>I={res['Is'][i]:.3f}" for i in range(len(q))]
    fig = go.Figure(go.Choroplethmapbox(
        geojson=gj, locations=[str(i) for i in range(len(q))], z=q,
        colorscale=cs, zmin=0, zmax=n, marker_opacity=0.80, marker_line_width=0,
        colorbar=dict(title="LISA", thickness=14,
                      tickvals=[i+0.5 for i in range(n)],
                      ticktext=["Not Sig.","HH Hotspot","LH Outlier","LL Coldspot","HL Outlier"][:n],
                      tickfont=dict(size=8)),
        hovertemplate="%{customdata}<extra></extra>", customdata=hover, name="LISA"))
    fig.update_layout(**_lay()); return fig

def fig_stdb(cdf):
    gw = localize_gdf(cdf)
    nc = int(gw["cluster"].max())+1
    pal = px.colors.qualitative.Tab10
    fig = go.Figure()
    ns = gw[gw["cluster"]==-1]
    if len(ns): fig.add_trace(go.Scattermapbox(lat=ns.geometry.y, lon=ns.geometry.x,
        mode="markers", marker=dict(size=3,color="rgba(120,120,120,0.3)"), name="Noise", hoverinfo="skip"))
    for cid in range(min(nc,10)):
        s = gw[gw["cluster"]==cid]
        if not len(s): continue
        fig.add_trace(go.Scattermapbox(lat=s.geometry.y, lon=s.geometry.x, mode="markers",
            marker=dict(size=6,color=pal[cid%10],opacity=0.75), name=f"Cluster {cid} (n={len(s)})"))
    fig.update_layout(**_lay()); return fig

def fig_gwr_coef(gwr, stc, gj, idx, name):
    if gwr is None:
        return go.Figure().add_annotation(text="GWR not available",showarrow=False).update_layout(**_lay())
    if idx == "r2":
        g = gwr.local_r2_.reshape(stc.nrows, stc.ncols)
        cmap, zmin, zmax = "viridis", 0, 1
    else:
        g = gwr.to_grid(int(idx), stc)
        vlim = max(abs(g.min()), abs(g.max()), 0.01)
        cmap, zmin, zmax = "RdBu_r", -vlim, vlim
    _, vals = grid_values_to_series(g, stc)
    hover = [f"{name}: {v:.4f}" for v in vals]
    fig = go.Figure(_choro(gj, vals, cmap, zmin, zmax, hover, name, title=name))
    fig.update_layout(**_lay()); return fig

def fig_forecast(fc, lo, hi, stc, gj, step, ci=False):
    grid = (hi[step]-lo[step]) if ci else fc[step]
    cmap = "Oranges" if ci else "plasma"
    _, vals = grid_values_to_series(grid, stc)
    hover = [f"Pred: {fc[step].ravel()[i]:.2f}  CI:[{lo[step].ravel()[i]:.2f},{hi[step].ravel()[i]:.2f}]"
             for i in range(len(vals))]
    fig = go.Figure(_choro(gj, vals, cmap, 0, max(vals) or .01, hover,
                           "CI Width" if ci else "Forecast", title="CI Width" if ci else "Forecast"))
    fig.update_layout(**_lay()); return fig


# ── Plotly chart helpers ───────────────────────────────────────────────────────

def ts_fig(stc, fc, lo, hi, row, col):
    obs  = stc.get_cell_timeseries(row, col)
    xlbl = [str(p) for p in stc.time_labels]
    fig  = go.Figure()
    fig.add_trace(go.Bar(x=xlbl, y=obs, name="Observed", marker_color="#3a86ff", opacity=0.85))
    if fc is not None:
        fl   = [str(stc.time_labels[-1]+i+1) for i in range(fc.shape[0])]
        fv   = fc[:, row, col]
        lv   = lo[:, row, col] if lo is not None else fv
        hv   = hi[:, row, col] if hi is not None else fv
        fig.add_trace(go.Bar(x=fl, y=fv, name="Forecast", marker_color="#e84855", opacity=0.85))
        fig.add_trace(go.Scatter(x=fl+fl[::-1], y=list(hv)+list(lv[::-1]),
            fill="toself", fillcolor="rgba(232,72,85,0.15)",
            line=dict(color="rgba(0,0,0,0)"), name="95% CI"))
        fig.add_vline(x=TRAIN_MONTHS-.5, line_dash="dash", line_color="#ffa500",
                      annotation_text="Train/Test", annotation_font_color="#ffa500")
    fig.update_layout(title=f"Cell ({row},{col})", xaxis_tickangle=-40,
                      height=290, legend=dict(orientation="h", y=1.1), **_dk())
    return fig

def roc_fig(d):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["fpr"],y=d["tpr"],mode="lines",
        name=f"ROC AUC={d['auc']:.3f}",line=dict(color="#e84855",width=2)))
    fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",name="Random",
        line=dict(color="#555",dash="dash",width=1)))
    fig.update_layout(title="ROC Curve",xaxis_title="FPR",yaxis_title="TPR",height=320,**_dk())
    return fig

def pr_fig(d):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["recall"],y=d["precision"],mode="lines",
        name=f"PR  AP={d['average_precision']:.3f}",line=dict(color="#58a6ff",width=2)))
    fig.update_layout(title="Precision-Recall",xaxis_title="Recall",yaxis_title="Precision",height=320,**_dk())
    return fig

def pai_fig(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["area_fraction"]*100,y=df["pai"],mode="lines+markers",
        name="PAI",line=dict(color="#e84855",width=2)))
    fig.add_hline(y=1,line_dash="dash",line_color="#555",annotation_text="Random baseline")
    fig.update_layout(title="PAI vs Area Coverage",xaxis_title="Area %",yaxis_title="PAI",height=320,**_dk())
    return fig

def ripley_fig(r):
    d = r["distances"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.concatenate([d,d[::-1]]),
        y=np.concatenate([r["L_upper"],r["L_lower"][::-1]]),
        fill="toself",fillcolor="rgba(100,100,100,0.2)",
        line=dict(color="rgba(0,0,0,0)"),name="CSR envelope"))
    fig.add_trace(go.Scatter(x=d,y=r["L"],mode="lines",name="Observed L(d)",
        line=dict(color="#e84855",width=2)))
    fig.add_hline(y=0,line_dash="dash",line_color="#555",annotation_text="CSR")
    fig.update_layout(title="Ripley's L(d)",xaxis_title="Distance (m)",yaxis_title="L(d)",height=310,**_dk())
    return fig

def cm_fig(d):
    fig = go.Figure(go.Heatmap(z=d["cm"],x=["No Hot","Hotspot"],y=["No Hot","Hotspot"],
        colorscale=[[0,"#0d1117"],[1,"#e84855"]],
        text=d["cm"].astype(str),texttemplate="%{text}",showscale=False))
    fig.update_layout(title=f"Confusion Matrix  F1={d['f1']:.3f}",
                      xaxis_title="Predicted",yaxis_title="Actual",height=290,**_dk())
    return fig

def cat_dist_fig(s):
    s2 = s.sort_values(ascending=False).head(10)
    cols = [CATEGORY_COLORS.get(c,"#555") for c in s2.index]
    fig = go.Figure(go.Bar(x=[c.replace("_","<br>").title() for c in s2.index],y=s2.values,
        marker_color=cols,text=s2.values,textposition="outside"))
    fig.update_layout(title="Emerging Hotspot Distribution",yaxis_title="Cells",height=310,**_dk())
    return fig


# ════════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🗃️ Data Source")
    src = st.radio("", ["🧪 Synthetic", "🌆 NYC Open Data", "🌃 Chicago Open Data"])
    if src == "🌆 NYC Open Data":
        nyc_lim = st.slider("Records", 5000, 50000, 15000, 5000)
        nyc_dt  = st.text_input("Start date", "2023-06-01")
    if src == "🌃 Chicago Open Data":
        chi_lim = st.slider("Records", 5000, 50000, 15000, 5000)
        chi_dt  = st.text_input("Start date", "2023-06-01")
    st.divider()

    st.markdown("## 🎛️ Map Layer")
    layer = st.radio("", LAYERS, index=0, label_visibility="collapsed")

    time_layers = {"📦  Space-Time Cube", "⚡  Getis-Ord Gi*", "🗺️  LISA Clusters"}
    t_idx = st.slider("Month", 0, TRAIN_MONTHS-1, TRAIN_MONTHS-1) if any(l in layer for l in time_layers) else TRAIN_MONTHS-1

    animate = st.checkbox("▶️ Animated") if "Gi*" in layer else False
    fc_step = st.slider("Forecast Step", 0, N_FORECAST-1, 0) if "Forecast" in layer or "Uncertainty" in layer else 0

    coef_sel = "Temporal Lag β₁"
    if "GWR" in layer:
        coef_sel = st.selectbox("Coefficient", ["Temporal Lag β₁","Dist to Centre β₂","Seasonality β₃","Local R²"])

    use_prophet = st.checkbox("🔮 Use Prophet (slower)")
    st.divider()

    st.markdown("### 🔍 Cell Inspector")
    row_sel = st.number_input("Row", 0, 19, 10)
    col_sel = st.number_input("Col", 0, 19, 10)


# ════════════════════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════════════════════

st.markdown("""
<h1 style='color:#e84855;margin-bottom:0;font-size:1.55rem;'>
🗺️ Spatiotemporal Crime Intelligence Dashboard
</h1>
<p style='color:#8b949e;margin-top:2px;font-size:.82rem;'>
KDE · Moran's I · LISA · Ripley's K · Space-Time Cube · Gi* · Emerging Hotspots ·
ST-DBSCAN · ARIMA · Prophet · GWR · OLS/SAR/SEM · ROC · PAI
</p><hr>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# RUN ANALYSIS (all cached)
# ════════════════════════════════════════════════════════════════════════════════

with st.spinner("⏳ Running analysis (cached after first load)…"):
    # Data
    if src == "🌆 NYC Open Data":
        gdf = _nyc(nyc_lim, nyc_dt)
        if gdf.empty: st.error("NYC download failed."); st.stop()
        stc = _stc(gdf, False)
        bbox = tuple(gdf.total_bounds)
    elif src == "🌃 Chicago Open Data":
        gdf = _chi(chi_lim, chi_dt)
        if gdf.empty: st.error("Chicago download failed."); st.stop()
        stc = _stc(gdf, False)
        bbox = tuple(gdf.total_bounds)
    else:
        gdf  = _gen_synthetic()
        stc  = _stc(gdf, True)
        bbox = BBOX

    eha       = _eha(stc)
    kde_grid  = _kde(gdf, stc)
    lisa_all  = _lisa_all(stc)
    cdf       = _stdb(gdf)
    fc_obj, fc_cube = _forecast(stc, use_prophet)
    lo_cube, hi_cube = fc_obj.lower_cube, fc_obj.upper_cube
    ols_m, ols_p = _ols(stc)
    slm_m, slm_p = _slm(stc)
    sem_m, sem_p = _sem(stc)
    gwr_m     = _gwr(stc)
    gj_cells, geojson = _cells_wgs84(stc)
    gdf_wgs   = _gdf_wgs84(gdf)
    mi        = _morans(stc)
    rl        = _ripley(gdf, bbox)
    stat_df   = _stat_df(stc)
    cat_sum   = eha.summary()

    # Evaluation
    t_start    = min(TRAIN_MONTHS, stc.T-1)
    test_evts  = gdf[gdf["datetime"] >= stc.time_labels[t_start].to_timestamp()]
    fc_total   = fc_cube.sum(axis=0)
    ev         = PredictiveAccuracyEvaluator(stc, fc_total, test_evts)
    metrics    = ev.report(80)
    roc_d      = ev.roc_data()
    pr_d       = ev.precision_recall_data()
    cm_d       = ev.confusion_matrix_data(80)
    pai_df     = ev.pai_curve(np.arange(50, 100, 5))

    evals = {"ARIMA": ev}
    if ols_p is not None: evals["OLS"] = PredictiveAccuracyEvaluator(stc, ols_p, test_evts)
    if slm_p is not None: evals["Spatial Lag"] = PredictiveAccuracyEvaluator(stc, slm_p, test_evts)
    if sem_p is not None: evals["Spatial Error"] = PredictiveAccuracyEvaluator(stc, sem_p, test_evts)


# ════════════════════════════════════════════════════════════════════════════════
# MAP
# ════════════════════════════════════════════════════════════════════════════════

coef_map = {"Temporal Lag β₁":(1,"Temporal Lag β"),
            "Dist to Centre β₂":(2,"Dist Centre β"),
            "Seasonality β₃":(3,"Seasonality β"),
            "Local R²":("r2","Local R²")}

if   "Density" in layer and "KDE" not in layer: fig = fig_density(gdf_wgs)
elif "KDE" in layer:                             fig = fig_kde(kde_grid, stc, geojson)
elif "Cube" in layer:                            fig = fig_stc(stc, t_idx, geojson)
elif "Gi*" in layer:
    fig = _animated_gi(eha.z_cube.tolist(), [str(p) for p in stc.time_labels], json.dumps(geojson)) if animate else fig_gi(eha, t_idx, geojson)
elif "Emerging" in layer:                        fig = fig_emerging(eha, geojson)
elif "LISA" in layer:                            fig = fig_lisa(lisa_all, t_idx, geojson)
elif "DBSCAN" in layer:                          fig = fig_stdb(cdf)
elif "GWR" in layer:
    idx, name = coef_map.get(coef_sel, (1,"β"))
    fig = fig_gwr_coef(gwr_m, stc, geojson, idx, name)
elif "Uncertainty" in layer:                     fig = fig_forecast(fc_cube, lo_cube, hi_cube, stc, geojson, fc_step, ci=True)
else:                                            fig = fig_forecast(fc_cube, lo_cube, hi_cube, stc, geojson, fc_step)

st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})


# ════════════════════════════════════════════════════════════════════════════════
# METRIC CARDS
# ════════════════════════════════════════════════════════════════════════════════

st.markdown("<hr style='margin:.25rem 0;'>", unsafe_allow_html=True)
mc = st.columns(7)
mc[0].metric("Events",      f"{len(gdf):,}")
mc[1].metric("Grid",        f"{stc.nrows}×{stc.ncols}")
mc[2].metric("PAI @ 80%",   f"{metrics['pai']:.2f}", "↑ vs 1.0 random")
mc[3].metric("Hit Rate",    f"{metrics['hit_rate']*100:.1f}%")
mc[4].metric("ROC-AUC",     f"{roc_d['auc']:.3f}")
mc[5].metric("Moran's I",   f"{mi['I']:.3f}", "p<.05 ✓" if mi['p_value']<0.05 else "NS")
mc[6].metric("F1 Score",    f"{cm_d['f1']:.3f}")


# ════════════════════════════════════════════════════════════════════════════════
# ANALYSIS TABS
# ════════════════════════════════════════════════════════════════════════════════

st.markdown("<hr style='margin:.3rem 0;'>", unsafe_allow_html=True)
T = st.tabs(["📈 Time Series", "🔬 Exploratory (Phase 2)",
              "🏷️ Hotspot Categories", "📐 GWR Analysis",
              "✅ Full Validation", "📊 Model Comparison", "🗃️ Data & Config"])


# ─── Tab 0: Time Series ───────────────────────────────────────────────────────
with T[0]:
    c1, c2, c3 = st.columns([2.4, 0.8, 0.8])
    cell_cat = eha.category_grid[row_sel, col_sel]
    cell_tot = int(stc.total_counts()[row_sel, col_sel])
    sr       = test_stationarity(stc.get_cell_timeseries(row_sel, col_sel), f"({row_sel},{col_sel})")
    with c1:
        st.plotly_chart(ts_fig(stc, fc_cube, lo_cube, hi_cube, row_sel, col_sel), use_container_width=True)
    with c2:
        st.markdown("**Cell Info**")
        st.metric("Total crimes",  cell_tot)
        st.metric("Category",      cell_cat.replace("_"," ").title().split()[0])
    with c3:
        st.markdown("**Stationarity**")
        st.metric("ADF p",   f"{sr.get('adf_p_value','N/A')}")
        st.metric("KPSS p",  f"{sr.get('kpss_p_value','N/A')}")
        st.metric("Verdict", "✅ Stat." if sr.get("adf_stationary") else "❌ Non-stat.")


# ─── Tab 1: Exploratory ───────────────────────────────────────────────────────
with T[1]:
    st.markdown("### Phase 2 — Exploratory Spatial Analysis")
    e1, e2 = st.columns(2)
    with e1:
        st.plotly_chart(ripley_fig(rl), use_container_width=True)
        st.caption("L(d) > 0 = clustered beyond CSR  ·  Grey band = 39-simulation Monte Carlo envelope")
    with e2:
        st.markdown(f"""
**Global Moran's I** (last time step)

| | |
|---|---|
| I | `{mi['I']:.4f}` |
| Z-score | `{mi['z']:.2f}` |
| p-value | `{mi['p_value']:.4f}` |
| Result | {'✅ Significant positive spatial autocorrelation' if mi['p_value']<0.05 else '⬜ Not significant'} |

**Cell Stationarity** (ADF + KPSS on monthly time series)
""")
        if not stat_df.empty:
            for v, n in stat_df["verdict"].value_counts().items():
                st.write(f"- **{v}**: {n} cells ({100*n/len(stat_df):.0f}%)")
            with st.expander("Full stationarity table"):
                st.dataframe(stat_df[["row","col","verdict","adf_p_value","kpss_p_value"]],
                             hide_index=True, height=180)

    st.markdown("---")
    st.markdown("""
**LISA Interpretation** (Local Moran's I — see map layer 🗺️ LISA)

| Quadrant | Meaning |
|----------|---------|
| 🔴 **HH** | High crime cell surrounded by high-crime neighbours → **hotspot core** |
| 🟠 **LH** | Low crime cell surrounded by high-crime neighbours → **spatial outlier** |
| 🔵 **LL** | Low crime surrounded by low crime → **coldspot** |
| 🟣 **HL** | High crime cell surrounded by low-crime neighbours → **isolated spike** |
""")


# ─── Tab 2: Hotspot Categories ────────────────────────────────────────────────
with T[2]:
    h1, h2 = st.columns([2,1])
    with h1: st.plotly_chart(cat_dist_fig(cat_sum), use_container_width=True)
    with h2:
        st.markdown("**Summary**")
        df_c = cat_sum.reset_index()
        df_c.columns = ["Category","Cells"]
        df_c["Category"] = df_c["Category"].str.replace("_"," ").str.title()
        st.dataframe(df_c, hide_index=True, height=300)


# ─── Tab 3: GWR ───────────────────────────────────────────────────────────────
with T[3]:
    if gwr_m is None:
        st.warning("GWR model unavailable (try reducing grid to speed up bandwidth selection).")
    else:
        st.markdown(f"**GWR** · bandwidth = `{gwr_m.bandwidth:.0f}m` · kernel = Gaussian")
        g1, g2 = st.columns(2)
        with g1:
            gdf_gwr = gwr_m.summary_df(stc)
            show_cols = [c for c in gdf_gwr.columns if c not in ["flat_idx","row","col","residual"]]
            st.markdown("Local coefficient statistics:")
            st.dataframe(gdf_gwr[show_cols].describe().round(4), height=200)
            st.markdown("""
**Interpretation**
- β₁ (Temporal Lag) > 0 → crime is persistent here month-to-month
- β₁ varies spatially → crime persistence differs across the city
- High Local R² → local features explain crime well
- Low Local R² → unmeasured drivers (policing, social factors) dominate
""")
        with g2:
            # Local R² map (mini)
            r2v = gwr_m.local_r2_.tolist()
            fr2 = go.Figure(_choro(geojson, r2v, "viridis", 0, 1,
                                   [f"Local R²={v:.3f}" for v in r2v], "Local R²", title="Local R²"))
            fr2.update_layout(**base_mapbox_layout(height=320))
            st.plotly_chart(fr2, use_container_width=True)


# ─── Tab 4: Full Validation ───────────────────────────────────────────────────
with T[4]:
    st.markdown("### Phase 7 — Validation (most projects skip this — don't)")
    v1, v2, v3 = st.columns(3)
    with v1: st.plotly_chart(roc_fig(roc_d), use_container_width=True)
    with v2: st.plotly_chart(pr_fig(pr_d),   use_container_width=True)
    with v3: st.plotly_chart(pai_fig(pai_df), use_container_width=True)

    st.markdown("---")
    vc1, vc2 = st.columns([1, 2])
    with vc1: st.plotly_chart(cm_fig(cm_d), use_container_width=True)
    with vc2:
        st.markdown(f"""
**Confusion Matrix Summary** (hotspot threshold: top 20% cells)

| | Value |
|---|---|
| TP (true hotspots) | {cm_d['TP']} |
| FP (false alarms)  | {cm_d['FP']} |
| FN (missed hotspots) | {cm_d['FN']} |
| TN (true negatives) | {cm_d['TN']} |
| **Precision** | **{cm_d['precision']:.3f}** |
| **Recall** | **{cm_d['recall']:.3f}** |
| **F1** | **{cm_d['f1']:.3f}** |
| Accuracy | {cm_d['accuracy']:.3f} |
""")


# ─── Tab 5: Model Comparison ──────────────────────────────────────────────────
with T[5]:
    rows = {}
    for name, ev_m in evals.items():
        r = ev_m.roc_data(); c = ev_m.confusion_matrix_data(80)
        rows[name] = {"PAI": f"{ev_m.pai(80):.3f}",
                      "Hit Rate": f"{ev_m.hit_rate(80)*100:.1f}%",
                      "ROC-AUC": f"{r['auc']:.3f}",
                      "F1": f"{c['f1']:.3f}",
                      "Precision": f"{c['precision']:.3f}",
                      "Recall": f"{c['recall']:.3f}",
                      "RMSE": f"{ev_m.rmse():.3f}",
                      "MAE": f"{ev_m.mae():.3f}"}
    st.dataframe(pd.DataFrame(rows).T, height=200)
    st.markdown("""
| Metric | Interpretation |
|--------|---------------|
| **PAI > 1** | Hotspot prediction better than random |
| **ROC-AUC > 0.7** | Strong discriminative ability |
| **High Recall** | Most actual hotspots are captured |
| **High Precision** | Predicted hotspots are mostly real |
| **Low RMSE** | Accurate count predictions |
""")
    if slm_m:
        st.success(f"Spatial Lag ρ = {slm_m.rho_:.4f} — crime clusters spread to adjacent cells (contagion effect detected)")
    if sem_m:
        st.info(f"Spatial Error λ = {sem_m.lambda_:.4f} — unmeasured spatial factors drive residual correlation")


# ─── Tab 6: Data & Config ─────────────────────────────────────────────────────
with T[6]:
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**Dataset**")
        st.dataframe(pd.DataFrame({
            "Field": ["Source","Events","Train","Test","Grid","Cell size","CRS"],
            "Value": [src, f"{len(gdf):,}",
                      f"{stc.time_labels[0]} → {stc.time_labels[t_start-1]}",
                      f"{stc.time_labels[t_start]} → {stc.time_labels[-1]}",
                      f"{stc.nrows}×{stc.ncols}={stc.n_cells():,} cells",
                      f"{stc.cell_size:.0f}m", str(gdf.crs)]}),
            hide_index=True, height=260)
    with d2:
        st.markdown("**Regression Models**")
        mr = []
        if ols_m: mr.append({"Model":"OLS Baseline",    "R²":f"{ols_m.r2_:.4f}","Key param":"—"})
        if slm_m: mr.append({"Model":"Spatial Lag SAR", "R²":f"{slm_m.r2_:.4f}","Key param":f"ρ={slm_m.rho_:.4f}"})
        if sem_m: mr.append({"Model":"Spatial Error SEM","R²":f"{sem_m.r2_:.4f}","Key param":f"λ={sem_m.lambda_:.4f}"})
        if gwr_m: mr.append({"Model":"GWR","R²":f"{gwr_m.local_r2_.mean():.4f} (mean local)","Key param":f"bw={gwr_m.bandwidth:.0f}m"})
        if mr: st.dataframe(pd.DataFrame(mr), hide_index=True)

        st.markdown("**Forecasting**")
        st.write(f"- ARIMA cells:   {fc_obj._arima_count}")
        st.write(f"- Prophet cells: {fc_obj._prophet_count}")
        st.write(f"- SES cells:     {fc_obj._ses_count}")
        st.write(f"- Mean cells:    {fc_obj._mean_count}")

        st.markdown("**Data Sources**")
        st.code("""
# NYC Open Data
load_nyc_crimes(limit=20_000, start_date='2023-01-01')

# Chicago Open Data
load_chicago_crimes(limit=20_000, start_date='2023-01-01')

# Local file
load_from_file('crimes.csv', lat_col='latitude', lon_col='longitude')
        """)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""<hr style='border-color:#21262d;margin-top:.4rem;'>
<p style='text-align:center;color:#484f58;font-size:.72rem;'>
Spatiotemporal Crime Intelligence · KDE · Moran's I · LISA · Ripley's K ·
Space-Time Cube · Gi* · ST-DBSCAN · ARIMA · Prophet · GWR · OLS/SAR/SEM · PAI · ROC
</p>""", unsafe_allow_html=True)
