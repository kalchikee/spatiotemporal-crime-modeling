"""
visualization.py
================
Publication-quality figures for every stage of the spatiotemporal analysis.

Functions
---------
plot_crime_map            — raw event density map
plot_space_time_cube      — time-slice grid panels
plot_gi_star              — Gi* Z-score heatmaps
plot_emerging_hotspots    — emerging hotspot category choropleth
plot_stdbscan_clusters    — ST-DBSCAN cluster scatter
plot_forecast_surface     — predicted risk surface
plot_pai_curve            — PAI vs area-fraction curve
plot_model_comparison     — bar chart of evaluation metrics
plot_temporal_profile     — monthly count time series for selected cells
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns


# Colour palette for emerging hotspot categories
_CATEGORY_COLORS = {
    "NEW_HOTSPOT":           "#FF0000",
    "CONSECUTIVE_HOTSPOT":   "#FF6600",
    "INTENSIFYING_HOTSPOT":  "#FFAA00",
    "PERSISTENT_HOTSPOT":    "#FFD700",
    "DIMINISHING_HOTSPOT":   "#AAAAFF",
    "SPORADIC_HOTSPOT":      "#FFCCCC",
    "OSCILLATING_HOTSPOT":   "#FF00FF",
    "HISTORICAL_HOTSPOT":    "#BBBBBB",
    "NEW_COLDSPOT":          "#0000FF",
    "CONSECUTIVE_COLDSPOT":  "#0066FF",
    "INTENSIFYING_COLDSPOT": "#00AAFF",
    "PERSISTENT_COLDSPOT":   "#00CCFF",
    "DIMINISHING_COLDSPOT":  "#CCCCFF",
    "SPORADIC_COLDSPOT":     "#CCDDFF",
    "OSCILLATING_COLDSPOT":  "#AA00FF",
    "HISTORICAL_COLDSPOT":   "#DDDDFF",
    "NO_PATTERN":            "#F5F5F5",
}

_DEFAULT_FIGSIZE = (10, 8)
_CMAP_COUNTS = "YlOrRd"
_CMAP_GI     = "RdBu_r"


# ---------------------------------------------------------------------------
# 1. Raw crime density map
# ---------------------------------------------------------------------------

def plot_crime_map(gdf, title="Crime Event Density", figsize=_DEFAULT_FIGSIZE, save_path=None):
    """Hexbin density map of raw crime events."""
    fig, ax = plt.subplots(figsize=figsize)
    x = gdf.geometry.x.values
    y = gdf.geometry.y.values
    hb = ax.hexbin(x, y, gridsize=40, cmap=_CMAP_COUNTS, mincnt=1)
    cb = fig.colorbar(hb, ax=ax, label="Events per hexbin")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_aspect("equal")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax


# ---------------------------------------------------------------------------
# 2. Space-Time Cube — grid of time slices
# ---------------------------------------------------------------------------

def plot_space_time_cube(stc, n_cols: int = 4, figsize: tuple | None = None, save_path=None):
    """
    Grid of heatmaps — one panel per time step in the space-time cube.
    """
    T = stc.T
    n_rows = int(np.ceil(T / n_cols))
    if figsize is None:
        figsize = (n_cols * 3, n_rows * 2.5)

    vmax = stc.cube.max()
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    for t in range(T):
        r, c = divmod(t, n_cols)
        ax = axes[r][c]
        im = ax.imshow(
            stc.get_time_slice(t),
            origin="lower",
            cmap=_CMAP_COUNTS,
            vmin=0,
            vmax=vmax,
            aspect="auto",
        )
        ax.set_title(str(stc.time_labels[t]), fontsize=8)
        ax.axis("off")

    # Hide unused panels
    for t in range(T, n_rows * n_cols):
        r, c = divmod(t, n_cols)
        axes[r][c].set_visible(False)

    fig.suptitle("Space-Time Cube — Monthly Crime Counts", fontsize=14, fontweight="bold", y=1.01)
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Crime Count", shrink=0.6)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 3. Gi* Z-score heatmap
# ---------------------------------------------------------------------------

def plot_gi_star(z_cube, time_labels, n_cols: int = 4, figsize=None, save_path=None):
    """
    Grid of Gi* Z-score heatmaps for each time step.
    Hot colours = hotspot (Z > 1.96); cool colours = coldspot.
    """
    T = z_cube.shape[0]
    n_rows = int(np.ceil(T / n_cols))
    if figsize is None:
        figsize = (n_cols * 3, n_rows * 2.5)

    vlim = max(abs(z_cube.min()), abs(z_cube.max()))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    for t in range(T):
        r, c = divmod(t, n_cols)
        ax = axes[r][c]
        im = ax.imshow(
            z_cube[t],
            origin="lower",
            cmap=_CMAP_GI,
            vmin=-vlim,
            vmax=vlim,
            aspect="auto",
        )
        ax.set_title(str(time_labels[t]), fontsize=8)
        ax.axis("off")

    for t in range(T, n_rows * n_cols):
        r, c = divmod(t, n_cols)
        axes[r][c].set_visible(False)

    fig.suptitle("Getis-Ord Gi* Z-scores Over Time", fontsize=14, fontweight="bold", y=1.01)
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Gi* Z-score", shrink=0.6)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 4. Emerging Hotspot Category choropleth
# ---------------------------------------------------------------------------

def plot_emerging_hotspots(category_gdf, figsize=_DEFAULT_FIGSIZE, save_path=None):
    """
    Choropleth map of emerging hotspot categories.
    """
    gdf = category_gdf.copy()
    cats = gdf["category"].unique()
    color_map = {cat: _CATEGORY_COLORS.get(cat, "#CCCCCC") for cat in cats}
    gdf["color"] = gdf["category"].map(color_map)

    fig, ax = plt.subplots(figsize=figsize)
    gdf.plot(color=gdf["color"], ax=ax, linewidth=0, edgecolor="none")

    # Legend
    present_cats = [c for c in _CATEGORY_COLORS if c in cats]
    patches = [
        mpatches.Patch(color=_CATEGORY_COLORS[c], label=c.replace("_", " ").title())
        for c in present_cats
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=7, framealpha=0.9,
              title="Pattern", title_fontsize=8)

    ax.set_title("Emerging Hotspot Analysis", fontsize=14, fontweight="bold")
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax


# ---------------------------------------------------------------------------
# 5. ST-DBSCAN cluster scatter
# ---------------------------------------------------------------------------

def plot_stdbscan_clusters(gdf_clustered, datetime_col="datetime", figsize=(12, 5), save_path=None):
    """
    Two-panel plot: (left) spatial cluster map; (right) temporal density per cluster.
    """
    gdf = gdf_clustered.copy()
    gdf[datetime_col] = pd.to_datetime(gdf[datetime_col])
    n_clusters = gdf["cluster"].max() + 1

    palette = sns.color_palette("tab10", n_colors=max(n_clusters, 1))
    noise_color = (0.7, 0.7, 0.7)

    fig, (ax_sp, ax_t) = plt.subplots(1, 2, figsize=figsize)

    # Spatial scatter
    noise = gdf[gdf["cluster"] == -1]
    ax_sp.scatter(noise.geometry.x, noise.geometry.y, c=[noise_color], s=2, alpha=0.3, label="Noise")
    for cid in range(n_clusters):
        sub = gdf[gdf["cluster"] == cid]
        if len(sub) == 0:
            continue
        ax_sp.scatter(sub.geometry.x, sub.geometry.y, c=[palette[cid % 10]],
                      s=6, alpha=0.6, label=f"Cluster {cid}")
    ax_sp.set_title("ST-DBSCAN Spatial Clusters")
    ax_sp.set_xlabel("Easting (m)")
    ax_sp.set_ylabel("Northing (m)")
    ax_sp.set_aspect("equal")
    if n_clusters <= 10:
        ax_sp.legend(fontsize=7, markerscale=2)

    # Temporal density per cluster
    for cid in range(min(n_clusters, 8)):
        sub = gdf[gdf["cluster"] == cid]
        if len(sub) < 5:
            continue
        monthly = sub.set_index(datetime_col).resample("ME").size()
        ax_t.plot(monthly.index, monthly.values, label=f"Cluster {cid}",
                  color=palette[cid % 10], linewidth=1.5)
    ax_t.set_title("Monthly Event Count per Cluster")
    ax_t.set_xlabel("Date")
    ax_t.set_ylabel("Event Count")
    ax_t.legend(fontsize=7)
    plt.setp(ax_t.xaxis.get_majorticklabels(), rotation=30, ha="right")

    plt.suptitle("ST-DBSCAN Spatiotemporal Clustering", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 6. Forecast risk surface
# ---------------------------------------------------------------------------

def plot_forecast_surface(
    forecast_grid,
    actual_grid=None,
    hotspot_mask=None,
    period_label="Next Period",
    figsize=(14, 5),
    save_path=None,
):
    """
    Side-by-side: predicted risk surface | actual crime counts | hotspot overlay.
    """
    n_panels = 2 + (hotspot_mask is not None)
    fig, axes = plt.subplots(1, n_panels, figsize=figsize)

    ax = axes[0]
    im0 = ax.imshow(forecast_grid, origin="lower", cmap=_CMAP_COUNTS, aspect="auto")
    fig.colorbar(im0, ax=ax, label="Predicted count")
    ax.set_title(f"Predicted Risk Surface\n({period_label})")
    ax.axis("off")

    if actual_grid is not None:
        ax = axes[1]
        im1 = ax.imshow(actual_grid, origin="lower", cmap=_CMAP_COUNTS, aspect="auto")
        fig.colorbar(im1, ax=ax, label="Actual count")
        ax.set_title("Actual Crime Counts")
        ax.axis("off")

    if hotspot_mask is not None:
        ax = axes[-1]
        ax.imshow(forecast_grid, origin="lower", cmap="Greys", alpha=0.3, aspect="auto")
        overlay = np.ma.masked_where(~hotspot_mask, hotspot_mask.astype(float))
        ax.imshow(overlay, origin="lower", cmap="Reds", alpha=0.7, aspect="auto")
        ax.set_title("Predicted Hotspot Zones\n(top 20% risk cells)")
        ax.axis("off")

    plt.suptitle("Forecast vs Actual", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 7. PAI curve
# ---------------------------------------------------------------------------

def plot_pai_curve(pai_df, model_name="ARIMA Forecast", figsize=(10, 4), save_path=None):
    """
    Plot PAI and Hit Rate as a function of area coverage fraction.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    ax1.plot(pai_df["area_fraction"] * 100, pai_df["pai"], marker="o", linewidth=2, color="#E84855")
    ax1.axhline(1, linestyle="--", color="grey", linewidth=1, label="Random baseline (PAI=1)")
    ax1.set_xlabel("Area Coverage (%)")
    ax1.set_ylabel("PAI")
    ax1.set_title("Predictive Accuracy Index (PAI)")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    ax2.plot(pai_df["area_fraction"] * 100, pai_df["hit_rate"] * 100, marker="o", linewidth=2, color="#3A86FF")
    ax2.plot(pai_df["area_fraction"] * 100, pai_df["area_fraction"] * 100, linestyle="--",
             color="grey", linewidth=1, label="Random (Hit Rate = Area %)")
    ax2.set_xlabel("Area Coverage (%)")
    ax2.set_ylabel("Hit Rate (%)")
    ax2.set_title("Hit Rate vs Area Coverage")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.suptitle(f"Predictive Performance — {model_name}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 8. Model comparison bar chart
# ---------------------------------------------------------------------------

def plot_model_comparison(comparison_df, figsize=(12, 5), save_path=None):
    """
    Grouped bar chart comparing PAI, Hit Rate, RMSE across models.
    """
    metrics = ["Hit Rate", "PAI", "RMSE", "Pearson r"]
    available = [m for m in metrics if m in comparison_df.columns]
    n = len(available)

    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    palette = sns.color_palette("Set2", n_colors=len(comparison_df))

    for ax, metric in zip(axes, available):
        vals = comparison_df[metric]
        bars = ax.bar(vals.index, vals.values, color=palette)
        ax.set_title(metric, fontweight="bold")
        ax.set_ylabel(metric)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(vals.index, rotation=20, ha="right", fontsize=9)
        for bar, v in zip(bars, vals.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    plt.suptitle("Model Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 9. Temporal profile for a selected cell
# ---------------------------------------------------------------------------

def plot_temporal_profile(stc, rows_cols: list[tuple], forecast_cube=None,
                           train_t_end=None, figsize=(12, 4), save_path=None):
    """
    Monthly count time series for one or more grid cells, with optional forecast.

    Parameters
    ----------
    rows_cols : list of (row, col) tuples
    forecast_cube : np.ndarray (n_steps, nrows, ncols) or None
    train_t_end : int — draw a vertical line at train/test split
    """
    n = len(rows_cols)
    fig, axes = plt.subplots(1, n, figsize=(figsize[0], figsize[1]), squeeze=False)

    time_idx = range(stc.T)
    labels = [str(p) for p in stc.time_labels]

    for ax, (row, col) in zip(axes[0], rows_cols):
        series = stc.get_cell_timeseries(row, col)
        ax.bar(time_idx, series, color="#3A86FF", alpha=0.7, label="Observed")

        if forecast_cube is not None and train_t_end is not None:
            fc_start = train_t_end
            fc_idx = range(fc_start, fc_start + len(forecast_cube))
            ax.bar(fc_idx, forecast_cube[:, row, col], color="#E84855", alpha=0.7, label="Forecast")

        if train_t_end is not None:
            ax.axvline(train_t_end - 0.5, color="black", linestyle="--", linewidth=1.5,
                       label="Train/Test split")

        xticks = list(range(0, stc.T, max(1, stc.T // 6)))
        ax.set_xticks(xticks)
        ax.set_xticklabels([labels[i] for i in xticks], rotation=30, ha="right", fontsize=8)
        ax.set_title(f"Cell ({row}, {col})", fontsize=10, fontweight="bold")
        ax.set_ylabel("Crime Count")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Monthly Crime Count — Selected Cells", fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 10. Category distribution bar chart
# ---------------------------------------------------------------------------

def plot_category_distribution(summary_series, figsize=(10, 4), save_path=None):
    """Bar chart of cell counts per emerging-hotspot category."""
    s = summary_series.sort_values(ascending=False)
    colors = [_CATEGORY_COLORS.get(cat, "#CCCCCC") for cat in s.index]

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(range(len(s)), s.values, color=colors)
    ax.set_xticks(range(len(s)))
    ax.set_xticklabels(
        [c.replace("_", "\n").title() for c in s.index],
        rotation=35, ha="right", fontsize=8,
    )
    ax.set_ylabel("Number of Grid Cells")
    ax.set_title("Emerging Hotspot Category Distribution", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
