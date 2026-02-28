"""
clustering.py
=============
ST-DBSCAN — Spatiotemporal Density-Based Spatial Clustering of Applications with Noise.

ST-DBSCAN extends classic DBSCAN with a second (temporal) neighbourhood threshold.
A point q is a spatial-temporal neighbour of p if:
    spatial_dist(p, q)  <= eps1   AND
    temporal_dist(p, q) <= eps2

Cluster labels are returned in the same order as the input GeoDataFrame.
Label  -1  means noise.

Reference
---------
Birant D. & Kut A. (2007). "ST-DBSCAN: An algorithm for clustering
spatial–temporal data". Data & Knowledge Engineering, 60(1), 208–221.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.neighbors import BallTree


class STDBSCAN:
    """
    Spatiotemporal DBSCAN clustering.

    Parameters
    ----------
    eps_spatial : float
        Maximum spatial distance between two points to be neighbours (metres).
    eps_temporal : float
        Maximum temporal distance (seconds) between two points to be neighbours.
    min_samples : int
        Minimum number of points in a neighbourhood to form a core point.
    """

    def __init__(
        self,
        eps_spatial: float = 1_000.0,
        eps_temporal: float = 30 * 24 * 3600,  # 30 days in seconds
        min_samples: int = 10,
    ):
        self.eps_spatial = eps_spatial
        self.eps_temporal = eps_temporal
        self.min_samples = min_samples

        self.labels_: np.ndarray | None = None
        self.n_clusters_: int = 0
        self.n_noise_: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, gdf: gpd.GeoDataFrame, datetime_col: str = "datetime") -> "STDBSCAN":
        """
        Fit ST-DBSCAN to a point GeoDataFrame.

        Parameters
        ----------
        gdf : GeoDataFrame
            Projected CRS (metres).
        datetime_col : str
            Column holding datetime values.
        """
        gdf = gdf.copy()
        gdf[datetime_col] = pd.to_datetime(gdf[datetime_col])

        coords = np.column_stack([gdf.geometry.x.values, gdf.geometry.y.values])
        timestamps = gdf[datetime_col].values.astype("int64") / 1e9  # → seconds

        n = len(coords)
        labels = np.full(n, -2, dtype=int)  # -2 = unvisited

        # Spatial index for fast radius queries
        tree = BallTree(coords, metric="euclidean")
        spatial_nbrs = tree.query_radius(coords, r=self.eps_spatial)

        cluster_id = 0

        for i in range(n):
            if labels[i] != -2:          # already visited
                continue
            labels[i] = -1               # tentatively noise

            # Spatial + temporal neighbours of i
            nbrs = self._temporal_filter(i, spatial_nbrs[i], timestamps)

            if len(nbrs) < self.min_samples:
                continue                  # noise remains -1

            # i is a core point — start a new cluster
            labels[i] = cluster_id
            queue = list(nbrs)

            while queue:
                j = queue.pop()
                if labels[j] == -1:
                    labels[j] = cluster_id   # border point
                if labels[j] != -2:
                    continue
                labels[j] = cluster_id

                j_nbrs = self._temporal_filter(j, spatial_nbrs[j], timestamps)
                if len(j_nbrs) >= self.min_samples:
                    queue.extend(j_nbrs)

            cluster_id += 1

        # Remaining -2 are isolated; mark as noise
        labels[labels == -2] = -1

        self.labels_ = labels
        self.n_clusters_ = cluster_id
        self.n_noise_ = (labels == -1).sum()
        print(
            f"[ST-DBSCAN] Found {self.n_clusters_} clusters, "
            f"{self.n_noise_} noise points ({100*self.n_noise_/n:.1f}%)"
        )
        return self

    def fit_transform(self, gdf: gpd.GeoDataFrame, datetime_col: str = "datetime") -> gpd.GeoDataFrame:
        """Fit and return GDF with an added 'cluster' column."""
        self.fit(gdf, datetime_col)
        out = gdf.copy()
        out["cluster"] = self.labels_
        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _temporal_filter(
        self,
        i: int,
        spatial_candidates: np.ndarray,
        timestamps: np.ndarray,
    ) -> list[int]:
        """Keep only spatial candidates within eps_temporal of point i."""
        t_i = timestamps[i]
        return [
            j
            for j in spatial_candidates
            if j != i and abs(timestamps[j] - t_i) <= self.eps_temporal
        ]

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def cluster_summary(self, gdf: gpd.GeoDataFrame, datetime_col: str = "datetime") -> pd.DataFrame:
        """
        Return a DataFrame summarising each cluster:
            cluster, n_events, start_date, end_date, centroid_x, centroid_y
        """
        if self.labels_ is None:
            raise RuntimeError("Call .fit() first.")

        gdf = gdf.copy()
        gdf["cluster"] = self.labels_
        gdf[datetime_col] = pd.to_datetime(gdf[datetime_col])

        records = []
        for cid in sorted(set(self.labels_)):
            if cid == -1:
                continue
            sub = gdf[gdf["cluster"] == cid]
            records.append(
                dict(
                    cluster=cid,
                    n_events=len(sub),
                    start_date=sub[datetime_col].min().date(),
                    end_date=sub[datetime_col].max().date(),
                    centroid_x=sub.geometry.x.mean(),
                    centroid_y=sub.geometry.y.mean(),
                    duration_days=(sub[datetime_col].max() - sub[datetime_col].min()).days,
                )
            )
        return pd.DataFrame(records)
