"""Geographic item-item graph G_GEO (Haversine distance)."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0


def haversine_to_all(lat: np.ndarray, lon: np.ndarray, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Distances in km from each point in (lat, lon) [shape B] to all points (lats, lons) [shape N].

    Inputs are in degrees; returns an array of shape ``[B, N]``.
    """
    la1, lo1 = np.radians(lat)[:, None], np.radians(lon)[:, None]
    la2, lo2 = np.radians(lats)[None, :], np.radians(lons)[None, :]
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def build_geo_edges(
    business: pd.DataFrame,
    item2idx: dict,
    k: int = 10,
    max_dist_km: float = 50.0,
    chunk_size: int = 500,
    verbose: bool = True,
) -> list[tuple[str, str, float]]:
    """For every item, connect its ``k`` nearest items that lie within ``max_dist_km``.

    Returns directed edges ``(business_a, business_b, distance_km)``.
    """
    business = business[business["business_id"].isin(item2idx)].copy()
    business = business.dropna(subset=["latitude", "longitude"])
    business = business.drop_duplicates(subset=["business_id"])
    business = business.sort_values("business_id").reset_index(drop=True)

    lats = business["latitude"].values
    lons = business["longitude"].values
    b_ids = business["business_id"].values
    n = len(lats)

    edges: list[tuple[str, str, float]] = []
    t0 = time.time()
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        dist = haversine_to_all(lats[start:end], lons[start:end], lats, lons)
        dist[np.arange(end - start), np.arange(start, end)] = np.inf  # no self-loops

        top_k = np.argpartition(dist, k, axis=1)[:, :k]
        for row, global_idx in enumerate(range(start, end)):
            for j in top_k[row]:
                d = dist[row, j]
                if d <= max_dist_km:
                    edges.append((b_ids[global_idx], b_ids[j], float(d)))

        if verbose and (start // chunk_size) % 5 == 0:
            print(f"  geo: {end}/{n} items | {time.time() - t0:.1f}s | {len(edges):,} edges")

    if verbose:
        print(f"  geo: {len(edges):,} directed edges built")
    return edges


def geo_weight(distance_km: float) -> float:
    """Inverse-distance edge weight: 0 km -> 1.0, 5 km -> 0.17, 50 km -> 0.02."""
    return 1.0 / (1.0 + distance_km)
