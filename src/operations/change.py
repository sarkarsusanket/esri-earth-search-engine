"""
Change-detection search.

Compares visual embeddings from two different time periods to find areas
that have changed.  The user provides a query (e.g. "new buildings"),
a from-time and to-time (past/recent/present mapping to 2014/2020/2026),
and a mode (new/removed/increased/decreased).

Use from_time="recent", to_time="present" for changes in the past 5 years.
Use from_time="past", to_time="present" for long-term changes over 10 years.
"""

from typing import Optional, Dict, List, Tuple
from pathlib import Path

import numpy as np
import geopandas as gpd
from scipy.spatial import cKDTree
from shapely.geometry import Point

import config
from schema import empty_gdf, from_geometries
from operations.threshold import compute_threshold


def _nearest_match_coords(
    from_coords: np.ndarray,
    to_coords: np.ndarray,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Finds nearest neighbor within threshold using KDTree.
    
    Returns array pairs of matching indices (to_indices, from_indices).
    """
    if len(from_coords) == 0 or len(to_coords) == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)

    tree = cKDTree(from_coords)
    dists, from_indices = tree.query(to_coords, distance_upper_bound=threshold)

    valid_mask = dists <= threshold
    to_indices = np.where(valid_mask)[0]
    matched_from_indices = from_indices[valid_mask].astype(int)

    return to_indices, matched_from_indices


def change(
    query: str,
    from_time: str,
    to_time: str,
    mode: str,
    region: Optional[gpd.GeoDataFrame] = None,
    vision_encoder=None,
    vision_year_indices: Optional[Dict[int, Dict[str, "TurboQuantSearchIndex"]]] = None,
    resolution: str = config.DEFAULT_RESOLUTION,
    nprobe: int = config.VISION_NPROBE_DEFAULT,
) -> gpd.GeoDataFrame:
    """Detect change between *from_time* and *to_time* for *query*."""
    if vision_year_indices is None:
        print("No year-specific vision indices loaded.")
        return empty_gdf()

    from_year = config.VISION_YEARS[from_time]
    to_year = config.VISION_YEARS[to_time]

    from_index = vision_year_indices.get(from_year, {}).get(resolution)
    to_index = vision_year_indices.get(to_year, {}).get(resolution)

    if from_index is None:
        print(f"No vision index for from_time='{from_time}' (year {from_year}, resolution '{resolution}').")
        return empty_gdf()
    if to_index is None:
        print(f"No vision index for to_time='{to_time}' (year {to_year}, resolution '{resolution}').")
        return empty_gdf()

    # --- Early index filtering using mode specific search bounds ---
    # Passing confidence thresholds directly to the search index pre-filters candidates,
    # drastically reducing the dataset size before k-d tree spatial matching.
    from_thresh = 0.2 if mode in ("removed", "decreased") else None
    to_thresh = 0.2 if mode in ("new", "increased") else None

    # --- Encode query and search both time periods ---
    query_vector = vision_encoder.encode_text(query)
    query_np = query_vector.squeeze(0).detach().cpu().numpy()

    print(f"[{resolution}] Searching {from_time} ({from_year}) index...")
    from_scores, from_lat, from_lon = from_index.search(
        query_np, region=region, nprobe=nprobe, confidence_thresh=from_thresh
    )
    print(f"[{resolution}] Searching {to_time} ({to_year}) index...")
    to_scores, to_lat, to_lon = to_index.search(
        query_np, region=region, nprobe=nprobe, confidence_thresh=to_thresh
    )

    if len(from_scores) == 0 and len(to_scores) == 0:
        print("[change] No results in either time period.")
        return empty_gdf()

    # Convert search results directly into contiguous NumPy arrays for zero-copy vectorized operations
    from_scores = np.asarray(from_scores, dtype=np.float64)
    from_coords = np.column_stack((from_lat, from_lon))

    to_scores = np.asarray(to_scores, dtype=np.float64)
    to_coords = np.column_stack((to_lat, to_lon))

    Path("results").mkdir(parents=True, exist_ok=True)

    # --- Save raw search points to Shapefiles ---
    if len(from_scores) > 0:
        gdf_from = gpd.GeoDataFrame(
            {"score": from_scores},
            geometry=gpd.points_from_xy(from_coords[:, 1], from_coords[:, 0]),
            crs="EPSG:4326"
        )
        gdf_from.to_file(f"results/from_pts_{from_time}_{from_year}.shp")

    if len(to_scores) > 0:
        gdf_to = gpd.GeoDataFrame(
            {"score": to_scores},
            geometry=gpd.points_from_xy(to_coords[:, 1], to_coords[:, 0]),
            crs="EPSG:4326"
        )
        gdf_to.to_file(f"results/to_pts_{to_time}_{to_year}.shp")

    # --- Match points across time periods ---
    to_idx, from_idx = _nearest_match_coords(from_coords, to_coords, config.CHANGE_DISTANCE_THRESHOLD)

    if len(to_idx) == 0:
        print(f"[{resolution}] No '{mode}' changes detected.")
        return empty_gdf()

    m_to_scores = to_scores[to_idx]
    m_from_scores = from_scores[from_idx]
    minus_scores = m_to_scores - m_from_scores

    gdf_matched = gpd.GeoDataFrame(
        {
            "to_score": m_to_scores,
            "from_score": m_from_scores,
            "minus_score": minus_scores
        },
        geometry=gpd.points_from_xy(to_coords[to_idx, 1], to_coords[to_idx, 0]),
        crs="EPSG:4326"
    )
    print(gdf_matched)
    gdf_matched.to_file(rf"D:\Code\query-earth\results\to_matched_{to_time}_{to_year}.shp")

    # --- Vectorized Mode Filtering ---
    if mode == "new":
        mask = (m_from_scores < 0.18) & (m_to_scores > 0.2)
        res_scores = minus_scores[mask]
        res_time = to_time
    elif mode == "removed":
        mask = (m_from_scores > 0.2) & (m_to_scores < 0.18)
        res_scores = minus_scores[mask]
        res_time = to_time
    elif mode == "increased":
        mask = (m_to_scores > m_from_scores) & ((m_to_scores - m_from_scores)>0.01) & (m_to_scores > 0.2)
        res_scores = minus_scores[mask]
        res_time = f"{from_time}->{to_time}"
    elif mode == "decreased":
        mask = (m_to_scores < m_from_scores) & ((m_from_scores - m_to_scores)>0.01) & (m_from_scores > 0.2)
        res_scores = m_from_scores[mask] - m_to_scores[mask]
        res_time = f"{from_time}->{to_time}"
    else:
        mask = np.zeros(len(to_idx), dtype=bool)
        res_scores = np.empty(0)
        res_time = ""

    if not np.any(mask):
        print(f"[{resolution}] No '{mode}' changes detected.")
        return empty_gdf()

    matched_to_idx = to_idx[mask]
    matched_lons = to_coords[matched_to_idx, 1]
    matched_lats = to_coords[matched_to_idx, 0]

    # Batch create Shapely Point objects via high-speed vectorized constructor
    result_points = gpd.points_from_xy(matched_lons, matched_lats)

    gdf = from_geometries(list(result_points), scores=res_scores.tolist())
    gdf["time"] = res_time
    print(f"[{resolution}] {mode}: {len(gdf)} change(s) detected.")
    return gdf