"""
Change-detection search.

Compares visual embeddings from two different time periods to find areas
that have changed.  The user provides a query (e.g. "new buildings"),
a from-time and to-time (past/recent/present mapping to 2014/2020/2026),
and a mode (new/removed/increased/decreased).

Use from_time="recent", to_time="present" for changes in the past 5 years.
Use from_time="past", to_time="present" for long-term changes over 10 years.

Internally the operation:
  1. Loads TurboQuant vision indices for both years at the specified resolution.
  2. Encodes the query with the shared CLIP text encoder.
  3. Searches both indices to get (score, lat, lon) tuples.
  4. Matches points across the two time periods by spatial proximity.
  5. Filters the matched/unmatched points according to `mode`.
"""

from typing import Optional, Dict, List, Tuple

import numpy as np
import geopandas as gpd
from shapely.geometry import Point

import config
from schema import empty_gdf, from_geometries
from operations.threshold import compute_threshold


def _nearest_match(
    from_pts: List[Tuple[float, float, float]],
    to_pts: List[Tuple[float, float, float]],
    threshold: float,
) -> Dict[int, int]:
    """For each point in *to_pts*, find the nearest point in *from_pts*
    within *threshold* degrees.  Returns {to_idx: from_idx}."""
    if not from_pts or not to_pts:
        return {}

    from_arr = np.array(from_pts)  # (N, 3) — score, lat, lon
    to_arr = np.array(to_pts)

    matches: Dict[int, int] = {}
    for ti in range(len(to_arr)):
        dists = np.sqrt(
            (from_arr[:, 1] - to_arr[ti, 1]) ** 2
            + (from_arr[:, 2] - to_pts[ti][2]) ** 2
        )
        nearest = int(np.argmin(dists))
        if dists[nearest] <= threshold:
            matches[ti] = nearest
    return matches


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
    """Detect change between *from_time* and *to_time* for *query*.

    Parameters
    ----------
    query : str
        Visual concept to search for (e.g. "new buildings").
    from_time, to_time : str
        One of "past", "recent", "present".
    mode : str
        One of "new", "removed", "increased", "decreased".
    region : GeoDataFrame, optional
        Spatial filter applied to both time periods.
    vision_encoder : models.VisionEncoder
        Shared CLIP text encoder.
    vision_year_indices : dict
        ``{year: {resolution: TurboQuantSearchIndex}}``.
    resolution : str
        Vision resolution ("low" or "high").
    top_n, nprobe : int
        Search parameters forwarded to TurboQuant.

    Returns
    -------
    GeoDataFrame
        Points representing changed areas, with ``time`` and ``score`` columns.
    """
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

    # --- Encode query and search both time periods ---
    query_vector = vision_encoder.encode_text(query)
    query_np = query_vector.squeeze(0).detach().cpu().numpy()

    print(f"[{resolution}] Searching {from_time} ({from_year}) index...")
    from_scores, from_lat, from_lon = from_index.search(
        query_np, region=region, nprobe=nprobe,
    )
    print(f"[{resolution}] Searching {to_time} ({to_year}) index...")
    to_scores, to_lat, to_lon = to_index.search(
        query_np, region=region, nprobe=nprobe,
    )

    # Adaptive Thresholding
    from_mask = compute_threshold(from_scores)
    from_scores = from_scores[from_mask == 1]
    from_lat = from_lat[from_mask == 1]
    from_lon = from_lon[from_mask == 1]

    to_mask = compute_threshold(to_scores)
    to_scores = to_scores[to_mask == 1]
    to_lat = to_lat[to_mask == 1]
    to_lon = to_lon[to_mask == 1]

    if len(from_scores) == 0 and len(to_scores) == 0:
        print("[change] No results in either time period.")
        return empty_gdf()

    # --- Build point lists: (score, lat, lon) ---
    from_pts = [(float(s), float(la), float(lo)) for s, la, lo in zip(from_scores, from_lat, from_lon)]
    to_pts   = [(float(s), float(la), float(lo)) for s, la, lo in zip(to_scores, to_lat, to_lon)]

    # --- Save raw search points to Shapefiles ---
    if from_pts:
        gdf_from = gpd.GeoDataFrame(
            {"score": [p[0] for p in from_pts]},
            geometry=[Point(p[2], p[1]) for p in from_pts],
            crs="EPSG:4326"  # Set your coordinate reference system if different
        )
        gdf_from.to_file(f"results/from_pts_{from_time}_{from_year}.shp")

    if to_pts:
        gdf_to = gpd.GeoDataFrame(
            {"score": [p[0] for p in to_pts]},
            geometry=[Point(p[2], p[1]) for p in to_pts],
            crs="EPSG:4326"
        )
        gdf_to.to_file(f"results/to_pts_{to_time}_{to_year}.shp")

    # --- Match points across time periods ---
    threshold = config.CHANGE_DISTANCE_THRESHOLD
    matches = _nearest_match(from_pts, to_pts, threshold)

    to_matched = set(matches.keys())
    from_matched = set(matches.values())

    # --- Collect results according to mode ---
    result_points: List[Point] = []
    result_scores: List[float] = []
    result_times: List[str] = []

    if mode == "new":
        for ti, fi in matches.items():
            if to_pts[ti][0] - from_pts[fi][0] > 0.1:
                result_points.append(Point(to_pts[ti][2], to_pts[ti][1]))
                result_scores.append(to_pts[ti][0] - from_pts[fi][0])
                result_times.append(to_time)

    elif mode == "removed":
        for ti, fi in matches.items():
            if from_pts[fi][0] - to_pts[ti][0] > 0.1:
                result_points.append(Point(from_pts[fi][2], from_pts[fi][1]))
                result_scores.append(from_pts[fi][0] - to_pts[ti][0])
                result_times.append(from_time)

    elif mode == "increased":
        for ti, fi in matches.items():
            if to_pts[ti][0] > from_pts[fi][0]:
                result_points.append(Point(to_pts[ti][2], to_pts[ti][1]))
                result_scores.append(to_pts[ti][0] - from_pts[fi][0])
                result_times.append(f"{from_time}->{to_time}")

    elif mode == "decreased":
        for ti, fi in matches.items():
            if to_pts[ti][0] < from_pts[fi][0]:
                result_points.append(Point(to_pts[ti][2], to_pts[ti][1]))
                result_scores.append(from_pts[fi][0] - to_pts[ti][0])
                result_times.append(f"{from_time}->{to_time}")

    if not result_points:
        print(f"[{resolution}] No '{mode}' changes detected.")
        return empty_gdf()

    gdf = from_geometries(result_points, scores=result_scores)
    gdf["time"] = result_times
    print(f"[{resolution}] {mode}: {len(gdf)} change(s) detected.")
    return gdf
