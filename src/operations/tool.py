"""
Spatial tool operations.

These are pure functions over GeoDataFrames that already conform to the
pipeline's standard schema (a `geometry` column, optionally a `score`
column). They never touch embeddings or models — only geometry.
"""
from typing import Optional

import geopandas as gpd


from schema import *

# Rough degrees-per-km at mid-latitudes; used only as a buffer fallback
# when a GeoDataFrame has no projected CRS to buffer in meters directly.
_KM_TO_DEGREES = 1 / 111.0


def buffer(gdf: gpd.GeoDataFrame, distance_km: float) -> gpd.GeoDataFrame:
    """Buffer every geometry in `gdf` outward by `distance_km` kilometers."""
    if gdf is None or gdf.empty:
        return empty_gdf()
    if distance_km is None:
        raise ValueError("buffer operation requires a 'buffer_distance_km' parameter.")

    gdf = ensure_crs(gdf)
    # Project to a metric CRS for an accurate buffer, then back to EPSG:4326.
    metric = gdf.to_crs(gdf.estimate_utm_crs())
    buffered = metric.copy()
    buffered[GEOMETRY_COL] = metric.geometry.buffer(distance_km * 1000)
    return ensure_crs(buffered.to_crs(CRS))


def union(a: gpd.GeoDataFrame, b: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Combine two GeoDataFrames' geometries into their spatial union."""
    a, b = ensure_crs(a), ensure_crs(b)
    if a.empty:
        return b.copy()
    if b.empty:
        return a.copy()
    merged_geom = a.geometry.unary_union.union(b.geometry.unary_union)
    return from_geometries([merged_geom])


def intersection(a: gpd.GeoDataFrame, b: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Keep only the overlapping area between two GeoDataFrames."""
    a, b = ensure_crs(a), ensure_crs(b)
    if a.empty or b.empty:
        return empty_gdf()
    result_geom = a.geometry.unary_union.intersection(b.geometry.unary_union)
    if result_geom.is_empty:
        return empty_gdf()
    return from_geometries([result_geom])


def difference(a: gpd.GeoDataFrame, b: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Remove from `a` any area that overlaps with `b` (a minus b)."""
    a, b = ensure_crs(a), ensure_crs(b)
    if a.empty:
        return empty_gdf()
    if b.empty:
        return a.copy()
    result_geom = a.geometry.unary_union.difference(b.geometry.unary_union)
    if result_geom.is_empty:
        return empty_gdf()
    return from_geometries([result_geom])


def add(a: gpd.GeoDataFrame, b: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Concatenate two GeoDataFrames' rows (set addition, not spatial union)."""
    import pandas as pd

    a, b = ensure_crs(a), ensure_crs(b)
    if a.empty:
        return b.copy()
    if b.empty:
        return a.copy()
    combined = pd.concat([a, b], ignore_index=True)
    return gpd.GeoDataFrame(combined, geometry=GEOMETRY_COL, crs=CRS)


# Dispatch table used by the executor.
TOOL_DISPATCH = {
    "buffer": buffer,
    "union": union,
    "intersection": intersection,
    "difference": difference,
    "add": add,
}
