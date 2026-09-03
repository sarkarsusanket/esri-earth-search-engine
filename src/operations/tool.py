"""
Spatial tool operations.

These are pure functions over GeoDataFrames that perform straightforward
spatial operations over standard spatial schemas.
"""
import pandas as pd
import geopandas as gpd

from schema import GEOMETRY_COL, CRS, empty_gdf, ensure_crs


def buffer(gdf: gpd.GeoDataFrame, distance_km: float) -> gpd.GeoDataFrame:
    """Buffer every geometry in `gdf` outward by `distance_km` kilometers."""
    if gdf is None or gdf.empty:
        return empty_gdf()
    if distance_km is None:
        raise ValueError("buffer operation requires a 'distance_km' parameter.")

    gdf = ensure_crs(gdf)
    # Project to metric CRS for accurate calculation, then project back to standard CRS
    metric = gdf.to_crs(gdf.estimate_utm_crs())
    buffered = metric.copy()
    buffered[GEOMETRY_COL] = metric.geometry.buffer(distance_km * 1000)
    return ensure_crs(buffered.to_crs(CRS))


def get_centroid(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Calculate and return the centroid for every geometry in `gdf`."""
    if gdf is None or gdf.empty:
        return empty_gdf()

    gdf = ensure_crs(gdf)
    # Estimate metric CRS for accurate spatial centroid calculation
    metric = gdf.to_crs(gdf.estimate_utm_crs())
    result = metric.copy()
    result[GEOMETRY_COL] = metric.geometry.centroid
    return ensure_crs(result.to_crs(CRS))


def union(a: gpd.GeoDataFrame, b: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Perform a spatial union overlay between two GeoDataFrames."""
    a, b = ensure_crs(a), ensure_crs(b)
    if a.empty:
        return b.copy()
    if b.empty:
        return a.copy()
    return gpd.overlay(a, b, how="union")


def intersection(a: gpd.GeoDataFrame, b: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Keep only the overlapping spatial intersection between two GeoDataFrames."""
    a, b = ensure_crs(a), ensure_crs(b)
    if a.empty or b.empty:
        return empty_gdf()
    return gpd.overlay(a, b, how="intersection")


def difference(a: gpd.GeoDataFrame, b: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Subtract spatial features of `b` from `a` (a minus b)."""
    a, b = ensure_crs(a), ensure_crs(b)
    if a.empty:
        return empty_gdf()
    if b.empty:
        return a.copy()
    return gpd.overlay(a, b, how="difference")


def add(a: gpd.GeoDataFrame, b: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Concatenate two GeoDataFrames' rows together without spatial merging."""
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
    "get_centroid": get_centroid,
    "union": union,
    "intersection": intersection,
    "difference": difference,
    "add": add,
}