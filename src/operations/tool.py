"""
Spatial tool operations.

These are pure functions over GeoDataFrames that perform straightforward
spatial operations over standard spatial schemas.
"""
import pandas as pd
import geopandas as gpd
import shapely

from schema import GEOMETRY_COL, CRS, empty_gdf, ensure_crs


def shapely_overlay(
    df1: gpd.GeoDataFrame, 
    df2: gpd.GeoDataFrame, 
    how: str = "intersection"
) -> gpd.GeoDataFrame:
    """Fast, robust overlay alternative to gpd.overlay.
    
    Handles mixed geometry types (Points, Lines, Polygons) without crashing.
    Supported modes for `how`: 'intersection', 'difference', 'union', 
    'symmetric_difference', 'identity'.
    """
    if df1.empty or df2.empty:
        if how in ("intersection", "inner"):
            return gpd.GeoDataFrame(columns=df1.columns, crs=df1.crs)
        elif how == "difference":
            return df1.copy()

    operations = {
        "intersection": shapely.intersection,
        "difference": shapely.difference,
        "union": shapely.union,
        "symmetric_difference": shapely.symmetric_difference,
    }
    
    how_op = "intersection" if how in ("intersection", "inner", "identity") else how
    if how_op not in operations:
        raise ValueError(f"Unsupported 'how' mode: {how}.")

    geom_col = df1._geometry_column_name

    # 1. Spatial Join to identify overlapping pairs
    joined_raw = gpd.sjoin(df1, df2, how="inner", predicate="intersects")

    if not joined_raw.empty:
        # Extract corresponding geometries
        geoms1 = joined_raw.geometry.to_numpy()
        
        # FIX: Use .loc[] instead of .iloc[] because index_right holds index labels
        geoms2 = df2.geometry.loc[joined_raw["index_right"]].to_numpy()

        # Execute vectorized C-level spatial operation
        intersected_geoms = operations[how_op](geoms1, geoms2)

        # Replace geometries and drop invalid/empty geometries
        joined = joined_raw.copy()
        joined[geom_col] = intersected_geoms
        joined = joined[~joined.geometry.is_empty & joined.geometry.notna()].copy()
        joined = joined.drop(columns=["index_right"], errors="ignore")
    else:
        joined = gpd.GeoDataFrame(columns=df1.columns, crs=df1.crs)

    # 2. Handle non-intersecting geometry portions for modes that require them
    if how in ("difference", "identity") or (how == "intersection" and joined.empty):
        # Reuse the sjoin already computed above instead of re-running it —
        # `joined_raw.index` (still df1's original index labels at this
        # point, before any dedup/filtering) already tells us exactly which
        # df1 rows had ANY match; a second `gpd.sjoin` call here was doing
        # the identical spatial join twice for every difference/identity
        # overlay.
        matched_idx = joined_raw.index.unique()
        unmatched_idx = df1.index.difference(matched_idx)
        unmatched_df1 = df1.loc[unmatched_idx].copy()

        if how in ("difference", "identity"):
            joined = pd.concat([joined, unmatched_df1], ignore_index=True)
            if not isinstance(joined, gpd.GeoDataFrame):
                joined = gpd.GeoDataFrame(joined, crs=df1.crs, geometry=geom_col)

    return joined


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

def add(a: gpd.GeoDataFrame, b: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Concatenate two GeoDataFrames' rows together without spatial merging."""
    a, b = ensure_crs(a), ensure_crs(b)
    if a.empty:
        return b.copy()
    if b.empty:
        return a.copy()
    combined = pd.concat([a, b], ignore_index=True)
    return gpd.GeoDataFrame(combined, geometry=GEOMETRY_COL, crs=CRS)


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
    return add(a, b)
    # a, b = ensure_crs(a), ensure_crs(b)
    # if a.empty:
    #     return b.copy()
    # if b.empty:
    #     return a.copy()
    # return shapely_overlay(a, b, how="union")


def intersection(a: gpd.GeoDataFrame, b: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Keep only the overlapping spatial intersection between two GeoDataFrames."""
    a, b = ensure_crs(a), ensure_crs(b)
    if a.empty or b.empty:
        return empty_gdf()
    return shapely_overlay(a, b, how="intersection")


def difference(a: gpd.GeoDataFrame, b: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Subtract spatial features of `b` from `a` (a minus b)."""
    a, b = ensure_crs(a), ensure_crs(b)
    if a.empty:
        return empty_gdf()
    if b.empty:
        return a.copy()
    return shapely_overlay(a, b, how="difference")


# Dispatch table used by the executor.
TOOL_DISPATCH = {
    "buffer": buffer,
    "get_centroid": get_centroid,
    "union": union,
    "intersection": intersection,
    "difference": difference,
    "add": add,
}