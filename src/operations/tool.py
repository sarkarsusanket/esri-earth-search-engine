"""
Spatial tool operations.

These are pure functions over GeoDataFrames that already conform to the
pipeline's standard schema (a `geometry` column, optionally a `score`
column). They never touch embeddings or models — only geometry.

Key design decisions:
  - intersection: auto-buffers point/line geometries before intersecting
  - All operations preserve individual feature rows (no unary_union collapse)
  - Geometry types are preserved where possible
"""
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString

from schema import GEOMETRY_COL, CRS, empty_gdf, from_geometries, ensure_crs


# Default buffer distances for intersection when geometries aren't polygons
_POINT_BUFFER_KM = 5.0
_LINE_BUFFER_KM = 0.01  # ~10 meters


def _needs_buffer(geom) -> bool:
    """True if the geometry is a Point or MultiPoint."""
    return isinstance(geom, (Point,))


def _is_line(geom) -> bool:
    """True if the geometry is a LineString or MultiLineString."""
    return isinstance(geom, LineString)


def _auto_buffer(gdf: gpd.GeoDataFrame, distance_km: float) -> gpd.GeoDataFrame:
    """Buffer geometries that aren't polygons. Returns a copy."""
    gdf = gdf.copy()
    metric = gdf.to_crs(gdf.estimate_utm_crs())
    new_geoms = []
    for geom in metric.geometry:
        if geom is None or geom.is_empty:
            new_geoms.append(geom)
            continue
        if _needs_buffer(geom):
            new_geoms.append(geom.buffer(distance_km * 1000))
        elif _is_line(geom):
            new_geoms.append(geom.buffer(_LINE_BUFFER_KM * 1000))
        else:
            new_geoms.append(geom)
    gdf[GEOMETRY_COL] = gpd.GeoSeries(new_geoms, crs=metric.crs).to_crs(CRS)
    return gdf


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
    """Combine two GeoDataFrames' rows into one (set addition, not spatial merge).

    Preserves individual features from both inputs. Use `add` for the same
    result without deduplication.
    """
    a, b = ensure_crs(a), ensure_crs(b)
    if a.empty:
        return b.copy()
    if b.empty:
        return a.copy()
    combined = pd.concat([a, b], ignore_index=True)
    return gpd.GeoDataFrame(combined, geometry=GEOMETRY_COL, crs=CRS)


def intersection(a: gpd.GeoDataFrame, b: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Keep only the overlapping area between two GeoDataFrames.

    For each feature in `a`, intersects it with the union of all features
    in `b`. Point/line geometries are auto-buffered before intersecting.
    Preserves individual features from `a`.
    """
    a = a.to_crs("EPSG:4326")
    b = b.to_crs("EPSG:4326")

    print(a.crs, b.crs)
    if a.empty or b.empty:
        return empty_gdf()

    print(len(a), len(b))

    # Auto-buffer non-polygon geometries for meaningful intersection
    a_buf = _auto_buffer(a, _POINT_BUFFER_KM) if any(_needs_buffer(g) or _is_line(g) for g in a.geometry) else a
    b_buf = _auto_buffer(b, _POINT_BUFFER_KM) if any(_needs_buffer(g) or _is_line(g) for g in b.geometry) else b

    b_union = b_buf.geometry.unary_union
    if b_union is None or b_union.is_empty:
        return empty_gdf()

    result_rows = []
    for idx, geom in a_buf.geometry.items():
        if geom is None or geom.is_empty:
            continue
        inter = geom.intersection(b_union)
        if not inter.is_empty:
            row = a_buf.loc[idx].copy()
            row[GEOMETRY_COL] = inter
            result_rows.append(row)

    print(len(result_rows))

    if not result_rows:
        return empty_gdf()
    return gpd.GeoDataFrame(result_rows, geometry=GEOMETRY_COL, crs=CRS)


def difference(a: gpd.GeoDataFrame, b: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Remove from `a` any area that overlaps with `b` (a minus b).

    For each feature in `a`, subtracts the union of all features in `b`.
    Preserves individual features from `a`.
    """
    a, b = ensure_crs(a), ensure_crs(b)
    if a.empty:
        return empty_gdf()
    if b.empty:
        return a.copy()

    b_union = b.geometry.unary_union
    if b_union is None or b_union.is_empty:
        return a.copy()

    result_rows = []
    for idx, geom in a.geometry.items():
        if geom is None or geom.is_empty:
            continue
        diff = geom.difference(b_union)
        if not diff.is_empty:
            row = a.loc[idx].copy()
            row[GEOMETRY_COL] = diff
            result_rows.append(row)

    if not result_rows:
        return empty_gdf()
    return gpd.GeoDataFrame(result_rows, geometry=GEOMETRY_COL, crs=CRS)


def add(a: gpd.GeoDataFrame, b: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Concatenate two GeoDataFrames' rows (set addition, not spatial union)."""
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
