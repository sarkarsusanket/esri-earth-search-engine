"""
Shared GeoDataFrame conventions.

Every pipeline step (geocode / demo / vision / tool) returns a GeoDataFrame
that always has a `geometry` column and, where applicable, a `score`
column. Standardizing on this shape means tool operations (buffer, union,
intersection, difference) can operate on the output of *any* step without
needing to know which operation produced it.
"""
import geopandas as gpd

GEOMETRY_COL = "geometry"
SCORE_COL = "score"
CRS = "EPSG:4326"


def empty_gdf() -> gpd.GeoDataFrame:
    """Return a well-formed, empty GeoDataFrame matching the pipeline schema."""
    return gpd.GeoDataFrame(columns=[GEOMETRY_COL, SCORE_COL], geometry=GEOMETRY_COL, crs=CRS)


def from_geometries(geometries, scores=None) -> gpd.GeoDataFrame:
    """Build a standard-schema GeoDataFrame from a list of geometries (+ optional scores)."""
    data = {GEOMETRY_COL: geometries}
    if scores is not None:
        data[SCORE_COL] = scores
    return gpd.GeoDataFrame(data, geometry=GEOMETRY_COL, crs=CRS)


def ensure_crs(gdf: gpd.GeoDataFrame, crs: str = CRS) -> gpd.GeoDataFrame:
    """Make sure a GeoDataFrame is tagged with the pipeline's working CRS."""
    if gdf.crs is None:
        return gdf.set_crs(crs)
    if str(gdf.crs) != crs:
        return gdf.to_crs(crs)
    return gdf
