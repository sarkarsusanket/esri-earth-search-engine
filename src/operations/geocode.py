"""Geocoding: resolve a place name to a bounding-box polygon."""

from typing import Optional

import geopandas as gpd
from geopy.geocoders import Nominatim
from shapely.geometry import Polygon

from schema import *

_GEOLOCATOR = Nominatim(user_agent="queryearth_orchestrator")


def geocode(target: str) -> gpd.GeoDataFrame:
    """Resolve a place name to its bounding box, returned as a single-row
    GeoDataFrame in the standard pipeline """
    if not target:
        return empty_gdf()

    try:
        location = _GEOLOCATOR.geocode(target, timeout=10)
    except Exception as e:
        print(f"Geocoding failed for '{target}': {e}")
        return empty_gdf()

    if location is None or "boundingbox" not in location.raw:
        print(f"No bounding box found for '{target}'.")
        return empty_gdf()

    lat_min, lat_max, lon_min, lon_max = map(float, location.raw["boundingbox"])
    polygon = Polygon(
        [
            (lon_min, lat_min),
            (lon_max, lat_min),
            (lon_max, lat_max),
            (lon_min, lat_max),
        ]
    )
    return from_geometries([polygon])
