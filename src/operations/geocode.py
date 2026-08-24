"""
Geocoding: resolve a place name to a bounding-box polygon.

NOTE: this is the one remaining network dependency in the pipeline (calls
Nominatim's public geocoding API). If you need fully offline operation,
swap this for a local gazetteer/offline geocoder — everything downstream
(demo/vision/tool) is already fully local.
"""

# import geopandas as gpd
# from geopy.geocoders import Nominatim
# from shapely.geometry import Polygon

# from schema import empty_gdf, from_geometries

# _GEOLOCATOR = Nominatim(user_agent="queryearth_dlpk")


# def geocode(target: str) -> gpd.GeoDataFrame:
#     """Resolve a place name to its bounding box, returned as a single-row
#     GeoDataFrame in the standard pipeline schema."""
#     if not target:
#         return empty_gdf()

#     try:
#         location = _GEOLOCATOR.geocode(target, timeout=10)
#     except Exception as e:
#         print(f"Geocoding failed for '{target}': {e}")
#         return empty_gdf()

#     if location is None or "boundingbox" not in location.raw:
#         print(f"No bounding box found for '{target}'.")
#         return empty_gdf()

#     lat_min, lat_max, lon_min, lon_max = map(float, location.raw["boundingbox"])
#     polygon = Polygon(
#         [
#             (lon_min, lat_min),
#             (lon_max, lat_min),
#             (lon_max, lat_max),
#             (lon_min, lat_max),
#         ]
#     )
#     return from_geometries([polygon])
import geopandas as gpd
import requests
from shapely.geometry import Polygon, shape
from schema import empty_gdf, from_geometries


def geocode(target: str) -> gpd.GeoDataFrame:
    """Resolve a place name to its EXACT administrative boundary polygon.

    Falls back to Bounding Box if exact polygon geometry is unavailable.
    """
    if not target:
        return empty_gdf()

    headers = {
        "User-Agent": "QueryEarthPipeline/1.0 (contact: admin@queryearth.local)"
    }

    # 1. Try Nominatim search with polygon_geojson enabled
    nominatim_url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": target,
        "format": "json",
        "polygon_geojson": 1,
        "limit": 1,
    }

    try:
        res = requests.get(
            nominatim_url, params=params, headers=headers, timeout=10
        )

        if res.status_code == 200:
            data = res.json()
            if data and len(data) > 0:
                item = data[0]
                geojson_geom = item.get("geojson")

                # Ensure geometry is a valid Polygon or MultiPolygon
                if geojson_geom and geojson_geom.get("type") in [
                    "Polygon",
                    "MultiPolygon",
                ]:
                    poly = shape(geojson_geom)
                    return from_geometries([poly])

                # If no polygon geometry, fallback to bounding box from Nominatim
                if "boundingbox" in item:
                    # Nominatim returns [south, north, west, east]
                    s, n, w, e = map(float, item["boundingbox"])
                    bbox_poly = Polygon(
                        [(w, s), (e, s), (e, n), (w, n), (w, s)]
                    )
                    return from_geometries([bbox_poly])

    except Exception as e:
        print(f"Nominatim polygon lookup failed for '{target}': {e}")

    # 2. Fallback to Esri Geocoding Bounding Box if Nominatim fails
    try:
        esri_url = "https://geocode.arcgis.com/arcgis/rest/services/World/GeocodeServer/findAddressCandidates"
        esri_params = {
            "SingleLine": target,
            "f": "json",
            "maxLocations": 1,
            "outFields": "extent",
        }
        res = requests.get(esri_url, params=esri_params, timeout=10)
        if res.status_code == 200:
            data = res.json()
            candidates = data.get("candidates", [])
            if candidates and "extent" in candidates[0]:
                ext = candidates[0]["extent"]
                bbox_poly = Polygon(
                    [
                        (ext["xmin"], ext["ymin"]),
                        (ext["xmax"], ext["ymin"]),
                        (ext["xmax"], ext["ymax"]),
                        (ext["xmin"], ext["ymax"]),
                        (ext["xmin"], ext["ymin"]),
                    ]
                )
                return from_geometries([bbox_poly])
    except Exception as e:
        print(f"Esri bbox fallback failed for '{target}': {e}")

    return empty_gdf()