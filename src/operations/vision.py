"""
Visual similarity search, backed by a TurboQuant-compressed embedding index
per resolution (see turboquant_index.TurboQuantSearchIndex).

Replaces the previous raw-NPZ memory-mapped brute-force GPU scan: the index
is already built (rotated, IVF-clustered, bit-packed) ahead of time by
build_turboquant_index.py, so a query here only ever decompresses and scores
a small fraction of the index (nprobe clusters for global search, or the
region-filtered candidate rows for spatially-restricted search).
"""
from typing import Optional

import geopandas as gpd

import config
from schema import empty_gdf, from_geometries


def search_vision(target: str,
                   region: Optional[gpd.GeoDataFrame],
                   vision_encoder: "models.VisionEncoder",
                   turbo_index: Optional["turboquant_index.TurboQuantSearchIndex"],
                   resolution: str = config.DEFAULT_RESOLUTION,
                   top_n: int = config.VISION_TOP_N_DEFAULT,
                   nprobe: int = config.VISION_NPROBE_DEFAULT) -> gpd.GeoDataFrame:
    """Find the top-N tile locations whose visual embedding best matches a
    text query, optionally restricted to `region`."""
    if turbo_index is None:
        print(f"No vision index loaded for resolution '{resolution}'.")
        return empty_gdf()

    query_vector = vision_encoder.encode_text(target)  # (1, D), normalized, on DEVICE
    query_np = query_vector.squeeze(0).detach().cpu().numpy()

    if region is not None and not region.empty:
        print(f"[{resolution}] Region-filtered TurboQuant search...")
    else:
        print(f"[{resolution}] Global IVF-routed TurboQuant search (nprobe={nprobe})...")

    scores, lat, lon = turbo_index.search(query_np, top_k=top_n, region=region, nprobe=nprobe)

    if len(scores) == 0:
        print(f"No tiles at resolution '{resolution}' matched (or none fall inside the given region).")
        return empty_gdf()

    geometries = gpd.points_from_xy(lon, lat)
    return from_geometries(geometries, scores=scores)
