"""
Point-of-interest (POI) search.

Free-text query is embedded with the same local text embedder the demo search
uses (all-MiniLM-L6-v2, 384-dim), then cosine-scored against the per-amenity
class embeddings baked into poi_embeddings.parquet. Every class scoring above
config.POI_THRESHOLD "passes", and all real POIs (poi.parquet) tagged with
those amenity classes are kept, optionally spatially filtered to a prior
region. Result rows carry the POI's name and amenity class plus a score.
"""
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd

import config
from schema import GEOMETRY_COL, SCORE_COL, empty_gdf, ensure_crs


def search_poi(
    target: Optional[str],
    region: Optional[gpd.GeoDataFrame],
    poi_gdf: gpd.GeoDataFrame,
    poi_embedding_df: pd.DataFrame,
    text_embedder,
    threshold: float = config.POI_THRESHOLD,
) -> gpd.GeoDataFrame:
    """Match POI amenity classes against a free-text query by embedding
    similarity, keep classes scoring above `threshold`, then return the real
    POIs of those classes, optionally restricted to `region`.

    The result GeoDataFrame has the pipeline schema columns plus `name` and
    `amenity` (the POI's point geometry, name, and amenity class).
    """
    if not target:
        print("POI search needs a non-empty query.")
        return empty_gdf()

    query_emb = text_embedder.encode(target).astype(np.float32)  # (384,), normalized
    emb_matrix = np.stack(poi_embedding_df["embedding"].values).astype(np.float32)

    # Both sides are unit-normalized, so a dot product is the cosine similarity.
    sims = emb_matrix @ query_emb

    pass_mask = sims > threshold
    if not pass_mask.any():
        print(
            f"No POI amenity classes matched query {target!r} "
            f"above threshold {threshold} (max sim {sims.max():.4f})."
        )
        return empty_gdf()

    matched_amenities = poi_embedding_df.loc[pass_mask, "amenities"].tolist()
    amenity_to_score = dict(
        zip(poi_embedding_df.loc[pass_mask, "amenities"], sims[pass_mask])
    )
    print(
        f"Matched {len(matched_amenities)} amenity class(es) above {threshold}: "
        f"{matched_amenities[:10]}{'...' if len(matched_amenities) > 10 else ''}"
    )

    poi = poi_gdf[poi_gdf["amenity"].isin(matched_amenities)].copy()
    if poi.empty:
        print("No POIs found for the matched amenity classes.")
        return empty_gdf()

    if region is not None and not region.empty:
        region_ll = ensure_crs(region)
        region_union = region_ll.geometry.unary_union
        inside = poi.geometry.within(region_union)
        poi = poi[inside]
        if poi.empty:
            print("No matched POIs lie inside the given region.")
            return empty_gdf()

    poi[SCORE_COL] = poi["amenity"].map(amenity_to_score).astype(float)
    columns = [GEOMETRY_COL, SCORE_COL] + [
        c for c in ("name", "amenity") if c in poi.columns
    ]
    return ensure_crs(poi[columns])
