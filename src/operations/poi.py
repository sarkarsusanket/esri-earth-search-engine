"""
Point-of-interest (POI) search.

Tiered matching, from most specific to most vague:

  1. Amenity-class keyword match  ("cafe")     -> filter POIs by their real
                                                  OSM amenity tag directly.
  2. Brand / name search          ("starbucks")-> filter POIs whose *name*
                                                  column matches the query.
  3. Semantic search              ("emergency  -> embed the query and cosine-
                                     services")   score against the per-class
                                                  embeddings (all-MiniLM-L6-v2).

Every tier, when it produces a result, returns the real POIs (point geometry,
name, amenity class) optionally spatially filtered to a prior region, with the
pipeline schema columns plus `name` and `amenity`.
"""
import re
from typing import List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd

import config
from schema import GEOMETRY_COL, SCORE_COL, empty_gdf, ensure_crs

# Words that read as generic categories rather than specific destinations.
# When the query is just one of these and no amenity class matches, prefer the
# semantic tier over matching hundreds of "… Park" / "… Beach" names.
_GENERIC_CATEGORY_WORDS = {
    "park", "parks", "beach", "beaches", "plaza", "plazas", "square", "squares",
    "market", "markets", "mall", "malls", "area", "area", "zone", "zones",
}


def _normalize(query: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    return re.sub(r"[^a-z0-9]+", " ", query.lower()).strip()


def _tokenize(query: str) -> List[str]:
    return _normalize(query).split()


def _token_equiv(qt: str, ct: str) -> bool:
    """True if two tokens are the same modulo a trailing plural 's'."""
    if qt == ct:
        return True
    a, b = sorted((qt, ct), key=len)
    return len(b) == len(a) + 1 and b == a + "s"


def _amenity_class_hits(norm: str, poi_gdf: gpd.GeoDataFrame) -> List[str]:
    """Amenity tags that match any query token. Handles `;` multi-value tags
    and `_`-separated class names (e.g. `cafe;bakery`, `fast_food`)."""
    qtokens = _tokenize(norm)
    if not qtokens or "amenity" not in poi_gdf.columns:
        return []
    hits = []
    for tag in poi_gdf["amenity"].astype(str).unique():
        subtags = [s for s in tag.replace("_", " ").split(";") if s.strip()]
        if any(
            _token_equiv(qt, ct)
            for qt in qtokens
            for subtag in subtags
            for ct in _tokenize(subtag)
        ):
            hits.append(tag)
    return hits


def _name_hits(norm: str, poi_gdf: gpd.GeoDataFrame) -> List[int]:
    """Row indexes whose POI *name* contains a query token (case-insensitive).
    This is how an explicit brand/destination like "starbucks" is matched."""
    qtokens = _tokenize(norm)
    if not qtokens or "name" not in poi_gdf.columns:
        return []
    name_low = poi_gdf["name"].fillna("").astype(str).str.lower()
    return poi_gdf.index[name_low.apply(lambda nm: any(t in nm for t in qtokens))].tolist()


def _looks_specific(target: str) -> bool:
    """Heuristic: is this a specific place/brand rather than a generic
    category phrase? Brand names and multi-word destination phrases match."""
    toks = target.split()
    if not toks:
        return False
    if any(re.search(r"[A-Z]", t) for t in toks):
        return True
    if len(toks) >= 2:
        return True
    return _tokenize(target)[0] not in _GENERIC_CATEGORY_WORDS


def _trim(poi: gpd.GeoDataFrame, region: Optional[gpd.GeoDataFrame], score: float):
    """Apply optional region filter and shape the result to the pipeline
    schema. Returns None if nothing survives (so the caller can fall through
    to the next tier) or a schema-shaped GeoDataFrame."""
    if region is not None and not region.empty:
        region_union = ensure_crs(region).geometry.unary_union
        poi = poi[poi.geometry.within(region_union)]
        if poi.empty:
            return None
    poi[SCORE_COL] = float(score)
    columns = [GEOMETRY_COL, SCORE_COL] + [
        c for c in ("name", "amenity") if c in poi.columns
    ]
    return ensure_crs(poi[columns])


def search_poi(
    target: Optional[str],
    region: Optional[gpd.GeoDataFrame],
    poi_gdf: gpd.GeoDataFrame,
    poi_embedding_df: pd.DataFrame,
    text_embedder,
    threshold: float = config.POI_THRESHOLD,
) -> gpd.GeoDataFrame:
    """Search POIs by specificity: amenity class keyword > brand name > semantic."""
    if not target:
        print("POI search needs a non-empty query.")
        return empty_gdf()

    norm = _normalize(target)

    # --- Tier 1: amenity-class keyword match ("cafe") ---------------------
    class_hits = _amenity_class_hits(norm, poi_gdf)
    if class_hits:
        cand = poi_gdf[poi_gdf["amenity"].isin(class_hits)].copy()
        if not cand.empty:
            res = _trim(cand, region, score=1.0)
            if res is not None:
                print(f"POI [class] query {target!r} matched amenity class(es): "
                      f"{class_hits[:10]}{'...' if len(class_hits) > 10 else ''}")
                return res

    # --- Tier 2: brand / name search ("starbucks") -----------------------
    if _looks_specific(target):
        name_idx = _name_hits(norm, poi_gdf)
        if name_idx:
            cand = poi_gdf.loc[name_idx].copy()
            if not cand.empty:
                res = _trim(cand, region, score=1.0)
                if res is not None:
                    print(f"POI [name] query {target!r} matched {len(cand)} named POI(s).")
                    return res

    # --- Tier 3: semantic search ("emergency services") -------------------
    query_emb = text_embedder.encode(target).astype(np.float32)  # (384,), normalized
    emb_matrix = np.stack(poi_embedding_df["embedding"].values).astype(np.float32)
    sims = emb_matrix @ query_emb  # both sides unit-normalized -> cosine

    pass_mask = sims > threshold
    if not pass_mask.any():
        print(
            f"No POI amenity classes matched query {target!r} above threshold "
            f"{threshold} (max sim {sims.max():.4f})."
        )
        return empty_gdf()

    matched_amenities = poi_embedding_df.loc[pass_mask, "amenities"].tolist()
    amenity_to_score = dict(
        zip(poi_embedding_df.loc[pass_mask, "amenities"], sims[pass_mask])
    )
    print(
        f"POI [semantic] matched {len(matched_amenities)} amenity class(es) above "
        f"{threshold}: {matched_amenities[:10]}{'...' if len(matched_amenities) > 10 else ''}"
    )

    poi = poi_gdf[poi_gdf["amenity"].isin(matched_amenities)].copy()
    if poi.empty:
        print("No POIs found for the matched amenity classes.")
        return empty_gdf()

    if region is not None and not region.empty:
        region_union = ensure_crs(region).geometry.unary_union
        poi = poi[poi.geometry.within(region_union)]
        if poi.empty:
            print("No matched POIs lie inside the given region.")
            return empty_gdf()

    poi[SCORE_COL] = poi["amenity"].map(amenity_to_score).astype(float)
    columns = [GEOMETRY_COL, SCORE_COL] + [
        c for c in ("name", "amenity") if c in poi.columns
    ]
    return ensure_crs(poi[columns])