"""
OpenStreetMap (OSM) search.

Replaces the legacy POI search with a unified OSM-based search that covers
roads, waterways, buildings, landuse, natural features, and points of interest.

The search is mode-driven: each mode maps to a specific OSM parquet file
with its own category column and schema.

Modes and their schemas:
  - roads:      highway column (residential, primary, motorway, etc.)
  - waterways:  waterway column (river, stream, canal, etc.)
  - buildings:  amenity column + name (restaurant, school, etc.)
  - landuse:    landuse column (residential, commercial, forest, etc.)
  - natural:    natural column (peak, beach, forest, etc.)
  - pois:       amenity column + name (detailed POI data)

Search methods:
  - keyword:  Direct category/name matching (fast, exact)
  - semantic: Embedding-based similarity (flexible, fuzzy)

When method="keyword" is used but no matches are found, automatically
falls back to semantic search if embeddings are available.
"""
import os
import re
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd

import config
from schema import GEOMETRY_COL, SCORE_COL, empty_gdf, ensure_crs


# Maps mode -> (parquet filename, category column, has_name_col)
MODE_REGISTRY = {
    "roads":     ("roads.parquet",     "highway",  False),
    "waterways": ("waterway.parquet",  "waterway", False),
    "buildings": ("buildings.parquet", "amenity",  True),
    "landuse":   ("landuse.parquet",   "landuse",  False),
    "natural":   ("natural.parquet",   "natural",  False),
    "pois":      ("pois.parquet",      "amenity",  True),
}

SUPPORTED_MODES = set(MODE_REGISTRY.keys())

SUPPORTED_METHODS = {"keyword", "semantic"}


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


def _resolve_mode_and_query(
    arg1: Optional[str],
    arg2: Optional[str],
) -> Tuple[str, Optional[str]]:
    """Determine which argument is the mode and which is the query.

    The model may write osm("primary", "roads") or osm("roads", "primary").
    We detect the mode by checking which argument matches a supported mode.
    If both match, the first matching one is treated as mode.
    If neither matches, assume (arg1=query, arg2=mode) is invalid and return
    the first as query, second as mode (will fail validation downstream).

    Returns:
        (mode, query) tuple
    """
    if arg1 is None and arg2 is None:
        return "roads", None

    if arg1 is None:
        # Only arg2 provided - if it's a mode, use it with empty query
        if arg2 and arg2.lower() in SUPPORTED_MODES:
            return arg2.lower(), None
        return "roads", arg2

    if arg2 is None:
        # Only arg1 provided - if it's a mode, use it with empty query
        if arg1.lower() in SUPPORTED_MODES:
            return arg1.lower(), None
        return "roads", arg1

    # Both provided - check which is the mode
    arg1_lower = arg1.lower()
    arg2_lower = arg2.lower()

    arg1_is_mode = arg1_lower in SUPPORTED_MODES
    arg2_is_mode = arg2_lower in SUPPORTED_MODES

    if arg1_is_mode and not arg2_is_mode:
        # arg1 is mode, arg2 is query
        return arg1_lower, arg2
    elif arg2_is_mode and not arg1_is_mode:
        # arg2 is mode, arg1 is query
        return arg2_lower, arg1
    elif arg1_is_mode and arg2_is_mode:
        # Both are modes - treat first as mode, second as query
        # (e.g., osm("roads", "waterways") -> search waterways in roads mode)
        return arg1_lower, arg2
    else:
        # Neither is a mode - treat as (query, mode) with fallback
        # This handles cases where the model doesn't use a valid mode name
        return arg2_lower, arg1


def _category_hits(query: str, gdf: gpd.GeoDataFrame, category_col: str) -> List[str]:
    """Category tags that match any query token. Handles `;` multi-value tags
    and `_`-separated class names (e.g. `motorway_link`, `fast_food`)."""
    qtokens = _tokenize(query)
    if not qtokens or category_col not in gdf.columns:
        return []
    hits = []
    for tag in gdf[category_col].dropna().astype(str).unique():
        subtags = [s for s in tag.replace("_", " ").split(";") if s.strip()]
        if any(
            _token_equiv(qt, ct)
            for qt in qtokens
            for subtag in subtags
            for ct in _tokenize(subtag)
        ):
            hits.append(tag)
    return hits


def _name_hits(query: str, gdf: gpd.GeoDataFrame) -> List[int]:
    """Row indexes whose name contains a query token (case-insensitive)."""
    qtokens = _tokenize(query)
    if not qtokens or "name" not in gdf.columns:
        return []
    name_low = gdf["name"].fillna("").astype(str).str.lower()
    return gdf.index[name_low.apply(lambda nm: any(t in nm for t in qtokens))].tolist()


def _semantic_search(
    query: str,
    gdf: gpd.GeoDataFrame,
    category_col: str,
    category_embeddings: pd.DataFrame,
    text_embedder,
    top_k: int = 5,
) -> Tuple[List[str], Dict[str, float]]:
    """Embed the query and find the most similar category values.

    Args:
        query: Free-text search query.
        gdf: The OSM GeoDataFrame for this mode.
        category_col: Name of the category column (e.g. "highway", "amenity").
        category_embeddings: DataFrame with 'category' and 'embedding' columns.
        text_embedder: LocalTextEmbedder instance.
        top_k: Number of top matches to return.

    Returns:
        (matched_categories, category_to_score) tuple
    """
    if category_embeddings is None or category_embeddings.empty:
        return [], {}

    query_emb = text_embedder.encode(query).astype(np.float32)

    emb_matrix = np.stack(category_embeddings["embedding"].values).astype(np.float32)
    sims = emb_matrix @ query_emb

    # Get top-k matches
    top_indices = np.argsort(sims)[::-1][:top_k]
    matched_cats = []
    cat_to_score = {}
    for idx in top_indices:
        cat = category_embeddings.iloc[idx]["category"]
        score = float(sims[idx])
        if score > 0.2:  # minimum similarity threshold
            matched_cats.append(cat)
            cat_to_score[cat] = score

    return matched_cats, cat_to_score


def _keyword_search(
    query: str,
    gdf: gpd.GeoDataFrame,
    category_col: str,
    has_name: bool,
) -> Tuple[Optional[gpd.GeoDataFrame], List[str]]:
    """Perform keyword-based search: category match then name match.

    Returns:
        (result_gdf_or_None, matched_categories) tuple
    """
    norm = _normalize(query)

    # Tier 1: category keyword match
    cat_hits = _category_hits(norm, gdf, category_col)
    if cat_hits:
        cand = gdf[gdf[category_col].isin(cat_hits)].copy()
        if not cand.empty:
            return cand, cat_hits

    # Tier 2: name search (where available)
    if has_name:
        name_idx = _name_hits(norm, gdf)
        if name_idx:
            cand = gdf.loc[name_idx].copy()
            if not cand.empty:
                return cand, []

    return None, []


def _trim(
    result: gpd.GeoDataFrame,
    region: Optional[gpd.GeoDataFrame],
    score: float,
    extra_cols: Optional[List[str]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """Apply optional region filter and shape the result to the pipeline schema."""
    if region is not None and not region.empty:
        region_union = ensure_crs(region).geometry.unary_union
        result = result[result.geometry.within(region_union)]
        if result.empty:
            return None
    result = result.copy()
    result[SCORE_COL] = float(score)
    keep_cols = [GEOMETRY_COL, SCORE_COL]
    if extra_cols:
        keep_cols.extend(c for c in extra_cols if c in result.columns)
    return ensure_crs(result[keep_cols])


def load_osm_data(
    osm_dir: str = config.OSM_EMBEDDING_DIR,
    year: str = "latest",
) -> Dict[str, gpd.GeoDataFrame]:
    """Load all OSM parquet files for a given year into a dict keyed by mode."""
    year_dir = os.path.join(osm_dir, year)
    if not os.path.isdir(year_dir):
        print(f"OSM year directory not found: {year_dir}")
        return {}

    data = {}
    for mode, (filename, _, _) in MODE_REGISTRY.items():
        path = os.path.join(year_dir, filename)
        if os.path.isfile(path):
            print(f"Loading OSM {mode} from {path}...")
            data[mode] = gpd.read_parquet(path)
        else:
            print(f"OSM file not found for mode '{mode}': {path}")
    return data


def load_osm_category_embeddings(
    embed_dir: str,
) -> Dict[str, pd.DataFrame]:
    """Load category embeddings for each OSM mode.

    Expected directory structure:
        embed_dir/
            roads.parquet      (columns: category, embedding)
            waterway.parquet
            buildings.parquet
            landuse.parquet
            natural.parquet
            pois.parquet

    Returns:
        Dict of mode -> DataFrame with 'category' and 'embedding' columns.
    """
    if not os.path.isdir(embed_dir):
        print(f"OSM category embeddings directory not found: {embed_dir}")
        return {}

    embeddings = {}
    for mode, (filename, _, _) in MODE_REGISTRY.items():
        path = os.path.join(embed_dir, filename)
        if os.path.isfile(path):
            print(f"Loading OSM {mode} category embeddings from {path}...")
            df = pd.read_parquet(path)
            # Validate structure
            if "category" in df.columns and "embedding" in df.columns:
                embeddings[mode] = df
            else:
                print(f"Warning: {path} missing 'category' or 'embedding' columns, skipping.")
        else:
            print(f"OSM category embeddings not found for mode '{mode}': {path}")
    return embeddings


def search_osm(
    mode: str,
    query: Optional[str],
    region: Optional[gpd.GeoDataFrame],
    osm_data: Dict[str, gpd.GeoDataFrame],
    method: str = "keyword",
    text_embedder=None,
    category_embeddings: Optional[Dict[str, pd.DataFrame]] = None,
    threshold: float = 0.40,
) -> gpd.GeoDataFrame:
    """Search OSM data by mode with keyword or semantic matching.

    Args:
        mode: One of SUPPORTED_MODES (roads, waterways, buildings, landuse, natural, pois).
        query: Free-text search query (e.g. "primary", "rivers", "hospitals").
        region: Optional spatial filter GeoDataFrame.
        osm_data: Dict of mode -> GeoDataFrame (loaded by load_osm_data).
        method: "keyword" for direct matching, "semantic" for embedding similarity.
                When method="keyword" fails, falls back to semantic if available.
        text_embedder: LocalTextEmbedder for semantic search.
        category_embeddings: Dict of mode -> DataFrame with 'category' and 'embedding' columns.
        threshold: Minimum cosine similarity for semantic matches.

    Returns:
        GeoDataFrame with geometry + score columns, plus mode-specific metadata.
    """
    # Validate mode
    if mode not in SUPPORTED_MODES:
        print(f"OSM search: unsupported mode '{mode}'. Must be one of {SUPPORTED_MODES}.")
        return empty_gdf()

    if mode not in osm_data or osm_data[mode] is None or osm_data[mode].empty:
        print(f"OSM search: no data loaded for mode '{mode}'.")
        return empty_gdf()

    # Validate method
    if method not in SUPPORTED_METHODS:
        print(f"OSM search: unsupported method '{method}'. Falling back to 'keyword'.")
        method = "keyword"

    gdf = osm_data[mode]
    filename, category_col, has_name = MODE_REGISTRY[mode]

    # No query = return everything in region
    if not query:
        extra = [category_col]
        if has_name:
            extra.append("name")
        return _trim(gdf, region, score=1.0, extra_cols=extra) or empty_gdf()

    # --- Semantic search mode ---
    if method == "semantic":
        if text_embedder is None or not category_embeddings:
            print(f"OSM [{mode}] semantic search requested but no text_embedder or embeddings provided. "
                  f"Falling back to keyword.")
            method = "keyword"
        else:
            mode_embeddings = category_embeddings.get(mode)
            if mode_embeddings is None or mode_embeddings.empty:
                print(f"OSM [{mode}] no category embeddings available for semantic search. "
                      f"Falling back to keyword.")
                method = "keyword"

    # --- Keyword search ---
    if method == "keyword":
        cand, cat_hits = _keyword_search(query, gdf, category_col, has_name)
        if cand is not None:
            extra = [category_col]
            if has_name:
                extra.append("name")
            res = _trim(cand, region, score=1.0, extra_cols=extra)
            if res is not None:
                if cat_hits:
                    print(f"OSM [{mode}] query {query!r} matched category(es): "
                          f"{cat_hits[:10]}{'...' if len(cat_hits) > 10 else ''}")
                else:
                    print(f"OSM [{mode}] query {query!r} matched by name.")
                return res

        # Keyword failed - fall back to semantic if available
        if (text_embedder is not None and category_embeddings
                and mode in category_embeddings
                and category_embeddings[mode] is not None
                and not category_embeddings[mode].empty):
            print(f"OSM [{mode}] keyword search failed, falling back to semantic.")
            method = "semantic"
        else:
            print(f"OSM [{mode}] no matches found for query {query!r}.")
            return empty_gdf()

    # --- Semantic search (either direct or fallback) ---
    if method == "semantic":
        mode_embeddings = category_embeddings.get(mode)
        matched_cats, cat_to_score = _semantic_search(
            query, gdf, category_col, mode_embeddings, text_embedder
        )
        if matched_cats:
            cand = gdf[gdf[category_col].isin(matched_cats)].copy()
            if not cand.empty:
                # Use the highest similarity score as the result score
                best_score = max(cat_to_score.values()) if cat_to_score else 0.5
                extra = [category_col]
                if has_name:
                    extra.append("name")
                res = _trim(cand, region, score=best_score, extra_cols=extra)
                if res is not None:
                    print(f"OSM [{mode}] semantic search matched {len(matched_cats)} category(es): "
                          f"{matched_cats[:10]}{'...' if len(matched_cats) > 10 else ''}")
                    return res

    print(f"OSM [{mode}] no matches found for query {query!r}.")
    return empty_gdf()
