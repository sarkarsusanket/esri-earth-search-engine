"""OpenStreetMap (OSM) search module.

Provides unified OSM keyword search with spatial filtering and fuzzy term
normalization (plural/stem handling).
"""

import os
import re
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import pandas as pd

import config
from schema import GEOMETRY_COL, SCORE_COL, empty_gdf, ensure_crs

# Maps mode -> (parquet filename, category column, has_name_col)
MODE_REGISTRY = {
    "roads": ("roads.parquet", "highway", False),
    "waterways": ("waterway.parquet", "waterway", False),
    "buildings": ("buildings.parquet", "amenity", True),
    "landuse": ("landuse.parquet", "landuse", False),
    "natural": ("natural.parquet", "natural", False),
    "pois": ("pois.parquet", "amenity", True),
}

SUPPORTED_MODES = set(MODE_REGISTRY.keys())


def _stem(word: str) -> str:
    """Basic English lemmatizer/stemmer to strip common plurals.

    Handles cases like 'rivers' -> 'river', 'beaches' -> 'beach', 'cities' ->
    'city'.
    """
    if len(word) <= 3:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("shes", "ches", "sses", "boxes", "faxes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _normalize(query: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    return re.sub(r"[^a-z0-9]+", " ", query.lower()).strip()


def _tokenize(query: str) -> List[str]:
    return [_normalize(token) for token in _normalize(query).split() if token]


def _resolve_mode_and_query(
    arg1: Optional[str],
    arg2: Optional[str],
) -> Tuple[str, Optional[str]]:
    """Determine which argument is the mode and which is the query."""
    if arg1 is None and arg2 is None:
        return "roads", None

    if arg1 is None:
        if arg2 and arg2.lower() in SUPPORTED_MODES:
            return arg2.lower(), None
        return "roads", arg2

    if arg2 is None:
        if arg1.lower() in SUPPORTED_MODES:
            return arg1.lower(), None
        return "roads", arg1

    arg1_lower = arg1.lower()
    arg2_lower = arg2.lower()

    arg1_is_mode = arg1_lower in SUPPORTED_MODES
    arg2_is_mode = arg2_lower in SUPPORTED_MODES

    if arg1_is_mode and not arg2_is_mode:
        return arg1_lower, arg2
    elif arg2_is_mode and not arg1_is_mode:
        return arg2_lower, arg1
    elif arg1_is_mode and arg2_is_mode:
        return arg1_lower, arg2
    else:
        return arg2_lower, arg1


def _category_hits(query: str, gdf: gpd.GeoDataFrame, category_col: str) -> List[str]:
    """Find category tags that match any query token using root-stem comparison.

    Handles `;` multi-value tags and `_`-separated class names.
    """
    qtokens = _tokenize(query)
    if not qtokens or category_col not in gdf.columns:
        return []

    # Pre-stem query tokens
    q_stems = {_stem(qt) for qt in qtokens}

    hits = []
    unique_categories = gdf[category_col].dropna().astype(str).unique()

    for tag in unique_categories:
        # Split multi-value and compound tag names like "motorway_link" or "fast_food"
        subtags = [s for s in tag.replace("_", " ").split(";") if s.strip()]
        cat_tokens = [
            token for subtag in subtags for token in _tokenize(subtag)
        ]
        cat_stems = {_stem(ct) for ct in cat_tokens}

        # Match if any stemmed query token overlaps with any stemmed category token
        if not q_stems.isdisjoint(cat_stems):
            hits.append(tag)

    return hits


def _name_hits(query: str, gdf: gpd.GeoDataFrame) -> List[int]:
    """Vectorized row matching where name contains query tokens."""
    qtokens = _tokenize(query)
    if not qtokens or "name" not in gdf.columns:
        return []

    # Include original and stemmed tokens for regex matching
    all_variants = list(
        set(qtokens + [_stem(t) for t in qtokens if len(t) > 3])
    )
    pattern = "|".join([re.escape(t) for t in all_variants])

    name_series = gdf["name"].fillna("").astype(str)
    mask = name_series.str.contains(pattern, case=False, regex=True)

    return gdf.index[mask].tolist()


def _keyword_search(
    query: str,
    gdf: gpd.GeoDataFrame,
    category_col: str,
    has_name: bool,
) -> Tuple[Optional[gpd.GeoDataFrame], List[str]]:
    """Keyword search across category and name fields with union matching."""
    sub_queries = [q.strip() for q in query.split(",") if q.strip()]

    all_cat_hits = []
    all_name_indices = []

    for sq in sub_queries:
        norm = _normalize(sq)
        cat_hits = _category_hits(norm, gdf, category_col)
        all_cat_hits.extend(cat_hits)

        if has_name:
            name_idx = _name_hits(norm, gdf)
            all_name_indices.extend(name_idx)

    all_cat_hits = list(dict.fromkeys(all_cat_hits))
    all_name_indices = list(dict.fromkeys(all_name_indices))

    # Combine Category and Name search results using logical OR
    cat_mask = (
        gdf[category_col].isin(all_cat_hits)
        if all_cat_hits
        else pd.Series(False, index=gdf.index)
    )
    name_mask = (
        gdf.index.isin(all_name_indices)
        if all_name_indices
        else pd.Series(False, index=gdf.index)
    )

    combined_mask = cat_mask | name_mask

    if combined_mask.any():
        return gdf[combined_mask].copy(), all_cat_hits

    return None, []


def _trim(
    result: gpd.GeoDataFrame,
    region: Optional[gpd.GeoDataFrame],
    score: float,
    extra_cols: Optional[List[str]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """Fast spatial filter using spatial index (R-tree) and pipeline schema shaping."""
    if result.empty:
        return None

    if region is not None and not region.empty:
        region_clean = ensure_crs(region)
        result = ensure_crs(result)

        # Spatial Join using Spatial Index
        result = gpd.sjoin(
            result,
            region_clean[["geometry"]],
            how="inner",
            predicate="intersects",
        )
        if result.empty:
            return None

        # Clean up temporary sjoin index columns if created
        result = result.drop(columns=["index_right"], errors="ignore")

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


def search_osm(
    mode: str,
    query: Optional[str],
    region: Optional[gpd.GeoDataFrame],
    osm_data: Dict[str, gpd.GeoDataFrame],
) -> gpd.GeoDataFrame:
    """Search OSM features by mode, term, and bounding region."""
    if mode not in SUPPORTED_MODES:
        print(
            f"OSM search: unsupported mode '{mode}'. Must be one of"
            f" {SUPPORTED_MODES}."
        )
        return empty_gdf()

    if mode not in osm_data or osm_data[mode] is None or osm_data[mode].empty:
        print(f"OSM search: no data loaded for mode '{mode}'.")
        return empty_gdf()

    gdf = osm_data[mode]
    filename, category_col, has_name = MODE_REGISTRY[mode]

    extra = [category_col]
    if has_name:
        extra.append("name")

    if not query:
        res = _trim(gdf, region, score=1.0, extra_cols=extra)
        return res if res is not None else empty_gdf()

    cand, cat_hits = _keyword_search(query, gdf, category_col, has_name)
    if cand is not None:
        res = _trim(cand, region, score=1.0, extra_cols=extra)
        if res is not None:
            if cat_hits:
                print(
                    f"OSM [{mode}] query {query!r} matched category(es):"
                    f" {cat_hits[:10]}"
                )
            else:
                print(f"OSM [{mode}] query {query!r} matched by name.")
            return res

    print(
        f"OSM [{mode}] no keyword matches for {query!r}"
    )
    return empty_gdf()