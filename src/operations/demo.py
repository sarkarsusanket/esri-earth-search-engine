"""Demographic similarity search against the TabularTextCLIP embedding space."""
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import geopandas as gpd

from config import *
from models import *
from schema import *


def search_demographics(target: Optional[str],
                         region: Optional[gpd.GeoDataFrame],
                         demo_gdf: gpd.GeoDataFrame,
                         ae_embeddings: torch.Tensor,
                         clip_model: torch.nn.Module,
                         top_k: int = DEMO_TOP_K_DEFAULT) -> gpd.GeoDataFrame:
    """Rank demographic polygons by similarity to a free-text query, optionally
    restricted to a prior region.

    If `target` is None/empty, the entire (optionally region-restricted)
    demographic layer is returned unranked.
    """
    candidates = demo_gdf
    candidate_embeddings = ae_embeddings

    if region is not None and not region.empty:
        region_union = region.geometry.unary_union
        mask = demo_gdf.geometry.intersects(region_union)
        if mask.any():
            candidates = demo_gdf[mask]
            candidate_embeddings = ae_embeddings[mask.values]
        else:
            print("Demographic layer does not intersect the given region; searching globally instead.")

    if not target:
        result = candidates.copy()
        if SCORE_COL not in result.columns:
            result[SCORE_COL] = 1.0
        return ensure_crs(result)

    text_embedding = get_text_embedding(target)
    text_tensor = torch.from_numpy(np.array([text_embedding], dtype=np.float32)).to(DEVICE)

    with torch.no_grad():
        tabular_latents = F.normalize(clip_model.geo_projector(candidate_embeddings), p=2, dim=-1).float()
        text_latent = F.normalize(clip_model.text_projector(text_tensor), p=2, dim=-1).float()
        scores = torch.matmul(tabular_latents, text_latent.t()).squeeze(-1).cpu().numpy()

    scores = _min_max_normalize(scores)

    top_indices = np.argsort(scores)[::-1][:top_k]
    matched = candidates.iloc[top_indices].copy()
    matched[SCORE_COL] = scores[top_indices]
    return ensure_crs(matched)


def _min_max_normalize(scores: np.ndarray) -> np.ndarray:
    s_min, s_max = scores.min(), scores.max()
    if s_max - s_min > 0:
        return (scores - s_min) / (s_max - s_min)
    return np.ones_like(scores)
