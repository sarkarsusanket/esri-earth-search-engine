"""
Visual similarity search against CLIP image-tile embeddings.

Supports three spatial resolutions of the *same* underlying model's
embeddings, stored as separate NPZ files:

    low    -> embeddings_full.npz    (~2km tiles)
    medium -> embeddings_tile4.npz   (~500m tiles)
    high   -> embeddings_tile100.npz (~200m tiles)

Each NPZ is expected to contain:
    "embeddings": (M, D) float array of CLIP image embeddings
    "centers":    (M, 2) float array of [lat, lon] tile centers
"""
import os
from typing import Optional
import numpy as np
import tqdm
import torch
import geopandas as gpd

from config import *
from models import *
from schema import *

def _points_within_region(centers: np.ndarray, region: Optional[gpd.GeoDataFrame]) -> np.ndarray:
    """Return a boolean mask of which tile centers fall inside `region`."""
    if region is None or region.empty:
        return np.ones(len(centers), dtype=bool)

    points_gdf = gpd.GeoDataFrame(
        {"idx": np.arange(len(centers))},
        geometry=gpd.points_from_xy(centers[:, 1], centers[:, 0]),  # lon, lat
        crs=CRS,
    )
    region_union = ensure_crs(region).geometry.unary_union
    return points_gdf.within(region_union).values


def search_vision(target: str,
                  region: Optional[gpd.GeoDataFrame],
                  vision_encoder: "VisionEncoder",
                  resolution: str = DEFAULT_RESOLUTION,
                  top_n: int = VISION_TOP_N_DEFAULT,
                  batch_size: int = 10000) -> gpd.GeoDataFrame:
    """
    Find the top-N tiles whose visual embedding matches a text query using a 
    rolling memory-mapped stream search with GPU acceleration.
    """
    filename = RESOLUTION_TO_FILE[resolution]
    path = os.path.join(VISION_EMBEDDINGS_DIR, filename)
    if not os.path.exists(path):
        print(f"Vision embedding file not found at {path}.")
        return empty_gdf()

    # 1. Open NPZ file as a memory map (0 RAM allocation for vectors)
    archive = np.load(path, mmap_mode='r')
    embeddings_mmap = archive["emb"]
    centers = archive["locations"]  # Coordinates are small enough to pull directly
    
    total_tiles = embeddings_mmap.shape[0]

    # 2. Pre-filter geographically to avoid reading unnecessary embeddings from disk
    in_region_mask = _points_within_region(centers, region)
    valid_indices = np.where(in_region_mask)[0]
    
    if len(valid_indices) == 0:
        print(f"No tiles at resolution '{resolution}' fall inside the given region.")
        return empty_gdf()
    
    print(f"[{resolution}] Streaming similarity search over {len(valid_indices):,} candidates inside region...")

    # 3. Prepare text query on target acceleration device
    query_vector = vision_encoder.encode_text(target)  # Expected shape: (1, D), already normalized
    if not isinstance(query_vector, torch.Tensor):
        query_vector = torch.from_numpy(query_vector)
    query_vector = query_vector.to(DEVICE).float().reshape(1, -1)

    # Containers for the rolling Top-K tracker
    running_scores = torch.tensor([], device=DEVICE, dtype=torch.float32)
    running_indices = torch.tensor([], device=DEVICE, dtype=torch.long)

    # 4. Stream dataset chunks sequentially
    for i in tqdm.tqdm(range(0, len(valid_indices), batch_size)):
        batch_indices = valid_indices[i:i + batch_size]
        
        # Pull only this chunk into RAM from the memory-map
        batch_embs_np = embeddings_mmap[batch_indices]
        
        # Ship chunk directly to GPU
        features = torch.from_numpy(batch_embs_np).to(DEVICE).float()
        features = torch.nn.functional.normalize(features, p=2, dim=-1)

        with torch.no_grad():
            # Compute cosine similarities
            scores = torch.matmul(features, query_vector.t()).squeeze(-1)
            
            # Combine current batch results with previous running top scores
            combined_scores = torch.cat([running_scores, scores])
            combined_indices = torch.cat([running_indices, torch.from_numpy(batch_indices).to(DEVICE)])
            
            # Cull combined records down to maintain only top_n
            k = min(top_n, combined_scores.shape[0]) if top_n else combined_scores.shape[0]
            top_scores, top_indices = torch.topk(combined_scores, k=k, largest=True)
            
            running_scores = top_scores
            running_indices = combined_indices[top_indices]

    # 5. Move final results back to CPU for spatial mapping
    top_scores_np = running_scores.cpu().numpy()
    top_indices_np = running_indices.cpu().numpy()
    
    matched_centers = centers[top_indices_np]
    geometries = gpd.points_from_xy(matched_centers[:, 1], matched_centers[:, 0])
    
    return from_geometries(geometries, scores=top_scores_np)