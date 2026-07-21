"""
Central configuration for the QueryEarth pipeline.

All filesystem paths, model checkpoints, and constants live here so the
rest of the codebase never hardcodes a path inline.
"""

import os
import torch

# ------------------------------------------------------------------
# Device
# ------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------
# Data paths
# ------------------------------------------------------------------
DEMO_PARQUET_PATH = (
    rf"E:\Data\query-earth\embeddings_california\zip-demo-embs.parquet"
)
CLIP_CKPT_PATH = rf"E:\Results\mmdfm\weights\zip_nofusion_clip2.pth"

# Folder containing the three resolution variants of the vision embeddings.
# Expected files inside this folder:
#   embeddings_full.npz   -> ~2km tiles   (low resolution / wide context)
#   embeddings_tile4.npz  -> ~500m tiles  (medium resolution)
#   embeddings_tile100.npz-> ~200m tiles  (high resolution / fine detail)
VISION_EMBEDDINGS_DIR = rf"E:\Data\query-earth\embeddings_california"

OUTPUT_DIR = rf"E:\Results\query-earth"    
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Resolution mapping
# ------------------------------------------------------------------
# Maps the LLM-facing resolution vocabulary ("low"/"medium"/"high") to the
# actual NPZ file on disk. Keeping this as a single dict makes it trivial
# to add a fourth resolution tier later without touching pipeline logic.
RESOLUTION_TO_FILE = {
    "low": "hex7-skyclip-low.npz",      # 2km   x 2km  tiles
    # "medium": "embeddings_tile4.npz",   # 500m  x 500m tiles
    "high": "hex7-skyclip-high.npz",  # 200m  x 200m tiles
}
DEFAULT_RESOLUTION = "high"

# ------------------------------------------------------------------
# Model identifiers
# ------------------------------------------------------------------
CLIP_VISION_MODEL_NAME = "ViT-L-14"
CLIP_VISION_PRETRAINED = "laion2b_s32b_b82k"

OLLAMA_EMBED_MODEL = "qwen3-embedding:4b"
OLLAMA_ROUTER_MODEL = "ministral-3:3b"

# ------------------------------------------------------------------
# Search defaults
# ------------------------------------------------------------------
DEMO_TOP_K_DEFAULT = 30
VISION_TOP_N_DEFAULT = 100
