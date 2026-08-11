"""
Central configuration for the QueryEarth DLPK.

Every path here is resolved relative to this file's location (the DLPK
root) rather than hardcoded to a drive letter, so the package works
wherever ArcGIS Pro unpacks it.
"""

import glob
import os
import torch

# ------------------------------------------------------------------
# Layout
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _find_first(directory: str, patterns) -> str:
    """Return the first file matching any of `patterns` (globs) under
    `directory`, or None. Lets the DLPK pick up whatever filename the
    checkpoint/weights happen to have instead of hardcoding one."""
    if not os.path.isdir(directory):
        return None
    for pattern in patterns:
        hits = sorted(glob.glob(os.path.join(directory, pattern)))
        if hits:
            return hits[0]
    return None


# ------------------------------------------------------------------
# Demography assets
# ------------------------------------------------------------------
DEMO_PARQUET_PATH = os.path.join(EMBEDDINGS_DIR, "demography-emb.parquet")

# Trained TabularTextCLIP checkpoint. Expected in weights/, e.g.
# weights/demo_clip.pth — auto-detected so you don't have to hardcode a name.
DEMO_CLIP_CKPT_PATH = _find_first(WEIGHTS_DIR, ("*demo*.pth", "*demo*.pt", "*.pth", "*.pt"))

# Local, offline text embedder for demo-text queries. Smallest well-supported general
# text embedder available (~23M params, 384-dim, CPU-friendly, no network).
_LOCAL_EMBEDDER_DIR = os.path.join(WEIGHTS_DIR, "text-embedder")
TEXT_EMBED_MODEL = _LOCAL_EMBEDDER_DIR if os.path.isdir(_LOCAL_EMBEDDER_DIR) else "sentence-transformers/all-MiniLM-L6-v2"
TEXT_EMBED_DIM = 384  # all-MiniLM-L6-v2 output dim. Update this if you swap the embedder later.

# ------------------------------------------------------------------
# Vision assets
# ------------------------------------------------------------------
VISION_INDEX_DIRS = {
    "low": os.path.join(EMBEDDINGS_DIR, "lowres-vision"),
    "high": os.path.join(EMBEDDINGS_DIR, "highres-vision"),
}
DEFAULT_RESOLUTION = "high"

# Text encoder used to embed vision *queries* into the same space the offline
# image embeddings were computed in. This must match whatever model produced
# the embeddings baked into embeddings/{lowres,highres}-vision — unrelated to
# TurboQuant itself, which only compresses the already-computed image side.
CLIP_VISION_MODEL_NAME = "ViT-L-14"
CLIP_VISION_PRETRAINED = "laion2b_s32b_b82k"

# ------------------------------------------------------------------
# POI assets
# ------------------------------------------------------------------
POI_EMBEDDING_PATH = os.path.join(EMBEDDINGS_DIR, "poi_embeddings.parquet")
POI_PATH = os.path.join(EMBEDDINGS_DIR, "poi.parquet")

# ------------------------------------------------------------------
# Local query router (llama.cpp GGUF model, replaces the previous Ollama
# router dependency)
# ------------------------------------------------------------------
ROUTER_GGUF_PATH = os.path.join(WEIGHTS_DIR, rf"gemma-4-E4B-it-Q4_K_M.gguf")
ROUTER_N_CTX = 1500
ROUTER_N_THREADS = max(1, (os.cpu_count() or 4) - 1)

# ------------------------------------------------------------------
# Search defaults
# ------------------------------------------------------------------
DEMO_TOP_K_DEFAULT = 30
VISION_TOP_N_DEFAULT = 100
VISION_NPROBE_DEFAULT = 24  # IVF clusters probed per global (non-spatially-filtered) vision search
POI_THRESHOLD = 0.48 # Threshhold for poi search
