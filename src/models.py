"""
Model definitions and loaders.

Holds the contrastive tabular/text model (TabularTextCLIP) used for
demographic similarity search, the OpenCLIP text encoder used for visual
similarity search queries, and a small local text embedder used to embed
demo-search queries (replaces the previous Ollama dependency).
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import geopandas as gpd
import open_clip

import config


# ------------------------------------------------------------------
# Local text embedder for demographic similarity queries.
# Replaces the previous Ollama (qwen3-embedding:4b) dependency with a small,
# fully local, CPU-friendly model — no network call, minimal load latency.
# The TabularTextCLIP checkpoint's text_projector must be trained against
# whatever embedder is configured here (see config.TEXT_EMBED_DIM).
# ------------------------------------------------------------------
class LocalTextEmbedder:
    def __init__(self, model_name_or_path: str = config.TEXT_EMBED_MODEL):
        from sentence_transformers import SentenceTransformer
        print(f"Loading local text embedder ({model_name_or_path})...")
        print(str(config.DEVICE))
        self.model = SentenceTransformer(model_name_or_path, device=str(config.DEVICE))

    def encode(self, text: str) -> np.ndarray:
        return self.model.encode([text], normalize_embeddings=True)[0].astype(np.float32)


# ------------------------------------------------------------------
# Demographic contrastive model
# ------------------------------------------------------------------
class TabularTextCLIP(nn.Module):
    """Projects tabular geo-embeddings and text-sentence embeddings into a
    shared latent space so they can be compared via cosine similarity."""

    def __init__(self, geo_dim: int, text_dim: int = config.TEXT_EMBED_DIM, projection_dim: int = 128):
        super().__init__()
        self.geo_projector = nn.Sequential(
            nn.Linear(geo_dim, projection_dim * 2),
            nn.ReLU(),
            nn.Linear(projection_dim * 2, projection_dim),
        )
        self.text_projector = nn.Sequential(
            nn.Linear(text_dim, projection_dim * 2),
            nn.ReLU(),
            nn.Linear(projection_dim * 2, projection_dim),
        )
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, ae_embs: torch.Tensor, sentence_embs: torch.Tensor):
        geo_projected = self.geo_projector(ae_embs)
        zipped_latents = F.normalize(geo_projected, p=2, dim=-1)

        b, s, d = sentence_embs.shape
        flat_text = sentence_embs.view(-1, d)
        flat_text_projected = self.text_projector(flat_text)
        text_latents = F.normalize(flat_text_projected.view(b, s, -1), p=2, dim=-1)

        return zipped_latents, text_latents


def _extract_ae_embeddings(gdf: gpd.GeoDataFrame) -> np.ndarray:
    """Pull the autoencoder embedding matrix out of the demographic GeoDataFrame,
    supporting a couple of common storage conventions."""
    if "embedding" in gdf.columns:
        return np.stack(gdf["embedding"].values).astype(np.float32)

    embedding_cols = [c for c in gdf.columns if c.startswith("emb_") or c.isdigit()]
    if embedding_cols:
        return gdf[embedding_cols].values.astype(np.float32)

    raise ValueError(
        "Could not locate autoencoder embedding vectors in the demographic GeoParquet."
    )


def load_demographic_assets(
    parquet_path: str = config.DEMO_PARQUET_PATH,
    ckpt_path: str = config.DEMO_CLIP_CKPT_PATH,
):
    """Load the demographic GeoDataFrame, its embedding matrix, and the
    trained TabularTextCLIP model used to compare them against free-text
    queries."""
    if not ckpt_path:
        raise FileNotFoundError(
            "No TabularTextCLIP checkpoint (.pth/.pt) found under weights/. "
            "Place your trained demo-similarity checkpoint there."
        )

    print("Loading demographic GeoParquet...")
    gdf = gpd.read_parquet(parquet_path)

    ae_np = _extract_ae_embeddings(gdf)
    ae_embeddings = torch.from_numpy(ae_np).to(config.DEVICE)

    # Load the text embedder
    text_embedder = LocalTextEmbedder()

    # Load the text clip model
    model = TabularTextCLIP(geo_dim=ae_embeddings.shape[1], projection_dim=128)
    checkpoint = torch.load(ckpt_path, map_location=config.DEVICE)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model = model.to(config.DEVICE).eval()

    return gdf, ae_embeddings, model, text_embedder


# ------------------------------------------------------------------
# POI assets
# ------------------------------------------------------------------
def load_poi_assets(
    poi_path: str = config.POI_PATH,
    poi_embedding_path: str = config.POI_EMBEDDING_PATH,
):
    """Load the POI GeoDataFrame (amenity/name/point geometry) and the
    amenity-class embedding table (amenities + 384-dim embedding) used by
    POI similarity search. Both are fully local, plain parquet loads."""
    print("Loading POI GeoParquet...")
    poi_gdf = gpd.read_parquet(poi_path)

    print("Loading POI amenity embeddings...")
    poi_embedding_df = pd.read_parquet(poi_embedding_path)

    return poi_gdf, poi_embedding_df


# ------------------------------------------------------------------
# Vision text encoder (OpenCLIP) — loaded once, shared by every vision query.
# Encodes the *query text* into the same space the offline TurboQuant image
# embeddings live in; unrelated to TurboQuant's own (image-side) compression.
# ------------------------------------------------------------------
class VisionEncoder:
    """Thin wrapper around a globally-loaded OpenCLIP text encoder/tokenizer."""

    def __init__(
        self,
        model_name: str = config.CLIP_VISION_MODEL_NAME,
        pretrained: str = config.CLIP_VISION_PRETRAINED,
    ):
        print("Initializing OpenCLIP text encoder...")
        model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = model.to(config.DEVICE).eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        """Encode a single text query into a normalized embedding of shape (1, D)."""
        tokens = self.tokenizer([text]).to(config.DEVICE)
        embedding = self.model.encode_text(tokens)
        return F.normalize(embedding, p=2, dim=-1)
