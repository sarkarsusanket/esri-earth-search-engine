"""
Model definitions and loaders.

Holds the contrastive tabular/text model (TabularTextCLIP) used for
demographic similarity search, plus the global OpenCLIP vision-language
model used for visual similarity search.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import geopandas as gpd
import open_clip
import ollama

import config


# ------------------------------------------------------------------
# Demographic contrastive model
# ------------------------------------------------------------------
class TabularTextCLIP(nn.Module):
    """Projects tabular geo-embeddings and text-sentence embeddings into a
    shared latent space so they can be compared via cosine similarity."""

    def __init__(self, geo_dim: int, text_dim: int = 2560, projection_dim: int = 128):
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
    parquet_path: str = config.DEMO_PARQUET_PATH, ckpt_path: str = config.CLIP_CKPT_PATH
):
    """Load the demographic GeoDataFrame, its embedding matrix, and the
    trained TabularTextCLIP model used to compare them against free-text
    queries."""
    print("Loading demographic GeoParquet...")
    gdf = gpd.read_parquet(parquet_path)

    ae_np = _extract_ae_embeddings(gdf)
    ae_embeddings = torch.from_numpy(ae_np).to(config.DEVICE)

    model = TabularTextCLIP(geo_dim=ae_embeddings.shape[1], projection_dim=128)
    checkpoint = torch.load(ckpt_path, map_location=config.DEVICE)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model = model.to(config.DEVICE).eval()

    return gdf, ae_embeddings, model


# ------------------------------------------------------------------
# Vision model (OpenCLIP) — loaded once, shared by every vision query
# ------------------------------------------------------------------
class VisionEncoder:
    """Thin wrapper around a globally-loaded OpenCLIP model/tokenizer."""

    def __init__(
        self,
        model_name: str = config.CLIP_VISION_MODEL_NAME,
        pretrained: str = config.CLIP_VISION_PRETRAINED,
    ):
        print("Initializing OpenCLIP model...")
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


# ------------------------------------------------------------------
# Text embeddings for demographic similarity (via Ollama)
# ------------------------------------------------------------------
def get_text_embedding(text: str, model: str = config.OLLAMA_EMBED_MODEL) -> np.ndarray:
    response = ollama.embed(model=model, input=text)
    return np.array(response.embeddings).squeeze(0)
