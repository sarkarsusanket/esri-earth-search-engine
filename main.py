import os
import json
import tqdm
import torch
import numpy as np
import pandas as pd
import geopandas as gpd
import torch.nn as nn
import torch.nn.functional as F
from shapely.geometry import Point, Polygon, MultiPolygon
from geopy.geocoders import Nominatim
import ollama
import open_clip

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEMO_PARQUET_PATH = "/data/susanket/queryearth/california-demo-emb.parquet"
VISUAL_NPZ_PATH = "/data/susanket/california_embeddings_ckpt.npz" # Update path if needed
CLIP_CKPT_PATH = "/data/susanket/mmdfm/embeddings/zip_nofusion_clip.pth"

print("Initializing OpenCLIP model globally...")
# Load the model once at script startup, not inside the loop
CLIP_MODEL, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k')
CLIP_MODEL = CLIP_MODEL.to(DEVICE).eval()
CLIP_TOKENIZER = open_clip.get_tokenizer('ViT-L-14')

OUTPUT_DIR = "/home/susanket/query-earth/output_shapes"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. MODEL DEFINITIONS & LOADERS
# ==========================================
class TabularTextCLIP(nn.Module):
    def __init__(self, geo_dim=352, text_dim=2560, projection_dim=128):
        super().__init__()
        self.geo_projector = nn.Sequential(
            nn.Linear(geo_dim, projection_dim * 2),
            nn.ReLU(),
            nn.Linear(projection_dim * 2, projection_dim)
        )
        self.text_projector = nn.Sequential(
            nn.Linear(text_dim, projection_dim * 2),
            nn.ReLU(),
            nn.Linear(projection_dim * 2, projection_dim)
        )
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, ae_embs, sentence_embs):
        geo_projected = self.geo_projector(ae_embs)
        zipped_latents = F.normalize(geo_projected, p=2, dim=-1)
        b, s, d = sentence_embs.shape
        flat_text = sentence_embs.view(-1, d)
        flat_text_projected = self.text_projector(flat_text)
        text_latents = F.normalize(flat_text_projected.view(b, s, -1), p=2, dim=-1)
        return zipped_latents, text_latents

def load_demographic_assets():
    """Loads the geoparquet containing geometry/features and initialises the CLIP model."""
    print("Loading Demographic GeoParquet...")
    gdf = gpd.read_parquet(DEMO_PARQUET_PATH)
    
    # Extract the autoencoder embeddings from the dataframe
    # Assumes a column named 'ae_embedding' exists as arrays/lists
    if 'embedding' in gdf.columns:
        ae_np = np.stack(gdf['embedding'].values).astype(np.float32)
    else:
        # Fallback dummy if embeddings are stored as individual columns or structured differently
        embedding_cols = [c for c in gdf.columns if c.startswith('emb_') or c.isdigit()]
        if embedding_cols:
            ae_np = gdf[embedding_cols].values.astype(np.float32)
        else:
            raise ValueError("Could not automatically locate autoencoder embedding vectors in Geoparquet.")

    ae_embeddings = torch.from_numpy(ae_np).to(DEVICE)
    
    # Initialize and load model weights
    clip_model = TabularTextCLIP(geo_dim=ae_embeddings.shape[1], projection_dim=128)
    checkpoint = torch.load(CLIP_CKPT_PATH, map_location=DEVICE)
    
    # Adjust state dict if saved with a mismatching wrapper
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    clip_model.load_state_dict(state_dict)
    clip_model = clip_model.to(DEVICE).eval()
    
    return gdf, ae_embeddings, clip_model

# ==========================================
# 3. EMBEDDINGS & LLM PARSING PIPELINE
# ==========================================
def get_ollama_embeddings(text, model="qwen3-embedding:4b"):
    response = ollama.embed(model=model, input=text)
    embeddings = response.embeddings
    return np.array(embeddings).squeeze(0)

def parse_query_with_llm(user_query):
    """
    Parses an arbitrary geographic query into structured pipeline steps using Ministral.
    """
    prompt = f"""
    You are an advanced geospatial query parsing routing assistant. Your job is to extract specific intents from a user's natural language request.
    Break down the following user query into three distinct values:
    1. "geocoding": The physical location name or neighborhood to anchor the search (or null if none specified).
    2. "demography": Text describing socio-economic, population, or demographic filters (or null if none specified).
    3. "visual_query": The explicit visual object or land-use type to search for via computer vision features (or null if none specified).

    Return ONLY a valid JSON object with these three keys. Do not include markdown code blocks, explanations, or trailing text.

    Example 1: "Find all the parking lots in the poorer neighbourhoods of downtown la"
    Output: {{"geocoding": "downtown la", "demography": "poorer neighbourhoods", "visual_query": "parking lots"}}

    Example 2: "Where are industrial warehouses located in industrial sectors near San Diego?"
    Output: {{"geocoding": "San Diego", "demography": "industrial sectors", "visual_query": "industrial warehouses"}}

    Example 3: "Show me swimming pools in rich neighborhoods"
    Output: {{"geocoding": null, "demography": "rich neighborhoods", "visual_query": "swimming pools"}}

    User Query: "{user_query}"
    Output:
    """
    
    response = ollama.chat(
        model='ministral-3:3b',
        messages=[{'role': 'user', 'content': prompt}],
        options={'temperature': 0.0} # Low temperature for reliable JSON parsing
    )
    
    try:
        cleaned_content = response.message.content.strip().replace("```json", "").replace("```", "")
        parsed_task = json.loads(cleaned_content)
        return parsed_task
    except Exception as e:
        print(f"Failed to parse LLM response cleanly. Error: {e}. Raw content: {response.message.content}")
        # Manual fallback parser logic
        return {"geocoding": None, "demography": None, "visual_query": None}

# ==========================================
# 4. EXECUTION ENGINES
# ==========================================
def execute_geocoding(location_str):
    if not location_str:
        return None

    try:
        geolocator = Nominatim(user_agent="geo_orchestrator_agent")
        location = geolocator.geocode(location_str, timeout=10)



        if location and "boundingbox" in location.raw:
            bbox = location.raw["boundingbox"]

            lat_min, lat_max, lon_min, lon_max = map(float, bbox)

            return Polygon([
                (lon_min, lat_min),
                (lon_max, lat_min),
                (lon_max, lat_max),
                (lon_min, lat_max),
            ])

        return None

    except Exception as e:
        print(f"Geocoding failed: {e}")
        return None

def execute_demographic_search(query_str, gdf, ae_embeddings, clip_model, top_k=50):
    """Executes contrastive alignment search against the demographic embeddings."""
    if not query_str:
        return gdf.copy() # Return entire study area if no demographic restriction is asked
        
    raw_text_emb = get_ollama_embeddings(query_str)
    text_tensor = torch.from_numpy(np.array([raw_text_emb], dtype=np.float32)).to(DEVICE)
    
    with torch.no_grad():
        normalized_tabular_latents = F.normalize(clip_model.geo_projector(ae_embeddings), p=2, dim=-1).float()
        normalized_text_latent = F.normalize(clip_model.text_projector(text_tensor), p=2, dim=-1).float()
        
        scores = torch.matmul(normalized_tabular_latents, normalized_text_latent.t()).squeeze(-1)
        scores_np = scores.cpu().numpy()
        
        s_min, s_max = scores_np.min(), scores_np.max()
        if s_max - s_min > 0:
            normalized_scores = (scores_np - s_min) / (s_max - s_min)
        else:
            normalized_scores = scores_np * 0 + 1.0

    top_indices = np.argsort(normalized_scores)[::-1][:top_k]
    matched_gdf = gdf.iloc[top_indices].copy()
    matched_gdf['demo_score'] = normalized_scores[top_indices]
    return matched_gdf

# ==========================================
# 2. ACCELERATED SEARCH FUNCTION
# ==========================================
def execute_visual_search(visual_query, bounding_geometry, npz_path, top_n=100):
    """
    Performs lightning-fast GPU-accelerated similarity search on imagery embeddings.
    """
    if not os.path.exists(npz_path):
        print(f"Visual asset matrix not found at {npz_path}. Skipping.")
        return gpd.GeoDataFrame(columns=['geometry', 'score'], crs="EPSG:4326")

    # 1. Load NPZ features (keep on CPU initially)
    archive = np.load(npz_path)
    img_embs = archive['embeddings']  # Matrix shape: (M, text_dim)
    centers = archive['centers']      # Array shape: (M, 2)
    
    # 2. Build Vectorized GeoDataFrame (Fixed lon/lat order mapping)
    df_pts = pd.DataFrame({'npz_idx': np.arange(len(centers))})
    geometry = gpd.points_from_xy(centers[:, 1], centers[:, 0])  # Ensure lon is first coordinate
    pts_gdf = gpd.GeoDataFrame(df_pts, geometry=geometry, crs="EPSG:4326")
    
    # 3. Handle Spatial Boundaries
    roi_gdf = gpd.GeoDataFrame(geometry=[bounding_geometry], crs="EPSG:4326")
    spatial_mask = pts_gdf.within(roi_gdf.geometry.iloc[0])
    filtered_pts_gdf = pts_gdf[spatial_mask].copy()
    
    if filtered_pts_gdf.empty:
        print("⚠️ Zero visual targets found inside the spatial boundary.")
        return gpd.GeoDataFrame(columns=['geometry', 'score'], crs="EPSG:4326")
        
    # Gather spatial subsets
    spatial_indices = filtered_pts_gdf['npz_idx'].values
    
    # Slice your massive numpy matrix down to just the active candidate rows
    filtered_embs_np = img_embs[spatial_indices]
    print(f"Candidates inside ROI: {len(filtered_embs_np):,}")
    
    # 4. Generate Text Embedding using Global Model Instance
    with torch.no_grad():
        text_tokens = CLIP_TOKENIZER([f"{visual_query}"]).to(DEVICE)
        query_tensor = CLIP_MODEL.encode_text(text_tokens)  # Shape: (1, D)
        # Normalize text query vector
        query_tensor = torch.nn.functional.normalize(query_tensor, p=2, dim=-1)

    # 5. Ultra-Fast PyTorch CUDA Matrix Multiplication
    # Transfer ONLY the filtered rows to GPU, avoiding VRAM bloating
    features_tensor = torch.from_numpy(filtered_embs_np).to(DEVICE).float()
    features_tensor = torch.nn.functional.normalize(features_tensor, p=2, dim=-1)  # Shape: (N, D)
    
    with torch.no_grad():
        # Cosine similarity matrix multiplication via cuBLAS (Blazing fast!)
        scores_tensor = torch.matmul(features_tensor, query_tensor.t()).squeeze(-1)  # Shape: (N,)
        
        # Use torch.topk to sort and extract top entries on the GPU directly
        actual_top_n = min(top_n, len(scores_tensor))
        top_scores, top_local_indices = torch.topk(scores_tensor, k=actual_top_n, largest=True)
        
        # Pull back ONLY the tiny top_n slice to CPU
        top_scores_np = top_scores.cpu().numpy()
        top_local_indices_np = top_local_indices.cpu().numpy()

    # 6. Map results back to Geometries
    # Extract matching rows from our filtered data slice
    matched_rows = filtered_pts_gdf.iloc[top_local_indices_np].copy()
    matched_rows['score'] = top_scores_np
    
    # Clean up structure for output
    output_gdf = matched_rows[['geometry', 'score']].reset_index(drop=True)
    return output_gdf
# ==========================================
# 5. GENERALIZED END-TO-END ORCHESTRATOR
# ==========================================
def run_pipeline(user_query, demo_gdf, ae_embeddings, clip_model):
    print(f"\n⚡ Initiating Execution Pipeline For Query: '{user_query}'")
    
    # Step 1: Parse string queries using LLM Router
    tasks = parse_query_with_llm(user_query)
    print(f"Parsed Orchestration Routing: {tasks}")
    
    # Step 2: Extract Bounding Geometry via Geocoding
    geocode_boundary = None
    if tasks.get("geocoding"):
        print(f"-> Geocoding Anchor Space: {tasks['geocoding']}")
        geocode_boundary = execute_geocoding(tasks["geocoding"])
        
    # Step 3: Extract & Filter Demographic Profile Boundaries
    print(f"-> Target Socio-Demographic Filter: {tasks.get('demography')}")

    matched_demo_gdf = execute_demographic_search(
        query_str=tasks.get("demography"),
        gdf=demo_gdf,
        ae_embeddings=ae_embeddings,
        clip_model=clip_model,
        top_k=30
    )
    
    # Intersect geocoding boundaries with demographics to refine search space
    if geocode_boundary is not None:
        # Filter demographics polygons strictly interacting with our geocoded bounding box
        spatial_mask = matched_demo_gdf.geometry.intersects(geocode_boundary)
        refined_search_gdf = matched_demo_gdf[spatial_mask]
        
        # Fallback if demographic criteria find nothing within the geocoded box
        if refined_search_gdf.empty:
            print("Demographic features did not overlap geocoded envelope. Falling back to explicit geocode boundary.")
            roi_geometry = geocode_boundary
        else:
            roi_geometry = refined_search_gdf.unary_union
    else:
        # Fallback if no specific city/neighborhood was geocoded
        roi_geometry = matched_demo_gdf.unary_union

    # Step 4: Perform Object-Level Visual Search inside refined boundaries
    if tasks.get("visual_query") and roi_geometry is not None:
        print(f"-> Executing Vision-Based Similarity Search for: '{tasks['visual_query']}'")
        output_gdf = execute_visual_search(
            visual_query=tasks["visual_query"],
            bounding_geometry=roi_geometry,
            npz_path=VISUAL_NPZ_PATH,
            top_n=100
        )
    else:
        # Fallback if query lacks a visual component: export the demographic boundaries directly
        print("No explicit visual objects found in query. Defaulting output layers to demographic boundaries.")
        output_gdf = matched_demo_gdf.copy()

    # Step 5: Export results cleanly to a shapefile
    if not output_gdf.empty:
        safe_name = "".join([c if c.isalnum() else "_" for c in user_query]).strip("_")[:50]
        output_path = os.path.join(OUTPUT_DIR, f"{safe_name}.geojson")
        
        # Clean attribute data types for driver compatibility
        for col in output_gdf.columns:
            if output_gdf[col].dtype == 'object' and col != 'geometry':
                output_gdf[col] = output_gdf[col].astype(str)
                
        output_gdf.to_file(output_path)
        print(f" Saved output results to: {output_path}")
    else:
        print("⚠️ No features passed the spatial or demographic criteria. Shapefile generation skipped.")

# ==========================================
# 6. APPLICATION ENTRYPOINT
# ==========================================
if __name__ == "__main__":
    # Bootstrapping demographic indexes
    demo_gdf, ae_embeddings, clip_model = load_demographic_assets()
    
    # Complex test query
    sample_query = "Farmlands in regions with high poverty rates"
    run_pipeline(sample_query, demo_gdf, ae_embeddings, clip_model)