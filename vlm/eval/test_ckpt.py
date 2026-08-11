import argparse
import base64
from io import BytesIO
import json
import logging
import sys
from typing import List, Tuple

import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import warnings
warnings.filterwarnings("ignore")

# Keep the path structure aligned with your VLM module setup
sys.path.append(rf"D:\Code\query-earth\vlm")
try:
    from model import VLM
except ImportError:
    # Fallback to local import if VLM is in current workspace
    from model import VLM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ==============================================================================
# HARDCODED QUERIES
# Add, edit, or remove your search queries in this list:
# ==============================================================================
QUERIES = [
    # --- Original Queries ---
    "baseball field",
    "solar panels on roof or field",
    "shipping containers or port Terminal",
    "swimming pool in residential area",
    "dense traffic intersection",
    "industrial warehouse complex",
    "marina with docked boats",
    "tennis courts",
    "parking lot full of cars",
    "construction site with bare earth",

    # --- Sports & Recreation ---
    "football or soccer stadium with field markings",
    "golf course fairways and sand traps",
    "athletics running track oval",
    "outdoor basketball courts",
    "skate park with concrete ramps",
    "race track or motorsport circuit",
    "amusement park with roller coasters",
    "public park with green trees and walking paths",

    # --- Urban Infrastructure & Residential ---
    "suburban neighborhood with grid pattern houses",
    "dense high-rise apartment complex",
    "cul-de-sac suburban road layout",
    "mobile home park or trailer park",
    "cemetery with grid of headstones",
    "shopping mall complex with large parking area",
    "gas station or truck stop",
    "school ground with playground and field",
    "hospital complex with helipad",
    "urban plaza or pedestrian square",

    # --- Transportation & Logistics ---
    "airport runway and tarmac with parked airplanes",
    "railroad railyard with freight trains and tracks",
    "highway cloverleaf interchange",
    "bridge crossing over river or water",
    "roundabout or traffic circle",
    "bus depot or truck fleet terminal",
    "toll plaza on highway",
    "pier or jetty extending into ocean",
    "canal or artificial waterway with locks",
    "dry dock for ship maintenance",

    # --- Industrial & Energy Infrastructure ---
    "oil or gas storage tanks",
    "wind turbines in field or offshore",
    "coal power plant or cooling towers",
    "water treatment plant with circular settling tanks",
    "electrical substation with transformers and pylons",
    "mining pit or open pit quarry",
    "oil refinery with complex pipelines",
    "solar farm with rows of PV panels",
    "recycling yard or scrap yard with metal heaps",
    "lumber yard with stacked timber",

    # --- Agriculture & Land Use ---
    "center pivot circular irrigation fields",
    "terraced farming fields on hillside",
    "greenhouses or polytunnels in agricultural area",
    "orchard with neat rows of fruit trees",
    "vineyard with parallel rows of vines",
    "crop fields with rectangular boundaries",
    "hay bales in harvested field",
    "dairy farm or livestock feedlot",
    "fish farm or aquaculture ponds",
    "paddy fields flooded with water",

    # --- Natural & Physical Geography ---
    "meandering river through valley",
    "sand dunes in desert",
    "volcanic crater or caldera",
    "glacier or ice sheet",
    "coastal beach with ocean waves breaking",
    "coral reef under clear shallow ocean water",
    "forest canopy with dense tree cover",
    "mangrove swamp or coastal wetland",
    "mountain peak with snow cover",
    "river delta flowing into sea",
    "cliff face along ocean coastline",
    "lake or reservoir with dam structure",

    # --- Environmental & Land Disturbances ---
    "wildfire burn scar or scorched forest",
    "deforestation or clear-cut timber forest",
    "flooded agricultural fields or urban area",
    "landslide or mudslide scar",
    "coastal erosion or shrinking beach",
    "drying lake bed with salt flats",
    "smokestacks emitting plume or smoke",

    # --- Specialized & Rare Features ---
    "helipad on building rooftop",
    "archaeological ruins or ancient structures",
    "military base with barracks and airfield",
    "drive-in theater screen and parking",
    "solar evaporative salt pans with bright colors",
    "log boom or timber floating on river",
    "fountain or water display in park",
    "footbridge over highway",
    "sewage lagoon or oxidation pond",
    "radio or satellite communication dish antenna array",

    # --- Pattern & Texture Specifics ---
    "neat geometric grid of urban streets",
    "curved winding mountain road",
    "bright red roofed buildings",
    "bright blue roofs or tarps",
    "dense cluster of small boats",
    "checkerboard pattern of agricultural fields",
    "shadows cast by tall skyscrapers",
    "muddy brown water meeting clear blue ocean water",
    "zigzag mountain road switchbacks",
    "patches of bare soil mixed with forest"
]



class ParquetImageDataset(Dataset):
    """Reads raw image bytes directly from a Pandas/PyArrow DataFrame."""

    def __init__(self, df: pd.DataFrame, image_col: str):
        self.df = df
        self.image_col = image_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        raw_bytes = self.df.iloc[idx][self.image_col]
        
        # Handle dict wrapping if parquet uses PyArrow's image logical type
        if isinstance(raw_bytes, dict) and "bytes" in raw_bytes:
            raw_bytes = raw_bytes["bytes"]
            
        img = Image.open(BytesIO(raw_bytes)).convert("RGB")
        return img, idx


class ParquetCollate:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        images, indices = zip(*batch)
        pixel_values = self.processor(images=list(images), return_tensors="pt")[
            "pixel_values"
        ]
        return pixel_values, torch.tensor(indices, dtype=torch.long)


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    train_args = ckpt.get("args", {})

    model, processor, tokenizer = VLM.build_with_processor(
        text_model_name=train_args.get("text_model_name", "prajjwal1/bert-tiny"),
        vision_model_name=train_args.get("vision_model_name", "facebook/dino-vits16"),
        proj_dim=train_args.get("proj_dim", 256),
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device)
    model.eval()
    logger.info("Loaded checkpoint %s", checkpoint_path)
    return model, processor, tokenizer


@torch.no_grad()
def compute_image_embeddings(
    model: VLM,
    processor,
    df: pd.DataFrame,
    image_col: str,
    device: torch.device,
    batch_size: int = 128,
    num_workers: int = 4,
) -> torch.Tensor:
    dataset = ParquetImageDataset(df, image_col)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=ParquetCollate(processor),
        pin_memory=True,
    )

    all_embeds = []
    logger.info("Extracting embeddings for %d images...", len(dataset))
    
    for pixel_values, _ in loader:
        pixel_values = pixel_values.to(device, non_blocking=True)
        image_embeds = model.encode_image(pixel_values, normalize=True)
        all_embeds.append(image_embeds.cpu())

    return torch.cat(all_embeds, dim=0)  # (N, D)


@torch.no_grad()
def compute_text_embeddings(
    model: VLM,
    tokenizer,
    queries: List[str],
    device: torch.device,
) -> torch.Tensor:
    tok = tokenizer(
        queries, padding=True, truncation=True, max_length=77, return_tensors="pt"
    ).to(device)
    
    text_embeds = model.encode_text(
        tok["input_ids"], tok["attention_mask"], normalize=True
    )
    return text_embeds.cpu()  # (Num_queries, D)


def bytes_to_b64_url(raw_bytes: bytes) -> str:
    """Converts image raw bytes into a browser-renderable base64 string."""
    if isinstance(raw_bytes, dict) and "bytes" in raw_bytes:
        raw_bytes = raw_bytes["bytes"]
    b64_str = base64.b64encode(raw_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"


def generate_html_report(
    queries: List[str],
    top_results: List[List[Tuple[float, bytes, int]]],
    output_path: str,
    top_k: int,
):
    """Generates a styled, standalone HTML report displaying query results."""
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VLM Similarity Search Results</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #0f172a;
            color: #f8fafc;
            margin: 0;
            padding: 2rem;
        }}
        h1 {{
            font-size: 1.875rem;
            margin-bottom: 0.5rem;
            border-bottom: 2px solid #334155;
            padding-bottom: 0.75rem;
        }}
        .subtitle {{
            color: #94a3b8;
            margin-bottom: 2rem;
        }}
        .query-section {{
            background-color: #1e293b;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        }}
        .query-title {{
            font-size: 1.25rem;
            font-weight: 600;
            color: #38bdf8;
            margin-top: 0;
            margin-bottom: 1rem;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 1rem;
        }}
        .card {{
            background-color: #0f172a;
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid #334155;
            display: flex;
            flex-direction: column;
        }}
        .card img {{
            width: 100%;
            height: 180px;
            object-fit: cover;
            background-color: #020617;
        }}
        .card-meta {{
            padding: 0.75rem;
            font-size: 0.875rem;
            display: flex;
            justify-content: space-between;
            color: #cbd5e1;
        }}
        .score {{
            font-weight: bold;
            color: #4ade80;
        }}
        .idx {{
            color: #64748b;
        }}
    </style>
</head>
<body>
    <h1>VLM Visual Search Results</h1>
    <div class="subtitle">Showing Top-{top_k} similarity matches per query</div>
"""

    for query, results in zip(queries, top_results):
        html_content += f"""
    <div class="query-section">
        <div class="query-title">🔍 "{query}"</div>
        <div class="grid">"""
        
        for rank, (score, img_bytes, idx) in enumerate(results, 1):
            b64_url = bytes_to_b64_url(img_bytes)
            html_content += f"""
            <div class="card">
                <img src="{b64_url}" alt="Match #{rank}" loading="lazy" />
                <div class="card-meta">
                    <span class="idx">#{rank} (Row {idx})</span>
                    <span class="score">{score:.4f}</span>
                </div>
            </div>"""
            
        html_content += """
        </div>
    </div>"""

    html_content += """
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    logger.info("Saved search report HTML to %s", output_path)


def parse_args():
    p = argparse.ArgumentParser(description="Run vector search over Parquet images & build HTML gallery.")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to VLM checkpoint .pt file")
    p.add_argument("--parquet_path", type=str, required=True, help="Path to input Parquet file")
    p.add_argument("--image_col", type=str, default="image", help="Column name containing binary image bytes")
    p.add_argument("--top_k", type=int, default=5, help="Number of top matches to retrieve per query")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--output_html", type=str, default="search_results.html")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # 1. Load Model & Data
    model, processor, tokenizer = load_model_from_checkpoint(args.checkpoint, device)
    
    logger.info("Loading parquet file from %s...", args.parquet_path)
    df = pd.read_parquet(args.parquet_path)
    logger.info("Loaded %d rows from parquet.", len(df))

    print("\n" + "="*50)
    print("           PARQUET DATAFRAME DIAGNOSTICS          ")
    print("="*50)
    print(f"Total rows: {len(df)}")
    print(f"Columns present: {list(df.columns)}")
    print("\nDataFrame Preview (Head):")
    print(df.head(3))
    print("="*50 + "\n")

    # 2. Extract Embeddings
    img_embeds = compute_image_embeddings(
        model, processor, df, args.image_col, device, args.batch_size, args.num_workers
    )  # (N, D)
    
    logger.info("Running search for %d hardcoded queries...", len(QUERIES))
    text_embeds = compute_text_embeddings(
        model, tokenizer, QUERIES, device
    )  # (Q, D)

    # 3. Compute Similarity Matrix (Q x N)
    logger.info("Computing cosine similarities...")
    sim_matrix = text_embeds @ img_embeds.T  # (Q, N)

    # 4. Fetch Top K Results
    top_k = min(args.top_k, len(df))
    top_scores, top_indices = sim_matrix.topk(top_k, dim=-1)

    top_results_per_query = []
    for q_idx in range(len(QUERIES)):
        q_results = []
        for k in range(top_k):
            score = top_scores[q_idx, k].item()
            row_idx = top_indices[q_idx, k].item()
            img_bytes = df.iloc[row_idx][args.image_col]
            q_results.append((score, img_bytes, row_idx))
        top_results_per_query.append(q_results)

    # 5. Generate Output HTML Report
    generate_html_report(QUERIES, top_results_per_query, args.output_html, top_k)


if __name__ == "__main__":
    main()