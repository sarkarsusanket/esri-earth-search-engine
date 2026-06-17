import os
from io import BytesIO
import queue
from concurrent.futures import ThreadPoolExecutor
import geopandas as gpd
import numpy as np
from PIL import Image
import requests
from shapely.geometry import Point
import torch
import open_clip
import tqdm


# --- CONFIGURATION ---
GRID_SHAPEFILE = rf"/data/susanket/queryearth/ca_grid/california_grid.shp"
OUTPUT_DIR = rf"/data/susanket/queryearth/embedding_results3"
CHECKPOINT_INTERVAL = 100  # Save data to disk every N files
SERVICE_URL = "https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/export"
EMBEDDING_DIM = 768       # ViT-L-14 output dimension

# Thread Optimization
MAX_WORKERS = 80  # Number of parallel CPU threads downloading images simultaneously

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

FILE_PATHS = {
    "full": {
        "embs": os.path.join(OUTPUT_DIR, "embeddings_full_embs.dat"),
        "coords": os.path.join(OUTPUT_DIR, "embeddings_full_coords.dat")
    },
    "tile4": {
        "embs": os.path.join(OUTPUT_DIR, "embeddings_tile4_embs.dat"),
        "coords": os.path.join(OUTPUT_DIR, "embeddings_tile4_coords.dat")
    },
    "tile100": {
        "embs": os.path.join(OUTPUT_DIR, "embeddings_tile100_embs.dat"),
        "coords": os.path.join(OUTPUT_DIR, "embeddings_tile100_coords.dat")
    }
}

# --- INITIALIZE OPEN_CLIP ---
print(f"Loading CLIP model on {device}...")
model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k')
model = model.to(device).eval()

# --- HELPER FUNCTIONS ---

def get_arcgis_image(lat, lon, radius_meters=1000):
    """Fetches an image centered at lat/lon with a given radius."""
    buffered_geom = Point(lon, lat).buffer(radius_meters / 111 / 1000)
    bbox_series = gpd.GeoSeries([buffered_geom], crs="EPSG:4326").to_crs("EPSG:3857")
    xmin_3857, ymin_3857, xmax_3857, ymax_3857 = bbox_series.bounds.iloc[0]

    width, height = 1000, 1000

    export_params = {
        "bbox": f"{xmin_3857},{ymin_3857},{xmax_3857},{ymax_3857}",
        "bboxSR": "3857",
        "size": f"{width},{height}",
        "imageSR": "3857",
        "format": "png",
        "transparent": "true",
        "f": "image",
    }

    try:
        response = requests.get(SERVICE_URL, params=export_params, timeout=30)
        if response.status_code == 200:
            if b"error" in response.content or b"Exception" in response.content:
                return None
            return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception:
        pass
    return None

def download_worker(idx, lat, lon, result_queue):
    """Worker task executed concurrently by the thread pool."""
    img = get_arcgis_image(lat, lon, radius_meters=1000)
    # print(img)
    result_queue.put((idx, lat, lon, img))

def tile_image(image, grid_size):
    """Divides an image into grid_size x grid_size equal tiles."""
    img_w, img_h = image.size
    tile_w, tile_h = img_w // grid_size, img_h // grid_size
    tiles = []
    for i in range(grid_size):
        for j in range(grid_size):
            box = (j * tile_w, i * tile_h, (j + 1) * tile_w, (i + 1) * tile_h)
            tiles.append(image.crop(box))
    return tiles

def extract_embeddings(pil_images, model, preprocess, batch_size=32):
    """Batches PIL Images and extracts normalized CLIP embeddings."""
    if not pil_images:
        return np.array([])
    embeddings = []
    for i in range(0, len(pil_images), batch_size):
        batch_imgs = pil_images[i:i+batch_size]
        tensor_imgs = torch.stack([preprocess(img) for img in batch_imgs]).to(device)
        with torch.no_grad(), torch.amp.autocast('cuda'):
            feat = model.encode_image(tensor_imgs)
            feat /= feat.norm(dim=-1, keepdim=True)
            embeddings.append(feat.cpu().numpy())
    return np.vstack(embeddings)

# --- MAIN RUNNER ---

def main():
    print("Reading grid shapefile...")
    grid_gdf = gpd.read_file(GRID_SHAPEFILE)
    if grid_gdf.crs != "EPSG:4326":
        grid_gdf = grid_gdf.to_crs("EPSG:4326")
    
    centroids = grid_gdf.geometry.centroid
    total_records = len(grid_gdf)
    print(f"Total grid cells to process: {total_records}")

    state_file = os.path.join(OUTPUT_DIR, "processed_count.txt")
    start_idx = 0
    if os.path.exists(state_file):
        with open(state_file, "r") as f:
            start_idx = int(f.read().strip())
        print(f"Resuming pipeline starting from index: {start_idx}")

    shapes = {
        "full": {"embs": (total_records, EMBEDDING_DIM), "coords": (total_records, 2)},
        "tile4": {"embs": (total_records * 4, EMBEDDING_DIM), "coords": (total_records * 4, 2)},
        "tile100": {"embs": (total_records * 100, EMBEDDING_DIM), "coords": (total_records * 100, 2)}
    }

    mmaps = {}
    for key in ["full", "tile4", "tile100"]:
        mmaps[key] = {
            "embs": np.memmap(FILE_PATHS[key]["embs"], dtype="float32", mode="w+" if start_idx == 0 else "r+", shape=shapes[key]["embs"]),
            "coords": np.memmap(FILE_PATHS[key]["coords"], dtype="float32", mode="w+" if start_idx == 0 else "r+", shape=shapes[key]["coords"])
        }

    cache = {
        "full": {"embs": [], "coords": []},
        "tile4": {"embs": [], "coords": []},
        "tile100": {"embs": [], "coords": []}
    }
    
    pending_count = 0
    result_queue = queue.Queue(maxsize=MAX_WORKERS * 2)

    # 1. Start ThreadPoolExecutor for background downloads
    print(f"Spawning concurrent download pool using {MAX_WORKERS} workers...")
    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    
    # Pre-submit initial tasks to populate the pipeline queue
    current_submit_idx = start_idx
    def refill_queue():
        nonlocal current_submit_idx
        while current_submit_idx < total_records and not result_queue.full():
            p = centroids.iloc[current_submit_idx]
            executor.submit(download_worker, current_submit_idx, p.y, p.x, result_queue)
            current_submit_idx += 1

    refill_queue()

    # 2. Main Consumer Loop (Runs GPU inference synchronously)
    for expected_idx in tqdm.tqdm(range(start_idx, total_records)):
        # Refill download queue at every step iteration
        refill_queue()
        
        # Pull downloaded images from the queue
        idx, lat, lon, img = result_queue.get()
        
        # Handle index mismatches if threads finish slightly out of order
        # (Though memmap handles indexing via explicit array assignments, we track it linearly)
        if img is None:
            cache["full"]["embs"].append(np.zeros(EMBEDDING_DIM))
            cache["full"]["coords"].append([lat, lon])
            cache["tile4"]["embs"].extend([np.zeros(EMBEDDING_DIM)] * 4)
            cache["tile4"]["coords"].extend([[lat, lon]] * 4)
            cache["tile100"]["embs"].extend([np.zeros(EMBEDDING_DIM)] * 100)
            cache["tile100"]["coords"].extend([[lat, lon]] * 100)
        else:
            # Resolution 1: Full View
            emb_full = extract_embeddings([img], model, preprocess)
            cache["full"]["embs"].append(emb_full[0])
            cache["full"]["coords"].append([lat, lon])

            # Resolution 2: 4 Tiles
            tiles_4 = tile_image(img, grid_size=2)
            embs_4 = extract_embeddings(tiles_4, model, preprocess)
            cache["tile4"]["embs"].extend(embs_4)
            cache["tile4"]["coords"].extend([[lat, lon]] * 4)

            # Resolution 3: 100 Tiles
            tiles_100 = tile_image(img, grid_size=10)
            embs_100 = extract_embeddings(tiles_100, model, preprocess)
            cache["tile100"]["embs"].extend(embs_100)
            cache["tile100"]["coords"].extend([[lat, lon]] * 100)

        pending_count += 1

        # Checkpoint: Flush cache to disk allocation files
        if pending_count >= CHECKPOINT_INTERVAL or expected_idx == total_records - 1:
            chunk_start_idx = expected_idx - pending_count + 1
            
            mmaps["full"]["embs"][chunk_start_idx:expected_idx+1] = cache["full"]["embs"]
            mmaps["full"]["coords"][chunk_start_idx:expected_idx+1] = cache["full"]["coords"]

            mmaps["tile4"]["embs"][chunk_start_idx*4:(expected_idx+1)*4] = cache["tile4"]["embs"]
            mmaps["tile4"]["coords"][chunk_start_idx*4:(expected_idx+1)*4] = cache["tile4"]["coords"]

            mmaps["tile100"]["embs"][chunk_start_idx*100:(expected_idx+1)*100] = cache["tile100"]["embs"]
            mmaps["tile100"]["coords"][chunk_start_idx*100:(expected_idx+1)*100] = cache["tile100"]["coords"]

            for key in mmaps:
                mmaps[key]["embs"].flush()
                mmaps[key]["coords"].flush()

            for key in cache:
                cache[key]["embs"].clear()
                cache[key]["coords"].clear()
                
            pending_count = 0
            
            with open(state_file, "w") as f:
                f.write(str(expected_idx + 1))

    # Shutdown thread execution resources safely
    executor.shutdown(wait=True)

    # --- FINAL CONVERSION STEP ---
    print("\nProcessing finished. Packing binary blocks into final 3 compressed NPZ files...")
    for key in ["full", "tile4", "tile100"]:
        out_npz = os.path.join(OUTPUT_DIR, f"embeddings_{key}.npz")
        final_embs = np.memmap(FILE_PATHS[key]["embs"], dtype="float32", mode="r", shape=shapes[key]["embs"])
        final_coords = np.memmap(FILE_PATHS[key]["coords"], dtype="float32", mode="r", shape=shapes[key]["coords"])
        
        np.savez_compressed(out_npz, embeddings=final_embs, centroids=final_coords)
        
        del final_embs, final_coords
        os.remove(FILE_PATHS[key]["embs"])
        os.remove(FILE_PATHS[key]["coords"])

    if os.path.exists(state_file):
        os.remove(state_file)

    print(f"Success! Output datasets stored in {OUTPUT_DIR}.")

if __name__ == "__main__":
    main()