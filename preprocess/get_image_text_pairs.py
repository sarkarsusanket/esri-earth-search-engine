import os
import random
import time
import json
from io import BytesIO
import concurrent.futures
import pandas as pd
import numpy as np
import geopandas as gpd
import requests
from PIL import Image
from shapely.geometry import Point, Polygon
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from ollama import chat
import base64

# --- Configuration ---
# SHAPEFILE_PATH = rf"/data/susanket/data/global/world.geojson"
SHAPEFILE_PATH = rf"/data/susanket/data/global/world.geojson"
/data/susanket/queryearth/vlm_captions_1M.parquet
OUTPUT_PARQUET = r""
CHECKPOINT_FILE = r"/data/susanket/queryearth/vlm_captions_ckpt.json"

N_SAMPLES = 10_000_000               # Number of random points to sample inside the polygon
CHUNK_SIZE = 1000              # Balanced chunk scale suited for LLM/VLM generation latencies
MAX_DOWNLOAD_THREADS = 60      
VLM_MODEL = 'ministral-3:3b'

# --- Coordinate Projection Helper ---
def wgs84_to_web_mercator(lon, lat):
    lon_rad = np.radians(lon)
    x = 6378137.0 * lon_rad
    y = 6378137.0 * np.log(np.tan(np.pi / 4.0 + np.radians(lat) / 2.0))
    return x, y

def web_mercator_to_wgs84(x, y):
    lon = np.degrees(x / 6378137.0)
    lat = np.degrees(2.0 * np.arctan(np.exp(y / 6378137.0)) - np.pi / 2.0)
    return lon, lat

# --- Sample Point uniformly from Polygon ---
def sample_point_from_polygon(poly):
    minx, miny, maxx, maxy = poly.bounds
    while True:
        p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if poly.contains(p):
            return p.x, p.y

# --- Thread Worker: Dynamic Point Image Grabber ---
def download_point_image(lon, lat, radius = 224):
    service_url = "https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/export"
    try:
        size_dim = radius
        mx, my = wgs84_to_web_mercator(lon, lat)
        
        meters_per_pixel = 0.5 
        half_span = (size_dim * meters_per_pixel) / 2.0
        
        mx_min, mx_max = mx - half_span, mx + half_span
        my_min, my_max = my - half_span, my + half_span
        
        export_params = {
            "bbox": f"{mx_min},{my_min},{mx_max},{my_max}",
            "bboxSR": "3857",
            "size": f"{size_dim},{size_dim}", 
            "imageSR": "3857",
            "format": "png",
            "transparent": "true",
            "f": "image",
        }
        
        response = requests.get(service_url, params=export_params, timeout=15)
        if response.status_code == 200:
            # --- OPTIMIZATION: Re-encode PNG to lightweight JPEG bytes ---
            img = Image.open(BytesIO(response.content)).convert("RGB")
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=85, optimize=True)
            jpeg_bytes = buffer.getvalue()
            
            return jpeg_bytes, size_dim
    except Exception:
        pass
    return None, None

# --- VLM Worker ---
# def generate_caption(img_bytes):
#     try:
#         response = chat(
#             model=VLM_MODEL,
#             messages=[{
#                 'role': 'user',
#                 'content': 'Describe this aerial view in a crisp, single-sentence caption focusing on the feature characteristics, like structures, shapes, colors and proximity of objects.',
#                 'images': [img_bytes]
#             }],
#         )
#         return response.message.content.strip()
#     except Exception as e:
#         return f"Error generation failed: {str(e)}"
def generate_caption(img_bytes):
    try:
        # Ollama's direct HTTP endpoint requires images to be base64 strings
        b64_image = base64.b64encode(img_bytes).decode('utf-8')
        
        url = "http://localhost:11434/api/chat"
        payload = {
            "model": VLM_MODEL,
            "messages": [{
                "role": "user",
                "content": "Describe this aerial view in a crisp, single-sentence caption focusing on the feature characteristics, like structures, shapes, colors and proximity of objects.",
                "images": [b64_image]
            }],
            "stream": False  # Crucial: turning off streaming drops HTTP chunking overhead
        }
        
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            return result['message']['content'].strip()
        else:
            return f"Error: API returned status {response.status_code}"
            
    except Exception as e:
        return f"Error generation failed: {str(e)}"

def main():
    begin = time.time()
    print(f"Reading input shapefile structure from: {SHAPEFILE_PATH}")
    gdf = gpd.read_file(SHAPEFILE_PATH)
    print(gdf.crs)
    
    unified_poly = gdf.unary_union
    
    print(f"Generating {N_SAMPLES} target sampling spatial configurations...")
    sampled_points = [sample_point_from_polygon(unified_poly) for _ in range(N_SAMPLES)]
    
    # --- OPTIMIZATION: Updated Schema to support the image column as binary bytes ---
    target_schema = pa.schema([
        pa.field('sample_id', pa.int64()),
        pa.field('lat', pa.float32()),
        pa.field('lon', pa.float32()),
        pa.field('image_size', pa.int32()),
        pa.field('caption', pa.string()),
        pa.field('image_bytes', pa.binary()) 
    ])

    start_idx = 0
    if os.path.exists(CHECKPOINT_FILE) and os.path.exists(OUTPUT_PARQUET):
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                start_idx = json.load(f).get("processed_rows", 0)
            print(f"Resuming processing from checkpoint sequence: {start_idx}")
        except Exception:
            start_idx = 0

    if start_idx == 0 and os.path.exists(OUTPUT_PARQUET):
        os.remove(OUTPUT_PARQUET)
        
    # --- OPTIMIZATION: Enforce ZSTD compression to keep file size small ---
    writer = pq.ParquetWriter(OUTPUT_PARQUET, target_schema, compression='ZSTD')

    # --- Main Chunk Processing Architecture Loop ---
    for c_start in range(start_idx, N_SAMPLES, CHUNK_SIZE):
        c_end = min(c_start + CHUNK_SIZE, N_SAMPLES)
        chunk_points = sampled_points[c_start:c_end]
        
        print(f"\n--- Processing Samples {c_start} to {c_end} of {N_SAMPLES} ---")
        
        downloaded_images = [None] * len(chunk_points)
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_DOWNLOAD_THREADS) as executor:
            futures = {
                executor.submit(download_point_image, pt[0], pt[1]): idx 
                for idx, pt in enumerate(chunk_points)
            }
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Downloading Imagery Maps"):
                idx = futures[future]
                downloaded_images[idx] = future.result()

        # --- OPTIMIZATION: Parallel VLM Inference Execution ---
        valid_images = []
        valid_indices = []
        
        # Filter out failed downloads first
        for idx, img_data in enumerate(downloaded_images):
            if img_data is not None and img_data[0] is not None:
                valid_images.append(img_data)
                valid_indices.append(idx)
        
        captions = [None] * len(valid_images)
        
        # Parallelize VLM requests using a ThreadPoolExecutor
        # Note: Set max_workers depending on how many parallel streams your system/GPU can handle (e.g., 4-8)
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_DOWNLOAD_THREADS) as vlm_executor:
            vlm_futures = {
                vlm_executor.submit(generate_caption, img_bytes): i 
                for i, (img_bytes, _) in enumerate(valid_images)
            }
            for future in tqdm(concurrent.futures.as_completed(vlm_futures), total=len(vlm_futures), desc="VLM Captioning"):
                i = vlm_futures[future]
                captions[i] = future.result()

        # Build clean output lists straight into the Arrow Table structure
        flat_rows = []
        for i, idx in enumerate(valid_indices):
            lon, lat = chunk_points[idx]
            img_bytes, target_dim = valid_images[i]
            caption = captions[i]
            
            flat_rows.append({
                'sample_id': c_start + idx,
                'lat': float(lat),
                'lon': float(lon),
                'image_size': int(target_dim),
                'caption': caption,
                'image_bytes': img_bytes 
            })

        if flat_rows:
            out_df = pd.DataFrame(flat_rows)
            table = pa.Table.from_pandas(out_df, schema=target_schema)
            writer.write_table(table)

        with open(CHECKPOINT_FILE, 'w') as f:
            json.dump({"processed_rows": c_end}, f)

    if writer:
        writer.close()
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        
    print(f"Dataset compilation successful. Elements written to target workspace: {OUTPUT_PARQUET}. Took around {time.time()-begin} seconds.")

if __name__ == "__main__":
    main()