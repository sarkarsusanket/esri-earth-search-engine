"""
grounding.py - Spatial Grounding Pipeline for Vision Search
-----------------------------------------------------------
Downloads single ArcGIS satellite tiles around candidate lat/lon locations at zoom 17,
runs GroundingDINO object detection, and projects detected bounding box centroids
back to (latitude, longitude) coordinates.
"""

import io
import math
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import requests
import torch
import tqdm
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# ==============================================================================
# 1. TILE MATH & MAP PROJECTION UTILITIES (Web Mercator EPSG:3857)
# ==============================================================================
EARTH_RADIUS = 6378137.0  # meters


def latlon_to_single_tile(lat: float, lon: float, zoom: int = 17) -> Tuple[int, int]:
    """Converts a single lat/lon coordinate to its XYZ tile index."""
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile


def global_pixels_to_latlon(px: float, py: float, zoom: int, tile_size: int = 256) -> Tuple[float, float]:
    """Convert absolute continuous pixel coordinates back to lat/lon."""
    n = 2.0 ** zoom
    lon = (px / (n * tile_size)) * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * (py / (n * tile_size)))))
    lat = math.degrees(lat_rad)
    return lat, lon


# ==============================================================================
# 2. SPATIAL GROUNDING PIPELINE CLASS
# ==============================================================================
TILE_URL_TMPL = (
    "https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/"
    "WMTS/1.0.0/default028mm/MapServer/tile/{release_id}/{zoom}/{y}/{x}"
)


class SpatialGrounder:
    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-tiny",
        release_id: str = "32246",
        device: Optional[str] = None,
    ):
        """Initializes model and processor weights synchronously."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
        self.model.eval()
        self.release_id = release_id
        self.session = requests.Session()

    def _fetch_single_tile(self, zoom: int, x: int, y: int) -> Optional[Image.Image]:
        """Downloads a single satellite image tile."""
        url = TILE_URL_TMPL.format(release_id=self.release_id, zoom=zoom, y=y, x=x)
        try:
            resp = self.session.get(url, timeout=10)
            if resp.status_code == 200:
                return Image.open(io.BytesIO(resp.content)).convert("RGB")
        except requests.RequestException:
            pass
        return None

    def _ground_objects(
        self,
        image: Image.Image,
        query: str,
        threshold: float = 0.35,
        text_threshold: float = 0.25,
    ) -> List[Tuple[float, float, float]]:
        """Executes GroundingDINO detection and returns box centers in tile pixel coords."""
        formatted_query = query if query.endswith(".") else f"{query}."
        text_labels = [formatted_query]

        inputs = self.processor(
            images=image, text=text_labels, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]

        detections = []
        for box, score in zip(results["boxes"], results["scores"]):
            box_coords = box.tolist()  # [xmin, ymin, xmax, ymax]
            cx = (box_coords[0] + box_coords[2]) / 2.0
            cy = (box_coords[1] + box_coords[3]) / 2.0
            detections.append((cx, cy, score.item()))

        return detections

    def ground_locations(
        self,
        locations: List[Tuple[float, float]],
        query: str,
        threshold: float = 0.35,
    ) -> List[List[Dict[str, Any]]]:
        """
        Main entry point for downstream search pipelines.
        
        Args:
            locations: List of (lat, lon) tuples from similarity search
            query: Natural language target (e.g. "swimming pool", "farmlands")
            threshold: GroundingDINO object detection confidence threshold
            
        Returns:
            A list of detection result lists (one list of matches per location point)
        """
        all_results = []
        zoom = 17  # Fixed zoom level 17

        for lat, lon in tqdm.tqdm(locations):
            # 1. Get tile index for candidate lat/lon at zoom 17
            x_tile, y_tile = latlon_to_single_tile(lat, lon, zoom=zoom)

            # 2. Fetch single tile image
            tile_img = self._fetch_single_tile(zoom, x_tile, y_tile)
            if tile_img is None:
                all_results.append([])
                continue

            # 3. Run object detection on single tile
            pixel_detections = self._ground_objects(tile_img, query, threshold=threshold)

            # 4. Project pixel centers to global Lat/Lon
            grounded_locations = []
            origin_global_px_x = x_tile * 256
            origin_global_px_y = y_tile * 256

            for cx, cy, score in pixel_detections:
                global_px_x = origin_global_px_x + cx
                global_px_y = origin_global_px_y + cy

                target_lat, target_lon = global_pixels_to_latlon(global_px_x, global_px_y, zoom)

                grounded_locations.append({
                    "lat": target_lat,
                    "lon": target_lon,
                    "confidence": round(score, 4),
                    "query": query,
                })

            all_results.append(grounded_locations)

        return all_results