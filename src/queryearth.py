"""
QueryEarth DLPK entry point.

Loads all long-lived assets once (demographic embeddings + model, vision
text encoder, TurboQuant vision indices, local text embedder, local GGUF
query router), then serves `predict(query)` calls cheaply against them.

    from queryearth import QueryEarth
    qe = QueryEarth()                 # heavy, one-time load
    fs = qe.find("find golf courses near wealthy suburbs in Texas")
"""
import logging

# Silence Fiona's environment logger before importing fiona/geopandas
logging.getLogger("fiona._env").setLevel(logging.CRITICAL)
logging.getLogger("fiona").setLevel(logging.CRITICAL)

import json
import os
import time
import warnings
warnings.filterwarnings("ignore")

import geopandas as gpd
from shapely.geometry.polygon import orient

import config
import models
import query_parser
from executor import PipelineContext, PipelineExecutor
from turboquant_index import TurboQuantSearchIndex
from schema import GEOMETRY_COL, SCORE_COL
from operations.grounding import SpatialGrounder


# ------------------------------------------------------------------
# QueryEarth
# ------------------------------------------------------------------
class QueryEarth:
    """Loads every long-lived asset once at construction time; `predict()`
    is the cheap, repeatable per-query call."""

    def __init__(self, **kwargs):
        self.name = "QueryEarth"
        self.description = "Natural-language geospatial query pipeline."
        

    def initialize(self, **kwargs):
        demo_gdf, ae_embeddings, demo_model, text_embedder = models.load_demographic_assets()
        vision_encoder = models.VisionEncoder()

        poi_gdf, poi_embedding_df = models.load_poi_assets()

        grounder = SpatialGrounder()

        vision_indices = {}
        for resolution, folder in config.VISION_INDEX_DIRS.items():
            if os.path.isdir(folder) and os.path.exists(os.path.join(folder, "meta.json")):
                vision_indices[resolution] = TurboQuantSearchIndex(folder)
            else:
                print(f"No vision index found for resolution '{resolution}' at {folder} (skipping).")

        self.context = PipelineContext(
            demo_gdf, ae_embeddings, demo_model, text_embedder,
            vision_encoder, vision_indices, poi_gdf, poi_embedding_df, grounder
        )
        self.executor = PipelineExecutor(self.context)


    def find(self, query: str, save_gdf = False, **kwargs):
        """Given a natural-language query string, parse it into a pipeline
        plan, execute it, and return the result as an arcgis FeatureSet."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("QueryEarth.predict expects a non-empty query string.")
        
        begin = time.time()

        plan = query_parser.parse_query(query)
        result_gdf = self.executor.run_plan(plan)

        print(f"Found the objects in {(time.time() - begin)} seconds.")

        if save_gdf:
            os.makedirs(rf"results\{query.replace(" ", "_")}", exist_ok=True)
            result_gdf.to_file(rf"results\{query.replace(" ", "_")}\output .shp")

        return result_gdf


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="QueryEarth DLPK smoke test")
    parser.add_argument("-query", type=str, required=True, help="The query to search for")
    args = parser.parse_args()

    qe = QueryEarth()
    qe.initialize()
    result = qe.find(args.query)
