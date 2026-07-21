"""
End-to-end orchestrator.

Wires together query parsing (LLM -> DAG plan), DAG execution, and result
export. This is the main entry point external callers should use.
"""

import os
import re
import time
import warnings
warnings.filterwarnings("ignore")

import geopandas as gpd

import config, models
from executor import PipelineContext, PipelineExecutor
from query_parser import parse_query

import argparse


def build_context() -> PipelineContext:
    """Load all long-lived assets once: demographic embeddings, the
    demo-similarity model, and the vision encoder."""
    demo_gdf, ae_embeddings, demo_model = models.load_demographic_assets()
    vision_encoder = models.VisionEncoder()
    return PipelineContext(demo_gdf, ae_embeddings, demo_model, vision_encoder)


def _safe_filename(user_query: str, max_len: int = 50) -> str:
    safe = re.sub(r"[^0-9a-zA-Z]+", "_", user_query).strip("_")
    return safe[:max_len]


def export_result(
    gdf: gpd.GeoDataFrame, user_query: str, output_dir: str = config.OUTPUT_DIR
) -> str:
    """Write the final GeoDataFrame to a GeoJSON file and return its path."""
    if gdf is None or gdf.empty:
        print("No features passed the pipeline's criteria. Nothing to export.")
        return None

    for col in gdf.columns:
        if col != "geometry" and gdf[col].dtype == "object":
            gdf[col] = gdf[col].astype(str)

    output_dir = os.path.join(output_dir, _safe_filename(user_query))
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{_safe_filename(user_query)}.shp")
    gdf.to_file(output_path)
    print(f"Saved output to: {output_path}")
    return output_path


def run_pipeline(user_query: str, context: PipelineContext) -> gpd.GeoDataFrame:
    """Parse `user_query` into a DAG plan, execute it, and export the result."""
    print(f"\nQuery: '{user_query}'")

    plan = parse_query(user_query)
    print(f"Resolved {len(plan.steps)}-step plan:")
    for step in plan.steps:
        print(
            f"  {step.step_id}. [{step.operation}] {step.description} -> {step.output_variable}"
        )

    executor = PipelineExecutor(context)
    result_gdf = executor.run_plan(plan)

    export_result(result_gdf, user_query)
    return result_gdf


if __name__ == "__main__":
    begin = time.time()
    parser = argparse.ArgumentParser(description="Query Earth Engine")
    parser.add_argument("-query", type=str, help="The query to search for")
    args = parser.parse_args()

    context = build_context()
    run_pipeline(args.query, context)

    print(f"Processed the query in {time.time() - begin} seconds.")
