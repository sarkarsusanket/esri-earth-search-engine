"""
DAG executor.

Walks a `QueryPlan` (an ordered list of `PipelineStep`s) and executes each
step against an in-memory variable table, resolving `inputs` references to
prior steps' outputs as it goes. Every step's result is a GeoDataFrame in
the standard schema, so tool operations can consume the output of *any*
prior step regardless of which operation produced it.
"""

from typing import Dict

import geopandas as gpd

import schema
from query_parser import QueryPlan, PipelineStep
from operations import geocode as geocode_op
from operations import demo as demo_op
from operations import vision as vision_op
from operations.tool import TOOL_DISPATCH


class PipelineContext:
    """Holds the long-lived assets (models, embeddings) needed across steps,
    so they're loaded once per session rather than once per step."""

    def __init__(self, demo_gdf, ae_embeddings, demo_model, vision_encoder):
        self.demo_gdf = demo_gdf
        self.ae_embeddings = ae_embeddings
        self.demo_model = demo_model
        self.vision_encoder = vision_encoder


class PipelineExecutor:
    """Executes a QueryPlan step by step against a shared variable table."""

    def __init__(self, context: PipelineContext):
        self.context = context
        self.variables: Dict[str, gpd.GeoDataFrame] = {}

    def _resolve(self, variable_name: str) -> gpd.GeoDataFrame:
        if variable_name not in self.variables:
            raise KeyError(f"Step references undefined variable '{variable_name}'.")
        return self.variables[variable_name]

    def _single_input(self, step: PipelineStep) -> gpd.GeoDataFrame:
        """Most geocode/demo/vision steps take zero or one input region."""
        if not step.inputs:
            return None
        return self._resolve(step.inputs[0])

    def run_step(self, step: PipelineStep) -> gpd.GeoDataFrame:
        params = step.parameters

        if step.operation == "geocode":
            result = geocode_op.geocode(params.get("target"))

        elif step.operation == "demo":
            region = self._single_input(step)
            result = demo_op.search_demographics(
                target=params.get("target"),
                region=region,
                demo_gdf=self.context.demo_gdf,
                ae_embeddings=self.context.ae_embeddings,
                clip_model=self.context.demo_model,
            )

        elif step.operation == "vision":
            region = self._single_input(step)
            result = vision_op.search_vision(
                target=params.get("target"),
                region=region,
                vision_encoder=self.context.vision_encoder,
                resolution=params.get("resolution"),
            )

        elif step.operation == "tool":
            result = self._run_tool_step(step)

        else:
            raise ValueError(
                f"Unknown operation '{step.operation}' in step {step.step_id}."
            )

        self.variables[step.output_variable] = result
        return result

    def _run_tool_step(self, step: PipelineStep) -> gpd.GeoDataFrame:
        action = step.parameters.get("target")
        handler = TOOL_DISPATCH.get(action)
        if handler is None:
            raise ValueError(
                f"Unsupported tool action '{action}' in step {step.step_id}."
            )

        inputs = [self._resolve(name) for name in step.inputs]

        if action == "buffer":
            if len(inputs) != 1:
                raise ValueError(
                    f"'buffer' expects exactly 1 input, got {len(inputs)} in step {step.step_id}."
                )
            return handler(inputs[0], step.parameters.get("buffer_distance_km"))

        if len(inputs) != 2:
            raise ValueError(
                f"'{action}' expects exactly 2 inputs, got {len(inputs)} in step {step.step_id}."
            )
        return handler(inputs[0], inputs[1])

    def run_plan(self, plan: QueryPlan, verbose: bool = True) -> gpd.GeoDataFrame:
        for step in plan.steps:
            if verbose:
                print(f"[step {step.step_id}] {step.operation}: {step.description}")
            result = self.run_step(step)
            if verbose:
                print(f"  -> '{step.output_variable}': {len(result)} feature(s)")
        return self.variables[plan.final_variable]
