"""
Natural-language query parsing.

Converts an arbitrary user query into a directed acyclic graph (DAG) of
pipeline steps. Each step is one of:

  - "geocode": resolve a place name to a bounding geometry
  - "demo":    similarity search over demographic embeddings
  - "vision":  similarity search over visual (CLIP) embeddings, at a
               chosen spatial resolution
  - "tool":    a pure spatial/set operation (buffer, union, intersection,
               difference, etc.) applied to one or more prior outputs

Steps reference each other's outputs by `output_variable` name, so the
parser is effectively writing a small program that `executor.py` later
interprets.
"""
import re
import json
from dataclasses import dataclass, field
from typing import Any, Optional

import ollama

import config

SUPPORTED_OPERATIONS = {"geocode", "georeference", "demo", "vision", "tool"}
SUPPORTED_TOOL_ACTIONS = {"buffer", "union", "intersection", "difference", "add"}
SUPPORTED_RESOLUTIONS = set(config.RESOLUTION_TO_FILE.keys())

ROUTER_SYSTEM_PROMPT = """You are a geospatial query planning assistant. Convert the
user's natural-language request into a JSON pipeline of steps that, executed in
order, answer the query.

Each step has:
  - "step_id": integer, sequential starting at 1
  - "operation": one of "geocode", "demo", "vision", "tool"
  - "description": short human-readable description of what this step does
  - "parameters": object with keys "target", "resolution", "buffer_distance_km"
      - "target": for geocode -> place name.
                  for demo   -> the demographic/socioeconomic text query.
                  for vision -> the visual object/land-use text query.
                  for tool   -> one of "buffer", "union", "intersection", "difference", "add".
      - "resolution": only used for "vision" steps. One of "low", "high".
                       low = ~2km tiles (broad context),
                       high = ~200m tiles (fine detail, small objects). null otherwise.
      - "buffer_distance_km": only used for "tool" steps with target "buffer". null otherwise.
  - "inputs": list of output_variable names this step depends on (empty list if none)
  - "output_variable": a short variable name (e.g. "A", "B", "C_buffered") this
      step's result is stored as, for later steps to reference

Rules:
- "geocode" steps always have empty "inputs" (they start a region from a place name).
- "demo" and "vision" steps take exactly one input: the region they search within.
  If no prior region was established, "inputs" is empty (search the whole study area).
- "tool" steps take one input for "buffer", and two inputs for "union"/"intersection"/
  "difference"/"add" (in that order — for "difference", the result is inputs[0] minus inputs[1]).
- The final step's output_variable must be "FINAL_ANSWER".
- Choose "resolution" for vision steps based on the size of the object being searched for:
  small/fine objects (pools, vehicles, individual structures) -> "high";
  large objects/land-use patterns (forests, farmland, golf courses, large facilities) -> "low".
- **Return ONLY a valid JSON object** of the form {"pipeline_steps": [...]}. No markdown,
  no explanations, no trailing text.
- Do not output any tools or modes like "confirmation" thats not mwentiobned here.

Example 1:
Query: "find suburban neighbourhoods in Texas with backyard swimming pools but no golf courses within 5 kms"
Output:
You should return :
{"pipeline_steps": [
  {"step_id": 1, "operation": "geocode", "description": "Find boundaries for Texas",
   "parameters": {"target": "Texas", "resolution": null, "buffer_distance_km": null},
   "inputs": [], "output_variable": "A"},
  {"step_id": 2, "operation": "demo", "description": "Find suburban neighborhoods inside Texas",
   "parameters": {"target": "suburban neighborhoods", "resolution": null, "buffer_distance_km": null},
   "inputs": ["A"], "output_variable": "B"},
  {"step_id": 3, "operation": "vision", "description": "Identify golf courses in Texas using high resolution",
   "parameters": {"target": "golf courses", "resolution": "high", "buffer_distance_km": null},
   "inputs": ["A"], "output_variable": "C"},
  {"step_id": 4, "operation": "tool", "description": "Buffer golf courses by 5 kilometers",
   "parameters": {"target": "buffer", "resolution": null, "buffer_distance_km": 5.0},
   "inputs": ["C"], "output_variable": "C_buffered"},
  {"step_id": 5, "operation": "tool", "description": "Remove golf-course buffer zones from suburban neighborhoods",
   "parameters": {"target": "difference", "resolution": null, "buffer_distance_km": null},
   "inputs": ["B", "C_buffered"], "output_variable": "D"},
  {"step_id": 6, "operation": "vision", "description": "Find backyard swimming pools in the remaining area using low resolution",
   "parameters": {"target": "backyard swimming pools", "resolution": "low", "buffer_distance_km": null},
   "inputs": ["D"], "output_variable": "FINAL_ANSWER"}
]}

Example 2:
Query: "Locate all low-income areas anywhere that contain visible cargo shipping containers"
Output:
{"pipeline_steps": [
  {"step_id": 1, "operation": "demo", "description": "Isolate all low-income demographic regions globally", "parameters": {"target": "low-income areas", "resolution": null, "buffer_distance_km": null}, "inputs": [], "output_variable": "A"},
  {"step_id": 2, "operation": "vision", "description": "Detect individual cargo shipping containers inside low-income polygons using high resolution", "parameters": {"target": "cargo shipping containers", "resolution": "high", "buffer_distance_km": null}, "inputs": ["A"], "output_variable": "FINAL_ANSWER"}
]}

Example 3:
Query: "In France, pinpoint wealthy high-density urban polygons that also overlap with visible solar panel installations"
Output:
You should return :
{"pipeline_steps": [
  {"step_id": 1, "operation": "geocode", "description": "Get outline of France", "parameters": {"target": "France", "resolution": null, "buffer_distance_km": null}, "inputs": [], "output_variable": "A"},
  {"step_id": 2, "operation": "demo", "description": "Find wealthy demographics within France", "parameters": {"target": "wealthy neighborhoods", "resolution": null, "buffer_distance_km": null}, "inputs": ["A"], "output_variable": "B"},
  {"step_id": 3, "operation": "demo", "description": "Find high-density urban populations within France", "parameters": {"target": "high-density urban areas", "resolution": null, "buffer_distance_km": null}, "inputs": ["A"], "output_variable": "C"},
  {"step_id": 4, "operation": "tool", "description": "Intersect wealthy zones with high-density zones", "parameters": {"target": "intersection", "resolution": null, "buffer_distance_km": null}, "inputs": ["B", "C"], "output_variable": "D"},
  {"step_id": 5, "operation": "vision", "description": "Identify micro-solar panel installations inside target areas using high resolution", "parameters": {"target": "solar panels", "resolution": "high", "buffer_distance_km": null}, "inputs": ["D"], "output_variable": "FINAL_ANSWER"}
]}



Exanple 4:
Query: "Find farmlands in relatively poorer regions"
Output:
You should return :

{
  "pipeline_steps": [
    {
      "step_id": 1,
      "operation": "demo",
      "description": "Identify relatively poorer socio-economic regions",
      "parameters": {
        "target": "poor regions",
        "resolution": null,
        "buffer_distance_km": null
      },
      "inputs": [],
      "output_variable": "A"
    },
    {
      "step_id": 2,
      "operation": "vision",
      "description": "Identify farmland land-use patterns within the poorer regions using low resolution",
      "parameters": {
        "target": "farmlands",
        "resolution": "low",
        "buffer_distance_km": null
      },
      "inputs": ["A"],
      "output_variable": "FINAL_ANSWER"
    }
  ]
}

Example 5: Identify high-density residential zones that sit entirely within severe wildfire hazard regions
You should return :
{
  "pipeline_steps": [
    {
      "step_id": 1,
      "operation": "demo",
      "description": "Isolate wildlife areas",
      "parameters": {"target": "wildlife hazard zones", "resolution": null, "buffer_distance_km": null},
      "inputs": [],
      "output_variable": "A"
    },
    {
      "step_id": 2,
      "operation": "vision",
      "description": "Detect high densiy residential areas using low resolution inside A",
      "parameters": {"target": "high density residential areas", "resolution": "low", "buffer_distance_km": null},
      "inputs": ["A"],
      "output_variable": "FINAL_ANSWER"
    }]
}

**Return ONLY a valid JSON object** of the form {"pipeline_steps": [...]}. Remember the supported ops: {"geocode", "demo", "vision", "tool"}
"""


@dataclass
class PipelineStep:
    step_id: int
    operation: str
    description: str
    parameters: dict
    inputs: list
    output_variable: str

    def __post_init__(self):
        if self.operation not in SUPPORTED_OPERATIONS:
            raise ValueError(
                f"Unsupported operation '{self.operation}' in step {self.step_id}"
            )
        if self.operation == "tool":
            action = self.parameters.get("target")
            if action not in SUPPORTED_TOOL_ACTIONS:
                raise ValueError(
                    f"Unsupported tool action '{action}' in step {self.step_id}"
                )
        if self.operation == "vision":
            resolution = self.parameters.get("resolution") or config.DEFAULT_RESOLUTION
            if resolution not in SUPPORTED_RESOLUTIONS:
                raise ValueError(
                    f"Unsupported resolution '{resolution}' in step {self.step_id}"
                )
            self.parameters["resolution"] = resolution


@dataclass
class QueryPlan:
    steps: list  # list[PipelineStep], in execution order
    raw: dict = field(default_factory=dict)

    @property
    def final_variable(self) -> str:
        return self.steps[-1].output_variable if self.steps else None


def _coerce_plan(parsed: dict) -> QueryPlan:
    raw_steps = parsed.get("pipeline_steps", [])
    if not raw_steps:
        raise ValueError("Parsed plan contains no pipeline_steps.")

    steps = [
        PipelineStep(
            step_id=s["step_id"],
            operation=s["operation"],
            description=s.get("description", ""),
            parameters=s.get("parameters", {}) or {},
            inputs=s.get("inputs", []) or [],
            output_variable=s["output_variable"],
        )
        for s in raw_steps
    ]

    if steps[-1].output_variable != "FINAL_ANSWER":
        # Be lenient: rename the last step's output rather than failing outright.
        steps[-1].output_variable = "FINAL_ANSWER"

    return QueryPlan(steps=steps, raw=parsed)


def _fallback_plan(user_query: str) -> QueryPlan:
    """A degenerate single-step plan used if the LLM router fails entirely:
    treat the whole query as a vision search over the full study area."""
    step = PipelineStep(
        step_id=1,
        operation="vision",
        description="Fallback: treat entire query as a visual search term.",
        parameters={
            "target": user_query,
            "resolution": config.DEFAULT_RESOLUTION,
            "buffer_distance_km": None,
        },
        inputs=[],
        output_variable="FINAL_ANSWER",
    )
    return QueryPlan(steps=[step], raw={})


def parse_query(user_query: str, model: str = config.OLLAMA_ROUTER_MODEL) -> QueryPlan:
    """Parse a natural-language geospatial query into an executable QueryPlan."""
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": f'Query: "{user_query}"\nOutput:'},
        ],
        options={"temperature": 0.0},
    )

    raw_content = response.message.content.strip()
    
    pattern = r"```json\s*(.*?)\s*```"
    matches = re.findall(pattern, raw_content, flags=re.DOTALL | re.IGNORECASE)

    if matches:
        last_json_string = matches[-1]
        print(last_json_string)
    else:
        print(raw_content)
        print("No JSON blocks found.")
        quit()

    cleaned = last_json_string.replace("```json", "").replace("```", "").strip()
    
    try:
        parsed = json.loads(cleaned)
        return _coerce_plan(parsed)
    except Exception as e:
        print(f"Failed to parse query plan ({e}). Raw model output:\n{raw_content}")
        return _fallback_plan(user_query)
