import os
import time
from llama_cpp import Llama

llm = None

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

Exanple 4:
Query: "Find farmlands in relatively poorer regions"
Output:
You should return :

{
  "pipeline_steps": [
    {
      "operation": "demo",
      "parameters": {
        "target": "poor regions",
        "resolution": null,
        "buffer_distance_km": null
      },
      "inputs": [],
      "output_variable": "A"
    },
    {
      "operation": "vision",
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

**Return ONLY a valid JSON object** of the form {"pipeline_steps": [...]}. Remember the supported ops: {"geocode", "demo", "vision", "tool"}
"""

def initialize():
    global llm
    model_path = os.path.join(
        os.path.dirname(__file__),
        "models",
        rf"E:\Weights\gguf\qwen2.5-3b-instruct-q8_0.gguf"
    )

    llm = Llama(
        model_path=model_path,
        n_ctx=1024,
        n_threads=20,
        verbose=False,
    )
    _ = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": ROUTER_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": "Dont output anything. Just print hello fast."
            }
        ],
        temperature=0,
        max_tokens=8
    )

# 1. Initialize the model
print("Loading model...")
start_load = time.time()
initialize()
print(f"Model loaded in {time.time() - start_load:.2f} seconds.")

# 2. Benchmark response time
start_gen = time.time()
response = llm.create_chat_completion(
    messages=[
        {
            "role": "system",
            "content": ROUTER_SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": "QUERY: Find all the circular farmlands in the poorer regions of LA."
        }
    ],
    temperature=0,
    max_tokens=256
)
duration = time.time() - start_gen

# 3. Output results
content = response["choices"][0]["message"]["content"]
tokens_used = response["usage"]["completion_tokens"]

print("\n--- Response ---")
print(content)
print("----------------")
print(f"Time taken: {duration:.2f} seconds")
print(f"Speed: {tokens_used / duration:.2f} tokens/sec")