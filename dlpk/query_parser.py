"""
Natural-language query parsing.

Converts an arbitrary user query into a directed acyclic graph (DAG) of
pipeline steps, using a locally-loaded GGUF model (llama.cpp) instead of a
network call to Ollama — this is the one part of the pipeline where low
latency depends most on avoiding a round trip, so the router model is
loaded once and kept warm for the DLPK session's lifetime.

Each step is one of:
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
from dataclasses import dataclass, field

import config

SUPPORTED_OPERATIONS = {"geocode", "demo", "vision", "tool", "poi"}
SUPPORTED_TOOL_ACTIONS = {"buffer", "union", "intersection", "difference", "add"}
SUPPORTED_RESOLUTIONS = set(config.VISION_INDEX_DIRS.keys())

ROUTER_SYSTEM_PROMPT = """You are a geospatial query planning assistant. Convert the
user's natural-language request into a short program: a sequence of
assignment lines, one per step, executed top to bottom. Each line assigns
the result of one function call to a variable name, and later lines can
pass earlier variables in as arguments.

Allowed functions (letter code in parens is just the mnemonic, always write
the full name):

  geocode(place)                (G)   resolve a place name to a region.
                                       `place` is a plain string, e.g. "Los Angeles".
                                       Only places here like LA or Chicago, not things like rivers etc
                                       Takes no variable arguments.

  demo(region?, query)          (D)   demographic similarity search, e.g.
                                       "poorer regions", "areas with high
                                       unemployment". `region` is an optional
                                       prior variable to search within — omit
                                       it (single argument) to search the
                                       whole study area. `query` is a string.

  poi(region?, query)           (P)   point-of-interest search over discrete
                                       amenity classes, e.g. "emergency
                                       services", "coffee shops",
                                       "restaurants". `region` is an optional
                                       prior variable to search within — omit
                                       it (single argument) to search the
                                       whole study area. `query` is a string.
                                       Use this for any request about named/
                                       classified places and services (banks,
                                       schools, parks, hospitals, fuel, etc.).

  vision-high(region?, query)   (VH)  visual search for SMALL/fine objects:
                                       pools, vehicles, structures.
                                       Also if you are buffering  the points from this then no more than 0.01 buffer
  vision-low(region?, query)    (VL)  visual search for LARGE objects / land-use
                                       patterns: forests, farmland, golf
                                       courses, large facilities.
                                       Same argument shape as demo: optional
                                       leading region variable, then a string
                                       query.
                                       Also if you are buffering  the points from this then no more than 0.2 buffer


  buffer(region, km)            (B)   buffer a prior variable by a distance
                                       in kilometers. `region` is a variable,
                                       `km` is a number.

  intersection(a, b)            (I)   set operations on two prior variables.
  union(a, b)                   (U)
  difference(a, b)              (X)   result is a minus b.
  add(a, b)                     (A)

Syntax rules:
- One statement per line: `varname = function(args)`.
- Variable names are short bare identifiers (a, b, c, region1, ...).
- String arguments are double-quoted. Variable arguments are bare identifiers.
- Arguments are comma-separated, in the order shown above.
- The last line's variable name must be exactly `output` — that line's
  result is the answer returned to the user.
- Write ONLY the program. No markdown fences, no comments, no explanation,
  no trailing text — just the lines.

You have to be smart about what needs to be directed into vision and what get to go into POI.

Example:
Query: "Find circular farmlands in relatively poorer regions"
Output:
a = demo("poorer regions")
output = vision-low(a, "circular farmlands")

Example:
Query: "Find farmlands which are near a water source"
Output:
output = vision-low("farmlands near water source")

Example:
Query: "Find parking lots"
Output:
output = vision-high("parking lots")

Example:
Query: "Find all the emergency services near highway interections within 5 kms of LA"
Output:
a = geocode("Los Angeles")
b = buffer(a, 5)
c = vision-low(b, "highway inetrsection")
d = buffer(c, 2)
output = poi(d, "emergency services")

Example:
Query: "Find residential homes with pools in the poorer regions of downtown LA"
Output:
a = geocode("Los Angeles")
b = demo(a, "poorer regions")
output = vision-high(b, "residential homes with pools")

Example:
Query: "Find solar farms outside dense forests near Phoenix"
Output:
a = geocode("Phoenix")
b = buffer(a, 5)
c = vision-low(b, "dense forests")
d = buffer(c, 0.2)
e = vision-high(b, "solar farms")
output = difference(e, d)

Remember: only geocode, demo, poi, vision-high, vision-low, buffer, intersection,
union, difference, add are allowed. Write only the program lines. Solve the problem in steps using the tools available to you.
"""


@dataclass
class PipelineStep:
    step_id: int
    operation: str
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


# ------------------------------------------------------------------
# DSL parsing
#
# Lines look like:
#   a = geocode("Los Angeles")
#   b = demo(a, "poorer regions")
#   c = vision-high(a, "houses with pools")
#   output = intersection(c, b)
#
# Each function name maps to an internal (operation, resolution) pair that
# PipelineStep/executor.py already understand — the DSL is just a friendlier
# surface syntax over the same step model used before.
# ------------------------------------------------------------------

# function name -> (operation, resolution or None, tool-action or None)
_FUNC_MAP = {
    "geocode": ("geocode", None, None),
    "demo": ("demo", None, None),
    "poi": ("poi", None, None),
    "vision-high": ("vision", "high", None),
    "vision-low": ("vision", "low", None),
    "buffer": ("tool", None, "buffer"),
    "intersection": ("tool", None, "intersection"),
    "union": ("tool", None, "union"),
    "difference": ("tool", None, "difference"),
    "add": ("tool", None, "add"),
}

_LINE_RE = re.compile(
    r'^\s*(?P<var>[A-Za-z_]\w*)\s*=\s*(?P<func>[A-Za-z_][A-Za-z_\-]*)\s*\((?P<args>.*)\)\s*$'
)
_NUMBER_RE = re.compile(r'^-?\d+(\.\d+)?$')


def _split_args(arg_str: str):
    """Split a DSL argument list on top-level commas, respecting quotes.
    Arguments never nest function calls in this grammar, so we only need to
    track whether we're inside a quoted string."""
    args, current, quote = [], [], None
    for ch in arg_str:
        if quote:
            current.append(ch)
            if ch == quote:
                quote = None
        elif ch in ('"', "'"):
            quote = ch
            current.append(ch)
        elif ch == ',':
            args.append(''.join(current))
            current = []
        else:
            current.append(ch)
    if current:
        args.append(''.join(current))
    return [a.strip() for a in args if a.strip()]


def _classify_arg(token: str):
    """Return ('text'|'number'|'var', value) for a single DSL argument."""
    if (token.startswith('"') and token.endswith('"')) or (token.startswith("'") and token.endswith("'")):
        return ('text', token[1:-1])
    if _NUMBER_RE.match(token):
        num = float(token)
        return ('number', int(num) if num.is_integer() else num)
    return ('var', token)


def _parse_dsl_line(line: str, step_id: int) -> PipelineStep:
    match = _LINE_RE.match(line)
    if not match:
        raise ValueError(f"Could not parse line as 'var = func(args)': {line!r}")

    var_name = match.group("var")
    func_name = match.group("func").lower()
    if func_name not in _FUNC_MAP:
        raise ValueError(f"Unsupported function '{func_name}' in line: {line!r}")

    operation, resolution, tool_action = _FUNC_MAP[func_name]
    classified = [_classify_arg(tok) for tok in _split_args(match.group("args"))]

    var_args = [v for kind, v in classified if kind == 'var']
    text_args = [v for kind, v in classified if kind == 'text']
    number_args = [v for kind, v in classified if kind == 'number']

    parameters = {"target": None, "resolution": None, "buffer_distance_km": None}

    if operation == "geocode":
        if not text_args:
            raise ValueError(f"geocode() needs a string place name: {line!r}")
        parameters["target"] = text_args[0]
        inputs = []

    elif operation == "demo" or operation == "poi" or (operation == "vision"):
        if not text_args:
            raise ValueError(f"{func_name}() needs a string query: {line!r}")
        parameters["target"] = text_args[0]
        parameters["resolution"] = resolution
        inputs = var_args  # zero or one region variable

    elif operation == "tool" and tool_action == "buffer":
        if not var_args:
            raise ValueError(f"buffer() needs a region variable: {line!r}")
        if not number_args:
            raise ValueError(f"buffer() needs a distance in km: {line!r}")
        parameters["target"] = "buffer"
        parameters["buffer_distance_km"] = number_args[0]
        inputs = var_args[:1]

    else:  # intersection / union / difference / add
        if len(var_args) != 2:
            raise ValueError(f"{func_name}() needs exactly 2 variable arguments: {line!r}")
        parameters["target"] = tool_action
        inputs = var_args

    return PipelineStep(
        step_id=step_id,
        operation=operation,
        parameters=parameters,
        inputs=inputs,
        output_variable=var_name,
    )


def _extract_dsl(raw_content: str) -> str:
    """Strip a leading/trailing ```...``` fence if the model added one
    despite being told not to, and drop blank/comment lines."""
    raw_content = raw_content.strip()
    fence_match = re.search(r"```(?:\w+)?\s*(.*?)\s*```", raw_content, flags=re.DOTALL)
    if fence_match:
        raw_content = fence_match.group(1).strip()
    lines = [
        ln for ln in raw_content.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    return "\n".join(lines)


def _coerce_plan(dsl_text: str) -> QueryPlan:
    lines = [ln for ln in dsl_text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("Parsed plan contains no steps.")

    steps = [_parse_dsl_line(line, step_id=i + 1) for i, line in enumerate(lines)]

    if steps[-1].output_variable.lower() != "output":
        # Be lenient: rename the last step's output rather than failing outright.
        steps[-1].output_variable = "output"
    steps[-1].output_variable = "FINAL_ANSWER"

    return QueryPlan(steps=steps, raw={"dsl": dsl_text})


def _fallback_plan(user_query: str) -> QueryPlan:
    """A degenerate single-step plan used if the local router fails entirely:
    treat the whole query as a vision search over the full study area."""
    step = PipelineStep(
        step_id=1,
        operation="vision",
        parameters={
            "target": user_query,
            "resolution": config.DEFAULT_RESOLUTION,
            "buffer_distance_km": None,
        },
        inputs=[],
        output_variable="FINAL_ANSWER",
    )
    return QueryPlan(steps=[step], raw={})


# ------------------------------------------------------------------
# Local GGUF router model (llama.cpp), loaded once and kept warm
# ------------------------------------------------------------------
_LLM = None


def _get_llm():
    global _LLM
    if _LLM is not None:
        return _LLM

    if not config.ROUTER_GGUF_PATH:
        raise FileNotFoundError(
            "No .gguf router model found under weights/. Place your router GGUF file there."
        )

    from llama_cpp import Llama
    print(f"Loading router model: {config.ROUTER_GGUF_PATH} ...")
    _LLM = Llama(
        model_path=config.ROUTER_GGUF_PATH,
        n_ctx=config.ROUTER_N_CTX,
        n_threads=config.ROUTER_N_THREADS,
        verbose=False,
    )
    # Warm up: pay the first-inference cost now, at load time, not on the
    # first real user query.
    _LLM.create_chat_completion(
        messages=[
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": 'Query: "warmup"\nOutput:'},
        ],
        temperature=0,
        max_tokens=8,
    )
    return _LLM


def warmup():
    """Force the router model to load now, so the first predict() call isn't
    slowed down by a cold model load."""
    _get_llm()


def parse_query(user_query: str) -> QueryPlan:
    """Parse a natural-language geospatial query into an executable QueryPlan."""
    llm = _get_llm()

    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": f'Query: "{user_query}"\nOutput:'},
        ],
        temperature=0,
        max_tokens=1024,
    )
    raw_content = response["choices"][0]["message"]["content"].strip()
    print("Raw content:", raw_content)

    dsl_text = _extract_dsl(raw_content)
    if not dsl_text:
        print(f"Router produced no parseable program. Falling back to a single vision step. Raw output:\n{raw_content}")
        return _fallback_plan(user_query)

    try:
        return _coerce_plan(dsl_text)
    except Exception as e:
        print(f"Failed to parse DSL plan ({e}). Falling back to a single vision step.")
        return _fallback_plan(user_query)