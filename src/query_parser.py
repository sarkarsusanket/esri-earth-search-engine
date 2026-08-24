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

from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

import config

SUPPORTED_OPERATIONS = {"geocode", "demo", "vision", "tool", "poi"}
SUPPORTED_TOOL_ACTIONS = {"buffer", "union", "intersection", "difference", "add"}
SUPPORTED_RESOLUTIONS = set(config.VISION_INDEX_DIRS.keys())

ROUTER_SYSTEM_PROMPT = """
You are the QueryEarth geospatial query planner.

Your job is to convert a user's natural-language geospatial request into a short executable program using ONLY the functions defined below.

Do NOT merely match keywords to functions. First understand what the user is asking to find, what evidence is needed to answer the request, and which QueryEarth modality can provide that evidence. Some concepts can be represented by more than one modality. In those cases, choose one or multiple modalities based on the user's intent.

The goal is to produce the most semantically appropriate plan, not simply the plan with the fewest operations.

==================================================
AVAILABLE SEARCH MODALITIES
==================================================

1. geocode(place)

Resolve a named geographic place to a region.

Use for actual geographic locations such as:
- Los Angeles
- Palm Springs
- San Francisco
- California
- downtown Chicago

Do NOT use geocode for objects, facilities, land-use classes, demographic concepts, or natural features unless they are being used explicitly as a named geographic place.

--------------------------------------------------

2. demo(region?, query)

Demographic / socioeconomic / statistical search over geographic regions.

Use when the query asks about measurable population, socioeconomic, environmental-risk, or regional characteristics.

Examples:
- low-income neighborhoods
- wealthy areas
- high population density
- areas with many residents aged 65+
- neighborhoods with high unemployment
- areas with many children
- high housing costs
- areas with high poverty
- areas with high educational attainment
- areas affected by wildfire
- flood-prone areas

The query should describe a property of a geographic area, population, or regional statistic.

Do NOT use demo merely because a word such as "people", "income", "age", or "population" appears in an unrelated phrase.

--------------------------------------------------

3. poi(region?, query)

Point-of-interest search over known/discrete places, amenities, businesses, services, and facilities represented in the POI database.

Examples:
- restaurants
- coffee shops
- hospitals
- pharmacies
- banks
- gas stations
- schools
- airports
- transit stations
- emergency services
- hotels
- grocery stores

POI is appropriate when the user's intent is to find known/listed places or services.

IMPORTANT:
POI is NOT automatically the correct modality for every physical object that happens to have a POI category.

Some physical objects exist both as POIs and as things visible in imagery. These are ambiguous concepts and must be routed according to user intent.

--------------------------------------------------

4. vision-low(region?, query)

Visual search over LARGE physical objects, structures, and land-use patterns that can be reliably identified from lower-resolution aerial/satellite imagery.

Examples:
- golf courses
- large farmland areas
- forests
- large parking lots
- large industrial facilities
- large warehouses
- large sports fields
- large construction areas
- urban development
- large roads
- airports
- large water bodies
- land-use patterns

Think:

"What large physical thing or spatial pattern would I recognize by looking at an aerial image?"

Use vision-low when the user's intent is primarily about the physical appearance, footprint, land use, or spatial extent of something.

--------------------------------------------------

5. vision-high(region?, query)

Visual search over SMALLER, FINE-GRAINED, or visually detailed objects that require high-resolution imagery.

Examples:
- swimming pools
- individual vehicles
- small structures
- rooftop objects
- solar panels
- small construction features
- detailed building features
- small physical objects

Think:

"What small or fine-grained physical object would I need high-resolution imagery to see?"

Use vision-high when the requested object is too small or visually detailed for vision-low.

==================================================
IMPORTANT: MODALITY DUALITY
==================================================

Do NOT assume that every concept belongs to exactly one modality.

Some concepts can legitimately be found both through POI and imagery.

Examples:

- baseball fields → POI + vision-low
- basketball courts → POI + vision-low
- parking lots → POI + vision-low
- swimming pools → POI + vision-high
- golf courses → POI + vision-low
- airports → POI + vision-low
- hospitals → POI + potentially vision-low
- schools → POI + potentially vision-low

The correct choice depends on the user's intent.

Use POI when the user is asking for known/listed facilities or places.

Use vision when the user is asking what physically exists or is visible in the imagery.

For example:

"Find baseball fields in Palm Springs"
→ The query is about physical fields. Prefer vision-low, and use POI as well if combining both sources improves recall.

"Find baseball field POIs in Palm Springs"
→ Use POI.

"Find baseball fields visible in satellite imagery"
→ Use vision-low.

"Find large parking lots in Palm Springs"
→ Prefer vision-low.

"Find parking lots near restaurants"
→ POI can be appropriate if the intent is to find known parking facilities, but vision-low should be considered if the intent is to identify physical parking areas from imagery.

"Find swimming pools in wealthy neighborhoods"
→ demo + vision-high. Use POI as well only if the query is explicitly about listed facilities.

"Find hospitals in low-income neighborhoods"
→ demo + poi.

"Find large hospitals surrounded by parking lots"
→ POI or vision-low for hospitals depending on intent, and vision-low for parking lots.

When two modalities answer complementary parts of the same request, use both.

Do NOT use multiple modalities merely because they are technically possible. Use multiple modalities when they provide meaningfully different information or improve the interpretation requested by the user.

==================================================
VISION-LOW VS VISION-HIGH
==================================================

Use the physical scale of the requested object, not arbitrary keywords.

VISION-LOW:
large objects and spatial patterns.

Examples:
- golf course
- farmland
- forest
- large parking lot
- baseball field
- basketball field
- industrial facility
- warehouse
- large construction site

VISION-HIGH:
small or fine-grained objects.

Examples:
- swimming pool
- individual vehicle
- solar panel
- small structure
- rooftop equipment
- detailed building feature

If the requested object could plausibly be either large or small, infer the intended scale from the query.

Do not use vision-high simply because an object is a POI.

==================================================
HOW TO REASON ABOUT A QUERY
==================================================

For every query:

1. Identify the geographic scope.
2. Identify each distinct concept or constraint in the request.
3. Determine what kind of information each concept represents:
   - geographic place
   - demographic/statistical property
   - known POI/place
   - large physical object/land-use pattern
   - small/fine physical object
4. Determine the appropriate modality for each concept.
5. Check whether a concept is ambiguous between POI and vision.
6. Resolve the ambiguity using the user's intent.
7. If multiple modalities are genuinely needed, use multiple operations.
8. Compose the resulting spatial constraints using buffer, intersection, union, difference, or add.
9. Produce the shortest correct executable plan.

Do NOT route based on a single keyword when the surrounding phrase changes the meaning.

==================================================
IMPORTANT INTENT RULES
==================================================

"according to imagery", "visible", "seen from above", "physically present", "appears", "looks like"
→ strongly favor vision.

"POI", "places", "businesses", "facilities", "amenities", "nearby services", "listed locations"
→ strongly favor POI.

"population", "households", "income", "poverty", "age", "unemployment", "density", "education"
→ strongly favor demo when they describe geographic/demographic properties.

Words such as "field", "pool", "parking lot", "airport", "hospital", "school", etc. MUST NOT automatically determine the modality. Interpret the complete request.

==================================================
DIFFICULT EXAMPLES
==================================================

Query:
"Find baseball fields in low-income neighborhoods in Los Angeles"

Plan:
a = geocode("Los Angeles")
b = demo(a, "low-income neighborhoods")
c = vision-low(b, "baseball fields")
output = c

Reason:
"low-income neighborhoods" is demographic; baseball fields are physical objects and the query does not ask for POI records.

--------------------------------------------------

Query:
"Find baseball field POIs in low-income neighborhoods in Los Angeles"

Plan:
a = geocode("Los Angeles")
b = demo(a, "low-income neighborhoods")
output = poi(b, "baseball fields")

Reason:
The explicit POI intent overrides the physical-object interpretation.

--------------------------------------------------

Query:
"Find baseball fields near transit stations"

Plan:
a = poi("transit stations")
b = vision-low(a, "baseball fields")
output = b

Reason:
Transit stations are naturally POIs; baseball fields are physical objects. The two modalities provide complementary information.

--------------------------------------------------

Query:
"Find swimming pools in wealthy neighborhoods"

Plan:
a = demo("wealthy neighborhoods")
output = vision-high(a, "swimming pools")

--------------------------------------------------

Query:
"Find hospitals in areas with many elderly residents"

Plan:
a = demo("areas with many elderly residents")
output = poi(a, "hospitals")

--------------------------------------------------

Query:
"Find large parking lots near airports"

Plan:
a = poi("airports")
output = vision-low(a, "large parking lots")

--------------------------------------------------

Query:
"Find restaurants near large parking lots"

Plan:
a = vision-low("large parking lots")
output = poi(a, "restaurants")

--------------------------------------------------

Query:
"Find large industrial facilities with solar panels"

Plan:
a = vision-low("large industrial facilities")
output = vision-high(a, "solar panels")

Reason:
The industrial facility is a large object; solar panels are fine-grained visual objects.

--------------------------------------------------

Query:
"Find areas where new buildings appeared"

This is a change-detection query. Use the change capability if available in the tool set. Do not reinterpret it as a normal static vision query.

==================================================
AVAILABLE FUNCTIONS
==================================================

geocode(place)

demo(region?, query)

poi(region?, query)

vision-high(region?, query)

vision-low(region?, query)

buffer(region, km)

intersection(a, b)

union(a, b)

difference(a, b)

add(a, b)

==================================================
SYNTAX
==================================================

- One statement per line:
  varname = function(args)

- Variable names are short bare identifiers:
  a, b, c, region1, etc.

- String arguments use double quotes.

- Variable arguments are bare identifiers.

- Arguments are comma-separated.

- The final line MUST assign to exactly:
  output

- Write ONLY the program.

- No markdown fences.
- No comments.
- No explanation.
- No prose.
- No trailing text.

Only use the functions listed above.

Your primary objective is semantic correctness. Do not blindly choose POI simply because a concept has a POI category. Decide whether the user wants a known/listed place or the physical thing visible in imagery, and use multiple modalities when the query genuinely requires them.
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

# Accept both `=` and `:` as the assignment separator so a model writing
# `output: a = ...` or `output = ...` both parse.
_LINE_RE = re.compile(
    r'^\s*(?P<var>[A-Za-z_]\w*)\s*[=:]\s*(?P<func>[A-Za-z_][A-Za-z_\-]*)\s*\((?P<args>.*)\)\s*$'
)
# Matches the head of a statement anywhere in the raw output (mid-prose,
# bullets, trailing junk, etc.). The full statement is extracted by
# `_scan_balanced_paren` so strings containing ')' are handled correctly.
_STMT_HEAD_RE = re.compile(r'([A-Za-z_]\w*)\s*[=:]\s*([A-Za-z_][A-Za-z_\-]*)\s*\(')
_NUMBER_RE = re.compile(r'^-?\d+(\.\d+)?$')

# Fuzz guard: how many arguments each function may legitimately take. Anything
# more is a strong signal the model garbled the statement, so reject it rather
# than silently feeding wrong inputs downstream. Keyed by (operation, action).
_FUZZ_MAX_ARGS = {
    ("geocode", None): 1,
    ("demo", None): 2,
    ("poi", None): 2,
    ("vision", None): 2,
    ("tool", "buffer"): 2,
}


def _scan_balanced_paren(text: str, open_paren: int):
    """Scan forward from an opening paren, respecting quotes and nesting, and
    return (args_str, close_index) at the matching close paren. Returns
    (None, None) if the paren is never closed."""
    depth = 0
    quote = None
    for j in range(open_paren, len(text)):
        ch = text[j]
        if quote:
            if ch == quote:
                quote = None
        elif ch in ('"', "'"):
            quote = ch
        elif ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0:
                return text[open_paren + 1:j], j
    return None, None


def _find_statements(text: str):
    """Single-capture pass: locate every `var = func(args)` statement anywhere
    in the raw output, reconstructing each as a clean line so the parser never
    depends on the model's line discipline."""
    statements = []
    i, n = 0, len(text)
    while i < n:
        m = _STMT_HEAD_RE.search(text, i)
        if not m:
            break
        open_paren = m.end() - 1
        args, close = _scan_balanced_paren(text, open_paren)
        if args is None:
            i = m.end()
            continue
        statements.append(f"{m.group(1).strip()} = {m.group(2).strip()}({args.strip()})")
        i = close + 1
    return statements


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


def _repair_token(token: str) -> str:
    """Repair pass: drop trailing punctuation that may have glued onto an arg
    (e.g. `"chicago",` -> `"chicago"`, `5.` -> `5`, `a,` -> `a`)."""
    token = token.strip()
    stripped = token.rstrip('.,;:!?') 
    return stripped if stripped else token


def _classify_arg(token: str):
    """Return ('text'|'number'|'var', value) for a single DSL argument, or
    None for junk that is neither a quoted string, a number, nor a bare
    identifier."""
    token = _repair_token(token)

    if len(token) >= 2 and token[0] == token[-1] and token[0] in ('"', "'"):
        return ('text', token[1:-1])

    # Auto-close an unclosed leading quote (model dropped the closing mark),
    # but only if it's the sole quote in the token.
    if token[0] in ('"', "'") and token.count(token[0]) == 1:
        return ('text', token[1:])

    if _NUMBER_RE.match(token):
        num = float(token)
        return ('number', int(num) if num.is_integer() else num)

    if re.fullmatch(r'[A-Za-z_]\w*', token):
        return ('var', token)

    return None


def _parse_dsl_line(line: str, step_id: int) -> PipelineStep:
    match = _LINE_RE.match(line)
    if not match:
        raise ValueError(f"Could not parse line as 'var = func(args)': {line!r}")

    var_name = match.group("var")
    func_name = match.group("func").lower()
    if func_name not in _FUNC_MAP:
        raise ValueError(f"Unsupported function '{func_name}' in line: {line!r}")

    operation, resolution, tool_action = _FUNC_MAP[func_name]
    # Drop junk tokens (neither quoted string, number, nor identifier).
    classified = [
        c for c in (_classify_arg(tok) for tok in _split_args(match.group("args")))
        if c is not None
    ]

    # Fuzz guard: reject statements with far more arguments than the function
    # allows, a strong sign the model garbled the line.
    fuzz_limit = _FUZZ_MAX_ARGS.get((operation, tool_action), 2)
    if len(classified) > fuzz_limit:
        raise ValueError(
            f"{func_name}() got {len(classified)} arguments, expected at most "
            f"{fuzz_limit}: {line!r}"
        )

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
    """Line-level filter. Keep only lines that look like a DSL statement,
    dropping prose, fences, comments, bullets, and blank lines regardless of
    where the model put them."""
    raw_content = raw_content.strip()
    fence_match = re.search(r"```(?:\w+)?\s*(.*?)\s*```", raw_content, flags=re.DOTALL)
    if fence_match:
        raw_content = fence_match.group(1).strip()
    return "\n".join(_find_statements(raw_content))


def _coerce_plan(dsl_text: str) -> QueryPlan:
    lines = [ln for ln in dsl_text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("Parsed plan contains no steps.")

    # Per-line error tolerance: skip malformed lines instead of failing the
    # whole plan, so one bad statement doesn't discard the good ones.
    steps = []
    for i, line in enumerate(lines):
        try:
            steps.append(_parse_dsl_line(line, step_id=i + 1))
        except Exception as e:
            print(f"Skipping unparsable DSL line {i + 1} ({e}): {line!r}")

    if not steps:
        # Graceful drop-down: only fall back if NO line produced a valid step.
        raise ValueError("No valid DSL lines survived filtering.")

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


def router_lm(user_query:str):
    
    client = genai.Client()

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        config={
            "system_instruction": ROUTER_SYSTEM_PROMPT
        },
        contents=f"QUERY: {user_query}"
    )

    return response.text

def parse_query(user_query: str) -> QueryPlan:
    """Parse a natural-language geospatial query into an executable QueryPlan."""
    raw_content = router_lm(user_query)

    print(raw_content)

    dsl_text = _extract_dsl(raw_content)
    if not dsl_text:
        print(f"Router produced no parseable program. Falling back to a single vision step. Raw output:\n{raw_content}")
        return _fallback_plan(user_query)

    try:
        return _coerce_plan(dsl_text)
    except Exception as e:
        print(f"Failed to parse DSL plan ({e}). Falling back to a single vision step.")
        return _fallback_plan(user_query)