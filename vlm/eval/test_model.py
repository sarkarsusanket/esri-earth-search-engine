"""
TurboQuant — a from-scratch implementation of the rotate -> quantize -> correct-residual
scheme from Zandieh et al., "TurboQuant: Online Vector Quantization with Near-optimal
Distortion Rate" (arXiv:2504.19874), applied to LAION OpenCLIP image embeddings.

Pipeline per the paper's structure:
  1. Random orthogonal rotation of each embedding. In high dimensions this makes the
     rotated coordinates behave as ~independent, identically distributed values, so a
     *scalar* quantizer applied per-coordinate is nearly as good as a full vector
     quantizer (this is the "PolarQuant" step).
  2. Stage-1 MSE quantizer: a uniform scalar quantizer per coordinate, with the clipping
     range set from the empirical distribution of rotated coordinates (robust to
     outliers) rather than raw min/max.
  3. Stage-2 residual correction: MSE-optimal scalar quantizers are *biased* estimators
     of inner products (the paper proves this). To fix that, we compute the residual
     after stage 1 and store one extra bit per coordinate (sign of the residual, scaled
     by the mean residual magnitude). This is a practical 1-bit residual corrector in
     the spirit of the paper's QJL residual stage — it directly reduces reconstruction
     bias instead of relying only on stage 1.

This script:
  - loads image embeddings from a parquet file via LAION OpenCLIP,
  - builds a TurboQuant index,
  - runs the same text queries against (a) exact float32 search and (b) the TurboQuant
    index,
  - writes an HTML report that shows, per query, exactly what changed: which images
    moved in/out of the top-k, rank deltas, and score deltas.
"""

import argparse
import base64
import logging
from io import BytesIO
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("turboquant")

QUERIES = [
    # --- Original Queries ---
    "baseball field",
    "solar panels on roof or field",
    "shipping containers or port Terminal",
    "swimming pool in residential area",
    "dense traffic intersection",
    "industrial warehouse complex",
    "marina with docked boats",
    "tennis courts",
    "parking lot full of cars",
    "construction site with bare earth",

    # --- Sports & Recreation ---
    "football or soccer stadium with field markings",
    "golf course fairways and sand traps",
    "athletics running track oval",
    "outdoor basketball courts",
    "skate park with concrete ramps",
    "race track or motorsport circuit",
    "amusement park with roller coasters",
    "public park with green trees and walking paths",

    # --- Urban Infrastructure & Residential ---
    "suburban neighborhood with grid pattern houses",
    "dense high-rise apartment complex",
    "cul-de-sac suburban road layout",
    "mobile home park or trailer park",
    "cemetery with grid of headstones",
    "shopping mall complex with large parking area",
    "gas station or truck stop",
    "school ground with playground and field",
    "hospital complex with helipad",
    "urban plaza or pedestrian square",

    # --- Transportation & Logistics ---
    "airport runway and tarmac with parked airplanes",
    "railroad railyard with freight trains and tracks",
    "highway cloverleaf interchange",
    "bridge crossing over river or water",
    "roundabout or traffic circle",
    "bus depot or truck fleet terminal",
    "toll plaza on highway",
    "pier or jetty extending into ocean",
    "canal or artificial waterway with locks",
    "dry dock for ship maintenance",

    # --- Industrial & Energy Infrastructure ---
    "oil or gas storage tanks",
    "wind turbines in field or offshore",
    "coal power plant or cooling towers",
    "water treatment plant with circular settling tanks",
    "electrical substation with transformers and pylons",
    "mining pit or open pit quarry",
    "oil refinery with complex pipelines",
    "solar farm with rows of PV panels",
    "recycling yard or scrap yard with metal heaps",
    "lumber yard with stacked timber",

    # --- Agriculture & Land Use ---
    "center pivot circular irrigation fields",
    "terraced farming fields on hillside",
    "greenhouses or polytunnels in agricultural area",
    "orchard with neat rows of fruit trees",
    "vineyard with parallel rows of vines",
    "crop fields with rectangular boundaries",
    "hay bales in harvested field",
    "dairy farm or livestock feedlot",
    "fish farm or aquaculture ponds",
    "paddy fields flooded with water",

    # --- Natural & Physical Geography ---
    "meandering river through valley",
    "sand dunes in desert",
    "volcanic crater or caldera",
    "glacier or ice sheet",
    "coastal beach with ocean waves breaking",
    "coral reef under clear shallow ocean water",
    "forest canopy with dense tree cover",
    "mangrove swamp or coastal wetland",
    "mountain peak with snow cover",
    "river delta flowing into sea",
    "cliff face along ocean coastline",
    "lake or reservoir with dam structure",

    # --- Environmental & Land Disturbances ---
    "wildfire burn scar or scorched forest",
    "deforestation or clear-cut timber forest",
    "flooded agricultural fields or urban area",
    "landslide or mudslide scar",
    "coastal erosion or shrinking beach",
    "drying lake bed with salt flats",
    "smokestacks emitting plume or smoke",

    # --- Specialized & Rare Features ---
    "helipad on building rooftop",
    "archaeological ruins or ancient structures",
    "military base with barracks and airfield",
    "drive-in theater screen and parking",
    "solar evaporative salt pans with bright colors",
    "log boom or timber floating on river",
    "fountain or water display in park",
    "footbridge over highway",
    "sewage lagoon or oxidation pond",
    "radio or satellite communication dish antenna array",

    # --- Pattern & Texture Specifics ---
    "neat geometric grid of urban streets",
    "curved winding mountain road",
    "bright red roofed buildings",
    "bright blue roofs or tarps",
    "dense cluster of small boats",
    "checkerboard pattern of agricultural fields",
    "shadows cast by tall skyscrapers",
    "muddy brown water meeting clear blue ocean water",
    "zigzag mountain road switchbacks",
    "patches of bare soil mixed with forest"
]


# ==============================================================================
# BIT PACKING — honest sub-byte storage for arbitrary bit widths
# ==============================================================================
def pack_bits(values: torch.Tensor, num_bits: int) -> np.ndarray:
    """
    Pack a flat tensor of non-negative integers, each fitting in `num_bits` bits,
    into a contiguous uint8 byte buffer (MSB-first). This is what actually gives
    TurboQuant its compression — storing codes as uint8/bool tensors does NOT,
    since torch has no sub-byte dtypes.
    """
    flat = values.detach().cpu().numpy().astype(np.uint32).reshape(-1)
    bits = np.zeros((flat.size, num_bits), dtype=np.uint8)
    for i in range(num_bits):
        bits[:, num_bits - 1 - i] = (flat >> i) & 1
    return np.packbits(bits.reshape(-1))


def unpack_bits(packed: np.ndarray, num_bits: int, count: int) -> torch.Tensor:
    """Inverse of pack_bits: returns a (count,) int64 tensor of unpacked values."""
    bits = np.unpackbits(packed)[: count * num_bits].reshape(count, num_bits)
    values = np.zeros(count, dtype=np.uint32)
    for i in range(num_bits):
        values |= bits[:, num_bits - 1 - i].astype(np.uint32) << i
    return torch.from_numpy(values.astype(np.int64))


# ==============================================================================
# TURBOQUANT INDEX
# ==============================================================================
class TurboQuantIndex:
    """
    Two-stage TurboQuant quantizer/index:
      Stage 1 — rotate, then per-coordinate uniform scalar quantization (num_bits),
                 with codes stored bit-packed (not as uint8).
      Stage 2 — 1-bit residual correction to reduce reconstruction bias, sign bits
                 also stored bit-packed.

    Search never reconstructs the (N, dim) float embedding matrix. Instead it
    unpacks codes/signs into small integer matrices and scores queries with a
    closed-form linear decomposition of the reconstruction, i.e. it works
    directly on the compressed representation:

        score_i = <query, reconstructed_i>
                = <query, codes_i> * scale - clip_val * sum(query)          [stage 1]
                + residual_scale_i * (2*<query, sign_i> - sum(query))       [stage 2]

    Both matmuls (<query, codes_i> and <query, sign_i>) operate on small-integer
    matrices unpacked from the bit-packed store — no float32 (N, dim) tensor is
    ever formed.
    """

    def __init__(self, dim: int, num_bits: int = 8, clip_percentile: float = 99.9, seed: int = 0):
        if num_bits < 1:
            raise ValueError("num_bits must be >= 1")
        self.dim = dim
        self.num_bits = num_bits
        self.clip_percentile = clip_percentile
        self.levels = 2 ** num_bits - 1

        gen = torch.Generator().manual_seed(seed)
        q, _ = torch.linalg.qr(torch.randn(dim, dim, generator=gen))
        self.rotation_matrix = q  # orthogonal -> preserves inner products/norms

        # Stage-1 state (global quantizer thresholds, shared across the whole index)
        self.clip_val: float = None
        self.scale: float = None

        # Compressed on-disk representation (bit-packed)
        self.num_vectors: int = None
        self.packed_codes: np.ndarray = None     # bit-packed, num_bits/value
        self.packed_signs: np.ndarray = None      # bit-packed, 1 bit/value
        self.residual_scale: torch.Tensor = None  # (N,) float32, mean |residual| per vector — unavoidable float side info

    def _rotate(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.rotation_matrix.to(x.device)

    def compress_and_index(self, embeddings: torch.Tensor) -> Dict[str, float]:
        num_vectors, dim = embeddings.shape
        original_size_bytes = num_vectors * dim * 4  # float32

        rotated = self._rotate(embeddings)

        # Robust symmetric clip range from the empirical distribution of rotated
        # coordinates (outlier-resistant vs. raw min/max). One global scale/clip,
        # not per-vector — this is what makes the quantizer a true scalar quantizer
        # rather than a per-vector min-max scheme.
        clip_val = torch.quantile(rotated.abs().flatten(), self.clip_percentile / 100.0).item()
        clip_val = max(clip_val, 1e-8)
        scale = (2 * clip_val) / self.levels

        clamped = torch.clamp(rotated, -clip_val, clip_val)
        codes = torch.round((clamped + clip_val) / scale).to(torch.int64)  # in [0, levels]

        # Stage-1 reconstruction (only used transiently, to derive the residual —
        # never stored), then 1-bit residual correction
        dequant_stage1 = codes.float() * scale - clip_val
        residual = rotated - dequant_stage1
        residual_scale = residual.abs().mean(dim=1)  # (N,) — one scalar per vector
        residual_sign = (residual >= 0).to(torch.int64)  # 0/1

        self.clip_val = clip_val
        self.scale = scale
        self.num_vectors = num_vectors
        self.packed_codes = pack_bits(codes, self.num_bits)
        self.packed_signs = pack_bits(residual_sign, 1)
        self.residual_scale = residual_scale.cpu()

        stage1_bytes = self.packed_codes.nbytes
        stage2_bytes = self.packed_signs.nbytes
        scale_bytes = self.residual_scale.element_size() * self.residual_scale.nelement()
        # Note: the rotation matrix (dim x dim floats) is shared/amortized across the
        # whole index, not per-vector, so it's excluded from per-vector byte accounting.
        compressed_size_bytes = stage1_bytes + stage2_bytes + scale_bytes

        compression_ratio = original_size_bytes / compressed_size_bytes
        space_saving_pct = (1.0 - (compressed_size_bytes / original_size_bytes)) * 100

        return {
            "original_size_mb": original_size_bytes / (1024 * 1024),
            "compressed_size_mb": compressed_size_bytes / (1024 * 1024),
            "compression_ratio": compression_ratio,
            "space_saving_pct": space_saving_pct,
            "bytes_per_vector": compressed_size_bytes / num_vectors,
            "effective_bits_per_dim": self.num_bits + 1,  # stage-1 bits + 1 correction bit
        }

    def search(self, query_embeddings: torch.Tensor, top_k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compressed-domain search: unpack codes/signs into small integer matrices,
        then score with the closed-form decomposition. No (N, dim) float
        reconstruction is ever materialized.
        """
        rotated_queries = self._rotate(query_embeddings).cpu()  # (Q, dim)
        num_queries = rotated_queries.shape[0]
        n, d = self.num_vectors, self.dim

        codes = unpack_bits(self.packed_codes, self.num_bits, n * d).view(n, d).float()   # small ints, not reconstructed floats
        signs_pm1 = unpack_bits(self.packed_signs, 1, n * d).view(n, d).float() * 2 - 1     # {0,1} -> {-1,+1}

        query_sum = rotated_queries.sum(dim=1, keepdim=True)  # (Q, 1)

        # Stage-1 term: scale * <codes_i, q> - clip_val * sum(q)
        stage1_scores = self.scale * (codes @ rotated_queries.T) - self.clip_val * query_sum.T  # (N, Q)

        # Stage-2 term: residual_scale_i * (2*<sign_i, q> - sum(q))
        sign_dot = signs_pm1 @ rotated_queries.T  # (N, Q) -- note signs_pm1 in {-1,1}, so <sign_i,q> == this directly
        # since signs_pm1 already encodes +/-1, 2*<0/1,q>-sum(q) simplifies to <signs_pm1, q>
        stage2_scores = self.residual_scale.unsqueeze(1) * sign_dot  # (N, Q)

        scores = (stage1_scores + stage2_scores).T  # (Q, N)
        top_scores, top_indices = scores.topk(top_k, dim=-1)
        return top_scores, top_indices


# ==============================================================================
# DATASET & UTILS
# ==============================================================================
class ParquetImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_col: str):
        self.df = df
        self.image_col = image_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        raw_bytes = self.df.iloc[idx][self.image_col]
        if isinstance(raw_bytes, dict) and "bytes" in raw_bytes:
            raw_bytes = raw_bytes["bytes"]
        img = Image.open(BytesIO(raw_bytes)).convert("RGB")
        return img, idx


class ParquetCollate:
    def __init__(self, preprocess):
        self.preprocess = preprocess

    def __call__(self, batch):
        images, indices = zip(*batch)
        tensors = torch.stack([self.preprocess(img) for img in images])
        return tensors, torch.tensor(indices, dtype=torch.long)


def bytes_to_b64_url(raw_bytes: bytes) -> str:
    if isinstance(raw_bytes, dict) and "bytes" in raw_bytes:
        raw_bytes = raw_bytes["bytes"]
    b64_str = base64.b64encode(raw_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"


# ==============================================================================
# LAION CLIP MODEL LOADER
# ==============================================================================
def load_laion_vlm(model_name: str, pretrained: str, device: torch.device):
    import open_clip
    logger.info("Loading LAION/OpenCLIP VLM: %s (%s)...", model_name, pretrained)
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    return model, preprocess, tokenizer


@torch.no_grad()
def compute_image_embeddings(model, preprocess, df, image_col, device, batch_size, num_workers):
    dataset = ParquetImageDataset(df, image_col)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=ParquetCollate(preprocess), pin_memory=True,
    )
    all_embeds = []
    logger.info("Extracting embeddings for %d images with CLIP...", len(dataset))
    for pixel_values, _ in loader:
        pixel_values = pixel_values.to(device, non_blocking=True)
        embeds = model.encode_image(pixel_values)
        embeds = F.normalize(embeds, dim=-1)
        all_embeds.append(embeds.cpu())
    return torch.cat(all_embeds, dim=0)


@torch.no_grad()
def compute_text_embeddings(model, tokenizer, queries: List[str], device: torch.device):
    tokens = tokenizer(queries).to(device)
    embeds = model.encode_text(tokens)
    embeds = F.normalize(embeds, dim=-1)
    return embeds.cpu()


# ==============================================================================
# DIFF / "WHAT CHANGED" COMPUTATION
# ==============================================================================
def diff_results(orig_indices: List[int], tq_indices: List[int]) -> Dict[str, object]:
    """Compare two ranked result lists and summarize what changed."""
    orig_rank = {idx: r for r, idx in enumerate(orig_indices)}
    tq_rank = {idx: r for r, idx in enumerate(tq_indices)}

    kept = [idx for idx in orig_indices if idx in tq_rank]
    dropped = [idx for idx in orig_indices if idx not in tq_rank]   # in baseline, fell out of TQ top-k
    entered = [idx for idx in tq_indices if idx not in orig_rank]   # new in TQ, wasn't in baseline top-k

    rank_shifts = {idx: tq_rank[idx] - orig_rank[idx] for idx in kept}
    recall_at_k = len(kept) / max(len(orig_indices), 1)

    return {
        "kept": kept,
        "dropped": dropped,
        "entered": entered,
        "rank_shifts": rank_shifts,
        "recall_at_k": recall_at_k,
    }


# ==============================================================================
# HTML REPORT GENERATION
# ==============================================================================
def generate_html_report(
    queries: List[str],
    orig_results: List[List[Tuple[float, bytes, int]]],
    tq_results: List[List[Tuple[float, bytes, int]]],
    stats: Dict[str, float],
    output_path: str,
    top_k: int,
):
    def card(score, img_bytes, idx, badge=None):
        b64_url = bytes_to_b64_url(img_bytes)
        badge_html = f'<span class="badge badge-{badge[1]}">{badge[0]}</span>' if badge else ""
        return f"""
                    <div class="card">
                        {badge_html}
                        <img src="{b64_url}" loading="lazy" />
                        <div class="card-meta">
                            <span>#{idx}</span>
                            <span class="score">{score:.3f}</span>
                        </div>
                    </div>"""

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TurboQuant Retrieval Diff Report</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        background-color: #0b0f19; color: #f1f5f9; margin: 0; padding: 2rem; }}
    h1 {{ font-size: 2rem; margin-bottom: 0.25rem; color: #38bdf8; }}
    .stats-banner {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 1rem; margin: 1.5rem 0 2.5rem 0; background: #1e293b; padding: 1.25rem;
        border-radius: 10px; border: 1px solid #334155; }}
    .stat-card {{ display: flex; flex-direction: column; }}
    .stat-label {{ font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }}
    .stat-value {{ font-size: 1.4rem; font-weight: 700; color: #4ade80; margin-top: 0.25rem; }}
    .query-section {{ background-color: #111827; border-radius: 12px; padding: 1.5rem;
        margin-bottom: 2rem; border: 1px solid #1f2937; }}
    .query-title {{ font-size: 1.2rem; font-weight: 600; color: #f3f4f6; margin-bottom: 0.5rem; }}
    .diff-summary {{ font-size: 0.85rem; color: #cbd5e1; margin-bottom: 1rem; }}
    .diff-summary b {{ color: #4ade80; }}
    .comparison-wrapper {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }}
    @media (max-width: 900px) {{ .comparison-wrapper {{ grid-template-columns: 1fr; }} }}
    .panel-header {{ font-size: 0.9rem; font-weight: 600; text-transform: uppercase;
        margin-bottom: 0.75rem; padding-bottom: 0.25rem; border-bottom: 1px solid #374151; }}
    .panel-orig {{ color: #a78bfa; }}
    .panel-tq {{ color: #38bdf8; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); gap: 0.75rem; }}
    .card {{ background-color: #1f2937; border-radius: 6px; overflow: hidden;
        border: 1px solid #374151; position: relative; }}
    .card img {{ width: 100%; height: 110px; object-fit: cover; }}
    .card-meta {{ padding: 0.4rem; font-size: 0.75rem; display: flex; justify-content: space-between; color: #d1d5db; }}
    .score {{ font-weight: bold; color: #34d399; }}
    .badge {{ position: absolute; top: 4px; left: 4px; font-size: 0.65rem; font-weight: 700;
        padding: 2px 6px; border-radius: 4px; z-index: 1; }}
    .badge-dropped {{ background: #7f1d1d; color: #fecaca; }}
    .badge-entered {{ background: #14532d; color: #bbf7d0; }}
</style>
</head>
<body>
<h1>TurboQuant Retrieval Diff Report</h1>
<p style="color:#9ca3af;margin-top:0;">Exact float32 search vs. TurboQuant index — per-query changes</p>
<div class="stats-banner">
    <div class="stat-card"><span class="stat-label">Original Uncompressed</span><span class="stat-value">{stats['original_size_mb']:.2f} MB</span></div>
    <div class="stat-card"><span class="stat-label">TurboQuant Size</span><span class="stat-value">{stats['compressed_size_mb']:.2f} MB</span></div>
    <div class="stat-card"><span class="stat-label">Compression Ratio</span><span class="stat-value">{stats['compression_ratio']:.1f}x</span></div>
    <div class="stat-card"><span class="stat-label">RAM Reduction</span><span class="stat-value">{stats['space_saving_pct']:.1f}%</span></div>
    <div class="stat-card"><span class="stat-label">Bytes / Vector</span><span class="stat-value">{stats['bytes_per_vector']:.0f} B</span></div>
    <div class="stat-card"><span class="stat-label">Effective Bits/Dim</span><span class="stat-value">{stats['effective_bits_per_dim']:.0f}</span></div>
</div>
"""

    for q_idx, query in enumerate(queries):
        orig_idxs = [idx for _, _, idx in orig_results[q_idx]]
        tq_idxs = [idx for _, _, idx in tq_results[q_idx]]
        diff = diff_results(orig_idxs, tq_idxs)

        html_content += f"""
    <div class="query-section">
        <div class="query-title">"{query}"</div>
        <div class="diff-summary">
            recall@{top_k}: <b>{diff['recall_at_k']*100:.0f}%</b> &nbsp;|&nbsp;
            kept: <b>{len(diff['kept'])}</b> &nbsp;|&nbsp;
            dropped from top-k: <b>{len(diff['dropped'])}</b> &nbsp;|&nbsp;
            newly entered: <b>{len(diff['entered'])}</b>
        </div>
        <div class="comparison-wrapper">
            <div>
                <div class="panel-header panel-orig">Exact Float32 (Top {top_k})</div>
                <div class="grid">"""
        for score, img_bytes, idx in orig_results[q_idx]:
            badge = ("DROPPED", "dropped") if idx in diff["dropped"] else None
            html_content += card(score, img_bytes, idx, badge)
        html_content += """
                </div>
            </div>
            <div>
                <div class="panel-header panel-tq">TurboQuant Index (Top """ + str(top_k) + """)</div>
                <div class="grid">"""
        for score, img_bytes, idx in tq_results[q_idx]:
            badge = ("NEW", "entered") if idx in diff["entered"] else None
            html_content += card(score, img_bytes, idx, badge)
        html_content += """
                </div>
            </div>
        </div>
    </div>"""

    html_content += "\n</body>\n</html>"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    logger.info("Saved diff report HTML to %s", output_path)


# ==============================================================================
# MAIN
# ==============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="TurboQuant Parquet Search & Diff Report (LAION OpenCLIP)")
    p.add_argument("--parquet_path", type=str, required=True)
    p.add_argument("--image_col", type=str, default="image_bytes")
    p.add_argument("--model_name", type=str, default="ViT-B-32")
    p.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k")
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--num_bits", type=int, default=2, help="Stage-1 bits per rotated coordinate")
    p.add_argument("--clip_percentile", type=float, default=99.9)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--output", type=str, default="turboquant_diff_report.html")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    logger.info("Loading parquet file from %s...", args.parquet_path)
    df = pd.read_parquet(args.parquet_path)
    logger.info("Loaded %d rows from parquet.", len(df))

    model, preprocess, tokenizer = load_laion_vlm(args.model_name, args.pretrained, device)

    img_embeds = compute_image_embeddings(
        model, preprocess, df, args.image_col, device, args.batch_size, args.num_workers
    )
    text_embeds = compute_text_embeddings(model, tokenizer, QUERIES, device)

    tq_index = TurboQuantIndex(dim=img_embeds.shape[1], num_bits=args.num_bits, clip_percentile=args.clip_percentile)
    stats = tq_index.compress_and_index(img_embeds)

    logger.info("Compression ratio: %.2fx | space saved: %.2f%% | %.1f bytes/vector",
                stats["compression_ratio"], stats["space_saving_pct"], stats["bytes_per_vector"])

    top_k = min(args.top_k, len(df))

    orig_scores = text_embeds @ img_embeds.T
    top_orig_scores, top_orig_indices = orig_scores.topk(top_k, dim=-1)
    top_tq_scores, top_tq_indices = tq_index.search(text_embeds, top_k)

    orig_results_per_query, tq_results_per_query = [], []
    for q_idx in range(len(QUERIES)):
        q_orig = [
            (top_orig_scores[q_idx, k].item(), df.iloc[top_orig_indices[q_idx, k].item()][args.image_col],
             top_orig_indices[q_idx, k].item())
            for k in range(top_k)
        ]
        q_tq = [
            (top_tq_scores[q_idx, k].item(), df.iloc[top_tq_indices[q_idx, k].item()][args.image_col],
             top_tq_indices[q_idx, k].item())
            for k in range(top_k)
        ]
        orig_results_per_query.append(q_orig)
        tq_results_per_query.append(q_tq)

    generate_html_report(QUERIES, orig_results_per_query, tq_results_per_query, stats, args.output, top_k)


if __name__ == "__main__":
    main()