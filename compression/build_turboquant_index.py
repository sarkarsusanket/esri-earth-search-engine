"""
Build a persisted TurboQuant index (rotate -> IVF coarse cluster -> bit-packed
scalar quantizer + 1-bit residual correction) PLUS a spatial (haversine BallTree)
index, from a Parquet file containing location embeddings + lat/lon coordinates.

Usage:
    python build_turboquant_index.py \
        --parquet_path embeddings.parquet \
        --output_dir ./tq_index \
        --num_bits 1 \
        --nlist 8000 \
        --lat_col lat \
        --lon_col lon \
        --emb_col embedding
"""

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_turboquant_index")


# ==============================================================================
# BIT PACKING — row-aligned
# ==============================================================================
def pack_bits_rows(values: torch.Tensor, num_bits: int) -> np.ndarray:
    arr = values.detach().cpu().numpy().astype(np.uint32)
    n, d = arr.shape
    bits = np.zeros((n, d, num_bits), dtype=np.uint8)
    for i in range(num_bits):
        bits[:, :, num_bits - 1 - i] = (arr >> i) & 1
    bits = bits.reshape(n, d * num_bits)
    pad = (-bits.shape[1]) % 8
    if pad:
        bits = np.pad(bits, ((0, 0), (0, pad)))
    return np.packbits(bits, axis=1)  # (N, bytes_per_row)


def _kmeans(x: torch.Tensor, k: int, iters: int = 15, sample_size: int = 200_000, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    n = x.shape[0]
    train_x = x[torch.randperm(n, generator=gen)[:min(sample_size, n)]] if n > sample_size else x
    centroids = train_x[torch.randperm(train_x.shape[0], generator=gen)[:k]].clone()
    for _ in range(iters):
        assign = torch.empty(train_x.shape[0], dtype=torch.int64)
        chunk = 20_000
        for s in range(0, train_x.shape[0], chunk):
            assign[s:s + chunk] = torch.cdist(train_x[s:s + chunk], centroids).argmin(dim=1)
        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(k)
        new_centroids.index_add_(0, assign, train_x)
        counts.index_add_(0, assign, torch.ones(train_x.shape[0]))
        empty = counts == 0
        counts = counts.clamp(min=1).unsqueeze(1)
        new_centroids = new_centroids / counts
        new_centroids[empty] = centroids[empty]
        centroids = new_centroids
    return centroids


def _read_parquet_columns(parquet_path: str, columns: list):
    """
    Zero-copy efficient arrow dataset loader for specific columns.
    Converts embedding column (List / FixedSizeList) to a contiguous 2D float32 array.
    """
    dataset = ds.dataset(parquet_path, format="parquet")
    table = dataset.to_table(columns=columns)
    
    col_data = {}
    for col in columns:
        arr = table[col]
        # Check if the column is a list/nested array (embeddings)
        if hasattr(arr.type, "value_type"):
            # Flatten PyArrow list column into contiguous (N, D) numpy array
            flat = arr.to_numpy(zero_copy_only=False)
            col_data[col] = np.vstack(flat).astype(np.float32)
        else:
            col_data[col] = arr.to_numpy().astype(np.float32)
            
    return col_data


def build_index(
    parquet_path: str,
    output_dir: str,
    lat_col: str = "lat",
    lon_col: str = "lon",
    emb_col: str = "emb",
    num_bits: int = 1,
    nlist: int = None,
    clip_percentile: float = 99.9,
    seed: int = 0,
    chunk_size: int = 200_000,
    sample_size: int = 200_000,
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Loading Parquet data from %s ...", parquet_path)
    data = _read_parquet_columns(parquet_path, [emb_col, lat_col, lon_col])
    
    embeddings = data[emb_col]
    lat = data[lat_col]
    lon = data[lon_col]
    del data  # Free wrapper dictionary

    num_vectors, dim = embeddings.shape
    assert lat.shape[0] == num_vectors and lon.shape[0] == num_vectors, "lat/lon count must match embeddings"
    logger.info("Loaded %d embeddings, dim=%d", num_vectors, dim)

    if nlist is None:
        nlist = int(max(1, round(num_vectors ** 0.5)))
    nlist = max(1, min(nlist, num_vectors))
    logger.info("Using nlist=%d, chunk_size=%d", nlist, chunk_size)

    # --- 1. Rotation matrix ---
    gen = torch.Generator().manual_seed(seed)
    rotation_matrix, _ = torch.linalg.qr(torch.randn(dim, dim, generator=gen))

    # --- 2. Train IVF centroids + estimate quantizer thresholds on a SAMPLE ---
    logger.info("Sampling %d rows for k-means + threshold estimation...", min(sample_size, num_vectors))
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(num_vectors, size=min(sample_size, num_vectors), replace=False)
    sample_idx.sort()
    
    sample = torch.from_numpy(np.ascontiguousarray(embeddings[sample_idx]))
    rotated_sample = sample @ rotation_matrix
    del sample

    logger.info("Running k-means for IVF coarse quantizer...")
    centroids = _kmeans(rotated_sample, nlist, seed=seed)

    clip_val = float(np.quantile(rotated_sample.abs().flatten().numpy(), clip_percentile / 100.0))
    clip_val = max(clip_val, 1e-8)
    levels = 2 ** num_bits - 1
    scale = (2 * clip_val) / levels

    clamped_sample = torch.clamp(rotated_sample, -clip_val, clip_val)
    codes_sample = torch.round((clamped_sample + clip_val) / scale)
    dequant_sample = codes_sample * scale - clip_val
    residual_sample = rotated_sample - dequant_sample
    rs_max = float(np.quantile(residual_sample.abs().mean(dim=1).numpy(), 0.999))
    rs_max = max(rs_max, 1e-8)
    del rotated_sample, clamped_sample, codes_sample, dequant_sample, residual_sample

    # --- 3. Single sequential streaming pass ---
    assign_parts, codes_parts, signs_parts, rsq_parts = [], [], [], []
    n_chunks = (num_vectors + chunk_size - 1) // chunk_size
    
    for ci, start in enumerate(range(0, num_vectors, chunk_size)):
        end = min(start + chunk_size, num_vectors)
        chunk = torch.from_numpy(np.ascontiguousarray(embeddings[start:end]))
        rotated_chunk = chunk @ rotation_matrix
        del chunk

        assign_chunk = torch.cdist(rotated_chunk, centroids).argmin(dim=1)

        clamped = torch.clamp(rotated_chunk, -clip_val, clip_val)
        codes_chunk = torch.round((clamped + clip_val) / scale).to(torch.int64)
        dequant = codes_chunk.float() * scale - clip_val
        residual = rotated_chunk - dequant
        residual_scale_chunk = residual.abs().mean(dim=1)
        residual_sign_chunk = (residual >= 0).to(torch.int64)
        del rotated_chunk, clamped, dequant, residual

        rsq_chunk = torch.round(residual_scale_chunk / rs_max * 255).clamp(0, 255).to(torch.uint8)

        assign_parts.append(assign_chunk)
        codes_parts.append(pack_bits_rows(codes_chunk, num_bits))
        signs_parts.append(pack_bits_rows(residual_sign_chunk, 1))
        rsq_parts.append(rsq_chunk.numpy())

        if ci % max(1, n_chunks // 20) == 0 or end == num_vectors:
            logger.info("  chunk %d/%d (%d/%d vectors)", ci + 1, n_chunks, end, num_vectors)

    assign = torch.cat(assign_parts)
    packed_codes = np.concatenate(codes_parts, axis=0)
    packed_signs = np.concatenate(signs_parts, axis=0)
    residual_scale_q = np.concatenate(rsq_parts, axis=0)
    del assign_parts, codes_parts, signs_parts, rsq_parts, embeddings

    # --- 4. Sort by IVF cluster for contiguous storage ---
    sort_order = torch.argsort(assign).numpy()
    row_of_id = np.empty_like(sort_order)
    row_of_id[sort_order] = np.arange(num_vectors)

    counts = torch.bincount(assign, minlength=nlist)
    cluster_offsets = torch.cat([torch.zeros(1, dtype=torch.int64), torch.cumsum(counts, dim=0)]).numpy()

    packed_codes = packed_codes[sort_order]
    packed_signs = packed_signs[sort_order]
    residual_scale_q = residual_scale_q[sort_order]
    lat_sorted = lat[sort_order]
    lon_sorted = lon[sort_order]

    # --- 5. Spatial index (haversine BallTree) ---
    logger.info("Building spatial BallTree (haversine)...")
    from sklearn.neighbors import BallTree
    coords_rad = np.radians(np.stack([lat_sorted, lon_sorted], axis=1)).astype(np.float64)
    spatial_tree = BallTree(coords_rad, metric="haversine")

    # --- 6. Write artifacts ---
    logger.info("Writing index to %s ...", out)
    np.save(out / "rotation_matrix.npy", rotation_matrix.numpy())
    np.save(out / "centroids.npy", centroids.numpy())
    np.save(out / "cluster_offsets.npy", cluster_offsets)
    np.save(out / "packed_codes.npy", packed_codes)
    np.save(out / "packed_signs.npy", packed_signs)
    np.save(out / "residual_scale_q.npy", residual_scale_q)
    np.save(out / "sorted_to_original.npy", sort_order.astype(np.int32))
    np.save(out / "row_of_id.npy", row_of_id.astype(np.int32))
    np.save(out / "lat.npy", lat_sorted)
    np.save(out / "lon.npy", lon_sorted)
    
    with open(out / "spatial_tree.pkl", "wb") as f:
        pickle.dump(spatial_tree, f)

    meta = {
        "num_vectors": num_vectors,
        "dim": dim,
        "num_bits": num_bits,
        "clip_val": clip_val,
        "scale": scale,
        "residual_scale_max": rs_max,
        "nlist": nlist,
        "clip_percentile": clip_percentile,
        "seed": seed,
        "chunk_size": chunk_size,
        "sample_size": sample_size,
    }
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    original_bytes = num_vectors * dim * 4
    compressed_bytes = packed_codes.nbytes + packed_signs.nbytes + residual_scale_q.nbytes
    logger.info(
        "Done. %d vectors, dim=%d -> %.2fx compression (%.1f bytes/vector, excl. spatial index/meta)",
        num_vectors, dim, original_bytes / compressed_bytes, compressed_bytes / num_vectors,
    )


def parse_args():
    p = argparse.ArgumentParser(description="Build a TurboQuant + IVF + spatial index folder from a Parquet file")
    p.add_argument("--parquet_path", type=str, required=True, help="Path to input .parquet file")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--lat_col", type=str, default="lat", help="Latitude column name")
    p.add_argument("--lon_col", type=str, default="lon", help="Longitude column name")
    p.add_argument("--emb_col", type=str, default="emb", help="Embedding column name")
    p.add_argument("--num_bits", type=int, default=2, help="Stage-1 bits/coordinate")
    p.add_argument("--nlist", type=int, default=None, help="IVF clusters; default sqrt(N)")
    p.add_argument("--clip_percentile", type=float, default=99.9)
    p.add_argument("--chunk_size", type=int, default=200_000, help="Rows processed per streaming chunk")
    p.add_argument("--sample_size", type=int, default=200_000, help="Rows sampled for k-means + threshold fitting")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_index(
        parquet_path=args.parquet_path,
        output_dir=args.output_dir,
        lat_col=args.lat_col,
        lon_col=args.lon_col,
        emb_col=args.emb_col,
        num_bits=args.num_bits,
        nlist=args.nlist,
        clip_percentile=args.clip_percentile,
        seed=args.seed,
        chunk_size=args.chunk_size,
        sample_size=args.sample_size,
    )