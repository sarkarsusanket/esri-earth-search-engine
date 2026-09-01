"""
Extract GeoRSCLIP embeddings from fetched tiles, two ways:

  --mode raw     Embed each 200m tile as-is.
  --mode mosaic  Merge NxN adjacent tiles (default 10x10 = 2km at 200m/tile)
                 into one big image, THEN embed -- one embedding per 2km cell.

Design notes
------------
* RAW mode is a simple streaming batch job: read tiles.parquet in record
  batches (never load all 10M rows/images at once), decode JPEGs, batch
  onto the GPU, write embeddings out incrementally.

* MOSAIC mode is the trickier one. We can't just groupby in pandas --
  the image_bytes column alone can be 150-300GB, so we never want the
  whole table in memory. Instead:

    Pass 1 (cheap): read only `file_name` (no image bytes) for every row,
       parse (x, y) tile indices from it, compute block_id = (x//N, y//N),
       and record exactly which file_names belong to each block. This is
       tiny (a few hundred MB of strings/ints even at 10M rows).

    Pass 2 (streaming): scan the full parquet WITH image bytes in batches.
       For each row, drop its bytes into that tile's slot in its block's
       buffer. The instant a block has received all of its expected
       members, we mosaic it, run the CLIP embedding, write the result,
       and free that block's buffer. Memory is bounded by however many
       blocks are "in flight" (incomplete) at once -- not by total data
       size -- which is what makes this tractable at 10M-tile scale.

    Partially-covered blocks (e.g. near a polygon edge where some of the
    100 sub-tiles never existed / 404'd) are still embedded as long as
    they clear --min-coverage, with missing slots left black.

Both modes batch multiple images together before the GPU forward pass
(inference is where the real per-image work happens; JPEG decode is
comparatively cheap and could be further parallelized with a thread
pool if decode becomes the bottleneck).
"""

import argparse
import io
import re
from collections import defaultdict

import numpy as np
import torch
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from PIL import Image
from torchvision import transforms

import open_clip


FILENAME_RE = re.compile(r"^(?P<release>[^_]+)_(?P<x>-?\d+)_(?P<y>-?\d+)$")


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def _convert_to_rgb(image):
    return image.convert('RGB')
def load_model(ckpt_path: str, device: str = "cuda"):
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-H/14", pretrained="laion2b_s32b_b79k"
    )
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint, strict=False)
    model = model.to(device).eval()
    normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
        )
    preprocess = transforms.Compose([
            transforms.Resize(
                size=224,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(224),
            _convert_to_rgb,
            transforms.ToTensor(),
            normalize,
        ])
    return model, preprocess


@torch.no_grad()
def embed_images(model, preprocess, images, device="cuda", use_amp=True):
    """images: list of PIL.Image -> np.ndarray (n, dim) float32, L2-normalized."""
    if not images:
        return np.zeros((0, 0), dtype=np.float32)
    batch = torch.stack([preprocess(im) for im in images]).to(device, non_blocking=True)
    with torch.autocast(device_type="cuda", enabled=(use_amp and device == "cuda")):
        feats = model.encode_image(batch)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.float().cpu().numpy()


# --------------------------------------------------------------------------
# Common: parquet writer helper for embeddings
# --------------------------------------------------------------------------
def make_writer(out_path, dim):
    schema = pa.schema([
        ("file_name", pa.string()),
        ("lat", pa.float64()),
        ("lon", pa.float64()),
        ("embedding", pa.list_(pa.float32(), dim)),
    ])
    return pq.ParquetWriter(out_path, schema), schema


def write_batch(writer, schema, file_names, lats, lons, embs):
    table = pa.table({
        "file_name": file_names,
        "lat": lats,
        "lon": lons,
        "embedding": list(embs),   # list of 1D arrays -> pyarrow list column
    }, schema=schema)
    writer.write_table(table)


# --------------------------------------------------------------------------
# RAW mode
# --------------------------------------------------------------------------
def run_raw(args, model, preprocess, device):
    dataset = ds.dataset(args.tiles, format="parquet")
    scanner = dataset.scanner(
        columns=["lat", "lon", "file_name", "image_bytes"],
        batch_size=args.read_batch_size,
    )

    writer = None
    schema = None
    buf_imgs, buf_names, buf_lats, buf_lons = [], [], [], []
    n_done = 0

    def flush():
        nonlocal buf_imgs, buf_names, buf_lats, buf_lons, writer, schema, n_done
        if not buf_imgs:
            return
        embs = embed_images(model, preprocess, buf_imgs, device)
        if writer is None:
            writer, schema = make_writer(args.out, embs.shape[1])
        write_batch(writer, schema, buf_names, buf_lats, buf_lons, embs)
        n_done += len(buf_imgs)
        print(f"  embedded {n_done:,} tiles", end="\r")
        buf_imgs, buf_names, buf_lats, buf_lons = [], [], [], []

    for record_batch in scanner.to_batches():
        cols = record_batch.to_pydict()
        for name, lat, lon, img_bytes in zip(
            cols["file_name"], cols["lat"], cols["lon"], cols["image_bytes"]
        ):
            if img_bytes is None:
                continue
            try:
                im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception:
                continue
            buf_imgs.append(im)
            buf_names.append(name)
            buf_lats.append(lat)
            buf_lons.append(lon)
            if len(buf_imgs) >= args.infer_batch_size:
                flush()
    flush()
    if writer:
        writer.close()
    print(f"\nDone. {n_done:,} raw-tile embeddings -> {args.out}")


# --------------------------------------------------------------------------
# MOSAIC mode
# --------------------------------------------------------------------------
def parse_xy(file_name):
    m = FILENAME_RE.match(file_name)
    if not m:
        return None
    return int(m.group("x")), int(m.group("y"))


def build_block_index(tiles_path, block_n):
    """Pass 1: cheap metadata-only scan. Returns:
       block_members: {block_id: {(dx,dy): file_name}}   (expected members)
       block_latlon:  {block_id: (lat, lon)}   filled in during pass 2 (avg)
    """
    dataset = ds.dataset(tiles_path, format="parquet")
    scanner = dataset.scanner(columns=["file_name"])

    block_members = defaultdict(dict)
    for record_batch in scanner.to_batches():
        for name in record_batch.column("file_name").to_pylist():
            xy = parse_xy(name)
            if xy is None:
                continue
            x, y = xy
            bx, by = x // block_n, y // block_n
            dx, dy = x - bx * block_n, y - by * block_n
            block_members[(bx, by)][(dx, dy)] = name
    print(f"Pass 1 done: {len(block_members):,} candidate 2km blocks "
          f"({block_n}x{block_n} tiles each)")
    return block_members


def run_mosaic(args, model, preprocess, device):
    block_n = args.block_n
    tile_px = args.tile_px
    min_members = int(block_n * block_n * args.min_coverage)

    block_members = build_block_index(args.tiles, block_n)
    # remaining[block_id] = set of file_names still needed
    remaining = {bid: set(members.values()) for bid, members in block_members.items()}
    # buffers[block_id] = {(dx,dy): PIL.Image}
    buffers = defaultdict(dict)

    writer = None
    schema = None
    out_imgs, out_names, out_lats, out_lons = [], [], [], []
    n_blocks_done = 0

    def flush_embeddings():
        nonlocal out_imgs, out_names, out_lats, out_lons, writer, schema, n_blocks_done
        if not out_imgs:
            return
        embs = embed_images(model, preprocess, out_imgs, device)
        if writer is None:
            writer, schema = make_writer(args.out, embs.shape[1])
        write_batch(writer, schema, out_names, out_lats, out_lons, embs)
        n_blocks_done += len(out_imgs)
        print(f"  embedded {n_blocks_done:,} 2km blocks", end="\r")
        out_imgs, out_names, out_lats, out_lons = [], [], [], []

    def finalize_block(bid, lat_acc, lon_acc, n_acc):
        canvas = Image.new("RGB", (block_n * tile_px, block_n * tile_px), (0, 0, 0))
        for (dx, dy), im in buffers[bid].items():
            # dy=0 is smallest y (northernmost); paste top-down, left-right
            canvas.paste(im, (dx * tile_px, dy * tile_px))
        bx, by = bid
        out_imgs.append(canvas)
        out_names.append(f"block_{block_n}x{block_n}_{bx}_{by}")
        out_lats.append(lat_acc / n_acc)
        out_lons.append(lon_acc / n_acc)
        del buffers[bid]
        del remaining[bid]
        if len(out_imgs) >= args.infer_batch_size:
            flush_embeddings()

    # per-block running lat/lon average, accumulated as tiles arrive
    lat_acc = defaultdict(float)
    lon_acc = defaultdict(float)
    n_acc = defaultdict(int)

    dataset = ds.dataset(args.tiles, format="parquet")
    scanner = dataset.scanner(
        columns=["lat", "lon", "file_name", "image_bytes"],
        batch_size=args.read_batch_size,
    )

    for record_batch in scanner.to_batches():
        cols = record_batch.to_pydict()
        for name, lat, lon, img_bytes in zip(
            cols["file_name"], cols["lat"], cols["lon"], cols["image_bytes"]
        ):
            if img_bytes is None:
                continue
            xy = parse_xy(name)
            if xy is None:
                continue
            x, y = xy
            bid = (x // block_n, y // block_n)
            if bid not in remaining or name not in remaining[bid]:
                continue  # not part of any tracked block (shouldn't normally happen)
            try:
                im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception:
                remaining[bid].discard(name)
                continue

            dx, dy = x - bid[0] * block_n, y - bid[1] * block_n
            buffers[bid][(dx, dy)] = im
            lat_acc[bid] += lat
            lon_acc[bid] += lon
            n_acc[bid] += 1
            remaining[bid].discard(name)

            if not remaining[bid]:
                finalize_block(bid, lat_acc[bid], lon_acc[bid], n_acc[bid])
                del lat_acc[bid], lon_acc[bid], n_acc[bid]

    # End of scan: flush any blocks that never fully completed (edge blocks
    # with some missing/404 tiles) but still clear the coverage threshold.
    for bid in list(remaining.keys()):
        n_have = len(buffers.get(bid, {}))
        if n_have >= min_members:
            finalize_block(bid, lat_acc[bid], lon_acc[bid], n_acc[bid])
        else:
            buffers.pop(bid, None)
            lat_acc.pop(bid, None); lon_acc.pop(bid, None); n_acc.pop(bid, None)

    flush_embeddings()
    if writer:
        writer.close()
    print(f"\nDone. {n_blocks_done:,} 2km-mosaic embeddings -> {args.out}")


# --------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tiles", required=True, help="input tiles parquet (from fetch step)")
    p.add_argument("--out", required=True, help="output embeddings parquet")
    p.add_argument("--mode", choices=["raw", "mosaic"], required=True)
    p.add_argument("--ckpt", required=True, help="path to RS5M_ViT-H-14.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--infer-batch-size", type=int, default=1024,
                    help="images per GPU forward pass")
    p.add_argument("--read-batch-size", type=int, default=5120,
                    help="rows per parquet record-batch read")
    p.add_argument("--block-n", type=int, default=10,
                    help="tiles per mosaic side (10x10 tiles of 200m = 2km)")
    p.add_argument("--tile-px", type=int, default=256,
                    help="pixel size of each fetched tile")
    p.add_argument("--min-coverage", type=float, default=0.5,
                    help="min fraction of block_n^2 tiles present to still embed an edge block")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model, preprocess = load_model(args.ckpt, args.device)
    if args.mode == "raw":
        run_raw(args, model, preprocess, args.device)
    else:
        run_mosaic(args, model, preprocess, args.device)