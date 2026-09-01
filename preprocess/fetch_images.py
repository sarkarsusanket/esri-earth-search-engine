"""
Fetch ArcGIS Wayback imagery tiles for a large set of lat/lon centroids and
store them as parquet, efficiently, at 10M-point scale.

Key design choices (why this isn't just requests.get() in a loop):

1. DEDUPE TILES, NOT POINTS.
   At zoom 17 a tile covers roughly ~300x300m (varies with latitude) --
   close to a 200m grid spacing, so many centroids will land on the SAME
   tile. We compute (x, y) tile indices for every point, then fetch each
   UNIQUE tile exactly once. This alone can cut network/storage by a large
   factor at your scale. We still give you a full centroid->tile mapping
   so nothing is lost.

2. ASYNC + BOUNDED CONCURRENCY (aiohttp), not synchronous requests.
   Sequential requests.get() for 10M tiles is not viable. We use an
   asyncio.Semaphore to run many requests concurrently while keeping a
   ceiling that's polite to the tile server.

3. STREAMED / BATCHED PARQUET WRITES.
   Image bytes for millions of tiles won't fit in memory. We fetch in
   batches (e.g. 5,000 tiles), write each batch as a row-group with
   pyarrow.parquet.ParquetWriter, and drop the batch from memory.

4. RESUMABLE.
   Completed tile keys are appended to a checkpoint file as we go, so if
   the job dies partway through (network blip, OOM, rate limit ban) you
   can restart and it skips everything already fetched.

5. RETRIES WITH BACKOFF + graceful handling of missing tiles (404 at
   high zoom over water/edge-of-coverage is normal, not an error).

Usage:
    python fetch_wayback_tiles.py \
        --npy centroids_lonlat.npy \
        --release-id 45134 \
        --zoom 17 \
        --out tiles_2022.parquet \
        --mapping-out centroid_tile_map_2022.parquet \
        --concurrency 64 \
        --batch-size 5000
"""

import argparse
import asyncio
import math
import os
import time
from dataclasses import dataclass

import aiohttp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# --------------------------------------------------------------------------
# Tile math (vectorized version of the function you had, for speed on 10M pts)
# --------------------------------------------------------------------------
def latlon_to_tile_vec(lat: np.ndarray, lon: np.ndarray, zoom: int):
    """Vectorized lat/lon -> XYZ tile index. Same math as the slippy-map
    formula you used, just numpy'd so 10M points takes ms instead of minutes."""
    lat_rad = np.radians(lat)
    n = 2.0 ** zoom
    xtile = ((lon + 180.0) / 360.0 * n).astype(np.int64)
    ytile = ((1.0 - np.arcsinh(np.tan(lat_rad)) / np.pi) / 2.0 * n).astype(np.int64)
    return xtile, ytile


# --------------------------------------------------------------------------
# Fetch worker
# --------------------------------------------------------------------------
TILE_URL_TMPL = (
    "https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/"
    "WMTS/1.0.0/default028mm/MapServer/tile/{release_id}/{zoom}/{y}/{x}"
)


@dataclass
class FetchResult:
    x: int
    y: int
    file_name: str
    content: bytes | None   # None if tile genuinely doesn't exist (404)


async def fetch_one(session, sem, release_id, zoom, x, y, max_retries=4, timeout=15):
    url = TILE_URL_TMPL.format(release_id=release_id, zoom=zoom, y=y, x=x)
    file_name = f"{release_id}_{x}_{y}"

    async with sem:
        backoff = 1.0
        for attempt in range(max_retries):
            try:
                async with session.get(url, timeout=timeout) as resp:
                    if resp.status == 200:
                        content = await resp.read()
                        return FetchResult(x, y, file_name, content)
                    elif resp.status == 404:
                        # No imagery at this tile for this release -- not an error
                        return FetchResult(x, y, file_name, None)
                    elif resp.status in (429, 503):
                        # Rate limited / server busy -- back off and retry
                        await asyncio.sleep(backoff)
                        backoff *= 2
                        continue
                    else:
                        await asyncio.sleep(backoff)
                        backoff *= 2
                        continue
            except (aiohttp.ClientError, asyncio.TimeoutError):
                await asyncio.sleep(backoff)
                backoff *= 2
                continue
        # Exhausted retries
        return FetchResult(x, y, file_name, None)


async def fetch_batch(session, sem, release_id, zoom, xy_batch):
    tasks = [
        fetch_one(session, sem, release_id, zoom, int(x), int(y))
        for x, y in xy_batch
    ]
    return await asyncio.gather(*tasks)


# --------------------------------------------------------------------------
# Main driver
# --------------------------------------------------------------------------
def tile_center_lonlat(x: int, y: int, zoom: int):
    """Center lon/lat of a tile, for the 'lat'/'lon' columns in the output
    (representative location of the deduped tile, not any single input point)."""
    n = 2.0 ** zoom
    lon = (x + 0.5) / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * (y + 0.5) / n)))
    lat = math.degrees(lat_rad)
    return lon, lat


def load_checkpoint(ckpt_path):
    if os.path.exists(ckpt_path):
        with open(ckpt_path, "r") as f:
            return set(line.strip() for line in f if line.strip())
    return set()


async def run(args):
    coords = np.load(args.npy)   # shape (N, 2): [lon, lat]
    lons, lats = coords[:, 0], coords[:, 1]
    print(f"Loaded {len(lons):,} centroids")

    xs, ys = latlon_to_tile_vec(lats, lons, args.zoom)

    # ---- dedupe tiles ----
    xy_pairs = np.column_stack([xs, ys])
    unique_xy, inverse = np.unique(xy_pairs, axis=0, return_inverse=True)
    print(f"{len(unique_xy):,} unique tiles out of {len(xy_pairs):,} centroids "
          f"({100 * (1 - len(unique_xy) / len(xy_pairs)):.1f}% dedup savings)")

    # ---- write centroid -> tile filename mapping (cheap, no image bytes) ----
    file_names_all = np.array(
        [f"{args.release_id}_{x}_{y}" for x, y in unique_xy]
    )[inverse]
    mapping_table = pa.table({
        "lat": lats,
        "lon": lons,
        "file_name": file_names_all,
    })
    pq.write_table(mapping_table, args.mapping_out)
    print(f"Wrote centroid->tile mapping -> {args.mapping_out}")

    # ---- resume support ----
    ckpt_path = args.out + ".ckpt"
    done = load_checkpoint(ckpt_path)
    if done:
        print(f"Resuming: {len(done):,} tiles already fetched, skipping them")

    todo = [(x, y) for x, y in unique_xy if f"{args.release_id}_{x}_{y}" not in done]
    print(f"{len(todo):,} tiles left to fetch")

    schema = pa.schema([
        ("lat", pa.float64()),
        ("lon", pa.float64()),
        ("file_name", pa.string()),
        ("image_bytes", pa.binary()),
    ])

    # append mode if resuming an existing parquet, else fresh writer
    write_mode_new = not os.path.exists(args.out)
    writer = pq.ParquetWriter(args.out, schema) if write_mode_new else None
    if writer is None:
        # pyarrow ParquetWriter can't append to an existing file directly;
        # write resumed batches to a temp part file and merge at the end.
        part_path = args.out + ".resume_part"
        writer = pq.ParquetWriter(part_path, schema)
    else:
        part_path = None

    connector = aiohttp.TCPConnector(limit=args.concurrency, ttl_dns_cache=300)
    sem = asyncio.Semaphore(args.concurrency)

    n_fetched = 0
    n_missing = 0
    t0 = time.time()

    async with aiohttp.ClientSession(connector=connector) as session:
        for i in range(0, len(todo), args.batch_size):
            batch = todo[i:i + args.batch_size]
            results = await fetch_batch(session, sem, args.release_id, args.zoom, batch)

            lat_col, lon_col, name_col, bytes_col = [], [], [], []
            with open(ckpt_path, "a") as ckpt_f:
                for r in results:
                    if r.content is None:
                        n_missing += 1
                        # still checkpoint it so we don't retry a confirmed-missing tile
                        ckpt_f.write(r.file_name + "\n")
                        continue
                    lon, lat = tile_center_lonlat(r.x, r.y, args.zoom)
                    lat_col.append(lat)
                    lon_col.append(lon)
                    name_col.append(r.file_name)
                    bytes_col.append(r.content)
                    ckpt_f.write(r.file_name + "\n")
                    n_fetched += 1

            if name_col:
                batch_table = pa.table({
                    "lat": lat_col,
                    "lon": lon_col,
                    "file_name": name_col,
                    "image_bytes": bytes_col,
                }, schema=schema)
                writer.write_table(batch_table)

            elapsed = time.time() - t0
            rate = n_fetched / elapsed if elapsed > 0 else 0
            print(f"  [{i + len(batch):,}/{len(todo):,}] fetched={n_fetched:,} "
                  f"missing={n_missing:,} rate={rate:.1f} tiles/s", end="\r")

    writer.close()
    print()

    if part_path is not None:
        # Merge resumed part into the main parquet file
        _merge_parquet(args.out, part_path)

    print(f"Done. {n_fetched:,} tiles saved, {n_missing:,} missing/404 -> {args.out}")


def _merge_parquet(main_path, part_path):
    """Combine a resumed batch of newly-fetched rows into the existing parquet."""
    main_table = pq.read_table(main_path)
    part_table = pq.read_table(part_path)
    combined = pa.concat_tables([main_table, part_table])
    tmp_out = main_path + ".merged"
    pq.write_table(combined, tmp_out)
    os.replace(tmp_out, main_path)
    os.remove(part_path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--npy", required=True, help="npy of centroids, shape (N,2) = [lon, lat]")
    p.add_argument("--release-id", required=True, help="Wayback release id, e.g. 45134")
    p.add_argument("--zoom", type=int, default=17)
    p.add_argument("--out", required=True, help="output parquet path for tile images")
    p.add_argument("--mapping-out", required=True,
                    help="output parquet mapping every centroid to its tile file_name")
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=5000)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run(args)) 