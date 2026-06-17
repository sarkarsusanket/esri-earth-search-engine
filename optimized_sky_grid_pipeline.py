
from __future__ import annotations

import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor

import geopandas as gpd
import numpy as np
import open_clip
import pyproj
import torch
import tqdm
import xarray as xr
from PIL import Image
from shapely.geometry import box
from shapely.ops import transform
from shapely.prepared import prep

from mapminer.miners import ESRIBaseMapMiner, ESRILULCMiner


# -----------------------------
# Thread-local miners
# -----------------------------
_thread_local = threading.local()


def _get_lulc_miner() -> ESRILULCMiner:
    miner = getattr(_thread_local, "lulc_miner", None)
    if miner is None:
        miner = ESRILULCMiner()
        _thread_local.lulc_miner = miner
    return miner


def _get_basemap_miner() -> ESRIBaseMapMiner:
    miner = getattr(_thread_local, "basemap_miner", None)
    if miner is None:
        miner = ESRIBaseMapMiner()
        _thread_local.basemap_miner = miner
    return miner


# -----------------------------
# Geometry helpers
# -----------------------------
def _make_projectors(input_polygon):
    center_lat = input_polygon.centroid.y
    center_lon = input_polygon.centroid.x

    aeqd_proj = pyproj.Proj(proj="aeqd", lat_0=center_lat, lon_0=center_lon, datum="WGS84")
    to_m = pyproj.Transformer.from_crs("EPSG:4326", aeqd_proj.crs, always_xy=True)
    to_deg = pyproj.Transformer.from_crs(aeqd_proj.crs, "EPSG:4326", always_xy=True)
    return to_m, to_deg


def _slice_bounds_from_geom(geom, y_descending: bool):
    minx, miny, maxx, maxy = geom.bounds
    if y_descending:
        return slice(maxy, miny)
    return slice(miny, maxy)


def _build_target_mask_da(lulc_xr: xr.DataArray, target_classes: np.ndarray) -> xr.DataArray:
    mask = np.isin(lulc_xr.values, target_classes)
    return xr.DataArray(mask, coords=lulc_xr.coords, dims=lulc_xr.dims)


# -----------------------------
# Stage 1: LULC filtering
# -----------------------------
def get_filtered_grid(
    input_polygon,
    imp_lulc,
    med_lulc,
    micro_size_m=200,
    macro_size_m=10_000,
):
    print(f"Starting Filtered Grid Generation: Macro={macro_size_m}m, Micro={micro_size_m}m")

    miner_lulc = _get_lulc_miner()
    to_m, to_deg = _make_projectors(input_polygon)

    # Project AOI once. This is much cheaper than transforming every candidate repeatedly.
    input_poly_m = transform(to_m.transform, input_polygon)
    prepared_aoi_m = prep(input_poly_m)

    minx, miny, maxx, maxy = input_poly_m.bounds

    x_macro = np.arange(minx, maxx, macro_size_m)
    y_macro = np.arange(miny, maxy, macro_size_m)

    final_micro_centers = []
    final_micro_polys = []

    imp_arr = np.asarray(imp_lulc, dtype=np.int32)
    med_arr = np.asarray(med_lulc, dtype=np.int32)
    target_arr = np.unique(np.concatenate([imp_arr, med_arr])).astype(np.int32)
    max_class = int(target_arr.max(initial=0))

    for xm in tqdm.tqdm(x_macro, desc="Processing 10km Chunks"):
        for ym in y_macro:
            macro_poly_m = box(xm, ym, xm + macro_size_m, ym + macro_size_m)

            if not prepared_aoi_m.intersects(macro_poly_m):
                continue

            macro_poly_deg = transform(to_deg.transform, macro_poly_m)

            try:
                lulc_res = miner_lulc.fetch(polygon=macro_poly_deg)
                lulc_xr = lulc_res["data"].squeeze()
                vals = lulc_xr.values.astype(np.int32, copy=False).ravel()

                # Fast macro test: one bincount, no Python loop over every pixel/class.
                counts = np.bincount(vals, minlength=max_class + 1)
                total_px = vals.size

                keep_chunk = False
                if imp_arr.size and counts[imp_arr].any():
                    keep_chunk = True
                elif med_arr.size:
                    med_frac = counts[med_arr] / max(total_px, 1)
                    if np.any(med_frac >= 0.10):
                        keep_chunk = True

                if not keep_chunk:
                    continue

                active_area_m = input_poly_m.intersection(macro_poly_m)
                if active_area_m.is_empty:
                    continue

                m_centers, m_polys = _generate_filtered_micro_grid(
                    poly_m=active_area_m,
                    trans_to_deg=to_deg,
                    step=micro_size_m,
                    lulc_xr=lulc_xr,
                    target_classes=target_arr,
                )

                final_micro_centers.extend(m_centers)
                final_micro_polys.extend(m_polys)

            except Exception as e:
                print(f"LULC error at {xm},{ym}: {e}")
                continue

    return final_micro_centers, final_micro_polys


def _generate_filtered_micro_grid(poly_m, trans_to_deg, step, lulc_xr, target_classes):
    """
    Fast 200m grid generation.
    - Works in meters for grid creation.
    - Uses xarray slicing only for the final LULC check.
    - Avoids STRtree/Point construction entirely.
    """
    minx, miny, maxx, maxy = poly_m.bounds

    # Candidate cell centers in metric space.
    xs = np.arange(minx, maxx, step) + step * 0.5
    ys = np.arange(miny, maxy, step) + step * 0.5

    # Build a boolean target mask once for this macro chunk.
    target_da = _build_target_mask_da(lulc_xr, target_classes)

    # xarray y-axis is often descending in remote-sensing rasters.
    y_desc = bool(target_da.y[0] > target_da.y[-1]) if "y" in target_da.coords else True

    prepared_poly_m = prep(poly_m)

    res_centers = []
    res_polys = []

    # Local bindings reduce attribute lookup overhead in the inner loop.
    to_deg_transform = trans_to_deg.transform
    target_sel = target_da.sel

    half = step * 0.5

    for cy in ys:
        y0 = cy - half
        y1 = cy + half

        for cx in xs:
            x0 = cx - half
            x1 = cx + half

            cell_m = box(x0, y0, x1, y1)
            if not prepared_poly_m.intersects(cell_m):
                continue

            cell_deg = transform(to_deg_transform, cell_m)
            x_slice = slice(*sorted((cell_deg.bounds[0], cell_deg.bounds[2])))
            y_slice = _slice_bounds_from_geom(cell_deg, y_descending=y_desc)

            try:
                sub = target_sel(x=x_slice, y=y_slice)
                if bool(sub.any()):
                    lon, lat = to_deg_transform(cx, cy)
                    res_centers.append([lat, lon])
                    res_polys.append(cell_deg)
            except Exception:
                # Slightly out-of-bounds or weird edge case: skip it.
                continue

    return res_centers, res_polys


# -----------------------------
# Stage 2: inference pipeline
# -----------------------------
def run_sky_grid_pipeline_v4(
    input_polygon,
    imp_lulc,
    med_lulc,
    output_filename="/data/susanket/embeddings.npz",
    batch_size=512,
    num_workers=32,
    save_every=5000,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"

    centers, polygons = get_filtered_grid(input_polygon, imp_lulc, med_lulc)
    total_cells = len(polygons)
    print(f"Filtered to {total_cells} micro-cells.")

    if total_cells == 0:
        print("No cells passed the LULC filter.")
        return

    gdf = gpd.GeoDataFrame(geometry=polygons, crs="EPSG:4326")
    gdf.to_file("/data/susanket/boxes2.geojson")

    print("Loading SkyCLIP...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14",
        pretrained="laion2b_s32b_b82k",
    )
    model.to(device).eval()

    def fetch_worker(poly, center):
        try:
            miner = _get_basemap_miner()
            df = miner.fetch(polygon=poly)
            img = Image.fromarray(df.data.squeeze().transpose(1, 2, 0).astype(np.uint8, copy=False))
            return preprocess(img), center
        except Exception:
            return None

    def producer():
        # map() is simpler and usually lighter than submitting thousands of futures at once.
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for res in executor.map(fetch_worker, polygons, centers):
                if res is not None:
                    job_queue.put(res)
        job_queue.put(None)

    job_queue = Queue(maxsize=batch_size * 2)
    threading.Thread(target=producer, daemon=True).start()

    pbar = tqdm.tqdm(total=total_cells, desc="Processing Tiles")
    all_embs = []
    all_centers = []
    batch_t = []
    batch_c = []
    since_save = 0

    while True:
        item = job_queue.get()

        if item is None:
            if batch_t:
                pass
            else:
                break
        else:
            batch_t.append(item[0])
            batch_c.append(item[1])

        if len(batch_t) >= batch_size or (item is None and batch_t):
            imgs = torch.stack(batch_t, dim=0).to(device, non_blocking=True)

            with torch.inference_mode():
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        feat = model.encode_image(imgs)
                else:
                    feat = model.encode_image(imgs)

                feat = feat / feat.norm(dim=-1, keepdim=True)

            all_embs.append(feat.detach().cpu().numpy())
            all_centers.extend(batch_c)

            pbar.update(len(batch_t))
            since_save += len(batch_t)

            if since_save >= save_every:
                np.savez(
                    output_filename.replace(".npz", "_ckpt.npz"),
                    embeddings=np.vstack(all_embs),
                    centers=np.asarray(all_centers),
                )
                since_save = 0

            batch_t.clear()
            batch_c.clear()

        if item is None:
            break

    pbar.close()

    np.savez(
        output_filename,
        embeddings=np.vstack(all_embs),
        centers=np.asarray(all_centers),
    )
    print(f"Finished! Saved to {output_filename}")


# -----------------------------
# Execution
# -----------------------------
if __name__ == "__main__":
    # Classes: 7 = Built, 5 = Crops, 11 = Rangeland
    aoi = gpd.read_file("/home/susanket/satclip/sentinel/california.geojson").geometry.unary_union

    run_sky_grid_pipeline_v4(
        input_polygon=aoi,
        imp_lulc=[7],      # Priority: urban / built
        med_lulc=[5],      # Secondary: crops
        output_filename="/data/susanket/california_embeddings2.npz",
    )
