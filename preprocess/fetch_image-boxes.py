"""
Extract centroids of a regular 200x200m (or any size) grid that fall
INSIDE an irregular polygon (e.g. a US state boundary shapefile).

Why this is efficient:
    A naive approach generates every bbox in the polygon's bounding-box
    extent and then does a point-in-polygon / intersects test per box.
    At state-level extent with a 200m grid that's tens of millions of
    Python-level geometry tests -> very slow.

    Instead, this rasterizes the polygon onto a 200m grid using
    rasterio/GDAL's rasterize(), which is a compiled scanline
    algorithm (same class of approach as "walk each row, clip to the
    polygon's x-intersections at that row" -- but done in C, not
    Python). It burns 1s into a boolean array wherever a cell's
    CENTER falls inside the polygon, in effectively one pass over
    the grid. Getting the True cells and converting to coordinates
    is then a couple of vectorized numpy/pyproj calls.

    Practical numbers: a large US state (~1500km x 1000km) at 200m
    resolution is a raster of roughly 7500 x 5000 = 37.5M cells.
    As a uint8 array that's ~37MB, and rasterize() fills it in a
    few seconds. Compare to 37.5M individual shapely `.contains()`
    calls, which would take much, much longer.

Output: an (N, 2) float64 .npy array of [lon, lat] for every cell
whose center lies inside the polygon.
"""

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from pyproj import Transformer


def polygon_grid_centroids(
    shp_path: str,
    cell_size: float = 200.0,        # grid cell size in meters
    target_crs: str = "EPSG:5070",   # NAD83 / Conus Albers Equal Area (meters, low distortion for CONUS)
    all_touched: bool = False,       # False = cell CENTER must be inside poly. True = any overlap counts.
    out_path: str | None = "centroids_lonlat.npy",
    return_row_col: bool = False,
):
    """
    Parameters
    ----------
    shp_path : path to input polygon shapefile (or any OGR-readable format)
    cell_size : grid spacing in meters (e.g. 200 for 200x200m boxes)
    target_crs : a projected, meters-based CRS to grid in. Must be roughly
        equal-area/equidistant over your AOI so 200m cells stay ~200m.
        - EPSG:5070  -> CONUS Albers Equal Area (good default for US states)
        - EPSG:3857  -> Web Mercator (NOT equal-area, distorts at high latitude - avoid)
        - Or use a state-plane / local UTM zone CRS for smaller AOIs.
    all_touched : rasterize option. False (default) burns a cell only if its
        CENTER falls inside the polygon -- this is what you want for
        "centroid of the box is within the polygon". True would instead
        include any cell that the polygon merely touches/overlaps.
    out_path : where to save the (N,2) [lon, lat] float64 npy. Set None to skip saving.
    return_row_col : if True, also returns the (row, col) raster indices,
        useful if you want to align this later with raster data, imagery
        chips, etc. at the same grid.

    Returns
    -------
    coords : np.ndarray, shape (N, 2), columns = [lon, lat]
    (rows, cols) : optional, np.ndarray raster indices of each returned point
    """
    gdf = gpd.read_file(shp_path)
    if gdf.empty:
        raise ValueError("Input shapefile has no features.")

    gdf_m = gdf.to_crs(target_crs)

    # Dissolve all features into one geometry (handles multi-feature shapefiles,
    # e.g. state made of multiple polygon parts / islands)
    union_geom = (
        gdf_m.geometry.union_all()
        if hasattr(gdf_m.geometry, "union_all")
        else gdf_m.geometry.unary_union
    )

    minx, miny, maxx, maxy = union_geom.bounds

    # Snap bounds outward to whole grid cells so the grid is stable/reproducible
    minx = np.floor(minx / cell_size) * cell_size
    miny = np.floor(miny / cell_size) * cell_size
    maxx = np.ceil(maxx / cell_size) * cell_size
    maxy = np.ceil(maxy / cell_size) * cell_size

    width = int(round((maxx - minx) / cell_size))
    height = int(round((maxy - miny) / cell_size))
    n_cells = width * height
    print(f"Grid: {width} x {height} = {n_cells:,} candidate cells "
          f"({n_cells / 1e6:.1f}M), ~{n_cells / 1e6:.0f}MB as uint8")

    transform = rasterio.transform.from_origin(minx, maxy, cell_size, cell_size)

    mask = rasterize(
        [(union_geom, 1)],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=all_touched,
    )

    rows, cols = np.where(mask == 1)
    print(f"{len(rows):,} cells fall inside the polygon "
          f"({100 * len(rows) / n_cells:.1f}% of bbox)")

    # Vectorized pixel-center -> projected-CRS coordinates
    xs = transform.c + (cols + 0.5) * transform.a
    ys = transform.f + (rows + 0.5) * transform.e  # transform.e is negative (north-up)

    # Reproject centers back to lon/lat
    transformer = Transformer.from_crs(target_crs, "EPSG:4326", always_xy=True)
    lons, lats = transformer.transform(xs, ys)

    coords = np.column_stack([lons, lats]).astype(np.float64)  # [:,0]=lon, [:,1]=lat

    if out_path:
        np.save(out_path, coords)
        print(f"Saved {coords.shape[0]:,} centroids -> {out_path}")

    if return_row_col:
        return coords, np.column_stack([rows, cols])
    return coords


if __name__ == "__main__":
    # Example usage:
    coords = polygon_grid_centroids(
        shp_path=rf"E:\Data\Global\US\California\California.shp",
        cell_size=200,
        target_crs="EPSG:5070",   # swap for a UTM zone if AOI is small/local
        out_path=rf"E:\Data\query-earth\preprocess\images\centroids_lonlat.npy",
        all_touched = True,
        return_row_col=True
    )
    print(coords[:5])