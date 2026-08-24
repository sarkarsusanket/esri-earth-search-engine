import argparse
import shutil
import subprocess
from pathlib import Path

import pyogrio

from telemetry import StageTelemetry


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PBF = ROOT / "outputs" / "v2" / "hawaii" / "hawaii_raw.osm.pbf"
DEFAULT_OUTPUT = ROOT / "outputs" / "v2" / "hawaii" / "osm_features.gpkg"
DEFAULT_OSM_CONFIG = Path(__file__).resolve().parent / "osmconf.ini"
DEFAULT_OGR2OGR = Path(
    r"C:\Program Files\ArcGIS\Pro\bin\Python\envs\arcgispro-py3\Library\bin\ogr2ogr.exe"
)
REQUIRED_LAYERS = {"points", "lines", "multipolygons"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert an OSM PBF into a faithful v2 GeoPackage.")
    parser.add_argument("--pbf", type=Path, default=DEFAULT_PBF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--osm-config", type=Path, default=DEFAULT_OSM_CONFIG)
    parser.add_argument("--ogr2ogr", type=Path, default=DEFAULT_OGR2OGR)
    args = parser.parse_args()

    for path in [args.pbf, args.osm_config, args.ogr2ogr]:
        if not path.exists():
            raise FileNotFoundError(path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for output_path in [
        args.output,
        Path(f"{args.output}-journal"),
        Path(f"{args.output}-wal"),
        Path(f"{args.output}-shm"),
    ]:
        output_path.unlink(missing_ok=True)

    command = [
        str(args.ogr2ogr),
        "-f",
        "GPKG",
        str(args.output),
        str(args.pbf),
        "points",
        "lines",
        "multipolygons",
        "-lco",
        "SPATIAL_INDEX=YES",
        "-progress",
        "--config",
        "OSM_CONFIG_FILE",
        str(args.osm_config),
    ]
    with StageTelemetry("pbf_to_gpkg", args.output.parent):
        subprocess.run(command, check=True)

    layers = set(pyogrio.list_layers(args.output)[:, 0].tolist())
    missing_layers = REQUIRED_LAYERS - layers
    if missing_layers:
        args.output.unlink(missing_ok=True)
        raise RuntimeError(f"GeoPackage is missing required layers: {sorted(missing_layers)}")
    print(f"Created {args.output} ({args.output.stat().st_size:,} bytes); layers={sorted(layers)}")


if __name__ == "__main__":
    main()