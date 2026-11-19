# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "boto3",
#     "click>=8.1",
#     "icechunk>=2",
#     "numcodecs[pcodec]",
#     "numpy",
#     "obstore",
#     "pyarrow",
#     "xarray",
#     "zarr>=3.1.3",
# ]
# ///
"""
A tile of an `anomaly.py` output, against xarray computing the same tile's
anomaly from the run's repo in one process, over the run's window:

\b
    uv run benchmarks/anomaly_check.py results/anomaly-....json

A tile is 12 by 12 cells, numbered from the north pole and the prime meridian.
It reads the tile's pencils anonymously from us-east-1, about 60 MB a year of
the window, and the output with the credentials the `aws` CLI would use.
"""

import json
from datetime import datetime
from pathlib import Path

import click
import icechunk
import numpy as np
import xarray as xr
import zarr
from demo_check import BarePCodec, storage
from regridding_check import zarr_store

GROUP = "single/temporal"
TILE = 12

# K: the anomaly is written in float32, so up to a unit in its last place
TOLERANCE_K = 1e-5


@click.command(help=__doc__)
@click.argument("results", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--row", default=10, show_default=True, help="The tile's row, from the north.")
@click.option("--column", default=0, show_default=True, help="The tile's column, from 0°E.")
def main(results: Path, row: int, column: int) -> None:
    run = json.loads(results.read_text())
    start = np.datetime64(datetime.fromisoformat(run["start"]), "h")
    end = np.datetime64(datetime.fromisoformat(run["end"]), "h")
    rows = slice(row * TILE, (row + 1) * TILE)
    columns = slice(column * TILE, (column + 1) * TILE)

    written = xr.open_zarr(zarr_store(run["output"]), consolidated=False).anomaly
    written = written.isel(latitude=rows, longitude=columns).load()

    zarr.registry.register_codec("pcodec", BarePCodec)
    session = icechunk.Repository.open(storage(run["url"])).readonly_session("main")
    ds = xr.open_zarr(session.store, group=GROUP, chunks=None, consolidated=False)
    t2m = ds.t2m.isel(latitude=rows, longitude=columns).sel(
        valid_time=slice(start, end - np.timedelta64(1, "h"))
    )
    # in memory first: grouping the lazy array reads the store once a group
    by_hour = t2m.load().astype("float64").groupby("valid_time.hour")
    expected = (by_hour - by_hour.mean()).astype("float32")

    for name in ("valid_time", "latitude", "longitude"):
        if not np.array_equal(written[name], expected[name]):
            raise click.ClickException(f"the output's {name} isn't the window's")
    if np.isnan(written.values).any():
        raise click.ClickException(f"{int(np.isnan(written.values).sum()):,} cells weren't written")
    differences = np.abs(written.values - expected.values)
    at = np.unravel_index(differences.argmax(), differences.shape)
    lat, lon = written.latitude.values[at[1]], written.longitude.values[at[2]]
    lats, lons = written.latitude.values, written.longitude.values
    click.echo(
        f"{lats[0]:g}°N to {lats[-1]:g}°N, {lons[0]:g}°E to {lons[-1]:g}°E: "
        f"{written.size:,} cell-hours"
    )
    click.echo(
        f"  largest difference {differences[at]:.2e} K, "
        f"at {written.valid_time.values[at[0]]} {lat:g}°N {lon:g}°E"
    )
    if differences[at] >= TOLERANCE_K:
        raise click.ClickException("the anomalies differ")
    click.secho("all cells as expected", fg="green")


if __name__ == "__main__":
    main()
