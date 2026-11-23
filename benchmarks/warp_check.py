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
#     "pyproj",
#     "scipy",
#     "xarray",
#     "zarr>=3.1.3",
# ]
# ///
"""
A `warp.py` output, against xarray interpolating the same hours from the
run's repo in one process at the target cells' centres, projected with
pyproj, as xarray-sql checks its warp:

\b
    uv run benchmarks/warp_check.py results/warp-....json

By default the run's first 31 days at most; `--days` picks how many. It
reads a day at a time: a month of Earthmover's is about 3 GB, read
anonymously from us-east-1, so it is quickest run from there. The output is
read with the credentials the `aws` CLI would use.
"""

import json
from datetime import datetime
from pathlib import Path

import click
import icechunk
import numpy as np
import xarray as xr
import zarr
from demo_check import TOLERANCE_K, BarePCodec, storage
from regridding_check import zarr_store
from warp_weights import TARGET, target_points


@click.command(help=__doc__)
@click.argument("results", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--days", default=31, show_default=True, help="How many of the run's days.")
def main(results: Path, days: int) -> None:
    run = json.loads(results.read_text())
    start = np.datetime64(datetime.fromisoformat(run["start"]), "h")
    end = min(
        np.datetime64(datetime.fromisoformat(run["end"]), "h"),
        start + np.timedelta64(24 * days, "h"),
    )

    written = xr.open_zarr(zarr_store(run["output"]), consolidated=False).t2m
    y, x = TARGET
    if not (np.array_equal(written.y, y) and np.array_equal(written.x, x)):
        raise click.ClickException("the output's grid isn't the target grid")
    lat, lon = target_points()
    points = {
        "latitude": xr.DataArray(lat, dims=("y", "x")),
        "longitude": xr.DataArray(lon, dims=("y", "x")),
    }

    zarr.registry.register_codec("pcodec", BarePCodec)
    session = icechunk.Repository.open(storage(run["url"])).readonly_session("main")
    ds = xr.open_zarr(session.store, group="single/spatial", chunks=None, consolidated=False)
    t2m = ds.t2m.sel(valid_time=slice(start, end - np.timedelta64(1, "h")))

    worst = (0.0, None)
    for day in range(0, t2m.sizes["valid_time"], 24):
        hours = t2m.isel(valid_time=slice(day, day + 24)).astype("float64").load()
        # 0°E again as 360°E, so that a cell between the last column and the first interpolates
        wrapped = xr.concat(
            [hours, hours.isel(longitude=[0]).assign_coords(longitude=[360.0])], dim="longitude"
        )
        expected = wrapped.sortby("latitude").interp(points).values
        actual = written.sel(valid_time=hours.valid_time.values).values
        if np.isnan(actual).any():
            raise click.ClickException(
                f"{int(np.isnan(actual).sum()):,} cells of day {day // 24 + 1} weren't written"
            )
        differences = np.abs(actual - expected)
        at = np.unravel_index(differences.argmax(), differences.shape)
        if differences[at] > worst[0]:
            worst = (
                float(differences[at]),
                (hours.valid_time.values[at[0]], lat[at[1:]], lon[at[1:]]),
            )
        click.echo(f"  {day // 24 + 1} days checked\r", nl=False)

    hours = t2m.sizes["valid_time"]
    click.echo(f"\n{hours} hours, {hours * y.size * x.size:,} cells")
    difference, where = worst
    if where is not None:
        time, at_lat, at_lon = where
        click.echo(
            f"  largest difference {difference:.2e} K, at {time} {at_lat:.2f}°N {at_lon:.2f}°E"
        )
    if difference >= TOLERANCE_K:
        raise click.ClickException("the warped values differ")
    click.secho("all cells as expected", fg="green")


if __name__ == "__main__":
    main()
