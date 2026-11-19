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
A `wind_speed.py` output, against xarray computing the same hours from the
run's repo in one process:

\b
    uv run benchmarks/wind_speed_check.py results/wind-speed-....json

By default the run's first 31 days at most; `--days` picks how many. It
reads a day at a time: a month of Earthmover's `u10` and `v10` is about 6 GB,
read anonymously from us-east-1, so it is quickest run from there. The output
is read with the credentials the `aws` CLI would use.
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

# m/s: a float32's precision at the fastest winds
TOLERANCE = 1e-5


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

    zarr.registry.register_codec("pcodec", BarePCodec)
    session = icechunk.Repository.open(storage(run["url"])).readonly_session("main")
    ds = xr.open_zarr(session.store, group="single/spatial", chunks=None, consolidated=False)
    ds = ds[["u10", "v10"]].sel(valid_time=slice(start, end - np.timedelta64(1, "h")))

    written = xr.open_zarr(zarr_store(run["output"]), consolidated=False).wind_speed
    if not (
        np.array_equal(written.latitude, ds.latitude)
        and np.array_equal(written.longitude, ds.longitude)
    ):
        raise click.ClickException("the output's grid isn't the source's")

    worst = (0.0, None)
    for day in range(0, ds.sizes["valid_time"], 24):
        hours = ds.isel(valid_time=slice(day, day + 24))
        u10, v10 = hours.u10.values, hours.v10.values
        expected = np.sqrt(u10 * u10 + v10 * v10)
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
                (
                    hours.valid_time.values[at[0]],
                    float(ds.latitude[at[1]]),
                    float(ds.longitude[at[2]]),
                ),
            )
        click.echo(f"  {day // 24 + 1} days checked\r", nl=False)

    hours = ds.sizes["valid_time"]
    click.echo(f"\n{hours} hours, {hours * ds.sizes['latitude'] * ds.sizes['longitude']:,} cells")
    difference, where = worst
    if where is None:
        click.echo("  every cell matches exactly")
    else:
        time, at_lat, at_lon = where
        click.echo(f"  largest difference {difference:.2e} m/s, at {time} {at_lat}°N {at_lon}°E")
    if difference >= TOLERANCE:
        raise click.ClickException("the wind speeds differ")
    click.secho("all cells as expected", fg="green")


if __name__ == "__main__":
    main()
