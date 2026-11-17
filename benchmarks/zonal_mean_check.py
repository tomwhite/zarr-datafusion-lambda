# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "click>=8.1",
#     "icechunk>=2",
#     "numcodecs[pcodec]",
#     "numpy",
#     "xarray",
#     "zarr>=3.1.3",
# ]
# ///
"""
A `zonal_mean.py` run's every latitude, computed by xarray from its repo in
one process:

\b
    uv run benchmarks/zonal_mean_check.py results/zonal-mean-....json

It reads the run's window a day at a time: a month of Earthmover's is about
3 GB, read anonymously from us-east-1, so it is quickest run from there.
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


@click.command(help=__doc__)
@click.argument("results", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def main(results: Path) -> None:
    run = json.loads(results.read_text())
    start, end = datetime.fromisoformat(run["start"]), datetime.fromisoformat(run["end"])

    zarr.registry.register_codec("pcodec", BarePCodec)
    session = icechunk.Repository.open(storage(run["url"])).readonly_session("main")
    ds = xr.open_zarr(session.store, group="single/spatial", chunks=None, consolidated=False)
    t2m = ds.t2m.sel(valid_time=slice(start, np.datetime64(end) - np.timedelta64(1, "h")))

    sums = np.zeros(ds.sizes["latitude"])
    n = 0
    for day in range(0, t2m.sizes["valid_time"], 24):
        values = t2m.isel(valid_time=slice(day, day + 24)).values.astype("float64")
        sums += values.sum(axis=(0, 2))
        n += values.shape[0] * values.shape[2]
        click.echo(f"  {day // 24 + 1} days read\r", nl=False)
    means = sums / n - 273.15

    rows = run["rows"]
    click.echo(f"\n{start:%Y-%m-%d} to {end:%Y-%m-%d}, {t2m.sizes['valid_time']} hours")
    if not np.array_equal([r["latitude"] for r in rows], ds.latitude.values):
        raise click.ClickException("the latitudes differ")
    counts = [r["n"] for r in rows]
    if any(c != n for c in counts):
        raise click.ClickException(
            f"counts from {min(counts):,} to {max(counts):,}, expected {n:,}"
        )
    differences = means - np.array([r["air_mean_c"] for r in rows])
    worst = int(np.abs(differences).argmax())
    click.echo(
        f"  largest difference {differences[worst]:.2e} K, at latitude {rows[worst]['latitude']}"
    )
    if abs(differences[worst]) >= TOLERANCE_K:
        raise click.ClickException("the means differ")
    click.secho(f"all {len(rows)} latitudes as expected", fg="green")


if __name__ == "__main__":
    main()
