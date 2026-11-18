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
A `region_statistics.py` run's every region, computed by xarray from its
repo in one process:

\b
    uv run benchmarks/region_statistics_check.py results/region-statistics-....json

It reads the run's window a day at a time: each chunk is the whole globe,
so a month is about 3 GB, read anonymously from us-east-1, and quickest run
from there.
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
    regions = run["regions"]
    start, end = datetime.fromisoformat(run["start"]), datetime.fromisoformat(run["end"])

    zarr.registry.register_codec("pcodec", BarePCodec)
    session = icechunk.Repository.open(storage(run["url"])).readonly_session("main")
    ds = xr.open_zarr(session.store, group="single/spatial", chunks=None, consolidated=False)
    t2m = ds.t2m.sel(valid_time=slice(start, np.datetime64(end) - np.timedelta64(1, "h")))
    lat, lon = ds.latitude.values, ds.longitude.values
    boxes = [
        ((lat >= south) & (lat <= north), (lon >= west) & (lon <= east))
        for _, south, north, west, east in regions
    ]

    sums = np.zeros(len(regions))
    counts = np.zeros(len(regions), dtype="int64")
    for day in range(0, t2m.sizes["valid_time"], 24):
        values = t2m.isel(valid_time=slice(day, day + 24)).values.astype("float64")
        for i, (lats, lons) in enumerate(boxes):
            box = values[:, lats][:, :, lons]
            sums[i] += box.sum()
            counts[i] += box.size
        click.echo(f"  {day // 24 + 1} days read\r", nl=False)
    means = sums / counts - 273.15

    rows = run["rows"]
    click.echo(f"\n{start:%Y-%m-%d} to {end:%Y-%m-%d}, {t2m.sizes['valid_time']} hours")
    if [r["region"] for r in rows] != [r for r, *_ in regions]:
        raise click.ClickException("the regions differ")
    for row, count in zip(rows, counts, strict=True):
        if row["n"] != count:
            raise click.ClickException(f"{row['region']}: {row['n']:,} values, expected {count:,}")
    differences = means - np.array([r["avg_c"] for r in rows])
    worst = int(np.abs(differences).argmax())
    click.echo(f"  largest difference {differences[worst]:.2e} K, in {rows[worst]['region']}")
    if abs(differences[worst]) >= TOLERANCE_K:
        raise click.ClickException("the means differ")
    click.secho(f"all {len(rows)} regions as expected", fg="green")


if __name__ == "__main__":
    main()
