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
A `forecast_skill.py` run's RMSE at each lead, computed by xarray from its
repo in one process:

\b
    uv run benchmarks/forecast_skill_check.py results/forecast-skill-....json

It reads the run's window, and the hours its leads reach past it, an hour at
a time: about 4 MB an hour, read anonymously from us-east-1. Every hour's
error goes into the RMSE, so it checks only windows of a few weeks.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import click
import icechunk
import numpy as np
import xarray as xr
import zarr
from demo_check import BarePCodec, storage

MAX_DAYS = 31
# the sums of squares are added in another order
RELATIVE_TOLERANCE = 1e-9


@click.command(help=__doc__)
@click.argument("results", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def main(results: Path) -> None:
    run = json.loads(results.read_text())
    start, end = datetime.fromisoformat(run["start"]), datetime.fromisoformat(run["end"])
    if end - start > timedelta(days=MAX_DAYS):
        raise click.ClickException(f"the window is longer than {MAX_DAYS} days")
    leads = [r["lead_h"] for r in run["rows"]]

    zarr.registry.register_codec("pcodec", BarePCodec)
    session = icechunk.Repository.open(storage(run["url"])).readonly_session("main")
    ds = xr.open_zarr(session.store, group="single/spatial", chunks=None, consolidated=False)
    hours = int((end - start) / timedelta(hours=1))
    first = int(np.searchsorted(ds.valid_time.values, np.datetime64(start)))
    t2m = ds.t2m.isel(valid_time=slice(first, first + hours + max(leads)))
    if t2m.valid_time.values[0] != np.datetime64(start) or t2m.sizes["valid_time"] < hours:
        raise click.ClickException("the repo doesn't have every hour of the window")

    squares = dict.fromkeys(leads, 0.0)
    counts = dict.fromkeys(leads, 0)
    held: dict[int, np.ndarray] = {}
    for hour in range(hours):
        for i in [hour, *(hour + lead for lead in leads)]:
            if i not in held and i < t2m.sizes["valid_time"]:
                held[i] = t2m.isel(valid_time=i).values.astype("float64")
        for lead in leads:
            if hour + lead in held:
                squares[lead] += float(((held[hour] - held[hour + lead]) ** 2).sum())
                counts[lead] += held[hour].size
        del held[hour]
        click.echo(f"  {hour + 1} of {hours} hours\r", nl=False)

    click.echo(f"\n{start:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M}, {hours} hours")
    problems = []
    for row in run["rows"]:
        lead = row["lead_h"]
        rmse = np.sqrt(squares[lead] / counts[lead])
        relative = abs(row["rmse"] - rmse) / rmse
        click.echo(f"  {lead:3}h  RMSE {rmse:.6f} K, {relative:.1e} from the run's")
        if row["n"] != counts[lead]:
            problems.append(f"{lead}h: {row['n']:,} values, expected {counts[lead]:,}")
        if relative >= RELATIVE_TOLERANCE:
            problems.append(f"{lead}h: RMSE {row['rmse']} K, expected {rmse} K")
    if problems:
        raise click.ClickException("; ".join(problems))
    click.secho(f"all {len(leads)} leads as expected", fg="green")


if __name__ == "__main__":
    main()
