"""
xarray-sql's zonal mean of `t2m` over Earthmover's ERA5, `single/spatial`:
the mean of each of 721 latitudes over a window of hourly chunks. Every
invocation returns a partial row for each latitude, which the dispatchers
merge before they reach the client. The workers must be in the bucket's
region, us-east-1:

\b
    export AWS_REGION=us-east-1
    uv run zarr-datafusion-lambda deploy --from-source
    uv run python benchmarks/zonal_mean.py --start 2020-01-01 --end 2020-02-01

It checks each latitude's count and that its mean is plausible;
`zonal_mean_check.py` checks every latitude against xarray.
**This invokes real Lambdas and is billed.**
"""

import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import click
import numpy as np
from demo import GROUP, URL, bounds, metadata, print_problems
from map_lambda import LOCAL_ENDPOINT_URL

from zarr_datafusion_lambda import runs, sql

RESULTS = Path(__file__).parent.parent / "results"

QUERY = """
SELECT latitude, avg(t2m) - 273.15 AS air_mean_c, count(*) AS n
FROM era5
WHERE valid_time >= TIMESTAMP '{start}' AND valid_time < TIMESTAMP '{end}'
GROUP BY latitude
ORDER BY latitude DESC
"""

LATITUDES = 721
# celsius, far wider than any latitude's mean over a month in ERA5
PLAUSIBLE_MEAN = (-70, 40)


def check(rows: list[dict[str, Any]], start: datetime, end: datetime, longitudes: int) -> list[str]:
    """What is wrong with the rows, if anything, for hourly steps of `longitudes` values a row."""
    expected = 90 - 0.25 * np.arange(LATITUDES)
    if not np.array_equal([r["latitude"] for r in rows], expected):
        return [f"expected the {LATITUDES} latitudes from 90 to -90"]
    n = int((end - start) / timedelta(hours=1)) * longitudes
    problems = []
    for row in rows:
        label = f"{row['latitude']:.2f}"
        if row["n"] != n:
            problems.append(f"{label}: {row['n']:,} values, expected {n:,}")
        if not PLAUSIBLE_MEAN[0] < row["air_mean_c"] < PLAUSIBLE_MEAN[1]:
            problems.append(f"{label}: mean {row['air_mean_c']:.2f} C")
    return problems


@click.command(help=__doc__)
@click.option("--url", default=URL, show_default=True, help="The Icechunk repo.")
@click.option(
    "--start",
    type=click.DateTime(),
    metavar="DATE",
    default=datetime(2020, 1, 1),
    show_default=True,
)
@click.option(
    "--end", type=click.DateTime(), metavar="DATE", default=datetime(2020, 2, 1), show_default=True
)
@click.option("--concurrency", default=990, show_default=True)
@click.option("--dispatchers", type=int)
@click.option("--chunks-per-invocation", default=24, show_default=True)
@click.option("--chunks-in-flight", type=int)
@click.option("--local", is_flag=True, help=f"Against {LOCAL_ENDPOINT_URL}, with no CloudWatch.")
@click.option("--explain", is_flag=True, help="Print the plan and the workers' fragment, and stop.")
@click.option(
    "--report",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    metavar="RESULTS",
    help="Report on a run saved in RESULTS from CloudWatch, without running it again.",
)
def main(
    url: str,
    start: datetime,
    end: datetime,
    concurrency: int,
    dispatchers: int | None,
    chunks_per_invocation: int,
    chunks_in_flight: int | None,
    local: bool,
    explain: bool,
    report: Path | None,
) -> None:
    if report:
        runs.report(json.loads(report.read_text()), report)
        return

    meta = metadata(url)
    chunk_shape = meta["chunk_grid"]["configuration"]["chunk_shape"]
    chunk_bytes = math.prod(chunk_shape) * np.dtype(meta["data_type"]).itemsize
    tables: dict[str, sql.Table] = {"era5": {"url": url, "group": GROUP}}
    query = QUERY.format(**bounds(start, end))
    settings: dict[str, Any] = {
        "endpoint_url": LOCAL_ENDPOINT_URL if local else None,
        "concurrency": concurrency,
        "dispatchers": dispatchers,
        "chunks_per_invocation": chunks_per_invocation,
        "chunks_in_flight": chunks_in_flight,
    }
    if explain:
        click.echo(sql.explain(query, tables, **settings))
        return
    options = click.get_current_context().params
    result = {"query": "zonal-mean", **options, **runs.run(query, tables, **settings)}

    hours = (end - start) / timedelta(hours=1)
    runs.print_run(result, concurrency, math.ceil(hours / chunk_shape[0]), chunk_bytes)
    rows = result["rows"]
    for row in rows[:: (LATITUDES - 1) // 8]:
        click.echo(f"  {row['latitude']:7.2f}  mean {row['air_mean_c']:7.3f} C  n {row['n']:,}")
    print_problems(rows, check(rows, start, end, meta["shape"][2]))
    saved = runs.save(RESULTS, "zonal-mean", result)
    if not local:
        runs.report(result, saved)


if __name__ == "__main__":
    main()
