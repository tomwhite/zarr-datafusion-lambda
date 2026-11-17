"""
xarray-sql's region statistics of `t2m` over Earthmover's ERA5,
`single/spatial`: the mean inside each of five continental boxes over a
window of hourly chunks, a range join with the boxes as `VALUES`. The boxes
are written first, so they are the join's build side, and every worker is
sent them in its fragment. Each invocation returns a partial row for each
box, which the dispatchers merge, unless `--no-combine`. The workers must be
in the bucket's region, as the deploy spec fixes:

\b
    uv run zarr-datafusion-lambda deploy --from-source
    export AWS_REGION=us-east-1
    uv run python benchmarks/region_statistics.py --start 2020-01-01 --end 2020-02-01

It checks each box's count and that its mean is plausible;
`region_statistics_check.py` checks every box against xarray.
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

# xarray-sql's boxes, as (region, south, north, west, east), longitude 0 to 360,
# in the order the query sorts them
REGIONS = [
    ("Amazon", -10.0, 5.0, 290.0, 310.0),
    ("Australia_Outback", -30.0, -20.0, 125.0, 140.0),
    ("Greenland", 65.0, 80.0, 300.0, 340.0),
    ("SE_Asia", 5.0, 20.0, 95.0, 110.0),
    ("Sahara", 18.0, 30.0, 0.0, 30.0),
]

QUERY = """
SELECT r.region, avg(a.t2m) - 273.15 AS avg_c, count(*) AS n
FROM (VALUES {regions}) AS r(region, south, north, west, east)
JOIN era5 a
  ON a.latitude BETWEEN r.south AND r.north AND a.longitude BETWEEN r.west AND r.east
WHERE a.valid_time >= TIMESTAMP '{start}' AND a.valid_time < TIMESTAMP '{end}'
GROUP BY r.region
ORDER BY r.region
"""

# ERA5's 0.25° grid
LATITUDES = 90 - 0.25 * np.arange(721)
LONGITUDES = 0.25 * np.arange(1440)
# celsius, far wider than any of these boxes' means over a month in ERA5
PLAUSIBLE_MEAN = (-70, 50)


def query(start: datetime, end: datetime) -> str:
    regions = ", ".join(f"('{r}', {s}, {n}, {w}, {e})" for r, s, n, w, e in REGIONS)
    return QUERY.format(regions=regions, **bounds(start, end))


def cells(south: float, north: float, west: float, east: float) -> int:
    """The grid cells in a box, its edges included, as `BETWEEN` includes them."""
    lat, lon = LATITUDES, LONGITUDES
    return int(((lat >= south) & (lat <= north)).sum() * ((lon >= west) & (lon <= east)).sum())


def check(rows: list[dict[str, Any]], start: datetime, end: datetime) -> list[str]:
    """What is wrong with the rows, if anything."""
    if [r["region"] for r in rows] != [r for r, *_ in REGIONS]:
        return [f"expected the {len(REGIONS)} regions {', '.join(r for r, *_ in REGIONS)}"]
    hours = int((end - start) / timedelta(hours=1))
    problems = []
    for row, (region, *box) in zip(rows, REGIONS, strict=True):
        n = hours * cells(*box)
        if row["n"] != n:
            problems.append(f"{region}: {row['n']:,} values, expected {n:,}")
        if not PLAUSIBLE_MEAN[0] < row["avg_c"] < PLAUSIBLE_MEAN[1]:
            problems.append(f"{region}: mean {row['avg_c']:.2f} C")
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
@click.option(
    "--combine/--no-combine",
    default=True,
    show_default=True,
    help="Merge the workers' partial rows in the dispatchers.",
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
    combine: bool,
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
    settings: dict[str, Any] = {
        "endpoint_url": LOCAL_ENDPOINT_URL if local else None,
        "concurrency": concurrency,
        "dispatchers": dispatchers,
        "chunks_per_invocation": chunks_per_invocation,
        "chunks_in_flight": chunks_in_flight,
        "config": {"lambda.combine": str(combine).lower()},
    }
    if explain:
        click.echo(sql.explain(query(start, end), tables, **settings))
        return
    options = click.get_current_context().params
    result = {
        "query": "region-statistics",
        "regions": REGIONS,
        **options,
        **runs.run(query(start, end), tables, **settings),
    }

    hours = (end - start) / timedelta(hours=1)
    runs.print_run(result, concurrency, math.ceil(hours / chunk_shape[0]), chunk_bytes)
    rows = result["rows"]
    for row in rows:
        click.echo(f"  {row['region']:18}  mean {row['avg_c']:7.3f} C  n {row['n']:,}")
    print_problems(rows, check(rows, start, end))
    saved = runs.save(RESULTS, "region-statistics", result)
    if not local:
        runs.report(result, saved)


if __name__ == "__main__":
    main()
