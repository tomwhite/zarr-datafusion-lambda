"""
xarray-sql's forecast skill over Earthmover's ERA5, `single/spatial`, with a
persistence forecast: ERA5 at each hour of a window as the forecast for 6, 24
and 72 hours later, scored against ERA5 then by the RMSE of `t2m` at each
lead. Each invocation reads a forecast hour, and the hour each lead reaches,
joins them on the workers, and returns a partial row for each lead, which the
dispatchers merge before they reach the client:

\b
    export AWS_REGION=us-east-1
    uv run zarr-datafusion-lambda deploy --from-source
    uv run python benchmarks/forecast_skill.py --start 2020-01-06 --end 2020-01-13

It checks each lead's count and that its RMSE is plausible;
`forecast_skill_check.py` checks a short window's against xarray.
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

LEADS_H = (6, 24, 72)

# The truth first, and the leads before the forecast, as a join is built from
# its first table. The valid time is in a subquery: DataFusion drops an ON
# condition on a column from an earlier CROSS JOIN.
QUERY = """
SELECT f.lead_h,
       sqrt(avg(power(CAST(f.t2m AS DOUBLE) - e.t2m, 2))) AS rmse,
       count(*)                                           AS n
FROM era5 e
JOIN (
  SELECT l.lead_h, latitude, longitude, t2m, CAST(valid_time AS TIMESTAMP) + l.lead AS valid
  FROM (VALUES {leads}) AS l(lead_h, lead)
  CROSS JOIN era5
  WHERE valid_time >= TIMESTAMP '{start}' AND valid_time < TIMESTAMP '{end}'
) f
  ON e.latitude = f.latitude
 AND e.longitude = f.longitude
 AND CAST(e.valid_time AS TIMESTAMP) = f.valid
GROUP BY f.lead_h
ORDER BY f.lead_h
"""

# kelvin, far wider than ERA5's persistence error at any of the leads
PLAUSIBLE_RMSE = (0.5, 20)


def check(rows: list[dict[str, Any]], hours: int, cells: int) -> list[str]:
    """What is wrong with the rows, if anything, for `hours` forecasts of `cells` each."""
    if [r["lead_h"] for r in rows] != list(LEADS_H):
        return [f"expected the leads {LEADS_H}"]
    problems = []
    for row in rows:
        label = f"{row['lead_h']}h"
        if row["n"] != hours * cells:
            problems.append(f"{label}: {row['n']:,} values, expected {hours * cells:,}")
        if not PLAUSIBLE_RMSE[0] < row["rmse"] < PLAUSIBLE_RMSE[1]:
            problems.append(f"{label}: RMSE {row['rmse']:.3f} K")
    return problems


@click.command(help=__doc__)
@click.option("--url", default=URL, show_default=True, help="The Icechunk repo.")
@click.option(
    "--start",
    type=click.DateTime(),
    metavar="DATE",
    default=datetime(2020, 1, 6),
    show_default=True,
)
@click.option(
    "--end", type=click.DateTime(), metavar="DATE", default=datetime(2020, 1, 13), show_default=True
)
@click.option("--concurrency", default=990, show_default=True)
@click.option("--dispatchers", type=int)
@click.option("--chunks-per-invocation", default=1, show_default=True)
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
    leads = ", ".join(f"({h}, INTERVAL '{h} hours')" for h in LEADS_H)
    query = QUERY.format(leads=leads, **bounds(start, end))
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
    result = {"query": "forecast-skill", **options, **runs.run(query, tables, **settings)}

    hours = int((end - start) / timedelta(hours=1))
    # the forecast's chunks: the truth's, a chunk a lead each, aren't counted
    runs.print_run(result, concurrency, math.ceil(hours / chunk_shape[0]), chunk_bytes)
    rows = result["rows"]
    for row in rows:
        click.echo(f"  {row['lead_h']:3}h  RMSE {row['rmse']:6.3f} K  n {row['n']:,}")
    print_problems(rows, check(rows, hours, meta["shape"][1] * meta["shape"][2]))
    saved = runs.save(RESULTS, "forecast-skill", result)
    if not local:
        runs.report(result, saved)


if __name__ == "__main__":
    main()
