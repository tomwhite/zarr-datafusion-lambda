"""
xarray-sql's spectral index (NDVI), as the 10 m wind speed over Earthmover's
ERA5, `single/spatial`, every hour on its 0.25° grid, written to a Zarr store
by the workers: arithmetic on two variables per cell, `sqrt(u10² + v10²)`,
with no aggregate. Each worker writes a day's chunk, so write it to a bucket
the workers may write, in their region:

\b
    export AWS_REGION=us-east-1
    uv run zarr-datafusion-lambda deploy --from-source --allow-bucket <bucket>
    uv run python benchmarks/wind_speed.py s3://<bucket>/wind-speed-2020-01.zarr

By default a month, 31 invocations, writing about 3 GB before compression.
`wind_speed_check.py` checks the output against xarray. **This invokes real
Lambdas and is billed.**
"""

import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import click
import numpy as np
from demo import GROUP, URL, bounds, metadata
from map_lambda import LOCAL_ENDPOINT_URL
from regridding import objects

from zarr_datafusion_lambda import runs, sql

RESULTS = Path(__file__).parent.parent / "results"
CELLS = 721 * 1440
HOURS_PER_CHUNK = 24

# The output's coordinates keep the source's names, as the writer matches them
# to the source's axes.
QUERY = """
COPY (
  SELECT valid_time, latitude, longitude, sqrt(u10 * u10 + v10 * v10) AS wind_speed
  FROM era5
  WHERE valid_time >= TIMESTAMP '{start}' AND valid_time < TIMESTAMP '{end}'
) TO '{output}' STORED AS ZARR OPTIONS ('chunks' '{chunks},721,1440')
"""


@click.command(help=__doc__)
@click.argument("output", required=False)
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
    output: str | None,
    url: str,
    start: datetime,
    end: datetime,
    concurrency: int,
    dispatchers: int | None,
    chunks_in_flight: int | None,
    local: bool,
    explain: bool,
    report: Path | None,
) -> None:
    if report:
        runs.report(json.loads(report.read_text()), report)
        return
    if not output:
        raise click.UsageError("OUTPUT is needed.")
    meta = metadata(url)
    chunk_shape = meta["chunk_grid"]["configuration"]["chunk_shape"]
    chunk_bytes = math.prod(chunk_shape) * np.dtype(meta["data_type"]).itemsize
    tables: dict[str, sql.Table] = {"era5": {"url": url, "group": GROUP}}
    query = QUERY.format(output=output, chunks=HOURS_PER_CHUNK, **bounds(start, end))
    settings: dict[str, Any] = {
        "endpoint_url": LOCAL_ENDPOINT_URL if local else None,
        "concurrency": concurrency,
        "dispatchers": dispatchers,
        "chunks_per_invocation": HOURS_PER_CHUNK // chunk_shape[0],
        "chunks_in_flight": chunks_in_flight,
    }
    if explain:
        click.echo(sql.explain(query, tables, **settings))
        return
    chunks = f"{output.rstrip('/')}/wind_speed/c/"
    if objects(chunks):
        # its chunks would be rewritten in place, and not counted
        raise click.BadParameter(
            f"{output} already has chunks: write to a new store", param_hint="OUTPUT"
        )
    options = click.get_current_context().params
    result = {
        "query": "wind-speed",
        "chunks_per_invocation": settings["chunks_per_invocation"],
        **options,
        **runs.run(query, tables, **settings),
    }
    written = objects(chunks)

    hours = int((end - start) / timedelta(hours=1))
    # each of the scan's chunks is one of each variable's
    runs.print_run(result, concurrency, math.ceil(hours / chunk_shape[0]), 2 * chunk_bytes)
    expected = math.ceil(hours / HOURS_PER_CHUNK)
    [row] = result["rows"]
    click.echo(
        f"{row['count']:,} rows in {written:,} chunks, "
        f"expected {CELLS * hours:,} in {expected:,}, a day each"
    )
    ok = row["count"] == CELLS * hours and written == expected
    click.secho("all as expected" if ok else "PROBLEMS", fg="green" if ok else "red")
    saved = runs.save(RESULTS, "wind-speed", result)
    if not local:
        runs.report(json.loads(saved.read_text()), saved)


if __name__ == "__main__":
    main()
