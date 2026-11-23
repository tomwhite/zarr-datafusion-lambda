"""
xarray-sql's warp of `t2m` over Earthmover's ERA5, `single/spatial`, from
its 0.25° grid onto Europe's Lambert azimuthal equal-area grid (EPSG:3035)
at 25 km every hour, written to a Zarr store by the workers: a join with
bilinear weights, a weighted sum by target cell and hour. Every worker reads
the weights, Parquet in S3, whole, and an hour's cells are all in its chunks,
so it finishes the sum and writes a day's chunk itself. Write the weights
first (`warp_weights.py`), to a bucket the workers may read and write, in
their region:

\b
    uv run benchmarks/warp_weights.py s3://<bucket>/era5-laea-weights.parquet
    uv run zarr-datafusion-lambda deploy --from-source --allow-bucket <bucket>
    export AWS_REGION=us-east-1
    uv run python benchmarks/warp.py s3://<bucket>/warped-2020-01.zarr \\
        --weights s3://<bucket>/era5-laea-weights.parquet

By default a month, 31 invocations; `--start 1940-01-01 --end 2025-01-01` is
the whole record, about 31,000. `warp_check.py` checks the output against
xarray. **This invokes real Lambdas and is billed.**
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
from warp_weights import TARGET

from zarr_datafusion_lambda import runs, sql

RESULTS = Path(__file__).parent.parent / "results"
HOURS_PER_CHUNK = 24

# The output's time keeps the source's name, so that the workers' output chunks
# can be lined up with the hours they read.
QUERY = """
COPY (
  SELECT s.valid_time                AS valid_time,
         w.dst_y                     AS y,
         w.dst_x                     AS x,
         sum(s.t2m * w.weight)       AS t2m
  FROM weights w
  JOIN era5 s ON s.latitude = w.src_lat AND s.longitude = w.src_lon
  WHERE s.valid_time >= TIMESTAMP '{start}' AND s.valid_time < TIMESTAMP '{end}'
  GROUP BY 1, 2, 3
) TO '{output}' STORED AS ZARR OPTIONS ('chunks' '{chunks},{ny},{nx}')
"""


@click.command(help=__doc__)
@click.argument("output", required=False)
@click.option("--weights", metavar="URL", help="The weights, from warp_weights.py.")
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
    weights: str | None,
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
    if not output or not weights:
        raise click.UsageError("OUTPUT and --weights are needed.")
    meta = metadata(url)
    chunk_shape = meta["chunk_grid"]["configuration"]["chunk_shape"]
    chunk_bytes = math.prod(chunk_shape) * np.dtype(meta["data_type"]).itemsize
    tables: dict[str, sql.Table] = {
        "era5": {"url": url, "group": GROUP},
        "weights": {"url": weights, "format": "parquet"},
    }
    ny, nx = (axis.size for axis in TARGET)
    query = QUERY.format(output=output, chunks=HOURS_PER_CHUNK, ny=ny, nx=nx, **bounds(start, end))
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
    chunks = f"{output.rstrip('/')}/t2m/c/"
    if objects(chunks):
        # its chunks would be rewritten in place, and not counted
        raise click.BadParameter(
            f"{output} already has chunks: write to a new store", param_hint="OUTPUT"
        )
    options = click.get_current_context().params
    result = {
        "query": "warp",
        "chunks_per_invocation": settings["chunks_per_invocation"],
        **options,
        **runs.run(query, tables, **settings),
    }
    written = objects(chunks)

    hours = int((end - start) / timedelta(hours=1))
    runs.print_run(result, concurrency, math.ceil(hours / chunk_shape[0]), chunk_bytes)
    expected = math.ceil(hours / HOURS_PER_CHUNK)
    [row] = result["rows"]
    click.echo(
        f"{row['count']:,} rows in {written:,} chunks, "
        f"expected {ny * nx * hours:,} in {expected:,}, a day each"
    )
    ok = row["count"] == ny * nx * hours and written == expected
    click.secho("all as expected" if ok else "PROBLEMS", fg="green" if ok else "red")
    saved = runs.save(RESULTS, "warp", result)
    if not local:
        runs.report(json.loads(saved.read_text()), saved)


if __name__ == "__main__":
    main()
