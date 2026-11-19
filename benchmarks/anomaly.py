"""
xarray-sql's anomaly of `t2m` over Earthmover's ERA5 pencils,
`single/temporal`: each hour's departure from its cell's mean at that hour of
the day, over the same window, written to a Zarr store by the workers. Its
climatology CTE runs first, written to its own Zarr store, then the query
joined with that store, each worker reading a tile's pencils and the tile's
chunk of the climatology:

\b
    uv run zarr-datafusion-lambda deploy --from-source --allow-bucket <bucket>
    export AWS_REGION=us-east-1
    uv run python benchmarks/anomaly.py s3://<bucket>/anomaly-2020.zarr \\
        --climatology s3://<bucket>/anomaly-2020-climatology.zarr

By default a week, inside one band of pencils. A window over more than one
band is widened to the bands it reaches into, so that the output's chunks, a
pencil long, are the pencils: each invocation of the anomaly reads a tile's
pencil and writes its chunk, whatever the window, while each of the
climatology's reads a tile's whole window. `anomaly_check.py` checks a tile
of the output against xarray. **This invokes real Lambdas and is billed.**
"""

import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import click
from climatology import (
    BAND,
    CELLS,
    CHUNK_BYTES,
    GROUP,
    TILES,
    TIME_START,
    URL,
    ZARR_CHUNKS,
    bands,
    objects,
)
from map_lambda import LOCAL_ENDPOINT_URL

from zarr_datafusion_lambda import runs, sql

RESULTS = Path(__file__).parent.parent / "results"

# TODO: the CASTs work around zarr-datafusion's dictionary-encoded time.
CLIMATOLOGY = """
COPY (
  SELECT latitude,
         longitude,
         date_part('hour', CAST(valid_time AS TIMESTAMP)) AS hour,
         avg(t2m)                                         AS clim_t
  FROM era5
  WHERE valid_time >= TIMESTAMP '{start}' AND valid_time < TIMESTAMP '{end}'
  GROUP BY 1, 2, 3
) TO '{output}' STORED AS ZARR OPTIONS ('chunks' '{chunks}')
"""

# The climatology first, as a join is built from its first table. The output's
# coordinates keep the source's names, as the writer matches them to its axes.
ANOMALY = """
COPY (
  SELECT a.valid_time,
         a.latitude,
         a.longitude,
         CAST(a.t2m - c.clim_t AS REAL) AS anomaly
  FROM clim c
  JOIN era5 a
    ON a.latitude = c.latitude
   AND a.longitude = c.longitude
   AND date_part('hour', CAST(a.valid_time AS TIMESTAMP)) = c.hour
  WHERE a.valid_time >= TIMESTAMP '{start}' AND a.valid_time < TIMESTAMP '{end}'
) TO '{output}' STORED AS ZARR OPTIONS ('chunks' '{chunks},12,12')
"""


def widened(start: datetime, end: datetime) -> tuple[datetime, datetime]:
    """`[start, end)` widened to the bands of pencils it reaches into, if more than one."""
    if bands(start, end) == 1:
        return start, end
    hour = timedelta(hours=1)
    first = (start - TIME_START) // hour // BAND
    last = -(-((end - TIME_START) // hour) // BAND)
    return TIME_START + first * BAND * hour, TIME_START + last * BAND * hour


def refuse_existing(store: str, variable: str, option: str) -> str:
    """`store`'s chunks of `variable`, refused if there are any already."""
    chunks = f"{store.rstrip('/')}/{variable}/c/"
    if objects(chunks):
        # they would be rewritten in place, and not counted
        raise click.BadParameter(
            f"{store} already has chunks: write to a new store", param_hint=option
        )
    return chunks


@click.command(help=__doc__)
@click.argument("output")
@click.option("--climatology", metavar="URL", required=True, help="Where to write the climatology.")
@click.option(
    "--climatology-written",
    is_flag=True,
    help="The climatology is written already, over the same window: run only the anomaly.",
)
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
@click.option("--chunks-in-flight", type=int)
@click.option("--local", is_flag=True, help=f"Against {LOCAL_ENDPOINT_URL}, with no CloudWatch.")
@click.option(
    "--explain",
    is_flag=True,
    help="Print the plans and the workers' fragments, and stop: the anomaly's only once the "
    "climatology is written.",
)
def main(
    output: str,
    climatology: str,
    climatology_written: bool,
    url: str,
    start: datetime,
    end: datetime,
    concurrency: int,
    dispatchers: int | None,
    chunks_in_flight: int | None,
    local: bool,
    explain: bool,
) -> None:
    start, end = widened(start, end)
    click.echo(f"from {start} to {end}")
    window = {"start": f"{start:%Y-%m-%d %H:%M:%S}", "end": f"{end:%Y-%m-%d %H:%M:%S}"}
    hours = int((end - start) / timedelta(hours=1))
    # a pencil, or the whole window inside one
    hours_per_chunk = min(hours, BAND)
    era5: sql.Table = {"url": url, "group": GROUP}
    steps = [
        (
            "climatology",
            CLIMATOLOGY.format(output=climatology, chunks=ZARR_CHUNKS, **window),
            {"era5": era5},
        ),
        (
            "anomaly",
            ANOMALY.format(output=output, chunks=hours_per_chunk, **window),
            {"era5": era5, "clim": {"url": climatology}},
        ),
    ]
    pencils = bands(start, end)
    # the climatology's groups span a tile's whole window, the anomaly's rows don't
    per_invocation = {"climatology": pencils, "anomaly": 1}
    settings: dict[str, Any] = {
        "endpoint_url": LOCAL_ENDPOINT_URL if local else None,
        "concurrency": concurrency,
        "dispatchers": dispatchers,
        "chunks_in_flight": chunks_in_flight,
    }
    if explain:
        for name, query, tables in steps:
            if name == "anomaly" and not objects(f"{climatology.rstrip('/')}/clim_t/c/"):
                click.echo(f"{name}: not until {climatology} is written")
                continue
            plan = sql.explain(
                query, tables, chunks_per_invocation=per_invocation[name], **settings
            )
            click.echo(f"{name}:\n{plan}\n")
        return
    clim_chunks = f"{climatology.rstrip('/')}/clim_t/c/"
    if climatology_written:
        if (found := objects(clim_chunks)) != TILES:
            raise click.BadParameter(
                f"{climatology} has {found:,} chunks, not a climatology's {TILES:,}",
                param_hint="--climatology",
            )
        steps = steps[1:]
    else:
        refuse_existing(climatology, "clim_t", "--climatology")
    written_anomaly = refuse_existing(output, "anomaly", "OUTPUT")

    time_chunks = math.ceil(hours / hours_per_chunk)
    expected = {
        # rows, chunks
        "climatology": (CELLS * 24, TILES),
        "anomaly": (CELLS * hours, TILES * time_chunks),
    }
    written = {"climatology": clim_chunks, "anomaly": written_anomaly}
    options = click.get_current_context().params
    for name, query, tables in steps:
        click.secho(f"{name}:", bold=True)
        result = {
            "query": "anomaly" if name == "anomaly" else "anomaly-climatology",
            "chunks_per_invocation": per_invocation[name],
            **options,
            "start": start,
            "end": end,
            **runs.run(query, tables, chunks_per_invocation=per_invocation[name], **settings),
        }
        chunks = objects(written[name])
        runs.print_run(result, concurrency, pencils * TILES, CHUNK_BYTES)
        [row] = result["rows"]
        rows, chunks_expected = expected[name]
        click.echo(
            f"{row['count']:,} rows in {chunks:,} chunks, expected {rows:,} in {chunks_expected:,}"
        )
        ok = row["count"] == rows and chunks == chunks_expected
        saved = runs.save(RESULTS, result["query"], result)
        if not local:
            runs.report(json.loads(saved.read_text()), saved)
        if not ok:
            raise click.ClickException(f"the {name} isn't as expected")
    click.secho("all as expected", fg="green")


if __name__ == "__main__":
    main()
