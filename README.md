# zarr-datafusion-lambda

How fast can we query ERA5 Zarr data using SQL and the [map-lambda](https://github.com/tomwhite/map-lambda) framework?

**The monthly area-weighted global mean temperature over 85 years of hourly
ERA5 (745,128 chunks, 3.1 TB uncompressed) takes ~88s and costs ~$2.17**

**The hourly climatology of every grid cell over the same ERA5 record, written to
Zarr on S3, takes ~136s and costs ~$3.07**

Both answers match xarray for a sample, and the global means match those
Copernicus publishes from ERA5 to within 0.001°C for all but one of the ~1000
months.

## How it works

A query is planned by [DataFusion](https://datafusion.apache.org/), over Zarr
arrays flattened into tables by [zarr-datafusion](https://github.com/stratoscale-io/zarr-datafusion).

A new physical operator `MapLambdaZarrExec` is used to run partial
aggregates on Lambda workers. For the global means, the partial aggregate results
for each worker are streamed back to the client which completes the rest of the plan.

For the climatology, each worker processes all of a group's rows, so it runs
the final aggregate, and writes the output itself.

This project is set up for running benchmarks, it is not a general purpose SQL engine.

## Dependencies

map-lambda is pinned to a release, `v0.1.0.dev1`: `map-lambda==0.1.0.dev1`
for Python, and its tag for the Rust crates and the dispatcher, so that the
client and dispatcher come from the same build. zarr-datafusion is built
from a pinned commit on GitHub.

zarr-datafusion comes from the `lambda-zarr-write` branch of a fork,
<https://github.com/tomwhite/zarr-datafusion>, pinned to a commit, until its
Zarr writes are merged upstream.

## Running locally

First start `cargo lambda watch` from the `rust/` directory, building first,
because `watch` compiles a function on its first request, which would outlast
the dispatcher's invoke timeout:

```shell
cd rust && cargo build && cargo lambda watch
```

Then, in another terminal, run the tests:

```shell
uv run pytest
cd rust && cargo test
```

## Running on AWS

The lambdas are deployed as the map-lambda project `zarr-datafusion-lambda`,
described by `zarr_datafusion_lambda/deploy.toml`: one CloudFormation
stack holding `zarr-datafusion-lambda-dispatcher`, `zarr-datafusion-lambda-fragment`,
their roles and their logs.

The data is in us-east-1, so the workers must be there too (a query refuses
to run otherwise), which the spec fixes. ERA5 is read anonymously, so a
bucket only needs allowing for the climatology, which the workers write to.

### From a release

This needs only [uv](https://docs.astral.sh/uv/): the wheels and the lambda
zips come from this project's [GitHub
release](https://github.com/tomwhite/zarr-datafusion-lambda/releases), and
map-lambda's wheel from its own:

```shell
export UV_FIND_LINKS=https://github.com/tomwhite/zarr-datafusion-lambda/releases/expanded_assets/v0.1.0.dev1,https://github.com/tomwhite/map-lambda/releases/expanded_assets/v0.1.0.dev1
uvx --python 3.12 zarr-datafusion-lambda deploy --allow-bucket <bucket>
uvx --python 3.12 zarr-datafusion-lambda list      # what is deployed
uvx --python 3.12 zarr-datafusion-lambda delete    # when you're finished; the bucket and its data stay
```

`--python 3.12` is needed where the default Python is older, as it is in
CloudShell.

The same commands, and the query below, work in [AWS
CloudShell](https://aws.amazon.com/cloudshell/), which has credentials
already set, once uv is installed:

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
export UV_CACHE_DIR=/tmp/uv-cache   # CloudShell's home directory is limited to 1 GB
```

### From source

From a checkout, building needs Rust, [Zig](https://ziglang.org/) and
[cargo lambda](https://www.cargo-lambda.info/):

```shell
uv run zarr-datafusion-lambda deploy --from-source --allow-bucket <bucket>
uv run zarr-datafusion-lambda list
uv run zarr-datafusion-lambda delete
```

Deploy again after any change; the stack remembers the bucket.

### Running queries

The client finds AWS credentials in the same way as the `aws` CLI, so a
profile, SSO or CloudShell all work, and `AWS_*` variables take precedence.
The workers are in us-east-1, so the client's region must be too:

```shell
export AWS_REGION=us-east-1
```

With the deployment from a release, and `UV_FIND_LINKS` as above, this
queries a week of ERA5's 2 m temperature for its daily global mean, from the
package's `sql` module:

```shell
uv run --no-project --python 3.12 --with zarr-datafusion-lambda python - <<'EOF'
from zarr_datafusion_lambda import sql

tables = {"era5": {"url": "s3://earthmover-icechunk-era5/icechunkV2", "group": "single/spatial"}}
query = """
SELECT date_trunc('day', CAST(valid_time AS TIMESTAMP))                 AS day,
       sum(t2m * cos(radians(latitude))) / sum(cos(radians(latitude))) AS mean_t2m
FROM era5
WHERE valid_time >= TIMESTAMP '2020-01-01' AND valid_time < TIMESTAMP '2020-01-08'
GROUP BY 1
ORDER BY 1
"""
result = sql.run(query, tables, progress=True)
for row in result.table.to_pylist():
    print(f"{row['day']:%Y-%m-%d}  {row['mean_t2m']:.2f} K")
print(f"{result.chunks} chunks in {result.invocations} invocations")
EOF
```

Each benchmark below runs from a checkout, which builds the client from
source, and saves its run to `results/`, with a timeline of its invocations.

## Global mean temperature

`benchmarks/demo.py` computes the monthly area-weighted global mean of `t2m`
over [Earthmover's ERA5](https://registry.opendata.aws/earthmover-era5/),
dataset in Icechunk format. The full dataset goes back to 1940, but the
script accepts a time range to allow you to query part of it.

Start with a few months to check everything is working properly:

```shell
uv run python benchmarks/demo.py --start 2020-01-01 --end 2020-04-01
```

It should complete in a few seconds.

Then run the whole dataset, which costs about $2. The concurrency leaves room
under the account's default limit of 1,000:

```shell
uv run python benchmarks/demo.py --start 1940-01-01 --end 2025-01-01 --concurrency 990
```

In one run, the 1,020 months took 88.2s and cost approximately $2.17:
31,047 invocations of 24 hourly chunks.

<img src="./era5-1940-2024-c990-timeline.svg">

### Checking the results

```shell
# one month against xarray, reading about 3 GB from us-east-1
uv run benchmarks/demo_check.py $(ls results/demo-*.json | tail -1) --month 2020-02
# every month against Copernicus's published daily means, charted beside the run
uv run python benchmarks/copernicus_check.py $(ls results/demo-*.json | tail -1)
```

The month matches xarray to 9 decimal places, with identical extremes. Every
month from 1940 to 2022 agrees with Copernicus within about 0.0006°C, which
rounds each day to 0.001°C. Earthmover's copy of 2023 and 2024 differs a
little more, up to 0.0013°C in July 2024. This discrepancy was not explored further.

<img src="./era5-1940-2024-copernicus.svg">

### The plan

`--explain` prints the query's physical plan, then the fragment that each
fan-out sends to the workers. It doesn't run the query.

```shell
uv run python benchmarks/demo.py --start 1940-01-01 --end 2025-01-01 --concurrency 990 --explain
```

The client runs everything down to the `MapLambdaZarrExec`, which stands in
for the partial aggregate and the scan below it: its 745,128 chunks, those of
the window that pruning kept, go to 31,047 invocations of 24.

```
SortPreservingMergeExec: [month@0 ASC NULLS LAST]
  SortExec: expr=[month@0 ASC NULLS LAST], preserve_partitioning=[true]
    ProjectionExec: expr=[date_trunc(Utf8("month"),era5.valid_time)@0 as month, sum(era5.t2m * cos(radians(era5.latitude)))@1 / sum(cos(radians(era5.latitude)))@2 as mean_t2m, min(era5.t2m)@3 as min_t2m, max(era5.t2m)@4 as max_t2m, count(Int64(1))@5 as n]
      AggregateExec: mode=FinalPartitioned, gby=[date_trunc(Utf8("month"),era5.valid_time)@0 as date_trunc(Utf8("month"),era5.valid_time)], aggr=[sum(era5.t2m * cos(radians(era5.latitude))), sum(cos(radians(era5.latitude))), min(era5.t2m), max(era5.t2m), count(Int64(1))]
        RepartitionExec: partitioning=Hash([date_trunc(Utf8("month"),era5.valid_time)@0], 8), input_partitions=1
          MapLambdaZarrExec: function=fragment, chunks=745128, invocations=31047

fragment:
AggregateExec: mode=Partial, gby=[date_trunc(month, CAST(valid_time@1 AS Timestamp(ns))) as date_trunc(Utf8("month"),era5.valid_time)], aggr=[sum(era5.t2m * cos(radians(era5.latitude))), sum(cos(radians(era5.latitude))), min(era5.t2m), max(era5.t2m), count(Int64(1))]
  ProjectionExec: expr=[cos(radians(CAST(latitude@1 AS Float64))) as __common_expr_1, valid_time@0 as valid_time, t2m@2 as t2m]
    FilterExec: valid_time@0 >= -946771200000000 AND valid_time@0 < 1735689600000000
      ZarrExec: path=s3://earthmover-icechunk-era5/icechunkV2, filters=[valid_time BETWEEN -946771200000000 AND 1735689600000000]
```

Each worker binds the fragment's `ZarrExec` to its own chunks, and returns
one partial row per month it touches.

## Hourly climatology

`benchmarks/climatology.py` computes the mean `t2m` of every grid cell for
each hour of the day, over the same ERA5 in its time-series layout
(`single/temporal`), where a chunk is a "pencil": a year of hours (8,736) over
a 12 x 12 tile. The output is 24 x 721 x 1440 cells, written by the workers to
a Zarr store with a chunk a tile:

```sql
COPY (
  SELECT latitude, longitude,
         date_part('hour', CAST(valid_time AS TIMESTAMP)) AS hour,
         avg(t2m) AS clim, count(*) AS n
  FROM era5
  WHERE valid_time >= TIMESTAMP '{start}' AND valid_time < TIMESTAMP '{end}'
  GROUP BY 1, 2, 3
) TO 's3://<bucket>/climatology.zarr' STORED AS ZARR OPTIONS ('chunks' '24,12,12')
```

Each of the 7,320 invocations takes one tile's whole stack of pencils, so its
rows are final: it finishes the aggregate, writes its tile's chunk, and
returns only a count.

By default it runs over a week, which is useful for checking it works first
(costs about $0.12).

```shell
uv run python benchmarks/climatology.py s3://<bucket>/climatology-week.zarr
```

Then the whole record:

```shell
uv run python benchmarks/climatology.py s3://<bucket>/climatology.zarr --start 1940-01-01 --end 2026-01-01
```

In one run, the whole record took 135.8s and cost approximately $3.07: 7,320
invocations of 87 pencils, in about 7.4 waves at a concurrency of 990. The
cost includes about $0.07 for the workers' ~14,640 writes to the output, which
the run's report leaves out.

<img src="./era5-climatology-c990-timeline.svg">

### Checking the results

```shell
# one tile, 57.25 to 60°N and 0 to 2.75°E, against xarray reading the same pencils
uv run --with xarray --with "numcodecs[pcodec]" --with obstore python benchmarks/climatology_check.py \
  --output s3://<bucket>/climatology.zarr --start 1940-01-01 --end 2026-01-01
```

The tile matches xarray exactly. The check reads the tile's chunk alone, in
about 2 seconds.

## Releasing

Set the version in `pyproject.toml` (`0.1.0.devN` for now), run `uv lock`,
and push a tag of the same version prefixed with `v`. The release workflow
builds the dispatcher and fragment zips, and wheels for Linux x86-64 and
arm64 and macOS arm64 carrying the zips' checksums, then attaches them all to
a GitHub release. Then update the release URLs under Running on AWS.

## AI usage

Code and tests were written with AI assistance from Claude.
