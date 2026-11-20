# zarr-datafusion-lambda

How fast can we query ERA5 Zarr data using SQL and the [map-lambda](https://github.com/tomwhite/map-lambda) framework?

**The hourly climatology of every grid cell over the whole hourly ERA5 record,
1940 to 2025 (3.1 TB uncompressed), written to Zarr on S3, takes ~136s and
costs ~$3.07**

Each benchmark has a page explaining how it was run. This table is a summary.
Note that some of the benchmarks have not been run on the whole ERA5 dataset,
and have been run by restricting the time window to 5 or 10 years. It is
expected that they will scale to the whole dataset with a proportional increase
in time and cost.

| Benchmark | Shape | Window | Time | Cost | Checked against |
|---|---|---|---|---|---|
| [Global mean temperature](docs/benchmarks/global-mean.md) | `GROUP BY` month | 1940–2024 | 88s | $2.17 | xarray, Copernicus |
| [Zonal mean](docs/benchmarks/zonal-mean.md) | `GROUP BY` latitude, merged in the dispatchers | 1940–2024 | 92s | $2.25 | xarray |
| [Region statistics](docs/benchmarks/region-statistics.md) | joined with a small table | 1940–2024 | 102s | $2.49 | xarray |
| [Hourly climatology](docs/benchmarks/climatology.md) | finished and written to Zarr by the workers | 1940–2025 | 136s | $3.07 | xarray |
| [Regridding](docs/benchmarks/regridding.md) | joined with weights from S3, finished and written to Zarr by the workers | 2015–2024 | 66s | $0.87 | xarray |
| [Wind speed](docs/benchmarks/wind-speed.md) | no aggregate, written to Zarr by the workers | 2015–2024 | 58s | $0.78 | xarray |
| [Anomaly](docs/benchmarks/anomaly.md) | joined with its climatology, written to Zarr first, and written to Zarr by the workers | 2019–2024 | 152s | $1.26 | xarray |
| [Forecast skill](docs/benchmarks/forecast-skill.md) | joined with itself hours later, merged in the dispatchers | 2019–2023 | 78s | $1.69 | xarray |

## How it works

A query is planned by [DataFusion](https://datafusion.apache.org/), over Zarr
arrays flattened into tables by [zarr-datafusion](https://github.com/stratoscale-io/zarr-datafusion).

A new physical operator `MapLambdaZarrExec` is used to run partial
aggregates on Lambda workers. For the [global
mean](docs/benchmarks/global-mean.md), the partial aggregate results for each
worker are streamed back to the client which completes the rest of the plan.

For the [zonal mean](docs/benchmarks/zonal-mean.md), every worker returns a
partial row for every latitude, so the dispatchers merge them as they arrive
and send the client one state each.

For the [region statistics](docs/benchmarks/region-statistics.md), the boxes
are a small table joined with the scan. Each worker is sent the table in its
fragment and joins its own chunks with it, and the dispatchers merge the
partial rows as for the zonal mean.

For the [climatology](docs/benchmarks/climatology.md), each worker processes
all of a group's rows, so it runs the final aggregate, and writes the output
itself.

For [regridding](docs/benchmarks/regridding.md), the weights are a table in
S3, which each worker reads whole and joins its own chunks with. As for the
climatology, each worker has all of a group's rows, an hour's, so it finishes
the aggregate and writes the output itself.

For the [wind speed](docs/benchmarks/wind-speed.md), there is no aggregate:
each worker computes its own chunks' rows and writes them to the output, so
only a count comes back.

For the [anomaly](docs/benchmarks/anomaly.md), the climatology is written to
Zarr first, and the query is joined with that store: each worker reads a
pencil of ERA5 and the climatology's chunk over the same cells, and writes the
pencil's chunk of the output itself.

For the [forecast skill](docs/benchmarks/forecast-skill.md), ERA5 is joined
with itself at each lead: each worker reads an hour of ERA5 as the forecast
and the hours its leads reach as the truth, and returns a partial row for
each lead, which the dispatchers merge.

This project is set up for running benchmarks, it is not a general purpose SQL engine.

## Dependencies

map-lambda is pinned to a release: from PyPI for Python, and the release's
tag for the Rust crates and the dispatchers, so that the client and
dispatchers come from the same release. zarr-datafusion is built
from a pinned commit on GitHub.

zarr-datafusion comes from a fork, <https://github.com/tomwhite/zarr-datafusion>,
pinned to a commit, until the changes it carries are merged upstream.

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
stack holding `zarr-datafusion-lambda-dispatcher`,
`zarr-datafusion-lambda-dispatcher-partial-reduce`, which merges the workers'
partial rows, `zarr-datafusion-lambda-fragment`, their roles and their logs.

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

To edit queries and see their plans, `examples/query.ipynb` runs the same
query in Jupyter. From a release, having downloaded the notebook:

```shell
uv run --no-project --python 3.12 --with zarr-datafusion-lambda --with jupyterlab --with ipywidgets --with pandas jupyter lab
```

From a checkout, with the package built from source:

```shell
uv run --with jupyterlab --with ipywidgets --with pandas jupyter lab examples/query.ipynb
```

Each [benchmark](#zarr-datafusion-lambda) runs from a checkout, which builds
the client from source, and saves its run to `results/`, with a timeline of
its invocations.

## Releasing

Set the version in `pyproject.toml` (`0.1.0.devN` for now), run `uv lock`,
and push a tag of the same version prefixed with `v`. The release workflow
builds the dispatchers' and fragment's zips, and wheels for Linux x86-64 and
arm64 and macOS arm64 carrying the zips' checksums, then attaches them all to
a GitHub release. Then update the release URLs under Running on AWS.

## AI usage

Code and tests were written with AI assistance from Claude.
