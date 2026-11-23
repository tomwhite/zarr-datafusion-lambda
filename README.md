# zarr-datafusion-lambda

How fast can we query ERA5 Zarr data using SQL and the [map-lambda](https://github.com/tomwhite/map-lambda) framework?

**The hourly climatology of every grid cell over the whole hourly ERA5 record,
1940 to 2025 (3.1 TB uncompressed), written to Zarr on S3, takes ~136s and
costs ~$3.07**

See [quickstart](#quickstart) to run it yourself, or [benchmarks](#benchmarks)
for the time and costs for running other ERA5 queries.

## How it works

A query is planned by [Apache DataFusion](https://datafusion.apache.org/), over Zarr
arrays flattened into tables by [zarr-datafusion](https://github.com/stratoscale-io/zarr-datafusion).

The plan is cut where its work can be done chunk by chunk, and that part is
replaced by a new physical operator, `MapLambdaZarrExec`, which runs it on
Lambda workers through map-lambda, each invocation reading its own run of
chunks. What happens to the workers' results depends on the query:

- **An aggregate:** each worker returns partial results, the dispatchers
  merge them, and the client finishes the aggregate.
- **An aggregate whose groups each lie in one invocation:** each worker
  finishes its groups itself.
- **`COPY … STORED AS ZARR`:** each worker writes its own chunks of the
  output to S3, and returns only a count.
- **A join:** a small table is sent to every worker, and another Zarr store
  is read by each worker only where its own chunks join with it.

Each benchmark's page says which of these it uses. A query that can't be
split this way is refused, rather than run slowly in the client.

This project is set up for running benchmarks, it is not a general purpose SQL engine.

## Quickstart

This deploys the lambdas to your AWS account and queries ERA5 from a Jupyter
notebook. It needs [uv](https://docs.astral.sh/uv/), and AWS credentials,
found as the `aws` CLI finds them. ERA5 is read anonymously from Earthmover's
public bucket in us-east-1, which is where the lambdas are deployed too.

Deploy the lambdas from the latest release:

```shell
export UV_FIND_LINKS=https://github.com/tomwhite/zarr-datafusion-lambda/releases/expanded_assets/v0.1.0.dev2
uvx --python 3.12 zarr-datafusion-lambda deploy
```

Download the notebook, and open it in Jupyter:

```shell
curl -LO https://raw.githubusercontent.com/tomwhite/zarr-datafusion-lambda/v0.1.0.dev2/examples/quickstart.ipynb
uv run --no-project --python 3.12 --with zarr-datafusion-lambda --with "map-lambda[diagnostics]" --with jupyterlab --with ipywidgets --with pandas jupyter lab quickstart.ipynb
```

Run its cells: the query is the monthly global mean of the 2m
temperature over 2020, which prints its stats and its cost after it runs.

The lambdas cost nothing while they aren't running, apart from storing their
logs, so you can leave them deployed. To tidy up once you're done with them:

```shell
uvx --python 3.12 zarr-datafusion-lambda delete
```

## Benchmarks

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

Each benchmark runs from a checkout, deployed [from source](#from-source),
which builds the client from source too, and saves its run to `results/`,
with a timeline of its invocations.

## Deploying and querying

The lambdas are deployed as the map-lambda project `zarr-datafusion-lambda`,
described by `zarr_datafusion_lambda/deploy.toml`: one CloudFormation
stack holding `zarr-datafusion-lambda-dispatcher`,
`zarr-datafusion-lambda-dispatcher-partial-reduce`, which merges the workers'
partial rows, `zarr-datafusion-lambda-fragment`, their roles and their logs.

The data is in us-east-1, so the workers must be there too (a query refuses
to run otherwise), which the spec fixes. ERA5 is read anonymously, so a
bucket only needs allowing for a query that writes to it, such as a `COPY …
STORED AS ZARR`.

### From a release

This needs only uv: the wheels and the lambda zips come from this project's
[GitHub release](https://github.com/tomwhite/zarr-datafusion-lambda/releases),
and map-lambda from PyPI:

```shell
export UV_FIND_LINKS=https://github.com/tomwhite/zarr-datafusion-lambda/releases/expanded_assets/v0.1.0.dev2
uvx --python 3.12 zarr-datafusion-lambda deploy --allow-bucket <bucket>
uvx --python 3.12 zarr-datafusion-lambda list      # what is deployed
uvx --python 3.12 zarr-datafusion-lambda delete    # when you're finished; the bucket and its data stay
```

`--python 3.12` is needed where the default Python is older, as it is in
CloudShell.

The same commands, and the query below, work in [AWS
CloudShell](https://console.aws.amazon.com/cloudshell/home?region=us-east-1),
which has credentials already set, once uv is installed:

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
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

With the deployment from a release, and `UV_FIND_LINKS` as above, this runs
the Quickstart notebook's query from a script, with the package's `sql` module:

```shell
uv run --no-project --python 3.12 --with zarr-datafusion-lambda python - <<'EOF'
from zarr_datafusion_lambda import sql

era5 = sql.Session(
    {"era5": {"url": "s3://earthmover-icechunk-era5/icechunkV2", "group": "single/spatial"}},
    concurrency=990,
)
query = """
SELECT date_trunc('month', CAST(valid_time AS TIMESTAMP))              AS month,
       sum(t2m * cos(radians(latitude))) / sum(cos(radians(latitude))) AS mean_t2m
FROM era5
WHERE valid_time >= TIMESTAMP '2020-01-01' AND valid_time < TIMESTAMP '2021-01-01'
GROUP BY 1
ORDER BY 1
"""
result = era5.sql(query, progress=True)
for row in result.table.to_pylist():
    print(f"{row['month']:%Y-%m}  {row['mean_t2m']:.2f} K")
print(result.stats())
print(result.cost())
EOF
```

From a checkout, with the package built from source, the notebooks in
`examples/` open with:

```shell
uv run --with jupyterlab --with ipywidgets --with pandas jupyter lab examples/
```

## Development

### Running the tests locally

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

### Dependencies

map-lambda is pinned to a release: from PyPI for Python, and the release's
tag for the Rust crates and the dispatchers, so that the client and
dispatchers come from the same release. zarr-datafusion is built
from a pinned commit on GitHub.

zarr-datafusion comes from a fork, <https://github.com/tomwhite/zarr-datafusion>,
pinned to a commit, until the changes it carries are merged upstream.

### Releasing

Set the version in `pyproject.toml` (`0.1.0.devN` for now), run `uv lock`,
and push a tag of the same version prefixed with `v`. The release workflow
builds the dispatchers' and fragment's zips, and wheels for Linux x86-64 and
arm64 and macOS arm64 carrying the zips' checksums, then attaches them all to
a GitHub release. Then update the release URLs in the Quickstart and under Deploying and
querying.

## AI usage

Code and tests were written with AI assistance from Claude.
