# Global mean temperature

[zarr-datafusion-lambda](../../README.md) › Benchmarks

`benchmarks/demo.py` computes the monthly area-weighted global mean of `t2m`
over [Earthmover's ERA5](https://registry.opendata.aws/earthmover-era5/),
dataset in Icechunk format. The full dataset goes back to 1940, but the
script accepts a time range to allow you to query part of it.

Every worker's partial rows, one per month it touches, are streamed back to
the client, which finishes the aggregate.

## Running it

Like every benchmark, it runs from a checkout, deployed [from
source](../../README.md#from-source), and saves its run to `results/`, with a
timeline of its invocations.

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

## Results

In one run, the 1,020 months took 88.2s and cost approximately $2.17:
31,047 invocations of 24 hourly chunks.

<img src="./global-mean-timeline.svg">

## Checking the results

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

<img src="./global-mean-copernicus.svg">

## The plan

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
