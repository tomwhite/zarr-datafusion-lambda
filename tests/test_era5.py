import json
from operator import itemgetter

import datafusion
import numpy as np
import pyarrow as pa
import pytest
import xarray as xr
import zarr

from tests.stores import (
    CHUNKS,
    DISABLED,
    LOCAL,
    PENCILS,
    REGIONS_SQL,
    SHAPE,
    TIME_START,
    VARIABLE,
    rows_close,
    write_group,
)
from zarr_datafusion_lambda.sql import explain, location, run, session

# for what each worker measured, which a combine sums
UNCOMBINED = {"lambda.combine": "false"}

# neither bound falls on a chunk boundary
WINDOW = "time >= TIMESTAMP '1979-01-15 03:00:00' AND time < TIMESTAMP '1979-03-01 00:00:00'"

# The monthly global mean, and xarray-sql's zonal mean and climatology
# (https://xql.systems/xarray-sql/latest/geospatial/), over a window.
# TODO: the CASTs work around zarr-datafusion's dictionary-encoded time, which
# date_trunc and date_part reject.
WINDOWED = {
    "monthly-mean": f"""
SELECT date_trunc('month', CAST(time AS TIMESTAMP))                    AS month,
       sum(t2m * cos(radians(latitude))) / sum(cos(radians(latitude))) AS mean_t2m,
       min(t2m)                                                        AS min_t2m,
       max(t2m)                                                        AS max_t2m,
       count(*)                                                        AS n
FROM era5
WHERE {WINDOW}
GROUP BY 1
ORDER BY 1
""",
    "zonal-mean": f"""
SELECT latitude, avg(t2m) - 273.15 AS air_mean_c, count(*) AS n
FROM era5
WHERE {WINDOW}
GROUP BY latitude
ORDER BY latitude DESC
""",
    "climatology": f"""
SELECT latitude,
       longitude,
       date_part('hour', CAST(time AS TIMESTAMP)) AS hour,
       avg(t2m) - 273.15                          AS clim_c,
       count(*)                                   AS n
FROM era5
WHERE {WINDOW}
  AND latitude BETWEEN 25 AND 50
  AND longitude BETWEEN 235 AND 290
GROUP BY latitude, longitude, date_part('hour', CAST(time AS TIMESTAMP))
ORDER BY latitude DESC, longitude, hour
""",
}


@pytest.fixture
def era5(tmp_path):
    write_group(tmp_path)
    return tmp_path.as_uri()


@pytest.fixture
def era5_pencils(tmp_path):
    write_group(tmp_path, chunks=PENCILS)
    return tmp_path.as_uri()


@pytest.fixture
def era5_metadata(tmp_path):
    """A group with coordinates and no chunks written, which is all planning reads."""
    write_group(tmp_path, with_values=False)
    return tmp_path.as_uri()


def query_locally(url, query):
    """`query` over the whole Zarr group at `url` as table `era5`, flattened in one process."""
    group = zarr.open_group(location(url), mode="r")
    variable = group[VARIABLE]
    names = variable.metadata.dimension_names
    grids = np.meshgrid(*(group[name][:] for name in names), indexing="ij")
    columns = {name: grid.ravel() for name, grid in zip(names, grids, strict=True)}
    times = TIME_START + columns["time"].astype("timedelta64[h]")
    columns["time"] = pa.array(times.astype("datetime64[s]"))
    table = pa.table({**columns, VARIABLE: variable[:].ravel()})
    ctx = datafusion.SessionContext()
    ctx.from_arrow(table, name="era5")
    return ctx.sql(query).to_pylist()


# A spread of aggregates and groupings from the strong rows of "What fits".
# TODO: CAST(time AS TIMESTAMP) works around date_trunc rejecting
# zarr-datafusion's dictionary-encoded time.
QUERIES = [
    "SELECT latitude, avg(t2m) AS mean, stddev(t2m) AS sd, count(*) AS n "
    "FROM era5 GROUP BY latitude ORDER BY latitude",
    "SELECT count(*) AS n, sum(t2m) AS s FROM era5 WHERE latitude > 60 AND longitude < 90",
    "SELECT date_part('day', CAST(time AS TIMESTAMP)) AS day, max(t2m) - min(t2m) AS range "
    "FROM era5 WHERE time < TIMESTAMP '1979-01-08 00:00:00' GROUP BY 1 ORDER BY 1",
    # only the inner aggregate fans out
    "SELECT max(mean) AS warmest, count(*) AS n FROM "
    "(SELECT latitude, avg(t2m) AS mean FROM era5 GROUP BY latitude)",
    # each hour lies in one chunk, so the final aggregate runs on the workers
    "SELECT CAST(time AS TIMESTAMP) AS hour, avg(t2m) AS mean, count(*) AS n "
    "FROM era5 GROUP BY 1 ORDER BY 1",
]


@pytest.mark.lambda_integration
@pytest.mark.parametrize("combine", ["true", "false"])
@pytest.mark.parametrize("query", [*WINDOWED.values(), *QUERIES])
def test_queries_match_one_process(era5, query, combine):
    config = {"lambda.combine": combine}
    result = run(query, {"era5": era5}, **LOCAL, chunks_per_invocation=7, config=config)

    assert result.chunks
    rows_close(result.table.to_pylist(), query_locally(era5, query))


@pytest.mark.lambda_integration
def test_every_invocation_has_its_workers_measurements(era5):
    result = run(
        QUERIES[0],
        {"era5": era5},
        **LOCAL,
        chunks_per_invocation=7,
        return_stats=True,
        config=UNCOMBINED,
    )

    [report] = result.reports
    indexes = sorted(r["index"] for r in report["invocation_stats"])
    assert indexes == list(range(result.invocations))
    assert sorted(w["index"] for w in report["workers"]) == indexes
    # a local store isn't counted
    assert all(w["s3_requests"] == {"gets": 0, "lists": 0} for w in report["workers"])


@pytest.mark.lambda_integration
def test_each_dispatcher_merges_its_workers_into_one_state(era5):
    result = run(QUERIES[0], {"era5": era5}, **LOCAL, dispatchers=3, return_stats=True)

    [report] = result.reports
    states = report["workers"]
    assert len(states) == 3
    covered = sorted(i for state in states for i in state["indexes"])
    assert covered == list(range(result.invocations))
    assert all(s["s3_requests"] == {"gets": 0, "lists": 0} for s in states)
    assert all(s["timings_ms"]["execute"] > 0 for s in states)


@pytest.mark.lambda_integration
def test_progress_counts_up_to_every_invocation(era5):
    calls = []
    result = run(
        QUERIES[0],
        {"era5": era5},
        **LOCAL,
        chunks_per_invocation=7,
        on_progress=lambda done, total: calls.append((done, total)),
    )

    total = result.invocations
    assert calls[0] == (0, total)
    assert calls[-1] == (total, total)
    assert [done for done, _ in calls] == sorted(done for done, _ in calls)


@pytest.mark.lambda_integration
def test_a_progress_bar_gives_the_same_answer(era5):
    result = run(QUERIES[0], {"era5": era5}, **LOCAL, progress=True)

    rows_close(result.table.to_pylist(), query_locally(era5, QUERIES[0]))


@pytest.mark.lambda_integration
def test_dispatchers_share_the_invocations_with_the_same_answer(era5):
    result = run(QUERIES[0], {"era5": era5}, **LOCAL, dispatchers=3, return_stats=True)

    [report] = result.reports
    assert len(report["dispatcher_stats"]) == 3
    assert sum(c["dispatched"] for c in report["dispatcher_stats"]) == result.invocations
    rows_close(result.table.to_pylist(), query_locally(era5, QUERIES[0]))


@pytest.mark.lambda_integration
@pytest.mark.parametrize("in_flight", [1, 3])
def test_chunks_in_flight_give_the_same_answer(era5, in_flight):
    query = WINDOWED["zonal-mean"]
    result = run(
        query,
        {"era5": era5},
        **LOCAL,
        chunks_per_invocation=7,
        chunks_in_flight=in_flight,
        return_stats=True,
        config=UNCOMBINED,
    )

    [report] = result.reports
    assert {w["chunks_in_flight"] for w in report["workers"]} == {in_flight}
    rows_close(result.table.to_pylist(), query_locally(era5, query))


# A persistence-forecast error: each cell against itself a day later, which
# joins chunks along time.
# TODO: the CASTs work around interval arithmetic rejecting zarr-datafusion's
# dictionary-encoded time.
GLOBAL_CLIMATOLOGY = """
SELECT latitude,
       longitude,
       date_part('hour', CAST(time AS TIMESTAMP)) AS hour,
       avg(t2m) - 273.15                          AS clim_c,
       count(*)                                   AS n
FROM era5
GROUP BY latitude, longitude, date_part('hour', CAST(time AS TIMESTAMP))
ORDER BY latitude DESC, longitude, hour
"""


@pytest.mark.lambda_integration
@pytest.mark.parametrize(
    "query", [WINDOWED["climatology"], GLOBAL_CLIMATOLOGY], ids=["windowed", "global"]
)
@pytest.mark.parametrize("chunks_per_invocation", [1, 4, 6])
def test_the_climatology_over_pencils_matches_one_process(
    era5_pencils, query, chunks_per_invocation
):
    result = run(
        query, {"era5": era5_pencils}, **LOCAL, chunks_per_invocation=chunks_per_invocation
    )

    assert result.chunks
    rows_close(result.table.to_pylist(), query_locally(era5_pencils, query))


SELF_JOIN = (
    "SELECT count(*) AS n, avg(abs(a.t2m - b.t2m)) AS mae FROM era5 a JOIN era5 b "
    "ON a.latitude = b.latitude AND a.longitude = b.longitude "
    "AND CAST(b.time AS TIMESTAMP) = CAST(a.time AS TIMESTAMP) + INTERVAL '24 hours'"
)


def fragments(tables, query):
    return session(tables, **LOCAL).fragments(query)


def test_with_lambdas_disabled_a_query_runs_here_with_the_same_answer(era5):
    result = run(QUERIES[0], {"era5": era5}, **LOCAL, config=DISABLED)

    assert result.reports == []
    rows_close(result.table.to_pylist(), query_locally(era5, QUERIES[0]))


def test_set_turns_lambdas_off_and_on_in_a_session(era5_metadata):
    s = session({"era5": era5_metadata}, **LOCAL)

    s.sql("SET lambda.enabled = false")
    assert s.fragments(QUERIES[0]) == []
    s.sql("SET lambda.enabled = true")
    assert len(s.fragments(QUERIES[0])) == 1


def test_explain_with_lambdas_disabled_has_no_fragments(era5_metadata):
    plan = explain(QUERIES[0], {"era5": era5_metadata}, **LOCAL, config=DISABLED)

    assert "MapLambdaZarrExec" not in plan
    assert "fragment:" not in plan


def test_a_fan_out_combines_unless_set_not_to(era5_metadata):
    tables = {"era5": era5_metadata}

    assert "combiner=partial-reduce" in explain(QUERIES[0], tables, **LOCAL)
    assert "combiner=none" in explain(QUERIES[0], tables, **LOCAL, config=UNCOMBINED)


def test_a_misspelt_lambda_option_is_refused(era5_metadata):
    with pytest.raises(RuntimeError, match='"enable" not found'):
        session({"era5": era5_metadata}, **LOCAL, config={"lambda.enable": "false"})


def test_a_partial_aggregate_fans_out_one_chunk_per_invocation(era5_metadata):
    [(_, inputs)] = fragments({"era5": era5_metadata}, QUERIES[0])

    starts = [json.loads(i)[0]["outer"]["Range"][0] for i in inputs]
    assert starts == [k * CHUNKS[0] for k in range(SHAPE[0] // CHUNKS[0])]


def test_only_the_chunks_a_filter_keeps_fan_out(era5_metadata):
    [(_, inputs)] = fragments({"era5": era5_metadata}, QUERIES[2])

    # hours 0 to 167
    assert len(inputs) == 168 // CHUNKS[0] + 1


def boxes(inputs):
    """Each input's one box, as its sorted `(start, end)` on every axis it restricts."""
    specs = [json.loads(i) for i in inputs]
    assert all(len(spec) == 1 for spec in specs)
    return [
        sorted([tuple(s["outer"]["Range"]), *(tuple(r["Range"]) for _, r in s["extra"])])
        for [s] in specs
    ]


def test_a_scan_of_pencils_fans_out_one_pencil_per_invocation(tmp_path):
    write_group(tmp_path, with_values=False, chunks=PENCILS)
    [(_, inputs)] = fragments({"era5": tmp_path.as_uri()}, QUERIES[0])

    pencils = [
        sorted([(t, t + PENCILS[0]), (y, min(y + PENCILS[1], SHAPE[1])), (x, x + PENCILS[2])])
        for t in range(0, SHAPE[0], PENCILS[0])
        for y in range(0, SHAPE[1], PENCILS[1])
        for x in range(0, SHAPE[2], PENCILS[2])
    ]
    assert sorted(boxes(inputs)) == sorted(pencils)


def tiles(inputs):
    """Each input's tiles, as the sorted inner ranges of its boxes."""
    return [
        {tuple(sorted(tuple(r["Range"]) for _, r in spec["extra"])) for spec in json.loads(i)}
        for i in inputs
    ]


def pencil_inputs(tmp_path, query, chunks_per_invocation):
    write_group(tmp_path, with_values=False, chunks=PENCILS)
    tables = {"era5": tmp_path.as_uri()}
    settings = {**LOCAL, "chunks_per_invocation": chunks_per_invocation}
    [(_, inputs)] = session(tables, **settings).fragments(query)
    return inputs


def test_a_climatology_over_pencils_is_dealt_a_tile_at_a_time(tmp_path):
    bands = SHAPE[0] // PENCILS[0]
    inputs = pencil_inputs(tmp_path, GLOBAL_CLIMATOLOGY, chunks_per_invocation=bands)

    assert len(inputs) == 4 * 6
    assert all(len(t) == 1 for t in tiles(inputs))
    # each tile's run of bands, compacted into one spec
    assert all(len(json.loads(i)) == 1 for i in inputs)


def test_a_tile_longer_than_an_invocation_takes_several(tmp_path):
    inputs = pencil_inputs(tmp_path, GLOBAL_CLIMATOLOGY, chunks_per_invocation=4)

    # 6 bands a tile: 4, then 2
    assert len(inputs) == 4 * 6 * 2
    assert all(len(t) == 1 for t in tiles(inputs))


def test_a_group_key_without_the_tiles_axes_is_dealt_in_listing_order(tmp_path):
    inputs = pencil_inputs(tmp_path, QUERIES[0], chunks_per_invocation=6)

    # the 6 tiles of a row of one band
    assert all(len(t) == 6 for t in tiles(inputs))


@pytest.mark.lambda_integration
@pytest.mark.parametrize(
    "climatology",
    [GLOBAL_CLIMATOLOGY, GLOBAL_CLIMATOLOGY.replace("FROM era5\n", f"FROM era5\nWHERE {WINDOW}\n")],
    ids=["all-time", "windowed"],
)
# tiles of 5 latitudes by 6 longitudes: chunks the same, spanning two, and smaller and
# not dividing them, so two boxes read parts of one pencil
@pytest.mark.parametrize("chunks", ["24,5,6", "24,10,6", "24,3,4"])
def test_a_zarr_copy_of_the_climatology_over_pencils_matches_one_process(
    era5_pencils, tmp_path, climatology, chunks
):
    out = tmp_path / "climatology.zarr"
    query = f"COPY ({climatology}) TO '{out}' STORED AS ZARR OPTIONS ('chunks' '{chunks}')"
    tables = {"era5": era5_pencils}
    settings = {**LOCAL, "chunks_per_invocation": SHAPE[0] // PENCILS[0]}

    [fragment] = session(tables, **settings).explain_fragments(query)
    assert "ZarrDataSink" in fragment
    result = run(query, tables, **settings)

    expected = query_locally(era5_pencils, climatology)
    assert result.table.to_pylist() == [{"count": len(expected)}]
    group = zarr.open_group(out, mode="r")
    hours, lats, lons = (group[name][:] for name in ("hour", "latitude", "longitude"))
    assert list(group["clim_c"].metadata.dimension_names) == ["hour", "latitude", "longitude"]
    clim, n = group["clim_c"][:], group["n"][:]
    written = [
        {
            "latitude": lats[y],
            "longitude": lons[x],
            "hour": int(hours[h]),
            "clim_c": float(clim[h, y, x]),
            "n": int(n[h, y, x]),
        }
        for h in range(len(hours))
        for y in range(len(lats))
        for x in range(len(lons))
    ]
    key = itemgetter("latitude", "longitude", "hour")
    rows_close(sorted(written, key=key), sorted(expected, key=key))


@pytest.mark.parametrize(
    ("key", "chunks", "on_the_workers"),
    [
        ("time", CHUNKS, True),
        ("CAST(time AS TIMESTAMP)", CHUNKS, True),
        # a day spans chunks of 31 hours
        ("date_trunc('day', CAST(time AS TIMESTAMP))", CHUNKS, False),
        # an hour of the globe spans pencils' tiles
        ("time", PENCILS, False),
    ],
)
def test_the_final_aggregate_runs_on_the_workers_when_the_key_holds_the_outer_coordinate(
    tmp_path, key, chunks, on_the_workers
):
    write_group(tmp_path, with_values=False, chunks=chunks)
    query = f"SELECT {key} AS t, avg(t2m) AS mean FROM era5 GROUP BY 1"

    [fragment] = session({"era5": tmp_path.as_uri()}, **LOCAL).explain_fragments(query)

    assert ("AggregateExec: mode=Final," in fragment) == on_the_workers


HOURLY_FIELD = (
    "SELECT time, latitude, longitude, avg(t2m) AS t2m FROM era5 GROUP BY time, latitude, longitude"
)


def copy_hourly_field(out, chunks, where=""):
    field = HOURLY_FIELD.replace("FROM era5", f"FROM era5 {where}")
    return f"COPY ({field}) TO '{out}' STORED AS ZARR OPTIONS ('chunks' '{chunks}')"


def output_runs(fragments_and_inputs):
    """Each invocation's `(start, end)` hours of the source, merged."""
    [(_, inputs)] = fragments_and_inputs
    runs = []
    for i in inputs:
        ranges = sorted(tuple(spec["outer"]["Range"]) for spec in json.loads(i))
        runs.append((ranges[0][0], ranges[-1][1]))
    return sorted(runs)


@pytest.mark.parametrize(
    ("chunks", "boundaries"),
    [
        # dealt as the source is, 7 chunks of 31 hours an invocation
        ("31,19,36", list(range(0, SHAPE[0], 7 * 31))),
        # a day straddles the source's chunks, so invocations take 9 whole days
        ("24,19,36", list(range(0, SHAPE[0], 9 * 24))),
    ],
)
def test_a_field_by_hour_is_written_on_the_workers_a_whole_output_chunk_each(
    tmp_path, era5_metadata, chunks, boundaries
):
    s = session({"era5": era5_metadata}, **LOCAL, chunks_per_invocation=7)
    query = copy_hourly_field(tmp_path / "out.zarr", chunks)

    [fragment] = s.explain_fragments(query)
    assert "ZarrDataSink" in fragment
    assert [start for start, _ in output_runs(s.fragments(query))] == boundaries


FROM_MIDDAY = (
    "WHERE time >= TIMESTAMP '1979-01-15 03:00:00' AND time < TIMESTAMP '1979-03-01 00:00:00'"
)


@pytest.mark.lambda_integration
@pytest.mark.parametrize(
    ("chunks", "where"),
    [("31,19,36", ""), ("24,19,36", ""), ("24,19,36", FROM_MIDDAY)],
    ids=["source-chunks", "days", "days-from-a-window"],
)
def test_a_zarr_copy_of_a_field_by_hour_matches_the_source(era5, tmp_path, chunks, where):
    out = tmp_path / "out.zarr"
    query = copy_hourly_field(out, chunks, where)
    run(query, {"era5": era5}, **LOCAL, chunks_per_invocation=7)

    source = xr.open_zarr(location(era5), consolidated=False)
    if where:
        source = source.sel(time=slice("1979-01-15T03", "1979-02-28T23"))
    written = xr.open_zarr(out, consolidated=False)
    np.testing.assert_array_equal(written.time.values, source.time.values)
    np.testing.assert_allclose(written.t2m.values, source.t2m.values, rtol=1e-6)


def test_a_zarr_copy_whose_chunks_span_tiles_is_dealt_a_whole_output_box_each(tmp_path):
    write_group(tmp_path / "era5", with_values=False, chunks=PENCILS)
    out = tmp_path / "climatology.zarr"
    # a chunk of 10 latitudes spans two tiles of 5
    query = f"COPY ({GLOBAL_CLIMATOLOGY}) TO '{out}' STORED AS ZARR OPTIONS ('chunks' '24,10,6')"
    settings = {**LOCAL, "chunks_per_invocation": SHAPE[0] // PENCILS[0]}
    s = session({"era5": (tmp_path / "era5").as_uri()}, **settings)

    [fragment] = s.explain_fragments(query)
    [(_, inputs)] = s.fragments(query)

    assert "ZarrDataSink" in fragment
    # one box an invocation, through all of time, compacted into one spec
    specs = [json.loads(i) for i in inputs]
    assert all(len(spec) == 1 for spec in specs)
    boxes = [{coord: tuple(r["Range"]) for coord, r in spec["extra"]} for [spec] in specs]
    assert sorted({box[1] for box in boxes}) == [(0, 10), (10, 19)]
    assert sorted({box[2] for box in boxes}) == [(x, x + 6) for x in range(0, 36, 6)]
    assert len(boxes) == 2 * 6


@pytest.mark.parametrize(("chunks_per_invocation", "on_the_workers"), [(6, True), (4, False)])
def test_the_final_aggregate_runs_on_the_workers_when_each_tile_takes_one_invocation(
    tmp_path, chunks_per_invocation, on_the_workers
):
    write_group(tmp_path, with_values=False, chunks=PENCILS)
    tables = {"era5": tmp_path.as_uri()}
    settings = {**LOCAL, "chunks_per_invocation": chunks_per_invocation}

    [fragment] = session(tables, **settings).explain_fragments(GLOBAL_CLIMATOLOGY)
    plan = explain(GLOBAL_CLIMATOLOGY, tables, **settings).split("fragment:")[0]

    assert ("AggregateExec: mode=Final," in fragment) == on_the_workers
    assert ("mode=FinalPartitioned" in plan) != on_the_workers
    assert ("combiner=partial-reduce" in plan) != on_the_workers


def test_a_filtered_inner_axis_is_not_cut(tmp_path):
    write_group(tmp_path, with_values=False, chunks=PENCILS)
    [(_, inputs)] = fragments({"era5": tmp_path.as_uri()}, WINDOWED["climatology"])

    # the window's four runs of hours, each over the whole grid, where the filter applies
    assert len(inputs) == 4
    assert all(json.loads(i)[0].get("extra", []) == [] for i in inputs)


def test_a_fragment_runs_one_partition_per_chunk_whatever_the_clients_cores(era5_metadata):
    # 6 chunks survive, fewer than the partitions planned for here
    [plan] = session({"era5": era5_metadata}, **LOCAL, target_partitions=16).explain_fragments(
        QUERIES[2]
    )

    assert "RoundRobinBatch" not in plan


@pytest.mark.parametrize(
    ("query", "why"),
    [
        (SELF_JOIN, "two Zarr scans are joined"),
        ("SELECT t2m FROM era5 LIMIT 3", "can't be split into chunks"),
        (
            "SELECT max(r) AS r FROM (SELECT row_number() OVER (ORDER BY time) AS r FROM era5)",
            "don't work chunk by chunk: .*BoundedWindowAggExec",
        ),
        (
            "SELECT count(*) AS n FROM (SELECT t2m FROM era5 LIMIT 100)",
            "can't be split into chunks",
        ),
    ],
)
def test_a_query_that_cant_be_cut_by_chunk_is_refused(era5_metadata, query, why):
    with pytest.raises(RuntimeError, match=why):
        fragments({"era5": era5_metadata}, query)


def test_a_count_from_metadata_reads_no_chunks(era5):
    query = "SELECT count(*) AS n FROM era5"
    result = run(query, {"era5": era5}, **LOCAL)

    assert result.chunks is None
    rows_close(result.table.to_pylist(), query_locally(era5, query))


REGIONS = REGIONS_SQL.replace("AS regions(", "AS r(")
IN_REGION = "a.latitude BETWEEN r.south AND r.north AND a.longitude BETWEEN r.west AND r.east"


def by_region(joined):
    return f"SELECT r.region, avg(a.t2m) AS mean, count(*) AS n FROM {joined} GROUP BY r.region"


def test_a_join_with_a_small_table_fans_out_with_the_table_in_the_fragment(era5_metadata):
    s = session({"era5": era5_metadata}, **LOCAL)
    query = by_region(f"{REGIONS} JOIN era5 a ON {IN_REGION}")

    assert len(s.fragments(query)) == 1
    [fragment] = s.explain_fragments(query)
    assert "NestedLoopJoinExec" in fragment
    assert "DataSourceExec: partitions=1" in fragment


@pytest.mark.parametrize(
    ("joined", "why"),
    [
        (f"era5 a JOIN {REGIONS} ON {IN_REGION}", "build side: write the smaller side first"),
        # every worker would add the regions none of its rows fall in
        (f"{REGIONS} LEFT JOIN era5 a ON {IN_REGION}", "can't be sent to every worker"),
        # too big to send every worker
        (f"big r JOIN era5 a ON {IN_REGION}", "can't be sent to every worker"),
    ],
)
def test_a_join_that_cant_be_broadcast_is_refused(era5_metadata, joined, why):
    s = session({"era5": era5_metadata}, **LOCAL)
    # 1.6 MB as two float64 columns
    s.sql(
        "CREATE TABLE big AS SELECT CAST(value AS DOUBLE) AS south, "
        "CAST(value AS DOUBLE) + 90 AS north, 0.0 AS west, 360.0 AS east, "
        "CAST(value AS VARCHAR) AS region FROM generate_series(1, 100000)"
    )

    with pytest.raises(RuntimeError, match=why):
        s.fragments(by_region(joined))


# TODO: CAST(time AS TIMESTAMP) because zarr-datafusion's time is UTC, and the
# single-process check's is naive.
UNSPLITTABLE = [
    "SELECT CAST(time AS TIMESTAMP) AS time, latitude, longitude, t2m FROM era5 "
    "ORDER BY time, latitude, longitude LIMIT 5",
    SELF_JOIN,
]


@pytest.mark.parametrize("query", UNSPLITTABLE)
def test_a_query_that_cant_be_cut_runs_here_with_lambdas_disabled(era5, query):
    result = run(query, {"era5": era5}, **LOCAL, config=DISABLED)

    assert result.chunks is None
    rows_close(result.table.to_pylist(), query_locally(era5, query))
