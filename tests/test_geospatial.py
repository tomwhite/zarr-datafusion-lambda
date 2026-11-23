"""xarray-sql's geospatial cases (https://xql.systems/xarray-sql/latest/geospatial/)
over a small store chunked as ERA5 is, each answered fanned out, in the
client, and by xarray, and cut where expected."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from operator import itemgetter

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import xarray as xr

from tests.stores import (
    CHUNKS,
    DISABLED,
    LOCAL,
    PENCILS,
    REGIONS,
    REGIONS_SQL,
    SHAPE,
    SURFACE,
    VARIABLE,
    rows_close,
    weights,
    write_group,
    write_weights,
)
from zarr_datafusion_lambda.sql import Result, session

# What a fragment runs on the workers, one for each place `FanOutRule` cuts a plan.
PARTIAL = "a partial aggregate, combined in the dispatchers"
FINAL = "the final aggregate"
MAP = "no aggregate: rows"


@dataclass(frozen=True)
class Case:
    query: str
    # sorted, as `cut` gives them; `None` if it can't be cut, so is refused
    cut: tuple[str, ...] | None = None
    reference: Callable[[xr.Dataset], list[dict]] | None = None
    chunks: tuple[int, int, int] = CHUNKS
    chunks_per_invocation: int = 7
    variables: tuple[str, ...] = (VARIABLE,)
    weights: bool = False


def zonal_mean(ds):
    t2m = ds.t2m.astype("float64")
    mean = t2m.mean(["time", "longitude"]) - 273.15
    n = t2m.count(["time", "longitude"])
    rows = [
        {"latitude": float(lat), "air_mean_c": float(m), "n": int(c)}
        for lat, m, c in zip(ds.latitude.values, mean.values, n.values, strict=True)
    ]
    return sorted(rows, key=lambda r: -r["latitude"])


def climatology(ds):
    by_hour = ds.t2m.astype("float64").groupby("time.hour")
    clim = (by_hour.mean() - 273.15).transpose("hour", "latitude", "longitude")
    n = by_hour.count().transpose("hour", "latitude", "longitude")
    rows = [
        {
            "latitude": float(lat),
            "longitude": float(lon),
            "hour": int(hour),
            "clim_c": float(clim.values[h, y, x]),
            "n": int(n.values[h, y, x]),
        }
        for h, hour in enumerate(clim.hour.values)
        for y, lat in enumerate(clim.latitude.values)
        for x, lon in enumerate(clim.longitude.values)
    ]
    return sorted(rows, key=lambda r: (-r["latitude"], r["longitude"], r["hour"]))


def records(da, name, columns):
    """`da`'s values as rows of `columns`, each its value named `name` and its coordinates."""
    frame = da.rename(name).to_dataframe().reset_index()
    rows = frame[list(columns)].to_dict("records")
    for row in rows:
        if "time" in row:
            row["time"] = pd.Timestamp(row["time"])
    return rows


FIRST_DAY = slice(None, "1979-01-01T23")
POINT = ("time", "latitude", "longitude")


def derived_variable(ds):
    day = ds.sel(time=FIRST_DAY)
    # in float32, as the query computes it
    wind_speed = np.sqrt(day.u10 * day.u10 + day.v10 * day.v10)
    rows = records(wind_speed, "wind_speed", (*POINT, "wind_speed"))
    return sorted(rows, key=lambda r: (r["time"], -r["latitude"], r["longitude"]))


def anomaly(ds):
    t2m = ds.t2m.sel(
        time=slice("1979-01-15T03", "1979-02-28T23"),
        latitude=slice(50, 25),
        longitude=slice(235, 290),
    ).astype("float64")
    by_hour = t2m.groupby("time.hour")
    rows = records(by_hour - by_hour.mean(), "anomaly", (*POINT, "anomaly"))
    return sorted(rows, key=lambda r: (r["time"], -r["latitude"], r["longitude"]))


def persistence_skill(ds):
    week = 7 * 24
    rows = []
    for lead_h in (6, 24, 72):
        forecast = ds.t2m.isel(time=slice(0, week)).values
        truth = ds.t2m.isel(time=slice(lead_h, lead_h + week)).values
        # subtracted in float32, as the query does
        error = (forecast - truth).astype("float64")
        rmse = float(np.sqrt(np.mean(error**2)))
        rows.append({"lead_h": lead_h, "rmse": rmse, "n": error.size})
    return rows


def region_statistics(ds):
    rows = []
    for region, south, north, west, east in sorted(REGIONS):
        lat, lon = ds.latitude, ds.longitude
        t2m = ds.t2m.isel(
            latitude=((lat >= south) & (lat <= north)).values,
            longitude=((lon >= west) & (lon <= east)).values,
        ).astype("float64")
        rows.append({"region": region, "avg_c": float(t2m.mean()) - 273.15, "n": t2m.size})
    return rows


def regridding(ds):
    w = weights().to_pandas()
    day = ds.t2m.sel(time=slice(None, "1979-01-10T23")).astype("float64")
    src = day.sel(
        latitude=xr.DataArray(w.src_lat.values, dims="w"),
        longitude=xr.DataArray(w.src_lon.values, dims="w"),
    )
    weighted = pd.DataFrame((src * w.weight.values).values.T, columns=day.time.values)
    sums = weighted.groupby([w.dst_lat.rename("lat"), w.dst_lon.rename("lon")]).sum()
    rows = [
        {"time": pd.Timestamp(time), "lat": lat, "lon": lon, "regridded": float(total)}
        for (lat, lon), by_time in sums.iterrows()
        for time, total in by_time.items()
    ]
    return sorted(rows, key=lambda r: (r["time"], r["lat"], r["lon"]))


ZONAL_MEAN = """
SELECT latitude, avg(t2m) - 273.15 AS air_mean_c, count(*) AS n
FROM era5
GROUP BY latitude
ORDER BY latitude DESC
"""

# TODO: the CASTs work around date_part and interval arithmetic rejecting
# zarr-datafusion's dictionary-encoded time, and give naive timestamps, as
# xarray's are.
CLIMATOLOGY = """
SELECT latitude,
       longitude,
       date_part('hour', CAST(time AS TIMESTAMP)) AS hour,
       avg(t2m) - 273.15                          AS clim_c,
       count(*)                                   AS n
FROM era5
GROUP BY latitude, longitude, date_part('hour', CAST(time AS TIMESTAMP))
ORDER BY latitude DESC, longitude, hour
"""

# NDVI's per-pixel arithmetic, over ERA5's wind components
DERIVED_VARIABLE = """
SELECT CAST(time AS TIMESTAMP) AS time,
       latitude,
       longitude,
       sqrt(u10 * u10 + v10 * v10) AS wind_speed
FROM era5
WHERE time < TIMESTAMP '1979-01-02 00:00:00'
ORDER BY time, latitude DESC, longitude
"""

# xarray-sql's window, a CONUS box over a few weeks, on both sides. The
# climatology is written first: with no statistics on a Zarr scan, DataFusion
# builds a join from the table written first, and holds it in memory.
ANOMALY = """
WITH clim AS (
    SELECT latitude,
           longitude,
           date_part('hour', CAST(time AS TIMESTAMP)) AS hour,
           avg(t2m)                                   AS clim_t
    FROM era5
    WHERE time >= TIMESTAMP '1979-01-15 03:00:00' AND time < TIMESTAMP '1979-03-01 00:00:00'
      AND latitude BETWEEN 25 AND 50 AND longitude BETWEEN 235 AND 290
    GROUP BY latitude, longitude, date_part('hour', CAST(time AS TIMESTAMP))
)
SELECT CAST(a.time AS TIMESTAMP) AS time, a.latitude, a.longitude, a.t2m - c.clim_t AS anomaly
FROM clim c
JOIN era5 a
  ON a.latitude = c.latitude
 AND a.longitude = c.longitude
 AND date_part('hour', CAST(a.time AS TIMESTAMP)) = c.hour
WHERE a.time >= TIMESTAMP '1979-01-15 03:00:00' AND a.time < TIMESTAMP '1979-03-01 00:00:00'
  AND a.latitude BETWEEN 25 AND 50 AND a.longitude BETWEEN 235 AND 290
ORDER BY time, a.latitude DESC, a.longitude
"""

# Persistence: a week of the store as the forecast, against itself `lead` later.
# The valid time is in a subquery: DataFusion drops an ON condition on a
# column from an earlier CROSS JOIN, and in WHERE it isn't a join key. The
# truth first, and the leads before the forecast, as a join is built from its
# first table.
PERSISTENCE_SKILL = """
SELECT f.lead_h, sqrt(avg(power(f.t2m - e.t2m, 2))) AS rmse, count(*) AS n
FROM era5 e
JOIN (
    SELECT l.lead_h, latitude, longitude, t2m, CAST(time AS TIMESTAMP) + l.lead AS valid
    FROM (VALUES (6, INTERVAL '6 hours'), (24, INTERVAL '24 hours'),
                 (72, INTERVAL '72 hours')) AS l(lead_h, lead)
    CROSS JOIN era5
    WHERE time < TIMESTAMP '1979-01-08 00:00:00'
) f
  ON e.latitude = f.latitude AND e.longitude = f.longitude AND CAST(e.time AS TIMESTAMP) = f.valid
GROUP BY f.lead_h
ORDER BY f.lead_h
"""

# The regions first, as the anomaly's climatology is.
REGION_STATISTICS = f"""
SELECT r.region, avg(a.t2m) - 273.15 AS avg_c, count(*) AS n
FROM {REGIONS_SQL.replace("AS regions(", "AS r(")}
JOIN era5 a
  ON a.latitude BETWEEN r.south AND r.north AND a.longitude BETWEEN r.west AND r.east
GROUP BY r.region
ORDER BY r.region
"""

# A grid at a time, so as not to depend on a projection: the warp (9) is
# this over a target grid, then regridding.
REPROJECTION = """
SELECT latitude,
       longitude,
       reproject(longitude, latitude, 'EPSG:4326', 'EPSG:3857')['x'] AS x,
       reproject(longitude, latitude, 'EPSG:4326', 'EPSG:3857')['y'] AS y
FROM era5
WHERE time = TIMESTAMP '1979-01-01 00:00:00'
"""

# Each hour of ten days, as xarray-sql regrids a static field: 8 chunks
REGRIDDING = """
SELECT CAST(s.time AS TIMESTAMP) AS time,
       w.dst_lat                 AS lat,
       w.dst_lon                 AS lon,
       sum(s.t2m * w.weight)     AS regridded
FROM weights w
JOIN era5 s ON s.latitude = w.src_lat AND s.longitude = w.src_lon
WHERE s.time < TIMESTAMP '1979-01-11 00:00:00'
GROUP BY 1, 2, 3
ORDER BY 1, 2, 3
"""

CASES = {
    "zonal-mean": Case(ZONAL_MEAN, (PARTIAL,), zonal_mean),
    "climatology-pancakes": Case(CLIMATOLOGY, (PARTIAL,), climatology),
    # a tile's whole history an invocation, so no group spans two
    "climatology-pencils": Case(
        CLIMATOLOGY,
        (FINAL,),
        climatology,
        chunks=PENCILS,
        chunks_per_invocation=SHAPE[0] // PENCILS[0],
    ),
    "derived-variable": Case(DERIVED_VARIABLE, None, derived_variable, variables=SURFACE),
    "anomaly": Case(ANOMALY, None, anomaly),
    "persistence-skill": Case(PERSISTENCE_SKILL, (PARTIAL,), persistence_skill),
    "region-statistics": Case(REGION_STATISTICS, (PARTIAL,), region_statistics),
    "reprojection": Case(REPROJECTION),
    "regridding": Case(REGRIDDING, (FINAL,), regridding, weights=True),
}
CHECKED = [name for name, case in CASES.items() if case.reference]
UNCHECKED = [name for name, case in CASES.items() if not case.reference]
CUT = [name for name in CHECKED if CASES[name].cut is not None]
REFUSED = [name for name in CHECKED if CASES[name].cut is None]

# What stops a case running in the client.
GAPS = {
    "anomaly": "zarr-datafusion's CardinalityRule replaces the climatology, a FinalPartitioned "
    "aggregate under a partitioned hash join, with one partition",
    "reprojection": "no PROJ UDF: reproject",
}


def gapped(names):
    return [
        pytest.param(n, marks=pytest.mark.xfail(reason=GAPS[n], raises=RuntimeError, strict=True))
        if n in GAPS
        else n
        for n in names
    ]


def connect(tmp_path, case, with_values=True, **settings):
    """A session over the case's store, and its weights if it has any."""
    url = tmp_path / "era5"
    write_group(url, with_values=with_values, chunks=case.chunks, variables=case.variables)
    s = session(
        {"era5": url.as_uri()},
        **LOCAL,
        chunks_per_invocation=case.chunks_per_invocation,
        **settings,
    )
    if case.weights:
        weights = tmp_path / "weights.parquet"
        write_weights(weights)
        s.sql(f"CREATE EXTERNAL TABLE weights STORED AS PARQUET LOCATION '{weights}'")
    return s


def answer(tmp_path, case, **settings):
    stream, reports = connect(tmp_path, case, **settings).sql(case.query)
    return Result(pa.ipc.open_stream(stream).read_all(), json.loads(reports))


def reference(tmp_path, case):
    return case.reference(xr.open_zarr(tmp_path / "era5", consolidated=False))


@pytest.mark.lambda_integration
@pytest.mark.parametrize("name", CUT)
def test_a_case_with_lambdas_matches_xarray(tmp_path, name):
    case = CASES[name]
    result = answer(tmp_path, case)

    assert result.chunks
    rows_close(result.table.to_pylist(), reference(tmp_path, case))


@pytest.mark.parametrize("name", gapped(CHECKED))
def test_a_case_in_the_client_matches_xarray(tmp_path, name):
    case = CASES[name]
    result = answer(tmp_path, case, config=DISABLED)

    assert result.reports == []
    rows_close(result.table.to_pylist(), reference(tmp_path, case))


@pytest.mark.parametrize("name", gapped(UNCHECKED))
def test_a_case_without_a_reference_yet_runs_in_the_client(tmp_path, name):
    result = answer(tmp_path, CASES[name], config=DISABLED)

    assert result.table.num_rows > 0


def kind(fragment):
    if "AggregateExec: mode=Final," in fragment:
        return FINAL
    return PARTIAL if "AggregateExec" in fragment else MAP


def cut(tmp_path, case):
    """What each fragment of the plan runs, sorted."""
    s = connect(tmp_path, case, with_values=False)
    kinds = tuple(sorted(kind(f) for f in s.explain_fragments(case.query)))
    if PARTIAL in kinds:
        stream, _ = s.sql(f"EXPLAIN {case.query}")
        plans = pa.ipc.open_stream(stream).read_all().to_pylist()
        [plan] = [p["plan"] for p in plans if p["plan_type"] == "physical_plan"]
        assert "combiner=partial-reduce" in plan
    return kinds


@pytest.mark.parametrize("name", CUT)
def test_a_case_is_cut_where_expected(tmp_path, name):
    case = CASES[name]

    assert cut(tmp_path, case) == case.cut


@pytest.mark.parametrize("name", REFUSED)
def test_a_case_that_cant_be_cut_is_refused(tmp_path, name):
    with pytest.raises(RuntimeError, match="can't run on the workers"):
        cut(tmp_path, CASES[name])


# How each case should be cut, and what that needs.
SHOULD = {
    "derived-variable": ((MAP,), "map-only fragments returning rows, not writing them"),
    "anomaly": (
        (MAP, PARTIAL),
        "a sliced broadcast: the climatology's slice joined with each probe fragment's chunks",
    ),
    "reprojection": ((MAP,), "UDFs in the worker, and map-only fragments"),
}


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(n, marks=pytest.mark.xfail(reason=needs, strict=True))
        for n, (_, needs) in SHOULD.items()
    ],
)
def test_a_case_is_cut_as_it_should_be(tmp_path, name):
    should, _ = SHOULD[name]

    assert cut(tmp_path, CASES[name]) == should


REGRIDDING_COPY = """
COPY (
    SELECT s.time AS time, w.dst_lat AS lat, w.dst_lon AS lon, sum(s.t2m * w.weight) AS regridded
    FROM weights w
    JOIN era5 s ON s.latitude = w.src_lat AND s.longitude = w.src_lon
    WHERE s.time < TIMESTAMP '1979-01-11 00:00:00'
    GROUP BY 1, 2, 3
) TO '{output}' STORED AS ZARR OPTIONS ('chunks' '31,7,12')
"""


@pytest.mark.lambda_integration
def test_regridding_written_to_zarr_on_the_workers_matches_xarray(tmp_path):
    case = CASES["regridding"]
    output = tmp_path / "regridded.zarr"

    _, reports = connect(tmp_path, case).sql(REGRIDDING_COPY.format(output=output))

    [report] = json.loads(reports)
    assert report["invocations"] > 1
    written = xr.open_zarr(output, consolidated=False).regridded.to_dataframe().reset_index()
    rows = [
        {"time": pd.Timestamp(r.time), "lat": r.lat, "lon": r.lon, "regridded": r.regridded}
        for r in written.itertuples()
    ]
    key = itemgetter("time", "lat", "lon")
    rows_close(sorted(rows, key=key), reference(tmp_path, case))


# The derived variable over ten days, written in the source's chunks
DERIVED_VARIABLE_COPY = """
COPY (
    SELECT time, latitude, longitude, sqrt(u10 * u10 + v10 * v10) AS wind_speed
    FROM era5
    WHERE time < TIMESTAMP '1979-01-11 00:00:00'
) TO '{output}' STORED AS ZARR OPTIONS ('chunks' '31,19,36')
"""


def test_a_derived_variable_copied_to_zarr_is_written_by_the_workers(tmp_path):
    s = connect(tmp_path, CASES["derived-variable"], with_values=False)

    [fragment] = s.explain_fragments(DERIVED_VARIABLE_COPY.format(output=tmp_path / "out.zarr"))

    assert kind(fragment) == MAP
    assert "DataSinkExec" in fragment


@pytest.mark.lambda_integration
def test_a_derived_variable_written_to_zarr_on_the_workers_matches_xarray(tmp_path):
    output = tmp_path / "wind_speed.zarr"

    _, reports = connect(tmp_path, CASES["derived-variable"]).sql(
        DERIVED_VARIABLE_COPY.format(output=output)
    )

    [report] = json.loads(reports)
    assert report["invocations"] > 1
    ds = xr.open_zarr(tmp_path / "era5", consolidated=False).sel(time=slice(None, "1979-01-10"))
    # in float32, as the query computes it
    expected = np.sqrt(ds.u10 * ds.u10 + ds.v10 * ds.v10)
    written = xr.open_zarr(output, consolidated=False).wind_speed
    xr.testing.assert_allclose(written, expected.rename("wind_speed"), rtol=1e-6)


# The anomaly as two statements: its climatology CTE written to Zarr first,
# then the query joined with that store, summarised by latitude, as its rows
# would be too many to return.
def anomaly_window(table, box=True):
    window = (
        f"{table}time >= TIMESTAMP '1979-01-15 03:00:00' "
        f"AND {table}time < TIMESTAMP '1979-03-01 00:00:00'"
    )
    if box:
        window += f" AND {table}latitude BETWEEN 25 AND 50 AND {table}longitude BETWEEN 235 AND 290"
    return window


STORED_CLIMATOLOGY = """
COPY (
    SELECT latitude, longitude, date_part('hour', CAST(time AS TIMESTAMP)) AS hour,
           avg(t2m) AS clim_t
    FROM era5
    WHERE {window}
    GROUP BY 1, 2, 3
) TO '{output}' STORED AS ZARR OPTIONS ('chunks' '24,{chunks}')
"""

# the stored climatology first, as the smaller side is built from
ANOMALY_BY_LATITUDE = f"""
SELECT a.latitude,
       sum(power(a.t2m - c.clim_t, 2)) AS squares,
       min(a.t2m - c.clim_t)           AS low,
       max(a.t2m - c.clim_t)           AS high,
       count(*)                        AS n
FROM clim c
JOIN era5 a
  ON a.latitude = c.latitude
 AND a.longitude = c.longitude
 AND date_part('hour', CAST(a.time AS TIMESTAMP)) = c.hour
WHERE {anomaly_window("a.")}
GROUP BY 1
ORDER BY 1 DESC
"""


def anomaly_by_latitude(ds):
    t2m = ds.t2m.sel(
        time=slice("1979-01-15T03", "1979-02-28T23"),
        latitude=slice(50, 25),
        longitude=slice(235, 290),
    ).astype("float64")
    by_hour = t2m.groupby("time.hour")
    anomaly = by_hour - by_hour.mean()
    cells = ["time", "longitude"]
    stats = {
        "squares": (anomaly**2).sum(cells),
        "low": anomaly.min(cells),
        "high": anomaly.max(cells),
        "n": anomaly.count(cells),
    }
    return [
        {"latitude": float(lat), **{k: v.item(i) for k, v in stats.items()}}
        for i, lat in enumerate(anomaly.latitude.values)
    ]


def stored_climatology(tmp_path, chunks, box=True):
    """A session over a store chunked as `chunks`, and its climatology, written to Zarr."""
    era5 = tmp_path / "era5"
    write_group(era5, chunks=chunks)
    clim = tmp_path / "clim.zarr"
    query = STORED_CLIMATOLOGY.format(
        window=anomaly_window("", box), output=clim, chunks=f"{chunks[1]},{chunks[2]}"
    )
    session({"era5": era5.as_uri()}, **LOCAL, config=DISABLED).sql(query)
    tables = {"era5": era5.as_uri(), "clim": clim.as_uri()}
    # the window is a few chunks, so a few an invocation to deal it out
    return session(tables, **LOCAL, chunks_per_invocation=2)


def test_a_join_with_a_stored_climatology_reads_both_stores_in_one_fragment(tmp_path):
    s = stored_climatology(tmp_path, PENCILS)

    [fragment] = s.explain_fragments(ANOMALY_BY_LATITUDE)

    assert kind(fragment) == PARTIAL
    assert "HashJoinExec: mode=CollectLeft" in fragment
    assert fragment.count("ZarrExec") == 2


@pytest.mark.lambda_integration
@pytest.mark.parametrize("chunks", [PENCILS, CHUNKS], ids=["pencils", "pancakes"])
def test_a_join_with_a_stored_climatology_matches_xarray(tmp_path, chunks):
    s = stored_climatology(tmp_path, chunks)

    stream, reports = s.sql(ANOMALY_BY_LATITUDE)

    [report] = json.loads(reports)
    assert report["invocations"] > 1
    rows = pa.ipc.open_stream(stream).read_all().to_pylist()
    rows_close(rows, anomaly_by_latitude(xr.open_zarr(tmp_path / "era5", consolidated=False)))


# Over the whole globe, as the output of a write split by tile must cover each
# axis it is split on
ANOMALY_COPY = f"""
COPY (
    SELECT a.time, a.latitude, a.longitude, a.t2m - c.clim_t AS anomaly
    FROM clim c
    JOIN era5 a
      ON a.latitude = c.latitude
     AND a.longitude = c.longitude
     AND date_part('hour', CAST(a.time AS TIMESTAMP)) = c.hour
    WHERE {anomaly_window("a.", box=False)}
) TO '{{output}}' STORED AS ZARR OPTIONS ('chunks' '{{chunks}}')
"""


def test_an_anomaly_copied_to_zarr_is_written_by_the_workers_joining_both_stores(tmp_path):
    s = stored_climatology(tmp_path, PENCILS, box=False)
    query = ANOMALY_COPY.format(output=tmp_path / "anomaly.zarr", chunks="310,5,6")

    [fragment] = s.explain_fragments(query)

    assert kind(fragment) == MAP
    assert "DataSinkExec" in fragment
    assert fragment.count("ZarrExec") == 2


@pytest.mark.lambda_integration
@pytest.mark.parametrize(
    ("chunks", "output_chunks"),
    [(PENCILS, "310,5,6"), (CHUNKS, "31,19,36")],
    ids=["pencils", "pancakes"],
)
def test_an_anomaly_written_to_zarr_on_the_workers_matches_xarray(tmp_path, chunks, output_chunks):
    s = stored_climatology(tmp_path, chunks, box=False)
    output = tmp_path / "anomaly.zarr"

    _, reports = s.sql(ANOMALY_COPY.format(output=output, chunks=output_chunks))

    [report] = json.loads(reports)
    assert report["invocations"] > 1
    ds = xr.open_zarr(tmp_path / "era5", consolidated=False)
    t2m = ds.t2m.sel(time=slice("1979-01-15T03", "1979-02-28T23")).astype("float64")
    by_hour = t2m.groupby("time.hour")
    expected = (by_hour - by_hour.mean()).drop_vars("hour").rename("anomaly")
    written = xr.open_zarr(output, consolidated=False).anomaly
    xr.testing.assert_allclose(written, expected, rtol=1e-9, atol=1e-9)


@pytest.mark.lambda_integration
@pytest.mark.parametrize(
    ("start", "end", "per_invocation", "invocations"),
    [
        # a tile's window spans pencils, which two at a time span invocations
        ("1979-01-15T03", "1979-03-01T00", 2, None),
        # two pencils, from the 310th hour: a pencil, so an output chunk, an invocation
        ("1979-01-13T22", "1979-02-08T18", 1, 4 * 6 * 2),
    ],
    ids=["unaligned", "aligned"],
)
def test_a_write_over_pencils_split_through_time_matches_xarray(
    tmp_path, start, end, per_invocation, invocations
):
    era5 = tmp_path / "era5"
    write_group(era5, chunks=PENCILS, variables=SURFACE)
    s = session({"era5": era5.as_uri()}, **LOCAL, chunks_per_invocation=per_invocation)
    output = tmp_path / "wind_speed.zarr"
    window = f"time >= TIMESTAMP '{start}:00:00' AND time < TIMESTAMP '{end}:00:00'"
    query = DERIVED_VARIABLE_COPY.replace(
        "WHERE time < TIMESTAMP '1979-01-11 00:00:00'", f"WHERE {window}"
    ).format(output=output)

    _, reports = s.sql(query.replace("'31,19,36'", "'310,5,6'"))

    [report] = json.loads(reports)
    if invocations:
        assert report["invocations"] == invocations
    last = np.datetime64(end) - np.timedelta64(1, "h")
    ds = xr.open_zarr(era5, consolidated=False).sel(time=slice(start, last))
    expected = np.sqrt(ds.u10 * ds.u10 + ds.v10 * ds.v10).rename("wind_speed")
    written = xr.open_zarr(output, consolidated=False).wind_speed
    xr.testing.assert_allclose(written, expected, rtol=1e-6)
