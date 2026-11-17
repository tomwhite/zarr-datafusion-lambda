"""xarray-sql's geospatial cases (https://xql.systems/xarray-sql/latest/geospatial/)
over a small store chunked as ERA5 is, each answered fanned out, in the
client, and by xarray, and cut where expected."""

import json
from collections.abc import Callable
from dataclasses import dataclass

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
    # sorted, as `cut` gives them
    cut: tuple[str, ...] = ()
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
    day = ds.t2m.sel(time=FIRST_DAY).astype("float64")
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
# column from an earlier CROSS JOIN, and in WHERE it isn't a join key.
PERSISTENCE_SKILL = """
SELECT f.lead_h, sqrt(avg(power(f.t2m - e.t2m, 2))) AS rmse, count(*) AS n
FROM (
    SELECT l.lead_h, latitude, longitude, t2m, CAST(time AS TIMESTAMP) + l.lead AS valid
    FROM era5
    CROSS JOIN (VALUES (6, INTERVAL '6 hours'), (24, INTERVAL '24 hours'),
                       (72, INTERVAL '72 hours')) AS l(lead_h, lead)
    WHERE time < TIMESTAMP '1979-01-08 00:00:00'
) f
JOIN era5 e
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

# Each hour of a day, as xarray-sql regrids a static field
REGRIDDING = """
SELECT CAST(s.time AS TIMESTAMP) AS time,
       w.dst_lat                 AS lat,
       w.dst_lon                 AS lon,
       sum(s.t2m * w.weight)     AS regridded
FROM weights w
JOIN era5 s ON s.latitude = w.src_lat AND s.longitude = w.src_lon
WHERE s.time < TIMESTAMP '1979-01-02 00:00:00'
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
    "derived-variable": Case(DERIVED_VARIABLE, (), derived_variable, variables=SURFACE),
    "anomaly": Case(ANOMALY, (PARTIAL,), anomaly),
    "persistence-skill": Case(PERSISTENCE_SKILL, (), persistence_skill),
    "region-statistics": Case(REGION_STATISTICS, (PARTIAL,), region_statistics),
    "reprojection": Case(REPROJECTION),
    "regridding": Case(REGRIDDING, (), regridding, weights=True),
}
CHECKED = [name for name, case in CASES.items() if case.reference]
UNCHECKED = [name for name, case in CASES.items() if not case.reference]

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
@pytest.mark.parametrize("combine", ["true", "false"])
@pytest.mark.parametrize("name", CHECKED)
def test_a_case_with_lambdas_matches_xarray(tmp_path, name, combine):
    case = CASES[name]
    result = answer(tmp_path, case, config={"lambda.combine": combine})

    assert bool(result.chunks) == bool(case.cut)
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


@pytest.mark.parametrize("name", CHECKED)
def test_a_case_is_cut_where_expected(tmp_path, name):
    case = CASES[name]

    assert cut(tmp_path, case) == case.cut


# How each case should be cut, and what that needs.
SHOULD = {
    "derived-variable": ((MAP,), "map-only fragments"),
    "anomaly": (
        (MAP, PARTIAL),
        "a sliced broadcast: the climatology's slice joined with each probe fragment's chunks",
    ),
    "persistence-skill": (
        (PARTIAL,),
        "two-scan fragments, with the truth's chunks derived from the join predicate",
    ),
    "regridding": (
        (FINAL,),
        "a broadcast join of a small table, then the final aggregate on the workers, "
        "as each group's time is in one chunk",
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
