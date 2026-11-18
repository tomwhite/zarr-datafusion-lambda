"""Small Zarr stores chunked as Earthmover's ERA5 is, and what the tests share over them."""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import zarr
from map_lambda import LOCAL_ENDPOINT_URL

LOCAL = {"endpoint_url": LOCAL_ENDPOINT_URL, "concurrency": 100}
DISABLED = {"lambda.enabled": "false"}

VARIABLE = "t2m"
# ERA5's other surface variables, for the cases that zip several
SURFACE = (VARIABLE, "u10", "v10", "d2m")
DIMS = ("time", "latitude", "longitude")
TIME_UNITS = "hours since 1979-01-01 00:00:00"
TIME_START = np.datetime64("1979-01-01T00", "h")

# 60 chunks of 31 hours, January to mid-March 1979 on a 10° grid
SHAPE = (31 * 60, 19, 36)
# as `single/spatial` is chunked: the globe, a run of hours
CHUNKS = (31, 19, 36)
# as `single/temporal` is chunked: a run of hours of a tile, here 6 runs of 4 by 6 tiles
PENCILS = (310, 5, 6)


def write_group(path, with_values=True, chunks=CHUNKS, variables=(VARIABLE,)):
    """An ERA5-shaped group, hourly from 1979 on a global grid, as xarray writes one."""
    group = zarr.open_group(path, mode="w", zarr_format=3)
    n_time, n_lat, n_lon = SHAPE
    coordinates = {
        "time": np.arange(n_time, dtype="int64"),
        "latitude": 90.0 - 180.0 * np.arange(n_lat) / (n_lat - 1),
        "longitude": 360.0 * np.arange(n_lon) / n_lon,
    }
    for name, values in coordinates.items():
        group.create_array(name, shape=values.shape, dtype=values.dtype, dimension_names=[name])[
            :
        ] = values
    group["time"].attrs.update({"units": TIME_UNITS, "calendar": "standard"})
    values = surface(coordinates["latitude"]) if with_values else {}
    for name in variables:
        array = group.create_array(
            name, shape=SHAPE, chunks=chunks, dtype="float32", dimension_names=DIMS
        )
        if with_values:
            array[:] = values[name]


def surface(latitude):
    # warm at the equator, cold at the poles, plus noise
    profile = 288 - 40 * np.sin(np.radians(latitude)) ** 2
    t2m = (profile[:, None] + np.random.default_rng(0).uniform(-15, 15, SHAPE)).astype("float32")
    rng = np.random.default_rng(1)
    return {
        "t2m": t2m,
        "u10": rng.uniform(-15, 15, SHAPE).astype("float32"),
        "v10": rng.uniform(-15, 15, SHAPE).astype("float32"),
        # the dew point is at most the temperature
        "d2m": t2m - rng.uniform(0, 10, SHAPE).astype("float32"),
    }


# Boxes as (region, south, north, west, east), small enough to travel in the plan
REGIONS = [
    ("arctic", 66.5, 90.0, 0.0, 360.0),
    ("tropics", -23.5, 23.5, 0.0, 360.0),
    ("conus", 25.0, 50.0, 235.0, 290.0),
]
REGIONS_SQL = (
    "(VALUES "
    + ", ".join(f"('{r}', {s}, {n}, {w}, {e})" for r, s, n, w, e in REGIONS)
    + ") AS regions(region, south, north, west, east)"
)

# The 10° grid averaged onto a 30° one
COARSE = (90.0 - 30.0 * np.arange(7), 30.0 * np.arange(12))


def weights():
    """
    Regridding weights as `(dst_lat, dst_lon, src_lat, src_lon, weight)`: each
    coarse cell the area-weighted mean of the 3 by 3 cells around it, 2 by 3 at
    the poles, so each fine cell is in exactly one.
    """
    rows = []
    for dst_lat in COARSE[0]:
        for dst_lon in COARSE[1]:
            cells = [
                (lat, lon % 360)
                for lat in (dst_lat - 10, dst_lat, dst_lat + 10)
                if -90 <= lat <= 90
                for lon in (dst_lon - 10, dst_lon, dst_lon + 10)
            ]
            areas = np.cos(np.radians([lat for lat, _ in cells]))
            for (lat, lon), weight in zip(cells, areas / areas.sum(), strict=True):
                rows.append((dst_lat, dst_lon, lat, lon, weight))
    names = ("dst_lat", "dst_lon", "src_lat", "src_lon", "weight")
    columns = zip(*rows, strict=True)
    return pa.table({n: pa.array(c, pa.float64()) for n, c in zip(names, columns, strict=True)})


def write_weights(path):
    pq.write_table(weights(), path)


def rows_close(actual, expected):
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected, strict=True):
        assert a.keys() == e.keys()
        for k in a:
            if isinstance(e[k], float):
                assert a[k] == pytest.approx(e[k], rel=1e-9), k
            else:
                assert a[k] == e[k], k
