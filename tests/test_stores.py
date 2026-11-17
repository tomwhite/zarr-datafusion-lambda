import pyarrow as pa
import pytest
import xarray as xr

from tests.stores import (
    COARSE,
    DISABLED,
    LOCAL,
    REGIONS,
    REGIONS_SQL,
    SHAPE,
    SURFACE,
    write_group,
    write_weights,
)
from zarr_datafusion_lambda.sql import session


def rows(s, query):
    stream, _ = s.sql(query)
    return pa.ipc.open_stream(stream).read_all().to_pylist()


def test_a_group_of_several_variables_reads_as_one_table(tmp_path):
    write_group(tmp_path, variables=SURFACE)
    s = session({"era5": tmp_path.as_uri()}, **LOCAL, config=DISABLED)

    means = ", ".join(f"avg({v}) AS {v}" for v in SURFACE)
    [actual] = rows(s, f"SELECT {means}, max(d2m - t2m) AS spread, count(*) AS n FROM era5")

    ds = xr.open_zarr(tmp_path, consolidated=False)
    for v in SURFACE:
        assert actual[v] == pytest.approx(float(ds[v].astype("float64").mean()), rel=1e-9), v
    assert actual["spread"] <= 0
    assert actual["n"] == SHAPE[0] * SHAPE[1] * SHAPE[2]


def test_the_weights_average_each_coarse_cell_from_fine_cells_in_no_other(tmp_path):
    write_weights(tmp_path / "weights.parquet")
    s = session({}, **LOCAL, config=DISABLED)
    rows(
        s, f"CREATE EXTERNAL TABLE weights STORED AS PARQUET LOCATION '{tmp_path}/weights.parquet'"
    )

    totals = rows(s, "SELECT sum(weight) AS total FROM weights GROUP BY dst_lat, dst_lon")
    uses = rows(s, "SELECT count(*) AS n FROM weights GROUP BY src_lat, src_lon")

    assert len(totals) == len(COARSE[0]) * len(COARSE[1])
    assert all(t["total"] == pytest.approx(1) for t in totals)
    assert len(uses) == SHAPE[1] * SHAPE[2]
    assert all(u["n"] == 1 for u in uses)


def test_the_regions_read_as_a_table():
    s = session({}, **LOCAL, config=DISABLED)

    regions = rows(s, f"SELECT region FROM {REGIONS_SQL}")

    assert [r["region"] for r in regions] == [r for r, *_ in REGIONS]
