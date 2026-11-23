# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "click>=8.1",
#     "numpy",
#     "pyarrow",
#     "pyproj",
# ]
# ///
"""
Bilinear weights from ERA5's 0.25° grid onto Europe's Lambert azimuthal
equal-area grid (EPSG:3035) at 25 km, as a Parquet table of
`(dst_y, dst_x, src_lat, src_lon, weight)`: each target cell's value is
interpolated from the four ERA5 cells around where its centre falls. The
target grid is projected onto ERA5's with pyproj, once, as it is the same
every hour. The workers read the table from S3 in `warp.py`, so write it to a
bucket they're allowed to read, in their region:

\b
    uv run benchmarks/warp_weights.py s3://<bucket>/era5-laea-weights.parquet

About 350k rows.
"""

import click
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from regrid_weights import SOURCE

CRS = "EPSG:3035"
RESOLUTION_M = 25_000
# EPSG:3035's area of use, projected, out to whole cells
X_EDGES = (-150_000, 7_825_000)
Y_EDGES = (175_000, 6_975_000)
# cell centres, ascending, as the output's axes are
TARGET = tuple(
    np.arange(low + RESOLUTION_M / 2, high, RESOLUTION_M, dtype="float64")
    for low, high in (Y_EDGES, X_EDGES)
)


def target_points() -> tuple[np.ndarray, np.ndarray]:
    """Each target cell's centre as `(y, x)` arrays of latitude and longitude, east from 0°."""
    from pyproj import Transformer

    y, x = np.meshgrid(*TARGET, indexing="ij")
    lon, lat = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True).transform(x, y)
    return lat, lon % 360


def stencil() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Each target cell's four ERA5 cells, as `(y, x, 4)` arrays of their row and
    column in ERA5's grid, and their weights.
    """
    lat, lon = target_points()
    n_lat, n_lon = (axis.size for axis in SOURCE)
    # ERA5's rows run south from 90°N, its columns east from 0°, both 0.25° apart
    row, col = (90 - lat) / 0.25, lon / 0.25
    r0 = np.minimum(np.floor(row), n_lat - 2).astype(int)
    c0 = np.floor(col).astype(int)
    dr, dc = row - r0, col - c0
    rows = np.stack([r0, r0, r0 + 1, r0 + 1], axis=-1)
    # a cell east of the last column takes the first as its eastern neighbour
    cols = np.stack([c0, c0 + 1, c0, c0 + 1], axis=-1) % n_lon
    weights = np.stack([(1 - dr) * (1 - dc), (1 - dr) * dc, dr * (1 - dc), dr * dc], axis=-1)
    return rows, cols, weights


def table() -> pa.Table:
    rows, cols, weights = stencil()
    y, x = np.meshgrid(*TARGET, indexing="ij")
    src_lat, src_lon = SOURCE
    return pa.table(
        {
            "dst_y": np.repeat(y.ravel(), 4),
            "dst_x": np.repeat(x.ravel(), 4),
            "src_lat": src_lat[rows].ravel(),
            "src_lon": src_lon[cols].ravel(),
            "weight": weights.ravel(),
        }
    )


@click.command(help=__doc__)
@click.argument("output")
def main(output: str) -> None:
    weights = table()
    pq.write_table(weights, output)
    click.echo(f"{weights.num_rows:,} weights written to {output}")


if __name__ == "__main__":
    main()
