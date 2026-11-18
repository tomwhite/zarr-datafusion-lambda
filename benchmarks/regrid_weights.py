"""
First-order conservative regridding weights from ERA5's 0.25° grid onto a 1°
grid, as a Parquet table of `(dst_lat, dst_lon, src_lat, src_lon, weight)`:
each 1° cell's value is the area-weighted mean of the 0.25° cells it
overlaps. The workers read it from S3 in `regridding.py`, so write it to a
bucket they're allowed to read, in their region:

\b
    uv run python benchmarks/regrid_weights.py s3://<bucket>/era5-1deg-weights.parquet

About 1.6M rows.
"""

import click
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# ERA5's grid, and the target's, as cell centres, north to south from 90°N and
# east from 0°; a pole's cell is half a cell, from the pole to its edge
SOURCE = (90 - 0.25 * np.arange(721), 0.25 * np.arange(1440))
TARGET = (90 - 1.0 * np.arange(181), 1.0 * np.arange(360))


def latitude_weights() -> np.ndarray:
    """A `(181, 721)` matrix: each target row's share of each source row, by area."""
    (source, _), (target, _) = SOURCE, TARGET

    def edges(centres: np.ndarray, half: float) -> tuple[np.ndarray, np.ndarray]:
        return np.clip(centres - half, -90, 90), np.clip(centres + half, -90, 90)

    s_low, s_high = edges(source, 0.125)
    t_low, t_high = edges(target, 0.5)
    sin = lambda degrees: np.sin(np.radians(degrees))  # noqa: E731
    low = np.maximum(s_low[None, :], t_low[:, None])
    high = np.minimum(s_high[None, :], t_high[:, None])
    overlap = np.where(high > low, sin(high) - sin(low), 0.0)
    return overlap / (sin(t_high) - sin(t_low))[:, None]


def longitude_weights() -> np.ndarray:
    """A `(360, 1440)` matrix: each target column's share of each source column."""
    (_, source), (_, target) = SOURCE, TARGET
    # the source's offset from each target centre, across the meridian too
    offset = (source[None, :] - target[:, None] + 180) % 360 - 180
    # in degrees, so a share of the 1° target
    overlap = np.minimum(offset + 0.125, 0.5) - np.maximum(offset - 0.125, -0.5)
    return np.clip(overlap, 0, None)


def regrid(values: np.ndarray) -> np.ndarray:
    """`values` on the source grid, `(..., 721, 1440)`, on the target grid."""
    return latitude_weights() @ values @ longitude_weights().T


def table() -> pa.Table:
    """Every non-zero weight, as rows of the separable weights' product."""
    lat, lon = latitude_weights(), longitude_weights()
    (src_lat, src_lon), (dst_lat, dst_lon) = SOURCE, TARGET
    lat_t, lat_s = np.nonzero(lat)
    lon_t, lon_s = np.nonzero(lon)
    # every pairing of a latitude weight with a longitude weight
    i, j = np.meshgrid(np.arange(lat_t.size), np.arange(lon_t.size), indexing="ij")
    i, j = i.ravel(), j.ravel()
    return pa.table(
        {
            "dst_lat": dst_lat[lat_t[i]],
            "dst_lon": dst_lon[lon_t[j]],
            "src_lat": src_lat[lat_s[i]],
            "src_lon": src_lon[lon_s[j]],
            "weight": lat[lat_t[i], lat_s[i]] * lon[lon_t[j], lon_s[j]],
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
