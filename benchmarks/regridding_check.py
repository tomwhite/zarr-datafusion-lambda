# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "boto3",
#     "click>=8.1",
#     "icechunk>=2",
#     "numcodecs[pcodec]",
#     "numpy",
#     "obstore",
#     "pyarrow",
#     "xarray",
#     "zarr>=3.1.3",
# ]
# ///
"""
A `regridding.py` output, against xarray regridding the same hours from the
run's repo in one process, with the same weights as matrices:

\b
    uv run benchmarks/regridding_check.py results/regridding-....json

By default the run's first 31 days at most; `--days` picks how many. It
reads a day at a time: a month of Earthmover's is about 3 GB, read
anonymously from us-east-1, so it is quickest run from there. The output is
read with the credentials the `aws` CLI would use.
"""

import json
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import click
import icechunk
import numpy as np
import xarray as xr
import zarr
from demo_check import TOLERANCE_K, BarePCodec, storage
from regrid_weights import TARGET, regrid


def zarr_store(output: str) -> str | zarr.abc.store.Store:
    parsed = urlparse(output)
    if parsed.scheme != "s3":
        return output
    import boto3
    from botocore.exceptions import ClientError
    from obstore.auth.boto3 import Boto3CredentialProvider
    from obstore.store import S3Store

    # S3 names the bucket's region even in refusing a request sent elsewhere.
    try:
        response = boto3.client("s3").head_bucket(Bucket=parsed.netloc)
    except ClientError as error:
        response = error.response
    region = response["ResponseMetadata"]["HTTPHeaders"]["x-amz-bucket-region"]
    credentials = Boto3CredentialProvider(boto3.Session(region_name=region))
    store = S3Store(parsed.netloc, prefix=parsed.path.strip("/"), credential_provider=credentials)
    return zarr.storage.ObjectStore(store, read_only=True)


@click.command(help=__doc__)
@click.argument("results", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--days", default=31, show_default=True, help="How many of the run's days.")
def main(results: Path, days: int) -> None:
    run = json.loads(results.read_text())
    start = np.datetime64(datetime.fromisoformat(run["start"]), "h")
    end = min(
        np.datetime64(datetime.fromisoformat(run["end"]), "h"),
        start + np.timedelta64(24 * days, "h"),
    )

    written = xr.open_zarr(zarr_store(run["output"]), consolidated=False).t2m
    lat, lon = TARGET
    if not (
        np.array_equal(np.sort(written.latitude), np.sort(lat))
        and np.array_equal(np.sort(written.longitude), np.sort(lon))
    ):
        raise click.ClickException("the output's grid isn't the 1° grid")
    # the output's axes ascend; the weights' latitude descends, as ERA5's does
    written = written.sel(latitude=lat, longitude=lon)

    zarr.registry.register_codec("pcodec", BarePCodec)
    session = icechunk.Repository.open(storage(run["url"])).readonly_session("main")
    ds = xr.open_zarr(session.store, group="single/spatial", chunks=None, consolidated=False)
    t2m = ds.t2m.sel(valid_time=slice(start, end - np.timedelta64(1, "h")))

    worst = (0.0, None)
    for day in range(0, t2m.sizes["valid_time"], 24):
        hours = t2m.isel(valid_time=slice(day, day + 24))
        expected = regrid(hours.values.astype("float64"))
        actual = written.sel(valid_time=hours.valid_time.values).values
        if np.isnan(actual).any():
            raise click.ClickException(
                f"{int(np.isnan(actual).sum()):,} cells of day {day // 24 + 1} weren't written"
            )
        differences = np.abs(actual - expected)
        at = np.unravel_index(differences.argmax(), differences.shape)
        if differences[at] > worst[0]:
            worst = (
                float(differences[at]),
                (hours.valid_time.values[at[0]], lat[at[1]], lon[at[2]]),
            )
        click.echo(f"  {day // 24 + 1} days checked\r", nl=False)

    hours = t2m.sizes["valid_time"]
    click.echo(f"\n{hours} hours, {hours * lat.size * lon.size:,} cells")
    difference, where = worst
    if where is not None:
        time, at_lat, at_lon = where
        click.echo(f"  largest difference {difference:.2e} K, at {time} {at_lat}°N {at_lon}°E")
    if difference >= TOLERANCE_K:
        raise click.ClickException("the regridded values differ")
    click.secho("all cells as expected", fg="green")


if __name__ == "__main__":
    main()
