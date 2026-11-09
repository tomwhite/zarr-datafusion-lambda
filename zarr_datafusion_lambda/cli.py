"""The `zarr-datafusion-lambda` command line."""

import click
import map_lambda.deploy

from zarr_datafusion_lambda.deploy import SPEC


@click.group()
def cli() -> None:
    """Query ERA5 Zarr with SQL on AWS Lambda."""


for _command in map_lambda.deploy.commands(SPEC):
    cli.add_command(_command)
