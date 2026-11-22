"""
SQL over Zarr tables. Each partial aggregate over a chunkwise Zarr scan runs on
`fragment` lambdas, a run of chunks each, and the rest of the query here. A
query that would scan or write Zarr here is refused, with why, unless
`lambda.enabled` is false, which runs all of it here.

    era5 = sql.Session({"era5": {"url": URL, "group": "single/spatial"}})
    result = era5.sql(query, progress=True)
    result.to_pandas()
    print(result.stats())
    print(result.cost())
"""

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import unquote, urlparse

import boto3
import pyarrow as pa
from map_lambda import DISPATCHER
from map_lambda import Session as Lambdas
from map_lambda import cost as lambda_cost
from map_lambda.stats import summarize_dispatchers, summarize_invocations
from tqdm.auto import tqdm

from zarr_datafusion_lambda._native import Session as NativeSession
from zarr_datafusion_lambda.deploy import PROJECT

WORKER = "fragment"

# keep in sync with `COMBINER` in rust/split
COMBINER = "partial-reduce"


def location(url: str) -> str:
    """A table location as zarr-datafusion takes it: a path for `file://`, else the URL."""
    parsed = urlparse(url)
    return unquote(parsed.path) if parsed.scheme == "file" else url


Table = str | dict[str, str]
"""A Zarr group's URL, or a dict of a table's `url` and its options:
zarr-datafusion's, such as the `group` to read in an icechunk repo, or a
`format` other than Zarr, such as `parquet`."""


def _table(table: Table) -> tuple[str, dict[str, str]]:
    if isinstance(table, str):
        return location(table), {}
    options = dict(table)
    return location(options.pop("url")), options


def export_credentials() -> None:
    """
    Put the credentials the `aws` CLI would use (a profile, SSO, CloudShell's) in the
    `AWS_*` variables, the only ones zarr-datafusion's S3 stores read, unless they're
    set already. Once per process, as changing the environment while queries run
    could race with the stores reading it.
    """
    global _exported
    if _exported or "AWS_ACCESS_KEY_ID" in os.environ:
        return
    _exported = True
    credentials = boto3.Session().get_credentials()
    if credentials is None:
        return
    frozen = credentials.get_frozen_credentials()
    os.environ["AWS_ACCESS_KEY_ID"] = frozen.access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = frozen.secret_key
    if frozen.token:
        os.environ["AWS_SESSION_TOKEN"] = frozen.token


_exported = False


def session(tables: dict[str, Table], **settings: Any) -> NativeSession:
    """
    The native session over `tables`, by name, calling this project's lambdas,
    which [`Session`][zarr_datafusion_lambda.sql.Session] runs queries
    through. `settings` are
    `endpoint_url`, for a local `cargo lambda watch`, `concurrency`,
    `dispatchers` (by default the fewest that hold `concurrency`),
    `chunks_per_invocation`, `chunks_in_flight` (how many of those a
    worker runs at once, by default 2), `return_stats`, `target_partitions`,
    `on_progress`, called with how many of a fan-out's invocations have
    succeeded and how many it has: with none before it invokes, then as they
    arrive, and `config`, DataFusion's options by key, such as
    `{"lambda.enabled": "false"}` to run every query here, as SQL's `SET`
    would, or `{"lambda.combine": "false"}` to have every worker's partial rows
    sent here rather than merged in the dispatchers. A fan-out over an S3 bucket
    in another region than the workers is refused before it starts.
    """
    export_credentials()
    return NativeSession(
        {name: _table(table) for name, table in tables.items()}, project=PROJECT, **settings
    )


@dataclass
class Result:
    """A query's rows, and what its fan-outs did on the lambdas."""

    table: pa.Table
    reports: list[dict[str, Any]]
    """What each fan-out did: none when the query read no chunks, or ran here."""
    wall_s: float | None = None
    """How long the query took, from the client's side."""
    _reported: dict[str, list[dict[str, Any]]] | None = field(default=None, repr=False)

    @property
    def chunks(self) -> int | None:
        return sum(r["chunks"] for r in self.reports) if self.reports else None

    @property
    def invocations(self) -> int:
        return sum(r["invocations"] for r in self.reports)

    def to_pandas(self) -> Any:
        return self.table.to_pandas()

    @property
    def invocation_stats(self) -> list[dict[str, Any]]:
        """
        map-lambda's record of each worker invocation, as in
        `map_lambda.Session.invocation_stats`, with its `fan_out` and the S3
        requests its worker made, `s3_gets` and `s3_lists`. With the
        dispatchers merging workers' results, a merged state's requests count
        against its first invocation.
        """
        records = []
        for fan_out, report in enumerate(self.reports):
            requests = {}
            for worker in report["workers"]:
                index = worker["indexes"][0] if "indexes" in worker else worker["index"]
                requests[index] = worker["s3_requests"]
            for r in report["invocation_stats"]:
                made = requests.get(r["index"], {"gets": 0, "lists": 0})
                records.append(
                    {**r, "fan_out": fan_out, "s3_gets": made["gets"], "s3_lists": made["lists"]}
                )
        return records

    @property
    def dispatcher_stats(self) -> list[dict[str, Any]]:
        """map-lambda's record of each dispatcher invocation, with its `fan_out`."""
        return [
            {**d, "fan_out": fan_out}
            for fan_out, report in enumerate(self.reports)
            for d in report["dispatcher_stats"]
        ]

    def stats(self) -> str:
        """What the query's fan-outs did, added together, as a few lines to print."""
        if not self.reports:
            return "no fan-outs: the query read no chunks, or ran in the client"
        invocations = self.invocation_stats
        if not invocations:
            return "no stats: the query ran with return_stats=False"
        gets = sum(r["s3_gets"] for r in invocations)
        lists = sum(r["s3_lists"] for r in invocations)
        return "\n".join(
            [
                f"chunks:         {self.chunks:,} in {len(self.reports)} fan-out(s)",
                summarize_invocations(invocations, elapsed=self.wall_s),
                summarize_dispatchers(self.dispatcher_stats),
                f"S3 requests:    {gets:,} GETs, {lists:,} LISTs from the workers",
            ]
        )

    def cost(self, *, wait_s: float = 300, progress: bool = True) -> lambda_cost.Cost:
        """
        What the query's fan-outs cost at list prices, added together, from
        the REPORT lines Lambda writes to CloudWatch for each invocation of the
        workers and the dispatchers, with the S3 requests the workers made.
        CloudWatch takes some seconds to ingest them, so this waits, up to
        `wait_s`, printing how many it has found if `progress`. Needs
        permission to run CloudWatch Logs Insights queries.
        """
        if not self.reports:
            raise ValueError("no fan-outs to cost: the query read no chunks, or ran in the client")
        if not self.invocation_stats:
            raise ValueError("no stats to cost: run the query with return_stats=True")
        on_progress = print if progress else None
        deployed = Lambdas(project=PROJECT)
        workers, dispatchers = [], []
        for fan_out, report in enumerate(self.reports):
            combined = any("indexes" in w for w in report["workers"])
            dispatcher = f"{DISPATCHER}-{COMBINER}" if combined else DISPATCHER
            workers += lambda_cost.fetch_reports(
                [r for r in self.invocation_stats if r["fan_out"] == fan_out],
                deployed.function_name(WORKER),
                wait_s=wait_s,
                on_progress=on_progress,
            )
            dispatchers += lambda_cost.fetch_reports(
                [d for d in self.dispatcher_stats if d["fan_out"] == fan_out],
                deployed.function_name(dispatcher),
                wait_s=wait_s,
                on_progress=on_progress,
            )
        self._reported = {"workers": workers, "dispatchers": dispatchers}
        priced = workers + dispatchers
        expected = len(self.invocation_stats) + len(self.dispatcher_stats)
        return lambda_cost.estimate_cost(priced, missing=expected - len(priced))

    def timeline(self) -> Any:
        """
        A matplotlib figure of when each worker invocation was invoked,
        returned, and, once [`cost`][zarr_datafusion_lambda.sql.Result.cost]
        has found their REPORT lines, ran. Needs map-lambda's `diagnostics`
        extra.
        """
        from map_lambda.timeline import generate_timeline

        ran = {w["req_id"]: w for w in (self._reported or {}).get("workers", [])}
        records = [ran.get(r["req_id"], r) for r in self.invocation_stats]
        return generate_timeline(records, title=f"{self.chunks:,} chunks")


class Session:
    """
    Tables to query by SQL, and how queries over them run on the deployed
    lambdas: `settings` as [`run`][zarr_datafusion_lambda.sql.run] takes them.
    """

    def __init__(self, tables: dict[str, Table], **settings: Any) -> None:
        self.tables = tables
        self.settings = settings

    def sql(self, query: str, *, progress: bool = False) -> Result:
        """Run `query`, with a progress bar on stderr for each fan-out if `progress`."""
        return run(query, self.tables, progress=progress, **self.settings)

    def explain(self, query: str) -> str:
        """`query`'s physical plan, then the fragment each of its fan-outs runs on the workers."""
        return explain(query, self.tables, **self.settings)


def explain(query: str, tables: dict[str, Table], **settings: Any) -> str:
    """`query`'s physical plan, then the fragment each of its fan-outs runs on the workers."""
    s = session(tables, **settings)
    stream, _ = s.sql(f"EXPLAIN {query}")
    rows = pa.ipc.open_stream(stream).read_all().to_pylist()
    [plan] = [r["plan"] for r in rows if r["plan_type"] == "physical_plan"]
    return "\n".join([plan, *(f"fragment:\n{f}" for f in s.explain_fragments(query))])


class _Progress:
    """A bar for each fan-out in turn, of its invocations that have succeeded."""

    def __init__(self) -> None:
        self.bar: tqdm | None = None

    def __call__(self, done: int, total: int) -> None:
        if done == 0:
            self.close()
            # no ETA: invocations finish in waves, so the rate swings
            self.bar = tqdm(
                total=total,
                bar_format="{n_fmt}/{total_fmt} invocations |{bar}| {elapsed}",
                disable=None,
            )
        elif self.bar is not None:
            self.bar.update(done - self.bar.n)

    def close(self) -> None:
        if self.bar is not None:
            self.bar.close()
            self.bar = None


def run(query: str, tables: dict[str, Table], progress: bool = False, **settings: Any) -> Result:
    """
    Run `query` over `tables`, with `session`'s settings, and with `progress`
    a bar on stderr for each fan-out, when it is a terminal. `return_stats` is
    on unless set off, so that the result can say what it did and what it
    cost.
    """
    settings.setdefault("return_stats", True)
    bar = _Progress() if progress else None
    if bar is not None:
        settings["on_progress"] = bar
    started = time.monotonic()
    try:
        stream, reports = session(tables, **settings).sql(query)
    finally:
        if bar is not None:
            bar.close()
    table = pa.ipc.open_stream(stream).read_all()
    return Result(table, json.loads(reports), wall_s=time.monotonic() - started)
