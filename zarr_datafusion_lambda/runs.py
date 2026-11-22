"""
A fanned-out query run with its workers' measurements, and reported on: what
it did, what CloudWatch's REPORT lines say it cost, and its timeline.
"""

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from map_lambda import DISPATCHER, Session
from map_lambda import cost as lambda_cost

from zarr_datafusion_lambda import sql
from zarr_datafusion_lambda.deploy import PROJECT
from zarr_datafusion_lambda.sql import COMBINER, WORKER


def describe(name: str, values: list[float]) -> str:
    """Percentiles of `values`, in milliseconds, on one line."""
    if not values:
        return f"{name:>9}: -"
    p50, p90, p99 = np.percentile(values, [50, 90, 99])
    return f"{name:>9}: p50 {p50:7.1f}  p90 {p90:7.1f}  p99 {p99:7.1f}  max {max(values):7.1f} ms"


def run(query: str, tables: dict[str, sql.Table], **settings: Any) -> dict[str, Any]:
    """
    Run `query` as `sql.run` does, with one fan-out and a progress bar: its
    rows, and what each invocation, worker and dispatcher did.
    """
    result = sql.run(query, tables, progress=True, **settings)
    [report] = result.reports
    # each worker's output, or with a combine, each state its dispatcher merged
    responses = report["workers"]
    combined = any("indexes" in r for r in responses)
    return {
        "chunks": report["chunks"],
        "combined": combined,
        "invocations": [
            {k: v for k, v in r.items() if k != "fan_out"} for r in result.invocation_stats
        ],
        "dispatchers": report["dispatcher_stats"],
        "arrivals": report["arrivals"],
        "states": len(responses) if combined else None,
        "timings": [] if combined else [r["timings_ms"] for r in responses],
        "chunks_in_flight": [] if combined else sorted({r["chunks_in_flight"] for r in responses}),
        "rows": result.table.to_pylist(),
        "wall_s": result.wall_s,
        "fan_out_s": report["elapsed_s"],
    }


def print_run(result: dict[str, Any], concurrency: int, expected: int, chunk_bytes: int) -> None:
    """How `run`'s run went: its pruning, timings and requests."""
    wall = result["wall_s"]
    print(
        f"{result['chunks']:,} chunks after pruning (expected {expected:,}), "
        f"{len(result['invocations']):,} invocations at concurrency {concurrency} "
        f"through {len(result['dispatchers'])} dispatcher(s)"
    )
    print(
        f"wall {wall:.1f}s: fan-out {result['fan_out_s']:.1f}s, "
        f"planning and finishing {wall - result['fan_out_s']:.1f}s"
    )
    scanned = result["chunks"] * chunk_bytes
    print(f"scanned {scanned / 1e12:.2f} TB uncompressed, {scanned / 1e9 / wall:.1f} GB/s")

    invokes = [(r["invoke_end"] - r["invoke_start"]) * 1000 for r in result["invocations"]]
    print(describe("invoke", invokes))
    if result["combined"]:
        print(
            f"merged in the dispatchers into {result['states']:,} states, without workers' timings"
        )
    else:
        for step in result["timings"][0]:
            print(describe(f"- {step}", [t[step] for t in result["timings"]]))
        print(f"chunks in flight on the workers: {result['chunks_in_flight']}")
    print(arrived(result))
    gets = sum(r["s3_gets"] for r in result["invocations"])
    lists = sum(r["s3_lists"] for r in result["invocations"])
    print(f"S3 requests from workers: {gets:,} GETs, {lists:,} LISTs")


def save(directory: Path, name: str, result: dict[str, Any]) -> Path:
    """`result` in `directory`, as JSON named for `name` and the time."""
    saved = directory / f"{name}-{time.strftime('%Y%m%dT%H%M%S')}.json"
    saved.parent.mkdir(exist_ok=True)
    saved.write_text(json.dumps(result, default=str))
    print(f"saved to {saved}")
    return saved


def arrived(result: dict[str, Any]) -> str:
    """When the workers' results reached the client, against when the workers finished."""
    arrivals = sorted(result["arrivals"], key=lambda a: a["at_s"])
    total = sum(a["bytes"] for a in arrivals)
    start = min(r["invoke_start"] for r in result["invocations"])
    last_worker = max(r["invoke_end"] for r in result["invocations"]) - start
    cumulative = np.cumsum([a["bytes"] for a in arrivals])
    at = {q: arrivals[int(np.searchsorted(cumulative, q * total))]["at_s"] for q in (0.5, 0.9, 1)}
    return (
        f" received: {total / 1e6:.1f} MB of results in {len(arrivals):,} frames, half by "
        f"{at[0.5]:.1f}s, 90% by {at[0.9]:.1f}s, all by {at[1]:.1f}s; "
        f"the last worker finished at {last_worker:.1f}s"
    )


def draw(result: dict[str, Any], invocations: list[dict[str, Any]], saved: Path) -> None:
    """The run's timeline, as an SVG beside its results."""
    # needs map-lambda's diagnostics extra, for matplotlib
    from map_lambda import timeline

    figure = timeline.generate_timeline(
        invocations,
        title=f"{result['query']}: {result['chunks']:,} chunks, "
        f"{result['chunks_per_invocation']} an invocation",
    )
    path = saved.with_suffix(".svg")
    figure.savefig(path)
    print(f"timeline in {path}")


def report(result: dict[str, Any], saved: Path) -> None:
    """
    What CloudWatch's REPORT lines say about a saved run, with its `query`
    and `chunks_per_invocation`, and what it cost, and its timeline.
    """
    print("waiting for CloudWatch Logs Insights...", flush=True)
    deployed = Session(project=PROJECT)
    dispatcher = f"{DISPATCHER}-{COMBINER}" if result.get("combined") else DISPATCHER

    def progress(line: str) -> None:
        print(f"  {line}", flush=True)

    workers = lambda_cost.fetch_reports(
        result["invocations"], deployed.function_name(WORKER), on_progress=progress
    )
    dispatchers = lambda_cost.fetch_reports(
        result["dispatchers"], deployed.function_name(dispatcher), on_progress=progress
    )
    cold = sum(w["init_ms"] is not None for w in workers)
    expected = len(result["invocations"]) + len(result["dispatchers"])
    print(
        f"{len(workers):,} of {len(result['invocations']):,} workers, {cold:,} cold; "
        f"{len(dispatchers)} of {len(result['dispatchers'])} dispatchers"
        + (
            " (the rest are missing from the cost)"
            if len(workers) + len(dispatchers) < expected
            else ""
        )
    )
    print(describe("handler", [w["execution_ms"] for w in workers]))
    # past its own elapsed_ms, it is still streaming results to the client
    print(describe("dispatch", [c["execution_ms"] for c in dispatchers]))
    if workers:
        print(
            f"   memory: max {max(w['max_memory_mb'] for w in workers):.0f} MB "
            f"of {max(w['memory_size_mb'] for w in workers):.0f} MB"
        )
    priced = workers + dispatchers
    print(
        lambda_cost.estimate_cost(priced, missing=expected - len(priced)) if priced else "no cost"
    )
    ran = {w["req_id"]: w for w in workers}
    draw(result, [ran.get(r["req_id"], r) for r in result["invocations"]], saved)
