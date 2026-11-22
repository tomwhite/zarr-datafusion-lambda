"""What a query's result says about its fan-outs, from reports written as literals."""

import pyarrow as pa
import pytest
from map_lambda import DISPATCHER

from zarr_datafusion_lambda import sql


def invocation(index, start):
    return {
        "index": index,
        "req_id": f"w{start}-{index}",
        "invoke_start": start,
        "invoke_end": start + 1.0,
        "call": 0,
        "call_start": start,
    }


def dispatcher(req_id, start):
    return {
        "req_id": req_id,
        "dispatched": 2,
        "retried": 0,
        "retry_wait_ms": 0.0,
        "cold": False,
        "connections": 1,
        "elapsed_ms": 1000.0,
        "call": 0,
        "call_start": start,
    }


def relayed(start):
    """A fan-out of two invocations whose workers' results came here as they were."""
    return {
        "chunks": 4,
        "invocations": 2,
        "elapsed_s": 1.0,
        "invocation_stats": [invocation(0, start), invocation(1, start)],
        "workers": [
            {"index": 0, "s3_requests": {"gets": 3, "lists": 1}},
            {"index": 1, "s3_requests": {"gets": 2, "lists": 0}},
        ],
        "dispatcher_stats": [dispatcher(f"d{start}", start)],
        "arrivals": [],
    }


def combined(start):
    """A fan-out of two invocations merged by its dispatcher into one state."""
    return {
        **relayed(start),
        "workers": [{"indexes": [0, 1], "s3_requests": {"gets": 5, "lists": 1}}],
    }


def result(*reports):
    return sql.Result(pa.table({"x": [1]}), list(reports), wall_s=2.0)


def test_each_invocation_carries_its_fan_out_and_its_workers_s3_requests():
    records = result(relayed(100.0), combined(200.0)).invocation_stats

    assert [(r["fan_out"], r["index"]) for r in records] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert [(r["s3_gets"], r["s3_lists"]) for r in records] == [(3, 1), (2, 0), (5, 1), (0, 0)]


def test_each_dispatcher_carries_its_fan_out():
    records = result(relayed(100.0), combined(200.0)).dispatcher_stats

    assert [(r["fan_out"], r["req_id"]) for r in records] == [(0, "d100.0"), (1, "d200.0")]


def test_stats_add_the_fan_outs():
    stats = result(relayed(100.0), combined(200.0)).stats()

    assert "8 in 2 fan-out(s)" in stats
    assert "invocations:    4" in stats
    assert "10 GETs, 2 LISTs" in stats


def test_stats_without_a_fan_out_say_so():
    assert "no fan-outs" in result().stats()


def test_cost_prices_every_fan_outs_workers_and_dispatchers(monkeypatch):
    asked = []

    def fetch_reports(records, function, **_):
        asked.append((function, [r["req_id"] for r in records]))
        # the second fan-out's dispatcher's REPORT line never turns up
        return (
            []
            if function.endswith("partial-reduce")
            else [{**r, "billed_ms": 1000, "memory_size_mb": 1024} for r in records]
        )

    monkeypatch.setattr(sql.lambda_cost, "fetch_reports", fetch_reports)

    cost = result(relayed(100.0), combined(200.0)).cost(progress=False)

    suffixes = ["fragment", DISPATCHER, "fragment", f"{DISPATCHER}-partial-reduce"]
    assert all(function.endswith(s) for (function, _), s in zip(asked, suffixes, strict=True))
    assert (cost.invocations, cost.missing) == (5, 1)
    assert (cost.s3_gets, cost.s3_lists) == (10, 2)


def test_cost_without_a_fan_out_is_refused():
    with pytest.raises(ValueError, match="no fan-outs"):
        result().cost()


class Logs:
    """Logs Insights over REPORT lines, as `{log group: {req_id: when}}`."""

    def __init__(self, lines):
        self.lines = lines
        self.queries = []

    def start_query(self, *, logGroupName, startTime, endTime, queryString):
        self.queries.append((logGroupName, startTime, endTime))
        return {"queryId": str(len(self.queries) - 1)}

    def get_query_results(self, *, queryId):
        group, start, end = self.queries[int(queryId)]
        report = {
            "@duration": "900",
            "@billedDuration": "1000",
            "@memorySize": "1024000000",
            "@maxMemoryUsed": "200000000",
        }
        results = [
            [{"field": "@requestId", "value": req_id}]
            + [{"field": k, "value": v} for k, v in report.items()]
            for req_id, at in self.lines.get(group, {}).items()
            if start <= at < end
        ]
        return {"status": "Complete", "results": results}


def test_cost_finds_the_report_lines_of_workers_and_dispatchers(monkeypatch):
    first, second = relayed(100.0), combined(200.0)
    group = f"/aws/lambda/{sql.PROJECT}-"
    logs = Logs(
        {
            f"{group}fragment": {
                r["req_id"]: r["invoke_end"]
                for r in first["invocation_stats"] + second["invocation_stats"]
            },
            f"{group}{DISPATCHER}": {"d100.0": 101.0},
            f"{group}{DISPATCHER}-partial-reduce": {"d200.0": 201.0},
        }
    )
    monkeypatch.setattr("boto3.client", lambda *_, **__: logs)

    cost = result(first, second).cost(progress=False)

    assert (cost.invocations, cost.missing) == (6, 0)
