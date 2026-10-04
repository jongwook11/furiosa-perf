"""Run service-simulation points with AIPerf and summarize them against the preset's SLO.

One point is one ``aiperf profile`` run. For chat presets a point's ``concurrency``
is the number of users in a conversation at the same time: AIPerf keeps a session's
slot during the user's think time between turns (checked against a logging mock
server), and each turn carries the conversation so far, including the responses
the server generated, so prefix caching is exercised as in a real chat service.
"""

import json
from pathlib import Path
from typing import Any

from furiosa_perf.clients.base import BenchmarkClient
from furiosa_perf.runner.server import option_args

AIPERF_VERSION = "0.13.0"
RESULT_FILE = "profile_export_aiperf.json"
# AIPerf writes these next to RESULT_FILE; all are removed before a point reruns.
RESULT_FILE_PREFIX = "profile_export"

# A point is not trusted when more than this share of its requests failed.
MAX_ERROR_RATE_PCT = 5.0
# Preset SLO key -> (AIPerf metric, statistic) it limits.
SLO_METRICS = {"ttft_p95_ms": ("time_to_first_token", "p95"), "tpot_p95_ms": ("inter_token_latency", "p95")}
LATENCY_STATS = ("avg", "p50", "p90", "p95", "p99")


def metric(result: dict[str, Any] | None, name: str, stat: str = "avg") -> float | None:
    """One statistic of an AIPerf metric (``{"unit": ..., "avg": ..., "p95": ...}``), ``None`` if absent."""
    entry = (result or {}).get(name)
    if not isinstance(entry, dict) or entry.get(stat) is None:
        return None
    return float(entry[stat])


def judge(point: dict[str, Any], result: dict[str, Any] | None, returncode: int) -> str:
    """Decide whether a point's numbers can be trusted.

    Args:
        point: AIPerf options of the point.
        result: AIPerf's JSON export, ``None`` if it wrote none.
        returncode: AIPerf's exit code.

    Returns:
        ``""`` if the point is valid, otherwise the reason it is not.
    """
    if returncode != 0 or result is None:
        return f"client exited with code {returncode}" + ("" if result is not None else " without a result file")
    completed = metric(result, "completed_request_count") or 0
    if completed == 0:
        return "no request completed"
    error_rate = metric(result, "request_error_rate") or 0.0
    if error_rate > MAX_ERROR_RATE_PCT:
        return f"{error_rate:.1f} % of requests failed"
    if point.get("streaming") and metric(result, "time_to_first_token", "p95") is None:
        return "no time to first token measured"
    return ""


def summary_row(
    point: dict[str, Any], result: dict[str, Any] | None, fail_reason: str, slo: dict[str, float]
) -> dict[str, Any]:
    """One summary row: users, validity, SLO verdict, throughput and latency distributions.

    Args:
        point: AIPerf options of the point.
        result: AIPerf's JSON export, ``None`` if it wrote none.
        fail_reason: Output of :func:`judge`.
        slo: The preset's SLO, e.g. ``{"ttft_p95_ms": 3000, "tpot_p95_ms": 50}``.

    Returns:
        The row, with the same keys for every point. ``slo_met`` is ``None`` for an
        invalid point or a missing metric.
    """

    def rounded(name: str, stat: str = "avg", digits: int = 2) -> float | None:
        value = metric(result, name, stat)
        return None if value is None else round(value, digits)

    slo_values = {key: metric(result, *SLO_METRICS[key]) for key in slo}
    slo_met = None
    if not fail_reason and slo and all(v is not None for v in slo_values.values()):
        slo_met = all(slo_values[key] <= limit for key, limit in slo.items())  # type: ignore[operator]

    row: dict[str, Any] = {
        "Users": point.get("concurrency"),
        "ok": not fail_reason,
        "fail_reason": fail_reason,
        "slo_met": slo_met,
        "completed": rounded("completed_request_count", digits=0),
        "Error Rate(%)": rounded("request_error_rate"),
        "Request Throughput(req/s)": rounded("request_throughput", digits=3),
        "Goodput(req/s)": rounded("goodput", digits=3),
        "Output Throughput(tok/s)": rounded("output_token_throughput"),
        "Total Throughput(tok/s)": rounded("total_token_throughput"),
        "Mean ISL": rounded("input_sequence_length", digits=1),
        "Mean OSL": rounded("output_sequence_length", digits=1),
    }
    for label, name in (("TTFT", "time_to_first_token"), ("TPOT", "inter_token_latency"), ("E2EL", "request_latency")):
        for stat in LATENCY_STATS:
            row[f"{stat.title()} {label}(ms)"] = rounded(name, stat)
    return row


def max_users_within_slo(rows: list[dict[str, Any]]) -> int | None:
    """Largest user count whose point is valid and meets the SLO, ``None`` if none does.

    Points above the first one that misses the SLO are ignored: a higher count
    passing again after a failure is noise, not capacity.
    """
    best = None
    for row in sorted(rows, key=lambda r: r["Users"] or 0):
        if not row["ok"] or not row["slo_met"]:
            break
        best = row["Users"]
    return best


class AiperfClient(BenchmarkClient):
    """``aiperf profile``: service-simulation points such as multi-turn chat."""

    NAME = "aiperf"
    PACKAGE = "aiperf"
    VERSION = AIPERF_VERSION
    EXECUTABLE = "aiperf"

    def command(self, point: dict[str, Any], point_dir: Path) -> list[str]:
        """``aiperf profile`` command line for one point.

        The preset's SLO is also passed as ``--goodput`` (per-request thresholds) unless
        the point sets ``goodput`` itself.
        """
        command = [str(self.env / "bin" / "aiperf") if self.env else "aiperf", "profile"]
        command += ["--model", self.model, "--url", f"http://{self.host}:{self.port}"]
        command += ["--artifact-dir", str(point_dir), "--ui-type", "none"]
        if "tokenizer" not in point:
            command += ["--tokenizer", self.tokenizer]
        if self.preset.slo and "goodput" not in point:
            command += ["--goodput", " ".join(f"{SLO_METRICS[k][0]}:{v}" for k, v in self.preset.slo.items())]
        for key, value in point.items():
            command += option_args(key, value)
        return command

    def read_result(self, point_dir: Path) -> dict[str, Any] | None:
        """AIPerf's JSON export of a point."""
        path = point_dir / RESULT_FILE
        return json.loads(path.read_text()) if path.exists() else None

    def clear_result(self, point_dir: Path) -> None:
        """Remove every export file of a previous run."""
        for path in point_dir.glob(f"{RESULT_FILE_PREFIX}*"):
            path.unlink()

    def judge(self, point: dict[str, Any], result: dict[str, Any] | None, returncode: int) -> str:
        """See :func:`judge`."""
        return judge(point, result, returncode)

    def summary_row(self, point: dict[str, Any], result: dict[str, Any] | None, fail_reason: str) -> dict[str, Any]:
        """See :func:`summary_row`."""
        return summary_row(point, result, fail_reason, self.preset.slo)

    def concurrency(self, point: dict[str, Any]) -> int:
        """Users of the point (``concurrency``)."""
        return int(point.get("concurrency") or 0)

    def run_summary(self) -> dict[str, Any]:
        """``capacity``: the SLO and the most users that met it."""
        if not self.preset.slo:
            return {}
        return {"capacity": {"slo": self.preset.slo, "max_users": max_users_within_slo(self.total_results["results"])}}
