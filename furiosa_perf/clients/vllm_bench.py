"""Run a preset's points with ``vllm bench serve`` and summarize them.

The client lives in its own virtual environment, pinned to
:data:`VLLM_BENCH_VERSION` so that results from different hosts and dates come
from the same client. Each point writes the client's result JSON next to its
log; the summary is built from that JSON, never from the console output.
"""

import json
import os
import signal
import subprocess
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pytablewriter import MarkdownTableWriter

from furiosa_perf.log import logger
from furiosa_perf.results.store import RECORD_FILE, check_unique_point_names, point_name, read_json, write_json
from furiosa_perf.runner.monitor import HardwareMonitor
from furiosa_perf.runner.server import option_args
from furiosa_perf.spec import Preset

VLLM_BENCH_VERSION = "0.30.0"
CLIENT_PYTHON = "3.12"
RESULT_FILE = "result.json"
BENCH_LOG = "bench.log"

# A point is not trusted when more requests than this fraction failed.
MAX_FAILED_FRACTION = 0.05
# With ignore-eos every request must return exactly the requested output length;
# fewer tokens than this fraction of it means the client did not count them all.
MIN_OUTPUT_FRACTION = 0.9

# Statistics read from the result JSON for each latency metric.
LATENCY_STATS = ("mean", "median", "p25", "p50", "p75", "p90", "p95", "p99")
# result JSON metric -> (summary column suffix, divisor from ms, digits). The units match
# the earlier summary files, which the report reads.
LATENCY_COLUMNS = {"ttft": ("TTFT(s)", 1000.0, 4), "tpot": ("TPOT(ms)", 1.0, 3), "e2el": ("E2EL(s)", 1000.0, 4)}


def client_dir() -> Path:
    """Directory of the pinned client environment (shared by runs on this host)."""
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return cache / "furiosa-perf" / "clients" / f"vllm-{VLLM_BENCH_VERSION}"


def ensure_client() -> Path:
    """Create the client environment on first use and return its interpreter.

    Returns:
        Path of the environment's ``python``.

    Raises:
        RuntimeError: If creating the environment or installing vLLM fails.
    """
    env = client_dir()
    python = env / "bin" / "python"
    if (env / "bin" / "vllm").exists():
        return python
    logger.info(f"Installing the benchmark client vllm=={VLLM_BENCH_VERSION} into {env}")
    for step in (
        ["uv", "venv", "--python", CLIENT_PYTHON, str(env)],
        ["uv", "pip", "install", "--python", str(python), f"vllm=={VLLM_BENCH_VERSION}"],
    ):
        done = subprocess.run(step, capture_output=True, text=True)
        if done.returncode != 0:
            raise RuntimeError(
                f"failed to set up the benchmark client ({' '.join(step)}); cause: {done.stderr.strip()}"
            )
    return python


def judge(point: dict[str, Any], result: dict[str, Any] | None, returncode: int) -> str:
    """Decide whether a point's numbers can be trusted.

    Args:
        point: Client options of the point.
        result: The client's result JSON, ``None`` if it wrote none.
        returncode: The client's exit code.

    Returns:
        ``""`` if the point is valid, otherwise the reason it is not.
    """
    if returncode != 0 or result is None:
        return f"client exited with code {returncode}" + ("" if result is not None else " without a result file")
    requested = int(result.get("num_prompts") or point.get("num-prompts") or 0)
    failed = int(result.get("failed") or 0)
    if requested and failed / requested > MAX_FAILED_FRACTION:
        return f"{failed} of {requested} requests failed"

    output_len = int(point.get("random-output-len") or 0)
    if output_len <= 0:
        return ""
    produced = int(result.get("total_output_tokens") or 0)
    if produced == 0:
        return "no output tokens counted"
    expected = int(result.get("completed") or 0) * output_len
    if point.get("ignore-eos") and produced < MIN_OUTPUT_FRACTION * expected:
        return f"outputs shorter than requested ({produced} of {expected} tokens)"
    return ""


def summary_row(point: dict[str, Any], result: dict[str, Any] | None, fail_reason: str) -> dict[str, Any]:
    """One summary row: the point's identity, its validity and the client's metrics.

    Column names and units are those of the earlier summary files, which the report
    reads. Metrics are kept as measured even when the point is invalid; ``ok`` and
    ``fail_reason`` say whether to trust them.

    Args:
        point: Client options of the point.
        result: The client's result JSON, ``None`` if it wrote none.
        fail_reason: Output of :func:`judge`.

    Returns:
        The row, with the same keys for every point.
    """
    result = result or {}

    def number(key: str, divisor: float = 1.0, digits: int = 2) -> float | None:
        value = result.get(key)
        return None if value is None else round(float(value) / divisor, digits)

    row: dict[str, Any] = {
        "Input Tokens": point.get("random-input-len"),
        "Output Tokens": point.get("random-output-len") or 0,
        "Concurrent": point.get("max-concurrency"),
        "ok": not fail_reason,
        "fail_reason": fail_reason,
        "completed": result.get("completed"),
        "failed": result.get("failed"),
        "Output Throughput(tok/s)": number("output_throughput"),
        "Total Throughput(tok/s)": number("total_token_throughput"),
        "Request Throughput(req/s)": number("request_throughput", digits=3),
    }
    for metric, (suffix, divisor, digits) in LATENCY_COLUMNS.items():
        for stat in LATENCY_STATS:
            row[f"{stat.title()} {suffix}"] = number(f"{stat}_{metric}_ms", divisor, digits)
    return row


class VllmPerformanceBenchmark:
    """Run every point of a preset against a running API server."""

    def __init__(
        self,
        preset: Preset,
        model: str,
        host: str,
        port: int,
        run_dir: Path,
        identity: dict[str, Any],
        resume: bool = False,
    ) -> None:
        """Store what to run and where results go.

        Args:
            preset: Points and task of the benchmark.
            model: Model ID the server reports on ``/v1/models``.
            host: API server host.
            port: API server port.
            run_dir: Directory of this run (see :mod:`furiosa_perf.results.store`).
            identity: Fields copied into every ``record.json`` so a point is
                self-describing: kind, task, model, hardware, parallelism, versions.
            resume: Keep points whose ``record.json`` is valid instead of rerunning them.

        Raises:
            ValueError: If two points of the preset would share a directory.
        """
        check_unique_point_names(preset.points)
        self.preset = preset
        self.task = preset.task
        self.model = model
        self.host = host
        self.port = port
        self.run_dir = run_dir
        self.identity = identity
        self.resume = resume
        self.python: Path | None = None
        self.bench_process: subprocess.Popen[str] | None = None
        self.total_results: dict[str, Any] = {"results": []}

    def setup(self) -> None:
        """Make sure the pinned client environment exists."""
        self.python = ensure_client()
        logger.info(f"Benchmark client: vllm {VLLM_BENCH_VERSION} ({self.python})")

    def run(self, monitor: HardwareMonitor | None = None, server_alive: Callable[[], bool] | None = None) -> None:
        """Run all points in order and collect their summary rows.

        An invalid point does not stop the run. If the server is found dead after
        one, the run stops, because every later point would fail as well.

        Args:
            monitor: Running hardware monitor; each point is bracketed with
                :meth:`HardwareMonitor.begin` / :meth:`HardwareMonitor.end`.
            server_alive: Returns whether the server still answers; checked after an
                invalid point.

        Raises:
            RuntimeError: If the server stopped during the run.
        """
        self.total_results = {"model": self.model, "task": self.task, "results": []}
        points = self.preset.points
        for index, point in enumerate(points):
            kept = self._kept_row(point)
            if kept is not None:
                logger.info(f"point {point_name(point)} is already recorded as valid; keeping it")
                self.total_results["results"].append(kept)
                continue
            row = self._run_point(point, monitor)
            self.total_results["results"].append(row)
            if row["ok"]:
                continue
            logger.warning(f"point {point_name(point)} is invalid: {row['fail_reason']}")
            if server_alive is not None and not server_alive():
                raise RuntimeError(
                    f"the server stopped answering during point {point_name(point)}; "
                    f"{len(points) - index - 1} remaining points were not run"
                )

        num_invalid = sum(not row["ok"] for row in self.total_results["results"])
        logger.info(f"{len(points)} points run, {num_invalid} invalid")

    def _kept_row(self, point: dict[str, Any]) -> dict[str, Any] | None:
        """With ``resume``, the summary row of a point already recorded as valid."""
        if not self.resume:
            return None
        record = read_json(self.point_dir(point_name(point)) / RECORD_FILE)
        if record is None or record.get("point") != point or not record.get("summary", {}).get("ok"):
            return None
        return dict(record["summary"])

    def _command(self, point: dict[str, Any], result_dir: Path) -> list[str]:
        """``vllm bench serve`` command line for one point."""
        command = [str(self.python), "-m", "vllm.entrypoints.cli.main", "bench", "serve"]
        command += ["--model", self.model, "--host", self.host, "--port", str(self.port)]
        # The server was already checked to be ready, so the client need not wait for it.
        command += ["--save-result", "--result-dir", str(result_dir), "--result-filename", RESULT_FILE]
        command += ["--ready-check-timeout-sec", "0"]
        for key, value in point.items():
            command += option_args(key, value)
        return command

    def _run_point(self, point: dict[str, Any], monitor: HardwareMonitor | None) -> dict[str, Any]:
        """Run one point and return its summary row (plus monitor fields)."""
        name = point_name(point)
        result_dir = self.point_dir(name)
        result_path = result_dir / RESULT_FILE
        result_path.unlink(missing_ok=True)  # never read a previous run's result
        command = self._command(point, result_dir)
        logger.info(f"Executing: {' '.join(command)}")

        started = datetime.now(timezone.utc).isoformat()
        window = monitor.begin(name) if monitor is not None else None
        with (result_dir / BENCH_LOG).open("w") as log:
            self.bench_process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=result_dir,
                bufsize=1,
                start_new_session=True,
            )
            assert self.bench_process.stdout is not None, "stdout should be a pipe"
            for line in self.bench_process.stdout:
                log.write(line)
                logger.info(line.rstrip())
            returncode = self.bench_process.wait()
        self.bench_process = None

        result = None
        if result_path.exists():
            result = json.loads(result_path.read_text())
        row = summary_row(point, result, judge(point, result, returncode))
        if monitor is not None and window is not None:
            row.update(monitor.end(window, int(point.get("max-concurrency") or 0), slice_dir=result_dir))
        record = {
            **self.identity,
            "client": "vllm-bench",
            "client_version": VLLM_BENCH_VERSION,
            "name": name,
            "point": point,
            "started": started,
            "finished": datetime.now(timezone.utc).isoformat(),
            "summary": row,
        }
        write_json(result_dir / RECORD_FILE, record)
        logger.info(f"Parsed result: {row}")
        return row

    def finish_info(self, desc: str = "") -> None:
        """Write Markdown and CSV summary files, overall and per ISL/OSL pair.

        Args:
            desc: Text prepended to the generated description block.
        """
        rows = self.total_results["results"]
        if not rows:
            return
        headers = list(rows[0])
        report_desc = (
            f"{desc}\n"
            f"* Performance Benchmark - {self.preset.kind} {self.task} with vllm bench {VLLM_BENCH_VERSION}\n"
            f"* Model Info: {self.model}\n"
            f"* Device Info: {self.identity['hardware']} x {self.identity['num_devices']} {self.identity['layout']}\n"
            f"* date: {datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}\n"
            f"* Scenario Results:"
        )
        by_length: dict[str, list[list[Any]]] = defaultdict(list)
        for row in rows:
            by_length[self._length_key(row["Input Tokens"], row["Output Tokens"])].append(list(row.values()))
        result_root = self.run_dir
        for key, values in by_length.items():
            self._write_summary_files(result_root, headers, values, report_desc, suffix=f"_{key}", log=True)
        all_values = [value for values in by_length.values() for value in values]
        self._write_summary_files(result_root, headers, all_values, report_desc, suffix="")

    @staticmethod
    def _write_summary_files(
        result_dir: Path, headers: list[str], rows: list[list[Any]], desc: str, suffix: str, log: bool = False
    ) -> None:
        """Write ``summary<suffix>.md`` and ``.csv``; empty cells for missing values."""
        md_writer = MarkdownTableWriter()
        md_writer.headers = headers
        md_writer.value_matrix = rows
        csv_lines = [",".join(headers)] + [",".join("" if v is None else str(v) for v in row) for row in rows]
        if log:
            logger.info("\n".join([desc, md_writer.dumps()]))
        (result_dir / f"summary{suffix}.md").write_text(desc + "\n" + md_writer.dumps() + "\n")
        (result_dir / f"summary{suffix}.csv").write_text(desc + "\n" + "\n".join(csv_lines) + "\n")

    @staticmethod
    def _length_key(input_tokens: int, output_tokens: int) -> str:
        """Compact ``"<ISL>_<OSL>"`` key such as ``"1k_1k"`` for grouping summary files."""
        isl = f"{input_tokens // 1024}k" if input_tokens >= 1024 else str(input_tokens)
        osl = f"{output_tokens // 1024}k" if output_tokens >= 1024 else str(output_tokens)
        return f"{isl}_{osl}"

    def point_dir(self, name: str) -> Path:
        """Return (and create) the directory of one point inside the run directory."""
        path = self.run_dir / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def stop(self) -> None:
        """Terminate the running client process group, if any."""
        if self.bench_process is None:
            return
        if self.bench_process.poll() is None:
            try:
                os.killpg(os.getpgid(self.bench_process.pid), signal.SIGTERM)
                self.bench_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self.bench_process.pid), signal.SIGKILL)
        self.bench_process = None
