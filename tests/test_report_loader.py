"""Tests for how the current report finds the device and version of a result."""

import json
from pathlib import Path

from furiosa_perf.reporting.schemas import BenchmarkMetricLoader


def test_new_runs_are_described_by_run_json(tmp_path: Path) -> None:
    """With run.json next to summary.csv its fields are used and the layout is appended to the device."""
    run = tmp_path / "baseline" / "anything" / "text-gen" / "org--m"
    run.mkdir(parents=True)
    (run / "summary.csv").write_text("Input Tokens\n1\n")
    (run / "run.json").write_text(
        json.dumps(
            {
                "hardware": "H100-80GB-HBM3",
                "layout": "tp1-dp2",
                "num_devices": 2,
                "backend": "vllm",
                "backend_version": "0.28.0",
            }
        )
    )

    assert BenchmarkMetricLoader._extract_device_and_version(run / "summary.csv") == (
        "H100-80GB-HBM3-tp1-dp2",
        "2",
        "vllm",
        "0.28.0",
    )
    assert BenchmarkMetricLoader.source_label(run / "summary.csv") == "H100-80GB-HBM3-tp1-dp2_2_vllm_0.28.0"


def test_legacy_runs_are_described_by_their_path(tmp_path: Path) -> None:
    """bench_space results have no run.json; the device folder name is parsed as before."""
    summary = tmp_path / "RNGD_4_furiosa-llm_2026.4.0b15" / "vllm" / "offline" / "m" / "summary.csv"
    summary.parent.mkdir(parents=True)
    summary.write_text("Input Tokens\n1\n")

    assert BenchmarkMetricLoader._extract_device_and_version(summary) == ("RNGD", "4", "furiosa-llm", "2026.4.0b15")
