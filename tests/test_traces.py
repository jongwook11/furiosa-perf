"""Tests for replaying a request trace: download check, slicing and stretching, and the AIPerf command."""

import hashlib
from pathlib import Path

import pytest

from furiosa_perf.clients.aiperf import AiperfClient, max_load_within_slo
from furiosa_perf.clients.traces import fetch_trace, load_trace, scale_trace
from furiosa_perf.spec import load_preset

ROWS = [
    {"timestamp": 0, "input_length": 6000, "output_length": 100, "hash_ids": [0, 1]},
    {"timestamp": 500, "input_length": 50000, "output_length": 100, "hash_ids": [0, 2]},
    {"timestamp": 1000, "input_length": 3000, "output_length": 50, "hash_ids": [0, 3]},
    {"timestamp": 4000, "input_length": 3000, "output_length": 50, "hash_ids": [0, 4]},
]


def test_scale_trace_cuts_the_slice_that_fits_and_stretches_it() -> None:
    """At 50 % for 2 s of wall time a point replays 1 s of trace, twice as slowly, minus over-long requests."""
    rows, window = scale_trace(ROWS, speed_pct=50, wall_seconds=2, max_total_tokens=40960)

    assert [r["timestamp"] for r in rows] == [0, 2000]  # the 50k-token request is dropped, 4 s is past the slice
    assert [r["hash_ids"] for r in rows] == [[0, 1], [0, 3]]  # prefix structure is kept as recorded
    assert (window.trace_seconds, window.requests, window.dropped_too_long) == (1.0, 2, 1)
    assert window.offered_req_per_s == 1.0


def test_scale_trace_at_full_speed_keeps_the_recorded_times() -> None:
    """100 % replays the recorded timestamps unchanged."""
    rows, _ = scale_trace(ROWS, speed_pct=100, wall_seconds=5, max_total_tokens=None)

    assert [r["timestamp"] for r in rows] == [0, 500, 1000, 4000]


def test_fetch_trace_refuses_a_file_with_another_hash(tmp_path: Path) -> None:
    """A changed trace file never silently changes results: its SHA-256 must match the preset."""
    path = tmp_path / "trace.jsonl"
    path.write_text('{"timestamp": 0, "input_length": 1, "output_length": 1}\n')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    assert fetch_trace(str(path), digest) == path
    assert load_trace(path)[0]["input_length"] == 1
    with pytest.raises(ValueError, match="the preset expects"):
        fetch_trace(str(path), "0" * 64)


def test_trace_points_replay_a_written_slice_at_fixed_times(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The command replays the point's trace file on a fixed schedule bounded by the slice length."""
    preset = load_preset("simulation/chat-mooncake")
    client = AiperfClient(preset, "m", "h", 1, tmp_path, {}, tokenizer="t")
    client.env = Path("/env")
    client._trace_rows = ROWS
    command = client.command({**preset.points[0], "trace-speed-pct": 50, "trace-duration": 2}, tmp_path)

    assert command[command.index("--custom-dataset-type") + 1] == "mooncake_trace"
    assert command[command.index("--input-file") + 1] == str(tmp_path / "trace.jsonl")
    assert "--fixed-schedule" in command
    assert command[command.index("--benchmark-duration") + 1] == "2"
    assert "--trace-speed-pct" not in command and "--trace-duration" not in command
    assert len(load_trace(tmp_path / "trace.jsonl")) == 2


def test_capacity_of_a_trace_replay_is_a_speed() -> None:
    """For a trace the highest passing speed, not a user count, is the capacity."""
    rows = [
        {"Trace Speed(%)": 10, "ok": True, "slo_met": True},
        {"Trace Speed(%)": 40, "ok": True, "slo_met": False},
        {"Trace Speed(%)": 25, "ok": True, "slo_met": True},
    ]

    assert max_load_within_slo(rows, "Trace Speed(%)") == 25
