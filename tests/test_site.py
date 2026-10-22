"""Tests for collecting results into the report site's data files."""

import json
from pathlib import Path

from furiosa_perf.reporting.site import collect_runs, write_data, write_viewer

LEGACY_SUMMARY = """* Framework Version: furiosa-llm_2026.3.0
* API Server Command: furiosa-llm serve furiosa-ai/m --port 8000 --enable-overlap-scheduling --enable-payload-logging
* Model Info: furiosa-ai/m
* date: 20260708-045349
* Scenario Results:
Input Tokens,Output Tokens,Concurrent,Output Throughput(tok/s),mean_power
128,128,1,70.77,231.0
128,128,4,0,99.9
"""


def _results(tmp_path: Path) -> Path:
    legacy = tmp_path / "bench_space" / "RNGD_4_furiosa-llm_2026.3.0" / "vllm" / "offline" / "m"
    legacy.mkdir(parents=True)
    (legacy / "summary.csv").write_text(LEGACY_SUMMARY)
    flag_experiment = legacy.parent / "m__overlap"
    flag_experiment.mkdir()
    (flag_experiment / "summary.csv").write_text(LEGACY_SUMMARY)

    run = tmp_path / "results" / "baseline" / "H100_2-tp1-dp2_vllm" / "text" / "org--m"
    (run / "i1024-o128-c1").mkdir(parents=True)
    (run / "run.json").write_text(
        json.dumps(
            {
                "kind": "baseline",
                "task": "text",
                "model": "org/m",
                "hardware": "H100",
                "hardware_product": "H100-80GB-HBM3",
                "num_devices": 2,
                "layout": "tp1-dp2",
                "backend": "vllm",
                "backend_version": "0.30.0",
                "started": "2026-10-18T02:28:08+00:00",
                "server": {"command": "vllm serve org/m --no-enable-prefix-caching"},
            }
        )
    )
    summary = {
        "Input Tokens": 1024,
        "Output Tokens": 128,
        "Concurrent": 1,
        "ok": False,
        "Output Throughput(tok/s)": 5.0,
    }
    (run / "i1024-o128-c1" / "record.json").write_text(json.dumps({"summary": summary}))
    (run / "summary.csv").write_text("Input Tokens\n1024\n")  # read through run.json, not as a legacy run
    return tmp_path


def test_both_result_layouts_become_runs(tmp_path: Path) -> None:
    """Legacy and new runs carry the same fields; flag experiments and the new tree's summary.csv are skipped."""
    new, legacy = collect_runs([_results(tmp_path)])  # sorted by id

    assert (legacy["task"], legacy["model"], legacy["hw"], legacy["n"], legacy["ver"]) == (
        "text",
        "m",
        "RNGD",
        4,
        "2026.3.0",
    )
    assert legacy["flags"] == ["enable-overlap-scheduling"]
    assert legacy["date"] == "2026-07-08T04:53:49+00:00"
    assert legacy["points"][0]["Output Throughput(tok/s)"] == 70.77
    assert (new["model"], new["model_id"], new["hw"], new["layout"], new["flags"]) == (
        "m",
        "org/m",
        "H100-80GB-HBM3",
        "tp1-dp2",
        [],
    )
    assert new["id"] == "H100-80GB-HBM3_2-tp1-dp2_vllm_0.30.0--baseline--text--m--202610180228"


def test_site_has_an_index_and_one_file_per_run(tmp_path: Path) -> None:
    """index.json describes each run and counts failed points; each run file has its points as columns."""
    runs = collect_runs([_results(tmp_path)])
    site = tmp_path / "site"
    write_viewer(site, theme=None)
    write_data(runs, site)

    index = json.loads((site / "data" / "index.json").read_text())["runs"]
    assert [(r["num_points"], r["failed"], r["shapes"]) for r in index] == [
        (1, 1, ["1024/128"]),
        (2, 1, ["128/128"]),
    ]
    points = json.loads((site / "data" / "runs" / f"{index[1]['id']}.json").read_text())
    assert points["cols"][:4] == ["Input Tokens", "Output Tokens", "Concurrent", "Output Throughput(tok/s)"]
    assert points["rows"][1][3] == 0
    assert (site / "index.html").exists() and (site / "theme" / "theme.css").exists()
