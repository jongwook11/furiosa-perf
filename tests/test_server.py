"""Tests for building the API server command line."""

from pathlib import Path

import pytest

from furiosa_perf.runner.server import option_args, server_command
from furiosa_perf.spec import parse_server

ENV = Path("/opt/serving-env")


def command(backend: str, devices: str, **extra: object) -> tuple[list[str], dict[str, str]]:
    """``server_command`` for a parsed config with the given backend and devices."""
    raw = {"backend": backend, "model": "org/model", "devices": devices, "enable-prefix-caching": True, **extra}
    return server_command(parse_server(raw, ENV))


def test_furiosa_llm_counts_tensor_parallelism_in_pes() -> None:
    """Two RNGD cards as two TP-1 replicas: 8 PEs per replica, devices as npu:<i>, all sizes explicit."""
    argv, env = command("furiosa-llm", "2,3", parallel={"dp": 2})

    assert argv[:3] == [str(ENV / "bin" / "furiosa-llm"), "serve", "org/model"]
    assert argv[argv.index("--devices") + 1] == "npu:2,npu:3"
    assert argv[argv.index("--tensor-parallel-size") + 1] == "8"
    assert argv[argv.index("--data-parallel-size") + 1] == "2"
    assert argv[argv.index("--pipeline-parallel-size") + 1] == "1"
    assert "CUDA_VISIBLE_DEVICES" not in env


def test_vllm_selects_gpus_through_the_environment() -> None:
    """vLLM sees only the configured GPUs and counts tensor parallelism in GPUs."""
    argv, env = command("vllm", "4,5,6,7", parallel={"tp": 2})

    assert env["CUDA_VISIBLE_DEVICES"] == "4,5,6,7"
    assert "--devices" not in argv
    assert argv[argv.index("--tensor-parallel-size") + 1] == "2"
    assert argv[argv.index("--data-parallel-size") + 1] == "2"


def test_server_runs_from_its_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """The env's bin comes first on PATH and variables pointing at another environment are dropped."""
    monkeypatch.setenv("VIRTUAL_ENV", "/somewhere/else")
    monkeypatch.setenv("PYTHONPATH", "/somewhere/else/lib")
    _, env = command("vllm", "0")

    assert env["PATH"].startswith(str(ENV / "bin"))
    assert "VIRTUAL_ENV" not in env
    assert "PYTHONPATH" not in env


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, ["--enforce-eager"]),
        (False, []),
        (None, []),
        (4096, ["--enforce-eager", "4096"]),
        (["a", "b"], ["--enforce-eager", "a", "b"]),
        ({"k": 1}, ["--enforce-eager", '{"k": 1}']),
    ],
)
def test_option_args(value: object, expected: list[str]) -> None:
    """Flags, omitted values, scalars, lists and JSON objects render as the backends' argparse expects."""
    assert option_args("enforce-eager", value) == expected
