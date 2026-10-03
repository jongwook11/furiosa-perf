"""Entry point for the furiosa-perf CLI: the ``run`` and ``report`` commands."""

import warnings

# requests emits RequestsDependencyWarning when urllib3 or chardet/charset-normalizer
# versions are newer than the range it was tested against. Suppress at the CLI boundary
# since the functionality is unaffected.
warnings.filterwarnings("ignore", message=".*doesn't match a supported version.*")

from pathlib import Path  # noqa: E402

import click  # noqa: E402

from furiosa_perf.log import logger, setup_logger  # noqa: E402
from furiosa_perf.reporting.command import report  # noqa: E402
from furiosa_perf.runner.env import SystemDetector  # noqa: E402
from furiosa_perf.runner.runner import BenchmarkRunner  # noqa: E402
from furiosa_perf.spec import APIServerConfigLoader, PerformanceBenchConfigLoader  # noqa: E402


@click.command()
@click.option(
    "--model-id",
    type=str,
    default="LGAI-EXAONE/EXAONE-4.0-32B-FP8",
    show_default=True,
    help="Local path to model directory, or HuggingFace model ID.",
)
@click.option(
    "--hardware-type",
    type=str,
    default="npu",
    show_default=True,
    help="Hardware type to benchmark (e.g. npu, gpu).",
)
@click.option(
    "--backend",
    type=str,
    default="furiosa-llm",
    show_default=True,
    help="Backend framework to use (e.g. furiosa-llm, vllm).",
)
@click.option(
    "--server-config",
    required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    help="LLM API server launch configuration file (YAML).",
)
@click.option(
    "--benchmark-config",
    required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    help="Benchmark scenario configuration file (YAML).",
)
def run(
    model_id: str,
    hardware_type: str,
    backend: str,
    server_config: Path,
    benchmark_config: Path,
) -> None:
    """Launch an API server, run benchmark scenarios, and save results."""
    setup_logger("INFO")
    logger.info("Starting furiosa-perf")
    logger.info(f"Model:         {model_id}")
    logger.info(f"Hardware type: {hardware_type}")
    logger.info(f"Backend:       {backend}")

    system_valid, system_info = SystemDetector.check_system_compatibility(hardware_type, backend)
    if not system_valid:
        logger.error("Hardware compatibility check failed. Exiting.")
        raise SystemExit(1)

    api_server_config = APIServerConfigLoader.api_server_setup(backend, server_config)
    benchmark_config_obj = PerformanceBenchConfigLoader.benchmark_config_setup(benchmark_config)

    runner = BenchmarkRunner(system_info, hardware_type, api_server_config, benchmark_config_obj)
    logger.info("Benchmarking started.")
    runner.execute(model_id)
    logger.info("Benchmarking completed.")


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli() -> None:
    """furiosa-perf: LLM serving performance benchmark CLI."""


cli.add_command(run)
cli.add_command(report)

if __name__ == "__main__":
    cli()
