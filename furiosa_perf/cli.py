"""Entry point for the furiosa-perf CLI: the ``run`` and ``report`` commands."""

import warnings

# requests emits RequestsDependencyWarning when urllib3 or chardet/charset-normalizer
# versions are newer than the range it was tested against. Suppress at the CLI boundary
# since the functionality is unaffected.
warnings.filterwarnings("ignore", message=".*doesn't match a supported version.*")

import shlex  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import click  # noqa: E402

from furiosa_perf.log import logger, setup_logger  # noqa: E402
from furiosa_perf.reporting.command import report  # noqa: E402
from furiosa_perf.runner.env import SystemDetector  # noqa: E402
from furiosa_perf.runner.runner import BenchmarkRunner  # noqa: E402
from furiosa_perf.runner.server import server_command  # noqa: E402
from furiosa_perf.spec import PerformanceBenchConfigLoader, Server, load_server  # noqa: E402


@click.command()
@click.option(
    "--server-config",
    required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    help="API server config (YAML): backend, model, devices, env, and options passed to `<backend> serve`.",
)
@click.option(
    "--benchmark-config",
    required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    help="Benchmark scenario configuration file (YAML).",
)
@click.option(
    "--reuse-server",
    is_flag=True,
    help="Benchmark a server already running on the configured port instead of launching one.",
)
@click.option(
    "--check",
    is_flag=True,
    help="Validate both configs and print the server command and scenario count, then exit without running.",
)
def run(server_config: Path, benchmark_config: Path, reuse_server: bool, check: bool) -> None:
    """Launch an API server, run benchmark scenarios, and save results."""
    setup_logger("WARNING" if check else "INFO")  # --check prints only the plan
    try:
        server = load_server(server_config, default_env=Path(sys.prefix))
        benchmark = PerformanceBenchConfigLoader.benchmark_config_setup(benchmark_config)
    except ValueError as e:
        raise click.UsageError(str(e)) from None

    if check:
        _print_check(server, len(benchmark.scenarios))
        return

    logger.info(f"Starting furiosa-perf: {server.backend} serving {server.model} on devices {server.devices}")
    system_valid, system_info = SystemDetector.check_system_compatibility(
        server.hardware_type, server.backend, str(server.python)
    )
    if not system_valid:
        logger.error("Hardware compatibility check failed. Exiting.")
        raise SystemExit(1)
    num_detected = len(system_info.hardware[server.hardware_type]["devices"])
    missing = [index for index in server.devices if index >= num_detected]
    if missing:
        raise click.UsageError(
            f"the device indices {missing} are invalid; {num_detected} {server.hardware_type.upper()} devices are detected"
        )

    runner = BenchmarkRunner(system_info, server, benchmark)
    logger.info("Benchmarking started.")
    runner.execute(reuse_server=reuse_server)
    logger.info("Benchmarking completed.")


def _print_check(server: Server, num_scenarios: int) -> None:
    """Print what ``run`` would launch, for validating configs in scripts and CI."""
    argv, env = server_command(server)
    if "CUDA_VISIBLE_DEVICES" in env:
        click.echo(f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} \\")
    click.echo(shlex.join(argv))
    click.echo(f"{num_scenarios} scenarios")
    if not server.executable.exists():
        click.echo(f"warning: {server.executable} does not exist on this host", err=True)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli() -> None:
    """furiosa-perf: LLM serving performance benchmark CLI."""


cli.add_command(run)
cli.add_command(report)

if __name__ == "__main__":
    cli()
