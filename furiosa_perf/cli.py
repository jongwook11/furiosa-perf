"""Entry point for the furiosa-perf CLI: the ``run``, ``env`` and ``report`` commands."""

import warnings

# requests emits RequestsDependencyWarning when urllib3 or chardet/charset-normalizer
# versions are newer than the range it was tested against. Suppress at the CLI boundary
# since the functionality is unaffected.
warnings.filterwarnings("ignore", message=".*doesn't match a supported version.*")

import json  # noqa: E402
import shlex  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import click  # noqa: E402

from furiosa_perf.log import logger, setup_logger  # noqa: E402
from furiosa_perf.reporting.command import report  # noqa: E402
from furiosa_perf.runner import env as environment  # noqa: E402
from furiosa_perf.runner.runner import BenchmarkRunner  # noqa: E402
from furiosa_perf.runner.server import server_command  # noqa: E402
from furiosa_perf.spec import Server, apply_server_overrides, load_preset, load_server  # noqa: E402


@click.command()
@click.option(
    "--server-config",
    required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    help="API server config (YAML): backend, model, devices, env, and options passed to `<backend> serve`.",
)
@click.option(
    "--preset",
    required=True,
    help="Benchmark preset: a YAML file, or a bundled name such as baseline/text-gen.",
)
@click.option(
    "--output-dir",
    default="results",
    show_default=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Root of the results tree; each run goes to <kind>/<device>/<task>/<model> under it.",
)
@click.option("--resume", is_flag=True, help="Finish a run already in the results directory, keeping its valid points.")
@click.option("--overwrite", is_flag=True, help="Delete a run already in the results directory and start over.")
@click.option(
    "--reuse-server",
    is_flag=True,
    help="Benchmark a server already running on the configured port instead of launching one.",
)
@click.option(
    "--check",
    is_flag=True,
    help="Validate the server config and preset, print the server command and point count, then exit.",
)
def run(
    server_config: Path, preset: str, output_dir: Path, resume: bool, overwrite: bool, reuse_server: bool, check: bool
) -> None:
    """Launch an API server, run benchmark scenarios, and save results."""
    setup_logger("WARNING" if check else "INFO")  # --check prints only the plan
    if resume and overwrite:
        raise click.UsageError("--resume and --overwrite are mutually exclusive")
    try:
        benchmark = load_preset(preset)
        server = apply_server_overrides(load_server(server_config, default_env=Path(sys.prefix)), benchmark.server)
    except ValueError as e:
        raise click.UsageError(str(e)) from None

    if check:
        _print_check(server, len(benchmark.points))
        return

    logger.info(f"Starting furiosa-perf: {server.backend} serving {server.model} on devices {server.devices}")
    env = environment.collect(server)
    problems = environment.check(env, server)
    if problems:
        raise click.UsageError("; ".join(problems))

    runner = BenchmarkRunner(env, server, benchmark, output_dir, resume=resume, overwrite=overwrite)
    logger.info("Benchmarking started.")
    runner.execute(reuse_server=reuse_server)
    logger.info("Benchmarking completed.")


def _print_check(server: Server, num_points: int) -> None:
    """Print what ``run`` would launch, for validating configs in scripts and CI."""
    argv, env = server_command(server)
    if "CUDA_VISIBLE_DEVICES" in env:
        click.echo(f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} \\")
    click.echo(shlex.join(argv))
    click.echo(f"{num_points} points")
    if not server.executable.exists():
        click.echo(f"warning: {server.executable} does not exist on this host", err=True)


@click.command(name="env")
@click.option(
    "--server-config",
    required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    help="Server config whose environment, devices and model are inspected.",
)
@click.option(
    "--output",
    type=click.Path(file_okay=False, path_type=Path),
    help="Also write the raw probe outputs to this directory (the env/ of a run).",
)
def env_command(server_config: Path, output: Path | None) -> None:
    """Show the environment a run with this server config would record, without running it."""
    setup_logger("WARNING")
    try:
        server = load_server(server_config, default_env=Path(sys.prefix))
    except ValueError as e:
        raise click.UsageError(str(e)) from None
    env = environment.collect(server)
    click.echo(json.dumps(env.summary(server.devices), indent=2, default=str))
    for problem in environment.check(env, server):
        click.echo(f"warning: {problem}", err=True)
    if output is not None:
        env.write_raw(output)
        click.echo(f"raw probe outputs written to {output}", err=True)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli() -> None:
    """furiosa-perf: LLM serving performance benchmark CLI."""


cli.add_command(run)
cli.add_command(env_command)
cli.add_command(report)

if __name__ == "__main__":
    cli()
