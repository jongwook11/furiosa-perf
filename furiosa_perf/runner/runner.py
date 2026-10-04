"""Benchmark orchestration: server -> monitor -> benchmark client -> teardown."""

import signal
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from furiosa_perf.clients.vllm_bench import VLLM_BENCH_VERSION, VllmPerformanceBenchmark
from furiosa_perf.log import logger
from furiosa_perf.results.store import ENV_DIR, RUN_FILE, SERVER_LOG, prepare_run_dir, run_dir, write_json
from furiosa_perf.runner.env import Environment, device_label, freeze, power_settings
from furiosa_perf.runner.monitor import HardwareMonitor, MonitorConfig
from furiosa_perf.runner.server import APIServerManager
from furiosa_perf.spec import Preset, Server


class BenchmarkRunner:
    """Orchestrate the full benchmark lifecycle: server → benchmark → teardown.

    Owns the :class:`APIServerManager` and :class:`VllmPerformanceBenchmark`
    instances, coordinates their startup, monitoring and cleanup, and keeps
    ``run.json`` of the run directory up to date.
    """

    def __init__(
        self,
        environment: Environment,
        server: Server,
        preset: Preset,
        output_dir: Path,
        resume: bool = False,
        overwrite: bool = False,
    ) -> None:
        """Resolve the run directory and the fields every record carries.

        Args:
            environment: Collected environment (see :func:`furiosa_perf.runner.env.collect`).
            server: What API server to launch, with the preset's server options applied.
            preset: What to measure.
            output_dir: Root of the results tree.
            resume: Continue a run already in the run directory.
            overwrite: Replace a run already in the run directory.
        """
        self.environment = environment
        self.server = server
        self.preset = preset
        self.resume = resume
        self.overwrite = overwrite

        used = environment.used_devices(server.devices)
        product = used[0]["name"] if used else "Unknown"
        backend, backend_version = server.backend, str(environment.server_env.get("version"))
        # The served ID is known before launch, so a conflicting run is found without starting the server.
        self.model = str(server.args.get("served-model-name") or server.model)
        self.identity: dict[str, Any] = {
            "kind": preset.kind,
            "task": preset.task,
            "model": self.model,
            "hardware": device_label(product, used),
            "hardware_product": product,
            "num_devices": len(server.devices),
            "devices": server.devices,
            "layout": server.parallel.label,
            "parallel": asdict(server.parallel),
            "backend": backend,
            "backend_version": backend_version,
        }
        self.run_dir = run_dir(
            output_dir,
            preset.kind,
            self.identity["hardware"],
            len(server.devices),
            server.parallel.label,
            f"{backend}_{backend_version}",
            preset.task,
            self.model,
        )

    def execute(self, reuse_server: bool = False) -> list[Any]:
        """Run the full benchmark pipeline and return per-point summary rows.

        Sequence:
            1. Prepare the run directory (refusing to mix runs) and write ``run.json``.
            2. Start the API server.
            3. Set up the client and start hardware monitoring.
            4. Run every point.
            5. On completion or error, stop everything and finish ``run.json``.

        Args:
            reuse_server: Use a server already running on the configured port.

        Returns:
            List of per-point summary rows.

        Raises:
            Exception: Re-raises any exception from server startup or benchmark execution.
        """
        prepare_run_dir(self.run_dir, self.resume, self.overwrite)
        logger.info(f"Results: {self.run_dir}")
        run_info = self._run_info()
        write_json(self.run_dir / RUN_FILE, run_info)
        self.environment.write_raw(self.run_dir / ENV_DIR)

        api_server = APIServerManager(self.server, log_path=self.run_dir / SERVER_LOG)
        benchmark: VllmPerformanceBenchmark | None = None
        monitor: HardwareMonitor | None = None
        desc = ""
        status = "failed"

        try:
            logger.info(f"Starting server for {self.server.model}")
            server_command = api_server.start(reuse_running=reuse_server)
            run_info["server"]["command"] = server_command
            write_json(self.run_dir / RUN_FILE, run_info)
            desc = (
                f"* OS: {self.environment.host.get('os')}\n"
                f"* Python Version: {self.environment.server_env.get('python_version')}\n"
                f"* Framework Version: {self.identity['backend']} {self.identity['backend_version']}\n"
                f"* API Server Command: {server_command}"
            )
            served = self._fetch_pretrained_id(api_server)
            if served != self.model:
                logger.warning(f"the server reports model '{served}', the results are stored as '{self.model}'")

            benchmark = VllmPerformanceBenchmark(
                preset=self.preset,
                model=served,
                host=self.server.host,
                port=self.server.port,
                run_dir=self.run_dir,
                identity=self.identity,
                resume=self.resume,
            )
            benchmark.setup()
            if benchmark.python is not None:
                (self.run_dir / ENV_DIR / "client-freeze.txt").write_text(freeze(benchmark.python))

            monitor = HardwareMonitor(
                MonitorConfig(
                    host=self.server.host,
                    port=self.server.port,
                    server_pid=api_server.server_pid,
                    # The monitor picks its sampler by name: "RNGD" for every NPU, the product for GPUs.
                    device_name="RNGD" if self.server.hardware_type == "npu" else self.identity["hardware_product"],
                    device_indices=self.server.devices,
                    log_dir=self.run_dir,
                )
            )
            monitor.start()
            benchmark.run(monitor, server_alive=lambda: api_server.served_model() is not None)
            status = "finished"

        except KeyboardInterrupt:
            status = "interrupted"
            raise
        except Exception:
            logger.exception("Benchmark execution failed.")
            raise
        finally:
            old_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
            try:
                self._cleanup(benchmark, monitor, api_server, desc)
                rows = benchmark.total_results["results"] if benchmark is not None else []
                run_info["power_settings_changed"] = self._power_settings_changed()
                run_info.update(
                    status=status,
                    finished=datetime.now(timezone.utc).isoformat(),
                    num_points=len(rows),
                    num_invalid=sum(not row["ok"] for row in rows),
                )
                write_json(self.run_dir / RUN_FILE, run_info)
            finally:
                signal.signal(signal.SIGINT, old_sigint)

        return benchmark.total_results["results"]  # type: ignore[no-any-return]

    def _power_settings_changed(self) -> bool | None:
        """Whether a used device's power limit or governor changed since the run started.

        Power and efficiency figures assume the settings recorded at the start; a
        change in between is logged and flagged in ``run.json``.
        """
        keys = ("index", "governor") if self.server.hardware_type == "npu" else ("index", "power_limit_w")
        before = [{key: d.get(key) for key in keys} for d in self.environment.used_devices(self.server.devices)]
        try:
            after = power_settings(self.server.hardware_type, self.server.devices)
        except Exception as e:
            logger.warning(f"cannot re-read the power settings at the end of the run; cause: {e}")
            return None
        if after != before:
            logger.warning(f"the device power settings changed during the run: {before} -> {after}")
        return after != before

    def _run_info(self) -> dict[str, Any]:
        """Initial content of ``run.json``: what is run, on what, with which versions."""
        return {
            **self.identity,
            "status": "running",
            "started": datetime.now(timezone.utc).isoformat(),
            "finished": None,
            "client": self.preset.client,
            "client_version": VLLM_BENCH_VERSION,
            "environment": self.environment.summary(self.server.devices),
            "server": {
                "backend": self.server.backend,
                "model": self.server.model,
                "env": str(self.server.env),
                "host": self.server.host,
                "port": self.server.port,
                "args": self.server.args,
                "command": None,
            },
            "preset": {"server": self.preset.server, "points": self.preset.points},
            "num_points": len(self.preset.points),
            "num_invalid": None,
        }

    def _fetch_pretrained_id(self, api_server: APIServerManager) -> str:
        """Query ``/v1/models`` and return the first model's ID.

        Args:
            api_server: Running server manager whose host/port to query.

        Returns:
            The ``id`` field of the first model returned by the server.

        Raises:
            requests.HTTPError: If the server responds with a non-2xx status.
        """
        resp = requests.get(f"{api_server.base_url}/v1/models", timeout=5)
        resp.raise_for_status()
        return resp.json()["data"][0]["id"]  # type: ignore[no-any-return]

    def _cleanup(
        self,
        benchmark: VllmPerformanceBenchmark | None,
        monitor: HardwareMonitor | None,
        api_server: APIServerManager | None,
        desc: str,
    ) -> None:
        """Stop all components in reverse start order, logging failures individually.

        Args:
            benchmark: Benchmark instance to stop and finalise (may be ``None``).
            monitor: Hardware monitor to stop (may be ``None``).
            api_server: API server to stop (may be ``None``).
            desc: Passed through to :meth:`VllmPerformanceBenchmark.finish_info`.
        """
        if benchmark is not None:
            try:
                benchmark.stop()
            except Exception:
                logger.exception("Failed to stop benchmark.")

        if monitor is not None:
            try:
                monitor.stop()
            except Exception:
                logger.exception("Failed to stop monitoring process.")

        if api_server is not None:
            try:
                api_server.stop()
            except Exception:
                logger.exception("Failed to stop API server.")

        if benchmark is not None:
            try:
                benchmark.finish_info(desc)
            except Exception:
                logger.exception("Failed to finalise benchmark info.")
