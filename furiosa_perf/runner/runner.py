"""Benchmark orchestration: server -> monitor -> benchmark client -> teardown."""

import os
import signal
from typing import Any

import requests

from furiosa_perf.clients.vllm_bench import VllmPerformanceBenchmark
from furiosa_perf.log import logger
from furiosa_perf.runner.monitor import HardwareMonitor, MonitorConfig
from furiosa_perf.runner.server import APIServerManager
from furiosa_perf.spec import PerformanceBenchConfig, Server


class BenchmarkRunner:
    """Orchestrate the full benchmark lifecycle: server → benchmark → teardown.

    Owns the :class:`APIServerManager` and :class:`VllmPerformanceBenchmark`
    instances and coordinates their startup, monitoring, and cleanup.
    """

    def __init__(self, system_info: Any, server: Server, benchmark_config: PerformanceBenchConfig) -> None:
        """Store configuration for later use in :meth:`execute`.

        Args:
            system_info: :class:`SystemDetector.SystemEnv` namedtuple with OS/HW info.
            server: What API server to launch.
            benchmark_config: Parsed benchmark config dataclass with scenarios.
        """
        self.system_info = system_info
        self.server = server
        self.benchmark_config = benchmark_config

    def execute(self, reuse_server: bool = False) -> list[Any]:
        """Run the full benchmark pipeline and return per-scenario results.

        Sequence:
            1. Start the API server.
            2. Query ``/v1/models`` to resolve the served model ID.
            3. Set up and run the vLLM benchmark.
            4. Start hardware monitoring alongside the benchmark.
            5. On completion or error, stop all components in reverse order.

        Args:
            reuse_server: Use a server already running on the configured port.

        Returns:
            List of per-scenario result dicts from :attr:`VllmPerformanceBenchmark.total_results`.

        Raises:
            Exception: Re-raises any exception from server startup or benchmark execution.
        """
        api_server = APIServerManager(self.server)
        benchmark: VllmPerformanceBenchmark | None = None
        monitor: HardwareMonitor | None = None
        desc = ""

        try:
            logger.info(f"Starting server for {self.server.model}")
            server_command = api_server.start(reuse_running=reuse_server)
            desc = (
                f"* Ubuntu Version: {self.system_info.os}\n"
                f"* Python Version: {self.system_info.python}\n"
                f"* Framework Version: {self.system_info.runtime}\n"
                f"* API Server Command: {server_command}"
            )

            pretrained_id = self._fetch_pretrained_id(api_server)
            self._configure_benchmark(pretrained_id)

            benchmark = VllmPerformanceBenchmark(
                config=self.benchmark_config,
                backend=self.system_info.runtime,
                host=self.server.host,
                port=self.server.port,
                env=os.environ.copy(),
            )
            benchmark.setup()

            monitor = HardwareMonitor(
                MonitorConfig(
                    host=self.server.host,
                    port=self.server.port,
                    server_pid=api_server.server_pid,
                    device_name=benchmark.device_name,
                    device_indices=self.server.devices,
                    log_dir=benchmark.get_vllm_result_dir(""),
                )
            )
            monitor.start()
            benchmark.run(monitor)

        except Exception:
            logger.exception("Benchmark execution failed.")
            raise
        finally:
            old_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
            try:
                self._cleanup(benchmark, monitor, api_server, desc)
            finally:
                signal.signal(signal.SIGINT, old_sigint)

        return benchmark.total_results["results"]  # type: ignore[no-any-return]

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

    def _configure_benchmark(self, pretrained_id: str) -> None:
        """Populate benchmark_config fields resolved at runtime.

        Args:
            pretrained_id: Model ID returned by the running API server.
        """
        self.benchmark_config.model = pretrained_id
        self.benchmark_config.device_name = self.system_info.hardware[self.server.hardware_type]["name"]
        self.benchmark_config.used_device_num = len(self.server.devices)
        self.benchmark_config.parallel_label = self.server.parallel.label

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
