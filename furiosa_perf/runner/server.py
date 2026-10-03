"""Launch, health-check and stop the LLM API server process."""

import os
import signal
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import TextIO

import psutil
import requests

from furiosa_perf.log import logger
from furiosa_perf.spec import (
    APIServerConfig,
    FuriosaLLMServerConfig,
    VllmServerConfig,
)


class APIServerManager:
    """Manage the lifecycle of a vLLM or furiosa-llm API server subprocess.

    Handles starting, health-checking, and stopping the server process, and
    streams server logs to the logger while waiting for startup.
    """

    def __init__(
        self,
        config: APIServerConfig,
        model: str = "furiosa-ai/Llama-3.1-8B-Instruct",
    ) -> None:
        """Initialise the manager without starting the server.

        Args:
            config: Server configuration (host, port, devices, …).
            model: HuggingFace model ID or local path to serve.
        """
        self.model = model
        self.config = config
        self.server_process: subprocess.Popen[str] | None = None
        self.server_ready = False
        self.server_pid = -1
        self._log_file: TextIO | None = None
        self._log_path: Path | None = None

    def __del__(self) -> None:
        self.stop()

    def start(self) -> str:
        """Start the API server and wait until it is ready.

        Returns:
            The full server command as a space-joined string, or ``""`` if the
            server was already running.

        Raises:
            ValueError: If the command cannot be built from the config.
            RuntimeError: If the server does not become healthy within the timeout.
        """
        if self._is_api_server_ready(self.model):
            logger.warning(f"API server already running for model: {self.model}")
            self._get_opened_server_pid()
            return ""

        if self.server_process and self.server_process.poll() is None:
            logger.warning("Server already running.")
            self._get_opened_server_pid()
            return ""

        command, env = self._build_command()
        if not command:
            raise ValueError("Failed to build server command.")

        logger.info(f"Starting server: {' '.join(command)}")

        log_dir = Path("./serve_logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        safe_model_name = self.model.replace("/", "_")
        self._log_path = log_dir / f"serve_{safe_model_name}_{int(time.time())}.log"
        logger.info(f"Server log: {self._log_path}")
        self._log_file = open(self._log_path, "w", buffering=1)  # noqa: SIM115, PTH123

        env["PYTHONUNBUFFERED"] = "1"
        self.server_process = subprocess.Popen(
            command,
            env=env,
            stdout=self._log_file,
            stderr=self._log_file,
            text=True,
            start_new_session=True,
        )

        self._wait_for_startup(url=f"http://{self.config.host}:{self.config.port}/v1/models")
        self.server_ready = True
        self.server_pid = self.server_process.pid
        logger.info("Server is ready.")
        return " ".join(command)

    def stop(self) -> None:
        """Terminate the server process gracefully, falling back to SIGKILL."""
        if not self.server_process:
            return

        if self.server_process.poll() is None:
            logger.info("Stopping server process…")
            try:
                os.killpg(os.getpgid(self.server_process.pid), signal.SIGTERM)
                self.server_process.wait(timeout=5)
                logger.info("Server process stopped successfully.")
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self.server_process.pid), signal.SIGKILL)
                logger.warning("Server process force-killed.")
        else:
            logger.info(f"Server already exited with code {self.server_process.poll()}.")

        self.server_process = None
        self.server_ready = False
        if self._log_file:
            self._log_file.close()
            self._log_file = None

    def _to_kebab_case(self, s: str) -> str:
        """Convert an underscore-separated string to kebab-case."""
        return s.replace("_", "-")

    def _build_command(self) -> tuple[list[str], dict[str, str]]:
        """Construct the server CLI command and environment from the config.

        Returns:
            A tuple of ``(command_argv, env_dict)``.
        """
        env = os.environ.copy()
        command: list[str] = []

        if isinstance(self.config, VllmServerConfig):
            command = ["vllm", "serve", self.model]
            for key, value in asdict(self.config).items():
                if value is None:
                    continue
                cli_key = self._to_kebab_case(key)
                if isinstance(value, bool):
                    if value:
                        command.append(f"--{cli_key}")
                elif isinstance(value, int | str | float):
                    if isinstance(value, str) and cli_key == "devices":
                        env["CUDA_VISIBLE_DEVICES"] = value
                        continue
                    command.extend([f"--{cli_key}", str(value)])
                else:
                    logger.warning(f"Unsupported config type for {key}: {type(value)}")

        elif isinstance(self.config, FuriosaLLMServerConfig):
            command = ["furiosa-llm", "serve", self.model]
            for key, value in asdict(self.config).items():
                if value is None:
                    continue
                cli_key = self._to_kebab_case(key)
                if isinstance(value, bool):
                    if value:
                        command.append(f"--{cli_key}")
                elif isinstance(value, int | str | float):
                    if isinstance(value, str) and cli_key == "devices":
                        # Convert "0,1,2,3" → "npu:0,npu:1,npu:2,npu:3"
                        value = ",".join(f"npu:{idx}" for idx in value.split(","))
                    if cli_key == "tensor-parallel-size":
                        # furiosa-llm counts PEs (8 per NPU); vLLM counts NPUs.
                        value = value * 8
                    command.extend([f"--{cli_key}", str(value)])
                else:
                    logger.warning(f"Unsupported config type for {key}: {type(value)}")
        else:
            logger.error(f"Unsupported API server config type: {type(self.config)}")

        return command, env

    def _is_api_server_ready(self, expected_model_id: str) -> bool:
        """Return True when the server is already serving *expected_model_id*.

        Args:
            expected_model_id: Model ID to match against ``/v1/models`` response.
        """
        url = f"http://{self.config.host}:{self.config.port}/v1/models"
        try:
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            model_id = resp.json()["data"][0]["id"]
            return model_id == expected_model_id  # type: ignore[no-any-return]
        except (requests.RequestException, KeyError, IndexError, ValueError):
            return False

    def _wait_for_startup(
        self, url: str = "http://localhost:8000/v1/models", timeout: int = 1800
    ) -> None:
        """Poll *url* until the server responds 200, streaming its log to the logger.

        Args:
            url: Health-check URL to poll.
            timeout: Maximum seconds to wait before raising.

        Raises:
            RuntimeError: If the server process exits early or the timeout elapses.
        """
        start = time.time()
        log_pos = 0

        while True:
            if self._log_path and self._log_path.exists():
                with self._log_path.open() as f:
                    f.seek(log_pos)
                    for line in f:
                        logger.info(f"[server] {line.rstrip()}")
                    log_pos = f.tell()

            if self.server_process and self.server_process.poll() is not None:
                raise RuntimeError(
                    f"Server exited unexpectedly (code {self.server_process.poll()}). "
                    f"See {self._log_path}"
                )

            try:
                resp = requests.get(url, timeout=5)
                if resp.status_code == 200:
                    logger.info("Server startup complete.")
                    return
            except requests.RequestException as e:
                logger.debug(f"Server not ready yet: {e}")

            if time.time() - start > timeout:
                raise RuntimeError(
                    f"Server did not start within {timeout}s. See {self._log_path}"
                )

            logger.info("Waiting for server launch — checking again in 30 s…")
            time.sleep(30)

    def _get_opened_server_pid(self) -> None:
        """Populate :attr:`server_pid` by scanning running processes by name."""
        if isinstance(self.config, VllmServerConfig):
            process_name = "vllm"
        elif isinstance(self.config, FuriosaLLMServerConfig):
            process_name = "furiosa-llm"
        else:
            return

        for proc in psutil.process_iter(["pid", "name"]):
            if proc.info["name"] == process_name:
                self.server_pid = proc.info["pid"]
