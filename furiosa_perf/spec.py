"""Configuration dataclasses and loaders for API server and benchmark runs."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeAlias

import yaml

from furiosa_perf.log import logger

# Serving backend -> accelerator family it runs on.
BACKENDS: dict[str, str] = {"furiosa-llm": "npu", "vllm": "gpu"}

# Server config keys furiosa-perf reads itself; every other key is passed through.
SERVER_FIELDS = ("backend", "model", "devices", "parallel", "env", "host", "port")

# Backend options furiosa-perf sets from ``devices`` and ``parallel``. They are not
# accepted as pass-through keys: their units differ between backends (furiosa-llm
# counts tensor parallelism in PEs), and two sources for one setting can disagree.
PARALLEL_SERVER_ARGS = ("tensor-parallel-size", "pipeline-parallel-size", "data-parallel-size")

# Applied unless the config sets either option: the fixed-length baseline compares
# raw inference performance, which prefix-cache hits on the random prompts would skew.
DEFAULT_SERVER_ARGS: dict[str, Any] = {"no-enable-prefix-caching": True}
PREFIX_CACHING_ARGS = ("enable-prefix-caching", "no-enable-prefix-caching")


@dataclass(frozen=True)
class Parallel:
    """Parallelism of the server, in units of devices (cards).

    ``tp * pp * dp`` equals the number of devices.

    Attributes:
        tp: Devices one model replica's tensors are split across.
        pp: Pipeline stages of one replica.
        dp: Independent replicas.
    """

    tp: int
    pp: int = 1
    dp: int = 1

    @property
    def label(self) -> str:
        """``""`` for pure tensor parallelism, else e.g. ``"tp2-dp2"`` for result names."""
        if self.pp == 1 and self.dp == 1:
            return ""
        parts = [f"tp{self.tp}"] + [f"{name}{size}" for name, size in (("pp", self.pp), ("dp", self.dp)) if size > 1]
        return "-".join(parts)


def parse_parallel(raw: Any, num_devices: int) -> Parallel:
    """Resolve ``parallel`` against the device count.

    Any one of ``tp``, ``pp`` and ``dp`` may be omitted and is then derived; ``pp``
    and ``dp`` default to 1, so an absent ``parallel`` is tensor parallelism over
    every device.

    Args:
        raw: Value of ``parallel`` in the server config (a mapping or ``None``).
        num_devices: Number of entries in ``devices``.

    Returns:
        The resolved parallelism.

    Raises:
        ValueError: If a key is unknown, a size is not a positive integer, or the
            sizes do not multiply to ``num_devices``.
    """
    raw = raw or {}
    if not isinstance(raw, dict) or set(raw) - {"tp", "pp", "dp"}:
        raise ValueError(f"the parallel setting '{raw}' is invalid; use a mapping with keys tp, pp and dp")
    sizes = {}
    for key, value in raw.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"the parallel size {key}={value!r} is invalid; use a positive integer")
        sizes[key] = value

    pp, dp = sizes.get("pp", 1), sizes.get("dp", 1)
    tp = sizes.get("tp")
    if tp is None:
        tp = num_devices // (pp * dp) if num_devices % (pp * dp) == 0 else 0
    elif "dp" not in sizes and num_devices % (tp * pp) == 0:
        dp = num_devices // (tp * pp)
    if tp < 1 or tp * pp * dp != num_devices:
        raise ValueError(
            f"the parallel setting {raw} does not fit {num_devices} devices; tp * pp * dp should equal the device count"
        )
    return Parallel(tp=tp, pp=pp, dp=dp)


@dataclass(frozen=True)
class Server:
    """How to launch the LLM API server.

    Attributes:
        backend: ``furiosa-llm`` or ``vllm``.
        model: Hugging Face model ID or local path to serve.
        devices: Accelerator indices to serve on.
        parallel: How the devices are split into tensor/pipeline/data parallelism.
        env: Root of the Python environment whose ``bin/<backend>`` is launched. Its
            interpreter is also where the backend version is read from, so the
            recorded version is the one that actually served.
        host: Address the server binds and the client connects to.
        port: Server port.
        args: Remaining options, passed to ``<backend> serve`` as ``--<key> <value>``.
    """

    backend: str
    model: str
    devices: list[int]
    parallel: Parallel
    env: Path
    host: str = "0.0.0.0"
    port: int = 8000
    args: dict[str, Any] = field(default_factory=dict)

    @property
    def hardware_type(self) -> str:
        """Accelerator family of the backend: ``npu`` or ``gpu``."""
        return BACKENDS[self.backend]

    @property
    def executable(self) -> Path:
        """The backend's command-line entry point inside :attr:`env`."""
        return self.env / "bin" / self.backend

    @property
    def python(self) -> Path:
        """The interpreter of :attr:`env`."""
        return self.env / "bin" / "python"


def parse_devices(raw: Any) -> list[int]:
    """Read ``devices`` given as ``0``, ``"0,1"`` or ``[0, 1]``.

    Args:
        raw: Value of ``devices`` in the server config.

    Returns:
        The device indices in the given order.

    Raises:
        ValueError: If the value is empty, not a list of non-negative integers, or has duplicates.
    """
    items = raw if isinstance(raw, list) else str(raw).split(",")
    try:
        devices = [int(str(item).strip()) for item in items]
    except ValueError:
        devices = []
    if not devices or min(devices) < 0 or len(set(devices)) != len(devices):
        raise ValueError(f"the device list '{raw}' is invalid; use distinct indices such as 0,1,2,3")
    return devices


def parse_server(raw: dict[str, Any], default_env: Path) -> Server:
    """Build a :class:`Server` from the parsed server config.

    Keys may be written in snake_case or kebab-case; pass-through keys are
    normalised to kebab-case, the form both backends' command lines use.

    Args:
        raw: Mapping loaded from the server config YAML.
        default_env: Environment to use when ``env`` is not set.

    Returns:
        The validated server description.

    Raises:
        ValueError: If a required key is missing, the backend is unsupported, the
            parallelism does not fit the devices, or a pass-through key sets
            parallelism directly.
    """
    config = {key.replace("_", "-"): value for key, value in raw.items()}
    for key in ("backend", "model", "devices"):
        if config.get(key) in (None, ""):
            raise ValueError(f"the server config is missing '{key}'")
    backend = str(config["backend"])
    if backend not in BACKENDS:
        raise ValueError(f"the backend '{backend}' is unsupported; use one of {', '.join(BACKENDS)}")

    args = {key: value for key, value in config.items() if key not in SERVER_FIELDS}
    for key in args:
        if key in PARALLEL_SERVER_ARGS:
            raise ValueError(
                f"the server option '{key}' is not allowed; set 'parallel: {{tp, pp, dp}}' in units of devices instead"
            )
    if not any(key in args for key in PREFIX_CACHING_ARGS):
        args = {**DEFAULT_SERVER_ARGS, **args}

    devices = parse_devices(config["devices"])
    return Server(
        backend=backend,
        model=str(config["model"]),
        devices=devices,
        parallel=parse_parallel(config.get("parallel"), len(devices)),
        env=Path(config.get("env") or default_env).expanduser(),
        host=str(config.get("host", "0.0.0.0")),
        port=int(config.get("port", 8000)),
        args=args,
    )


def load_server(path: Path, default_env: Path) -> Server:
    """Load and validate a server config YAML.

    Args:
        path: Server config file.
        default_env: Environment to use when the config does not set ``env``.

    Returns:
        The validated server description.

    Raises:
        ValueError: If the file is not a mapping or fails :func:`parse_server`.
    """
    with path.open() as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"the server config {path} should be a mapping of option names to values")
    try:
        server = parse_server(raw, default_env)
    except ValueError as e:
        raise ValueError(f"{path}: {e}") from None
    logger.info(f"Loaded server config: {server}")
    return server


@dataclass
class BaseScenarioConfig:
    """Fields shared by every vLLM ``bench serve`` scenario.

    These map onto the placeholders in ``VllmPerformanceBenchmark.COMMAND``
    (``--max-concurrency``, ``--num-prompts``, ``--request-rate``) plus the
    input/output token counts used for result-directory naming and parsing.
    """

    input_tokens: int = 1024
    max_concurrency: int = 1
    num_prompts: int | None = None
    request_rate: int | str = "inf"
    model: str = ""


@dataclass
class LLMScenarioConfig(BaseScenarioConfig):
    """Scenario config for the ``offline`` task."""

    random_range_ratio: float = 0.0
    output_tokens: int = 1024

    def __post_init__(self) -> None:
        """Default num_prompts to max_concurrency × 3 when not set."""
        if self.num_prompts is None:
            self.num_prompts = self.max_concurrency * 3


@dataclass
class VLScenarioConfig(LLMScenarioConfig):
    """Scenario config for the ``vl-offline`` (vision-language) task."""

    random_range_ratio: float = 0.0
    random_mm_base_items_per_request: int = 1
    random_mm_bucket_config: str = "{(256, 256, 1): 1.0}"
    random_mm_limit_mm_per_prompt: str = '{"image": 1}'


@dataclass
class RerankerScenarioConfig(BaseScenarioConfig):
    """Scenario config for the ``reranker`` task."""

    random_batch_size: int = 5

    def __post_init__(self) -> None:
        """Default num_prompts based on request_rate when not set."""
        if self.num_prompts is None:
            if isinstance(self.request_rate, (int, float)):
                self.num_prompts = int(self.request_rate * 3)
            else:
                # "inf" or any non-numeric rate: fall back to a small finite run size.
                self.num_prompts = self.max_concurrency * 3


@dataclass
class EmbeddingScenarioConfig(BaseScenarioConfig):
    """Scenario config for the ``embeddings`` task."""

    def __post_init__(self) -> None:
        """Default num_prompts based on request_rate when not set."""
        if self.num_prompts is None:
            if isinstance(self.request_rate, (int, float)):
                self.num_prompts = int(self.request_rate * 3)
            else:
                # "inf" or any non-numeric rate: fall back to a small finite run size.
                self.num_prompts = self.max_concurrency * 3


ScenarioConfig: TypeAlias = (
    LLMScenarioConfig | VLScenarioConfig | RerankerScenarioConfig | EmbeddingScenarioConfig
)


@dataclass
class PerformanceBenchConfig:
    """Top-level benchmark configuration holding all scenarios to run."""

    name: str
    task: str
    model: str = ""
    device_name: str = ""
    used_device_num: int = 1
    parallel_label: str = ""  # Parallel.label; empty for pure tensor parallelism
    scenarios: list[ScenarioConfig] = field(default_factory=list)


class PerformanceBenchConfigLoader:
    """Load and construct a :class:`PerformanceBenchConfig` from a YAML file."""

    CONFIG_REGISTRY: dict[str, type[ScenarioConfig]] = {
        "offline": LLMScenarioConfig,
        "vl-offline": VLScenarioConfig,
        "reranker": RerankerScenarioConfig,
        "embeddings": EmbeddingScenarioConfig,
    }

    @classmethod
    def _create_config(cls, configs: dict[str, Any]) -> PerformanceBenchConfig:
        """Parse raw YAML data into a :class:`PerformanceBenchConfig`.

        Args:
            configs: Raw YAML dict containing ``name``, ``task``, and ``scenarios``.

        Returns:
            Fully expanded :class:`PerformanceBenchConfig` with one
            :class:`ScenarioConfig` per concurrency level.

        Raises:
            ValueError: If the task name is not registered.
        """
        task = configs.get("task")
        scenario_class = cls.CONFIG_REGISTRY.get(task)  # type: ignore[arg-type]
        if scenario_class is None:
            raise ValueError(f"No scenario config registered for task {task!r}.")

        base = {k: v for k, v in configs.items() if k != "scenarios"}
        expanded: list[ScenarioConfig] = []
        for scenario in configs.get("scenarios", []):
            concurrencies = scenario["max_concurrency"]
            if isinstance(concurrencies, list):
                base_fields = {k: v for k, v in scenario.items() if k != "max_concurrency"}
                expanded.extend(scenario_class(**base_fields, max_concurrency=c) for c in concurrencies)
            else:
                expanded.append(scenario_class(**scenario))

        return PerformanceBenchConfig(**base, scenarios=expanded)

    @classmethod
    def benchmark_config_setup(cls, benchmark_config_path: Path) -> PerformanceBenchConfig:
        """Load a benchmark config YAML and return the parsed config object.

        Args:
            benchmark_config_path: Path to the YAML configuration file.

        Returns:
            Parsed :class:`PerformanceBenchConfig` instance.

        Raises:
            FileNotFoundError: If the config file does not exist.
        """
        if not benchmark_config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {benchmark_config_path}")

        with benchmark_config_path.open() as f:
            benchmark_config = cls._create_config(yaml.safe_load(f))

        logger.info(f"Loaded benchmark config: {benchmark_config}")
        return benchmark_config
