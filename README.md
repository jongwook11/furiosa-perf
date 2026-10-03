# furiosa-perf

`furiosa-perf` is a performance benchmarking CLI for **LLM serving**.

It automates the full benchmark pipeline: launch an API server → run [vLLM benchmark](https://github.com/vllm-project/vllm/tree/main/benchmarks) scenarios → collect hardware metrics → save CSV/Markdown summaries.

**Supported backends:**

| Backend | Hardware |
|---|---|
| `furiosa-llm` | Furiosa RNGD (NPU) |
| `vllm` | NVIDIA GPU |

See [Benchmark Methodology](examples/BENCHMARK.md) for a detailed explanation of the benchmark design, metrics, and how to interpret results.

---

## Key features

- **End-to-end automation** — server launch, benchmark execution, and teardown in one command
- **Hardware monitoring** — power, temperature, utilization (NPU/GPU), host CPU/memory, and server `/metrics`, logged to CSV
- **Multiple tasks** — `offline` (LLM text), `vl-offline` (vision-language), `reranker`, `embeddings`
- **Flexible scenarios** — define token lengths and concurrency lists in a single YAML; `max_concurrency` can be a list and is auto-expanded

---

## Requirements

- **OS**: Linux
- **Python**: 3.10+ (the default `python3` on Ubuntu 22.04 works)
- **APT**: `python3-venv`
- **[uv](https://docs.astral.sh/uv/)**: used internally to create a separate venv for the vLLM benchmark tool

```bash
# ubuntu 22.04
sudo apt install python3-venv -y
pip install uv
```

### Backend prerequisites

**Furiosa NPU + `furiosa-llm`**

```bash
pip install furiosa-llm
```

**NVIDIA GPU + vLLM**

```bash
pip install vllm
# nvidia-smi must be available for GPU monitoring
```

---

## Installation

```bash
git clone https://github.com/furiosa-ai/furiosa-perf.git
cd furiosa-perf

python3 -m venv .venv
source .venv/bin/activate

pip install -e .
```

Backend-specific extras:

```bash
pip install ".[furiosa-llm]"   # Furiosa NPU
pip install ".[vllm]"          # NVIDIA GPU
```

> **Note:** activate the venv (`source .venv/bin/activate`) before every session, or add it to your shell profile.

### HuggingFace token (gated models)

```bash
export HF_TOKEN=hf_...
# or
huggingface-cli login
```

---

## Quick start

```bash
furiosa-perf run \
  --server-config examples/server/exaone4.yaml \
  --benchmark-config examples/bench/llm_scenario.yaml
```

See [`examples/script.sh`](examples/script.sh) for a complete working example.

### CLI options

| Option | Required | Default | Description |
|---|---|---|---|
| `--server-config` | ✓ | — | API server config YAML (backend, model, devices, …) |
| `--benchmark-config` | ✓ | — | Benchmark scenario config YAML |
| `--reuse-server` | | off | Benchmark a server already running on the configured port instead of launching one |
| `--check` | | off | Validate both configs, print the server command and scenario count, and exit (no hardware needed) |

---

## Config reference

### Server config (`--server-config`)

Controls how the API server is launched.

**Example** (`examples/server/exaone4.yaml`):

```yaml
backend: furiosa-llm
model: furiosa-ai/EXAONE-4.0-32B-FP8
devices: 0,1,2,3
```

**Keys furiosa-perf reads:**

| Key | Default | Description |
|---|---|---|
| `backend` | — (required) | `furiosa-llm` (RNGD) or `vllm` (NVIDIA GPU) |
| `model` | — (required) | Hugging Face model ID or local path to serve |
| `devices` | — (required) | Device indices to serve on, e.g. `0,1,2,3` |
| `parallel` | `{tp: <number of devices>}` | Split of the devices into tensor (`tp`), pipeline (`pp`) and data (`dp`) parallelism, in units of devices; `tp * pp * dp` must equal the device count and one of them may be omitted |
| `env` | the environment furiosa-perf runs in | Python environment whose `bin/<backend>` is launched; the recorded backend version is read from it |
| `host` | `0.0.0.0` | Server bind address |
| `port` | `8000` | Server port |

**Every other key** is passed to `<backend> serve` as `--<key> <value>` (`true` → bare flag, `false` → omitted; snake_case keys are converted to kebab-case), for example `served-model-name`, `max-model-len` or `enable-prefix-caching`. Prefix caching is disabled unless one of `enable-prefix-caching` / `no-enable-prefix-caching` is set, because the fixed-length benchmarks compare raw inference performance. The parallel sizes are set only through `parallel`: the backends count them in different units (furiosa-llm counts tensor parallelism in PEs, 8 per RNGD).

```yaml
# Two RNGDs as two independent replicas, served from a dedicated SDK environment
backend: furiosa-llm
model: furiosa-ai/Qwen3-8B-FP8
devices: 0,1
parallel: {dp: 2}
env: /opt/venvs/furiosa-llm-2026.4.0
max-model-len: 32768
```

**Choosing `devices` (Furiosa RNGD)**

`devices` specifies which RNGD cards to use and how many to allocate for the model. By default all of them form one tensor-parallel replica; use `parallel` to split them differently. Larger models require more cards per replica.

First, check how many RNGDs are available in your environment:

```bash
$ furiosa-smi info
+------+--------+-------------------+---------+---------+--------------+
| Arch | Device | Firmware          | Temp.   | Power   | PCI-BDF      |
+------+--------+-------------------+---------+---------+--------------+
| rngd | npu0   | 2026.2.0, a5db283 | 36.33°C | 39.00 W | 0000:03:00.0 |
+------+--------+-------------------+---------+---------+--------------+
| rngd | npu1   | 2026.2.0, a5db283 | 35.99°C | 38.00 W | 0000:04:00.0 |
+------+--------+-------------------+---------+---------+--------------+
| rngd | npu2   | 2026.2.0, a5db283 | 34.03°C | 39.00 W | 0000:44:00.0 |
+------+--------+-------------------+---------+---------+--------------+
| rngd | npu3   | 2026.2.0, a5db283 | 35.76°C | 39.00 W | 0000:45:00.0 |
+------+--------+-------------------+---------+---------+--------------+
```

Use the numbers from the **Device** column (`npu0` → `0`, `npu1` → `1`, …) as the `devices` value. For example, to use the first 4 cards: `devices: 0,1,2,3`. If two users share an 8-card server, one can use `0,1,2,3` and the other `4,5,6,7`.

**Supported models and recommended device count**

Furiosa AI supported models are available at [huggingface.co/furiosa-ai](https://huggingface.co/furiosa-ai).

| Model (HuggingFace ID) | Required RNGDs | Example `devices` |
|---|:---:|---|
| `furiosa-ai/EXAONE-4.0-32B-FP8` | 4 | `0,1,2,3` or `4,5,6,7` |
| `furiosa-ai/Llama-3.3-70B-Instruct` | 4 | `0,1,2,3` or `4,5,6,7` |
| `furiosa-ai/Qwen3-32B-FP8` | 4 | `0,1,2,3` or `4,5,6,7` |
| `furiosa-ai/Llama-3.1-8B-Instruct` | 1 | `0` or any device available in your environment |
| `furiosa-ai/Qwen2.5-0.5B-Instruct` | 1 | `0` or any device available in your environment |
| `furiosa-ai/Qwen3-Embedding-8B` | 1 | `0` or any device available in your environment |
| `furiosa-ai/Qwen3-Reranker-8B` | 1 | `0` or any device available in your environment |

**Notes:**

- `furiosa-llm` backend: `devices: 0,1,2,3` → `--devices npu:0,npu:1,npu:2,npu:3`, and `parallel` → `--tensor-parallel-size <tp × 8> --pipeline-parallel-size <pp> --data-parallel-size <dp>`
- `vllm` backend: `devices` sets `CUDA_VISIBLE_DEVICES`, and `parallel` → `--tensor-parallel-size <tp> --pipeline-parallel-size <pp> --data-parallel-size <dp>`
- Results of non-default layouts get the layout in their directory name, e.g. `RNGD_4-tp2-dp2_furiosa-llm_<version>`

---

### Benchmark config (`--benchmark-config`)

Defines the benchmark name, task, and scenarios.

**Example** (`examples/bench/llm_scenario.yaml`):

```yaml
name: vllm
task: offline
scenarios:
  - input_tokens: 128
    output_tokens: 128
    max_concurrency: [1, 4, 8, 16, 32, 64, 128, 256, 512]
  - input_tokens: 1024
    output_tokens: 1024
    max_concurrency: [1, 4, 8, 16, 32, 64, 128, 256]
```

`max_concurrency` can be a single integer or a list — lists are automatically expanded into separate runs.

**Supported tasks:**

| `task` | Description | Required extra fields |
|---|---|---|
| `offline` | LLM text generation | `output_tokens`, `random_range_ratio` |
| `vl-offline` | Vision-language generation | `output_tokens`, `random_mm_*` fields |
| `reranker` | Reranking | `random_batch_size` |
| `embeddings` | Text embeddings | — |

---

## Outputs

Results are written under `./bench_space/` relative to your working directory.

```
bench_space/
└── <DEVICE>_<NUM>_<BACKEND>_<VERSION>/
    └── <name>/<task>/<model>/
        ├── summary_<ISL>_<OSL>.md        # Markdown table
        ├── summary_<ISL>_<OSL>.csv       # CSV summary
        ├── monitor.csv                   # 1 Hz: device sum/mean, server /metrics, server/client/host CPU and memory
        ├── devices.csv                   # 1 Hz: one row per device (power, temperatures, utilization, clocks, throttle)
        ├── metrics_snapshots.jsonl       # full /metrics text and NPU counters at every scenario start and end
        └── <ISL>.<OSL>.<CONC>/
            ├── monitor.csv               # the rows of this scenario
            ├── devices.csv
            └── vllm-*.json               # raw vLLM benchmark output
```

The monitor columns merged into each scenario's summary row:

| Column | Meaning |
|---|---|
| `active_s` | Seconds from the first to the last sample with a running request; client start-up and teardown are excluded |
| `mean_power`, `energy_j` | Energy integrated over the active window, and that energy divided by `active_s` |
| `gen_tok_per_j`, `total_tok_per_j` | Generated (or prompt + generated) tokens per joule, from the server's token counters over the same window |
| `power_steady_w`, `power_p95_w` | Mean power while at least 90 % of `max_concurrency` requests run, and the 95th percentile over the active window |
| `window_ok` | `False` when there are fewer than 10 active samples or no server metrics. RNGD power readings take a few seconds to follow the load, so treat power, energy and busy figures of such a scenario as indicative only |
| `throttled` | `True` if any device reported a slowdown (power, thermal or clock cap) while active |
| `npu_busy_pct` | RNGD only, when `window_ok`: executed PE cycles over the active window's cycle capacity |
| `server_<ttft\|itl\|e2el>_mean_ms` | Server-side mean latencies from the `/metrics` histograms; the gap to the client's means is client, network and tokenization overhead |
| `finished_<stop\|length\|abort>` | Requests the server finished per reason during the scenario |
| `client_cpu_peak_pct` | Peak CPU of the benchmark client while requests run (100 = one core); near 100 the client, not the server, may limit throughput |

Server logs are written to `./serve_logs/`.

**Example path:**

```
bench_space/RNGD_4_furiosa-llm/vllm/offline/EXAONE-4.0-32B-FP8/summary.csv
```

---

## Reports

Turn one or more benchmark runs into a self-contained interactive HTML report:

```bash
furiosa-perf report \
  --benchmark-result-path ./bench_space \
  --model-list all \
  --task-list all \
  --report-path ./report
```

The command walks `--benchmark-result-path` for `<task>/<model>/summary.csv`
files (the aggregate summaries written by `run`), and renders per-(ISL/OSL)
charts — throughput table, interactive per-user curve, rack-level performance,
TTFT/TPOT, and TPS/Watt — into `--report-path` (default `./report`). Open
`report/index.html` in a browser. The raw CSVs are also bundled under
`report/csv/` and indexed in `report/manifest.json`.

The landing page opens with a **RNGD vs RTX Pro 6000** comparison: one bar chart
per metric (TPS/User, TPS/Watt, Median TTFT, Median TPOT) for the 1024/1024
scenario, broken down by concurrency. Bars are oriented so taller always means
RNGD is better — TPS/User and TPS/Watt as RNGD / RTX Pro 6000, Median TTFT and
Median TPOT as RTX Pro 6000 / RNGD — with a dashed 90% reference line. Each
model is a separate bar you can toggle from the legend; models unsupported on
RNGD simply show no bar. At each concurrency RNGD uses its best-performing device
count and RTX Pro 6000 is matched to the device count closest to it (shown in the
bar's hover tooltip). The comparison targets (scenario, hardware families,
reference line, metrics) are configured in `furiosa_perf/reporting/theme.py`.

### CLI options

| Option | Required | Default | Description |
|---|---|---|---|
| `--benchmark-result-path` | ✓ | — | Root directory to scan for `summary.csv` results |
| `--model-list` | | `all` | Comma-separated model names, or `all` |
| `--task-list` | | `all` | Comma-separated task names, or `all` |
| `--report-contents` | | — | Optional YAML overriding the report title/abstract/theme |
| `--report-path` | | `./report` | Output directory for the generated report |

---

## Troubleshooting

**Device columns in the monitoring CSV are empty**
- NPU: requires `furiosa-smi-py` (installed by the `furiosa-llm` extra) and the `furiosa-smi` apt package it wraps.
- GPU: requires `nvidia-smi` on `PATH`.
- The monitor keeps logging server `/metrics` and host usage either way; the missing probe is reported once as `[monitor] device sampling is disabled; cause: ...`.

**Server fails to start**
- Check `./serve_logs/` for the server process output.
- Run the backend command directly to inspect startup errors.

**`HF_TOKEN` not set for gated models**
- Export `HF_TOKEN` or run `huggingface-cli login` before benchmarking.

---

## Development

```bash
pip install -e ".[dev]"

# Install lint tools (shfmt, shellcheck, ruff, mypy, yamlfmt, yamllint, actionlint)
make install-tools

# Run all linters
make lint

# Run tests
make test

# Clean caches
make clean
```
