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
- **Presets** — bundled `baseline/text-gen`, `baseline/vision`, `baseline/embedding`, `baseline/rerank`, or your own YAML
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
  --preset baseline/text-gen
```

See [`examples/script.sh`](examples/script.sh) for a complete working example.

### CLI options

| Option | Required | Default | Description |
|---|---|---|---|
| `--server-config` | ✓ | — | API server config YAML (backend, model, devices, …) |
| `--preset` | ✓ | — | Benchmark preset: a bundled name such as `baseline/text-gen`, or a YAML file |
| `--output-dir` | | `results` | Root of the results tree |
| `--search-min`, `--search-max` | | from the preset | Range of the preset's capacity search for this run |
| `--resume` | | off | Finish a run already in its results directory, keeping valid points |
| `--overwrite` | | off | Delete a run already in its results directory and start over |
| `--reuse-server` | | off | Benchmark a server already running on the configured port instead of launching one |
| `--check` | | off | Validate the server config and preset, print the server command (preset server options applied) and point count, and exit (no hardware needed) |

### Inspecting the environment

```bash
furiosa-perf env --server-config examples/server/exaone4.yaml [--output ./env-dump]
```

Prints what a run with this server config would record, without launching anything: host (OS, kernel, CPU, memory, container), the devices (name, driver, firmware, PCIe link, NUMA node, links between devices, idle power; GPU power limits — current, default, min, max — or the RNGD governor), the server environment's Python and backend versions and where the backend is imported from, driver and runtime apt packages, the model's Hub revision, and the relevant environment variables (secrets masked). Problems that would stop `run` (backend missing from `env`, unknown device indices) are printed as warnings. `--output` also writes the raw probe outputs.

`run` collects the same information before launching the server: it goes into the `environment` section of `run.json` and raw into `env/`. At the end of the run the power settings are read again; if they changed, `run.json` has `power_settings_changed: true`.

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

**Every other key** is passed to `<backend> serve` as `--<key> <value>` (`true` → bare flag, `false` → omitted; snake_case keys are converted to kebab-case), for example `served-model-name`, `max-model-len` or `enable-prefix-caching`. The parallel sizes are set only through `parallel`: the backends count them in different units (furiosa-llm counts tensor parallelism in PEs, 8 per RNGD).

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

### Presets (`--preset`)

A preset says what to measure: which client, which server options the measurement needs, and the points to run. Pass a bundled name or a YAML file of your own.

| Bundled preset | Task |
|---|---|
| `baseline/text-gen` | Text generation, fixed ISL/OSL from 128/128 to 128k/1k |
| `baseline/vision` | One 256x256 image plus text, the same total lengths as `text-gen` |
| `baseline/embedding` | Embeddings, 256 to 32k input tokens |
| `baseline/rerank` | Reranking, 5 documents per query, 256 to 8k tokens |
| `simulation/chat-synthetic` | General chatbot: multi-turn synthetic conversations, capacity search over 8 to 256 users, SLO TTFT p95 ≤ 3 s and TPOT p95 ≤ 50 ms |
| `simulation/chat-mooncake` | Production chatbot trace (Kimi, Mooncake FAST'25 conversation trace) replayed at its recorded times, capacity search over 5 to 100 % of its recorded speed, same SLO |

The bundled files are in [`furiosa_perf/presets/`](furiosa_perf/presets); copy one to change the points. The format:

```yaml
kind: baseline            # fixed-length baseline sweep
task: text-gen            # label used in result paths
client: vllm-bench        # runs `vllm bench serve` (pinned version, own venv)
server:                   # applied over the server config; null removes an option
  no-enable-prefix-caching: true
  reasoning-parser: null
args:                     # `vllm bench serve` options shared by every point, same names as its CLI
  backend: vllm           # request format of the client
  dataset-name: random
  ignore-eos: true
matrix:                   # one point per combination; list values are expanded
  - {random-input-len: 1024, random-output-len: 1024, max-concurrency: [1, 4, 8, 16]}
```

- `num-prompts` defaults to three times `max-concurrency` (see `examples/BENCHMARK.md`).
- furiosa-perf sets the client's target (`model`, `host`, `port`) and result file itself; presets cannot.
- The baseline presets turn prefix caching off (random prompts must not hit a cache) and remove any reasoning parser, so thinking tokens stay in `content`, which is what the client counts.

**Service simulation (`kind: simulation`, `client: aiperf`).** Points run [AIPerf](https://github.com/ai-dynamo/aiperf) (pinned, own venv). In `simulation/chat-synthetic` every user holds one conversation: a system prompt shared by all users (1,000 tokens), a context of their own (500), then 5 ± 2 turns of 150 ± 50 tokens in and 300 ± 100 out, with 10 ± 5 s of think time between a reply and the next message. Each turn re-sends the conversation so far, including the replies the server generated, so the server keeps prefix caching on. A point's `concurrency` is the number of users talking at once (a user keeps their slot during think time). The preset also sets `max-model-len: 8192` and turns thinking off for models that support it (`default-chat-template-kwargs: {enable_thinking: false}`).

```yaml
slo:                      # p95 limits; a point meets the SLO when both hold
  ttft_p95_ms: 3000
  tpot_p95_ms: 50
```

Instead of a fixed list of user counts, a simulation preset can bisect for capacity:

```yaml
search:
  key: concurrency        # users talking at once
  min: 8                  # expected to meet the SLO
  max: 256                # expected to miss it
  step: 4                 # stop when the passing and failing counts are 4 apart
```

`restart-server: per-point` (set in `simulation/chat-synthetic`) launches a fresh server before every measured point, so no point inherits the previous one's prefix cache, queue or load; the preset's AIPerf warm-up (`warmup-request-count`) then primes the new server outside the measurement. The default, `never`, keeps one server for the whole run, as the baseline presets do.

Both ends are measured first; if `min` already misses the SLO or `max` still meets it, the run says so (`bound: below_min` / `at_least_max`) instead of searching. Otherwise each probe halves the range, so 8..256 with step 4 takes at most 8 points. Override the range per run with `--search-min` / `--search-max`; `--check` prints the plan. Passing assumes more users never improves latency, so a noisy point at the edge can move the result by about one step.

**Trace replay.** A preset with a `trace:` block replays a request trace in Mooncake format (`timestamp`, `input_length`, `output_length`, `hash_ids`) instead of synthetic traffic. The trace is downloaded once to `~/.cache/furiosa-perf/datasets/` and checked against the preset's `sha256`. It holds no text: AIPerf turns every hash id into a fixed 512-token block, so requests that shared a prompt prefix in production share it again and hit the prefix cache the same way. Requests are sent at their recorded times (open loop), stretched to `trace-speed-pct` percent of the recorded speed; each point replays the start of the trace that lasts `trace-duration` seconds at that speed, and requests longer than the server's `max-model-len` are left out and counted (`Dropped Too Long`). The search key is then `trace-speed-pct`, and `run.json` reports `capacity.max_speed_pct` with the trace's source, hash and license.

Each simulation summary row has `slo_met`, and `run.json` gets `capacity: {slo, max_users, search}`: the most users whose point was valid and met the SLO, and with a search its range, step, bound and every probe with its verdict. The SLO is also passed to AIPerf as `--goodput`, reported per point as `Goodput(req/s)`.

**Point validity.** Each summary row has `ok` and `fail_reason`. A baseline point is invalid when the client exits with an error, more than 5 % of its requests fail, a generation point counts no output tokens, or, with `ignore-eos`, fewer than 90 % of the requested output tokens come back. A simulation point is invalid when AIPerf exits with an error, no request completes, more than 5 % fail, or a streaming run measures no time to first token. Invalid points keep their measured values. The run goes on after an invalid point unless the server stopped answering, in which case it stops and reports how many points were not run.

---

## Outputs

Results are written under `--output-dir` (default `./results`), one directory per run:

```
results/
└── <kind>/                                        # baseline
    └── <DEVICE>_<NUM>[-<LAYOUT>]_<BACKEND>_<VERSION>/   # e.g. RNGD_4_furiosa-llm_2026.4.0, H100-80GB-HBM3_2-tp1-dp2_vllm_0.30.0
        └── <task>/<org>--<model>/                 # e.g. text-gen/furiosa-ai--Qwen3-8B-FP8
            ├── run.json                  # what was run, on what, with which versions; status and point counts
            ├── env/                      # raw environment probes: host.txt, smi.txt, topology.txt, devices.json,
            │                             #   packages.txt, server-freeze.txt, client-freeze.txt, env-vars.json
            ├── server.log                # API server output
            ├── summary_<ISL>_<OSL>.md|.csv  summary.md|.csv
            ├── monitor.csv               # 1 Hz: device sum/mean, server /metrics, server/client/host CPU and memory
            ├── devices.csv               # 1 Hz: one row per device (power, temperatures, utilization, clocks, throttle)
            ├── metrics_snapshots.jsonl   # full /metrics text and NPU counters at every point start and end
            └── i<ISL>-o<OSL>-c<CONC>/    # one directory per point (i<ISL>-b<DOCS>-c<CONC> for rerank, i<ISL>-c<CONC> for embedding)
                ├── record.json           # the point, self-describing: model, hardware, versions, options, ok, metrics
                ├── result.json           # raw vllm bench result
                ├── bench.log             # client console output
                └── monitor.csv  devices.csv   # the rows of this point
```

- `DEVICE` is the full product name (`H100-80GB-HBM3`, `H100-NVL`, `RTX-PRO-6000-Blackwell-Server-Edition`, `RNGD`), with `@<W>W` when a GPU power limit is not its default or `@powersave` for an RNGD on the PowerSave governor. `LAYOUT` appears only for non-default parallelism.
- Read `run.json` / `record.json` rather than parsing paths; they hold every field the path encodes and more.
- A run directory that already holds a run is not touched: pass `--resume` to finish it (points recorded as valid are kept, the rest run again) or `--overwrite` to start over.

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


---

## Reports

Turn one or more benchmark runs into a self-contained interactive HTML report:

```bash
furiosa-perf report \
  --benchmark-result-path ./results \
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
- Check `server.log` in the run directory; the error message also shows its last lines.
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
