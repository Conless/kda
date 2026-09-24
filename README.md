# KDA forward kernels for NVIDIA Blackwell

Two independent implementations of the Kimi Delta Attention (KDA) forward pass
(chunked, per-channel-gated delta rule; K3 gate `-5 * sigmoid(exp(A_log) * (g + dt_bias))`,
in-kernel q/k L2 normalization, `beta = sigmoid(beta)`, fp32 V-first recurrent state),
both following the KDA-internal `kda_forward` task ABI:

    run(q, k, v, g, beta, A_log, dt_bias, scale, initial_state, cu_seqlens) -> (output, final_state)

| directory | language | entry point | source |
|---|---|---|---|
| `cute/` | CuTe DSL (Python) | `cute/kernel.py` (`run`) | humanfia/kda-for-kda `yahui-2.88x-cute` @ `dde00d0`, with two fixes (see below) |
| `tirx/` | TIRx (TVM) | `tirx/kernel.py` (`prepare` / `run`) | humanfia/kda-tirx `20260922-b300-tune` @ `def3dfc` (judge submission `2b64c874`), unchanged |

Inputs: bf16 `q/k/v/g [1, T, H, 128]`, bf16 beta logits `[1, T, H]`, fp32 `A_log [H]`,
fp32 `dt_bias [H*128]`, fp32 `initial_state [N, H, 128, 128]`, int64 `cu_seqlens [N+1]` or `None`.
Outputs: bf16 `output [1, T, H, 128]`, fp32 `final_state [N, H, 128, 128]`.

## Results (NVIDIA B300, 2026-09-24)

Measured with the KDA-internal judge (`bench_kda_forward_standalone.py`): speedup over
FlashKDA 7afb9f4's fused CUTLASS forward, executed live on every workload; 8192 total tokens.

| workload | CuTe | TIRx |
|---|---:|---:|
| H96 fixed (1 x 8192) | 2.760x | 3.108x |
| H96 mixed varlen (6 seqs) | 3.244x | 3.030x |
| H96 uniform varlen (8 x 1024) | 2.582x | 2.466x |
| H64 fixed (1 x 8192) | 2.512x | 3.611x |
| H64 mixed varlen (6 seqs) | 3.528x | 3.285x |
| H64 uniform varlen (8 x 1024) | 2.544x | 2.425x |
| **geomean** | **2.837x** | **2.957x** |
| judge correctness (workloads, stress / exact probes, holdout, 5 real probes) | 24/24 | 24/24 |

## CuTe: changes from the source branch

- The per-chunk cumulative log-decay table is always kept in FP32 (the FP16 storage path
  `GCS_FP16_` / `GCS_PACKCVT_` and the unused M64 kernel `pkdx.py` are removed).
- The split-sequence state handoff clears its flags on the launch stream before every launch,
  so eager calls, CUDA graph captures and replays can be mixed in any order.

## TIRx: provenance

jinhongyii's TIRx gist 4c50fafe (a fused persistent packed-varlen kernel plus an embedded
two-kernel split-concurrent route for single sequences, derived from the tirx-kernels
`agent_evolved` KDA forward kernels), tuned in place for B300 by AI agents (untouched gist:
2.544x). `kernel.py` wraps the gist's `setup(data, B, T, H) -> run` protocol as
`prepare(...) -> launch`; `prepare` compiles, builds the host work list from `cu_seqlens`,
and records a private CUDA graph, and `launch` replays it on the current buffer contents.

## Environment

Both kernels, the FlashKDA baseline and the benchmark share one uv environment, pinned in
`pyproject.toml` / `uv.lock`:

| component | version | used by |
|---|---|---|
| Python | 3.12 | all |
| torch | 2.12.1+cu130 | all |
| nvidia-cutlass-dsl (CuTe DSL) | 4.7.0 | CuTe kernel |
| apache-tvm (with TIRx), apache-tvm-ffi, tirx-kernels | git ed5e2fed3, be35ec1, 65d9a075 | TIRx kernel |
| flash-kda (FlashKDA) | git 7afb9f4 | baseline |
| fla-core | 0.5.2 | baseline wrapper, correctness reference |
| flashinfer-python, cupti-python | 0.6.13, 13.0.1 | CUPTI timer |

TVM and FlashKDA are compiled from source during `uv sync`; everything else installs as
wheels.

### Prerequisites

- An NVIDIA Blackwell GPU (sm_100a / sm_103a) with a driver for CUDA 13.
- The CUDA 13 toolkit (`nvcc`); FlashKDA's CUTLASS extension is built with it.
- A C++17 compiler, CMake >= 3.18, and the LLVM 18 development files TVM needs for host
  code generation.
- [uv](https://docs.astral.sh/uv/) >= 0.8.

On Ubuntu 24.04:

```bash
sudo apt-get install build-essential cmake llvm-18-dev libxml2-dev zlib1g-dev libzstd-dev
curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv is not installed
```

### Install

```bash
export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH
uv sync
```

The first `uv sync` compiles TVM and FlashKDA from source (30-60 minutes); later syncs reuse uv's
cache. Notes:

- FlashKDA builds for the GPU it sees. On a build host without a GPU, set
  `FLASH_KDA_CUDA_ARCHS` (for example `103a` for B300, `100a` for B200) before `uv sync`.
- TVM finds LLVM through `llvm-config-18` (set in `[tool.uv.config-settings-package]` in
  `pyproject.toml`). If your LLVM 18 `llvm-config` has another name or path, change it
  there and run `uv sync --reinstall-package apache-tvm`.

### Check

```bash
uv run python -c "import cutlass, tvm, tirx_kernels.tirx_lite, flash_kda, fla; print('ok', cutlass.__version__, tvm.__version__)"
```

## Benchmark

```bash
uv run python bench.py cute    # or: tirx, or a path to another kernel.py
```

`bench.py` runs the six timed workloads of the KDA-internal `kda_forward` task (the Int21
set above), checks both outputs against FLA's Triton `chunk_kda` with the task's
tolerance, and reports each workload's FlashKDA and kernel times and the geomean speedup.
It follows the task's timing protocol (CUPTI, cold L2, CUDA graph, median of 30 iterations
x 3 trials), with the task's input distributions and fixed seeds. On the B300 above it
reports 2.849x (CuTe) and 2.965x (TIRx), within 0.5% of the judge's geomeans.

A kernel passed by path must expose `run(...)` with the signature above and may expose
`prepare(...) -> launch`, which is then planned once per workload and only `launch()` is
timed.

## Supported inputs

Head size 128, bf16 activations and an fp32 state; tested with H = 32, 64 and 96. The TIRx
kernel handles packed batches with `ceil(T / 64) + num_seqs <= 160`, fewer than 64 sequences
and `T < 65536`; single sequences with `T % 32 == 0` are not bound by the first limit.

## Hacking example

[`hacking_example/`](hacking_example/) keeps a disqualified forward kernel that a
multi-agent optimization run produced against a loose verifier: it claimed 3.74x over
FlashKDA by replacing the q/k norms with a distribution constant, dropping initial-state
channels, skipping the cross-CTA state handoff and the intra-chunk solve. Its README
documents each cheat and what a verifier needs to catch them. Do not use it.

## License

MIT, see [LICENSE](LICENSE).
