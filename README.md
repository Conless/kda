# KDA forward kernels for NVIDIA Blackwell

Two independent implementations of the Kimi Delta Attention (KDA) forward pass
(chunked, per-channel-gated delta rule; K3 gate `-5 * sigmoid(exp(A_log) * (g + dt_bias))`,
in-kernel q/k L2 normalization, `beta = sigmoid(beta)`, fp32 V-first recurrent state),
both following the KDA-internal `kda_forward` task ABI:

    run(q, k, v, g, beta, A_log, dt_bias, scale, initial_state, cu_seqlens) -> (output, final_state)

| directory | language | entry point | source |
|---|---|---|---|
| `cute/` | CuTe DSL (Python) | `cute/kernel.py` (`run`) | humanfia/kda-for-kda `yahui-2.88x-cute` @ `dde00d0`, with the FP16 gate table removed (see below) |
| `tirx/` | TIRx (TVM) | `tirx/kernel.py` (`prepare` / `run`) | humanfia/kda-tirx `20260922-b300-tune` @ `def3dfc` (judge submission `2b64c874`), unchanged |

Inputs: bf16 `q/k/v/g [1, T, H, 128]`, bf16 beta logits `[1, T, H]`, fp32 `A_log [H]`,
fp32 `dt_bias [H*128]`, fp32 `initial_state [N, H, 128, 128]`, int64 `cu_seqlens [N+1]` or `None`.
Outputs: bf16 `output [1, T, H, 128]`, fp32 `final_state [N, H, 128, 128]`.

## Results (NVIDIA B300, 2026-09-24)

Measured with the KDA-internal judge (`bench_kda_forward_standalone.py`): speedup over
FlashKDA 7afb9f4's fused CUTLASS forward, executed live on every workload; 8192 total tokens.

| workload | CuTe | TIRx |
|---|---:|---:|
| H96 fixed (1 x 8192) | 2.761x | 3.108x |
| H96 mixed varlen (6 seqs) | 3.241x | 3.030x |
| H96 uniform varlen (8 x 1024) | 2.594x | 2.466x |
| H64 fixed (1 x 8192) | 2.514x | 3.611x |
| H64 mixed varlen (6 seqs) | 3.554x | 3.285x |
| H64 uniform varlen (8 x 1024) | 2.558x | 2.425x |
| **geomean** | **2.845x** | **2.957x** |
| judge correctness (workloads, stress / exact probes, holdout, 5 real probes) | 24/24 | 24/24 |

Real-workload accuracy: 151 cases built from Kimi-Linear-48B-A3B-Instruct prefill captures
(GSM8K / MATH-500; all 20 captured layers; H = 32 / 64 / 96; captured lengths, lengths cut to
multiples of 64, tails started from a real recurrent state, 6- and 8-sequence packs, packs of
up to 12277 tokens and 72 sequences), judged with the task's tolerance against an fp64
token-by-token recurrence. FLA `chunk_kda` passes all 151.

| | CuTe | TIRx |
|---|---|---|
| pass | 144 | 139 |
| fail | 7, each one final-state element over tolerance (measured worst 1.01-1.29x); rel L2 <= 0.005 | 4, each one element over tolerance (1.02-1.20x); rel L2 <= 0.006 |
| refused | 0 | 8 (see limitations) |

## CuTe: changes from the source branch

The source branch stored the per-chunk cumulative log-decay table in shared memory as FP16
whenever every sequence length is a multiple of 32 (`gcs_fp16=full_chunks`). That table spans
0 to -230.8 bits (5 nats/token x 32 tokens), where FP16 spacing reaches 0.125 bits, so
intra-chunk decay factors and the chunk-end state decay are off by up to ~9% under the deep
gates of real models. It passed the synthetic workloads but failed 23 of the 24
aligned-length real cases (worst element up to 3.5x tolerance). This release removes the
FP16 storage path (`GCS_FP16_`, `GCS_PACKCVT_`) so the table is always FP32, and removes the
unused M64 kernel (`pkdx.py`), which also used an FP16 table. Cost: geomean 2.903x -> 2.845x;
real-workload passes 119 -> 144 of 151. Nothing else changed.

## TIRx: provenance

jinhongyii's TIRx gist 4c50fafe (a fused persistent packed-varlen kernel plus an embedded
two-kernel split-concurrent route for single sequences, derived from the tirx-kernels
`agent_evolved` KDA forward kernels), tuned in place for B300 by AI agents (untouched gist:
2.544x). `kernel.py` wraps the gist's `setup(data, B, T, H) -> run` protocol as
`prepare(...) -> launch`; `prepare` compiles, builds the host work list from `cu_seqlens`,
and records a private CUDA graph, and `launch` replays it on the current buffer contents.

## Environment

- CuTe: Python 3.12, CUDA 13.2, torch 2.12.1+cu130, `nvidia-cutlass-dsl` **4.7.0**
  (KDA-internal image `kda-runtime:gpu` with the 4.7.0 CuTe DSL wheels).
- TIRx: the KDA-internal image `kda-runtime:tirx-kdafwd` (TVM ed5e2fed3); `kernel.py`
  retargets the gist's `sm_100a` to the running device's architecture.

Both target Blackwell; all numbers above were measured on a B300 (sm_103a). To judge one with KDA-internal, submit the
directory's contents as the task's `solution/` (entry point `solution/kernel.py`).

## Known limitations

- TIRx asserts, rather than computes, when a packed batch exceeds its work-list capacity:
  `ceil(T / 64) + num_seqs <= 160` (varlen, or a single sequence whose length is not a
  multiple of 32), `num_seqs < 64`, and `T < 65536`. Single sequences with `T % 32 == 0`
  take the split route instead (verified at 12256 tokens).
- TIRx bakes `cu_seqlens` into its work list at `prepare`; a new layout needs a new `prepare`.
- CuTe plans its schedule on the host per `cu_seqlens` layout (cached per tensor); a new
  layout costs up to about a second of host time on the first call.

## License

MIT, see [LICENSE](LICENSE).
