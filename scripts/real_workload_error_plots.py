"""FlashKDA vs our CuTe and TIRx kernels on real Kimi-Linear prefill workloads.

    uv run --with matplotlib --with safetensors python scripts/real_workload_error_plots.py
    uv run --with matplotlib python scripts/real_workload_error_plots.py --plot-only

Runs FlashKDA, CuTe and TIRx on real Kimi-Linear-48B-A3B-Instruct prefill captures and
scores them against an fp64 token-by-token recurrence with FlashKDA's test metric
(tests/test_fwd.py: relative RMS error ``rms(gold - x) / rms(gold)``):

figures/real_workload_accuracy.png shows one real long prefill (a MATH-500 prompt, 8183
tokens, H = 96): the output error against context length (per 64-token window, smoothed
over 8 windows; the output is causal, so this is the error at that context length) and the
final-state error.

The numbers are saved to figures/real_workload_error.json (``--plot-only`` redraws from it).

Recomputing needs the Kimi-Linear prefill captures (WORKLOADS) and the KDA-internal
real-workload harness (REAL_TESTS, KDA_BENCH; optional cached fp64 references in REFS);
``--plot-only`` needs only matplotlib.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
WINDOW, SMOOTH = 64, 8
trw = None  # the real-workload harness, imported by compute()

KERNELS = [("FlashKDA", "#2a78d6"), ("Ours (CuTe)", "#e8702a"), ("Ours (TIRx)", "#1baf7a")]


def load_kernel(name, path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


FIGURE_CASE = "MATH-500, 1 seq x 8183"
CASES = {FIGURE_CASE: ("std-math500-multi-8192-H96", "math500-multi-8192", 96)}


def case_args(ref_name, sample, H):
    import torch

    dev = torch.device("cuda")
    layers = trw.load_layers(Path(os.environ["WORKLOADS"]) / sample, trw.LAYER_SETS[H], dev)
    return ref_name, trw.task_inputs(layers, dev)[0]


def err_ratio(gold, x):
    """FlashKDA's metric: rms(gold - x) / rms(gold)."""
    return float((gold.float() - x.float()).square().mean().sqrt() / (gold.float().square().mean().sqrt() + 1e-8))


def windowed(gold, x):
    """err_ratio per WINDOW-token window along the (packed) sequence."""
    T = gold.shape[1]
    n = T // WINDOW
    g = gold[:, : n * WINDOW].float().reshape(n, -1)
    d = (gold[:, : n * WINDOW].float() - x[:, : n * WINDOW].float()).reshape(n, -1)
    return (d.square().mean(1).sqrt() / (g.square().mean(1).sqrt() + 1e-8)).tolist()


def compute():
    global trw
    import torch
    from safetensors.torch import load_file

    sys.path[:0] = [os.environ["KDA_BENCH"], os.environ["REAL_TESTS"]]
    import test_real_workloads as trw

    refs = Path(os.environ["REFS"]) if "REFS" in os.environ else None

    cute = load_kernel("cute_kernel", ROOT / "cute/kernel.py")
    tirx = load_kernel("tirx_kernel", ROOT / "tirx/kernel.py")
    fns = {"FlashKDA": trw.bk.flashkda_forward, "Ours (CuTe)": cute.run, "Ours (TIRx)": tirx.run}
    data = []
    for title, key in CASES.items():
        ref_name, args = case_args(*key)
        ref_file = refs / f"{ref_name}.safetensors" if refs else None
        if ref_file is not None and ref_file.exists():
            ref = load_file(str(ref_file), device="cuda")
            gold, gold_ht = ref["o"], ref["final_state"]
        else:
            with torch.no_grad():
                gold, gold_ht = trw.bk.recurrence_forward(*trw.bc.clone_args(args))
        row = {"case": title, "case_id": ref_name, "T": int(args[0].shape[1]), "H": int(args[0].shape[2]), "kernels": {}}
        for label, fn in fns.items():
            with torch.no_grad():
                o, ht = fn(*trw.bc.clone_args(args))
            torch.cuda.synchronize()
            row["kernels"][label] = {
                "output": err_ratio(gold, o), "final_state": err_ratio(gold_ht, ht), "windowed_output": windowed(gold, o),
            }
            print(f"{ref_name:45s} {label:12s} output {row['kernels'][label]['output']:.3e} "
                  f"final_state {row['kernels'][label]['final_state']:.3e}", flush=True)
        data.append(row)
        del args, gold, gold_ht
        torch.cuda.empty_cache()
    (FIGURES / "real_workload_error.json").write_text(json.dumps(data, indent=1))
    return data


def smooth(xs, w):
    half = w // 2
    return [sum(xs[max(0, i - half): i + half + 1]) / len(xs[max(0, i - half): i + half + 1]) for i in range(len(xs))]


def plot(data):
    """One figure: a real long prefill, output error vs context length and final-state error."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 15})
    row = next(r for r in data if r["case"] == FIGURE_CASE)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(15, 5.8), width_ratios=[2, 1], layout="constrained")

    for label, color in KERNELS:
        ys = smooth(row["kernels"][label]["windowed_output"], SMOOTH)
        xs = [(i + 1) * WINDOW for i in range(len(ys))]
        ax.plot(xs, [100 * y for y in ys], color=color, lw=3, label=label)
    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("output relative RMSE (%)")
    ax.set_title("Output: relative RMSE vs context length")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=16, frameon=False, loc="upper left")

    values = [100 * row["kernels"][label]["final_state"] for label, _ in KERNELS]
    bars = bx.bar([label for label, _ in KERNELS], values, color=[c for _, c in KERNELS], width=0.6)
    bx.bar_label(bars, fmt="%.2f%%", padding=3)
    bx.set_ylabel("final-state relative RMSE (%)")
    bx.set_title(f"Final state after {row['T']} tokens")
    bx.set_ylim(0, max(values) * 1.15)

    for a in (ax, bx):
        a.grid(axis="y", alpha=0.3)
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Kimi-Linear-48B prefill of a MATH-500 prompt ({row['T']} tokens, {row['H']} heads)\n"
                 "relative RMSE = rms(x - x_fp64) / rms(x_fp64), FlashKDA's test metric; lower is better",
                 fontsize=16)
    fig.savefig(FIGURES / "real_workload_accuracy.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    data = json.loads((FIGURES / "real_workload_error.json").read_text()) if args.plot_only else compute()
    plot(data)


if __name__ == "__main__":
    main()
