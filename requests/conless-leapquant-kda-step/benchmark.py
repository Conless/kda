# SPDX-License-Identifier: Apache-2.0
"""Correctness and timing of a KDA decode-step implementation against definition.json (see README.md for the criterion).

  python benchmark.py                      # the baseline in this directory
  python benchmark.py --impl my_step.py    # a candidate exposing run(...) and, for timing, Pool / step_inplace

Correctness (functional form `run`, every batch size in workloads.jsonl, three inputs each: a synthetic state with edge-case
heads, the step after it (the first step's update appended to the buffer), and unstructured in-domain random tensors):
    o, k_row, u_row : per (sequence, head), with M = the same expression evaluated on the absolute values of every term
                      (the usual forward-error scale: o and u are sums that can cancel, so their own size is not a usable
                      denominator):  ||x - ref||_2 <= REL_L2 * ||M||_2  and  |x - ref| <= REL_MAX * M  elementwise
    g_row           : per element relative error <= REL_G (fp16 storage of a decay factor in (0, 1])
    w_new, pcum_new : |error| <= REL_SCALAR * (|reference| + 1)
The order of operations and the precision of intermediates are free.  Timing (deployment form `step_inplace` on a
caller-owned pool, every sequence decoding, buffer filled to entry 1): CUDA kernel time from the torch profiler, L2-cold --
NSETS disjoint slot sets are used in rotation so no call finds its checkpoints in L2.
"""
import argparse, importlib.util, json, sys
from pathlib import Path
import torch
from torch.profiler import profile, ProfilerActivity

HERE = Path(__file__).resolve().parent
H, K, V, L, R = 32, 128, 128, 16, 4
REL_L2, REL_MAX, REL_G, REL_SCALAR = 2.5e-3, 1.5e-2, 2e-3, 1e-5
QMAX = 127.0
NSETS, WARMUP, ITERS = 8, 6, 32
READ_B = V * K + (K + V) * 4 + R * (K + V) * 2 + L * (K + V) * 2 + L * K * 2 + L * 4 + K * 4 + (2 * K + V) * 2 + K * 2   # + gate history, cumulative gate, gate input
WRITE_B = 3 * 2 * V + K * 2 + K * 4 + 8                                                                                   # output, appended rows, decay factor, cumulative gate
REF_TBPS = 6.54


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod; spec.loader.exec_module(mod); return mod


def gate_consts(dev):
    gg = torch.Generator().manual_seed(1)
    return (torch.log(torch.rand(H, generator=gg) * 1.5 + 0.5)).to(dev), (torch.randn(H, K, generator=gg) * 0.5 - 1.0).to(dev)


def structured_inputs(bs, dev, seed=0):
    """A synthetic window state: full-range codes with positive scales, a hot row, Compensator Tokens, a partly filled buffer
    (h in [0, L)) with per-channel decay histories in [0.5, 1] and a cumulative log-gate consistent with h.  Heads 1..6 are edge
    cases: empty buffer and no residual, all-zero state, h = L - 1, tiny (1e-4), huge (3e2), almost fully decayed checkpoint."""
    g = torch.Generator(device=dev).manual_seed(seed)
    rn = lambda *s: torch.randn(*s, device=dev, generator=g)
    codes = (rn(bs, H, V, K) * 40).round().clamp(-127, 127).to(torch.int8)
    s_k = torch.rand(bs, H, K, device=dev, generator=g) * 0.2 + 0.05; s_v = torch.rand(bs, H, V, device=dev, generator=g) * 0.5 + 0.1
    s_v[..., 5] *= 30
    u = (rn(bs, H, R, K) * 0.5).to(torch.float16); q = (rn(bs, H, R, V) * 0.5).to(torch.float16)
    h = torch.randint(0, L, (bs, H), device=dev, generator=g, dtype=torch.int32)
    kbuf = (rn(bs, H, L, K) * 0.1).to(torch.bfloat16); ubuf = (rn(bs, H, L, V) * 0.3).to(torch.bfloat16)
    codes[:, 1] = 0; h[:, 1] = 0
    codes[:, 2] = 0; u[:, 2] = 0; q[:, 2] = 0; h[:, 2] = 0
    h[:, 3] = L - 1
    s_k[:, 4] *= 1e-4; u[:, 4] *= 1e-2; ubuf[:, 4] *= 1e-4
    s_k[:, 5] *= 3e2; u[:, 5] *= 17; ubuf[:, 5] *= 3e2
    live = torch.arange(L, device=dev) < h[..., None]
    gbuf = torch.where(live[..., None], torch.rand(bs, H, L, K, device=dev, generator=g) * 0.5 + 0.5, torch.ones(bs, H, L, K, device=dev)).half()
    w = live.float()
    pcum = gbuf.float().log().sum(-2) - torch.rand(bs, H, K, device=dev, generator=g)       # log-gate since the window start
    pcum[:, 6] -= 200.0                                                                         # a checkpoint decayed to nothing
    qx = rn(bs, H, K).to(torch.bfloat16); kx = rn(bs, H, K).to(torch.bfloat16); vx = rn(bs, H, V).to(torch.bfloat16)
    a = rn(bs, H, K).to(torch.bfloat16); b = (rn(bs, H) * 2).to(torch.bfloat16)
    A_log, dt_bias = gate_consts(dev)
    return codes, s_k, s_v, u, q, kbuf, ubuf, gbuf, w, pcum, h, qx, kx, vx, a, b, A_log, dt_bias


def next_step(inp, ref_out, dev, seed):
    """The following decode step: the reference's update appended to the buffer, fresh q / k / v and gates."""
    codes, s_k, s_v, u, q, kbuf, ubuf, gbuf, w, pcum, h, qx, kx, vx, a, b, A_log, dt_bias = inp
    _, k_row, u_row, g_row, w_new, pcum_new = ref_out
    kbuf, ubuf, gbuf = kbuf.clone(), ubuf.clone(), gbuf.clone(); hh = h.long(); ar = torch.arange(h.shape[0], device=dev)[:, None]; hv = torch.arange(H, device=dev)[None, :]
    kbuf[ar, hv, hh] = k_row; ubuf[ar, hv, hh] = u_row; gbuf[ar, hv, hh] = g_row
    full = h >= L - 1                                               # heads already at L - 1 keep h (they overwrite their last entry)
    h2 = torch.where(full, h, h + 1)
    gbuf = torch.where((torch.arange(L, device=dev) < h2[..., None])[..., None], gbuf, torch.ones_like(gbuf))
    g = torch.Generator(device=dev).manual_seed(seed)
    rn = lambda *s: torch.randn(*s, device=dev, generator=g)
    return (codes, s_k, s_v, u, q, kbuf, ubuf, gbuf, w_new * (torch.arange(L, device=dev) < h2[..., None]), pcum_new, h2.to(torch.int32),
            rn(*qx.shape).to(torch.bfloat16), rn(*kx.shape).to(torch.bfloat16), rn(*vx.shape).to(torch.bfloat16),
            rn(*a.shape).to(torch.bfloat16), (rn(*b.shape) * 2).to(torch.bfloat16), A_log, dt_bias)


def random_inputs(bs, dev, seed=1):
    """Unstructured tensors inside the deployment domain: positive scales, decay factors in (0, 1] (1 past h), w = 1 below h."""
    g = torch.Generator(device=dev).manual_seed(seed)
    rn = lambda *s, dt=torch.float32: torch.randn(*s, device=dev, generator=g).to(dt)
    h = torch.randint(0, L, (bs, H), device=dev, generator=g, dtype=torch.int32)
    live = torch.arange(L, device=dev) < h[..., None]
    gbuf = torch.where(live[..., None], torch.rand(bs, H, L, K, device=dev, generator=g) * 0.999 + 0.001, torch.ones(bs, H, L, K, device=dev)).half()
    A_log, dt_bias = gate_consts(dev)
    return (torch.randint(-127, 128, (bs, H, V, K), device=dev, generator=g, dtype=torch.int8), torch.exp(rn(bs, H, K) * 0.7) * 0.05,
            torch.exp(rn(bs, H, V) * 0.7) * 0.3, rn(bs, H, R, K, dt=torch.float16), rn(bs, H, R, V, dt=torch.float16),
            rn(bs, H, L, K, dt=torch.bfloat16), rn(bs, H, L, V, dt=torch.bfloat16), gbuf, live.float(), -torch.rand(bs, H, K, device=dev, generator=g) * 20, h,
            rn(bs, H, K, dt=torch.bfloat16), rn(bs, H, K, dt=torch.bfloat16), rn(bs, H, V, dt=torch.bfloat16),
            rn(bs, H, K, dt=torch.bfloat16), (rn(bs, H) * 2).to(torch.bfloat16), A_log, dt_bias)


def magnitudes(inp):
    """|A||x|-style scale of o, k_row and u_row: the reference's expressions with every term replaced by its absolute value."""
    codes, s_k, s_v, u, q, kbuf, ubuf, gbuf, w, pcum, h, qx, kx, vx, a, b, A_log, dt_bias = inp
    Kd = codes.shape[-1]; Ln = kbuf.shape[-2]
    kf = kx.float(); qf = qx.float()
    kn = kf * torch.rsqrt((kf * kf).sum(-1, keepdim=True) + 1e-6); qn = qf * torch.rsqrt((qf * qf).sum(-1, keepdim=True) + 1e-6) * Kd ** -0.5
    x = a.float() + dt_bias; g = -torch.exp(A_log)[:, None] * torch.where(x <= 20.0, torch.log1p(torch.exp(x)), x)
    en = torch.exp(g); pc = torch.exp(pcum + g); beta = torch.sigmoid(b.float()).to(torch.bfloat16).float()
    j = torch.arange(Ln, device=w.device)
    e = torch.where((j[:, None] < h[..., None, None]), gbuf.float(), torch.ones_like(gbuf, dtype=torch.float32))
    after = torch.flip(torch.cumprod(torch.flip(e, [-2]), -2), [-2]); after = torch.cat([after[..., 1:, :], torch.ones_like(after[..., :1, :])], -2)
    D = en[..., None, :] * after * (j[:, None] < h[..., None, None])
    s0a = torch.einsum("bhrv,bhrk->bhvk", q.float().abs(), u.float().abs()) + (codes.float() * (s_k / QMAX)[..., None, :] * s_v[..., :, None]).abs()
    sa = s0a * pc[..., None, :] + torch.einsum("bhj,bhjv,bhjk->bhvk", w.abs(), ubuf.float().abs(), (kbuf.float() * D).abs())
    ua = beta[..., None] * (vx.float().abs() + torch.einsum("bhvk,bhk->bhv", sa, kn.abs()))
    oa = torch.einsum("bhvk,bhk->bhv", sa, qn.abs()) + (kn.abs() * qn.abs()).sum(-1, keepdim=True) * ua
    return oa, kn.abs(), ua


def judge(out, ref, inp):
    worst = 0.0; ok = True
    for x, y, m in zip(out[:3], ref[:3], magnitudes(inp)):
        x, y = x.float(), y.float(); d = x - y
        l2 = d.flatten(2).norm(dim=-1) / m.flatten(2).norm(dim=-1).clamp_min(1e-30)
        mx = (d.abs() / m.clamp_min(1e-30)).flatten(2).amax(-1)
        zero = m.flatten(2).amax(-1) == 0                           # a head whose terms are all zero must come out all zero
        r = torch.where(zero, (x.flatten(2).abs().amax(-1) > 0).float() * 1e9, torch.maximum(l2 / REL_L2, mx / REL_MAX))
        ok &= bool((r <= 1).all()) and bool(torch.isfinite(x).all())
        worst = max(worst, r.max().item())
    x, y = out[3].float(), ref[3].float()
    e = ((x - y).abs() / y.abs().clamp_min(1e-30)).max().item(); ok &= e <= REL_G and bool(torch.isfinite(x).all()); worst = max(worst, e / REL_G)
    for x, y in zip(out[4:], ref[4:]):
        x, y = x.float(), y.float()
        e = ((x - y).abs() / (y.abs() + 1)).max().item(); ok &= e <= REL_SCALAR; worst = max(worst, e / REL_SCALAR)
    return ok, worst


def check(bs, impl, ref_run, dev):
    first = structured_inputs(bs, dev, seed=bs); r1 = ref_run(*[t.clone() for t in first])
    res = []
    for inp, ref in ((first, r1), (next_step(first, r1, dev, bs + 1), None), (random_inputs(bs, dev, seed=bs + 2), None)):
        ref = ref_run(*[t.clone() for t in inp]) if ref is None else ref
        out = impl.run(*[t.clone() for t in inp]); torch.cuda.synchronize()
        res.append(judge(out, ref, inp))
    return res


def ktime(fn):
    for i in range(WARMUP): fn(i % NSETS)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for i in range(ITERS): fn(i % NSETS)
        torch.cuda.synchronize()
    ev = [e for e in prof.events() if e.device_type.name == "CUDA" and "fill" not in e.name.lower() and "mem" not in e.name.lower()]
    return sum(e.device_time for e in ev) / ITERS / 1000


def timing(bs, impl, dev):
    pool = impl.Pool(NSETS * bs + 8, dev)
    pool.raw.normal_(0, 0.1); pool.uq.normal_(0, 0.1)
    pool.kbuf.copy_(torch.randn(pool.kbuf.shape, device=dev)); pool.ubuf.copy_(torch.randn(pool.ubuf.shape, device=dev)); pool.w.uniform_(0, 1)
    idxs = [torch.arange(1 + j * bs, 1 + (j + 1) * bs, dtype=torch.int32, device=dev) for j in range(NSETS)]
    mixed = torch.randn(bs, 2 * H * K + H * V, device=dev).to(torch.bfloat16)
    a = torch.randn(bs, H, K, device=dev).to(torch.bfloat16); b = (torch.randn(bs, H, device=dev) * 2).to(torch.bfloat16)
    A_log, dt_bias = gate_consts(dev); o = torch.empty(bs, H, V, dtype=torch.bfloat16, device=dev)
    def call(j):
        pool.hcnt.fill_(1)
        impl.step_inplace(pool, idxs[j], mixed, a, b, A_log, dt_bias, o)
    return ktime(call)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--impl", default=str(HERE / "baseline.py")); ap.add_argument("--skip-timing", action="store_true"); ap.add_argument("--bs", default="")
    a = ap.parse_args(); dev = torch.device("cuda")
    name = torch.cuda.get_device_name(0); assert any(m in name.split() for m in ("B200", "B300")), name
    d = json.loads((HERE / "definition.json").read_text()); ns = {}; exec(d["reference"], ns); ref_run = ns["run"]
    sizes = [json.loads(l)["workload"]["axes"]["batch_size"] for l in (HERE / "workloads.jsonl").read_text().splitlines() if l.strip()]
    if a.bs: sizes = [int(x) for x in a.bs.split(",")]
    impl = load(a.impl, "step_impl")
    print(f"{name}, torch {torch.__version__}, impl {Path(a.impl).name}\n")
    print("| batch_size | programs | worst error / tolerance (structured / next step / random) | correctness | latency (ms) | memory bound (ms) |")
    print("| --- | --- | --- | --- | --- | --- |")
    allok = True
    for bs in sizes:
        res = check(bs, impl, ref_run, dev); ok = all(r[0] for r in res); allok &= ok
        t_ms = float("nan") if a.skip_timing or not hasattr(impl, "step_inplace") else timing(bs, impl, dev)
        bound = bs * H * (READ_B + WRITE_B) / REF_TBPS / 1e12 * 1e3
        print(f"| {bs} | {bs * H} | {res[0][1]:.2f} / {res[1][1]:.2f} / {res[2][1]:.2f} | {'PASS' if ok else 'FAIL'} | {t_ms:.4f} | {bound:.4f} |", flush=True)
    assert allok, "correctness failed"


if __name__ == "__main__":
    main()
