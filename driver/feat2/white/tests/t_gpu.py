"""GPU test of white.scores (optional; not part of the default CPU run).  No model is loaded: the gates are moved to the card and scored there,
and the library's forward hook runs on a linear module of the real width in the module dtype.  Through the queue, about 1 GB:

    GQ=$RESCOPE_ROOT/02_research/tools/gpuq/gq
    $GQ run --mem 1500 --prio 4 --name white_gpu -- $RESCOPE_PYTHON tests/run_tests.py gpu
"""
from __future__ import annotations

import os

import tcommon as T
from tcommon import C, test
from white import scores as WS

torch = C.use_library()

from rescope import Hook, add_guards, calibrate_two_block, decision_band, feature_gate  # noqa: E402


@test("gpu", "every new gate on the GPU (ScopedLog.to('cuda'), development-log keys): scores and decisions == CPU within the decision band; hook on a cuda module in the module dtype")
def g1_gpu_device():
    if not torch.cuda.is_available():
        return {"pass": False, "note": "no CUDA device visible: run through the queue (gq run --mem 1500 --prio 4 -- python tests/run_tests.py gpu)"}
    if "GQ_MEM" not in os.environ:
        return {"pass": False, "note": "not started through the GPU queue (GQ_MEM unset): gq run --mem 1500 --prio 4 -- python tests/run_tests.py gpu"}
    cap = (int(os.environ["GQ_MEM"]) - 600) * 2 ** 20                    # the queue's rule: the job caps itself
    torch.cuda.set_per_process_memory_fraction(cap / torch.cuda.get_device_properties(0).total_memory, 0)
    dev = torch.device("cuda:0")
    out, ok, worst, n_dec, n_diff = {}, True, 0.0, 0, 0
    for name in C.DEV_LOGS.values():
        ctx = T.context(name)
        dt = torch.bfloat16 if ctx.backbone == "qwen25vl" else torch.float16          # module dtype of the backbone (key cache meta.json)
        assert str(dt) == ctx.kc.meta["module_dtype"]
        rows = ctx.rows[::5]
        Kq = ctx.kc.keys("tf", rows)                                      # float32 images of module-dtype values (exact)
        gates = [(lab, WS.feature_gate(ctx.log, feat, stats=ctx.stats, kappa=kappa, null_keys=ctx.null_keys)) for lab, feat, kappa in WS.variants()]
        gates.append(("rescope_nullW", WS.null_w(feature_gate(ctx.log, "ncoord"), stats=ctx.stats)[0]))
        e = {}
        g = torch.Generator().manual_seed(T.SEED)
        m = ctx.log.m
        W0 = (torch.randn(m, ctx.log.d, generator=g) / ctx.log.d ** 0.5).to(dt).to(dev)        # a stand-in pre-edit weight of the real shape and dtype
        mod = torch.nn.Linear(ctx.log.d, m, bias=False).to(dev, dt)
        with torch.no_grad():
            mod.weight.copy_(W0)
        for lab, S in gates:
            S, _ = add_guards(S, *ctx.own)
            G, _ = calibrate_two_block(S, coords=ctx.kc.coords(S, ctx.draw_rows(0), "tf"), alpha=0.05, seed=0)
            Gd = G.replace(guard_key=G.guard_key[:0], guard_edit=G.guard_edit[:0], W0=W0).to(dev)      # as the evaluation moves a gate to the model
            same_class = type(Gd) is type(G) and all(getattr(Gd, f).device.type == "cuda" for f in ("det_w", "det_s", "det_tau") + (("q_mu",) if getattr(G, "q_mu", None) is not None else ())
                                                     + (("q_F",) if getattr(G, "q_F", None) is not None else ()))
            r_cpu = G.scores(Kq)
            r_gpu = Gd.scores(Kq.to(dev)).cpu()                           # one batch on the card
            r_one = torch.cat([Gd.scores(Kq[i:i + 1].to(dev, dt)) for i in range(0, Kq.shape[0], 9)]).cpu()        # single keys in the module dtype, as the hook sees them
            band = decision_band(G, Kq, r_cpu)
            ratio = float(((r_gpu - r_cpu).abs() / band).max())
            ratio1 = float(((r_one - r_cpu[::9]).abs() / band[::9]).max())
            th = G.theta()
            near = ((r_cpu - th[None, :]).abs() <= band).any(1)
            f_cpu, f_gpu = (r_cpu >= th[None, :]).any(1), (r_gpu >= th[None, :]).any(1)
            diff = int(((f_cpu != f_gpu) & ~near).sum())
            # the hook on the card: 12 sequences whose last prompt position holds a cached key
            bad, fired = 0, 0
            for t in range(12):
                x = torch.zeros(1, 6, ctx.log.d, dtype=dt, device=dev)
                x[0] = Kq[torch.randint(0, Kq.shape[0], (6,), generator=g)].to(dev, dt)
                lp = 3
                if t % 2 == 0:                                            # an edit's own B = 1 Rel key at the last prompt position: must fire
                    x[0, lp] = ctx.own[0][t].to(dev, dt)
                with torch.no_grad():
                    plain = mod(x)
                    r = Gd.scores(x[0, lp][None])[0]
                    cand = r >= Gd.theta()
                    want = bool(cand.any())
                    top = int(torch.where(cand, Gd.rank_key(r[None])[0], torch.full_like(r, -float("inf"))).argmax()) if want else -1
                    with Hook(mod, Gd) as hk:
                        hk.begin(torch.tensor([lp]))
                        o = mod(x)
                        fl = hk.summary()
                good = fl["fired"] == want and fl["top_last"] == top
                if want:
                    fired += 1
                    tgt = (plain[0, lp].double() + Gd.values[top]).to(dt)   # W_0 k + v_e in fp64, ONE cast to the module dtype
                    good &= bool(torch.equal(o[0, lp], tgt)) and bool(torch.equal(o[0, :lp], plain[0, :lp])) and bool(torch.equal(o[0, lp + 1:], plain[0, lp + 1:]))
                else:
                    good &= bool(torch.equal(o, plain))
                if t % 2 == 0:
                    good &= want
                bad += int(not good)
            e[lab] = {"class_and_device_kept": same_class, "max_gpu_minus_cpu_over_band": ratio, "max_single_key_module_dtype_over_band": ratio1,
                      "rows_decided_otherwise_outside_band": diff, "rows_near": int(near.sum()), "hook_sequences_bad": bad, "hook_sequences_fired": fired}
            worst = max(worst, ratio, ratio1)
            n_dec += int(f_cpu.numel())
            n_diff += diff
            ok &= same_class and ratio <= 0.1 and ratio1 <= 0.1 and diff == 0 and bad == 0 and fired >= 6
            del Gd
        out[name] = e
        del mod, W0
        torch.cuda.empty_cache()
    return {"pass": ok, "per_log": out, "device": torch.cuda.get_device_name(0), "peak_MiB": torch.cuda.max_memory_allocated() / 2 ** 20,
            "note": f"2 logs x 9 gates on {torch.cuda.get_device_name(0)}: |GPU - CPU| / decision band <= {worst:.1e}; {n_diff} of {n_dec} decisions differ outside the band; hook on a "
                    f"cuda linear module (bf16 / fp16, real width): firing, top edit and output == W_0 k + v_e with one cast, silent sequences bitwise untouched: {ok}; "
                    f"peak {torch.cuda.max_memory_allocated() / 2 ** 20:.0f} MiB"}
