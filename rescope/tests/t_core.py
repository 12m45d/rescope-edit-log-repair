"""CPU unit tests, part 1 (synthetic logs): rebase identity, exact replay, bitwise non-firing, the hook against
an independent position-by-position reference, least repair, conformal calibration.
Each test returns a dict with 'pass' and the key numbers; run_tests.py writes them to tests/out/."""
from __future__ import annotations

import itertools

import numpy as np
import torch
import torch.nn.functional as Fnn

from rescope import calibrate as CA, loglib as LL, repair as RP
from rescope.scoped import COS, Hook, INF, LIN, ScopedLog

INT = {torch.float64: torch.int64, torch.float32: torch.int32, torch.bfloat16: torch.int16, torch.float16: torch.int16}


def bits_equal(a, b):
    """Bitwise equality (distinguishes -0.0 from +0.0, unlike torch.equal)."""
    return a.shape == b.shape and a.dtype == b.dtype and bool(torch.equal(a.contiguous().view(INT[a.dtype]), b.contiguous().view(INT[b.dtype])))


def linear(W0, dtype):
    lin = torch.nn.Linear(W0.shape[1], W0.shape[0], bias=False, dtype=dtype)
    with torch.no_grad():
        lin.weight.copy_(W0.to(dtype))
    return lin.requires_grad_(False)


def mid_tau(S):
    """Thresholds halfway between the largest log cross-talk and the own coordinate: every stored key fires only its own edit."""
    pool, _ = CA.pooled_crosstalk(S)
    return 0.5 * (float(pool.max()) + 1.0)


# ----------------------------------------------------------------------------- a1
def test_rebase_identity():
    out = {"cases": []}
    worst = 0.0
    for seed, (T, batch, l2) in enumerate([(24, 4, 0.5), (30, 10, 5.0), (12, 1, 2.0), (16, 16, 1.0), (40, 7, 20.0)]):
        log, tr = LL.synthetic_log(T=T, batch=batch, seed=seed, l2=l2)
        ref = tr["Z"] - log.K @ tr["W0"].T                              # z*_i - W_0 k_i (mid = 0)
        e = LL.rebase_error(log, ref)
        ex = LL.rebase_error(log, ref, LL.rebase_exact(log, _deltas(log)))
        fac = float((tr["W0"] + log.delta() - tr["WE"]).abs().max())
        worst = max(worst, e["rebased"]["max"], ex["rebased"]["max"])
        out["cases"].append({"T": T, "batch": batch, "l2": l2, "factor_identity_max_abs": fac, "rebased_max_rel": e["rebased"]["max"],
                             "exact_form_max_rel": ex["rebased"]["max"], "not_rebased_max_rel": e["not_rebased"]["max"],
                             "not_rebased_later_batches_median": None if e["not_rebased_later_batches"] is None else e["not_rebased_later_batches"]["median"]})
    # low-precision writes (round-off term E): factor form misses E_{<b} k_i, exact form does not
    log, tr = LL.synthetic_log(T=24, batch=4, seed=7, l2=0.5, write_dtype=torch.bfloat16)
    ref = tr["Z"] - log.K @ tr["W0"].T
    dl = _deltas(log, write_dtype=torch.bfloat16, W0=tr["W0"])
    e_f, e_x = LL.rebase_error(log, ref), LL.rebase_error(log, ref, LL.rebase_exact(log, dl))
    out["bf16_writes"] = {"factor_form_max_rel": e_f["rebased"]["max"], "exact_form_max_rel": e_x["rebased"]["max"],
                          "not_rebased_max_rel": e_f["not_rebased"]["max"]}
    out["max_rel_err_exact_writes"] = worst
    out["pass"] = bool(worst < 1e-10 and e_x["rebased"]["max"] < 1e-10 and e_f["rebased"]["max"] < 0.05
                       and all(c["not_rebased_max_rel"] > 0.05 for c in out["cases"] if c["batch"] < c["T"]))
    return out


def _deltas(log, write_dtype=None, W0=None):
    """Applied batch deltas of a synthetic log (replays the writes)."""
    out = []
    W = None if W0 is None else W0.clone()
    for b in range(int(log.batch.max()) + 1):
        ids = (log.batch == b).nonzero().flatten()
        D = log.U[ids].T @ log.W[ids]
        if write_dtype is None:
            out.append(D)
        else:
            Wn = (W.to(write_dtype) + D.to(write_dtype)).double()
            out.append(Wn - W)
            W = Wn
    return out


# ----------------------------------------------------------------------------- a2
def test_replay_attains_targets():
    out = {"cases": [], "pass": True}
    for seed, dtype in itertools.product(range(3), (torch.float64, torch.float32, torch.bfloat16)):
        log, tr = LL.synthetic_log(T=24, batch=4, seed=seed, l2=3.0)
        S0 = ScopedLog.from_log(log)
        S = S0.replace(det_tau=torch.full((log.T,), mid_tau(S0), dtype=torch.float64))
        only_own = bool(torch.equal(S.fires_on(log.K), torch.eye(log.T, dtype=torch.bool)))
        lin = linear(tr["W0"], dtype)
        W0d = lin.weight.data.clone()
        x = log.K.to(dtype)                                              # sequence i: one position = the solver key of edit i
        res = {}
        for resp, fire in (("repl", "top1"), ("gain", "top1"), ("repl", "all"), ("gain", "all")):
            g = S.replace(response=resp, fire=fire, W0=W0d)
            ys, es = [], []
            with Hook(lin, g) as hk:
                for i in range(log.T):
                    hk.begin(torch.tensor([0]))
                    ys.append(lin(x[i][None, None])[0, 0])
            for i in range(log.T):                                       # dtype(W_0 k (module dtype, same call shape) + coef v), one cast
                coef = 1.0 if resp == "repl" else float(g.scores(x[i][None])[0, i].clamp(max=1.0))
                es.append((Fnn.linear(x[i][None, None], W0d)[0, 0].double() + coef * g.values[i]).to(dtype))
            y, expect = torch.stack(ys), torch.stack(es)
            res[f"{resp}/{fire}"] = {"bitwise_expected": bits_equal(y, expect),
                                     "max_abs_err_to_target": float((y.double() - tr["Z"]).abs().max())}
        ok = only_own and all(v["bitwise_expected"] for v in res.values())
        if dtype == torch.float64:                                       # exact arithmetic: the target itself
            ok = ok and all(v["max_abs_err_to_target"] < 1e-10 for v in res.values())
        out["cases"].append({"seed": seed, "dtype": str(dtype), "stored_keys_fire_only_own_edit": only_own, "responses": res, "ok": ok})
        out["pass"] = out["pass"] and ok
    out["max_abs_err_fp64"] = max(v["max_abs_err_to_target"] for c in out["cases"] if c["dtype"] == "torch.float64" for v in c["responses"].values())
    return out


# ----------------------------------------------------------------------------- a3
def test_nonfiring_bitwise():
    out = {"n_calls": 0, "n_nonfiring_calls_same_object": 0, "n_nonfiring_calls": 0, "n_nonfiring_rows": 0, "n_nonfiring_rows_bitwise": 0,
           "n_negzero_rows": 0, "n_negzero_preserved": 0}
    g = torch.Generator().manual_seed(11)
    for seed, dtype, role in itertools.product(range(3), (torch.float64, torch.float32, torch.bfloat16, torch.float16), ("last", "last+answer", "all")):
        log, tr = LL.synthetic_log(T=20, batch=5, seed=seed, l2=3.0)
        lin = linear(tr["W0"], dtype)
        W0d = lin.weight.data.clone()
        S = ScopedLog.from_log(log, role=role, tau=0.6, W0=W0d)
        seen = {}
        h0 = lin.register_forward_hook(lambda m, a, o: seen.__setitem__("out", o))      # runs before the gate: the module's own tensor
        with Hook(lin, S) as hk:
            for trial in range(8):
                Tn, lp = 12, int(torch.randint(2, 9, (1,), generator=g))
                x = torch.randn(1, Tn, log.d, generator=g, dtype=torch.float64)
                if trial % 2 == 0:
                    x[0, lp] = log.K[trial]                              # an own key at the last prompt position: fires
                if trial % 4 == 1:
                    x[0, lp + 1] = log.K[trial]                          # only eligible for 'last+answer' / 'all'
                x = x.to(dtype)
                hk.begin(torch.tensor([lp]))
                y = lin(x)
                y0 = Fnn.linear(x, W0d)
                f = hk.last
                fired = torch.zeros(Tn, dtype=torch.bool)
                if f is not None:
                    fired[f.pos] = True
                out["n_calls"] += 1
                if f is None:
                    out["n_nonfiring_calls"] += 1
                    out["n_nonfiring_calls_same_object"] += int(y is seen["out"])
                nf = ~fired
                out["n_nonfiring_rows"] += int(nf.sum())
                out["n_nonfiring_rows_bitwise"] += int(sum(bits_equal(y[0, p], y0[0, p]) for p in nf.nonzero().flatten()))
                o = y0.clone()
                o[0, :, 0] = -0.0                                        # "W_0 k" with -0.0 entries: adding a zero would flip them to +0.0
                z = S.apply(x, o, torch.tensor([lp]))
                for p in nf.nonzero().flatten():
                    out["n_negzero_rows"] += 1
                    out["n_negzero_preserved"] += int(bits_equal(z[0, p], o[0, p]) and bool(torch.signbit(z[0, p, 0])))
                out["n_apply_same_object_when_silent"] = out.get("n_apply_same_object_when_silent", 0) + int(f is None and z is o)
        h0.remove()
    out["pass"] = bool(out["n_nonfiring_rows"] == out["n_nonfiring_rows_bitwise"] and out["n_nonfiring_calls"] > 0
                       and out["n_nonfiring_calls"] == out["n_nonfiring_calls_same_object"] == out["n_apply_same_object_when_silent"]
                       and out["n_negzero_rows"] == out["n_negzero_preserved"] > 0)
    return out


# ----------------------------------------------------------------------------- a9: the hook vs an independent reference
def reference_forward(S, x, y0, lp, pos0=0):
    """Position-by-position numpy implementation of the semantics in scoped.py's docstring (B = 1).
    Returns (y (Tn, m) torch in y0's dtype, fired positions, releasing edits per fired position)."""
    w, s, tau, ed, kind, key = (t.numpy() for t in (S.det_w, S.det_s, S.det_tau, S.det_edit, S.det_kind, S.det_key))
    V, LW, LU = S.values.numpy(), S.lin_w.numpy(), S.lin_u.numpy()
    xd = x.double().numpy()
    y = y0.clone()
    fired, released = [], []
    for p in range(xd.shape[0]):
        ap = p + pos0
        if not (S.role == "all" or (S.role == "last" and ap == lp) or (S.role == "last+answer" and ap >= lp)):
            continue
        k = xd[p]
        r = np.empty(len(s))
        for j in range(len(s)):
            if kind[j] == LIN:
                r[j] = (k @ w[j]) / s[j]
            elif kind[j] == COS:
                r[j] = (k @ w[j]) / s[j] / max(np.linalg.norm(k), 1e-300)
            else:
                r[j] = -np.linalg.norm(k - key[j])
        cand = [j for j in range(len(s)) if S.eta != INF and r[j] >= S.eta * tau[j]]
        if not cand:
            continue
        if S.fire == "top1":
            jt = max(cand, key=lambda j: r[j])
            rel = {int(ed[jt]): r[jt]}
        else:
            rel = {}
            for j in cand:
                rel[int(ed[j])] = max(rel.get(int(ed[j]), -np.inf), r[j])
        delta = np.zeros(V.shape[1])
        for e, re in sorted(rel.items()):
            if S.response == "repl":
                delta += V[e]
            elif S.response == "gain":
                delta += min(1.0, re) * V[e]
            else:
                delta += (k @ LW[e]) * LU[e]
        y[p] = (y0[p].double() + torch.from_numpy(delta)).to(y0.dtype)
        fired.append(ap)
        released.append(sorted(rel))
    return y, fired, released


def test_hook_vs_reference():
    out = {"n_cases": 0, "n_decisions_equal": 0, "n_nonfiring_rows_bitwise_ok": 0, "n_firing_rows": 0, "n_firing_rows_bitwise": 0,
           "max_rel_diff_firing_rows": 0.0, "n_cases_with_firing": 0, "by_dtype": {}}
    g = torch.Generator().manual_seed(5)
    combos = list(itertools.product(("last", "last+answer", "all"), ("top1", "all"), ("repl", "gain", "lin"), (1.0, 0.7, INF), (False, True)))
    for ci, (role, fire, resp, eta, extra) in enumerate(combos):
        for dtype in (torch.float32, torch.bfloat16):
            log, tr = LL.synthetic_log(T=12, d=48, m=10, batch=4, seed=ci % 5, l2=4.0)
            lin = linear(tr["W0"], dtype)
            W0d = lin.weight.data.clone()
            S = ScopedLog.from_log(log, role=role, fire=fire, response=resp, tau=0.55, W0=W0d).replace(eta=eta)
            Tn, lp = 14, 6
            x = torch.randn(Tn, log.d, generator=g, dtype=torch.float64)
            for p, i in ((lp, 1), (lp + 2, 3), (2, 5), (lp + 4, 1)):
                x[p] = log.K[i] * (0.8 + 0.4 * torch.rand(1, generator=g, dtype=torch.float64)) + 0.3 * torch.randn(log.d, generator=g, dtype=torch.float64)
            if extra:                                                    # why-not detectors: one cosine, one supplied linear
                S = RP.witness_repair(S.replace(eta=1.0), 2, x[lp + 1], mode="detector", gamma=0.9)[0]
                S = RP.witness_repair(S, 4, x[lp + 3], mode="detector", detector=x[lp + 3], tau=0.5)[0].replace(eta=eta)
            x = x.to(dtype)
            with Hook(lin, S) as hk:
                hk.begin(torch.tensor([lp]))
                y = lin(x[None])[0]
                f = hk.last
            y0 = Fnn.linear(x[None], W0d)[0]
            yr, fired, released = reference_forward(S, x, y0, lp)
            mine = [] if f is None else f.pos.tolist()
            mine_rel = [] if f is None else ([[int(e)] for e in f.edit] if fire == "top1" else [S._edit_any(f.cand[i:i + 1])[0].nonzero().flatten().tolist() for i in range(len(mine))])
            dec = mine == fired and mine_rel == released
            mask = torch.zeros(Tn, dtype=torch.bool)
            mask[fired] = True
            nf_ok = all(bits_equal(y[p], y0[p]) for p in (~mask).nonzero().flatten())
            fb = sum(bits_equal(y[p], yr[p]) for p in fired)
            rd = max([float(((y[p].double() - yr[p].double()).abs() / yr[p].double().abs().clamp_min(1e-3)).max()) for p in fired], default=0.0)
            out["n_cases"] += 1
            out["n_decisions_equal"] += int(dec)
            out["n_nonfiring_rows_bitwise_ok"] += int(nf_ok)
            out["n_firing_rows"] += len(fired)
            out["n_firing_rows_bitwise"] += fb
            out["n_cases_with_firing"] += int(bool(fired))
            out["max_rel_diff_firing_rows"] = max(out["max_rel_diff_firing_rows"], rd)
            d = out["by_dtype"].setdefault(str(dtype), {"firing_rows": 0, "bitwise": 0})
            d["firing_rows"] += len(fired)
            d["bitwise"] += fb
    # padded batches: per-row last prompt index; decisions of a B = 3 call == the three B = 1 calls
    bt = {"n": 0, "decisions_equal": 0, "nonfiring_rows_bitwise": 0}
    for trial in range(40):
        log, tr = LL.synthetic_log(T=12, d=48, m=10, batch=4, seed=trial % 5, l2=4.0)
        role = ("last", "last+answer", "all")[trial % 3]
        lin = linear(tr["W0"], torch.float32)
        S = ScopedLog.from_log(log, role=role, fire=("top1", "all")[trial % 2], tau=0.55, W0=lin.weight.data.clone())
        x = torch.randn(3, 10, log.d, generator=g, dtype=torch.float64)
        lps = torch.tensor([3, 6, 8])
        for b in range(3):
            x[b, int(lps[b])] = log.K[(trial + b) % log.T]
            x[b, (int(lps[b]) + 1) % 10] = log.K[(trial + b + 5) % log.T]
        x = x.to(torch.float32)
        with Hook(lin, S) as hk:
            hk.begin(lps)
            yb = lin(x)
            fb = hk.last
            got = sorted(zip(fb.row.tolist(), fb.pos.tolist(), fb.top.tolist())) if fb is not None else []
            single = []
            for b in range(3):
                hk.begin(lps[b:b + 1])
                lin(x[b:b + 1])
                f1 = hk.last
                single += [] if f1 is None else [(b, p, t) for p, t in zip(f1.pos.tolist(), f1.top.tolist())]
        y0 = Fnn.linear(x, lin.weight.data)
        mask = torch.ones(3, 10, dtype=torch.bool)
        for (b, p, _) in got:
            mask[b, p] = False
        bt["n"] += 1
        bt["decisions_equal"] += int(got == sorted(single) and len(got) > 0)
        bt["nonfiring_rows_bitwise"] += int(bits_equal(yb[mask], y0[mask]))
    out["batched"] = bt
    # cheap gates: fp32 coordinates agree with fp64 to 1e-5; the module-dtype (bf16) ones only to about 1e-2
    log, tr = LL.synthetic_log(T=12, d=48, m=10, batch=4, seed=1, l2=4.0)
    S = ScopedLog.from_log(log, tau=0.55)
    xb = (log.K + 0.1 * torch.randn(12, 48, generator=g, dtype=torch.float64)).to(torch.bfloat16)
    r64 = S.scores(xb)
    out["precision"] = {"fp32_max_abs_diff": float((S.replace(precision="fp32").scores(xb) - r64).abs().max()),
                        "bf16_max_abs_diff": float((S.replace(precision="module").scores(xb) - r64).abs().max())}
    # firing rows: the reference sums its fp64 terms in another order (numpy dot vs torch matmul), so a row may differ
    # by one unit in the last place of the module dtype; decisions and non-firing rows must agree exactly
    out["pass"] = bool(out["n_decisions_equal"] == out["n_cases"] and out["n_nonfiring_rows_bitwise_ok"] == out["n_cases"]
                       and out["max_rel_diff_firing_rows"] < 2e-2 and out["n_firing_rows_bitwise"] >= 0.99 * out["n_firing_rows"]
                       and out["n_cases_with_firing"] > 0.5 * out["n_cases"] * (2 / 3)
                       and bt["decisions_equal"] == bt["n"] == bt["nonfiring_rows_bitwise"]
                       and out["precision"]["fp32_max_abs_diff"] < 1e-5 < out["precision"]["bf16_max_abs_diff"] < 0.1)
    return out


# ----------------------------------------------------------------------------- a15: input guards of the hook (review F3, F4, PAD)
def test_hook_guards():
    """F4: one last prompt index per row, for every gate (a shorter vector used to disable the remaining rows, a
    single index was broadcast by the router).  F3: a gate without W0 must not silently run on the weight the module
    holds.  PAD: with seq_len the PAD positions of a padded batch never fire (decisions == the unpadded B = 1 calls)."""
    from rescope import baselines as BL
    from rescope.scoped import ROLES
    log, tr = LL.synthetic_log(T=12, d=48, m=10, batch=4, seed=2, l2=4.0)
    lin = linear(tr["W0"], torch.float32)
    W0d, WEd = lin.weight.data.clone(), tr["WE"].to(torch.float32)
    S = ScopedLog.from_log(log, tau=0.55, W0=W0d)
    g = torch.Generator().manual_seed(17)
    x = torch.randn(3, 10, log.d, generator=g, dtype=torch.float64)
    lps = torch.tensor([3, 6, 8])
    for b in range(3):
        x[b, int(lps[b])] = log.K[b]
    x = x.to(torch.float32)
    out = {"lp_length": {}, "W0": {}, "pad": {}}
    gates = {f"scoped/{role}": S.replace(role=role) for role in ROLES}
    gates.update({"router": BL.router_gate(log, WEd, W0d, eps=0.0), "zscore": BL.zscore_gate(WEd, W0d, 1.0), "cosine": BL.cosine_gate(log, 0.9, W0=W0d)})
    for nm, gate in gates.items():
        for tag, lp in (("short", lps[:1]), ("long", torch.tensor([3, 6, 8, 2]))):
            try:
                with Hook(lin, gate) as hk:
                    hk.begin(lp)
                    lin(x)
                out["lp_length"][f"{nm}/{tag}"] = False
            except ValueError:
                out["lp_length"][f"{nm}/{tag}"] = True
        with Hook(lin, gate) as hk:                                    # the right length works
            hk.begin(lps)
            lin(x)
            out["lp_length"][f"{nm}/ok_rows_firing"] = sorted(set(hk.last.row.tolist())) if hk.last is not None else []
    direct = {}
    for nm, fn in (("scoped", lambda: S.firing(x, lps[:1])), ("router", lambda: gates["router"].firing(x, lps[:1], 0, {}))):
        try:                                                           # the gates themselves, without the hook
            fn()
            direct[nm] = False
        except ValueError:
            direct[nm] = True
    out["lp_length"]["direct_gate_call_raises"] = direct
    # ---- F3
    with torch.no_grad():
        lin.weight.copy_(WEd)                                          # the module holds the deployed weight
    Sn = S.replace(W0=None)
    w = out["W0"]
    try:
        with Hook(lin, Sn):
            pass
        w["no_W0_raises"] = False
    except RuntimeError:
        w["no_W0_raises"] = True
    with Hook(lin, Sn, set_weight=False):
        w["no_W0_set_weight_false_allowed_weight_untouched"] = bool(torch.equal(lin.weight.data, WEd))
    with Hook(lin, S):
        w["with_W0_module_is_W0_inside"] = bool(torch.equal(lin.weight.data, W0d))
    w["restored_after"] = bool(torch.equal(lin.weight.data, WEd))
    try:
        with Hook(lin, S, set_weight=False):                           # W0 given, module at W_E, weight not to be set: refused
            pass
        w["wrong_weight_set_weight_false_raises"] = False
    except RuntimeError:
        w["wrong_weight_set_weight_false_raises"] = True
    w["no_hook_left"] = len(lin._forward_hooks) == 0
    # ---- PAD positions of a padded batch
    with torch.no_grad():
        lin.weight.copy_(W0d)
    lens, lps = torch.tensor([5, 8, 10]), torch.tensor([2, 4, 7])
    p = {"n": 0, "masked_equals_unpadded_B1": 0, "pad_outputs_bitwise_W0": 0, "n_pad_firing_unmasked": {r: 0 for r in ROLES}, "n_pad_firing_masked": 0,
         "dense_masked_no_pad": 0, "n_dense": 0}
    for trial in range(30):
        role = ROLES[trial % 3]
        x = torch.randn(3, 10, log.d, generator=g, dtype=torch.float64)
        for b in range(3):
            x[b, int(lps[b])] = log.K[(trial + b) % log.T]
            x[b, int(lps[b]) + 1] = log.K[(trial + b + 4) % log.T]
            x[b, int(lens[b]):] = log.K[(trial + 7) % log.T]           # PAD rows: a key that fires
        x = x.to(torch.float32)
        Sx = S.replace(role=role, fire=("top1", "all")[trial % 2])
        key = lambda f: [] if f is None else sorted(zip(f.row.tolist(), f.pos.tolist(), f.top.tolist()))     # noqa: E731
        with Hook(lin, Sx) as hk:
            hk.begin(lps)
            lin(x)
            raw = key(hk.last)
            hk.begin(lps, seq_len=lens)
            y = lin(x)
            masked = key(hk.last)
            single = []
            for b in range(3):
                hk.begin(lps[b:b + 1])
                lin(x[b:b + 1, :int(lens[b])])
                single += [(b, q, t) for (_, q, t) in key(hk.last)]
        y0 = Fnn.linear(x, W0d)
        pad = torch.arange(10)[None, :] >= lens[:, None]
        p["n"] += 1
        p["masked_equals_unpadded_B1"] += int(masked == sorted(single) and len(masked) > 0)
        p["pad_outputs_bitwise_W0"] += int(bits_equal(y[pad], y0[pad]))
        p["n_pad_firing_unmasked"][role] += sum(q >= int(lens[b]) for (b, q, _) in raw)
        p["n_pad_firing_masked"] += sum(q >= int(lens[b]) for (b, q, _) in masked)
        for gate in (gates["router"], gates["zscore"]):               # dense gates: the mask is applied by the hook, too
            with Hook(lin, gate) as hk:
                hk.begin(lps, seq_len=lens)
                lin(x)
                f = hk.last
            p["n_dense"] += 1
            p["dense_masked_no_pad"] += int(f is None or bool((f.pos < lens[f.row]).all()))
    bad = 0
    for kw in (dict(gen=True, seq_len=lens[:1]), ):
        try:
            Hook(lin, S).begin(torch.tensor([3]), **kw)
        except ValueError:
            bad += 1
    try:
        with Hook(lin, S) as hk:
            hk.begin(lps, seq_len=lens[:2])
            lin(x)
    except ValueError:
        bad += 1
    p["bad_arguments_refused"] = bad
    out["pad"] = p
    # ---- exact top-1 tie: two edits with the SAME detector have the same coordinate on every key; the lower index releases
    dup = lambda t: torch.cat([t[:1], t[:1], t[2:]])                                             # noqa: E731  (edit 1 <- edit 0)
    Sd = S.replace(det_w=dup(S.det_w), det_s=dup(S.det_s), det_key=dup(S.det_key))
    tie = {"n": 0, "tied": 0, "lowest_index_releases": 0, "value_of_edit_0": 0}
    for trial in range(20):
        k = (log.K[0] * (0.8 + 0.02 * trial) + 0.05 * torch.randn(log.d, generator=g, dtype=torch.float64)).to(torch.float32)
        f = Sd.firing(k[None, None], torch.tensor([0]))
        y0 = Fnn.linear(k[None, None], W0d)
        y = Sd.apply(k[None, None], y0, torch.tensor([0]))
        tie["n"] += 1
        tie["tied"] += int(f is not None and bool(f.cand[0, 0] and f.cand[0, 1]) and float(f.r[0, 0]) == float(f.r[0, 1]) == float(f.r[0].max()))
        tie["lowest_index_releases"] += int(f is not None and int(f.top[0]) == 0 and int(f.edit[0]) == 0)
        tie["value_of_edit_0"] += int(bits_equal(y[0, 0], (y0[0, 0].double() + Sd.values[0]).to(torch.float32)))
    out["tie"] = tie
    ll = out["lp_length"]
    out["n_lp_length_cases"] = sum(k.endswith(("/short", "/long")) for k in ll)
    out["n_lp_length_raise"] = sum(bool(v) for k, v in ll.items() if k.endswith(("/short", "/long")))
    out["pass"] = bool(out["n_lp_length_raise"] == out["n_lp_length_cases"] == 2 * len(gates) and all(direct.values())
                       and all(ll[f"{nm}/ok_rows_firing"] == [0, 1, 2] for nm in ("scoped/last", "scoped/all", "router", "cosine"))
                       and all(w.values()) and p["masked_equals_unpadded_B1"] == p["n"] == p["pad_outputs_bitwise_W0"] and p["n_pad_firing_masked"] == 0
                       and p["n_pad_firing_unmasked"]["last"] == 0 and p["n_pad_firing_unmasked"]["last+answer"] > 0 and p["n_pad_firing_unmasked"]["all"] > 0
                       and p["dense_masked_no_pad"] == p["n_dense"] and bad == 2
                       and tie["tied"] == tie["n"] == tie["lowest_index_releases"] == tie["value_of_edit_0"])
    return out


# ----------------------------------------------------------------------------- a4
def test_least_repair():
    out = {"n_trials": 0, "order_independent_bitwise": 0, "equals_coordinatewise_max": 0, "resolved": 0, "minimal": 0, "idempotent": 0,
           "only_candidates_changed": 0, "margin_order_independent": 0, "eta_fold_same_firing": 0}
    g = torch.Generator().manual_seed(3)
    for trial in range(200):
        log, _ = LL.synthetic_log(T=16, d=40, m=6, batch=4, seed=trial % 7, l2=4.0)
        eta = (1.0, 0.8, 1.3)[trial % 3]
        S = CA.calibrate_log_only(ScopedLog.from_log(log), 0.9)[0].replace(eta=eta)
        mk = lambda n: 0.25 * torch.randn(n, log.d, generator=g, dtype=torch.float64) + 0.35 * log.K[torch.randint(0, log.T, (n,), generator=g)]  # noqa: E731
        A, B = mk(3), mk(4)
        if trial % 5 == 0:
            B[0] = A[0] * (1 + 1e-15)                                    # near-tie between complaints
        kw = dict(on_conflict="force", margin=0.0, abs_margin=0.0)       # the textbook least repair: next float above
        ab = RP.least_repair(RP.least_repair(S, A, **kw)[0], B, **kw)[0]
        ba = RP.least_repair(RP.least_repair(S, B, **kw)[0], A, **kw)[0]
        jo, rec = RP.least_repair(S, [A, B], **kw)
        Sf = S.fold()
        mref = np.maximum(Sf.scores(A).numpy().max(0), Sf.scores(B).numpy().max(0))
        t0 = Sf.det_tau.numpy()
        ref = np.maximum(t0, np.nextafter(mref, np.inf))                 # coordinatewise maximum, independent (numpy) reference
        out["n_trials"] += 1
        out["order_independent_bitwise"] += int(torch.equal(ab.det_tau, ba.det_tau) and torch.equal(ab.det_tau, jo.det_tau))
        out["equals_coordinatewise_max"] += int(np.array_equal(jo.det_tau.numpy(), ref))
        out["resolved"] += int(rec["resolved"] and not bool(jo.fires_on(A).any()) and not bool(jo.fires_on(B).any()))
        ch = jo.det_tau.numpy() > t0
        out["only_candidates_changed"] += int(np.array_equal(ch, mref >= t0))
        # minimal: one float lower on any raised threshold and the complaint fires again
        out["minimal"] += int(all(mref[j] >= np.nextafter(jo.det_tau.numpy()[j], -np.inf) for j in np.nonzero(ch)[0]))
        out["idempotent"] += int(torch.equal(RP.least_repair(jo, [A, B], **kw)[0].det_tau, jo.det_tau))
        kd = dict(on_conflict="force")                                   # default margins (1e-6 relative + 1e-12)
        m1 = RP.least_repair(RP.least_repair(S, A, **kd)[0], B, **kd)[0]
        m2 = RP.least_repair(RP.least_repair(S, B, **kd)[0], A, **kd)[0]
        m3, rec3 = RP.least_repair(S, [A, B], **kd)
        out["margin_order_independent"] += int(torch.equal(m1.det_tau, m2.det_tau) and torch.equal(m1.det_tau, m3.det_tau) and rec3["resolved"]
                                               and bool((m3.det_tau >= jo.det_tau).all())
                                               and bool(((m3.det_tau - jo.det_tau) <= 2e-6 * jo.det_tau.abs() + 2e-12).all()))
        X = torch.randn(200, log.d, generator=g, dtype=torch.float64) + 0.3 * log.K[torch.randint(0, log.T, (200,), generator=g)]
        out["eta_fold_same_firing"] += int(torch.equal(S.fires_on(X), Sf.fires_on(X)))
    # own-key conflict: the complaint IS (a scaled copy of) an edit's own solver key
    log, _ = LL.synthetic_log(T=16, d=40, m=6, batch=4, seed=1, l2=4.0)
    S = CA.calibrate_log_only(ScopedLog.from_log(log), 0.9)[0]
    key = 1.2 * log.K[5][None]
    S_ref, rec_ref = RP.least_repair(S, key)
    S_par, rec_par = RP.least_repair(S, key, on_conflict="partial")
    S_for, rec_for = RP.least_repair(S, key, on_conflict="force")
    out["own_key_conflict"] = {"refused": rec_ref["refused"], "unchanged": bool(S_ref is S), "conflict_edits": sorted({c["edit"] for c in rec_ref["conflicts"]}),
                               "partial_keeps_edit_5": bool(S_par.det_tau[5] == S.fold().det_tau[5]), "partial_resolved": rec_par["resolved"],
                               "force_resolved": rec_for["resolved"], "force_own_keys_silenced": int((~S_for.guard_ok()).sum())}
    n = out["n_trials"]
    oc = out["own_key_conflict"]
    out["pass"] = bool(all(out[k] == n for k in ("order_independent_bitwise", "equals_coordinatewise_max", "resolved", "minimal", "idempotent",
                                                 "only_candidates_changed", "margin_order_independent", "eta_fold_same_firing"))
                       and oc["refused"] and oc["unchanged"] and 5 in oc["conflict_edits"] and oc["partial_keeps_edit_5"]
                       and not oc["partial_resolved"] and oc["force_resolved"] and oc["force_own_keys_silenced"] >= 1)
    return out


# ----------------------------------------------------------------------------- a5
def test_conformal():
    """Exchangeable synthetic queries: n calibration keys and 200 fresh keys per trial from the same distribution, 2000
    trials per (n, alpha, ties) cell, two independent seeds.  Without ties the expected false-firing rate is exactly
    1 - rank / (n + 1) <= alpha (equal to alpha when (1 - alpha)(n + 1) is an integer), so 'rate <= alpha' is tested as
    'mean <= alpha + 3 standard errors' and the z-score against the exact expectation is reported."""
    out = {"cells": [], "pass": True}
    log, _ = LL.synthetic_log(T=10, d=24, m=4, batch=5, seed=2, l2=4.0)
    S = CA.calibrate_log_only(ScopedLog.from_log(log), 0.95)[0]
    assert float(S.det_tau.min()) > 0
    n_trials, n_test = 2000, 200
    for seed in (1234, 4321):
      g = torch.Generator().manual_seed(seed)
      poolK = torch.randn(40, log.d, generator=g, dtype=torch.float64) + 0.4 * log.K[torch.randint(0, log.T, (40,), generator=g)]
      for (n, alpha), ties in itertools.product([(19, 0.1), (50, 0.1), (99, 0.05), (100, 0.2), (200, 0.01), (30, 0.01)], (False, True)):
        rates, naive, n_checked, n_same = [], [], 0, 0
        for a in range(0, n_trials, 250):
            nb = min(250, n_trials - a)
            if ties:                                                     # queries drawn from 40 distinct keys: heavy ties between calibration and test
                keys = poolK[torch.randint(0, 40, (nb, n + n_test), generator=g)]
            else:
                keys = torch.randn(nb, n + n_test, log.d, generator=g, dtype=torch.float64) + 0.4 * log.K[torch.randint(0, log.T, (nb, n + n_test), generator=g)]
            sc = S.score(keys.reshape(-1, log.d)).reshape(nb, n + n_test).numpy()
            for t in range(nb):
                S2, rec = CA.calibrate_conformal(S, sc[t, :n], alpha, allow_silence=True)
                fire = S2.fires_on(keys[t, n:]).any(1).numpy()           # the package's firing rule, not the score
                rates.append(fire.mean())
                naive.append((sc[t, n:] >= rec["quantile"]).mean())      # non-strict rule AT the quantile (what the guard avoids)
                n_checked += 1
                n_same += int(np.array_equal(fire, sc[t, n:] > rec["quantile"]))     # == the strict conformal event
        rates, naive = np.array(rates), np.array(naive)
        k = CA.conformal_rank(n, alpha)
        theory = 0.0 if k > n else 1 - k / (n + 1)
        se = float(rates.std(ddof=1) / np.sqrt(len(rates)))
        cell = {"seed": seed, "n": n, "alpha": alpha, "ties": ties, "rank": k, "eta_infinite": k > n, "mean_false_firing_rate": float(rates.mean()), "se": se,
                "theory_no_ties": theory, "z_vs_theory": None if (ties or se == 0) else float((rates.mean() - theory) / se),
                "le_alpha": bool(rates.mean() <= alpha), "le_alpha_plus_3se": bool(rates.mean() <= alpha + 3 * se),
                "naive_nonstrict_at_quantile_rate": float(naive.mean()), "n_trials": len(rates), "n_test_per_trial": n_test,
                "firing_equals_strict_event": [n_same, n_checked]}
        ok = (cell["le_alpha_plus_3se"] and (ties or abs(rates.mean() - theory) <= 3 * se + 1e-12) and (k <= n or rates.max() == 0.0)
              and n_same == n_checked)
        cell["ok"] = bool(ok)
        out["cells"].append(cell)
        out["pass"] = out["pass"] and bool(ok)
    zs = [c["z_vs_theory"] for c in out["cells"] if c["z_vs_theory"] is not None]
    out["z_vs_theory_no_ties"] = {"n_cells": len(zs), "mean": float(np.mean(zs)), "min": float(np.min(zs)), "max": float(np.max(zs))}
    g = torch.Generator().manual_seed(1234)
    # own-key rule: a quantile so large that the literal thresholds silence own keys
    big = np.full(50, 5.0)
    S_lit, r_lit = CA.calibrate_conformal(S, big, 0.1, allow_silence=True)
    S_cap, r_cap = CA.calibrate_conformal(S, big, 0.1)
    S_inf, r_inf = CA.calibrate_conformal(S, big[:5], 0.1)
    out["own_keys"] = {"literal_silenced_reported": r_lit["n_own_keys_silenced_by_literal_rule"], "literal_silenced_actual": int((~S_lit.guard_ok()).sum()),
                       "capped_silenced_actual": int((~S_cap.guard_ok()).sum()), "capped_guarantee": r_cap["guarantee"],
                       "eta_inf_capped_silenced": int((~S_inf.guard_ok()).sum()), "eta_inf_literal_fires_anything": bool(
                           CA.calibrate_conformal(S, big[:5], 0.1, allow_silence=True)[0].fires_on(log.K).any())}
    o = out["own_keys"]
    out["pass"] = bool(out["pass"] and o["literal_silenced_reported"] == o["literal_silenced_actual"] == log.T and o["capped_silenced_actual"] == 0
                       and o["eta_inf_capped_silenced"] == 0 and not o["eta_inf_literal_fires_anything"] and r_inf["eta_infinite"])
    out["max_excess_over_alpha"] = max(c["mean_false_firing_rate"] - c["alpha"] for c in out["cells"])
    out["naive_rule_max_excess_with_ties"] = max(c["naive_nonstrict_at_quantile_rate"] - c["alpha"] for c in out["cells"] if c["ties"])
    return out


TESTS = {"a1_rebase_identity": test_rebase_identity, "a2_replay_attains_targets": test_replay_attains_targets,
         "a3_nonfiring_bitwise": test_nonfiring_bitwise, "a9_hook_vs_reference": test_hook_vs_reference, "a15_hook_guards": test_hook_guards,
         "a4_least_repair": test_least_repair, "a5_conformal": test_conformal}
