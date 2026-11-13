"""CPU unit tests, part 2: null repair, scope cut vs brute force, baselines vs their definitions (incl. KV-cache
position bookkeeping), witness repair / session, log-only calibration, and CPU checks on the two real logs
(rebased values against the cached clean-model targets, the splits, P11a's thresholds)."""
from __future__ import annotations

import dataclasses
import itertools
import json
import os
import types

import numpy as np
import torch
import torch.nn.functional as Fnn

from rescope import baselines as BL, calibrate as CA, evaluate as EV, loglib as LL, repair as RP, splits as SP
from rescope.scoped import COS, Hook, INF, LIN, TIE, ScopedLog
from t_core import bits_equal, linear


# ----------------------------------------------------------------------------- a6
def test_null_repair():
    log, tr = LL.synthetic_log(T=12, d=60, m=6, batch=4, seed=4, l2=4.0)
    S = ScopedLog.from_log(log, tau=0.5)
    g = torch.Generator().manual_seed(8)
    Kb = torch.randn(20, log.d, generator=g, dtype=torch.float64) + 0.5 * log.K.mean(0)
    Kb[7], Kb[12] = Kb[3], 2.0 * Kb[5]                                   # duplicate and collinear benign keys: singular Gram
    W, K, s = log.W.numpy(), log.K.numpy(), log.own().numpy()
    out = {}
    # exact variant vs the explicit minimum-norm solution  w' = w - A^+ (A w - b),  A = [Kb; k_i],  b = [0; s_i]
    S1, rec = RP.null_repair(S, Kb)
    W1 = S1.det_w.numpy()
    ref = np.stack([W[i] - np.linalg.pinv(np.vstack([Kb.numpy(), K[i]])) @ (np.vstack([Kb.numpy(), K[i]]) @ W[i] - np.r_[np.zeros(20), s[i]])
                    for i in range(log.T)])
    out["exact"] = {"max_abs_benign_r": float(np.abs(W1 @ Kb.numpy().T / s[:, None]).max()),
                    "max_own_rel_err": float(np.abs((W1 * K).sum(1) / s - 1).max()), "max_abs_diff_vs_pinv": float(np.abs(W1 - ref).max()),
                    "rank_benign": rec["rank_benign"], "rel_change_max": rec["rel_change_max"], "guards_lost": rec["n_guards_lost"]}
    # ridge variant vs the KKT system of  min ||w' - w||^2 + ||Kb w'||^2 / rho  s.t. <w', k_i> = s_i
    out["ridge"] = []
    for ridge in (1e-3, 1e-1, 10.0):
        S2, r2 = RP.null_repair(S, Kb, ridge=ridge)
        W2 = S2.det_w.numpy()
        Hm = np.eye(log.d) + Kb.numpy().T @ Kb.numpy() / r2["rho"]
        ref2 = np.stack([np.linalg.solve(np.block([[Hm, K[i][:, None]], [K[i][None, :], np.zeros((1, 1))]]), np.r_[W[i], s[i]])[:log.d]
                         for i in range(log.T)])
        out["ridge"].append({"ridge": ridge, "max_abs_diff_vs_kkt": float(np.abs(W2 - ref2).max()),
                             "max_own_rel_err": float(np.abs((W2 * K).sum(1) / s - 1).max()),
                             "benign_norm_ratio": float(np.linalg.norm(W2 @ Kb.numpy().T) / np.linalg.norm(W @ Kb.numpy().T))})
    S0, _ = RP.null_repair(S, Kb, ridge=1e-12)
    out["ridge_to_zero_vs_exact_max_abs"] = float((S0.det_w - S1.det_w).abs().max())
    # joint zero-forcing on the other solver keys (P11b's NULL)
    S3, r3 = RP.null_repair(S, Kb, zero_force_log=True)
    W3 = S3.det_w.numpy()
    A = np.vstack([K, Kb.numpy()])
    ref3 = np.stack([W[i] - np.linalg.pinv(A) @ (A @ W[i] - np.r_[s[i] * np.eye(log.T)[i], np.zeros(20)]) for i in range(log.T)])
    out["zero_force_log"] = {"max_abs_cross_r": float(np.abs(W3 @ K.T / s[:, None] - np.eye(log.T)).max()),
                             "max_abs_benign_r": float(np.abs(W3 @ Kb.numpy().T / s[:, None]).max()), "max_abs_diff_vs_pinv": float(np.abs(W3 - ref3).max())}
    # infeasible: the benign set contains edit 2's own key
    S4, r4 = RP.null_repair(S, torch.cat([Kb, log.K[2:3]]))
    out["infeasible"] = {"reported": r4["infeasible"], "detector_2_unchanged": bool(torch.equal(S4.det_w[2], S.det_w[2])),
                         "others_zero_on_key_2": float((S4.det_w @ log.K[2] / log.own()).abs()[[i for i in range(log.T) if i != 2]].max())}
    # nothing else changes; the 'lin' response still uses the log's own read-out vectors
    x = torch.randn(1, 9, log.d, generator=g, dtype=torch.float64)
    lin = linear(tr["W0"], torch.float64)
    ys = []
    for Sx in (S, S1):
        with Hook(lin, Sx.replace(role="all", fire="all", response="lin", det_tau=torch.full((log.T,), -INF, dtype=torch.float64), W0=lin.weight.data.clone())):
            ys.append(lin(x))
    out["untouched"] = {"tau": bool(torch.equal(S1.det_tau, S.det_tau)), "values": S1.values is S.values, "lin_w": S1.lin_w is S.lin_w,
                        "lin_response_bitwise_equal": bits_equal(ys[0], ys[1])}
    e, z = out["exact"], out["zero_force_log"]
    out["pass"] = bool(e["max_abs_benign_r"] < 1e-9 and e["max_own_rel_err"] < 1e-9 and e["max_abs_diff_vs_pinv"] < 1e-8 and e["rank_benign"] == 18
                       and e["guards_lost"] == 0 and all(r["max_abs_diff_vs_kkt"] < 1e-8 and r["max_own_rel_err"] < 1e-9 and r["benign_norm_ratio"] < 1 for r in out["ridge"])
                       and out["ridge"][0]["benign_norm_ratio"] < out["ridge"][1]["benign_norm_ratio"] < out["ridge"][2]["benign_norm_ratio"]
                       and out["ridge_to_zero_vs_exact_max_abs"] < 1e-6 and z["max_abs_cross_r"] < 1e-9 and z["max_abs_benign_r"] < 1e-9
                       and z["max_abs_diff_vs_pinv"] < 1e-8 and out["infeasible"]["reported"] == [2] and out["infeasible"]["detector_2_unchanged"]
                       and out["infeasible"]["others_zero_on_key_2"] < 1e-9 and all(out["untouched"].values()))
    return out


# ----------------------------------------------------------------------------- a7
def test_scope_cut():
    rng = np.random.default_rng(20261110)
    out = {"n_instances": 0, "n_equal_brute_force": 0, "n_box_thresholds_reproduce_optimum": 0, "n_problems": 600, "n_problems_with_ties": 0,
           "n_rows_never_clean": 0, "n_exempt_entries": 0, "failures": []}
    vals = np.arange(0, 6).astype(float)                                 # few distinct values: many ties
    for it in range(600):
        T, N = int(rng.integers(1, 4)), int(rng.integers(1, 7))
        C = rng.choice(vals, size=(N, T))
        ex = rng.random((N, T)) < 0.15
        C[ex] = -np.inf
        cap = rng.choice(np.array([2.0, 3.0, 4.0, 5.0, 6.0]), size=T)
        we, wx = [], []
        for i in range(T):
            for _ in range(int(rng.integers(0, 4))):
                we.append(i)
                wx.append(float(rng.choice(vals)))
        P = RP.CutProblem(C, cap, we, wx, rng.integers(0, 6, size=N).astype(float), rng.integers(0, 6, size=len(we)).astype(float), scale=2)
        out["n_rows_never_clean"] += int((~P.feasible).sum())
        out["n_exempt_entries"] += int(ex.sum())
        fin = C[np.isfinite(C)]
        out["n_problems_with_ties"] += int(len(np.unique(np.concatenate([fin, np.array(wx)]))) < len(fin) + len(wx))
        for lam in (0.5, 1.0, 2.0):                                      # lam * weight * scale is an integer: the stated objective, exactly
            clean, info = P.solve(lam)
            bf = RP.brute_force(P, lam)
            m, hi = P.box(clean)
            lo = np.where(np.isneginf(m), -np.inf, np.nextafter(m, np.inf))
            with np.errstate(invalid="ignore"):
                mid = np.where(np.isneginf(lo), hi, 0.5 * (lo + hi))
            box_ok = all(P.objective(t, lam)[0] == bf for t in (lo, hi, mid))
            out["n_instances"] += 1
            out["n_equal_brute_force"] += int(info["value_int"] == bf)
            out["n_box_thresholds_reproduce_optimum"] += int(box_ok)
            if not (info["value_int"] == bf and box_ok) and len(out["failures"]) < 5:
                out["failures"].append({"it": it, "lam": lam, "cut": info["value_int"], "bf": bf})
    # the public operator on scoped logs: objective == brute force, clean rows silent, covered witnesses and own keys fire
    g = torch.Generator().manual_seed(13)
    pub = {"n": 0, "objective_equals_brute_force": 0, "clean_rows_silent": 0, "covered_witnesses_fire": 0, "own_keys_fire": 0, "inside_box": 0,
           "n_raised": 0, "n_lowered": 0}
    for trial, rule in itertools.product(range(40), ("nearest", "lo", "mid", "hi")):
        log, _ = LL.synthetic_log(T=3, d=30, m=4, batch=1, seed=trial, l2=4.0)
        S = CA.calibrate_log_only(ScopedLog.from_log(log), 0.8)[0]
        rows = [torch.randn(2, log.d, generator=g, dtype=torch.float64) + 0.5 * log.K[int(torch.randint(0, 3, (1,), generator=g))] for _ in range(7)]
        wk = [(i, log.K[i] * (0.4 + 0.5 * float(torch.rand(1, generator=g))) + 0.4 * torch.randn(log.d, generator=g, dtype=torch.float64)) for i in (0, 0, 1, 2, 2)]
        R = S.query_max(rows)
        wr = torch.stack([S.key_scores(k[None])[0, i] for i, k in wk])
        lam = (0.3, 1.0, 3.0)[trial % 3]
        S2, rec = RP.scope_cut(S, R, ([i for i, _ in wk], wr), lam, rule=rule, scale=20)
        S2k, _ = RP.scope_cut(S, rows, wk, lam, rule=rule, scale=20)      # the same problem given as keys
        pub["keys_interface_equal"] = pub.get("keys_interface_equal", 0) + int(torch.equal(S2k.det_tau, S2.det_tau))
        P = RP.CutProblem(R.numpy(), CA.own_caps(S.fold()).numpy(), [i for i, _ in wk], wr.numpy(), scale=20)
        val, clean, unc = P.objective(S2.det_tau.numpy(), lam)
        pub["n"] += 1
        pub["objective_equals_brute_force"] += int(val == RP.brute_force(P, lam) == rec["value_int"])
        pub["clean_rows_silent"] += int(not any(bool(S2.fires_on(rows[r]).any()) for r in np.nonzero(clean)[0]))
        pub["covered_witnesses_fire"] += int(all(bool(S2.fires_on(wk[w][1][None])[0, wk[w][0]]) for w in np.nonzero(~unc)[0]))
        pub["own_keys_fire"] += int(bool(S2.guard_ok().all()))
        pub["inside_box"] += int(bool(np.all(S2.det_tau.numpy() >= np.array(rec["tau_lo"])) and np.all(S2.det_tau.numpy() <= np.array(rec["tau_hi"]))))
        pub["n_raised"] += rec["n_raised"]
        pub["n_lowered"] += rec["n_lowered"]
    out["public_operator"] = pub
    out["pass"] = bool(out["n_instances"] >= 500 and out["n_equal_brute_force"] == out["n_instances"] == out["n_box_thresholds_reproduce_optimum"]
                       and out["n_problems_with_ties"] > 300 and all(pub[k] == pub["n"] for k in ("objective_equals_brute_force", "clean_rows_silent",
                                                                                              "covered_witnesses_fire", "own_keys_fire", "inside_box",
                                                                                              "keys_interface_equal"))
                       and pub["n_raised"] > 0 and pub["n_lowered"] > 0)
    return out


# ----------------------------------------------------------------------------- a8
def _eligible(role, p, lp):
    return role == "all" or (role == "last" and p == lp) or (role == "last+answer" and p >= lp)


def _ref_key_gate(kind, x, K, thr, role, lp, pos0=0):
    """Definition of the cosine / ball gates, position by position (numpy fp64): [(absolute position, edit)]."""
    out = []
    for p in range(x.shape[0]):
        if not _eligible(role, p + pos0, lp):
            continue
        k = x[p]
        if kind == "cos":
            c = (K @ k) / (np.linalg.norm(K, axis=1) * np.linalg.norm(k))
            j = int(c.argmax())
            if c[j] >= thr:
                out.append((p + pos0, j))
        else:
            dist = np.linalg.norm(K - k[None, :], axis=1)
            j = int(dist.argmin())
            if dist[j] < thr:
                out.append((p + pos0, j))
    return out


def test_baselines():
    log, tr = LL.synthetic_log(T=10, d=40, m=8, batch=5, seed=3, l2=4.0)
    dtype = torch.float32
    lin = linear(tr["W0"], dtype)
    W0d, WEd = lin.weight.data.clone(), tr["WE"].to(dtype)
    C = (WEd.double() - W0d.double()).numpy()
    K, V = log.K.numpy(), LL.rebase(log)
    g = torch.Generator().manual_seed(21)
    dmin = float(BL.key_distances(log).min())
    a_own = np.linalg.norm(K @ C.T, axis=1)
    specs = [("cos", 0.9, "last"), ("cos", 0.99, "all"), ("cos", 0.9, "last+answer"), ("exact", 0.999, "last"), ("exact", 0.999, "all"),
             ("ball", 0.25 * dmin, "last"), ("ball", 0.6 * dmin, "all"), ("router", float(np.quantile(a_own, 0.25)), None),
             ("router", float(np.quantile(a_own, 0.9)), None), ("zscore", 1.0, None), ("zscore", 2.0, None), ("zscore-call", 2.0, None)]
    out = {"n_cases": 0, "n_decisions_equal": 0, "n_outputs_bitwise": 0, "n_firing_cases": 0, "by_gate": {}, "kv_cache": {}}
    for trial in range(40):
        Tn, lp = 16, int(torch.randint(3, 10, (1,), generator=g))
        x = torch.randn(Tn, log.d, generator=g, dtype=torch.float64) + 0.3 * log.K.mean(0)
        for p in (lp, lp + 2, 1, lp + 4):                                # near-copies of stored keys at several noise levels
            i = int(torch.randint(0, log.T, (1,), generator=g))
            x[p] = log.K[i] + (0.0, 0.02, 0.1, 0.3, 1.0)[int(torch.randint(0, 5, (1,), generator=g))] * torch.randn(log.d, generator=g, dtype=torch.float64)
        x = x.to(dtype)
        xn = x.double().numpy()
        y0 = Fnn.linear(x[None], W0d)[0]
        yE = Fnn.linear(x[None], WEd)[0]
        for kind, thr, role in specs:
            if kind in ("cos", "exact"):
                gate = BL.cosine_gate(log, thr, role, W0=W0d) if kind == "cos" else BL.exact_gate(log, role, W0=W0d)
                ref = _ref_key_gate("cos", xn, K, thr, role, lp)
            elif kind == "ball":
                gate = BL.ball_gate(log, thr, role, W0=W0d)
                ref = _ref_key_gate("ball", xn, K, thr, role, lp)
            elif kind == "router":
                gate = BL.router_gate(log, WEd, W0d, eps=thr)
                ref = [(p, -1) for p in range(lp, Tn)] if np.linalg.norm(C @ xn[lp]) >= thr else []
            else:                                                    # a full-sequence forward: kv 'stale' and 'call' are the same gate
                gate = BL.zscore_gate(WEd, W0d, thr, kv="call" if kind == "zscore-call" else "stale")
                M = np.linalg.norm(xn @ C.T, axis=1)
                z = (M - M.mean()) / M.std(ddof=1)
                ref = [(p, -1) for p in range(Tn) if z[p] >= thr]
            with Hook(lin, gate) as hk:
                hk.begin(torch.tensor([lp]))
                y = lin(x[None])[0]
                f = hk.last
            mine = [] if f is None else list(zip(f.pos.tolist(), f.edit.tolist()))
            yr = y0.clone()
            for p, j in ref:
                yr[p] = yE[p] if j < 0 else (y0[p].double() + V[j]).to(dtype)
            nm = f"{kind}/{role}"
            d = out["by_gate"].setdefault(nm, {"cases": 0, "decisions_equal": 0, "outputs_bitwise": 0, "firing_positions": 0})
            d["cases"] += 1
            d["decisions_equal"] += int(mine == ref)
            d["outputs_bitwise"] += int(bits_equal(y, yr))
            d["firing_positions"] += len(ref)
            out["n_cases"] += 1
            out["n_decisions_equal"] += int(mine == ref)
            out["n_outputs_bitwise"] += int(bits_equal(y, yr))
            out["n_firing_cases"] += int(bool(ref))
    # ---- KV-cache decoding: a full prefix forward, then one-token forwards; absolute positions must be kept
    kv = {"n": 0, "positions_equal": 0, "outputs_bitwise": 0, "role_last_never_fires_in_steps": 0}
    S = ScopedLog.from_log(log, tau=0.5, W0=W0d)
    for trial in range(30):
        lp, n0, n_step = 5, 8, 6
        x = torch.randn(n0 + n_step, log.d, generator=g, dtype=torch.float64)
        x[lp], x[n0 + 1], x[n0 + 4], x[2] = log.K[trial % log.T], log.K[(trial + 3) % log.T], log.K[(trial + 5) % log.T], log.K[(trial + 7) % log.T]
        x = x.to(dtype)
        xn = x.double().numpy()
        Mp = np.linalg.norm(xn[:n0] @ C.T, axis=1)                       # zscore: statistics of the last full-sequence forward
        zs = (np.linalg.norm(xn @ C.T, axis=1) - Mp.mean()) / Mp.std(ddof=1)
        kv["n_step_positions_zscore_stale"] = kv.get("n_step_positions_zscore_stale", 0) + int(sum(zs[p] >= 1.0 for p in range(n0, n0 + n_step)))
        for nm, gate, ref in (
                ("scoped/last", S, [p for p in range(n0 + n_step) if p == lp]),
                ("scoped/last+answer", S.replace(role="last+answer"), None), ("scoped/all", S.replace(role="all"), None),
                ("router", BL.router_gate(log, WEd, W0d, eps=float(np.quantile(a_own, 0.25))), list(range(lp, n0 + n_step)) if np.linalg.norm(C @ xn[lp]) >= np.quantile(a_own, 0.25) else []),
                ("zscore", BL.zscore_gate(WEd, W0d, 1.0), [p for p in range(n0 + n_step) if zs[p] >= 1.0]),
                # kv='call' = the official hook: statistics per call, so a one-token step never swaps
                ("zscore-call", BL.zscore_gate(WEd, W0d, 1.0, kv="call"), [p for p in range(n0) if zs[p] >= 1.0])):
            if ref is None:                                              # scoped log: the semantics, position by position
                r = gate.scores(x).numpy()
                ref = [p for p in range(n0 + n_step) if _eligible(gate.role, p, lp) and (r[p] >= gate.theta().numpy()).any()]
            with Hook(lin, gate) as hk:
                hk.begin(torch.tensor([lp]), gen=True)
                ys = [lin(x[None, :n0])[0]] + [lin(x[None, p:p + 1])[0] for p in range(n0, n0 + n_step)]
                got = sorted(p for (_, p) in hk._pos)
            y = torch.cat(ys)
            yr = torch.cat([Fnn.linear(x[None, :n0], W0d)[0]] + [Fnn.linear(x[None, p:p + 1], W0d)[0] for p in range(n0, n0 + n_step)])
            yEs = torch.cat([Fnn.linear(x[None, :n0], WEd)[0]] + [Fnn.linear(x[None, p:p + 1], WEd)[0] for p in range(n0, n0 + n_step)])
            for p in ref:
                if nm.startswith("scoped"):
                    rr = gate.scores(x[p][None])[0]
                    j = int(torch.where(rr >= gate.theta(), rr, torch.full_like(rr, -INF)).argmax())
                    yr[p] = (yr[p].double() + gate.values[j]).to(dtype)
                else:
                    yr[p] = yEs[p]
            kv["n"] += 1
            kv["positions_equal"] += int(got == ref)
            kv["outputs_bitwise"] += int(bits_equal(y, yr))
            if nm == "scoped/last":
                kv["role_last_never_fires_in_steps"] += int(all(p < n0 for p in got))
            if nm == "zscore-call":
                kv["n_step_positions_zscore_call"] = kv.get("n_step_positions_zscore_call", 0) + sum(p >= n0 for p in got)
    out["kv_cache"] = kv
    out["coachhook_alpha"] = {"literal": BL.coachhook_alpha([3.0, 2.5, 1.7, 1.9], step=[0, 0, 1, 2]), "no_steps": BL.coachhook_alpha([3.0, 1.2]),
                              "init_only": BL.coachhook_alpha([1.0, 1.1], step=[0, 0])}
    out["pass"] = bool(out["n_decisions_equal"] == out["n_cases"] == out["n_outputs_bitwise"] and out["n_firing_cases"] > 0.4 * out["n_cases"]
                       and all(d["firing_positions"] > 0 for d in out["by_gate"].values())
                       and kv["positions_equal"] == kv["n"] == kv["outputs_bitwise"] and kv["role_last_never_fires_in_steps"] == 30
                       and kv["n_step_positions_zscore_stale"] > 0 and kv["n_step_positions_zscore_call"] == 0
                       and out["coachhook_alpha"] == {"literal": 1.7, "no_steps": 1.2, "init_only": 2.2})
    return out


# ----------------------------------------------------------------------------- a10
def test_witness_repair():
    log, tr = LL.synthetic_log(T=12, d=50, m=6, batch=4, seed=6, l2=4.0)
    S = CA.calibrate_log_only(ScopedLog.from_log(log), 1.0)[0]
    tau3 = float(S.det_tau[3])
    g = torch.Generator().manual_seed(4)
    wit = 0.45 * tau3 * log.K[3] + 0.02 * torch.randn(log.d, generator=g, dtype=torch.float64)       # a "rephrase" of edit 3 below its threshold
    r_w = float(S.scores(wit[None])[0, 3])
    out = {"r_witness": r_w, "tau_3": tau3, "fires_before": bool(S.fires_on(wit[None]).any())}
    S1, r1 = RP.witness_repair(S, 3, wit, margin=0.0, abs_margin=0.0)
    S2, r2 = RP.witness_repair(S, 3, wit)
    out["threshold"] = {"tau_equals_coordinate": float(S1.det_tau[3]) == r_w, "fires": bool(S1.fires_on(wit[None])[0, 3]), "applied": r1["applied"],
                        "edit_is_top1_on_witness": r1["edit_is_top1_on_witness"],
                        "others_unchanged": bool(torch.equal(S1.det_tau[torch.arange(12) != 3], S.det_tau[torch.arange(12) != 3])),
                        "guard_added": S1.guard_key.shape[0] == 13 and int(S1.guard_edit[-1]) == 3,
                        "default_margin_fires": bool(S2.fires_on(wit[None])[0, 3]), "default_margin_tau_below": float(S2.det_tau[3]) < r_w}
    ben = torch.stack([1.5 * wit, torch.randn(log.d, generator=g, dtype=torch.float64)])              # benign[0]: same direction, larger coordinate, still silent
    S3, r3 = RP.witness_repair(S, 3, wit, benign_keys=ben)
    S3f, r3f = RP.witness_repair(S, 3, wit, benign_keys=ben, force=True)
    out["conflict"] = {"benign_silent_before": not bool(S.fires_on(ben).any()), "conflict": r3["conflict"], "unchanged": S3 is S,
                       "conflicting_benign": [c["benign"] for c in r3["conflicts"]], "applied": r3["applied"], "forced_applied": r3f["applied"],
                       "forced_benign_fires": bool(S3f.fires_on(ben[:1]).any())}
    S4, r4 = RP.witness_repair(S, 5, log.K[5])
    out["already"] = {"already_fired": r4["already_fired"], "tau_unchanged": bool(torch.equal(S4.det_tau, S.det_tau)), "guard_added": S4.guard_key.shape[0] == 13}
    S5, r5 = RP.witness_repair(S, 3, wit, mode="detector", gamma=0.97, benign_keys=ben[1:])
    near = wit + 0.01 * wit.norm() / log.d ** 0.5 * torch.randn(log.d, generator=g, dtype=torch.float64)
    lin = linear(tr["W0"], torch.float64)
    with Hook(lin, S5.replace(W0=lin.weight.data.clone())) as hk:
        hk.begin(torch.tensor([0]))
        y = lin(wit[None, None])[0, 0]
    out["detector"] = {"D": S5.D, "kind_cos": int(S5.det_kind[-1]) == COS, "edit": int(S5.det_edit[-1]), "tau": float(S5.det_tau[-1]),
                       "r_at_witness": float(S5.scores(wit[None])[0, -1]), "fires_witness": bool(S5.edit_fires(wit[None])[0, 3]),
                       "fires_near": bool(S5.edit_fires(near[None])[0, 3]), "cos_near": float(S5.scores(near[None])[0, -1]),
                       "fires_random": bool(S5.fires_on(ben[1:]).any()), "fires_scaled_witness": bool(S5.edit_fires(3.0 * wit[None])[0, 3]),
                       "releases_value_of_edit_3": float((y - (lin.weight.data @ wit + S5.values[3])).abs().max()), "description": r5["detector_kind"]}
    S6, r6 = RP.witness_repair(S, 3, wit, mode="detector", benign_keys=ben)                           # 1.5 * wit has cosine 1: conflict
    out["detector_conflict"] = {"conflict": r6["conflict"], "unchanged": S6 is S}
    S7, r7 = RP.witness_repair(S, 3, wit, mode="detector", detector=wit, tau=0.8)
    out["detector_supplied"] = {"kind_lin": int(S7.det_kind[-1]) == LIN, "r_at_witness": float(S7.scores(wit[None])[0, -1]),
                                "s_equals_norm_sq": abs(float(S7.det_s[-1]) - float(wit @ wit)) < 1e-9, "fires": bool(S7.edit_fires(wit[None])[0, 3])}
    _, r8 = RP.least_repair(S1, wit[None])                                                           # the granted witness is now a guard
    out["witness_is_guarded"] = {"refused": r8["refused"], "conflict_edits": [c["edit"] for c in r8["conflicts"]]}
    # a repair that would not make the edit fire on its witness is refused (the witness becomes a guard = a must-fire key;
    # a guard that does not fire would be enforced by the next calibration, review F2)
    nf = {}
    for nm, kw in (("cosine_gamma_above_1", dict(mode="detector", gamma=1.0 + 1e-9)), ("supplied_detector_tau_above_1", dict(mode="detector", detector=wit, tau=1.5))):
        try:
            RP.witness_repair(S, 3, wit, **kw)
            nf[nm] = False
        except ValueError:
            nf[nm] = True
    nf["every_guard_of_every_granted_repair_fires"] = all(bool(Sx.guard_ok().all()) for Sx in (S1, S2, S3f, S4, S5, S7))
    out["nonfiring_witness_refused"] = nf
    ses = RP.Session(S)
    comp = 0.9 * tau3 * log.K[3] / float(S.scores(log.K[3][None])[0, 3]) * 1.3                        # fires edit 3
    rc = ses.complain(comp[None])
    rw = ses.why_not(3, 0.98 * comp)             # coordinate BELOW the repaired complaint's: granting it would make the complaint fire again
    rw2 = ses.why_not(3, 1.02 * comp)            # coordinate ABOVE it: the least repair left this key firing, nothing to do
    out["session"] = {"complaint_fired_before": bool(S.fires_on(comp[None]).any()), "complaint_resolved": rc["resolved"], "n_benign_recorded": len(ses.benign),
                      "why_not_conflict": rw["conflict"], "why_not_applied": rw["applied"], "above_already_fired": rw2["already_fired"],
                      "complaint_still_silent": not bool(ses.S.fires_on(comp[None]).any()), "n_history": len(ses.history)}
    t, c, d, s = out["threshold"], out["conflict"], out["detector"], out["session"]
    out["pass"] = bool(not out["fires_before"] and all(t.values()) and c["benign_silent_before"] and c["conflict"] and c["unchanged"] and c["conflicting_benign"] == [0]
                       and not c["applied"] and c["forced_applied"] and c["forced_benign_fires"] and all(out["already"].values())
                       and d["D"] == 13 and d["kind_cos"] and d["edit"] == 3 and d["tau"] == 0.97 and abs(d["r_at_witness"] - 1) < 1e-12 and d["fires_witness"]
                       and d["fires_near"] and not d["fires_random"] and d["fires_scaled_witness"] and d["releases_value_of_edit_3"] < 1e-12
                       and out["detector_conflict"]["conflict"] and out["detector_conflict"]["unchanged"] and all(out["detector_supplied"].values())
                       and out["witness_is_guarded"]["refused"] and out["witness_is_guarded"]["conflict_edits"] == [3] and all(out["nonfiring_witness_refused"].values())
                       and s["complaint_fired_before"] and s["complaint_resolved"] and s["n_benign_recorded"] == 1 and s["why_not_conflict"]
                       and not s["why_not_applied"] and s["above_already_fired"] and s["complaint_still_silent"] and s["n_history"] == 3)
    return out


# ----------------------------------------------------------------------------- a11
def test_calibrate_log_only():
    out = {"cases": [], "pass": True}
    for seed in range(4):
        log, _ = LL.synthetic_log(T=20, d=64, m=5, batch=5, seed=seed, l2=6.0)
        S = ScopedLog.from_log(log)
        R = (log.K @ log.W.T / log.own()[None, :]).numpy()
        pool = R[~np.eye(log.T, dtype=bool)]
        for q in (0.5, 0.9, 0.999, 1.0):
            S1, rec = CA.calibrate_log_only(S.replace(eta=1.7), q)
            ok = (rec["theta"] == float(np.quantile(pool, q)) and bool((S1.det_tau == rec["theta"]).all()) and S1.eta == 1.0 and bool(S1.guard_ok().all())
                  and rec["log_cross_firing_pairs"] == int((pool >= rec["theta"]).sum()) and (q < 1.0 or rec["log_cross_firing_pairs"] == 1))
            out["cases"].append({"seed": seed, "q": q, "theta": rec["theta"], "cross_firing_pairs": rec["log_cross_firing_pairs"], "ok": bool(ok)})
            out["pass"] = out["pass"] and bool(ok)
    Sg = S.replace(guard_key=0.3 * log.K)                                # own keys with coordinate 0.3: the q = 1 rule would silence them
    Sc, rc = CA.calibrate_log_only(Sg, 1.0)
    Sl, rl = CA.calibrate_log_only(Sg, 1.0, allow_silence=True)
    out["cap"] = {"theta": rc["theta"], "literal_would_silence": rc["n_own_keys_silenced_by_literal_rule"], "capped_tau_max": float(Sc.det_tau.max()),
                  "capped_own_keys_fire": bool(Sc.guard_ok().all()), "literal_silenced": int((~Sl.guard_ok()).sum()), "cap_value": 0.3 * (1 - TIE)}
    # add_guards: the B = 1 style own keys (here 0.8 k_i) become protected as well
    Sa, ra = RP.add_guards(S, 0.8 * log.K, torch.arange(log.T))
    Sa1, _ = CA.calibrate_conformal(CA.calibrate_log_only(Sa, 0.9)[0], np.full(30, 50.0), 0.1)
    S01, _ = CA.calibrate_conformal(CA.calibrate_log_only(S, 0.9)[0], np.full(30, 50.0), 0.1)
    out["add_guards"] = {"n_added": ra["n_added"], "n_firing_now": ra["n_firing_now"], "guards_fire_after_conformal_cap": bool(Sa1.guard_ok().all()),
                         "cap_with_extra_guards": float(Sa1.theta().max()), "cap_without": float(S01.theta().max()),
                         "extra_keys_fire_without_guarding": bool(S01.edit_fires(0.8 * log.K).diagonal().all())}
    # several detectors per edit: a granted cosine witness detector is not dragged down by the edit's own solver key
    S0 = CA.calibrate_log_only(S, 1.0)[0]
    g = torch.Generator().manual_seed(9)
    wit = 0.2 * log.K[3] + 0.02 * torch.randn(log.d, generator=g, dtype=torch.float64)    # r_3 about 0.2: below the threshold
    Sw, rw = RP.witness_repair(S0, 3, wit, mode="detector", gamma=0.95)
    Sm, rm = CA.calibrate_conformal(Sw, np.full(30, 3.0), 0.1)            # eta = 3: literal thresholds silence every guard
    Sq, rq = CA.calibrate_log_only(Sw, 1.0)
    th = Sm.theta()
    out["multi_detector"] = {"witness_applied": rw["applied"], "n_guards": int(Sw.guard_key.shape[0]), "literal_silences_guards": rm["n_guards_silenced_by_literal_rule"],
                             "literal_silences_own_keys": rm["n_own_keys_silenced_by_literal_rule"], "all_guards_fire": bool(Sm.guard_ok().all()),
                             "primary_3_capped_at_own": abs(float(th[3]) - (1 - TIE)) < 1e-9, "cos_detector_threshold": float(th[-1]),
                             "cos_detector_capped_at_witness": abs(float(th[-1]) - (1 - TIE)) < 1e-9,
                             "log_only_keeps_cos_threshold": float(Sq.det_tau[-1]) == 0.95, "log_only_guards_fire": bool(Sq.guard_ok().all())}
    md = out["multi_detector"]
    out["pass"] = bool(out["pass"] and md["witness_applied"] and md["n_guards"] == log.T + 1 and md["literal_silences_guards"] == log.T + 1
                       and md["literal_silences_own_keys"] == log.T and md["all_guards_fire"] and md["primary_3_capped_at_own"]
                       and md["cos_detector_capped_at_witness"] and md["log_only_keeps_cos_threshold"] and md["log_only_guards_fire"])
    bad = 0
    for q in (0.0, -0.1, 1.5):
        try:
            CA.calibrate_log_only(S, q)
        except ValueError:
            bad += 1
    c = out["cap"]
    ag = out["add_guards"]
    out["pass"] = bool(out["pass"] and ag["n_added"] == ag["n_firing_now"] == log.T and ag["guards_fire_after_conformal_cap"]
                       and abs(ag["cap_with_extra_guards"] - 0.8 * (1 - TIE)) < 1e-9 and abs(ag["cap_without"] - (1 - TIE)) < 1e-9
                       and not ag["extra_keys_fire_without_guarding"])
    out["pass"] = bool(out["pass"] and bad == 3 and c["theta"] > 0.3 and c["literal_would_silence"] == log.T and c["capped_own_keys_fire"]
                       and c["literal_silenced"] == log.T and abs(c["capped_tau_max"] - c["cap_value"]) < 1e-12)
    return out


# ----------------------------------------------------------------------------- a16: the guard contract (review F2)
def test_guard_contract():
    """A guard is a MUST-FIRE key.  add_guards refuses a key on which its edit does not fire now (default); with
    allow_nonfiring=True the key is accepted and every operator that SETS thresholds (calibrate_log_only,
    calibrate_conformal, scope_cut) lowers the edit's threshold to the key's coordinate, as documented, and counts it
    (n_guards_not_firing_before); least_repair (which only raises thresholds) sees no conflict in it."""
    log, _ = LL.synthetic_log(T=16, d=40, m=6, batch=4, seed=1, l2=4.0)
    S = CA.calibrate_log_only(ScopedLog.from_log(log), 0.9)[0]
    tau5 = float(S.det_tau[5])
    low = 0.5 * tau5 * log.K[5]                                          # r_5 = 0.5 tau_5: edit 5 does not fire on it
    r_low = float(S.key_scores(low[None])[0, 5])
    g = torch.Generator().manual_seed(31)
    X = torch.randn(4000, log.d, generator=g, dtype=torch.float64) + 0.3 * log.K[torch.randint(0, log.T, (4000,), generator=g)]
    rate = lambda Sx: float(Sx.fires_on(X)[:, 5].double().mean() * 100)                       # noqa: E731
    out = {"tau_5": tau5, "r_low": r_low, "fires_now": bool(S.edit_fires(low[None])[0, 5])}
    try:
        RP.add_guards(S, low[None], [5])
        out["nonfiring_refused"] = False
    except ValueError:
        out["nonfiring_refused"] = True
    try:                                                                 # one bad key among good ones: refused as a whole
        RP.add_guards(S, torch.cat([log.K[:3], low[None]]), [0, 1, 2, 5])
        out["mixed_refused"] = False
    except ValueError:
        out["mixed_refused"] = True
    Sa, ra = RP.add_guards(S, 0.999 * log.K, torch.arange(log.T))        # keys that fire: accepted
    Sg, rg = RP.add_guards(S, low[None], [5], allow_nonfiring=True)
    out["firing_keys"] = {"n_added": ra["n_added"], "n_firing_now": ra["n_firing_now"], "not_firing_now": ra["not_firing_now"]}
    out["allow_nonfiring"] = {"n_added": rg["n_added"], "n_firing_now": rg["n_firing_now"], "not_firing_now": rg["not_firing_now"],
                              "thresholds_unchanged": bool(torch.equal(Sg.det_tau, S.det_tau)), "firing_rate_edit_5_pct": rate(Sg)}
    cap = r_low * (1 - TIE)
    res = {}
    S1, r1 = CA.calibrate_log_only(Sg, 0.9)
    S0, r0 = CA.calibrate_log_only(S, 0.9)
    res["calibrate_log_only"] = {"tau_5": float(S1.theta()[5]), "tau_5_without_the_guard": float(S0.theta()[5]), "n_guards_not_firing_before": r1["n_guards_not_firing_before"],
                                 "without_guard_n_not_firing_before": r0["n_guards_not_firing_before"], "guard_fires": bool(S1.guard_ok().all()),
                                 "firing_rate_edit_5_pct": rate(S1), "firing_rate_without_the_guard_pct": rate(S0)}
    S2, r2 = CA.calibrate_conformal(Sg, np.full(30, 1.5), 0.1)
    S2b, r2b = CA.calibrate_conformal(S, np.full(30, 1.5), 0.1)
    res["calibrate_conformal"] = {"tau_5": float(S2.theta()[5]), "tau_5_without_the_guard": float(S2b.theta()[5]),
                                  "n_guards_not_firing_before": r2["n_guards_not_firing_before"], "without_guard_n_not_firing_before": r2b["n_guards_not_firing_before"],
                                  "guard_fires": bool(S2.guard_ok().all()), "guarantee": r2["guarantee"], "guarantee_without_the_guard": r2b["guarantee"],
                                  "firing_rate_edit_5_pct": rate(S2), "firing_rate_without_the_guard_pct": rate(S2b)}
    rows = [k[None] for k in X[:40]]
    S3, r3 = RP.scope_cut(Sg, rows, [], 1.0)
    S3b, r3b = RP.scope_cut(S, rows, [], 1.0)
    res["scope_cut"] = {"tau_5": float(S3.theta()[5]), "tau_5_without_the_guard": float(S3b.theta()[5]), "n_guards_not_firing_before": r3["n_guards_not_firing_before"],
                        "without_guard_n_not_firing_before": r3b["n_guards_not_firing_before"], "guard_fires": bool(S3.guard_ok().all()),
                        "firing_rate_edit_5_pct": rate(S3), "firing_rate_without_the_guard_pct": rate(S3b)}
    out["setters"] = res
    comp = 1.3 * tau5 * log.K[5]                                         # fires edit 5 (r = 1.3 tau), below its own key (r = 1)
    S4, r4 = RP.least_repair(Sg, comp[None])
    out["least_repair"] = {"complaint_fired": bool(Sg.edit_fires(comp[None])[0, 5]), "refused": r4["refused"], "conflicts": r4["conflicts"],
                           "resolved": r4["resolved"], "own_key_fires": bool(S4.guard_ok()[5]), "nonfiring_guard_still_silent": not bool(S4.guard_ok()[-1])}
    lr = out["least_repair"]
    out["pass"] = bool(not out["fires_now"] and out["nonfiring_refused"] and out["mixed_refused"]
                       and out["firing_keys"] == {"n_added": log.T, "n_firing_now": log.T, "not_firing_now": []}
                       and out["allow_nonfiring"]["not_firing_now"] == [0] and out["allow_nonfiring"]["thresholds_unchanged"]
                       and all(abs(v["tau_5"] - cap) < 1e-12 and v["tau_5_without_the_guard"] > 1.5 * cap and v["n_guards_not_firing_before"] == 1
                               and v["without_guard_n_not_firing_before"] == 0 and v["guard_fires"]
                               and v["firing_rate_edit_5_pct"] > v["firing_rate_without_the_guard_pct"] for v in res.values())
                       and res["calibrate_conformal"]["guarantee"].startswith("void") and res["calibrate_conformal"]["guarantee_without_the_guard"] == "exact"
                       and lr["complaint_fired"] and not lr["refused"] and lr["conflicts"] == [] and lr["resolved"] and lr["own_key_fires"]
                       and lr["nonfiring_guard_still_silent"])
    return out


# ----------------------------------------------------------------------------- a12: the real logs (CPU)
def test_real_logs():
    hz = EV.harness()
    Z = hz.TB.Z
    from vendor_v7.vlm import DEFAULT_DTYPE
    P11 = EV.EXP + "/08_rescope/p11_retrofit/out"
    out = {"logs": {}, "pass": True}
    for name in ("TB-Q100-ae-l2_3", "TB-L100-ae-l2_3"):
        log = LL.from_testbed(name)
        cfg = hz.TB.LOGS[name]
        zcfg = dataclasses.replace(Z.ZConfig.for_backbone(cfg["backbone"]), dtype=Z._dtype_tag(DEFAULT_DTYPE[cfg["backbone"]]))
        tg = [Z.load_cached(cfg["backbone"], cfg["layer"], u, zcfg, None) for u in log.meta["uids"]]
        res = {"n_targets_found": int(sum(t is not None for t in tg)), "T": log.T}
        ref = torch.stack([t.z for t in tg]).double() - torch.stack([t.h for t in tg]).double()      # z*_i - h_i(W_0), clean model
        e_f = LL.rebase_error(log, ref)
        e_x = LL.rebase_error(log, ref, LL.rebase_exact(log, LL.batch_deltas(name)))
        with open(os.path.join(P11, name, "rebase.json")) as f:
            th = json.load(f)["vs_cache"]
        res["rebase"] = {"factor_form_max": e_f["rebased"]["max"], "factor_form_median": e_f["rebased"]["median"], "exact_form_max": e_x["rebased"]["max"],
                         "not_rebased_max": e_f["not_rebased"]["max"], "not_rebased_median": e_f["not_rebased"]["median"],
                         "p11a_factor_form_max": th["factor_form"]["max"], "p11a_exact_form_max": th["exact_form"]["max"],
                         "p11a_not_rebased_max": th["no_rebase_U"]["max"],
                         "oracle_targets": int(sum(bool(t.oracle) for t in tg))}
        rb = res["rebase"]
        ok_rb = (abs(rb["factor_form_max"] - rb["p11a_factor_form_max"]) < 1e-9 and abs(rb["exact_form_max"] - rb["p11a_exact_form_max"]) < 1e-9
                 and abs(rb["not_rebased_max"] - rb["p11a_not_rebased_max"]) < 1e-9 and rb["factor_form_max"] < 0.05 < rb["not_rebased_max"])
        rows, pool = EV.load_rows(name)
        res["split"] = SP.summary(rows, pool)
        sp = res["split"]
        ok_sp = (sp["n_twins"] == 25 and sp["n_twins_by_ptype"] == {"image": 25, "text": 0} and sp["n_heldout"] - sp["n_heldout_disjoint"] == 25
                 and len(SP.select(rows, pool, "heldout_disjoint")) == sp["n_heldout_disjoint"] and len(SP.select(rows, pool, "repair")) == sp["n_repair"])
        # review F8, recomputed here from the raw rows (not with splits.py): disjoint rows that share their QUESTION with a
        # REPAIR row (other image), and disjoint rows that are the same query as another disjoint row
        rep_query = {(r["ptype"], r["prompt"], r["image"]) for r in rows if pool[r["probe_idx"]] == "repair"}
        rep_text = {(r["ptype"], r["prompt"]) for r in rows if pool[r["probe_idx"]] == "repair"}
        dj = [r for r in rows if r["y0_valid"] and pool[r["probe_idx"]] == "heldout" and (r["ptype"], r["prompt"], r["image"]) not in rep_query]
        cnt = {}
        for r in dj:
            cnt[(r["ptype"], r["prompt"], r["image"])] = cnt.get((r["ptype"], r["prompt"], r["image"]), 0) + 1
        ind = {"n_disjoint": len(dj), "n_question_in_repair": {p: sum((r["ptype"], r["prompt"]) in rep_text for r in dj if r["ptype"] == p) for p in ("image", "text")},
               "n_distinct_queries": len(cnt), "n_rows_in_duplicate_groups": sum(c for c in cnt.values() if c > 1)}
        strict = SP.select(rows, pool, "heldout_strict")
        an = SP.annotate(rows, pool)
        pairs = {((r["ptype"], r["prompt"], r["image"]), a["query_id"]) for a, r in zip(an, rows)}      # query <-> query_id must be a bijection
        res["split_independent"] = ind
        ok_sp = (ok_sp and ind["n_disjoint"] == sp["n_heldout_disjoint"] and ind["n_question_in_repair"] == sp["n_disjoint_question_in_repair"]
                 and ind["n_distinct_queries"] == sp["n_disjoint_distinct_queries"] and ind["n_rows_in_duplicate_groups"] == sp["n_disjoint_rows_in_duplicate_groups"]
                 and len(strict) == sp["n_heldout_strict"] == ind["n_disjoint"] - sum(ind["n_question_in_repair"].values())
                 and all((r["ptype"], r["prompt"]) not in rep_text for r in strict)
                 and len(pairs) == len({p[0] for p in pairs}) == len({p[1] for p in pairs}))
        # hard negatives (CPU: a stub handle with the log's items): the row count P11a evaluated; a wrong path is an error
        items_log, _ = hz.TB.load_stream(cfg["stream"])
        stub = types.SimpleNamespace(backbone=cfg["backbone"], items=items_log, name=name)
        with open(os.path.join(P11, name, "C", "_meta_main.json")) as f:
            n_hn_p11 = sum(m["sid"].startswith("HN-") for m in json.load(f)["meta"])
        try:
            EV.load_hard_negatives(stub, hn_dir=os.path.join(EV.HN_DIR, "no_such_dir"))
            missing_raises = False
        except FileNotFoundError:
            missing_raises = True
        res["hard_negatives"] = {"n": len(EV.load_hard_negatives(stub)), "n_p11a": n_hn_p11, "missing_file_raises": missing_raises,
                                 "missing_ok_gives_empty": EV.load_hard_negatives(stub, hn_dir=os.path.join(EV.HN_DIR, "no_such_dir"), missing_ok=True) == []}
        hn = res["hard_negatives"]
        ok_sp = ok_sp and hn["n"] == hn["n_p11a"] > 0 and hn["missing_file_raises"] and hn["missing_ok_gives_empty"]
        draw = SP.sample(list(range(200)), 50, 3)
        res["sample"] = {"seeded": draw == SP.sample(list(range(200)), 50, 3), "sorted_unique": draw == sorted(set(draw)) and len(draw) == 50,
                         "not_the_first_n": draw != list(range(50)), "other_seed_differs": draw != SP.sample(list(range(200)), 50, 4)}
        ok_sp = ok_sp and all(res["sample"].values())
        S = ScopedLog.from_log(log)
        res["log_only"] = {}
        ok_lo = True
        for q, cfgname in ((1.0, "R1-POOL1-REPL"), (0.999, "R1-POOL0.999-REPL")):
            S1, rec = CA.calibrate_log_only(S, q)
            with open(os.path.join(P11, name, "C", cfgname + ".json")) as f:
                tc = json.load(f)["config"]["tau_info"]
            res["log_only"][str(q)] = {"theta": rec["theta"], "p11a_theta": tc["theta"], "cross_firing_pairs": rec["log_cross_firing_pairs"],
                                       "p11a_cross_firing_pairs": tc["log_cross_firing_pairs"], "own_keys_silenced": rec["n_own_keys_silenced"]}
            ok_lo = ok_lo and abs(rec["theta"] - tc["theta"]) < 1e-12 and rec["log_cross_firing_pairs"] == tc["log_cross_firing_pairs"] and rec["n_own_keys_silenced"] == 0
        res["s_median"] = float(log.own().median())
        res["ok"] = bool(ok_rb and ok_sp and ok_lo and res["n_targets_found"] == log.T)
        out["logs"][name] = res
        out["pass"] = out["pass"] and res["ok"]
    return out


# ----------------------------------------------------------------------------- a13: firing sets of the real logs (CPU)
def test_real_logs_firing():
    """Thresholds of this package applied to the per-sequence coordinates STORED by P10 (fp64: last prompt position,
    maximum over positions, raw cosines) must reproduce the firing flags that P10 / P11a observed in their forwards
    (1279 / 1288 B = 1 sequences: edits x {rel, tgen, igen} and every valid probe row)."""
    X = EV.EXP + "/08_rescope"
    out = {"logs": {}, "pass": True}
    for name in ("TB-Q100-ae-l2_3", "TB-L100-ae-l2_3"):
        D = f"{X}/p10_scope_probe/out/{name}/coords"
        with open(D + "/seqs.json") as f:
            seqs = json.load(f)["seqs"]
        s64 = np.load(D + "/seq64.npz")
        sid = [f"edit:{q['metric']}:{q['edit']}" if q["kind"] == "edit" else f"probe:{q['ptype']}:{q['probe_idx']}" for q in seqs]
        log = LL.from_testbed(name)
        s = log.own().numpy()
        S = ScopedLog.from_log(log)
        with open(f"{X}/p10_scope_probe/out/{name}/calib.json") as f:
            tau_p10 = np.array(json.load(f)["tau_self"])
        r_last, r_max = s64["c_last"] / s[None, :], s64["c_max"] / s[None, :]
        pred = {"R1-REL0.35-REPL": (r_last >= S.replace(det_tau=torch.full((log.T,), 0.35, dtype=torch.float64)).theta().numpy()[None, :]).any(1),
                "R1-POOL1-REPL": (r_last >= CA.calibrate_log_only(S, 1.0)[0].theta().numpy()[None, :]).any(1),
                "NDB-R1-0.95": s64["cos_last"].max(1) >= float(BL.cosine_gate(log, 0.95).det_tau[0]),
                "P10-HARD-SELF-noE": (r_max >= (tau_p10 / s)[None, :]).any(1)}
        with open(f"{X}/p11_retrofit/out/{name}/C/_meta_main.json") as f:
            m11 = json.load(f)["meta"]
        res = {}
        for cfg, p in pred.items():
            if cfg.startswith("P10"):
                with open(f"{X}/p10_scope_probe/out/{name}/E/HARD-SELF-noE.json") as f:
                    th = {r["sid"]: r["n_fire_pos"] > 0 for r in json.load(f)["rows"]}
            else:
                with open(f"{X}/p11_retrofit/out/{name}/C/{cfg}.json") as f:
                    rows = json.load(f)["rows"]
                th = {m["sid"]: rows[k]["nfp"] > 0 for k, m in enumerate(m11)}
            pr = [k for k, x in enumerate(sid) if x.startswith("probe")]
            res[cfg] = {"n": len(sid), "fired_equal": int(sum(bool(p[k]) == th[x] for k, x in enumerate(sid))),
                        "probe_fire_rate": float(np.mean(p[pr]) * 100), "their_probe_fire_rate": float(np.mean([th[sid[k]] for k in pr]) * 100)}
            out["pass"] = out["pass"] and res[cfg]["fired_equal"] == len(sid)
        out["logs"][name] = res
    return out


# ----------------------------------------------------------------------------- a14: operators on real keys (CPU)
def test_real_keys_operators():
    """The operators on REAL last-prompt keys (captured by P11b: p11_scopecut/out/<log>/keys, read-only; role 'last').
    1. conformal: P11b's experiment (score = max_i c_i / (1.05 x largest coordinate on the other solver keys), its seeded
       draws of n REPAIR rows, false-firing rate on the HELD-OUT rows) through calibrate_conformal / fires_on: the
       mean rate over the 500 draws must equal P11b's stored number.
    2. least repair of every firing REPAIR row (thresholds 0.35): all silent afterwards, same thresholds in reversed order.
    3. null repair at real scale (d = 11008): constraint residuals.
    4. witness repair (threshold mode) of the half-A T-Gen keys against the REPAIR rows as recorded benign queries."""
    X = EV.EXP + "/08_rescope/p11_scopecut/out"
    NS, ALPHAS, SEED, DRAWS = (20, 50, 100, 200), (0.01, 0.05, 0.1), 20261112, 500        # P11b's protocol constants
    out = {"logs": {}, "pass": True}
    for name in ("TB-Q100-ae-l2_3", "TB-L100-ae-l2_3"):
        with open(f"{X}/{name}/keys/meta.json") as f:
            seqs = json.load(f)["seqs"]
        K = torch.from_numpy(np.load(f"{X}/{name}/keys/K_last.npy")).double()
        log = LL.from_testbed(name)
        T, s = log.T, log.own()
        fit = [q["n"] for q in seqs if q["kind"] == "probe" and q["pool"] == "repair"]
        test = [q["n"] for q in seqs if q["kind"] == "probe" and q["pool"] == "heldout"]
        rel = [next(q["n"] for q in seqs if q["kind"] == "edit" and q["metric"] == "rel" and q["edit"] == i) for i in range(T)]
        tgen = [next(q["n"] for q in seqs if q["kind"] == "edit" and q["metric"] == "tgen" and q["edit"] == i) for i in range(T)]
        Kfit, Ktest = K[fit], K[test]
        res = {"n_fit": len(fit), "n_test": len(test)}
        # ---- 1. conformal regression against P11b
        M = log.K @ log.W.T
        ms = torch.where(torch.eye(T, dtype=torch.bool), torch.full_like(M, -INF), M).max(0).values       # largest c_i on the other solver keys
        S = ScopedLog.from_log(log).replace(det_tau=1.05 * ms / s)
        S, _ = RP.add_guards(S, K[rel], torch.arange(T))                 # P11b's "own" = stored key AND B = 1 key
        sc_fit = CA.benign_scores(S, Kfit)
        R_test = S.scores(Ktest)
        with open(f"{X}/{name}/conformal.json") as f:
            theirs = json.load(f)["cells"]
        cells = []
        for (n, alpha) in ((50, 0.05), (200, 0.1), (100, 0.01), (20, 0.01), (100, 0.1)):
            rng = np.random.default_rng([SEED, NS.index(n), ALPHAS.index(alpha)])
            idx = np.stack([rng.choice(len(fit), n, replace=False) for _ in range(DRAWS)])
            ffr, own, same = [], [], 0
            for dr in range(DRAWS):
                S2, rec = CA.calibrate_conformal(S, sc_fit[idx[dr]], alpha, allow_silence=True)
                fire = (R_test >= S2.theta()[None, :]).any(1)
                if dr < 3:
                    same += int(torch.equal(fire, S2.fires_on(Ktest).any(1)))
                ffr.append(float(fire.double().mean()))
                g = S2.guard_ok()
                own.append(float((g[:T] & g[T:]).double().mean()))
            th = theirs[f"SELF|{n}|{alpha}"]
            c = {"n": n, "alpha": alpha, "mean_ffr": float(np.mean(ffr)), "p11b_mean_ffr": th["ffr"]["mean"], "max_ffr": float(np.max(ffr)),
                 "p11b_max_ffr": th["ffr"]["max"], "own_coverage": float(np.mean(own)), "p11b_own_coverage": th["coverage"]["own"]["mean"],
                 "eta_infinite": rec["eta_infinite"], "fires_on_equals_precomputed": same,
                 # the guarantee is marginal over the calibration draw (review F7): spread of the single draws
                 "q95_ffr": float(np.quantile(ffr, 0.95)), "share_of_draws_above_alpha": float(np.mean(np.array(ffr) > alpha))}
            c["ok"] = bool(abs(c["mean_ffr"] - c["p11b_mean_ffr"]) < 1e-12 and abs(c["max_ffr"] - c["p11b_max_ffr"]) < 1e-12
                           and abs(c["own_coverage"] - c["p11b_own_coverage"]) < 1e-12 and same == 3)
            cells.append(c)
        res["conformal"] = cells
        # ---- 2. least repair of every firing REPAIR row
        S35 = ScopedLog.from_log(log, tau=0.35)
        row_fires = lambda Sx, Kx: (Sx.key_scores(Kx) >= Sx.theta()[None, :]).any(1)               # noqa: E731  (one key per sequence, like the hook)
        f0, h0 = row_fires(S35, Kfit), row_fires(S35, Ktest)
        firing = f0.nonzero().flatten().tolist()
        ses, ses_r = RP.Session(S35), RP.Session(S35)
        recs = [ses.complain(Kfit[i][None]) for i in firing]
        for i in reversed(firing):
            ses_r.complain(Kfit[i][None])
        f1, h1 = row_fires(ses.S, Kfit), row_fires(ses.S, Ktest)
        refused = [i for i, r in zip(firing, recs) if r["refused"]]
        res["least_repair"] = {"n_firing_repair_rows": len(firing), "n_refused_own_key_conflict": len(refused),
                               "n_repaired_rows_still_firing": int(f1[[i for i in firing if i not in refused]].sum()) if len(firing) > len(refused) else 0,
                               "n_repair_rows_firing_after": int(f1.sum()), "own_keys_fire": bool(ses.S.guard_ok().all()),
                               "order_independent_bitwise": bool(torch.equal(ses.S.det_tau, ses_r.S.det_tau)),
                               "n_thresholds_raised": int((ses.S.det_tau > S35.det_tau).sum()),
                               "heldout_firing_pct_before": float(h0.double().mean() * 100), "heldout_firing_pct_after": float(h1.double().mean() * 100)}
        # ---- 3. null repair at real scale
        Sn, rn = RP.null_repair(S35, Kfit[:100])
        Sz, rz = RP.null_repair(S35, Kfit[:100], zero_force_log=True)
        cross = (log.K @ Sz.det_w.T) / s[None, :] - torch.eye(T, dtype=torch.float64)
        res["null_repair"] = {"max_abs_benign_r": rn["max_abs_benign_r"], "max_own_rel_err": rn["max_own_rel_err"], "rank_benign": rn["rank_benign"],
                              "rel_change_median": rn["rel_change_median"], "infeasible": rn["infeasible"],
                              "zero_force_max_abs_benign_r": rz["max_abs_benign_r"], "zero_force_max_abs_cross_r": float(cross.abs().max()),
                              "zero_force_max_own_rel_err": rz["max_own_rel_err"],
                              "heldout_firing_pct_before": float(h0.double().mean() * 100),
                              "heldout_firing_pct_after": float(row_fires(Sn, Ktest).double().mean() * 100)}
        # ---- 4. witness repair of the half-A T-Gen keys, REPAIR rows recorded as benign
        Sw, n_app, n_conf, n_already = S35, 0, 0, 0
        for i in range(T // 2):
            Sw, rw = RP.witness_repair(Sw, i, K[tgen[i]], benign_keys=Kfit)
            n_app += int(rw["applied"])
            n_conf += int(rw["conflict"])
            n_already += int(rw["already_fired"])
        fw = row_fires(Sw, Kfit)
        ok_w = Sw.guard_ok()
        res["witness_repair"] = {"n_witnesses": T // 2, "already_fired": n_already, "applied": n_app, "conflicts": n_conf,
                                 "granted_witnesses_fire": bool(ok_w[T:].all()), "n_guards": int(ok_w.numel()),
                                 "n_silent_repair_rows_firing_after": int((fw & ~f0).sum()), "own_keys_fire": bool(ok_w[:T].all()),
                                 "heldout_firing_pct_after": float(row_fires(Sw, Ktest).double().mean() * 100)}
        lr, nr, wr = res["least_repair"], res["null_repair"], res["witness_repair"]
        res["ok"] = bool(all(c["ok"] for c in cells) and lr["n_repaired_rows_still_firing"] == 0 and lr["own_keys_fire"] and lr["order_independent_bitwise"]
                         and lr["n_repair_rows_firing_after"] <= lr["n_refused_own_key_conflict"])
        res["ok"] = bool(res["ok"] and nr["max_abs_benign_r"] < 1e-9 and nr["max_own_rel_err"] < 1e-9 and nr["infeasible"] == []
                         and nr["zero_force_max_abs_benign_r"] < 1e-9 and nr["zero_force_max_abs_cross_r"] < 1e-9 and nr["zero_force_max_own_rel_err"] < 1e-9
                         and wr["granted_witnesses_fire"] and wr["n_silent_repair_rows_firing_after"] == 0 and wr["own_keys_fire"]
                         and wr["already_fired"] + wr["applied"] + wr["conflicts"] == T // 2 and wr["n_guards"] == T + wr["already_fired"] + wr["applied"])
        out["logs"][name] = res
        out["pass"] = out["pass"] and res["ok"]
    return out


TESTS = {"a6_null_repair": test_null_repair, "a7_scope_cut": test_scope_cut, "a8_baselines": test_baselines, "a10_witness_repair": test_witness_repair,
         "a11_calibrate_log_only": test_calibrate_log_only, "a16_guard_contract": test_guard_contract, "a12_real_logs": test_real_logs,
         "a13_real_logs_firing": test_real_logs_firing, "a14_real_keys_operators": test_real_keys_operators}
