"""P12 -- linear repair streams: n reported complaints repaired with a linear method; what happens to the
HELD-OUT pool, the logged edits and the non-complaints as a function of n?

usage: CUDA_VISIBLE_DEVICES=k python run_streams.py --log TB-Q100-ae-l2_3 [--seeds 0,1,2] [--ns 1,10,ALL]
                                                   [--methods COMP_B,...] [--cap_gb 11] [--limit K]

Reported sets: the first n complaints of the nested stream of (log, seed) (p12_common.build_stream; harness
sampler, REPAIR pool, stratified image / text), n in {1, 2, 5, 10, 20, 50, 100, ALL}; seeds 0, 1, 2 for
n <= 20, seed 0 for n >= 50.  Every method is a call of the shared harness (07_repair_v3/harness), with the
options spelled out here (alpha = 1 everywhere, nothing tuned):

  NONE           repair("NONE"): the unrepaired log W_E.            (evaluated on every valid REPAIR row)
  ALL            repair("ROLLBACK_ALL"): W_0.                       (evaluated on every valid REPAIR row)
  COMP_B         repair("COMP", first n, protect=("K_ans",), query="answer", form="adapter") -- one joint
                 closed-form compensation (gate 1e-8, tol_rel 1e-9, fp64 solve on the CPU).
  COMP_BP        the same with protect=("K_ans", "probe:100000") = K_ans + the answer-position keys of every
                 REPAIR-pool non-complaint (known-good regression suite).
  COMP_S         sequential: repair("COMP", [c], protect=("K_ans",), query="answer", form="adapter",
                 prior=<adapter so far>, extra_basis=<keys of every earlier repaired complaint>) for the
                 complaints in stream order (rq4's COMP_S loop; the extra basis is orthonormalised against
                 the protection basis with two passes of block Gram-Schmidt).
  COMP_SP        the same with protect=("K_ans", "probe:100000").
  COMP_B_ALLPOS  repair("COMP", first n, protect=("K_ans",), query="all", form="adapter"), n in {1,2,5,10};
                 feasibility = no dropped direction and exact restoration (recorded).
  REEDIT_B       z* per complaint: repair("REEDIT", [c], variant="k0+kp", return_targets=True) on W_E (vendor
                 zopt, B=1, the harness oracle fallbacks; cached per complaint, the same z* for every n / seed);
                 install: repair("REEDIT", first n, variant="k0+kp", targets=<z*>, chunk=1) -- ONE AlphaEdit
                 solve ([redacted: describes a related submission under double-blind review] + the log's Kp + the log's l2), bf16/fp16 write like the editor;
                 complaints whose z* never becomes an oracle are not installed (recorded).
  REEDIT_S       the stream continued like the original log: batches of 10 complaints in stream order; batch b
                 is repair("REEDIT", batch, variant="k0+kp", targets=<z* on W_E>, chunk=1) on the log handle
                 advanced by the earlier batches (W := W_after, Kp := [Kp, keys of the installed complaints],
                 residual = z* - h at the current weights), exactly Editor.edit_batch(update_Kp=True) with
                 targets precomputed before the stream (the convention of the testbed build).
  REEDIT_SC      (exploratory variant, seed 0, n >= 20) the same stream with z* re-optimised on the CURRENT
                 weights for every batch (repair("REEDIT", [c], return_targets=True) on the advanced handle).
  RB_TOT         repair("ROLLBACK_FO", first n, budget=30, offload=True, record_flags=True): greedy first-order
                 rollback on the summed reported mean log p(y0); stops when every reported complaint is
                 reproduced or after 30 rollbacks.
  RB_UNION       W(union_c R_c), R_c = repair("ROLLBACK_FO", [c], budget=10).meta["R"] (cached per complaint).

Each config -> out/<log>/<METHOD>__n<n>__s<seed>.json (harness evaluate(per_item=True) record + cost + method
diagnostics); finished configs are skipped, several workers may share the directory (lock files).  The
corrections of the D2 configs (n in {10, 100, ALL}, seed 0) are saved to out/<log>/corr/ for run_d2.py.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import os
import re
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p12_common import *  # noqa: F401,F403,E402
from p12_common import _orth  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402


# =========================================================================== per-complaint caches
class TargetCache:
    """z* of every complaint on W_E (harness REEDIT z* phase, B=1) + its last-prompt key; disk-backed."""

    def __init__(self, H, fh):
        self.H, self.fh = H, fh
        self.path = os.path.join(MYCACHE, f"targets_{H.name}.pt")
        self.mem = {}
        self._mtime = None
        self._reload()

    def _reload(self):
        if os.path.exists(self.path):
            mt = os.path.getmtime(self.path)
            if mt != self._mtime:
                d = torch.load(self.path, map_location="cpu", weights_only=False)
                for k, v in d.items():
                    self.mem.setdefault(k, v)
                self._mtime = mt

    def _save(self):
        with flock(self.path + ".lock"):
            if os.path.exists(self.path):
                d = torch.load(self.path, map_location="cpu", weights_only=False)
                for k, v in d.items():
                    self.mem.setdefault(k, v)
            tmp = self.path + f".tmp{os.getpid()}"
            torch.save(self.mem, tmp)
            os.replace(tmp, self.path)
            self._mtime = os.path.getmtime(self.path)

    @staticmethod
    def target(e):
        return VZ.Target(**{k: v for k, v in e["blob"].items() if k in VZ.Target.__dataclass_fields__})

    def ensure(self, cs):
        self._reload()
        new = 0
        for c in cs:
            g = c["group"]
            e = self.mem.get(g)
            # valid: a first-attempt oracle (the ladder is a no-op for it) or an entry made with the ladder
            if e is not None and (e["oracle"] or e.get("zproc") == ZPROC):
                continue
            self.mem[g] = compute_target(self.H, c)
            if e is not None:
                n = self.mem[g]
                log(f"    z* of {g} recomputed with the oracle fallback ladder: oracle {e['oracle']} (attempts "
                    f"{e['attempts']}) -> {n['oracle']} (attempts {n['attempts']})", self.fh)
            new += 1
            if new % 25 == 0:
                self._save()
                log(f"    z* targets: {new} new ({len(self.mem)} cached)", self.fh)
        if new:
            self._save()
        return [self.mem[c["group"]] for c in cs]


def capture_k1(H, c):
    """Edit-module input at the last prompt token of the complaint's editor item (B=1): the key the
    editor's solve uses and appends to Kp (Editor.capture_keys)."""
    it = H.complaint_item(c)
    b = H.vlm.build_batch([it["image"]], [it["src"]], [it["alt"]], fmt="vqa")
    with torch.no_grad():
        with H.vlm.capture(b, [H.L], positions="last", module="edit", io=("in",), dtype=torch.float32) as ck:
            H.vlm.forward(b, stop_after_layer=H.L)
    return ck.inp[H.L][0].detach().cpu()


ZPROC = "vendor zopt B=1, then the harness oracle fallback ladder (REEDIT chunk=1 path)"


def compute_target(H, c):
    """z* of one complaint.  chunk=REEDIT_CHUNK is REQUIRED here: in harness/methods.py (as of
    2026-11-09 18:47, the RQ4 `chunk` edit) the documented oracle fallback ladder of m_reedit (fixed
    loop at 2x steps, then 200 steps / early_stop 0.01) is nested under the `chunk is not None` branch,
    so repair("REEDIT", ..., chunk=None) stops after the vendor attempt.  With one item, chunk=1 runs
    the identical vendor attempt (same B=1 batch, same seed) and then the ladder."""
    corr, cost = repair("REEDIT", H, [c], variant="k0+kp", return_targets=True, chunk=REEDIT_CHUNK)
    t = corr.meta.pop("_targets")[0]
    return {"blob": {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in t.to_blob().items()},
            "oracle": bool(t.oracle), "attempts": int(corr.meta["zopt_attempts"][0]), "zproc": ZPROC,
            "single_installed": corr.kind == "weight",
            "zcost": {"passes_zopt": cost["passes_zopt"], "flops_zopt": cost["flops_zopt"],
                      "sec_zopt": cost["sec"]["zopt"], "n_backward_zopt": cost["n_backward_zopt"],
                      "sec_total_single": cost["sec"]["total"]},
            "k1": capture_k1(H, c)}


class RollbackCache:
    """Per-complaint greedy first-order rollback set (harness ROLLBACK_FO, budget 10) on W_E; disk-backed."""

    def __init__(self, H, fh):
        self.H, self.fh = H, fh
        self.path = os.path.join(MYCACHE, f"rbfo_{H.name}.json")
        self.mem = {}
        self._reload()

    def _reload(self):
        if os.path.exists(self.path):
            for k, v in jload(self.path).items():
                self.mem.setdefault(k, v)

    def _save(self):
        with flock(self.path + ".lock"):
            self._reload()
            atomic_jdump(self.mem, self.path)

    def ensure(self, cs):
        self._reload()
        new = 0
        for c in cs:
            g = c["group"]
            if g in self.mem:
                continue
            corr, cost = repair("ROLLBACK_FO", self.H, [c], budget=RB_UNION_BUDGET)
            self.mem[g] = {"R": [int(i) for i in corr.meta["R"]], "resolved_internal": bool(corr.meta["resolved_internal"]),
                           "cost": cost_small(cost)}
            new += 1
            if new % 50 == 0:
                self._save()
                log(f"    per-complaint rollback sets: {new} new ({len(self.mem)} cached)", self.fh)
        if new:
            self._save()
        return [self.mem[c["group"]] for c in cs]


# =========================================================================== log view (stream continuation)
@contextlib.contextmanager
def log_view(H):
    """Inside the block the log handle may be advanced (advance()); restored exactly on exit."""
    isre = lambda k: isinstance(k, tuple) and len(k) > 0 and k[0] == "reedit"  # noqa: E731
    saved = (H.WE, H.Kp_log, {k: v for k, v in H._setup.items() if isre(k)}, H._D_act)
    try:
        yield
    finally:
        H.WE, H.Kp_log, H._D_act = saved[0], saved[1], saved[3]
        for k in [k for k in H._setup if isre(k)]:
            del H._setup[k]
        H._setup.update(saved[2])
        H.reset()


def advance(H, W_new, K1_new):
    """The log after one more batch: weights W_new, Kp extended by the installed keys K1_new (d, b)."""
    H.WE = W_new
    H.Kp_log = torch.cat([H.Kp_log, K1_new.to(H.Kp_log.device, H.Kp_log.dtype)], 1)
    H._D_act = None
    for k in [k for k in H._setup if isinstance(k, tuple) and len(k) > 0 and k[0] == "reedit"]:
        del H._setup[k]                                 # the AlphaEdit setup (V^T Kp) is rebuilt from the log


# =========================================================================== methods
def orth_against(Kq, blocks):
    X = Kq
    for _ in range(2):                                   # block Gram-Schmidt, twice
        for Q in blocks:
            if Q is not None:
                X = X - Q @ (Q.T @ X)
    B, st = _orth(X)
    return B


def key_counts(H, cs, query):
    """Number of query keys per complaint: answer tokens of y0 (stored n_ans_tok) or the sequence length."""
    if query == "answer":
        return [int(c["n_ans_tok"]) for c in cs]
    return [int(H.complaint_batch(c)["seq_len"][0]) for c in cs]


def restore_diag(H, corr, cs, Kq, counts):
    """Per complaint: ||(X + D) K_c|| / ||D K_c|| (how exactly W_0's module output is restored)."""
    X_L, X_R = corr.Lf.to(H.solve_dev), corr.Rf.to(H.solve_dev)
    D = H.D_act
    res, j = [], 0
    for c, A in zip(cs, counts):
        K = Kq[:, j:j + A]
        j += A
        DK = D @ K
        res.append({"group": c["group"], "n_keys": A,
                    "restore_rel_err": float((X_L @ (X_R.T @ K) + DK).norm() / max(float(DK.norm()), 1e-30))})
    assert j == Kq.shape[1]
    return res


def m_comp_batch(H, cs, protect, query):
    corr, cost = repair("COMP", H, cs, protect=protect, query=query, form="adapter", keep_keys=True)
    Kq = corr.meta.pop("_Kq")
    pc = restore_diag(H, corr, cs, Kq, key_counts(H, cs, query))
    del Kq
    m = corr.meta
    feas = bool(m["P1_rel"] < 1e-6 and max(p["restore_rel_err"] for p in pc) < 1e-6 and not m["infeasible"])
    extra = {"per_complaint": pc, "feasible_exact": feas,
             "overdetermined": bool(m["n_query"] + m["protect_rank"] > H.d_in)}
    rank = {"correction": int(m["rank_kept"]), "protection": int(m["protect_rank"]), "n_query_keys": int(m["n_query"]),
            "n_dropped": int(m["n_dropped"]), "d_in": H.d_in, "kind": "exact (adapter, fp64)"}
    c = cost_small(cost)
    return corr, {"passes": c["passes"], "sec_total": c["sec"]["total"], "sec": c["sec"], "n_backward": c["n_backward"],
                  "n_forward": c["n_forward"], "flops_total": c["flops_total"], "n_calls": 1,
                  "setup_sec_excluded": c.get("setup_sec_excluded")}, extra, rank


def m_comp_restore_all(H, cs):
    """Exploratory (outside the protocol): ONE joint compensation whose queries are the answer positions of
    EVERY valid REPAIR-pool row -- the reported complaints AND the known-good rows -- protecting K_ans.
    Where COMP-B+ pins the known-good rows to the edited output (protection), this restores W_0's module
    output on them too: the largest subspace the REPAIR pool defines.  The same harness call as COMP-B."""
    _, comp, nonc = repair_rows(H)
    assert {c["group"] for c in cs} == {c["group"] for c in comp}, "COMP_BR is defined for n = ALL only"
    corr, cost, extra, rank = m_comp_batch(H, list(cs) + list(nonc), PROT, "answer")
    extra["n_known_good_rows_restored"] = len(nonc)
    extra["per_complaint"] = extra["per_complaint"][:len(cs)]         # aligned with the reported complaints
    return corr, cost, extra, rank


def m_comp_seq(H, cs, protect, fh):
    Q, bst = protection_basis(H, protect)
    prior, E, costs, steps, Ks, cnt, t_basis = None, None, [], [], [], [], 0.0
    for k, c in enumerate(cs):
        corr, cost = repair("COMP", H, [c], protect=protect, query="answer", form="adapter", prior=prior,
                            extra_basis=E, keep_keys=True)
        Kq = corr.meta.pop("_Kq")
        with Timer() as tb:                                  # accumulating the protection = part of the cost
            if corr.meta["rank_kept"] > 0:                   # keys inside the protected span add nothing
                Eb = orth_against(Kq, [Q, E])
                E = Eb if E is None else torch.cat([E, Eb], 1)
        t_basis += tb.sec
        cost["sec"]["basis_update"] = tb.sec
        cost["sec"]["total"] += tb.sec
        costs.append(cost)
        Ks.append(Kq.cpu())
        cnt.append(int(Kq.shape[1]))
        m = corr.meta
        steps.append({"group": c["group"], "rank_kept": m["rank_kept"], "n_query": m["n_query"],
                      "protect_rank": m["protect_rank"], "total_rank": m.get("total_rank", m["rank_kept"]),
                      "new_fro": m.get("new_fro"), "fro": corr.fro(H), "P1_rel": m["P1_rel"], "cond": m["cond"],
                      "infeasible": m["infeasible"]})
        prior = corr
        if k and k % 50 == 0:
            log(f"    COMP seq step {k}/{len(cs)} rank {steps[-1]['total_rank']} protect {steps[-1]['protect_rank']} "
                f"fro {steps[-1]['fro']:.3f}", fh)
    corr = Correction("adapter", Lf=prior.Lf, Rf=prior.Rf, gate=prior.gate,
                      meta={"method": "COMP", "mode": "sequential", "order": [c["group"] for c in cs]})
    Kall = torch.cat(Ks, 1).to(H.solve_dev)
    pc = restore_diag(H, corr, cs, Kall, cnt)
    del Kall, Ks
    cost = add_costs(costs)
    cost["sec_basis_update"] = t_basis
    cost["setup_sec_excluded"] = bst.get("sec_setup", 0.0)
    n_drop = int(sum(s["n_query"] - s["rank_kept"] for s in steps))
    extra = {"steps": steps, "per_complaint": pc, "feasible_exact": bool(n_drop == 0 and max(p["restore_rel_err"] for p in pc) < 1e-6)}
    # diagnostic (not part of the repair cost): the sequential operator against the joint solve of the
    # same complaints -- with accumulating protection both satisfy the same constraints in the same row
    # space, so they are the same matrix in exact arithmetic
    cb, _ = repair("COMP", H, cs, protect=protect, query="answer", form="adapter")
    Xs = corr.Lf.to(H.solve_dev) @ corr.Rf.to(H.solve_dev).T
    Xb = cb.Lf.to(H.solve_dev) @ cb.Rf.to(H.solve_dev).T
    extra["identity_vs_batch"] = {"rel_fro_diff": float((Xs - Xb).norm() / Xb.norm()), "fro_seq": float(Xs.norm()),
                                  "fro_batch": float(Xb.norm()), "rank_batch": int(cb.Rf.shape[1])}
    del Xs, Xb, cb
    rank = {"correction": int(corr.Rf.shape[1]),
            "protection": int((0 if Q is None else Q.shape[1]) + (0 if E is None else E.shape[1])),
            "protection_static": int(0 if Q is None else Q.shape[1]),
            "n_query_keys": int(sum(cnt)), "n_dropped": n_drop, "d_in": H.d_in, "kind": "exact (adapter, fp64)"}
    return corr, cost, extra, rank


def _zsum(ents):
    z = {"passes_zopt": {}, "flops_zopt": 0.0, "sec_zopt": 0.0, "n_backward_zopt": 0}
    for e in ents:
        for k, v in e["zcost"]["passes_zopt"].items():
            z["passes_zopt"][k] = z["passes_zopt"].get(k, 0) + v
        z["flops_zopt"] += e["zcost"]["flops_zopt"]
        z["sec_zopt"] += e["zcost"]["sec_zopt"]
        z["n_backward_zopt"] += e["zcost"]["n_backward_zopt"]
    return z


def _merge_passes(*ps):
    out = {}
    for p in ps:
        for k, v in p.items():
            out[k] = out.get(k, 0) + v
    return out


def _reedit_protection(H, kp_cols):
    pr = H.tl.metrics["build"]["protected"]
    return {"K0_projected_rank": int(pr["protected_rank"]), "Kp_soft_columns": int(kp_cols), "d_in": H.d_in}


def m_reedit_batch(H, cs, tc):
    ents = tc.ensure(cs)
    tg = [tc.target(e) for e in ents]
    corr, cost = repair("REEDIT", H, cs, variant="k0+kp", targets=tg, chunk=REEDIT_CHUNK)
    m = corr.meta
    z = _zsum(ents)
    c = cost_small(cost)
    cst = {"passes": _merge_passes(z["passes_zopt"], c["passes"]), "sec_total": z["sec_zopt"] + c["sec"]["total"],
           "sec_zopt": z["sec_zopt"], "sec_install": c["sec"]["total"], "sec": c["sec"],
           "n_backward": z["n_backward_zopt"] + c["n_backward"],
           "n_forward": sum(z["passes_zopt"].values()) + c["n_forward"],
           "flops_total": z["flops_zopt"] + c["flops_total"], "n_calls": 1,
           "note": "z* cost = sum of the cached per-complaint z* optimisations (B=1) of the reported complaints",
           "setup_sec_excluded": c.get("setup_sec_excluded")}
    inst = list(m.get("installed_rows", []))
    extra = {"zproc": ZPROC, "installed": len(inst), "not_installed": [cs[j]["group"] for j in m.get("not_installed_rows", [])],
             "zopt_attempts": [e["attempts"] for e in ents],
             "solve_attainment": m.get("solve_attainment"), "dW_over_W0": m.get("dW_over_W0"),
             "applied_dW_over_W0": m.get("applied_dW_over_W0"),
             "z_delta_over_h_median": float(np.median([e["blob"]["delta_over_h"] for e in ents]))}
    rank = {"correction": len(inst), "protection": _reedit_protection(H, H.Kp_log.shape[1]),
            "kind": "nominal (solver rank = installed complaints; the bf16/fp16 write makes the applied "
                    "difference numerically full rank)"}
    corr.meta.pop("attempt_log", None)
    corr.meta.pop("targets", None)
    return corr, cst, extra, rank


def m_reedit_stream(H, cs, tc, zmode, fh):
    ents = tc.ensure(cs)                                 # keys (and, for zmode "WE", the targets)
    kp0 = int(H.Kp_log.shape[1])
    batches = [list(range(s, min(s + REEDIT_BATCH, len(cs)))) for s in range(0, len(cs), REEDIT_BATCH)]
    per_batch, costs, zc, installed, not_inst = [], [], [], [], []
    with log_view(H):
        for bi, idx in enumerate(batches):
            sub = [cs[j] for j in idx]
            if zmode == "WE":
                tg = [tc.target(ents[j]) for j in idx]
                zc += [ents[j]["zcost"] for j in idx]
                att = [ents[j]["attempts"] for j in idx]
            else:                                           # z* on the current (advanced) weights, B=1
                tg, att = [], []
                for c in sub:
                    c1, k1 = repair("REEDIT", H, [c], variant="k0+kp", return_targets=True, chunk=REEDIT_CHUNK)
                    tg.append(c1.meta.pop("_targets")[0])
                    att.append(int(c1.meta["zopt_attempts"][0]))
                    zc.append({"passes_zopt": k1["passes_zopt"], "flops_zopt": k1["flops_zopt"],
                               "sec_zopt": k1["sec"]["zopt"], "n_backward_zopt": k1["n_backward_zopt"]})
                    del c1
            corr, cost = repair("REEDIT", H, sub, variant="k0+kp", targets=tg, chunk=REEDIT_CHUNK)
            m = corr.meta
            ok = list(m.get("installed_rows", []))
            installed += [sub[j]["group"] for j in ok]
            not_inst += [sub[j]["group"] for j in m.get("not_installed_rows", [])]
            costs.append(cost)
            per_batch.append({"batch": bi, "n": len(sub), "installed": len(ok), "Kp_cols": int(H.Kp_log.shape[1]),
                              "solve_attainment": m.get("solve_attainment"), "dW_over_W0": m.get("dW_over_W0"),
                              "step_fro": cost["correction_fro"], "zopt_attempts": att,
                              "oracle": [bool(t.oracle) for t in tg]})
            if corr.kind == "weight" and ok:
                K1 = torch.stack([ents[idx[j]]["k1"] for j in ok], 1)          # (d, b) fp32
                advance(H, corr.W, K1)
            del corr
        W_final = H.WE
        kp_end = int(H.Kp_log.shape[1])
    meta = {"method": "REEDIT", "mode": "stream", "zmode": zmode, "batch": REEDIT_BATCH, "variant": "k0+kp",
            "l2": float(H.cfg["l2"]), "n_batches": len(batches)}
    corr = Correction("weight", W=W_final, meta=meta) if installed else Correction("none", meta=meta)
    z = {"passes_zopt": {}, "flops_zopt": 0.0, "sec_zopt": 0.0, "n_backward_zopt": 0}
    for e in zc:
        z["passes_zopt"] = _merge_passes(z["passes_zopt"], e["passes_zopt"])
        z["flops_zopt"] += e["flops_zopt"]
        z["sec_zopt"] += e["sec_zopt"]
        z["n_backward_zopt"] += e["n_backward_zopt"]
    a = add_costs(costs)
    cst = {"passes": _merge_passes(z["passes_zopt"], a["passes"]), "sec_total": z["sec_zopt"] + a["sec_total"],
           "sec_zopt": z["sec_zopt"], "sec_install": a["sec_total"], "n_backward": z["n_backward_zopt"] + a["n_backward"],
           "n_forward": sum(z["passes_zopt"].values()) + a["n_forward"],
           "flops_total": z["flops_zopt"] + a["flops_total"], "n_calls": len(costs),
           "note": "per-batch AlphaEdit setup (V^T Kp of the advanced log) is excluded like every per-log setup"}
    extra = {"zproc": ZPROC, "installed": len(installed), "not_installed": not_inst, "per_batch": per_batch}
    rank = {"correction": len(installed), "protection": _reedit_protection(H, kp_end), "Kp_cols_start": kp0,
            "kind": "nominal (sum of the per-batch solver ranks)"}
    return corr, cst, extra, rank


def m_rb_tot(H, cs):
    corr, cost = repair("ROLLBACK_FO", H, cs, budget=RB_TOT_BUDGET, offload=True, record_flags=True)
    traj = corr.meta["trajectory"]
    for t in traj:
        t.pop("flags", None)
    f0 = corr.meta.pop("flags0", [])
    c = cost_small(cost)
    cst = {"passes": c["passes"], "sec_total": c["sec"]["total"], "sec": c["sec"], "n_backward": c["n_backward"],
           "n_forward": c["n_forward"], "flops_total": c["flops_total"], "n_calls": 1}
    extra = {"k": corr.meta["k"], "R": corr.meta["R"], "resolved_internal": corr.meta["resolved_internal"],
             "flags0_resolved": int(sum(f0)), "budget": RB_TOT_BUDGET}
    rank = {"correction": int(corr.meta["k"]), "protection": 0,
            "kind": "nominal (number of rolled-back edits; rank-1 solver term per edit, exact batch delta for whole batches)"}
    return corr, cst, extra, rank


def m_rb_union(H, cs, rc):
    ents = rc.ensure(cs)
    U = sorted({int(i) for e in ents for i in e["R"]})
    with Timer() as t:
        W = H.rollback_weight(U) if U else None
    meta = {"method": "ROLLBACK_FO", "mode": "union", "R": U, "k": len(U)}
    corr = Correction("weight", W=W, meta=meta) if U else Correction("none", meta=meta)
    cst = add_costs([e["cost"] for e in ents])
    cst["sec_total"] += t.sec
    cst["sec_union_weight"] = t.sec
    extra = {"k": len(U), "R": U, "n_resolved_individually": int(sum(e["resolved_internal"] for e in ents)),
             "per_complaint_k": [len(e["R"]) for e in ents], "budget_each": RB_UNION_BUDGET}
    rank = {"correction": len(U), "protection": 0, "kind": "nominal (number of rolled-back edits)"}
    return corr, cst, extra, rank


OPTIONS = {
    "COMP_B": {"harness": "COMP", "protect": list(PROT), "query": "answer", "form": "adapter", "gate": 1e-8, "tol_rel": 1e-9},
    "COMP_BP": {"harness": "COMP", "protect": list(PROT_PLUS), "query": "answer", "form": "adapter", "gate": 1e-8, "tol_rel": 1e-9},
    "COMP_S": {"harness": "COMP", "protect": list(PROT), "query": "answer", "form": "adapter", "gate": 1e-8, "tol_rel": 1e-9,
               "sequential": "prior + extra_basis (keys of earlier repaired complaints), stream order"},
    "COMP_SP": {"harness": "COMP", "protect": list(PROT_PLUS), "query": "answer", "form": "adapter", "gate": 1e-8,
                "tol_rel": 1e-9, "sequential": "prior + extra_basis (keys of earlier repaired complaints), stream order"},
    "COMP_B_ALLPOS": {"harness": "COMP", "protect": list(PROT), "query": "all", "form": "adapter", "gate": 1e-8, "tol_rel": 1e-9},
    "COMP_BR": {"harness": "COMP", "protect": list(PROT), "query": "answer", "form": "adapter", "gate": 1e-8, "tol_rel": 1e-9,
                "queries": "every valid REPAIR-pool row (complaints and known-good rows)", "exploratory": True},
    "REEDIT_B": {"harness": "REEDIT", "variant": "k0+kp", "l2": "log", "targets": "z* on W_E per complaint (B=1, oracle required)",
                 "chunk": REEDIT_CHUNK, "form": "merged"},
    "REEDIT_S": {"harness": "REEDIT", "variant": "k0+kp", "l2": "log", "targets": "z* on W_E per complaint (B=1, oracle required)",
                 "chunk": REEDIT_CHUNK, "form": "merged", "stream": f"batches of {REEDIT_BATCH}, Kp accumulates, residual at the current weights"},
    "REEDIT_SC": {"harness": "REEDIT", "variant": "k0+kp", "l2": "log", "targets": "z* on the CURRENT weights per batch (B=1, oracle required)",
                  "chunk": REEDIT_CHUNK, "form": "merged", "stream": f"batches of {REEDIT_BATCH}, Kp accumulates"},
    "RB_TOT": {"harness": "ROLLBACK_FO", "budget": RB_TOT_BUDGET, "offload": True, "record_flags": True},
    "RB_UNION": {"harness": "ROLLBACK_FO", "budget": RB_UNION_BUDGET, "union": True},
}


def make(H, method, cs, tc, rc, fh):
    if method == "COMP_B":
        return m_comp_batch(H, cs, PROT, "answer")
    if method == "COMP_BP":
        return m_comp_batch(H, cs, PROT_PLUS, "answer")
    if method == "COMP_B_ALLPOS":
        return m_comp_batch(H, cs, PROT, "all")
    if method == "COMP_BR":
        return m_comp_restore_all(H, cs)
    if method == "COMP_S":
        return m_comp_seq(H, cs, PROT, fh)
    if method == "COMP_SP":
        return m_comp_seq(H, cs, PROT_PLUS, fh)
    if method == "REEDIT_B":
        return m_reedit_batch(H, cs, tc)
    if method == "REEDIT_S":
        return m_reedit_stream(H, cs, tc, "WE", fh)
    if method == "REEDIT_SC":
        return m_reedit_stream(H, cs, tc, "current", fh)
    if method == "RB_TOT":
        return m_rb_tot(H, cs)
    if method == "RB_UNION":
        return m_rb_union(H, cs, rc)
    raise ValueError(method)


# =========================================================================== driver
def summary_line(name, rec, n_rep):
    col = rec["collateral"]["all"]
    nv = col["n_noncomplaints"] + col["n_complaints"]
    d1 = 100.0 * (col["n_complaints"] - col["complaints_fixed"] + col["noncomplaints_lost"]) / nv
    e = rec["edits"]
    return (f"  {name}: reported resolved {sum(r['D1_resolved'] for r in rec['complaints'])}/{n_rep} | HELD-OUT D1 "
            f"{d1:.1f}% (fixed {col['complaints_fixed']}/{col['n_complaints']}, lost {col['noncomplaints_lost']}"
            f"/{col['n_noncomplaints']}) | Rel/TGen/IGen {e['rel']['exact']:.0f}/{e['tgen']['exact']:.0f}/"
            f"{e['igen']['exact']:.0f} broken {rec['edits_broken_total']} fixed {rec['edits_fixed_total']} | "
            f"T-Loc {rec['loc']['tloc']['after']:.2f} I-Loc {rec['loc']['iloc']['after']:.2f} | fro/log "
            f"{rec['fro_rel_to_log']:.3f}")


def run_endpoint(H, name, odir, rows, fh):
    path = os.path.join(odir, f"{name}.json")
    if os.path.exists(path):
        return
    t0 = time.time()
    corr, cost = repair("NONE" if name == "NONE" else "ROLLBACK_ALL", H, rows)
    rec = evaluate(H, corr, rows, d2=False, per_item=True)
    for r, row in zip(rec["complaints"], rows):
        r["is_complaint_D1"] = bool(row["is_complaint_D1"])
        r["stored_reproduced"] = bool(row["reproduced"])
    hr = rec["heldout_rows"]
    stored = {(r["probe_idx"], r["ptype"]): r for r in H.rows}
    hstore = [stored[(int(x["group"].split("_")[1]), x["ptype"])] for x in hr]
    san = {}
    if name == "NONE":
        san = {"repair_D1_set_equals_stored": all(r["D1_resolved"] == r["stored_reproduced"] for r in rec["complaints"]),
               "heldout_D1_set_equals_stored": all(x["reproduced_after"] == bool(s["reproduced"]) for x, s in zip(hr, hstore)),
               "n_repair_rows": len(rows), "n_heldout_rows": len(hr),
               "edits_identical_to_baseline": all(rec["edits"][m]["identical_em_acc"] for m in ("rel", "tgen", "igen"))}
    else:
        san = {"repair_D1_complaints_after": int(sum(not r["D1_resolved"] for r in rec["complaints"])),
               "heldout_D1_complaints_after": int(sum(not x["reproduced_after"] for x in hr)),
               "weight_equals_W0": bool(torch.equal(corr.W, H.W0))}
    out = {"log": H.name, "config": name, "method": name, "n": name, "n_reported": 0, "seed": None, "options": {},
           "eval": rec, "cost": {"passes": cost["passes"], "sec_total": cost["sec"]["total"], "n_backward": 0,
                                 "n_forward": 0, "n_calls": 1},
           "meta": meta_clean(corr.meta), "extra": {}, "sanity": san,
           "rank": {"correction": 0 if name == "NONE" else H.T, "protection": 0, "kind": "end point"},
           "eval_sec": time.time() - t0, "peak_gpu_bytes": torch.cuda.max_memory_allocated(), "gpu_procs": gpu_procs(),
           "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    atomic_jdump(clean(out), path)
    log(summary_line(name, rec, len(rows)) + f" | sanity {san} | {time.time() - t0:.0f}s", fh)


def copy_alias(odir, method, n, seed, fh):
    """A config that is the same computation as another one: copy its record (marked alias_of)."""
    ali = alias_of(method, n)
    name, src_name = config_name(method, n, seed), config_name(ali, n, seed)
    src = os.path.join(odir, f"{src_name}.json")
    if not os.path.exists(src):
        return False
    d = jload(src)
    d.update(config=name, method=method, alias_of=src_name, options=OPTIONS[method])
    atomic_jdump(d, os.path.join(odir, f"{name}.json"))
    if os.path.exists(os.path.join(odir, "corr", f"{src_name}.pt")):
        with open(os.path.join(odir, "corr", f"{name}.alias"), "w") as f:
            f.write(src_name)
    log(f"  {name}: identical computation to {src_name} -> record copied (alias)", fh)
    return True


def supersede_stale_reedit(odir, fh):
    """REEDIT records written before the z* fallback fix (no extra.zproc) in which a complaint was NOT
    installed are moved to superseded_zfallback/ (kept, never deleted) and re-run.  Records without a
    dropped complaint are unaffected: every target was a first-attempt oracle."""
    sdir = os.path.join(odir, "superseded_zfallback")
    moved = []
    for fn in sorted(os.listdir(odir)):
        if not (fn.startswith("REEDIT_") and fn.endswith(".json")) or fn.endswith(".failed.json"):
            continue
        d = jload(os.path.join(odir, fn))
        ex = d.get("extra", {})
        if ex.get("zproc") == ZPROC or not ex.get("not_installed"):
            continue
        os.makedirs(sdir, exist_ok=True)
        os.replace(os.path.join(odir, fn), os.path.join(sdir, fn))
        base = fn[:-5]
        for suf in (".pt", ".alias"):
            cp = os.path.join(odir, "corr", base + suf)
            if os.path.exists(cp):
                os.replace(cp, os.path.join(sdir, base + suf))
        dp = os.path.join(odir, "d2", fn)
        if os.path.exists(dp):
            os.replace(dp, os.path.join(sdir, "d2_" + fn))
        moved.append((base, ex["not_installed"]))
    if moved:
        log(f"  superseded {len(moved)} first-pass REEDIT records (z* without the oracle fallback, a complaint "
            f"not installed): {moved}", fh)
    return moved


def run_one(H, logname, odir, method, n, seed, streams, tc, rc, fh):
    name = config_name(method, n, seed)
    cs = reported(streams[seed], n)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    corr, cost, extra, rank = make(H, method, cs, tc, rc, fh)
    t1 = time.time()
    peak_repair = torch.cuda.max_memory_allocated()
    rec = evaluate(H, corr, cs, d2=False, per_item=True)
    out = {"log": logname, "config": name, "method": method, "n": n, "n_reported": len(cs), "seed": seed,
           "options": OPTIONS[method], "reported_groups": [c["group"] for c in cs],
           "reported_ptypes": [c["ptype"] for c in cs],
           "eval": rec, "cost": cost, "meta": meta_clean(corr.meta), "extra": extra, "rank": rank,
           "repair_wall_sec": t1 - t0, "eval_sec": time.time() - t1, "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
           "peak_gpu_bytes_repair": peak_repair, "gpu_procs": gpu_procs(), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
           "torch_threads": torch.get_num_threads()}
    if seed == D2_SEED and n in D2_NS and (method in METHODS or method == "COMP_BR"):
        save_corr(corr, os.path.join(odir, "corr", f"{name}.pt"))
    atomic_jdump(clean(out), os.path.join(odir, f"{name}.json"))
    log(summary_line(name, rec, len(cs)) + f" | rank {rank.get('correction')} | repair {cost.get('sec_total', 0):.1f}s "
        f"(wall {t1 - t0:.0f}s) eval {time.time() - t1:.0f}s peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GiB", fh)
    del corr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--ns", default=None, help="comma list (e.g. 1,10,ALL); default: the whole protocol")
    ap.add_argument("--methods", default=None)
    ap.add_argument("--cap_gb", type=float, default=None)
    ap.add_argument("--limit", type=int, default=None, help="stop after this many evaluated configs (smoke)")
    ap.add_argument("--tag", default="", help="suffix of the out directory (smoke runs)")
    ap.add_argument("--no_extra", action="store_true")
    ap.add_argument("--phase", default=None, help="s0 | rest | extra (p12_common.phase_plan); overrides --seeds")
    a = ap.parse_args()
    seeds = tuple(int(s) for s in a.seeds.split(","))
    ns = None if a.ns is None else {x if x == "ALL" else int(x) for x in a.ns.split(",")}
    methods = None if a.methods is None else set(a.methods.split(","))
    os.makedirs(LOGD, exist_ok=True)
    fh = open(os.path.join(LOGD, f"streams_{a.log}{a.tag}.log"), "a")
    odir = os.path.join(OUT, a.log + a.tag)
    os.makedirs(os.path.join(odir, "corr"), exist_ok=True)
    log(f"==== run_streams {vars(a)} pid {os.getpid()} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", fh)
    H, cap = load_log(a.log, a.cap_gb)
    start_watchdog(fh)
    if H.checks.get("collapsed_working_point"):
        raise SystemExit("collapsed working point")
    B = baseline(H)
    rows, comp, nonc = repair_rows(H)
    streams = {s: build_stream(H, s) for s in SEEDS}
    log(f"loaded {a.log}; cap {cap} GiB; solve device {H.solve_dev}; checks {H.checks}; baseline {B['checks']}; "
        f"REPAIR valid rows {len(rows)} complaints {len(comp)} non-complaints {len(nonc)}", fh)
    setup_path = os.path.join(odir, "_setup.json")
    if not os.path.exists(setup_path):
        atomic_jdump({"log": a.log, "load_checks": H.checks, "baseline_checks": B["checks"],
                      "n_repair_rows": len(rows), "n_repair_complaints": len(comp), "n_repair_noncomplaints": len(nonc),
                      "n_heldout_rows": B["checks"]["n_heldout_rows"], "d_in": H.d_in, "m_out": H.m_out, "T": H.T,
                      "l2": H.cfg["l2"], "backbone": H.backbone, "layer": H.L,
                      "fro_log": float(H.D_act.norm()), "fro_W0": float(H.W0.double().norm()),
                      "stored": {k: H.tl.metrics[k] for k in ("Rel_exact", "Rel_token", "TGen_exact", "IGen_exact",
                                                              "TLoc_probes", "ILoc_probes")},
                      "K0_protected": H.tl.metrics["build"]["protected"],
                      "streams": {str(s): [c["group"] for c in streams[s]] for s in SEEDS},
                      "split_seed": SM.SPLIT_SEED, "n_base": N_BASE}, setup_path)
    supersede_stale_reedit(odir, fh)
    run_endpoint(H, "NONE", odir, rows, fh)
    run_endpoint(H, "ALL", odir, rows, fh)
    tc, rc = TargetCache(H, fh), RollbackCache(H, fh)
    base = phase_plan(a.phase) if a.phase else plan(seeds, extra=not a.no_extra)
    todo = [(m, n, s) for (m, n, s) in base if (ns is None or n in ns) and (methods is None or m in methods)]
    n_done, n_oom = 0, 0
    for (method, n, seed) in todo:
        name = config_name(method, n, seed)
        path = os.path.join(odir, f"{name}.json")
        if os.path.exists(path) or os.path.exists(path.replace(".json", ".failed.json")):
            continue
        ali = alias_of(method, n)
        if ali is not None:
            copy_alias(odir, method, n, seed, fh)            # if the source is not there yet: the final pass
            continue
        yield_if_blocking(fh, f"(before {name})")
        lock = os.path.join(odir, f".lock_{name}")
        if not claim(lock):
            continue
        try:
            own = 0
            while True:
                try:
                    run_one(H, a.log, odir, method, n, seed, streams, tc, rc, fh)
                    n_done += 1
                    with contextlib.suppress(OSError):          # a stale marker of an earlier OOM
                        os.remove(path.replace(".json", ".oom.json"))
                    break
                except Exception as e:  # noqa: BLE001
                    if not is_oom(e):
                        raise
                    n_oom += 1
                    msg = str(e)
                    kind = oom_kind(msg) if isinstance(e, torch.OutOfMemoryError) else "card_full"
                    log(f"  {name}: CUDA out of memory ({kind}; cap {cap} GiB): {msg[:330]}", fh)
                    if kind != "own_cap":
                        release(lock)
                        leave_card(fh, f"card full during {name}")      # does not return
                    own += 1
                    H.reset()
                    gc.collect()
                    torch.cuda.empty_cache()
                    if own >= 2:
                        # NOT a result and NOT permanent: the config does not fit under this process's cap.  The
                        # marker does not block a later run; aggregate.py lists the config as unfinished.
                        atomic_jdump({"log": a.log, "config": name, "method": method, "n": n, "seed": seed,
                                      "error": "CUDA OOM", "kind": kind, "cap_gib": cap, "attempts": own,
                                      "detail": msg[:500], "time": time.strftime("%Y-%m-%d %H:%M:%S")},
                                     path.replace(".json", ".oom.json"))
                        break
        except Exception as e:  # noqa: BLE001
            log(f"  {name}: FAILED {type(e).__name__}: {e}\n{traceback.format_exc()}", fh)
            atomic_jdump({"log": a.log, "config": name, "method": method, "n": n, "seed": seed,
                          "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-3000:]},
                         path.replace(".json", ".failed.json"))
            H.reset()
            gc.collect()
            torch.cuda.empty_cache()
        finally:
            release(lock)
        if a.limit and n_done >= a.limit:
            break
    for (method, n, seed) in todo:                           # aliases whose source was produced in this run
        if alias_of(method, n) is not None and not os.path.exists(os.path.join(odir, f"{config_name(method, n, seed)}.json")):
            copy_alias(odir, method, n, seed, fh)
    log(f"run_streams finished: {n_done} configs evaluated, {n_oom} OOM events; peak GPU "
        f"{torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GiB", fh)


if __name__ == "__main__":
    main()
