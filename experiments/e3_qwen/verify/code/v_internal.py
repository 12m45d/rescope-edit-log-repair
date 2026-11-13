"""Verification of E3, checks 1, 3, 4 (CPU, no driver): every headline number recomputed by my own code from the experiments' per-row
files and gate-level cells, against results.json, paper_numbers.json, table.txt, the sanity run e0 and the testbed's stored metrics.

    python v_internal.py            -> out/internal.json, out/mine_<bb>.json (my numbers per log and method)
"""
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP
import glob
import gzip
import json
import os
import re
import sys

import numpy as np

FINAL = f"{_RP.ROOT}/02_research/experiments/10_final"
V = FINAL + "/e3_qwen/verify"
TESTBED = f"{_RP.ROOT}/02_research/experiments/06_testbed_v3"
BB = {"qwen": {"dir": FINAL + "/e3_qwen/out", "fam": "Q", "cache": "qwen25vl", "later": 500, "dev": "TB-Q100-ae-l2_3",
               "logs": ("TB-Q100-ae-l2_3", "TB-Q200-ae-l2_3", "TB-Q500-ae-l2_3", "TB-Q100-memitw", "TB-Q500-memitw", "TB-Q1000-memitw", "TB-Q2000-memitw")},
      "llava": {"dir": FINAL + "/e3_llava/out", "fam": "L", "cache": "llava", "later": 100, "dev": "TB-L100-ae-l2_3",
                "logs": ("TB-L100-ae-l2_3", "TB-L200-ae-l2_3", "TB-L500-ae-l2_3", "TB-L100-memitw", "TB-L500-memitw", "TB-L1000-memitw")}}
CAL = ("rescope", "rescope_shared", "rescope_null", "rescope_x", "cosine", "cosine_shared", "white_1_shared", "white_0.1_shared", "white_1_x", "retr_sum")
GL = ("rescope_nullW", "rescope_pac")
FIXED = ("deployed", "exact", "cosine_default", "lookup", "cosine_published", "where_logonly")
H3L = ("rescope", "rescope_x", "white_1_shared", "white_1_x", "retr_sum")
INTERNAL = ("TB-Q1000-ae-l2_3", "TB-Q2000-ae-l2_3", "TB-L1000-ae-l2_3")
TOL = 1e-6


def T_of(log):
    return int(re.search(r"TB-[QL](\d+)-", log).group(1))


def draws_of(log):
    return (0,) if T_of(log) >= 1000 else (0, 1, 2)


def jl(p):
    with (gzip.open(p, "rt") if p.endswith(".gz") else open(p)) as f:
        return [json.loads(x) for x in f if x.strip()]


def uf(s):
    return np.array([c == "1" for c in s], dtype=bool)


def pct(a):
    a = np.asarray(list(a), dtype=float)
    return 100.0 * float(a.mean()) if a.size else None


def mean(v):
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None


_ROWS = {}


def rows(bb, log, lab, d, part):
    p = f"{BB[bb]['dir']}/rows/{log}__{lab}__d{'x' if d is None else d}__{part}.jsonl.gz"
    if p not in _ROWS:
        _ROWS[p] = jl(p) if os.path.exists(p) else None
    return _ROWS[p]


def my_crossed_diff(A, B, n_boot=2000, seed=12345):
    """Paired crossed bootstrap (draws x units, resampled independently): A, B (draws, units) per-unit means.  -> diff, lo, hi."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    rng = np.random.default_rng(seed)
    nd, nu = A.shape
    st = np.empty(n_boot)
    for i in range(n_boot):
        d, u = rng.integers(0, nd, nd), rng.integers(0, nu, nu)
        st[i] = 100.0 * (A[np.ix_(d, u)].mean() - B[np.ix_(d, u)].mean())
    return float(100.0 * (A.mean() - B.mean())), float(np.percentile(st, 2.5)), float(np.percentile(st, 97.5))


def reph_grid(bb, log, lab, sub=None):
    """(draws, edits) matrix of (T-Gen exact + I-Gen exact) / 2 per edit; a row without calibration is repeated over the draws."""
    out = []
    for d in draws_of(log):
        r = rows(bb, log, lab, None if lab in FIXED else d, "tf")
        T = T_of(log)
        g = np.zeros((T, 2))
        for x in r:
            if x["kind"] == "edit" and x["metric"] in ("tgen", "igen"):
                g[x["edit"], int(x["metric"] == "igen")] = float(bool(x["exact"]))
        g = g.mean(1)
        out.append(g if sub is None else g[sub])
    return np.array(out)


def behaviour(bb, log, lab, ctx):
    """My behavioural numbers of one method on one log: mean over the behavioural draws of the per-draw rates (percent)."""
    fixed = lab in FIXED
    T, later0 = T_of(log), BB[bb]["later"]
    per = []
    dep = {(r["metric"], r["edit"]): bool(r["exact"]) for r in rows(bb, log, "deployed", None, "tf") if r["kind"] == "edit"}
    for d in ([None] if fixed else draws_of(log)):
        R = rows(bb, log, lab, d, "tf")
        if R is None:
            return None
        e = [r for r in R if r["kind"] == "edit"]
        p = [r for r in R if r["kind"] == "probe"]
        h = [r for r in R if r["kind"] == "hn" and r.get("part") != "gen"]
        o = {"n_edit_rows": len(e), "n_probe": len(p), "n_hn": len(h), "n_near": sum(r["sid"] in ctx["near_sids"] for r in p)}
        for m in ("rel", "tgen", "igen"):
            em = [r for r in e if r["metric"] == m]
            o[m] = pct(bool(r["exact"]) for r in em)
            o[m + "_later"] = pct(bool(r["exact"]) for r in em if r["edit"] >= later0)
            o[m + "_first"] = pct(bool(r["exact"]) for r in em if r["edit"] < 100)
            o["gained_" + m] = pct(bool(r["exact"]) and not dep[(m, r["edit"])] for r in em)
            o["kept_" + m] = pct(bool(r["exact"]) and dep[(m, r["edit"])] for r in em)
        o["reph"] = pct(bool(r["exact"]) for r in e if r["metric"] in ("tgen", "igen"))
        o["reph_later"] = pct(bool(r["exact"]) for r in e if r["metric"] in ("tgen", "igen") and r["edit"] >= later0)
        gate = p[0].get("fired") is not None
        near = [r for r in p if r["sid"] in ctx["near_sids"]]              # MY near-scope flag (question string of an edit of the log)
        o["hd_D1"], o["near_D1"] = pct(bool(r["D1"]) for r in p), pct(bool(r["D1"]) for r in near)
        o["hd_fired"] = pct(bool(r["fired"]) for r in p) if gate else None
        o["near_fired"] = pct(bool(r["fired"]) for r in near) if gate else None
        o["hn_D1"] = pct(bool(r["D1"]) for r in h)
        o["hn_fired"] = pct(bool(r["fired"]) for r in h) if gate else None
        for k in ("HN-I", "HN-Q", "HN-X"):
            o[f"hn_D1_{k}"] = pct(bool(r["D1"]) for r in h if r["hn_type"] == k)
        o["n_near_flag_differs"] = sum((r.get("near_edit") is not None) != (r["sid"] in ctx["near_sids"]) for r in p)
        o["sids_ok"] = ({r["sid"] for r in p} == ctx["hd_sids"]) and ({r["sid"] for r in h} == ctx["hn_sids"]) and len(e) == 3 * T
        o["n_D1_without_firing"] = sum(bool(r["D1"]) and not r["fired"] for r in p) if gate else None
        per.append(o)
    out = {k: mean([o[k] for o in per]) for k in per[0] if not isinstance(per[0][k], bool)}
    out["sids_ok"] = all(o["sids_ok"] for o in per)
    out["by_draw"] = {k: [o[k] for o in per] for k in ("rel", "tgen", "igen", "hd_fired", "hd_D1", "hn_D1")}
    d2 = rows(bb, log, lab, None if fixed else 0, "d2")
    if d2 is not None:
        pc = [r for r in d2 if r.get("pre_correct")]
        out["d2"], out["d2_n_pre_correct"] = pct(bool(r["D2"]) for r in pc), len(pc)
    q = [rows(bb, log, lab, d, "tfq") for d in draws_of(log)] if not fixed else []
    if q and all(x is not None for x in q):                                 # second read-out (R8): far rows restricted to question-disjoint rows
        out["qd_hd_D1"] = mean([pct(bool(r["D1"]) for r in x if r["kind"] == "probe" and r["sid"] in ctx["qd_sids"]) for x in q])
        out["qd_tgen"] = mean([pct(bool(r["exact"]) for r in x if r["kind"] == "edit" and r["metric"] == "tgen") for x in q])
    if lab in H3L:
        hv = []
        for d in draws_of(log):
            c = json.load(open(f"{BB[bb]['dir']}/behave/{log}/{lab}__d{d}__h3.json"))
            if c.get("reachable"):
                x = rows(bb, log, lab, d, "h3")
                t, i = (pct(bool(r["exact"]) for r in x if r["kind"] == "edit" and r["metric"] == m) for m in ("tgen", "igen"))
                hv.append(0.5 * (t + i))
            else:
                hv.append(0.0)
        out["h3_exact"], out["h3_exact_by_draw"] = float(np.mean(hv)), hv
    return out


def gate(bb, log, lab, ctx):
    p = f"{BB[bb]['dir']}/gate/{log}/{lab}__reg.json"
    if not os.path.exists(p):
        return None
    f = json.load(open(p))
    cs = [f["cells"][k] for k in sorted(f["cells"], key=int)]
    if "not_defined" in cs[0]:
        return None
    T, later0 = T_of(log), BB[bb]["later"]
    per = []
    for c in cs:
        if bb == "qwen":
            hd = uf(c["hd_flags"])
            lit = c.get("literal") or {}
            hdl = uf(lit["hd_flags"]) if lit.get("built") else (hd if lit else None)
            cov = {k: uf(c["per_edit"][k]) for k in ("rel", "tgen", "igen")}
            hn = uf(c["hn_flags"])
        else:
            pf = uf(c["probe_fired"])
            hd = pf[ctx["hd_mask_all"]]
            lit = c.get("literal") or {}
            hdl = uf(lit["probe_fired"])[ctx["hd_mask_all"]] if lit.get("built") else (hd if lit else None)
            cov = {k: uf(c["edit"][k]["own_top"]) for k in ("rel", "tgen", "igen")}
            hn = uf(c["hn_fired"])
        o = {"fire": pct(hd), "fire_near": pct(hd[ctx["near_mask"]]), "fire_far": pct(hd[~ctx["near_mask"]]), "fire_lit": None if hdl is None else pct(hdl),
             "fire_qd": pct(hd[ctx["qd_mask"]]), "cov_t": pct(cov["tgen"]), "cov_i": pct(cov["igen"]), "cov_rel": pct(cov["rel"]),
             "cov_t_later": pct(cov["tgen"][later0:]) if T > later0 else None, "cov_i_later": pct(cov["igen"][later0:]) if T > later0 else None,
             "capped": c.get("n_capped"), "hn_all": pct(hn), "n_hd": int(hd.size)}
        o["cov_reph_later"] = None if o["cov_t_later"] is None else 0.5 * (o["cov_t_later"] + o["cov_i_later"])
        for k in ("HN-I", "HN-Q", "HN-X"):
            m = ctx["hn_kind"] == k
            o["hn_" + k] = pct(hn[m]) if m.any() else None
        if ctx.get("hn_unseen") is not None:
            for k in ("HN-I", "HN-Q", "HN-X"):
                m = (ctx["hn_kind"] == k) & ctx["hn_unseen"]
                o["hnu_" + k], o["hnu_n_" + k] = (pct(hn[m]) if m.any() else None), int(m.sum())
        b = c.get("BEN-T") or {}
        o["ben_far"] = 100.0 * b["far"]["n_fired"] / b["far"]["n"] if "far" in b else None
        mm = c.get("MME-T") or {}
        o["mme"] = 100.0 * mm["n_fired"] / mm["n"] if "n_fired" in mm else None
        h = c.get("h3")
        if h is not None and "not_defined" not in h:
            if h["reachable"]:
                o["h3cov"] = 0.5 * (pct(uf(h["per_edit"]["tgen"])) + pct(uf(h["per_edit"]["igen"])))
                o["h3cov_cell"] = h["rephrase"]["mean"]
                o["h3_n_at_cap"] = h.get("n_edits_at_cap")
            else:
                o["h3cov"], o["h3cov_cell"], o["h3_n_at_cap"] = 0.0, 0.0, h.get("n_edits_at_cap")
            o["h3_reachable"] = bool(h["reachable"])
        per.append(o)
    out = {k: mean([o.get(k) for o in per]) for k in sorted({k for o in per for k in o}) if k != "h3_reachable"}
    out["n_draws"] = len(per)
    out["fire_by_draw"] = [o["fire"] for o in per]
    out["mme_by_draw"] = [o["mme"] for o in per]
    out["capped_by_draw"] = [o["capped"] for o in per]
    out["n_draws_above_bound"] = sum(o["fire"] > 100 * 5 / 101 for o in per)
    out["h3_n_unreachable"] = sum(1 for o in per if o.get("h3_reachable") is False)
    out["digest_by_draw"] = [c.get("digest") for c in cs]
    out["h3cov_flags_vs_cell_max_abs"] = max([abs(o["h3cov"] - o["h3cov_cell"]) for o in per if "h3cov" in o], default=None)
    return out


def context(bb, log):
    """Pools, near-scope flags and counts of a log, computed HERE from the raw files of the key cache and from the experiment's context."""
    B = BB[bb]
    T, fam = T_of(log), B["fam"]
    ix = INDEX[bb]
    nq = lambda s: (s or "").strip().lower()                                # noqa: E731
    eq = {nq(r["prompt"]) for r in ix if r["family"] == fam and r["kind"] == "rel" and r["edit"] < T}
    long_T = max(T_of(g) for g in B["logs"])
    allq = {nq(r["prompt"]) for r in ix if r["family"] == fam and r["kind"] in ("rel", "tgen") and r["edit"] < long_T}
    pr = [r for r in ix if r["family"] == fam and r["kind"] == "probe"]
    hd = [r for r in pr if r["pool"] == "heldout" and r["disjoint"] and r["y0_valid"]]
    hn = [r for r in ix if r["kind"] in ("hn_i", "hn_q", "hn_x") and r.get("family", fam) == fam and r["edit"] < T and (r.get("edit_b") is None or r["edit_b"] < T) and r["y0_valid"]]
    c = {"hd_sids": {r["lib_sid"] for r in hd}, "near_sids": {r["lib_sid"] for r in hd if nq(r["prompt"]) in eq}, "qd_sids": {r["lib_sid"] for r in hd if nq(r["prompt"]) not in allq},
         "hn_sids": {r.get("lib_sid") or r["sid"] for r in hn}, "n_probe_rows": len(pr), "n_probe_valid": sum(bool(r["y0_valid"]) for r in pr),
         "n_hd_all": sum(1 for r in pr if r["pool"] == "heldout" and r["disjoint"]), "n_hd_valid": len(hd), "n_repair_valid": sum(1 for r in pr if r["pool"] == "repair" and r["y0_valid"])}
    if bb == "qwen":
        x = json.load(open(f"{B['dir']}/gate/{log}/_context.json"))
        sid = [s.split(":", 1)[1] for s in x["hd"]["csid"]]
        c["near_mask"] = np.array([s in c["near_sids"] for s in sid])
        c["qd_mask"] = np.array([s in c["qd_sids"] for s in sid])
        c["hn_kind"] = np.array(x["hn"]["hn_type"])
        c["ctx_near_differs"] = int((c["near_mask"] != uf(x["hd"]["near"])).sum())
        c["ctx_qd_differs"] = int((c["qd_mask"] != uf(x["hd"]["qd"])).sum())
        c["ctx_hd_ok"] = set(sid) == c["hd_sids"] and set(x["hn"]["csid"]) == c["hn_sids"]
        c["later_ok"] = x["later_edits"] == list(range(B["later"], T))
        hp = f"{B['dir']}/behave/{log}/rescope_x__d0__h3.json"
        ufs = set((json.load(open(hp)).get("unseen_foreign") or {}).get("csids") or []) if os.path.exists(hp) else set()
        c["hn_unseen"] = np.array([s in ufs for s in x["hn"]["csid"]]) if ufs else None
    else:
        x = json.load(open(f"{B['dir']}/gate/{log}/_meta.json"))
        sid = [s.split(":", 1)[1] for s in x["probe"]["sid"]]
        hdm = np.array([s in c["hd_sids"] for s in sid])
        c["hd_mask_all"] = hdm
        c["near_mask"] = np.array([s in c["near_sids"] for s in sid])[hdm]
        c["qd_mask"] = np.array([s in c["qd_sids"] for s in sid])[hdm]
        c["hn_kind"] = np.array(x["hn"]["kind"])
        c["ctx_near_differs"] = int((c["near_mask"] != uf(x["probe"]["masks"]["hd_near"])[hdm]).sum())
        c["ctx_qd_differs"] = int((c["qd_mask"] != uf(x["probe"]["masks"]["hd_qd"])[hdm]).sum())
        c["ctx_hd_ok"] = bool((hdm == uf(x["probe"]["masks"]["hd"])).all()) and set(x["hn"]["sid"]) == c["hn_sids"]
        c["later_ok"] = x["later_edits"] == list(range(B["later"], T))
        c["hn_unseen"] = uf(x["hn"]["unseen_foreign"]) if x["hn"].get("unseen_foreign") else None
        c["n_invalid_reported"] = x["n_probe_invalid_excluded"]
        c["lp_shift_rows"] = x["lp_shift_rows"]
    return c


def stored(bb, log, mine):
    """The deployed log: the experiment's B = 1 rows against the testbed's own stored flags."""
    pe = jl(f"{TESTBED}/{log}/per_edit_metrics.jsonl")
    pr = {(r["probe_idx"], r["ptype"]): r for r in jl(f"{TESTBED}/{log}/probe_records_v2.jsonl")}
    met = json.load(open(f"{TESTBED}/{log}/metrics.json"))
    R = rows(bb, log, "deployed", None, "tf")
    o = {"non_oracle": [i for i, r in enumerate(pe) if not r.get("oracle", True)]}
    for m in ("rel", "tgen", "igen"):
        e = {r["edit"]: bool(r["exact"]) for r in R if r["kind"] == "edit" and r["metric"] == m}
        s = [bool(r[f"post_{m}_exact"]) for r in pe]
        o[m] = {"B1_rows": pct(e[i] for i in range(len(pe))), "stored_per_edit": pct(s), "stored_metrics_json": met.get({"rel": "Rel_exact", "tgen": "TGen_exact", "igen": "IGen_exact"}[m]),
                "n_flags_differ": sum(e[i] != s[i] for i in range(len(pe)))}
    p = [r for r in R if r["kind"] == "probe"]
    o["probe_D1"] = {"B1_rows": pct(bool(r["D1"]) for r in p), "stored_is_complaint_D1": pct(bool(pr[(r["probe_idx"], r["ptype"])]["is_complaint_D1"]) for r in p),
                     "n_flags_differ": sum(bool(r["D1"]) != bool(pr[(r["probe_idx"], r["ptype"])]["is_complaint_D1"]) for r in p), "n": len(p)}
    return o


def margins(log):
    import torch
    fac = torch.load(f"{TESTBED}/{log}/edit_factors.pt", map_location="cpu")
    W, K = fac["W"].double(), fac["K1_solver"].double()
    s = (W * K).sum(1)
    R = (K @ W.T) / s[None, :]                                              # R[j, i] = r_i(k_j)
    R.fill_diagonal_(-float("inf"))
    m = (1 - R.max(0).values).numpy()
    return {"min": float(m.min()), "n_nonpositive": int((m <= 0).sum()), "n_positive": int((m > 0).sum()), "median": float(np.median(m))}


INDEX = {}


def main():
    out = {"backbones": {}}
    for bb, B in BB.items():
        INDEX[bb] = jl(f"{FINAL}/cache/{B['cache']}/index.jsonl") + [dict(r, family=B["fam"]) for r in []]
        for r in INDEX[bb]:
            r.setdefault("family", B["fam"])
        res = json.load(open(B["dir"] + "/results.json"))
        pn = json.load(open(B["dir"] + "/paper_numbers.json"))
        tab = open(B["dir"] + "/table.txt").read()
        mine, O = {}, {"logs": {}}
        for log in B["logs"]:
            ctx = context(bb, log)
            mine[log] = {"_ctx": {k: v for k, v in ctx.items() if isinstance(v, (int, bool, dict))}, "_stored": None, "_margin": margins(log), "beh": {}, "gate": {}}
            for lab in FIXED + CAL:
                mine[log]["beh"][lab] = behaviour(bb, log, lab, ctx)
            for lab in CAL + GL + ("exact", "cosine_default", "lookup"):
                mine[log]["gate"][lab] = gate(bb, log, lab, ctx)
            mine[log]["_stored"] = stored(bb, log, mine)
            print(bb, log, "done", flush=True)
        json.dump(mine, open(f"{V}/out/mine_{bb}.json", "w"))
        # ---------------- my numbers against results.json
        ix = {}
        for r in res:
            ix.setdefault((r["log"], r["method"], r["metric"], r.get("variant") or "", r.get("draw")), []).append(r)

        def R(log, lab, metric, var="", draw=None):
            v = ix.get((log, lab, metric, var, draw))
            return None if not v else v[0]["value"]
        cmp, bad = 0, []

        def chk(log, lab, name, m, e):
            nonlocal cmp
            if m is None and e is None:
                return
            cmp += 1
            if m is None or e is None or abs(m - e) > TOL:
                bad.append({"log": log, "method": lab, "quantity": name, "mine": m, "experiment": e})
        QB = {"rel": "edit.all.rel.exact", "tgen": "edit.all.tgen.exact", "igen": "edit.all.igen.exact", "reph": "edit.all.rephrase.exact", "rel_later": "edit.later.rel.exact",
              "tgen_later": "edit.later.tgen.exact", "igen_later": "edit.later.igen.exact", "rel_first": "edit.first100.rel.exact", "tgen_first": "edit.first100.tgen.exact",
              "igen_first": "edit.first100.igen.exact", "hd_fired": "probe.fired", "hd_D1": "probe.D1", "near_fired": "probe.near.fired", "near_D1": "probe.near.D1", "hn_D1": "hn.all.D1",
              "hn_fired": "hn.all.fired", "d2": "d2.rate", "gained_tgen": "kg.all.tgen.gained", "kept_tgen": "kg.all.tgen.kept", "h3_exact": "h3.rephrase.exact", "qd_hd_D1": "qd.probe.D1",
              "qd_tgen": "qd.edit.all.tgen.exact"}
        QG = {"fire": "gate.fire.hd", "fire_near": "gate.fire.hd.near", "fire_lit": "gate.fire.hd.literal", "fire_qd": "gate.fire.hd.qd", "cov_t": "gate.cov.tgen", "cov_i": "gate.cov.igen",
              "capped": "gate.n_capped", "hn_all": "gate.hn.all.fired", "ben_far": "gate.BEN-T.far.fired", "mme": "gate.MME-T.fired", "h3cov": "gate.h3.cov", "hnu_HN-I": "gate.nsu.HN-I.fired"}
        LB = {"rel": "edit.rel.exact", "tgen": "edit.tgen.exact", "igen": "edit.igen.exact", "reph": "edit.reph.exact", "rel_later": "edit.rel.exact.later", "tgen_later": "edit.tgen.exact.later",
              "igen_later": "edit.igen.exact.later", "reph_later": "edit.reph.exact.later", "rel_first": "edit.rel.exact.first100", "tgen_first": "edit.tgen.exact.first100",
              "igen_first": "edit.igen.exact.first100", "hd_fired": "probe.fired", "hd_D1": "probe.D1", "near_fired": "probe.fired.near", "near_D1": "probe.D1.near", "hn_D1": "hn.all.D1",
              "hn_fired": "hn.all.fired", "gained_tgen": "kg.tgen.gained", "kept_tgen": "kg.tgen.kept", "qd_hd_D1": "probe.D1.qd"}
        LG = {"fire": "fire.hd", "fire_near": "fire.hd_near", "fire_far": "fire.hd_far", "fire_lit": "fire.hd.literal", "fire_qd": "fire.hd_qd", "cov_t": "cov.tgen", "cov_i": "cov.igen",
              "cov_t_later": "cov.tgen.later", "cov_i_later": "cov.igen.later", "capped": "n_capped", "hn_HN-I": "hn.HN-I.fired", "hn_HN-Q": "hn.HN-Q.fired", "hn_HN-X": "hn.HN-X.fired",
              "ben_far": "BEN-T.far.fired", "mme": "MME-T.fired", "h3cov": "h3.cov", "hnu_HN-I": "hnu.HN-I.fired"}
        for log in B["logs"]:
            for lab, m in mine[log]["beh"].items():
                if m is None:
                    bad.append({"log": log, "method": lab, "quantity": "behaviour rows", "mine": None, "experiment": "missing"})
                    continue
                fixed = lab in FIXED
                for k, name in (QB if bb == "qwen" else LB).items():
                    if k not in m or (k.endswith("_first") and bb == "llava" and T_of(log) == 100):
                        continue
                    if bb == "qwen":
                        e = R(log, lab, name, "qdisjoint" if k.startswith("qd_") else "", 0 if (k == "d2" and not fixed) else None)
                    elif k.startswith("qd_"):
                        e = R(log, lab, "b." + name + "@x", "qd") if k != "qd_hd_D1" else R(log, lab, "b.probe.D1.qd@x", "qd")
                    else:
                        e = R(log, lab, "b." + name, "reg") if fixed else R(log, lab, "b." + name + "@x", "reg")
                    if bb == "qwen" and k == "qd_hd_D1":
                        e = R(log, lab, "qd.probe.D1", "qdisjoint")
                    if bb == "qwen" and k == "qd_tgen":
                        e = R(log, lab, "qd.edit.all.tgen.exact", "qdisjoint")
                    if bb == "llava" and k == "qd_tgen":
                        e = R(log, lab, "b.edit.tgen.exact@x", "qd")
                    chk(log, lab, k, m[k], e)
                if bb == "llava":
                    if "d2" in m and R(log, lab, "b.D2", "", 0) is not None:
                        chk(log, lab, "d2", m["d2"], R(log, lab, "b.D2", "", 0))
                    if "h3_exact" in m:
                        for d, v in zip(draws_of(log), m["h3_exact_by_draw"]):
                            e = R(log, lab, "b.h3.exact", "", d)
                            if e is not None:
                                chk(log, lab, f"h3_exact_d{d}", v, e)
            for lab, g in mine[log]["gate"].items():
                if g is None:
                    continue
                fixed = g["n_draws"] == 1
                for k, name in (QG if bb == "qwen" else LG).items():
                    if g.get(k) is None:
                        continue
                    e = R(log, lab, name, "") if bb == "qwen" else (R(log, lab, "g." + name, "reg") if fixed else R(log, lab, "g." + name + "@mean", "reg"))
                    if e is None and k in ("fire_lit", "hnu_HN-I", "h3cov", "capped", "fire_qd"):
                        continue
                    chk(log, lab, "gate." + k, g[k], e)
        O["results_json"] = {"n_cells_compared": cmp, "n_disagree": len(bad), "disagreements": bad[:40], "disagreeing_quantities": sorted({x["quantity"] for x in bad})}
        # ---------------- my numbers against paper_numbers.json
        if bb == "qwen":
            ROMAN = {100: "C", 200: "CC", 500: "D", 1000: "M", 2000: "MM"}
            MC = {"deployed": "Deployed", "where_logonly": "Where", "rescope": "Rescope", "rescope_shared": "RescopeShared", "rescope_null": "RescopeNull", "rescope_x": "RescopeX",
                  "rescope_nullW": "RescopeNullW", "rescope_pac": "RescopePac", "cosine": "Cosine", "cosine_shared": "CosineShared", "white_1_shared": "WhiteOneShared",
                  "white_0.1_shared": "WhiteTenthShared", "white_1_x": "WhiteOneX", "retr_sum": "RetrSum", "lookup": "Lookup", "cosine_default": "CosineDefault",
                  "cosine_published": "CosinePublished", "exact": "Exact"}
            PB = {"RelExact": "rel", "TGenExact": "tgen", "IGenExact": "igen", "RelExactLater": "rel_later", "TGenExactLater": "tgen_later", "IGenExactLater": "igen_later",
                  "RelExactFirst": "rel_first", "TGenExactFirst": "tgen_first", "IGenExactFirst": "igen_first", "HdFired": "hd_fired", "HdDone": "hd_D1", "HdNearFired": "near_fired",
                  "HdNearDone": "near_D1", "HnDone": "hn_D1", "HnFired": "hn_fired", "DtwoRate": "d2", "HthreeExact": "h3_exact"}
            PG = {"GateHdFired": "fire", "GateHdFiredLiteral": "fire_lit", "GateHdNearFired": "fire_near", "GateCovTGen": "cov_t", "GateCovIGen": "cov_i", "GateCapped": "capped",
                  "GateBenTFar": "ben_far", "GateMmeT": "mme", "GateHnFired": "hn_all", "HthreeCov": "h3cov"}
            name = lambda log, lab, nm: "EthreeQ" + ("Ae" if "-ae-" in log else "Me") + ROMAN[T_of(log)] + MC[lab] + nm        # noqa: E731
            rnd = lambda v: round(float(v), 2)                               # noqa: E731
        else:
            TW = {100: "Hundred", 200: "TwoHundred", 500: "FiveHundred", 1000: "Thousand"}
            dig = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}

            def camel(s):
                s = re.sub(r"[^A-Za-z0-9]+", " ", str(s))
                return "".join(w[:1].upper() + w[1:] for w in "".join(dig.get(ch, ch) if ch.isdigit() else ch for ch in s).split())
            PB = {"Rel": "rel", "TGen": "tgen", "IGen": "igen", "RelLater": "rel_later", "TGenLater": "tgen_later", "IGenLater": "igen_later", "RephLater": "reph_later", "Reph": "reph",
                  "DOne": "hd_D1", "DOneNear": "near_D1", "HnDOne": "hn_D1"}
            PG = {"CovT": "cov_t", "CovI": "cov_i", "CovTLater": "cov_t_later", "CovILater": "cov_i_later", "Fire": "fire", "FireNear": "fire_near", "FireFar": "fire_far",
                  "FireLiteral": "fire_lit", "HnFireI": "hn_HN-I", "HnFireQ": "hn_HN-Q", "HnFireX": "hn_HN-X", "BenT": "ben_far", "MmeT": "mme", "Capped": "capped", "HthreeCov": "h3cov"}
            name = lambda log, lab, nm: "EthreeL" + ("Ae" if "-ae-" in log else "Me") + TW[T_of(log)] + camel(lab) + nm            # noqa: E731
            rnd = lambda v: float(v)                                        # noqa: E731
        seen, pbad = set(), []
        for log in B["logs"]:
            for lev, P in (("beh", PB), ("gate", PG)):
                for lab, m in mine[log][lev].items():
                    if m is None:
                        continue
                    for nm, k in P.items():
                        key = name(log, lab, nm)
                        if key in pn:
                            seen.add(key)
                            if m.get(k) is None or abs(m[k] - pn[key]) > (0.005 + 1e-9 if bb == "qwen" else 1e-6):
                                pbad.append({"name": key, "mine": m.get(k), "paper_number": pn[key]})
        rest = sorted(set(pn) - seen)
        # factor-level names: margins recomputed here; the others are listed as not recomputed
        fac_ok, fac_bad = 0, []
        for log in B["logs"]:
            pre = name(log, "rescope", "").replace("Rescope", "")
            for nm, k in (("MarginMin", "min"), ("MarginPositive", "n_positive")):
                if pre + nm in pn:
                    seen.add(pre + nm)
                    if abs(round(mine[log]["_margin"][k], 4) - round(pn[pre + nm], 4)) > 1e-4:
                        fac_bad.append({"name": pre + nm, "mine": mine[log]["_margin"][k], "paper_number": pn[pre + nm]})
                    else:
                        fac_ok += 1
            if pre + "NonOracle" in pn:
                seen.add(pre + "NonOracle")
                if len(mine[log]["_stored"]["non_oracle"]) != pn[pre + "NonOracle"]:
                    fac_bad.append({"name": pre + "NonOracle", "mine": len(mine[log]["_stored"]["non_oracle"]), "paper_number": pn[pre + "NonOracle"]})
                else:
                    fac_ok += 1
        rest = sorted(set(pn) - seen)
        O["paper_numbers"] = {"n_names": len(pn), "n_recomputed_here": len(seen), "n_disagree": len(pbad) + len(fac_bad), "disagreements": (pbad + fac_bad)[:30],
                              "n_not_recomputed": len(rest), "not_recomputed_kinds": sorted({re.sub(r"^Ethree[QL](Ae|Me)(C|CC|D|M|MM|Hundred|TwoHundred|FiveHundred|Thousand)", "", k) for k in rest})[:40]}
        # ---------------- discipline: cells, settings, digests, failures, internal logs
        D = {"n_cells": 0, "bad_cells": [], "n_failed_or_skipped_files": 0, "jobs_per_log": {}, "digest_gate_vs_behaviour_differ": [], "internal_logs_mentioned": {}}
        for log in B["logs"]:
            bd = f"{B['dir']}/behave/{log}"
            D["n_failed_or_skipped_files"] += len(glob.glob(bd + "/*FAILED*")) + len(glob.glob(bd + "/*SKIPPED*"))
            jobs = [json.load(open(p)) for p in sorted(glob.glob(bd + "/_job_*.json"))]
            D["jobs_per_log"][log] = {"n_jobs": len(jobs), "hook_mismatch": [j["assert_clean"].get("hook_mismatch") if isinstance(j.get("assert_clean"), dict) else j.get("assert_clean") for j in jobs],
                                      "blocks": [j.get("blocks") for j in jobs]}
            for lab in CAL:
                for d in draws_of(log):
                    for part in ("tf", "tfq"):
                        p = f"{bd}/{lab}__d{d}__{part}.json"
                        D["n_cells"] += 1
                        if not os.path.exists(p):
                            D["bad_cells"].append([log, lab, d, part, "missing"])
                            continue
                        c = json.load(open(p))
                        base = {"rescope_x": "x=external", "cosine_shared": "calibration=shared"}.get(lab, "")
                        want = base if part == "tf" else ",".join(x for x in ((base.split(",")[0] if lab == "cosine_shared" else ""), "mask=qdisjoint", (base if lab == "rescope_x" else "")) if x)
                        okv = (c.get("variant") or "") == want
                        if not (c.get("alpha") == 0.05 and c.get("n") == 200 and c.get("draw") == d and c.get("hook_mismatch") == 0 and c.get("memo_conflicts") == 0 and okv
                                and c.get("code", {}).get("fx") == "6a71f0d2cd71"):
                            D["bad_cells"].append([log, lab, d, part, {k: c.get(k) for k in ("alpha", "n", "draw", "variant", "hook_mismatch", "memo_conflicts", "code")}])
                        if part == "tf":
                            g = mine[log]["gate"].get(lab)
                            if g and g["digest_by_draw"][d] != c.get("digest"):
                                D["digest_gate_vs_behaviour_differ"].append([log, lab, d])
            for lab in FIXED:
                D["n_cells"] += 1
                if not os.path.exists(f"{bd}/{lab}__dx__tf.json"):
                    D["bad_cells"].append([log, lab, None, "tf", "missing"])
        blob = json.dumps(res) + json.dumps(pn) + tab
        D["internal_logs_mentioned"] = {g: blob.count(g) for g in INTERNAL}
        D["out_internal_dir_exists"] = os.path.exists(B["dir"] + "/internal")
        D["logs_in_results"] = sorted({r["log"] for r in res})
        O["discipline"] = D
        # ---------------- counts and pools
        O["counts"] = {log: dict({k: v for k, v in mine[log]["_ctx"].items()},
                                 n_edit_rows=mine[log]["beh"]["rescope"]["n_edit_rows"], n_probe_rows_in_files=mine[log]["beh"]["rescope"]["n_probe"], n_hn_rows_in_files=mine[log]["beh"]["rescope"]["n_hn"],
                                 all_row_files_have_my_sid_sets=all(m["sids_ok"] for m in mine[log]["beh"].values() if m),
                                 n_near_rows=mine[log]["beh"]["rescope"]["n_near"], near_flags_differing=max(m["n_near_flag_differs"] for m in mine[log]["beh"].values() if m),
                                 max_D1_without_firing=max((m["n_D1_without_firing"] or 0) for lab, m in mine[log]["beh"].items() if m and lab in CAL)) for log in B["logs"]}
        O["deployed_vs_stored"] = {log: mine[log]["_stored"] for log in B["logs"]}
        # ---------------- development log against the sanity run e0
        e0 = json.load(open(FINAL + "/e0_sanity/out/results.json"))["logs"][B["dev"]]
        MAP = {"M1": "rescope", "M2": "rescope_shared", "M3": "cosine", "M4": "cosine_shared", "M5": "cosine_default", "M6": "exact", "deployed": "deployed"}
        n, worst = 0, 0.0
        for m0, lab in MAP.items():
            g0 = e0["gate"]["methods"].get(m0)
            g = mine[B["dev"]]["gate"].get(lab)
            if g0 and g:
                for path, k in ((("probe", "heldout_disjoint", "fired"), "fire"), (("edit", "tgen", "own_top"), "cov_t"), (("edit", "igen", "own_top"), "cov_i"),
                                (("hn", "HN-I", "fired"), "hn_HN-I"), (("hn", "HN-Q", "fired"), "hn_HN-Q"), (("hn", "HN-X", "fired"), "hn_HN-X")):
                    v = g0["tf"]
                    for kk in path:
                        v = v[kk]
                    v = v["mean"] if isinstance(v, dict) else v
                    n += 1
                    worst = max(worst, abs(v - g[k]))
            b0 = e0["behaviour"]["methods"].get(m0)
            b = mine[B["dev"]]["beh"].get(lab)
            if b0 and b:
                per = [b0["single"]] if "single" in b0 else [b0["draws"][str(d)] for d in (0, 1, 2)]
                for path, k in ((("edit", "rel", "exact"), "rel"), (("edit", "tgen", "exact"), "tgen"), (("edit", "igen", "exact"), "igen"), (("probe", "heldout_disjoint", "D1"), "hd_D1")):
                    for i, v in enumerate(per):
                        for kk in path:
                            v = v[kk]
                        n += 1
                        worst = max(worst, abs(v - b["by_draw"][k][i]))
        O["dev_vs_e0"] = {"n_cells": n, "largest_abs_diff": worst}
        out["backbones"][bb] = O
        out.setdefault("mine_files", []).append(f"{V}/out/mine_{bb}.json")
    json.dump(out, open(V + "/out/internal.json", "w"), indent=1)
    for bb, O in out["backbones"].items():
        print(bb, "results.json", {k: v for k, v in O["results_json"].items() if k != "disagreements"}, "| paper", {k: v for k, v in O["paper_numbers"].items() if k != "disagreements"})
        print("  disagreements", json.dumps(O["results_json"]["disagreements"][:6])[:900], json.dumps(O["paper_numbers"]["disagreements"][:6])[:600])
        print("  discipline", json.dumps({k: v for k, v in O["discipline"].items() if k != "jobs_per_log"})[:900])
        print("  dev vs e0", O["dev_vs_e0"])


if __name__ == "__main__":
    main()
