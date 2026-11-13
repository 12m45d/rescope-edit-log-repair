"""E2 GPU job of ONE log (one model handle, one session): blocks that write cells (e2c.write_cell), resumable.

    gq run --mem 10500 --prio 3 -- python job.py TB-Q100-ae-l2_3 --blocks capture,base,cw,d ...

Blocks
  reedit_build  part d: REEDIT-B at n = 10 (seed 0) built ONCE through rescope.reedit, weight saved (fresh process: z* optimisation)
  capture       all-positions capture (one truncated forward per sequence): for every valid probe row, every edit sequence and every
                valid hard negative the vector over edits of the largest normalised coordinate over ALL positions of the teacher-forced
                sequence (= what the hook of ReScope-W scores) and the vector at the last prompt position -> out/capture/<log>.pt
  base          deployed log, pre-edit model, ReScope-W with all thresholds at -inf (round-off floor)
  cw            part c': ReScope-W with two-block thresholds from the captured vectors: gate level 20 draws, behaviour draws 0 - 2
  d             part d: linear rows (COMP-B ALL rebuilt, RB-UNION n = 10 / ALL, RB-TOT n = 10, REEDIT-B n = 10 rebuilt, REEDIT-B ALL saved weight)
  b1 / b2       part b: ReScope-W + least repair of the nested complaint sets (b1: n in {10, 100, ALL} seed 0; b2: the other cells) + blocklist
  c             part c: ReScope on label-free evidence, behaviour
  gen           free generation of the edit prompts at draw 0 / seed 0 for the main-table rows of this experiment
  e             running example (Qwen development log only)
Every method comes from the driver (fx.methods) or is a library object wrapped by fx.methods.wrap; every forward runs through
fx.behave (the library's Evaluator).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2c  # noqa: E402
from e2c import fx  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("log")
ap.add_argument("--blocks", default="capture,base,cw,d,b1,c,b2,gen,e")
ap.add_argument("--smoke", type=int, default=0, help="keep every k-th sequence (mechanics test; use with E2_OUT)")
args = ap.parse_args()
NAME, SM = args.log, int(args.smoke)
BLOCKS = [b for b in args.blocks.split(",") if b]
os.makedirs(e2c.LOGD, exist_ok=True)
FH = open(os.path.join(e2c.LOGD if not SM else e2c.OUT, f"job_{e2c.SHORT[NAME]}.log"), "a")


def say(msg):
    e2c.log(f"{e2c.SHORT[NAME]} {msg}", FH)


L = fx.logs.open(NAME)
H = fx.logs.load_model(NAME)
H.solve_dev = torch.device("cpu")              # P12: COMP and REEDIT solves on the CPU (fp64) for both logs
ses = fx.behave.session(L, H)
import rescope  # noqa: E402
from rescope import calibrate_two_block  # noqa: E402
from rescope.repair import ABS_MARGIN, MARGIN, _above, _guard_fires, _guards  # noqa: E402

hz = rescope.harness()
say(f"loaded {NAME}: blocks {BLOCKS} smoke {SM} cap {os.environ.get('GQ_MEM')} fx {fx.code_fingerprint()}")


def sub(rows):
    return list(rows)[::SM] if SM else list(rows)


EDIT_ROWS = sub([j for m in ("rel", "tgen", "igen") for j in L.edit_rows[m]])
HD_ROWS = sub(L.pool("heldout_disjoint"))
HO_ROWS = sub(L.pool("heldout"))
HN_ROWS = sub(L.rows(("hn_i", "hn_q", "hn_x"), valid=True))
D2_ROWS = sub(L.rows("probe", pool="heldout_disjoint", where=lambda e: bool(e.get("pre_correct_long"))))
assert set(L.pool("heldout_strict")) <= set(L.pool("heldout_disjoint")) <= set(L.pool("heldout"))
N_ALL = len(fx.draws.complaints(L, "ALL", 0))
N_REPAIR = len(L.pool("repair"))
DEP, PRE = fx.methods.deployed(L), fx.methods.preedit(L)
_dep = {}


def dep_flags():
    """Flags of the deployed log on every sequence this job evaluates (memoised by the driver), for the counts against it."""
    if not _dep:
        for r in ses.tf(DEP, sorted(set(EDIT_ROWS + HO_ROWS + HD_ROWS + HN_ROWS))):
            _dep[r["csid"]] = bool(r["exact"])
    return _dep


def annotate(rows):
    d = dep_flags()
    for r in rows:
        if r.get("csid") in d and "exact" in r and "line" not in r:
            if r["kind"] == "edit":
                r["exact_dep"] = d[r["csid"]]
            else:
                r["D1_dep"] = not d[r["csid"]]
    return rows


# ----------------------------------------------------------------------------- capture (all positions)
CAP_PATH = os.path.join(e2c.OUT, "capture", f"{NAME}.pt")


def where_base():
    return fx.methods.base_gate(L, "ncoord", "all", "all", "lin", True, True)[0]


def capture():
    if os.path.exists(CAP_PATH):
        return torch.load(CAP_PATH)
    rows = sorted(set((L.pool("repair") if SM else L.rows("probe", valid=True)) + EDIT_ROWS + HN_ROWS + HO_ROWS + HD_ROWS))
    S = where_base()
    assert S.D == S.T == L.T
    Sg = fx.methods._slim(S).to(H.dev)
    Rmax, Rlast, npos, lps = [], [], [], []
    t0 = time.time()
    with torch.no_grad(), ses.ev.hz.HE.vision_cache(H):
        for n, j in enumerate(rows):
            ses.ev.trim_vcache()
            x, p = ses.ev.capture(ses.item(j))
            keys = Sg.eligible_keys(x, torch.tensor([p]))               # role 'all': every position of the teacher-forced sequence
            assert keys.shape[0] == x.shape[0]
            r = Sg.scores(keys)                                         # (positions, T): one query, scored as the hook scores it
            Rmax.append(r.max(0).values.cpu())                          # = ScopedLog.query_max(keys): what least_repair / calibration use
            Rlast.append(r[p].cpu())                                    # the last prompt position
            npos.append(int(x.shape[0]))
            lps.append(int(p))
            if (n + 1) % 500 == 0:
                say(f"capture {n + 1}/{len(rows)} ({time.time() - t0:.0f}s)")
    del Sg
    torch.cuda.empty_cache()
    cap = {"log": NAME, "rows": rows, "sids": [L.sid(j) for j in rows], "Rmax": torch.stack(Rmax), "Rlast": torch.stack(Rlast), "n_pos": npos, "lp": lps,
           "what": "Rmax[r, i] = largest normalised coordinate of edit i over ALL positions of the teacher-forced sequence of row r (ScopedLog.query_max on "
                   "the model's device, role 'all'); Rlast: at the last prompt position", "seconds": time.time() - t0, "code": fx.code_fingerprint()}
    os.makedirs(os.path.dirname(CAP_PATH), exist_ok=True)
    torch.save(cap, CAP_PATH + ".tmp")
    os.replace(CAP_PATH + ".tmp", CAP_PATH)
    say(f"capture done: {len(rows)} sequences in {time.time() - t0:.0f}s")
    return cap


_CAP = {}


def cap_ix():
    if not _CAP:
        c = capture()
        _CAP.update(c=c, ix={j: i for i, j in enumerate(c["rows"])}, by_sid={s: i for i, s in enumerate(c["sids"])})
    return _CAP


def vec(rows):
    c = cap_ix()
    return c["c"]["Rmax"][[c["ix"][int(j)] for j in rows]]


def hook_vs_vectors(theta, rows):
    """Gate-level decision from the captured vectors against the hook's flags on every teacher-forced row of a cell."""
    c = cap_ix()
    th = theta.detach().cpu().double()
    out = {"n_rows": 0, "n_fired_differs": 0, "n_own_last_differs": 0, "examples": []}
    for r in rows:
        if r.get("fired") is None or "line" in r or r.get("csid") not in c["by_sid"]:
            continue
        i = c["by_sid"][r["csid"]]
        f = bool((c["c"]["Rmax"][i] >= th).any())
        out["n_rows"] += 1
        bad = f != bool(r["fired"])
        out["n_fired_differs"] += int(bad)
        if r["kind"] == "edit" and r.get("own_fired_last") is not None:
            o = bool(c["c"]["Rlast"][i][r["edit"]] >= th[r["edit"]])
            bo = o != bool(r["own_fired_last"])
            out["n_own_last_differs"] += int(bo)
            bad = bad or bo
        if bad and len(out["examples"]) < 5:
            out["examples"].append(r["csid"])
    return out


TOTAL_HV = {"n_rows": 0, "n_fired_differs": 0, "n_own_last_differs": 0}


def eval_cell(part, tag, M, meta, hn=False, d2=False, probes=None, theta=None):
    """Edits + held-out probes (+ hard negatives, + D2) of one method -> cell files.  Returns False if the cell existed."""
    if e2c.cell_done(part, NAME, tag):
        return False
    t0 = time.time()
    seqs = EDIT_ROWS + (HD_ROWS if probes is None else probes) + (HN_ROWS if hn else [])
    rows = annotate(ses.tf(M, seqs))
    t1 = time.time()
    hv = None
    if theta is not None:
        hv = hook_vs_vectors(theta, rows)
        for k in TOTAL_HV:
            TOTAL_HV[k] += hv[k]
    if d2:
        rows = rows + [dict(r, d2_row=True) for r in ses.d2(M, rows=D2_ROWS)]
    meta = dict(meta, rec=e2c.slim_rec(M.rec), digest=M.digest, kind=M.kind, has_hn=bool(hn), has_d2=bool(d2), smoke=SM,
                seconds={"tf": t1 - t0, "d2": time.time() - t1}, hook_vs_vectors=hv,
                theta=None if theta is None else [float(x) for x in theta.detach().cpu()])
    e2c.write_cell(part, NAME, tag, meta, rows)
    say(f"cell {part}/{tag}: {len(rows)} rows, tf {t1 - t0:.0f}s, d2 {time.time() - t1:.0f}s" + (f", hook vs vectors {hv['n_fired_differs']}/{hv['n_own_last_differs']} of {hv['n_rows']}" if hv else ""))
    return True


# ----------------------------------------------------------------------------- base
def block_base():
    eval_cell("base", "deployed", DEP, {"method": "deployed", "variant": "", "n": 0, "n_label": "0", "seed": None}, hn=True, d2=True, probes=HO_ROWS)
    eval_cell("base", "preedit", PRE, {"method": "preedit", "variant": "", "n": 0, "n_label": "0", "seed": None}, hn=True, d2=True, probes=HO_ROWS)
    M = fx.methods.where_least(L, [], H, start="deployed", label="where_floor")
    eval_cell("base", "where_floor", M, {"method": "where_floor", "variant": "all thresholds -inf", "n": 0, "n_label": "0", "seed": None}, hn=True,
              probes=HO_ROWS, theta=M.obj.theta())


# ----------------------------------------------------------------------------- part c': two-block thresholds for ReScope-W
def gate_from_vectors(theta):
    """Gate-level read-outs of a ReScope-W threshold vector from the captured vectors (no model)."""
    c = cap_ix()
    th = theta.detach().cpu().double()
    Rm, Rl = c["c"]["Rmax"], c["c"]["Rlast"]
    fired = (Rm >= th[None, :]).any(1).numpy()
    ncand = (Rm >= th[None, :]).sum(1).numpy()
    pct = lambda v: 100.0 * float(np.mean(v)) if len(v) else None      # noqa: E731
    ix = c["ix"]
    out = {"probe": {}, "edit": {}, "hn": {}}
    for pool in ("heldout_disjoint", "heldout_strict", "repair"):
        rr = [ix[j] for j in L.pool(pool) if j in ix]
        out["probe"][pool] = {"n": len(rr), "fired": pct(fired[rr]), "n_fired": int(fired[rr].sum()),
                              "width_mean": float(ncand[rr][fired[rr]].mean()) if fired[rr].any() else None}
    hd = [j for j in L.pool("heldout_disjoint") if j in ix]
    for nm, sel in (("near", [j for j in hd if L.near_edit(j) is not None]), ("far", [j for j in hd if L.near_edit(j) is None])):
        rr = [ix[j] for j in sel]
        out["probe"]["heldout_disjoint_" + nm] = {"n": len(rr), "fired": pct(fired[rr])}
    for m in ("rel", "tgen", "igen"):
        js = [j for j in L.edit_rows[m] if j in ix]
        own_any = [bool(Rm[ix[j]][L.own(j)] >= th[L.own(j)]) for j in js]
        own_last = [bool(Rl[ix[j]][L.own(j)] >= th[L.own(j)]) for j in js]
        out["edit"][m] = {"n": len(js), "fired": pct(fired[[ix[j] for j in js]]), "own_any": pct(own_any), "own_last": pct(own_last)}
    for t in ("HN-I", "HN-Q", "HN-X"):
        js = [j for j in L.rows(("hn_i", "hn_q", "hn_x"), valid=True) if j in ix and L.index[j]["hn_type"] == t]
        out["hn"][t] = {"n": len(js), "fired": pct(fired[[ix[j] for j in js]]),
                        "own_any": pct([bool(L.own(j) >= 0 and Rm[ix[j]][L.own(j)] >= th[L.own(j)]) for j in js])}
    return out


def where_cal(d, allow_silence=False):
    """ReScope-W (all positions, logged values, linear response) with the library's two-block rule on the captured vectors of the
    n = 200 benign REPAIR queries of the driver's calibration draw d (alpha = 0.05; guards as the library applies them)."""
    dr = fx.draws.calibration(L, 200, d)
    Q = vec(dr.rows)
    S2, crec = calibrate_two_block(where_base(), coords=Q, alpha=0.05, split=(1, 1), seed=int(dr.seed), rank="excess", allow_silence=allow_silence)
    rec = {"what": "ReScope-W with two-block thresholds (amendment R24): role 'all', fire 'all', response LIN, logged values; calibration scores = "
                   "largest normalised coordinate over all positions of the teacher-forced pre-edit trace of each query (captured on the model's device)",
           "cal": {k: v for k, v in crec.items() if k not in ("block1", "block2")}, "draw": {k: v for k, v in dr.rec.items() if k != "sids"},
           "n_cal": len(dr.rows), "allow_silence": allow_silence}
    M = fx.methods.wrap(L, S2, "where_cal" + ("_literal" if allow_silence else ""), rec, name="where_cal", alpha=0.05, n=200, draw=d,
                        variant="two_block" + (",allow_silence" if allow_silence else ""))
    return M, crec


def block_cw():
    pg = os.path.join(e2c.OUT, "parts", f"cw_gate_{NAME}.json")
    if not os.path.exists(pg):
        per = []
        for d in range(fx.draws.N_DRAWS):
            M, crec = where_cal(d)
            Ml, lrec = where_cal(d, allow_silence=True)
            g, gl = gate_from_vectors(M.obj.theta()), gate_from_vectors(Ml.obj.theta())
            per.append({"draw": d, "gate": g, "gate_literal": gl, "n_edits_capped": int(crec["n_edits_capped"]), "eta": float(crec.get("eta", float("nan"))),
                        "bound_false_firing": crec.get("bound_false_firing"), "n_own_keys_silenced_by_literal_rule": crec.get("n_own_keys_silenced_by_literal_rule"),
                        "n_own_keys_silenced_literal_run": lrec.get("n_own_keys_silenced"), "digest": M.digest})
        fx.io.jdump({"log": NAME, "method": "where_cal", "alpha": 0.05, "n": 200, "draws": per, "smoke": SM,
                     "what": "gate level from the captured all-position vectors; gate = with guards (as run), gate_literal = the literal rule without guards"}, pg, indent=1)
        say("cw gate level: 20 draws written")
    for d in (0, 1, 2):
        M, crec = where_cal(d)
        eval_cell("cw", f"where_cal__d{d}", M, {"method": "where_cal", "variant": "two_block", "n": 200, "n_label": "200", "seed": None, "draw": d, "alpha": 0.05,
                                               "n_edits_capped": int(crec["n_edits_capped"])}, hn=(d == 0), d2=(d == 0), theta=M.obj.theta())


# ----------------------------------------------------------------------------- part b: least repair of reported complaints
def least_from_vectors(S, Q, order):
    """The library's least_repair (one complaint at a time, on_conflict='refuse', default margins) on the complaints' captured
    per-query maxima Q (n, D): the same arithmetic (rescope.repair._above / _guards / _guard_fires), no keys needed."""
    S1 = S.fold()
    tau = S1.det_tau.clone()
    ge, Rg = _guards(S1)
    refused = []
    for i in order:
        new = torch.maximum(tau, _above(Q[int(i)].to(tau.device), MARGIN, ABS_MARGIN))
        before, after = _guard_fires(S1, tau, Rg, ge), _guard_fires(S1, new, Rg, ge)
        if bool((before & ~after).any()):
            refused.append(int(i))
            continue
        tau = new
    return tau, sorted(refused)


def b_cells(which):
    first = [(n, 0) for n in e2c.N_HN]
    rest = [(10, 1), (10, 2)] + [(n, s) for n in (1, 2, 5, 20) for s in (0, 1, 2)] + [(50, 0)]
    return first if which == "b1" else rest


def block_b(which):
    starts = {"deployed": where_base(), "logonly": fx.methods.where_logonly(L, H).obj}
    for n, s in b_cells(which):
        comp = fx.draws.complaints(L, n, s)
        if SM:
            comp = comp[:max(1, len(comp) // SM)]
        bt = f"blocklist__n{n}__s{s}"
        if not e2c.cell_done("bl", NAME, bt):
            rows = annotate(ses.blocklist(comp, EDIT_ROWS + HD_ROWS))
            e2c.write_cell("bl", NAME, bt, {"method": "blocklist", "variant": "deployed + exact-match list of the reported queries", "n": len(comp), "n_label": str(n),
                                            "seed": s, "n_listed_queries": len({L.index[j]["query_id"] for j in comp}), "smoke": SM}, rows)
        for start in ("deployed", "logonly"):
            tag = f"where_least__{start}__n{n}__s{s}"
            if e2c.cell_done("b", NAME, tag):
                continue
            t0 = time.time()
            M = fx.methods.where_least(L, comp, H, start=start, seed=s, label=f"where_least_{start}_n{n}")
            Q = vec(comp)
            tau0, ref0 = least_from_vectors(starts[start], Q, range(len(comp)))
            th = M.obj.theta().cpu()
            rng = np.random.default_rng([20261111, s, len(comp), 0 if start == "deployed" else 1])
            n_eq = 0
            for _ in range(20):
                tk, rk = least_from_vectors(starts[start], Q, rng.permutation(len(comp)))
                n_eq += int(torch.equal(tk, tau0) and rk == ref0)
            lr = M.rec["least_repair"]
            still = bool(len(comp)) and (Q >= th[None, :]).any(1)
            meta = {"method": "where_least", "variant": start, "n": len(comp), "n_label": str(n), "seed": s, "start": start,
                    "n_refused": lr["n_refused"], "n_resolved_by_record": lr["n_resolved"], "n_thresholds_raised_total": lr["n_thresholds_raised_total"],
                    "n_keys": lr["n_keys"], "n_reported_still_firing": int(still.sum()) if len(comp) else 0,
                    "order_check": {"n_orders": 20, "n_equal_thresholds": n_eq, "vector_rule_equals_driver_thresholds": bool(torch.equal(tau0, th)),
                                    "refused_equal": len(ref0) == lr["n_refused"]},
                    "n_finite_thresholds": int(torch.isfinite(th).sum()), "seconds_build": time.time() - t0}
            eval_cell("b", tag, M, meta, hn=n in e2c.N_HN, d2=n in e2c.N_HN, theta=M.obj.theta())


# ----------------------------------------------------------------------------- part c: ReScope on label-free evidence (behaviour)
def c_cells():
    out = [("rescope_n0__s0", lambda: fx.methods.evidence(L, 0, 0), 0, 0, False)]
    for n in (10, 1, 2, 5, 20):
        for s in (0, 1, 2):
            out.append((f"rescope_n{n}__s{s}", (lambda n=n, s=s: fx.methods.evidence(L, n, s)), n, s, n == 10))
    for n in (100, N_REPAIR, 200, 40, 50):
        for d in (0, 1, 2):
            out.append((f"rescope_n{n}__d{d}", (lambda n=n, d=d: fx.methods.evidence(L, n, d, 0.05)), n, d, n in (100, N_REPAIR)))
            out.append((f"rescope_shared_n{n}__d{d}", (lambda n=n, d=d: fx.methods.rescope_shared(L, alpha=0.05, n=n, draw=d, label=f"rescope_shared_n{n}")), n, d,
                        n in (100, N_REPAIR)))
    return out


def block_c():
    for tag, build, n, d, hn in c_cells():
        if e2c.cell_done("c", NAME, tag):
            continue
        M = build()
        meta = {"method": "rescope_shared" if "shared" in tag else "rescope", "variant": "logonly" if n == 0 else ("logonly+least" if n < 40 else M.variant), "n": n, "n_label": "ALL" if n == N_REPAIR else str(n),
                "seed": d if n < 40 else None, "draw": d, "alpha": M.alpha, "n_edits_capped": fx.methods.n_capped(M), "evidence_rule": M.rec.get("evidence", {}).get("rule")}
        eval_cell("c", tag, M, meta, hn=hn, d2=hn)


# ----------------------------------------------------------------------------- part d: linear rows
def p12_stream(seed):
    """The complaint dicts of the harness in the order of P12's nested stream (p12_common.build_stream), checked against the driver's."""
    HS = hz.HS
    base = HS.sample_complaints(H, 100, seed=seed)
    img, txt = [c for c in base if c["ptype"] == "image"], [c for c in base if c["ptype"] == "text"]
    stream = [txt[k // 2] if k % 2 == 0 else img[k // 2] for k in range(100)]
    used = {c["group"] for c in stream}
    pl = HS.pool_of(H)
    rows = sorted([r for r in H.rows if pl[r["probe_idx"]] == "repair" and r["y0_valid"]], key=lambda r: (r["probe_idx"], r["ptype"]))
    rest = [dict(r, complaint_kind="D1", sample_seed=seed) for r in rows if r["is_complaint_D1"] and r["group"] not in used]
    stream += [rest[int(i)] for i in np.random.default_rng([seed, 12]).permutation(len(rest))]
    if [c["group"] for c in stream] != [L.index[j]["group"] for j in fx.draws.complaint_stream(L, seed)]:
        raise RuntimeError("the harness stream differs from the driver's complaint stream")
    return stream


CORR = os.path.join(e2c.OUT, "corr", NAME)
P12D = os.path.join(e2c.P12, "out", NAME)


def record_flags(tag):
    r = fx.io.jload(os.path.join(P12D, tag + ".json"))
    by_group = {L.index[j]["group"]: L.sid(j) for j in L.probe_rows}
    ho = {by_group[h["group"]]: bool(h["reproduced_after"]) for h in r["eval"]["heldout_rows"]}
    ed = {L.sid(L.edit_rows[m][i]): bool(x) for m in ("rel", "tgen", "igen") for i, x in enumerate(r["eval"]["edits"][m]["em_list"])}
    return r, ho, ed


def compare_with_record(tag, rows):
    _, ho, ed = record_flags(tag)
    out = {"record": tag, "n_heldout": 0, "n_heldout_differ": 0, "n_heldout_disjoint_differ": 0, "n_edit": 0, "n_edit_differ": 0, "differing": []}
    for r in rows:
        if "line" in r:
            continue
        if r["kind"] == "probe" and r["csid"] in ho:
            out["n_heldout"] += 1
            if bool(r["exact"]) != ho[r["csid"]]:
                out["n_heldout_differ"] += 1
                out["n_heldout_disjoint_differ"] += int(bool(r.get("disjoint")))
                out["differing"].append(r["csid"])
        elif r["kind"] == "edit" and r["csid"] in ed:
            out["n_edit"] += 1
            if bool(r["exact"]) != ed[r["csid"]]:
                out["n_edit_differ"] += 1
                out["differing"].append(r["csid"])
    out["differing"] = out["differing"][:40]
    return out


def p12_saved(tag):
    p = os.path.join(P12D, "corr", tag + ".pt")
    return torch.load(p, map_location="cpu") if os.path.exists(p) else None


def reedit_build():
    p = os.path.join(CORR, "REEDIT_B__n10__s0.pt")
    if os.path.exists(p):
        return
    cs = p12_stream(0)[:10]
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    corr, info = rescope.reedit(H, cs)
    W = corr.W.detach().cpu()
    os.makedirs(CORR, exist_ok=True)
    torch.save({"kind": "weight", "W": W, "info": info, "groups": [c["group"] for c in cs], "seconds": time.time() - t0,
                "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()), "built": time.strftime("%Y-%m-%d %H:%M:%S"), "code": fx.code_fingerprint()}, p + ".tmp")
    os.replace(p + ".tmp", p)
    say(f"REEDIT-B n = 10 built in {time.time() - t0:.0f}s: installed {len(info['installed_rows'])}, attempts {info['zopt_attempts']}, peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


def d_cell(tag, M, meta, cmp_tag=None):
    if e2c.cell_done("d", NAME, tag):
        return
    t0 = time.time()
    rows = annotate(ses.tf(M, EDIT_ROWS + HO_ROWS + HN_ROWS))
    t1 = time.time()
    cmp_ = compare_with_record(cmp_tag, rows) if cmp_tag else None
    rows = rows + [dict(r, d2_row=True) for r in ses.d2(M, rows=D2_ROWS)]
    e2c.write_cell("d", NAME, tag, dict(meta, rec=e2c.slim_rec(M.rec), digest=M.digest, kind=M.kind, has_hn=True, has_d2=True, smoke=SM, flags_vs_record=cmp_,
                                         seconds={"tf": t1 - t0, "d2": time.time() - t1}), rows)
    say(f"cell d/{tag}: tf {t1 - t0:.0f}s d2 {time.time() - t1:.0f}s; flags vs record: {None if cmp_ is None else (cmp_['n_heldout_differ'], cmp_['n_edit_differ'])}")


def block_d():
    os.makedirs(CORR, exist_ok=True)
    # COMP-B on ALL reported complaints (seed 0), rebuilt once with the options of P12
    tag = "COMP_B__nALL__s0"
    if not e2c.cell_done("d", NAME, tag):
        cs = p12_stream(0)
        t0 = time.time()
        corr, cost = hz.pkg.repair("COMP", H, cs, protect=("K_ans",), query="answer", form="adapter")
        torch.save({"kind": "adapter", "Lf": corr.Lf.cpu(), "Rf": corr.Rf.cpu(), "gate": corr.gate}, os.path.join(CORR, tag + ".pt"))
        old = p12_saved(tag)
        same = None if old is None else {"Lf_max_abs_diff": float((old["Lf"].double() - corr.Lf.cpu().double()).abs().max()) if old["Lf"].shape == corr.Lf.shape else None,
                                         "Rf_max_abs_diff": float((old["Rf"].double() - corr.Rf.cpu().double()).abs().max()) if old["Rf"].shape == corr.Rf.shape else None,
                                         "bitwise": bool(old["Lf"].shape == corr.Lf.shape and torch.equal(old["Lf"], corr.Lf.cpu()) and torch.equal(old["Rf"], corr.Rf.cpu()))}
        m = {k: v for k, v in corr.meta.items() if isinstance(v, (int, float, str, bool))}
        M = fx.methods.wrap(L, corr, "comp_b_nall", {"harness": "COMP", "options": {"protect": ["K_ans"], "query": "answer", "form": "adapter"}, "meta": m},
                            name="COMP_B", n=len(cs), draw=0, variant="rebuilt")
        d_cell(tag, M, {"method": "COMP_B", "variant": "rebuilt", "n": len(cs), "n_label": "ALL", "seed": 0, "restores": e2c.RESTORES["COMP_B"],
                        "seconds_build": time.time() - t0, "vs_p12_saved_correction": same}, cmp_tag=tag)
        del corr
    # rollbacks from the removal sets of the records
    for meth, n in (("RB_UNION", 10), ("RB_UNION", "ALL"), ("RB_TOT", 10)):
        tag = f"{meth}__n{n}__s0"
        if e2c.cell_done("d", NAME, tag):
            continue
        rec, _, _ = record_flags(tag)
        R = [int(i) for i in rec["extra"]["R"]]
        M = fx.methods.rollback(L, H, R, label=f"{meth.lower()}_n{str(n).lower()}")
        old = p12_saved(tag)
        same = None if old is None or old.get("kind") != "weight" else bool(torch.equal(old["W"], M.obj.cpu()))
        torch.save({"kind": "weight", "W": M.obj.cpu(), "R": R}, os.path.join(CORR, tag + ".pt"))
        d_cell(tag, M, {"method": meth, "variant": "rebuilt", "n": e2c.n_int(L, n), "n_label": str(n), "seed": 0, "restores": e2c.RESTORES[meth], "rollback_R": R,
                        "rollback_k": len(R), "weight_equals_p12_saved": same}, cmp_tag=tag)
    # REEDIT-B at n = 10: built once (block reedit_build), the saved weight is evaluated
    tag = "REEDIT_B__n10__s0"
    p = os.path.join(CORR, tag + ".pt")
    if not e2c.cell_done("d", NAME, tag):
        if os.path.exists(p):
            b = torch.load(p, map_location="cpu")
            M = fx.methods.wrap(L, b["W"].to(H.dev), "reedit_b_n10", {"harness": "REEDIT", "info": b["info"], "file": p, "built": b["built"]}, name="REEDIT_B", n=10, draw=0,
                                variant="rebuilt")
            d_cell(tag, M, {"method": "REEDIT_B", "variant": "rebuilt", "n": 10, "n_label": "10", "seed": 0, "restores": e2c.RESTORES["REEDIT_B"],
                            "reedit_info": b["info"], "seconds_build": b["seconds"]}, cmp_tag=tag)
        else:
            say("REEDIT-B n = 10: no rebuilt weight (block reedit_build did not finish): cell skipped")
    # REEDIT-B at ALL reported complaints: the weight P12 saved (a re-edit is built once; not rebuilt here)
    tag = "REEDIT_B__nALL__s0"
    if not e2c.cell_done("d", NAME, tag):
        M = fx.methods.p12(L, H, "REEDIT_B", "ALL", 0)
        if M.available:
            d_cell(tag, M, {"method": "REEDIT_B", "variant": "p12_saved_weight", "n": N_ALL, "n_label": "ALL", "seed": 0, "restores": e2c.RESTORES["REEDIT_B"]}, cmp_tag=tag)
        else:
            say(f"REEDIT-B ALL: {M.rec.get('unavailable')}")


# ----------------------------------------------------------------------------- free generation of the edit prompts (main-table rows)
def block_gen():
    def comp_saved():
        b = torch.load(os.path.join(CORR, "COMP_B__nALL__s0.pt"), map_location="cpu")
        return fx.methods.wrap(L, hz.HL.Correction("adapter", Lf=b["Lf"].to(H.dev), Rf=b["Rf"].to(H.dev), gate=b["gate"], meta={}), "comp_b_nall")

    todo = [("d", "COMP_B__nALL__s0", comp_saved),
            ("d", "REEDIT_B__nALL__s0", lambda: fx.methods.p12(L, H, "REEDIT_B", "ALL", 0)),
            ("cw", "where_cal__d0", lambda: where_cal(0)[0]),
            ("base", "deployed", lambda: DEP)]
    for part, tag, build in todo:
        gt = tag + "__gen"
        if e2c.cell_done("gen", NAME, part + "__" + gt) or not e2c.cell_done(part, NAME, tag):
            continue
        t0 = time.time()
        M = build()
        rows = ses.generate(M, EDIT_ROWS)
        e2c.write_cell("gen", NAME, part + "__" + gt, {"method": fx.io.jload(e2c.cell_path(part, NAME, tag))["method"], "of_part": part, "of_tag": tag, "smoke": SM,
                                                      "seconds": time.time() - t0}, rows)
        say(f"gen {part}/{tag}: {len(rows)} decodes in {time.time() - t0:.0f}s")


# ----------------------------------------------------------------------------- main
def main():
    t0 = time.time()
    for b in BLOCKS:
        tb = time.time()
        say(f"block {b} starts")
        if b == "reedit_build":
            try:                                                        # first block of a fresh process (z* optimisation: backward passes)
                reedit_build()
            except Exception as ex:      # noqa: BLE001  (recorded; the cell REEDIT_B n = 10 is then skipped by block d)
                say(f"REEDIT-B n = 10 build FAILED: {type(ex).__name__}: {str(ex)[:300]}")
                fx.io.jdump({"log": NAME, "error": f"{type(ex).__name__}: {str(ex)[:1000]}", "time": time.strftime("%Y-%m-%d %H:%M:%S")},
                            os.path.join(e2c.OUT, "parts", f"reedit_build_failed_{NAME}_{int(time.time())}.json"), indent=1)
                import gc
                gc.collect()
            torch.cuda.empty_cache()
        elif b == "capture":
            cap_ix()
        elif b == "base":
            block_base()
        elif b == "cw":
            block_cw()
        elif b == "d":
            block_d()
        elif b in ("b1", "b2"):
            block_b(b)
        elif b == "c":
            block_c()
        elif b == "gen":
            block_gen()
        elif b == "e":
            import example
            example.run(sys.modules[__name__])
        else:
            raise ValueError(b)
        say(f"block {b} done in {time.time() - tb:.0f}s; session {ses.stats}")
    clean = ses.assert_clean()
    rec = {"log": NAME, "blocks": BLOCKS, "assert_clean": clean, "hook_vs_vectors_total": TOTAL_HV, "seconds": time.time() - t0, "stats": dict(ses.stats),
           "memo": ses.memo.summary() if hasattr(ses.memo, "summary") else None, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "smoke": SM}
    fx.io.jdump(rec, os.path.join(e2c.OUT, "parts", f"job_{NAME}_{'-'.join(BLOCKS)}_{int(time.time())}.json"), indent=1)
    say(f"job done: {clean}; hook vs vectors {TOTAL_HV}")
    assert TOTAL_HV["n_fired_differs"] == 0 and TOTAL_HV["n_own_last_differs"] == 0, TOTAL_HV


if __name__ == "__main__":
    main()
