"""E6, gate level for the two roles that look at more than the last prompt position (role = last + answer, role = all): 20 draws.

    gq run --mem <MiB> --prio 1 -- python e6_multi.py <log>

fx.gate decides role-'last' gates only (key cache).  For the other roles the coordinates at every eligible position are needed:
ONE truncated forward per sequence (library: Evaluator.capture), then everything is the library's arithmetic on the CPU:
  calibration   Q[q, j] = S.query_max(S.eligible_keys(x_q, lp_q)) for every valid REPAIR row (what fx.methods.calib_coords computes
                per draw), then fx.methods' own two-block rule (fx.methods._cal_two_block = rescope.calibrate_two_block) on the rows
                of the driver's draw fx.draws.calibration(L, 200, d).  CHECK: for draws 0 - 2 the calibrated gate must have the
                content digest of fx.methods.rescope(L, role=..., draw=d, H=H) (recorded; a mismatch stops the job).
  decision      R = S.scores(S.eligible_keys(x, lp)) (the hook's call), candidates R >= theta, top-1 per position by S.rank_key:
                the statements of ScopedLog.firing.  Per sequence: fired (any eligible position), the edits released, own-edit flags
                at the last prompt position (the driver's coverage definition, own_released_last) and at any position.
Sequences: valid heldout_disjoint probes, Rel / T-Gen / I-Gen of every edit, valid hard negatives.  The key-level pools (BEN-T) hold
only the last-prompt key: a multi-position role is not defined on them.
Output: out/multi_<log>.json in the layout of out/gate_<log>.json + per-sequence decisions of draws 0 - 2 (checked against the hook's
flags of the behavioural rows by e6_render.py).
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

import e6_common as C
from e6_common import fx

ROLES = {"role_lastans": "last+answer", "role_all": "all"}
INF = float("inf")


def decide_seq(g, R, last, own, own_b=None):
    """One sequence under one calibrated gate (CPU): R (P, D) scores at the eligible positions, `last` = row of the last prompt position."""
    cand = R >= g.theta()[None, :]
    anyp = cand.any(1)
    if not bool(anyp.any()):
        return {"fired": False, "rel": [], "n_cand": 0, "own_cand": False, "own_any": False, "own_last": False, "own_cand_last": False, "top_last": -1, "b_any": False}
    top = torch.where(cand, g.rank_key(R), torch.full_like(R, -INF)).argmax(1)
    ed = g.det_edit[top]
    rel = sorted({int(e) for e in ed[anyp].tolist()})
    ce = g._edit_any(cand)
    fl = bool(anyp[last])
    tl = int(ed[last]) if fl else -1
    return {"fired": True, "rel": rel, "n_cand": int(ce.any(0).sum()), "own_cand": bool(own >= 0 and ce[:, own].any()), "own_any": own >= 0 and own in rel,
            "own_last": own >= 0 and tl == own, "own_cand_last": bool(own >= 0 and ce[last, own]), "top_last": tl,
            "b_any": own_b is not None and own_b in rel}


def stats_of(L, seqs, dec, even):
    """Statistics in the layout of fx.metrics.gate_stats (the columns the tables use)."""
    pct = fx.metrics.pct
    out = {"probe": {}, "edit": {}, "hn": {}}
    pr = [(s, d) for s, d in zip(seqs, dec) if s["kind"] == "probe"]
    for nm, p in (("heldout_disjoint", lambda s: True), ("heldout_strict", lambda s: s["strict"]), ("heldout_disjoint_near", lambda s: s["near"]),
                  ("heldout_disjoint_far", lambda s: not s["near"]), ("heldout_disjoint_image", lambda s: s["ptype"] == "image"),
                  ("heldout_disjoint_text", lambda s: s["ptype"] == "text")):
        sub = [d for s, d in pr if p(s)]
        out["probe"][nm] = {"n": len(sub), "fired": pct(d["fired"] for d in sub)}
    w = [d["n_cand"] for s, d in pr if d["fired"]]
    out["probe"]["lineage"] = {"n_firing": len(w), "width_mean": float(np.mean(w)) if w else None, "width_max": int(max(w)) if w else None,
                               "multi": 100.0 * float(np.mean([x > 1 for x in w])) if w else None}
    par = {"even": {}, "odd": {}}
    for m in ("rel", "tgen", "igen"):
        sub = [(s, d) for s, d in zip(seqs, dec) if s["kind"] == m]
        out["edit"][m] = {"n": len(sub), "fired": pct(d["fired"] for s, d in sub), "own_cand": pct(d["own_cand_last"] for s, d in sub),
                          "own_top": pct(d["own_last"] for s, d in sub), "other_top": pct(d["fired"] and not d["own_last"] for s, d in sub),
                          "own_any": pct(d["own_any"] for s, d in sub), "own_cand_any": pct(d["own_cand"] for s, d in sub)}
        if m != "rel":
            for pn, pe in (("even", True), ("odd", False)):
                ss = [d for s, d in sub if (s["own"] in even) == pe]
                par[pn][m] = {"n": len(ss), "own_top": pct(d["own_last"] for d in ss), "own_cand": pct(d["own_cand_last"] for d in ss)}
    for t in ("HN-I", "HN-Q", "HN-X"):
        sub = [(s, d) for s, d in zip(seqs, dec) if s["kind"] == "hn" and s["hn_type"] == t]
        h = {"n": len(sub), "fired": pct(d["fired"] for s, d in sub), "own_cand": pct(d["own_cand_last"] for s, d in sub), "own_top": pct(d["own_last"] for s, d in sub),
             "own_any": pct(d["own_any"] for s, d in sub)}
        if t == "HN-X":
            h["b_any"] = pct(d["b_any"] for s, d in sub)
        out["hn"][t] = h
    allh = [d for s, d in zip(seqs, dec) if s["kind"] == "hn"]
    out["hn"]["all"] = {"n": len(allh), "fired": pct(d["fired"] for d in allh)}
    own_nr = sum(not d["own_last"] for s, d in zip(seqs, dec) if s["kind"] == "rel")
    own_nc = sum(not d["own_cand_last"] for s, d in zip(seqs, dec) if s["kind"] == "rel")
    return out, par, own_nr, own_nc


def main():
    name = sys.argv[1]
    nd = int(sys.argv[2]) if len(sys.argv) > 2 else C.N_DRAWS
    p_out = os.path.join(C.OUT, f"multi_{name}.json")
    if os.path.exists(p_out):
        print("[e6_multi] exists", p_out)
        return
    L = fx.logs.open(name)
    H = fx.logs.load_model(name)
    ses = fx.behave.session(L, H)
    base = {k: fx.methods.base_gate(L, "ncoord", r, "top1", "repl", True, True)[0] for k, r in ROLES.items()}
    dev = {k: fx.methods._slim(S).to(H.dev) for k, S in base.items()}
    pool = fx.draws.pool_rows(L, "repair", "tf")                         # the calibration pool: valid REPAIR rows
    ev_rows = L.pool("heldout_disjoint") + [j for m in ("rel", "tgen", "igen") for j in L.edit_rows[m]] + L.rows(("hn_i", "hn_q", "hn_x"), valid=True)
    strict = set(L.pool("heldout_strict"))
    even = set(fx.draws.even_edits(L))

    def scores(j):
        """{role key: (R (P, D) fp64 on the CPU, row of the last prompt position)} of sequence j: one truncated forward."""
        ses.ev.trim_vcache()
        x, lp = ses.ev.capture(ses.item(j))
        out = {}
        for k, Sg in dev.items():
            keys = Sg.eligible_keys(x, lp)                               # the hook's eligible positions (B = 1)
            out[k] = (Sg.scores(keys).cpu(), int(lp) if Sg.role == "all" else 0, Sg.query_max(keys).cpu())
        return out

    t0 = time.time()
    Q = {k: [] for k in ROLES}
    with torch.no_grad(), ses.ev.hz.HE.vision_cache(H):
        for j in pool:
            s = scores(j)
            for k in ROLES:
                Q[k].append(s[k][2])
    Q = {k: torch.cat(v) for k, v in Q.items()}
    pos = {j: i for i, j in enumerate(pool)}
    print(f"[e6_multi] {name}: Q of {len(pool)} REPAIR rows, {time.time() - t0:.0f}s", flush=True)

    # calibrated gates: per role, draw, (capped, literal)
    gates, cal, checks = {}, {}, {}
    for k, role in ROLES.items():
        for d in range(nd):
            dr = fx.draws.calibration(L, C.N, d)
            Qd = Q[k][[pos[j] for j in dr.rows]]
            for lit in (False, True):
                S2, crec = fx.methods._cal_two_block(base[k], Qd, C.ALPHA, dr.seed, (1, 1), "max", lit, "excess")
                gates[(k, d, lit)] = S2
                cal[(k, d, lit)] = crec
            if d < 3:                                                   # the driver's own constructor (200 truncated forwards per build)
                Md = fx.methods.rescope(L, alpha=C.ALPHA, n=C.N, draw=d, role=role, H=H)
                dg = fx.methods.content_digest(gates[(k, d, False)], L.name, "gate")
                checks[f"{k}_d{d}"] = {"driver_digest": Md.digest, "own_digest": dg, "equal": Md.digest == dg, "n_capped_driver": fx.methods.n_capped(Md)}
                print(f"[e6_multi] {name} {k} d{d}: digest check {checks[f'{k}_d{d}']}", flush=True)
                if Md.digest != dg:
                    raise RuntimeError(f"{k} draw {d}: the gate calibrated from the cached Q differs from the driver's gate")
    seqs, decs = [], {key: [] for key in gates}
    t0 = time.time()
    with torch.no_grad(), ses.ev.hz.HE.vision_cache(H):
        for n, j in enumerate(ev_rows):
            e = L.index[j]
            kind = e["kind"] if e["kind"] in ("rel", "tgen", "igen", "probe") else "hn"
            b = e.get("edit_b")
            seqs.append({"csid": e["sid"], "kind": kind, "own": L.own(j), "strict": j in strict, "near": kind == "probe" and L.near_edit(j) is not None,
                         "ptype": e.get("ptype"), "hn_type": e.get("hn_type"), "edit_b": None if b is None else int(b)})
            s = scores(j)
            for key, g in gates.items():
                R, last, _ = s[key[0]]
                decs[key].append(decide_seq(g, R, last, seqs[-1]["own"], seqs[-1]["edit_b"]))
            if (n + 1) % 500 == 0:
                print(f"[e6_multi] {name}: decided {n + 1}/{len(ev_rows)} sequences, {time.time() - t0:.0f}s", flush=True)
    res = {}
    for k in ROLES:
        res[k] = {}
        for d in range(nd):
            st, par, own_nr, own_nc = stats_of(L, seqs, decs[(k, d, False)], even)
            sl, _, lit_nr, _ = stats_of(L, seqs, decs[(k, d, True)], even)
            c, cl = cal[(k, d, False)], cal[(k, d, True)]
            g = gates[(k, d, False)]
            res[k][str(d)] = {"stats": st, "parity": par,
                              "cal": {"n_capped": c.get("n_edits_capped"), "own_not_released": own_nr, "own_not_candidate": own_nc, "eta": float(c["eta"]) if c.get("eta") is not None else None,
                                      "bound": c.get("bound_false_firing"), "n_own_keys_silenced_by_literal_rule": c.get("n_own_keys_silenced_by_literal_rule"),
                                      "n_edits_silenced_by_literal_rule": c.get("n_edits_silenced_by_literal_rule"), "variant": f"role={ROLES[k]}",
                                      "digest": fx.methods.content_digest(g, L.name, "gate"), "n": C.N, "draw": d},
                              "BEN-T": None,
                              "literal": {"same_as_row": bool((g.theta() == gates[(k, d, True)].theta()).all()), "heldout_disjoint": sl["probe"]["heldout_disjoint"]["fired"],
                                          "heldout_strict": sl["probe"]["heldout_strict"]["fired"], "BEN-T": None, "rel_own_top": sl["edit"]["rel"]["own_top"],
                                          "own_not_released": lit_nr}}
            if d < 3:
                res[k][str(d)]["per_seq"] = [{"csid": s["csid"], "fired": x["fired"], "own_last": x["own_last"], "own_cand_last": x["own_cand_last"], "top_last": x["top_last"],
                                              "rel": x["rel"]} for s, x in zip(seqs, decs[(k, d, False)])]
    fx.io.jdump(res, p_out)
    fx.io.jdump({"log": name, "n_draws": nd, "checks": checks, "n_pool": len(pool), "n_sequences": len(ev_rows), "code": fx.code_fingerprint(),
                 "BEN-T": "not defined: key-level pools hold the last-prompt key only", "assert_clean": ses.assert_clean(),
                 "written": time.strftime("%Y-%m-%d %H:%M:%S")}, os.path.join(C.OUT, f"multi_{name}_meta.json"), indent=1)
    print("[e6_multi] done", name, flush=True)


if __name__ == "__main__":
    main()
