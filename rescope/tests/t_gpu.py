"""GPU tests on a real log (run through run_tests.py gpu --log <name>).  Every test writes one JSON to tests/out/.

  b1_linear      (i)   NON-TAUTOLOGICAL: W_0 k + sum_i U_i <W_i, k> (response 'lin', role 'all', fire 'all', no
                       threshold, NO round-off term E, module weight W_0) against the real W_E forward, 30 sequences.
                       Cannot be bitwise: the deployed weight contains the dense write round-off E and the two
                       module outputs are rounded differently; reported: max |logit diff|, argmax agreement.
  b2_off         (ii)  thresholds +inf: bitwise the W_0 forward on the same 30 sequences (hook called, nothing fires).
  b345_pass      (iii) every non-firing row bitwise the plain W_0 forward (all rows, 5 configs);
                 (iv)  firing predicted from the captured keys == firing observed inside the hook (all rows);
                 (v)   regression against P10 (HARD-SELF-noE) and P11a (R1-REL0.35-REPL / -GAIN, R1-POOL1-REPL,
                       NDB-R1-0.95): Rel / T-Gen / I-Gen exact, D1 on all valid probe rows, hard-negative D1,
                       row-level agreement of the exact and firing flags.
  b6_harness     (vi)  harness linear corrections (NONE, COMP adapter, COMP merged weight, REEDIT with chunk=1) through
                       Evaluator: edit flags, HELD-OUT D1 flags and complaint flags equal to harness.evaluate.
  b7_decode            decode pipeline (decoder v2, D2): inactive gate == stored pre-edit answers, W_E == stored
                       post-edit answers / D2 flags, position bookkeeping of the hook through KV-cache generation.
  b9_paths             teacher-forced path vs decode path (review F1): where the last prompt position sits on each,
                       lp='prompt' scoring, least repair end to end on both paths with the documented protocol.
  b10_own              own prompts at B = 1 (review F5): B = 1 coordinates, guards from Evaluator.own_keys, capped thresholds.
  c_timing       (c)   per-forward overhead of the hook (role 'last' / 'all', gate in fp64 / module dtype).
"""
from __future__ import annotations

import gc
import json
import os
import time

import numpy as np
import torch

from rescope import baselines as BL, calibrate as CA, evaluate as EV
from rescope.scoped import INF, ScopedLog

P10 = EV.EXP + "/08_rescope/p10_scope_probe/out"
P11 = EV.EXP + "/08_rescope/p11_retrofit/out"
REG = ("P10-HARD-SELF-noE", "R1-REL0.35-REPL", "R1-REL0.35-GAIN", "R1-POOL1-REPL", "NDB-R1-0.95")


def jdump(obj, path):
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def jload(path):
    with open(path) as f:
        return json.load(f)


def seq30(ev):
    """10 edit sequences + 10 image probe rows + 10 text probe rows, seeded."""
    rng = np.random.default_rng(20261110)
    pr = ev.probe_items("all")
    pick = lambda xs, n: [xs[int(i)] for i in sorted(rng.choice(len(xs), n, replace=False))]          # noqa: E731
    return pick(ev.edit_items(), 10) + pick([p for p in pr if p["ptype"] == "image"], 10) + pick([p for p in pr if p["ptype"] == "text"], 10)


# ----------------------------------------------------------------------------- (i)
class _RoundedWE:
    """Control gate: the deployed function with ANOTHER rounding (module output computed in fp64, one cast), at every
    position.  Its distance to the native W_E forward is the round-off floor of the low-precision forward."""

    def __init__(self, WE, W0):
        self.WE, self.W0, self.role = WE, W0, "all"

    def firing(self, x, lp=None, pos0=0, state=None):
        from rescope.scoped import Firing
        n, Tn = x.shape[0] * x.shape[1], x.shape[1]
        idx = torch.arange(n, device=x.device)
        return Firing(idx=idx, row=idx // Tn, pos=idx % Tn + pos0, edit=torch.full_like(idx, -1))

    def respond(self, x, out, f):
        return torch.cat([(xs.double() @ self.WE.double().T).to(out.dtype) for xs in x.reshape(-1, x.shape[-1]).split(512)]).view(out.shape)


def b1_linear(ev, log):
    """Pre-specified criterion (fixed before the first run): PASS iff
       (1) every position fires under the un-thresholded gate,
       (2) module output: ||out_lin - out_WE||_F / ||out_WE - out_W0||_F <= 0.05 on every sequence (the write round-off E is
           0.5 % / 0.06 % of the log in Frobenius norm; a wrong factor of one edit in 100 gives >= 0.1),
       (3) logits: argmax equal to the W_E forward on >= 95 % of the answer tokens,
       (4) negative control: dropping the edit's own term from its Rel / T-Gen / I-Gen sequence moves the module output at the
           last prompt position >= 10 times further from W_E than the full linear log is (median over the edit sequences)."""
    H = ev.H
    S = ScopedLog.from_log(log, role="all", fire="all", response="lin", W0=H.W0)
    floor = _RoundedWE(H.WE, H.W0)

    def run(method, it, b):
        with ev.applied(method):
            with H.vlm.capture(b, [H.L], positions="all", module="edit", io=("out",), dtype=None) as cap:   # after the gate's hook
                r, l = ev.tf(it, b, logits=True)
        return r, l, cap.out[H.L][0].double()

    rows = []
    with torch.no_grad():                                               # no vision cache: plain forwards only
        for it in seq30(ev):
            b = ev.batch(it)
            lp = int(b["last_prompt_idx"][0])
            rE, lE, oE = run(None, it, b)
            r0, l0, o0 = run(H.W0, it, b)
            r, l, o = run(S, it, b)
            rF, lF, oF = run(floor, it, b)
            den, den_lp = float((oE - o0).norm()), float((oE[lp] - o0[lp]).norm())
            eq = l.argmax(-1) == lE.argmax(-1)
            row = {"sid": it["sid"], "kind": it["kind"], "seq_len": int(b["seq_len"][0]), "n_tok": r["n_tok"],
                   "max_abs_diff": float((l.float() - lE.float()).abs().max()),
                   "max_abs_diff_W0_vs_WE": float((l0.float() - lE.float()).abs().max()),
                   "max_abs_diff_floor": float((lF.float() - lE.float()).abs().max()),
                   "argmax_equal": bool(eq.all()), "n_argmax_tok_equal": int(eq.sum()),
                   "n_argmax_tok_equal_floor": int((lF.argmax(-1) == lE.argmax(-1)).sum()),
                   "exact_equal": r["exact"] == rE["exact"], "exact": r["exact"], "exact_WE": rE["exact"], "exact_W0": r0["exact"],
                   "bitwise": bool(torch.equal(l, lE)), "all_positions_fire": r["n_fire_pos"] == int(b["seq_len"][0]),
                   "module_rel_err": float((o - oE).norm()) / den, "module_rel_err_floor": float((oF - oE).norm()) / den,
                   "module_rel_err_last": float((o[lp] - oE[lp]).norm()) / den_lp}
            if it["kind"] == "edit":                                    # negative control: the same log without this edit's own term
                tau = S.det_tau.clone()
                tau[it["edit"]] = INF
                rD, lD, oD = run(S.replace(det_tau=tau), it, b)
                row.update(control_module_rel_err_last=float((oD[lp] - oE[lp]).norm()) / den_lp,
                           control_max_abs_diff=float((lD.float() - lE.float()).abs().max()), control_exact=rD["exact"])
            rows.append(row)
    ed = [x for x in rows if x["kind"] == "edit"]
    n_tok = sum(x["n_tok"] for x in rows)
    out = {"n": len(rows), "max_abs_logit_diff": max(x["max_abs_diff"] for x in rows),
           "median_abs_logit_diff": float(np.median([x["max_abs_diff"] for x in rows])),
           "median_max_abs_logit_diff_W0_vs_WE": float(np.median([x["max_abs_diff_W0_vs_WE"] for x in rows])),
           "roundoff_floor_max_abs_logit_diff": max(x["max_abs_diff_floor"] for x in rows),
           "n_argmax_equal": sum(x["argmax_equal"] for x in rows),
           "n_argmax_equal_by_kind": {k: [sum(x["argmax_equal"] for x in rows if x["kind"] == k), sum(x["kind"] == k for x in rows)] for k in ("edit", "probe")},
           "n_tok": n_tok, "n_argmax_tok_equal": sum(x["n_argmax_tok_equal"] for x in rows),
           "roundoff_floor_n_argmax_tok_equal": sum(x["n_argmax_tok_equal_floor"] for x in rows),
           "n_exact_flag_equal": sum(x["exact_equal"] for x in rows), "n_bitwise": sum(x["bitwise"] for x in rows),
           "n_all_positions_fire": sum(x["all_positions_fire"] for x in rows),
           "module_rel_err_max": max(x["module_rel_err"] for x in rows), "module_rel_err_median": float(np.median([x["module_rel_err"] for x in rows])),
           "roundoff_floor_module_rel_err_max": max(x["module_rel_err_floor"] for x in rows),
           "control_ratio_median": float(np.median([x["control_module_rel_err_last"] / max(x["module_rel_err_last"], 1e-12) for x in ed])),
           "control_module_rel_err_last_median": float(np.median([x["control_module_rel_err_last"] for x in ed])),
           "module_rel_err_last_median_edits": float(np.median([x["module_rel_err_last"] for x in ed])),
           "n_edit_sequences": len(ed), "n_edits_exact_WE": sum(x["exact_WE"] for x in ed), "n_edits_exact_lin": sum(x["exact"] for x in ed),
           "n_edits_exact_own_term_dropped": sum(x["control_exact"] for x in ed), "n_edits_exact_W0": sum(x["exact_W0"] for x in ed), "rows": rows,
           "what": "scoped log with response 'lin', role 'all', fire 'all', tau = -inf on module weight W_0 (= W_0 k + sum_i U_i <W_i, k>, fp64 side "
                   "path, no E term) vs the plain forward at the deployed W_E.  Not bitwise by construction: W_E contains the dense write round-off E "
                   "and the two module outputs are rounded differently; 'roundoff floor' = the deployed function itself with an fp64 module product.",
           "criterion": b1_linear.__doc__}
    out["pass"] = bool(out["n_all_positions_fire"] == len(rows) and out["module_rel_err_max"] <= 0.05
                       and out["n_argmax_tok_equal"] >= 0.95 * n_tok and out["control_ratio_median"] >= 10)
    return out


# ----------------------------------------------------------------------------- (ii)
def b2_off(ev, log):
    H = ev.H
    S = ScopedLog.from_log(log, role="all", tau=INF, W0=H.W0)
    rows = []
    with torch.no_grad():                                               # no vision cache: plain forwards only
        for it in seq30(ev):
            b = ev.batch(it)
            with ev.applied(H.W0):
                r0, l0 = ev.tf(it, b, logits=True)
            with ev.applied(None):
                rE, lE = ev.tf(it, b, logits=True)
            for role in ("all", "last"):
                with ev.applied(S.replace(role=role)) as hk:
                    r, l = ev.tf(it, b, logits=True)
                    rows.append({"sid": it["sid"], "role": role, "bitwise_W0": bool(torch.equal(l, l0)), "hook_calls": hk.calls,
                                 "fired": r["fired"], "differs_from_WE": not bool(torch.equal(l, lE))})
    out = {"n": len(rows), "n_bitwise_W0": sum(x["bitwise_W0"] for x in rows), "n_hook_called": sum(x["hook_calls"] >= 1 for x in rows),
           "n_fired": sum(bool(x["fired"]) for x in rows), "n_differs_from_WE": sum(x["differs_from_WE"] for x in rows), "rows": rows}
    out["pass"] = bool(out["n_bitwise_W0"] == out["n"] == out["n_hook_called"] and out["n_fired"] == 0)
    return out


# ----------------------------------------------------------------------------- (iii) (iv) (v)
def regression_gates(ev, log):
    H = ev.H
    cal = jload(os.path.join(P10, H.name, "calib.json"))
    S = ScopedLog.from_log(log, W0=H.W0)
    tau_abs = torch.tensor(cal["tau_self"], dtype=torch.float64)                 # P10's absolute thresholds on c_i
    r1 = S.replace(det_tau=torch.full((log.T,), 0.35, dtype=torch.float64))
    pool1, rec = CA.calibrate_log_only(S, 1.0)
    g = {"P10-HARD-SELF-noE": S.replace(role="all", fire="all", response="lin", det_tau=tau_abs / log.own()),
         "R1-REL0.35-REPL": r1, "R1-REL0.35-GAIN": r1.replace(response="gain"), "R1-POOL1-REPL": pool1,
         "NDB-R1-0.95": BL.cosine_gate(log, 0.95, role="last", W0=H.W0)}
    p11cfg = jload(os.path.join(P11, H.name, "C", "R1-POOL1-REPL.json"))["config"]
    info = {"pool1": rec, "p11_pool1_theta": p11cfg["tau_info"].get("theta"),
            "p10_s_vs_stored_s_max_abs": float((torch.tensor(cal["s"], dtype=torch.float64) - log.own()).abs().max())}
    return g, info


def their_flags(name):
    """{config: {sid: (exact, fired)}} from the exploratory experiments' per-row JSON files."""
    out = {"P10-HARD-SELF-noE": {r["sid"]: (bool(r["em"]), r["n_fire_pos"] > 0)
                                 for r in jload(os.path.join(P10, name, "E", "HARD-SELF-noE.json"))["rows"]}}
    meta = jload(os.path.join(P11, name, "C", "_meta_main.json"))["meta"]
    for c in REG[1:]:
        rows = jload(os.path.join(P11, name, "C", c + ".json"))["rows"]
        assert len(rows) == len(meta)
        out[c] = {m["sid"]: (bool(r["em"]), r["nfp"] > 0) for m, r in zip(meta, rows)}
    return out


def independent_firing(g, x, lp):
    """Firing of a scoped log on ONE captured sequence x (Tn, d), from the definition, without the package's gate code:
    -> (positions, candidate matrix, top detector) or None."""
    xd = x.double()
    pos = torch.arange(xd.shape[0], device=x.device)
    elig = {"last": pos == lp, "last+answer": pos >= lp, "all": torch.ones_like(pos, dtype=torch.bool)}[g.role]
    k = xd[elig]
    r = (k @ g.det_w.T) / g.det_s
    if bool((g.det_kind == 1).all()):
        r = r / k.norm(dim=1, keepdim=True)
    elif bool((g.det_kind != 0).any()):
        raise NotImplementedError
    cand = r >= g.eta * g.det_tau
    keep = cand.any(1)
    if not bool(keep.any()):
        return None
    top = torch.where(cand, r, torch.full_like(r, float("-inf"))).argmax(1)
    return pos[elig][keep], cand[keep], top[keep]


def same_firing(a, b):
    if a is None or b is None:
        return a is None and b is None
    return bool(torch.equal(a.idx, b.idx) and torch.equal(a.cand, b.cand) and torch.equal(a.top, b.top) and torch.equal(a.pos, b.pos))


def _numbers(flags):
    """flags: {sid: (exact, fired)} -> Rel / T-Gen / I-Gen exact %, probe D1 %, hard-negative D1 %, firing %."""
    def pct(v):
        return None if not v else float(np.mean(v) * 100)
    ed = {m: pct([e for s, (e, f) in flags.items() if s.startswith(f"edit:{m}:")]) for m in ("rel", "tgen", "igen")}
    pr = [(e, f) for s, (e, f) in flags.items() if s.startswith("probe:")]
    hn = [(e, f) for s, (e, f) in flags.items() if s.startswith("HN-")]
    return {"rel": ed["rel"], "tgen": ed["tgen"], "igen": ed["igen"], "probe_D1": pct([not e for e, f in pr]), "n_probe": len(pr),
            "probe_fire": pct([f for e, f in pr]), "hn_D1": pct([not e for e, f in hn]), "n_hn": len(hn), "hn_fire": pct([f for e, f in hn])}


def b345_pass(ev, log, out_dir, limit=0, log_fn=print, tag=None):
    """tag: fingerprint of the code; a partial state written by other code is discarded (never mixed into the rows)."""
    H = ev.H
    gates, info = regression_gates(ev, log)
    gates = {k: ev._prepare(g) for k, g in gates.items()}
    items = ev.edit_items() + ev.probe_items("all") + ev.hn_items()
    if limit:
        items = items[::limit]
    sids = [it["sid"] for it in items]
    part = os.path.join(out_dir, f"_partial_pass_{H.name}{'_lim%d' % limit if limit else ''}.json")
    st = jload(part) if os.path.exists(part) else None
    if st is None or st["sids"] != sids or st.get("tag") != tag:
        st = {"sids": sids, "tag": tag, "n_done": 0, "seconds": 0.0, "rows": {k: [] for k in gates}, "ref0": [],
              "iv": {k: {"n": 0, "equal": 0, "mismatch": []} for k in gates},
              "iii": {k: {kind: {"n_nonfiring": 0, "bitwise_W0": 0, "n_firing": 0, "firing_bitwise_W0": 0} for kind in ("edit", "probe", "hn")}
                      for k in gates}}
    t0 = time.time()
    with torch.no_grad(), ev.hz.HE.vision_cache(H):
        for n in range(st["n_done"], len(items)):
            it = items[n]
            ev.trim_vcache()
            b = ev.batch(it)
            with ev.applied(H.W0):                                      # plain full forward at the pre-edit weight, no hook
                r0, l0 = ev.tf(it, b, logits=True)
            st["ref0"].append({"sid": it["sid"], "exact": r0["exact"]})
            x, lp = ev.capture(it, batch=b)
            lpt = torch.tensor([lp], device=x.device)
            for k, g in gates.items():
                fp = g.firing(x[None], lpt, 0, {})                       # predicted from the captured keys (package gate)
                fi = independent_firing(g, x, lp)                        # ... and from the definition (test code)
                with ev.applied(g) as hk:
                    r, l = ev.tf(it, b, logits=True)
                    fo = hk.last                                         # observed inside the hook
                c = st["iv"][k]
                c["n"] += 1
                ok = same_firing(fp, fo) and ((fi is None and fo is None) or (fi is not None and fo is not None and torch.equal(fi[0], fo.pos)
                                                                                   and torch.equal(fi[1], fo.cand) and torch.equal(fi[2], fo.top)))
                c["equal"] += int(ok)
                if not ok:
                    c["mismatch"].append(it["sid"])
                c3 = st["iii"][k][it["kind"]]
                bw = bool(torch.equal(l, l0))
                if r["fired"]:
                    c3["n_firing"] += 1
                    c3["firing_bitwise_W0"] += int(bw)
                else:
                    c3["n_nonfiring"] += 1
                    c3["bitwise_W0"] += int(bw)
                st["rows"][k].append({a: v for a, v in r.items() if a not in ("uid", "group")})
            st["n_done"] = n + 1
            if (n + 1) % 100 == 0 or n + 1 == len(items):
                st["seconds"] += time.time() - t0
                t0 = time.time()
                jdump(st, part)
                log_fn(f"    pass {n + 1}/{len(items)} ({st['seconds']:.0f}s)")
    theirs = their_flags(H.name)
    res = {"n_items": len(items), "limit": limit, "seconds": st["seconds"], "info": info, "iii": st["iii"], "iv": st["iv"], "regression": {}}
    w0 = {r["sid"]: r["exact"] for r in st["ref0"]}
    res["W0_reference"] = {"probe_rows_reproduced": [sum(v for s, v in w0.items() if s.startswith("probe:")), sum(s.startswith("probe:") for s in w0)],
                           "hn_rows_reproduced": [sum(v for s, v in w0.items() if s.startswith("HN-")), sum(s.startswith("HN-") for s in w0)]}
    for k in gates:
        mine = {r["sid"]: (r["exact"], bool(r["fired"])) for r in st["rows"][k]}
        th = {s: v for s, v in theirs[k].items() if s in mine}
        common = {s: mine[s] for s in th}
        agree = {}
        for kind, pre in (("edit", "edit:"), ("probe", "probe:"), ("hn", "HN-")):
            ss = [s for s in th if s.startswith(pre)]
            agree[kind] = {"n": len(ss), "exact_equal": sum(mine[s][0] == th[s][0] for s in ss), "fired_equal": sum(mine[s][1] == th[s][1] for s in ss),
                           "exact_diff_sids": [s for s in ss if mine[s][0] != th[s][0]][:20], "fired_diff_sids": [s for s in ss if mine[s][1] != th[s][1]][:20]}
        res["regression"][k] = {"mine": _numbers(mine), "mine_on_their_rows": _numbers(common), "theirs": _numbers(th), "agreement": agree,
                                "n_rows_mine": len(mine), "n_rows_theirs_in_common": len(th), "n_rows_theirs": len(theirs[k])}
    res["rows"] = st["rows"]
    n3 = sum(c["n_nonfiring"] for k in gates for c in st["iii"][k].values())
    b3 = sum(c["bitwise_W0"] for k in gates for c in st["iii"][k].values())
    res["iii_total"] = {"n_nonfiring": n3, "bitwise_W0": b3,
                        "probe_rows_by_config": {k: [st["iii"][k]["probe"]["bitwise_W0"], st["iii"][k]["probe"]["n_nonfiring"]] for k in gates}}
    res["iv_total"] = {"n": sum(c["n"] for c in st["iv"].values()), "equal": sum(c["equal"] for c in st["iv"].values())}
    flags_equal = all(a["exact_equal"] == a["n"] and a["fired_equal"] == a["n"] for r in res["regression"].values() for a in r["agreement"].values())
    res["v_all_flags_equal"] = bool(flags_equal)
    res["pass_iii"] = bool(n3 == b3 and n3 > 0)
    res["pass_iv"] = bool(res["iv_total"]["n"] == res["iv_total"]["equal"])
    res["pass_v"] = bool(flags_equal) if not limit else None
    res["pass"] = bool(res["pass_iii"] and res["pass_iv"] and (limit or flags_equal))
    os.remove(part)
    return res


# ----------------------------------------------------------------------------- (vi)
def b6_harness(ev, log, log_fn=print, smoke=False):
    hz, H = ev.hz, ev.H
    cs = hz.pkg.sample_complaints(H, n=10, seed=0)
    # REEDIT first (evaluate.reedit: z* per complaint at B = 1 with chunk=1, then one joint install; P12 Anomaly 2): its z*
    # optimisation (backward passes) is the memory peak of the suite; its solve state (d x d fp64 matrices on the solve
    # device) is dropped again before COMP builds its own.
    torch.cuda.reset_peak_memory_stats()
    reedit, rinfo = EV.reedit(H, cs)
    peak_reedit = torch.cuda.max_memory_allocated() / 2 ** 30
    for k in [k for k in H._setup if isinstance(k, tuple) and k[:1] == ("reedit",)]:
        H._setup.pop(k)
    gc.collect()
    torch.cuda.empty_cache()
    corrs = {"NONE": hz.pkg.repair("NONE", H, cs)[0],
             "COMP-adapter": hz.pkg.repair("COMP", H, cs, protect=("K_last",), query="answer", form="adapter")[0],
             "COMP-merged": hz.pkg.repair("COMP", H, cs, protect=("K_last",), query="answer", form="merged")[0],
             "REEDIT": reedit}
    torch.cuda.empty_cache()
    if smoke:
        corrs.pop("NONE")
        corrs.pop("COMP-merged")
    citems = [p for p in ev.probe_items("repair") if any(p["probe_idx"] == c["probe_idx"] and p["ptype"] == c["ptype"] for c in cs)]
    out = {"complaints": [c["group"] for c in cs], "methods": {}, "harness_FAST_EVAL": bool(hz.HE.FAST_EVAL),
           "reedit": dict(rinfo, n_installed=len(rinfo["installed_rows"]), n_not_installed=len(rinfo["not_installed_rows"]), peak_gb_during_repair=peak_reedit),
           "what": "harness.evaluate (unmodified, its default code path) vs Evaluator.edits / probes(pool='heldout') / complaint rows (B = 1 full forwards)"}
    for tag, corr in corrs.items():
        t0 = time.time()
        rec = hz.pkg.evaluate(H, corr, cs, d2=False, loc=False, per_item=True)
        methods = {"corr": corr}
        if tag == "NONE":
            methods["None"] = None                                      # the deployed log through the None path
        if corr.kind == "weight":
            methods["tensor"] = corr.W                                  # the same correction passed as a plain weight
        res = {}
        for mk, m in methods.items():
            e = ev.edits(m)
            h = ev.probes(m, pool="heldout")
            c = ev._each(m, citems, "tf")
            he = {mm: rec["edits"][mm]["em_list"] for mm in EV.EDIT_METRICS}
            n_e = sum(r["exact"] == he[r["metric"]][r["edit"]] for r in e)
            hh = {r["group"]: r["reproduced_after"] for r in rec["heldout_rows"]}
            n_h = sum(r["exact"] == hh[r["group"]] for r in h)
            hc = {(r["probe_idx"], r["ptype"]): r["D1_resolved"] for r in rec["complaints"]}
            n_c = sum(r["exact"] == hc[(r["probe_idx"], r["ptype"])] for r in c)
            res[mk] = {"edit_flags_equal": [n_e, len(e)], "heldout_flags_equal": [n_h, len(h)], "complaint_flags_equal": [n_c, len(c)],
                       "n_heldout_rows_harness": len(hh), "heldout_D1_rate": float(np.mean([r["D1"] for r in h]) * 100),
                       "rel_exact": float(np.mean([r["exact"] for r in e if r["metric"] == "rel"]) * 100),
                       "complaints_resolved": int(sum(r["exact"] for r in c))}
        out["methods"][tag] = dict(res, kind=corr.kind, fro=rec["fro"], harness_rel_exact=rec["edits"]["rel"]["exact"],
                                   harness_complaints_resolved=int(sum(r["D1_resolved"] for r in rec["complaints"])), seconds=time.time() - t0)
        log_fn(f"    harness {tag}: {res}")
    H._setup.pop("evalcache", None)              # the harness' CPU suffix states (several GB) are not needed any more
    out["pass"] = bool(all(v[0] == v[1] and v[1] > 0 for t in out["methods"].values() for mk, m in t.items() if isinstance(m, dict)
                           for kk, v in m.items() if kk.endswith("_equal")) and
                       all(m["n_heldout_rows_harness"] == m["heldout_flags_equal"][1] for t in out["methods"].values() for m in t.values() if isinstance(m, dict)))
    return out


# ----------------------------------------------------------------------------- decode pipeline
def b7_decode(ev, log, n=24, n_edit=15, log_fn=print):
    H = ev.H
    rng = np.random.default_rng(7)
    pick = lambda xs, k: [xs[int(i)] for i in sorted(rng.choice(len(xs), min(k, len(xs)), replace=False))]         # noqa: E731
    pv = ev.probe_items("all")
    probes = pick([p for p in pv if p["ptype"] == "image"], n // 2) + pick([p for p in pv if p["ptype"] == "text"], n // 2)
    pc = pick(ev.probe_items("all", valid_only=False, pre_correct=True), n)
    S_off = ScopedLog.from_log(log, role="all", tau=INF, W0=H.W0)
    S_r1 = ScopedLog.from_log(log, tau=0.35, W0=H.W0)
    row = lambda p: H.row[(p["probe_idx"], p["ptype"])]                                                                # noqa: E731
    out = {}
    t0 = time.time()
    r_off, r_w0, r_we = ev.generate(S_off, probes), ev.generate(H.W0, probes), ev.generate(None, probes)
    out["inactive_gate"] = {"n": len(probes), "y_equals_stored_y0": sum(bool(r["agree"]) for r in r_off), "pos_ok": sum(bool(r["pos_ok"]) for r in r_off),
                            "long_equals_stored_pre": sum(r["long_eq_pre"] for r in r_off), "n_fired": sum(bool(r["fired"]) for r in r_off),
                            "min_hook_calls": min(r["n_calls"] for r in r_off),
                            "equal_to_plain_W0_decode": sum(a["y"] == b["y"] and a["long"] == b["long"] for a, b in zip(r_off, r_w0))}
    out["deployed"] = {"n": len(probes), "long_equals_stored_post": sum(r["long"] == row(p)["post_long_text"] for r, p in zip(r_we, probes)),
                       "y_equals_stored_yE": sum(r["y"] == row(p)["yE_text"] for r, p in zip(r_we, probes))}
    log_fn(f"    decode probes done ({time.time() - t0:.0f}s): {out}")
    dd = ev._each({"WE": None, "off": S_off}, pc, "decode")             # the function behind Evaluator.d2, on a sample
    d_we, d_off = dd["WE"], dd["off"]
    out["d2"] = {"n_pre_correct": len(pc), "D2_under_WE": sum(r["D2"] for r in d_we), "stored_D2": sum(p["D2_WE"] for p in pc),
                 "D2_flags_equal_stored": sum(r["D2"] == p["D2_WE"] for r, p in zip(d_we, pc)), "D2_under_inactive_gate": sum(r["D2"] for r in d_off)}
    eds = ev.edit_items(("rel",), range(n_edit))
    res = {}
    for name, S in (("R1", S_r1), ("R2", S_r1.replace(role="last+answer"))):
        g = ev.generate(S, eds)
        tf = ev.edits(S, ("rel",), range(n_edit))
        res[name] = {"n": len(g), "eq": sum(r["eq"] for r in g), "contains": sum(r["contains"] for r in g), "tf_exact": sum(r["exact"] for r in tf),
                     "eq_equals_tf_exact": sum(a["eq"] == b["exact"] for a, b in zip(g, tf)), "n_fired": sum(bool(r["fired"]) for r in g),
                     "n_fired_last": sum(r["fired_last"] for r in g), "n_fired_before_last_prompt": sum(r["fired_before"] for r in g),
                     "n_fire_pos": [r["n_fire_pos"] for r in g], "n_calls": [r["n_calls"] for r in g], "n_fire_calls": [r["n_fire_calls"] for r in g],
                     "own_released_last": sum(bool(r["own_released_last"]) for r in g), "pos_ok": sum(bool(r["pos_ok"]) for r in g),
                     "n_calls_last": [r["n_calls_last"] for r in g], "n_fire_calls_last": [r["n_fire_calls_last"] for r in g],
                     "n_decodes_last_decision_not_constant": sum(0 < r["n_fire_calls_last"] < r["n_calls_last"] for r in g),
                     "degenerate": {k: sum(r["degenerate"][k] for r in g) for k in ("empty", "qmark", "The", "single_char", "repeat")}}
    out["free_generation_rel"] = res
    # information (not a pass criterion): is the key at the last prompt position the same in the teacher-forced sequence
    # (prompt + answer) and in the prompt-only sequence of the decode?  (repairs are computed on captured keys)
    kc = {"n": 0, "bitwise_equal": 0, "n_last_prompt_index_differs": 0, "n_same_index_key_differs": 0, "max_rel_key_diff": 0.0, "max_abs_r_diff": 0.0,
          "by_kind": {}}
    Wd, sd = log.W.to(H.dev), log.own().to(H.dev)
    with torch.no_grad():
        for it in probes + eds:
            (xa, la), (xp, lq) = ev.capture(it, answer=True), ev.capture(it, answer=False)
            ka, kp = xa[la].double(), xp[lq].double()
            same = bool(torch.equal(xa[la], xp[lq]))
            kind = it["kind"] + ":" + it.get("ptype", it.get("metric", ""))
            bk = kc["by_kind"].setdefault(kind, {"n": 0, "bitwise_equal": 0, "index_differs": 0})
            kc["n"] += 1
            bk["n"] += 1
            kc["bitwise_equal"] += int(la == lq and same)
            bk["bitwise_equal"] += int(la == lq and same)
            kc["n_last_prompt_index_differs"] += int(la != lq)
            bk["index_differs"] += int(la != lq)
            kc["n_same_index_key_differs"] += int(la == lq and not same)
            kc["max_rel_key_diff"] = max(kc["max_rel_key_diff"], float((ka - kp).norm() / ka.norm()))
            kc["max_abs_r_diff"] = max(kc["max_abs_r_diff"], float(((ka - kp) @ Wd.T / sd).abs().max()))
    out["key_teacher_forced_vs_prompt_only"] = kc
    out["seconds"] = time.time() - t0
    a, b, d, e = out["inactive_gate"], out["deployed"], out["d2"], res["R1"]
    out["pass"] = bool(a["y_equals_stored_y0"] == a["n"] == a["equal_to_plain_W0_decode"] and a["n_fired"] == 0 and a["min_hook_calls"] >= 1
                       and a["long_equals_stored_pre"] == a["n"]
                       and d["D2_flags_equal_stored"] == d["n_pre_correct"] and d["D2_under_inactive_gate"] == 0
                       and a["pos_ok"] == a["n"] and e["pos_ok"] == e["n"] and res["R2"]["pos_ok"] == e["n"]
                       and e["n_fired_last"] == e["n"] and all(x == 1 for x in e["n_fire_pos"]) and res["R2"]["n_fired_before_last_prompt"] == 0
                       and all(f <= 11 for f in e["n_fire_calls"]))
    return out


# ----------------------------------------------------------------------------- teacher-forced path vs decode path (review F1)
def _kind(it):
    return it["kind"] + (":" + it.get("ptype", it.get("metric", "")) if it["kind"] != "hn" else "")


def _n_distinct(keys):
    """Number of bitwise-distinct tensors in a short list."""
    reps = []
    for k in keys:
        if not any(torch.equal(k, r) for r in reps):
            reps.append(k)
    return len(reps)


def _decode_with_keys(ev, it, method):
    """Decode of the item under `method` (Evaluator.decode) + the module input at the prompt's last token in every FULL
    forward of that decode, recorded by an independent forward hook: (row, [(sequence length, key)], last index, prompt length)."""
    H = ev.H
    pb = ev.batch(it, answer=False)
    lq, n0 = int(pb["last_prompt_idx"][0]), int(pb["seq_len"][0])
    seen = []

    def rec(mod, args, out):
        if args[0].shape[-2] > lq:                                       # a full-sequence forward (a KV-cache step holds one token)
            seen.append((int(args[0].shape[-2]), args[0][0, lq].detach().clone()))
    h = H.mod.register_forward_hook(rec)
    try:
        with ev.applied(method):
            row = ev.decode(it)
    finally:
        h.remove()
    return row, seen, lq, n0


def b9_paths(ev, log, log_fn=print, n_c=30, tau=0.25, smoke=False):
    """Pre-specified criterion (fixed before the first run): PASS iff
       (1) lp_shift (tokenizer only) is defined for every sequence of the log, equals on EVERY sequence the shift given by the
           vendor's own row encoding with and without the answer (encode_row: the code that builds the batches) and, on every
           shifted sequence and 30 others, the difference of the last prompt indices of the two batches actually built, the
           prompt-only ids being a prefix of the teacher-forced ids;
       (2) with lp='prompt' the hook sits on the prompt-only index on every one of those sequences, and where lp_shift = 0
           the rows of lp='tf' and lp='prompt' are identical;
       (3) least repair of firing REPAIR complaints with the two DOCUMENTED protocols for decode metrics,
           (A) keys of both paths (answer='both') with margin = repair.DECODE_MARGIN, and
           (B) teacher-forced keys + the keys of every full forward of a decode (answer='all') with the default margin:
           under each, no repaired complaint fires afterwards, teacher-forced (either position) or in the decode, and the own
           keys keep firing;
       (4) Evaluator.decode_keys are bitwise the keys that the gate sees at the last prompt position in the full forwards
           of the real decodes (every forward of every sampled complaint).
       Reported, not criteria: |dr| between the two positions, firing / D1 at both positions on the shifted rows, the other
       repair variants (teacher-forced keys, default margin), the relative margin each complaint would have needed."""
    from rescope import repair as RP
    H = ev.H
    S = ev._prepare(ScopedLog.from_log(log, tau=tau, W0=H.W0))
    th = S.theta()
    tok = H.vlm.tok
    items = ev.edit_items() + ev.probe_items("all") + ev.hn_items()
    sh = {it["sid"]: ev.lp_shift(it) for it in items}
    out = {"tau": tau, "decode_margin": RP.DECODE_MARGIN}
    scan = {"n": len(items), "n_shifted": sum(v != 0 for v in sh.values()), "n_undefined": sum(v is None for v in sh.values()),
            "values": sorted({str(v) for v in sh.values()}), "by_kind": {}, "shifted_sids": [s for s, v in sh.items() if v != 0]}
    for it in items:
        d = scan["by_kind"].setdefault(_kind(it), {"n": 0, "n_shifted": 0})
        d["n"] += 1
        d["n_shifted"] += int(sh[it["sid"]] != 0)
    scan["shifted_edits"] = sorted({it["edit"] for it in items if it["kind"] == "edit" and sh[it["sid"]] != 0})
    enc_bad = []                                                        # EVERY sequence: the vendor's own row encoding (the code that builds
    for it in items:                                                    # the batches), with and without the answer; tokenizer only
        q, ptype, ans = ev._query(it)
        prompt, fmt, n_img = (q["m_loc_q"], "vqa", 1) if ptype == "image" else (q["loc"], H.prober.text_fmt, None)
        ea, ep = H.vlm.encode_row(prompt, ans, fmt, n_img), H.vlm.encode_row(prompt, None, fmt, n_img)
        if not (ea["ids"][:len(ep["ids"])] == ep["ids"] and (ea["answer_start"] - 1) - (len(ep["ids"]) - 1) == sh[it["sid"]]):
            enc_bad.append(it["sid"])
    scan["encode_row_equal"] = [len(items) - len(enc_bad), len(items)]
    scan["encode_row_differs"] = enc_bad[:20]
    out["scan"] = scan
    log_fn(f"    paths: scan {scan['n_shifted']}/{scan['n']} shifted {scan['by_kind']}")
    shifted = [it for it in items if sh[it["sid"]] != 0]
    sample = (shifted[:6] if smoke else shifted) + [it for it in seq30(ev) if sh[it["sid"]] == 0]
    # ---- (1) + (2): the two batches, the keys at both positions, the hook position
    rows = []
    with torch.no_grad(), ev.hz.HE.vision_cache(H):
        for it in sample:
            ev.trim_vcache()
            bt, bp = ev.batch(it), ev.batch(it, answer=False)
            it_, ip_ = bt["inputs"]["input_ids"][0], bp["inputs"]["input_ids"][0]
            lt, lq = int(bt["last_prompt_idx"][0]), int(bp["last_prompt_idx"][0])
            xt, xp = H.capture_inputs(bt), H.capture_inputs(bp)
            r_tf, r_tfp, r_po = S.scores(xt[lt][None])[0], S.scores(xt[lq][None])[0], S.scores(xp[lq][None])[0]
            with ev.applied(S) as hk:
                a = ev.tf(it, bt)
                la = int(hk.lp[0])
                b = ev.tf(it, bt, lp="prompt")
                lb = int(hk.lp[0])
            same = {k: v for k, v in a.items() if k != "lp"} == {k: v for k, v in b.items() if k != "lp"}
            rows.append({"sid": it["sid"], "kind": _kind(it), "lp_shift": sh[it["sid"]],
                         "ok_tokenizer": bool(torch.equal(it_[:ip_.shape[0]], ip_)) and lt - lq == sh[it["sid"]] and lq == ip_.shape[0] - 1,
                         "ok_capture": ev.capture(it, lp="prompt")[1] == lq and ev.capture(it)[1] == lt,
                         "ok_hook": la == lt and lb == lq, "rows_identical": same,
                         "token_tf": tok.convert_ids_to_tokens(int(it_[lt])), "token_prompt": tok.convert_ids_to_tokens(int(it_[lq])),
                         "dr_tf_vs_decode_position": float((r_tf - r_po).abs().max()), "dr_same_token_two_sequences": float((r_tfp - r_po).abs().max()),
                         "fired_tf": bool(a["fired"]), "fired_prompt": bool(b["fired"]), "fires_decode_key": bool((r_po >= th).any()),
                         "exact_tf": a["exact"], "exact_prompt": b["exact"]})
    s1, s0 = [r for r in rows if r["lp_shift"] != 0], [r for r in rows if r["lp_shift"] == 0]
    mx = lambda rs, k: max([r[k] for r in rs], default=None)                                    # noqa: E731
    pr = [r for r in s1 if r["kind"].startswith(("probe", "hn"))]
    out["positions"] = {
        "n_checked": len(rows), "n_shifted_checked": len(s1), "ok_tokenizer": sum(r["ok_tokenizer"] for r in rows), "ok_capture": sum(r["ok_capture"] for r in rows),
        "ok_hook": sum(r["ok_hook"] for r in rows), "unshifted_rows_identical": [sum(r["rows_identical"] for r in s0), len(s0)],
        "tokens_shifted": sorted({(r["token_tf"], r["token_prompt"]) for r in s1}),
        "max_dr_shifted_tf_vs_decode_position": mx(s1, "dr_tf_vs_decode_position"), "max_dr_unshifted_tf_vs_decode_position": mx(s0, "dr_tf_vs_decode_position"),
        "max_dr_same_token_two_sequences": mx(rows, "dr_same_token_two_sequences"),
        "shifted_benign_rows": {"n": len(pr), "fired_tf": sum(r["fired_tf"] for r in pr), "fired_prompt": sum(r["fired_prompt"] for r in pr),
                                "fired_differs": sum(r["fired_tf"] != r["fired_prompt"] for r in pr),
                                "D1_tf": sum(not r["exact_tf"] for r in pr), "D1_prompt": sum(not r["exact_prompt"] for r in pr),
                                "D1_differs": sum(r["exact_tf"] != r["exact_prompt"] for r in pr)},
        "shifted_edit_rows": {"n": len(s1) - len(pr), "fired_tf": sum(r["fired_tf"] for r in s1 if r["kind"].startswith("edit")),
                              "fired_prompt": sum(r["fired_prompt"] for r in s1 if r["kind"].startswith("edit")),
                              "exact_tf": sum(r["exact_tf"] for r in s1 if r["kind"].startswith("edit")),
                              "exact_prompt": sum(r["exact_prompt"] for r in s1 if r["kind"].startswith("edit"))},
        "rows": rows}
    log_fn(f"    paths: positions {dict((k, v) for k, v in out['positions'].items() if k != 'rows')}")
    # ---- (3): least repair, end to end on both paths
    rep = ev.probe_items("repair")
    if smoke:
        rep = rep[::8]
        n_c = 6
    rows0 = ev.tf_rows(S, rep)
    fire_tf = {it["sid"]: bool(r["fired"]) for it, r in zip(rep, rows0)}
    fire_po = {it["sid"]: ev.predict(it, S, answer=False) is not None for it in rep if sh[it["sid"]] != 0}
    first = [it for it in rep if sh[it["sid"]] != 0 and (fire_tf[it["sid"]] or fire_po[it["sid"]])]        # every shifted row that fires on a path
    rest = [it for it in rep if sh[it["sid"]] == 0 and fire_tf[it["sid"]]]
    rng = np.random.default_rng(9)
    k = max(n_c - len(first), min(12, len(rest)))
    comp = first + [rest[int(i)] for i in sorted(rng.choice(len(rest), min(k, len(rest)), replace=False))]
    k_tf = [ev.eligible_keys(it, S) for it in comp]
    k_po = [ev.eligible_keys(it, S, answer=False) for it in comp]
    k_both = [ev.eligible_keys(it, S, answer="both") for it in comp]
    k_dec = [ev.eligible_keys(it, S, answer="decode") for it in comp]
    k_all = [ev.eligible_keys(it, S, answer="all") for it in comp]
    lr = {"n_repair_rows": len(rep), "n_firing_tf": sum(fire_tf.values()), "n_shifted_repair_rows": len(fire_po),
          "n_shifted_firing_tf": sum(fire_tf[s] for s in fire_po), "n_shifted_firing_decode_key": sum(fire_po.values()),
          "n_complaints": len(comp), "n_complaints_shifted": len(first), "complaints": [it["sid"] for it in comp],
          "both_is_cat_of_the_two_paths": all(torch.equal(c, torch.cat([a, b])) for a, b, c in zip(k_tf, k_po, k_both)),
          "all_is_cat_of_tf_and_decode": all(torch.equal(c[:1], a) and torch.equal(c[-b.shape[0]:], b) and c.shape[0] == 1 + int(sh[it["sid"]] != 0) + b.shape[0]
                                             for it, a, b, c in zip(comp, k_tf, k_dec, k_all)),
          "decode_keys_first_is_prompt_only_key": sum(bool(torch.equal(d[0], p[0])) for d, p in zip(k_dec, k_po))}
    # the decode before any repair + the keys the gate sees at the last prompt position in every full forward of it
    dec0, need, n_fw, n_fw_bitwise = [], [], 0, 0
    with torch.no_grad(), ev.hz.HE.vision_cache(H):
        for it, ka, kp, kd in zip(comp, k_tf, k_po, k_dec):
            ev.trim_vcache()
            row, seen, lq, n0 = _decode_with_keys(ev, it, S)
            dec0.append(row)
            for n_tok, key in seen:                                      # decode_keys must be the very keys of the decode's forwards
                n_fw += 1
                n_fw_bitwise += int(0 <= n_tok - n0 < kd.shape[0] and bool(torch.equal(key, kd[n_tok - n0])))
            ks = torch.stack([key for _, key in seen])
            rr = S.scores(ks).max(0).values                              # largest coordinate of every detector over the decode's forwards
            hot = rr >= th                                               # detectors that fire somewhere in the decode
            m_tf, m_both, m_all = (S.scores(x).max(0).values for x in (ka, torch.cat([ka, kp]), torch.cat([ka, kd])))

            def rel(m):                                                  # relative margin above the given keys that the decode needs
                if not bool(hot.any()):
                    return None
                if bool((m[hot] <= 0).any()):                            # (a non-positive coordinate: no relative margin covers it)
                    return float("inf")
                return float((rr[hot] / m[hot] - 1).max())
            need.append({"sid": it["sid"], "lp_shift": sh[it["sid"]], "n_forwards": len(seen), "n_detectors_firing_in_decode": int(hot.sum()),
                         "margin_needed_tf_keys": rel(m_tf), "margin_needed_both_paths": rel(m_both), "margin_needed_all": rel(m_all),
                         "n_distinct_keys_in_decode": _n_distinct([key for _, key in seen]),
                         "max_abs_dr_decode_vs_prompt_only_capture": float((S.scores(ks) - S.scores(kp)).abs().max())})

    def dist(key, shifted):
        v = [x[key] for x in need if x[key] is not None and (x["lp_shift"] != 0) == shifted]
        fin = [x for x in v if np.isfinite(x)]
        return {"n": len(v), "n_unbounded": len(v) - len(fin), "max": max(fin, default=None), "median": float(np.median(fin)) if fin else None,
                "n_above_decode_margin": sum(x > RP.DECODE_MARGIN for x in v), "n_above_default_margin": sum(x > RP.MARGIN for x in v)}
    lr["before"] = {"decode_fired": sum(bool(r["fired"]) for r in dec0), "decode_agree_y0": sum(bool(r["agree"]) for r in dec0),
                    "decode_last_decision_not_constant": sum(0 < r["n_fire_calls_last"] < r["n_calls_last"] for r in dec0)}
    lr["decode_keys_bitwise"] = {"n_forwards": n_fw, "bitwise_equal_to_decode_keys": n_fw_bitwise,
                                 "max_distinct_keys_in_one_decode": max([x["n_distinct_keys_in_decode"] for x in need], default=None)}
    lr["margin_needed"] = {f"{k}/{'shifted' if s else 'unshifted'}": dist(f"margin_needed_{k}", s) for k in ("tf_keys", "both_paths", "all") for s in (False, True)}
    lr["margin_needed"].update(max_abs_dr_decode_vs_prompt_only_capture=max([x["max_abs_dr_decode_vs_prompt_only_capture"] for x in need], default=None),
                               per_complaint=need)
    log_fn(f"    paths: {len(comp)} complaints ({len(first)} shifted); before {lr['before']}; decode keys {lr['decode_keys_bitwise']}; "
           f"margin needed {dict((k, v) for k, v in lr['margin_needed'].items() if k != 'per_complaint')}")
    variants = {"tf_keys_default_margin": (k_tf, {}), "both_paths_default_margin": (k_both, {}),
                "tf_keys_decode_margin": (k_tf, {"margin": RP.DECODE_MARGIN}), "both_paths_decode_margin": (k_both, {"margin": RP.DECODE_MARGIN}),
                "all_default_margin": (k_all, {})}
    for name, (keys, kw) in variants.items():
        ses = RP.Session(S)
        recs = [ses.complain(kx, **kw) for kx in keys]
        ok = [i for i, r in enumerate(recs) if not r["refused"]]
        tf1, tfp, de1 = ev.tf_rows(ses.S, comp), ev.tf_rows(ses.S, comp, lp="prompt"), ev.generate(ses.S, comp)
        lr[name] = {"n_refused": len(comp) - len(ok), "n_raised_total": sum(r["n_raised"] for r in recs),
                    "tf_still_firing": sum(bool(tf1[i]["fired"]) for i in ok), "tf_prompt_position_still_firing": sum(bool(tfp[i]["fired"]) for i in ok),
                    "decode_still_firing": sum(bool(de1[i]["fired"]) for i in ok), "decode_still_firing_shifted": sum(bool(de1[i]["fired"]) for i in ok if sh[comp[i]["sid"]] != 0),
                    "decode_still_firing_sids": [comp[i]["sid"] for i in ok if de1[i]["fired"]],
                    "decode_fire_calls": [[de1[i]["n_fire_calls_last"], de1[i]["n_calls_last"]] for i in ok if de1[i]["fired"]],
                    "tf_D1_after": sum(tf1[i]["D1"] for i in ok), "decode_agree_y0_after": sum(bool(de1[i]["agree"]) for i in ok),
                    "own_keys_fire": bool(ses.S.guard_ok().all()), "n_thresholds_raised": int((ses.S.det_tau > S.det_tau).sum())}
        log_fn(f"    paths: {name}: {lr[name]}")
    out["least_repair"] = lr
    H.reset()
    out["module_weight_is_WE_after"] = bool(torch.equal(H.mod.weight.data, H.WE))
    out["forward_hooks_left"] = len(H.mod._forward_hooks)
    p, dk = out["positions"], lr["decode_keys_bitwise"]
    silent = lambda d: (d["tf_still_firing"] == 0 and d["tf_prompt_position_still_firing"] == 0 and d["decode_still_firing"] == 0       # noqa: E731
                        and d["own_keys_fire"] and len(comp) - d["n_refused"] > 0)
    out["pass_1"] = bool(scan["n_undefined"] == 0 and scan["encode_row_equal"][0] == scan["encode_row_equal"][1] == len(items)
                         and p["ok_tokenizer"] == p["n_checked"] == p["ok_capture"])
    out["pass_2"] = bool(p["ok_hook"] == p["n_checked"] and p["unshifted_rows_identical"][0] == p["unshifted_rows_identical"][1] > 0)
    out["pass_3A"] = bool(lr["both_is_cat_of_the_two_paths"] and silent(lr["both_paths_decode_margin"]))
    out["pass_3B"] = bool(lr["all_is_cat_of_tf_and_decode"] and silent(lr["all_default_margin"]))
    out["pass_4"] = bool(dk["n_forwards"] == dk["bitwise_equal_to_decode_keys"] > 0 and lr["decode_keys_first_is_prompt_only_key"] == len(comp))
    out["pass"] = bool(out["pass_1"] and out["pass_2"] and out["pass_3A"] and out["pass_3B"] and out["pass_4"]
                       and out["module_weight_is_WE_after"] and out["forward_hooks_left"] == 0)
    out["criterion"] = b9_paths.__doc__
    return out


# ----------------------------------------------------------------------------- own prompts at B = 1 (review F5)
def b10_own(ev, log, log_fn=print):
    """Pre-specified criterion: PASS iff, with the B = 1 Rel keys of the edits added as guards (before calibrating),
    (1) thresholds set AT the caps and (2) a conformal calibration whose caps bind fire every edit on its own Rel prompt
    through the hook (T / T); (3) add_guards refuses the same keys on the log capped at the stored keys alone, and the
    number it reports as not firing equals the number of own prompts that are silent through the hook;
    (4) decode path: with the keys of every full forward of a decode added as guards as well (own_keys(answer='all')) and
    thresholds at the caps, every sampled edit (every 4th, own sequence not shifted) is a candidate at the last prompt
    position in EVERY full forward of the free generation of its Rel prompt.
    (5) Evaluator contracts: hn_items() of an Evaluator built without hard negatives raises; the strict pool is a non-empty
    proper subset of the disjoint pool without any row whose question occurs in REPAIR; one query per query_id.
    Reported, not criteria: the B = 1 coordinates and cosines, Rel under each, the exact / cosine gates on own prompts, the
    same decodes with the teacher-forced B = 1 guards only."""
    from rescope import repair as RP
    H = ev.H
    T = log.T
    S0 = ScopedLog.from_log(log, W0=H.W0)                                # tau = -inf: every key fires
    keys, edits = ev.own_keys("rel")
    R = S0.key_scores(keys.double().cpu())
    r_own = R[torch.arange(T), torch.tensor(edits)].numpy()
    kd, Kn = keys.double().cpu(), log.K / log.K.norm(dim=1, keepdim=True)
    cos = ((kd / kd.norm(dim=1, keepdim=True)) * Kn[torch.tensor(edits)]).sum(1).numpy()
    out = {"n_edits": T, "edits_in_order": edits == list(range(T)),
           "r_own_B1": {"min": float(r_own.min()), "q05": float(np.quantile(r_own, 0.05)), "median": float(np.median(r_own)), "max": float(r_own.max()),
                        "n_below_1": int((r_own < 1).sum()), "n_below_0.99": int((r_own < 0.99).sum()), "n_below_0.95": int((r_own < 0.95).sum())},
           "cos_B1_vs_stored": {"min": float(cos.min()), "median": float(np.median(cos)), "n_below_0.999": int((cos < 0.999).sum()),
                                "edits_below_0.999": [int(i) for i in np.nonzero(cos < 0.999)[0]], "n_below_0.95": int((cos < 0.95).sum())}}
    S_cap = S0.replace(det_tau=CA.own_caps(S0))                           # thresholds AT the caps of the stored (B = 10) keys
    Sg, rg = RP.add_guards(S0, keys, edits)
    S_capg = Sg.replace(det_tau=CA.own_caps(Sg))
    try:
        RP.add_guards(S_cap, keys, edits)
        refused = False
    except ValueError:
        refused = True
    _, ra = RP.add_guards(S_cap, keys, edits, allow_nonfiring=True)
    big = np.full(20, 50.0)                                              # a conformal quantile so large that every cap binds
    Sc, rc = CA.calibrate_conformal(CA.calibrate_log_only(S0, 1.0)[0], big, 0.1)
    Scg, rcg = CA.calibrate_conformal(CA.calibrate_log_only(Sg, 1.0)[0], big, 0.1)
    methods = {"cap_stored_keys": S_cap, "cap_with_B1_guards": S_capg, "conformal_capped_stored_keys": Sc, "conformal_capped_with_B1_guards": Scg,
               "exact_gate": BL.exact_gate(log, W0=H.W0), "cosine_0.95": BL.cosine_gate(log, 0.95, W0=H.W0)}
    res = ev.edits(methods, ("rel",))
    out["rel"] = {k: {"n": len(v), "own_fired_last": sum(bool(r["own_fired_last"]) for r in v), "exact": sum(r["exact"] for r in v),
                      "silent_edits": [r["edit"] for r in v if not r["own_fired_last"]][:100]} for k, v in res.items()}
    out["add_guards"] = {"before_calibration": {"n_added": rg["n_added"], "n_firing_now": rg["n_firing_now"]}, "on_capped_log_refused": refused,
                         "on_capped_log_allow_nonfiring": {"n_added": ra["n_added"], "n_firing_now": ra["n_firing_now"], "n_not_firing_now": len(ra["not_firing_now"])},
                         "max_cap_stored": float(CA.own_caps(S0).max()), "min_cap_with_B1_guards": float(CA.own_caps(Sg).min())}
    out["conformal"] = {"stored": {"n_detectors_capped": rc["n_detectors_capped"], "guarantee": rc["guarantee"]},
                        "with_B1_guards": {"n_detectors_capped": rcg["n_detectors_capped"], "guarantee": rcg["guarantee"],
                                           "all_guards_fire": bool(Scg.guard_ok().all())}}
    # ---- decode path: the own prompt in EVERY full forward of a free generation (a decode re-evaluates the last prompt
    #      position in up to 11 forwards of growing length, each with a slightly different key)
    sub = [it for it in ev.edit_items(("rel",), range(0, T, 4)) if ev.lp_shift(it) == 0]
    ka, ea = ev.own_keys("rel", answer="all")
    Sa, raa = RP.add_guards(S0, ka, ea)
    S_capa = Sa.replace(det_tau=CA.own_caps(Sa))
    dec = {"n_guard_keys": int(ka.shape[0]), "n_firing_when_added": raa["n_firing_now"], "n_shifted_edits_without_decode_guards": sum(ev.lp_shift(it) != 0 for it in ev.edit_items(("rel",))),
           "min_cap": float(CA.own_caps(Sa).min())}
    with torch.no_grad(), ev.hz.HE.vision_cache(H):
        for name, Sx in (("cap_with_B1_guards", S_capg), ("cap_with_B1_and_decode_guards", S_capa)):
            Sx = ev._prepare(Sx)
            th = Sx.theta()
            c = {"n_edits": len(sub), "n_edits_own_candidate_in_every_forward": 0, "n_forwards": 0, "n_forwards_own_candidate": 0,
                 "hook_n_calls_last": 0, "hook_n_fire_calls_last": 0, "eq": 0, "min_r_over_threshold": None}
            for it in sub:
                ev.trim_vcache()
                row, seen, lq, n0 = _decode_with_keys(ev, it, Sx)
                r = Sx.scores(torch.stack([k for _, k in seen]))[:, it["edit"]]
                ok = r >= th[it["edit"]]
                c["n_forwards"] += len(seen)
                c["n_forwards_own_candidate"] += int(ok.sum())
                c["n_edits_own_candidate_in_every_forward"] += int(bool(ok.all()))
                c["hook_n_calls_last"] += row["n_calls_last"]
                c["hook_n_fire_calls_last"] += row["n_fire_calls_last"]
                c["eq"] += int(row["eq"])
                ratio = float((r / th[it["edit"]]).min())
                c["min_r_over_threshold"] = ratio if c["min_r_over_threshold"] is None else min(c["min_r_over_threshold"], ratio)
            dec[name] = c
    out["decode"] = dec
    # ---- small contracts of the Evaluator (review, STYLE): hard negatives that were not loaded are an error, never an empty
    #      table; the strict pool is a subset of the disjoint pool; identical queries share a query_id
    ev2 = EV.Evaluator(H, hard_negatives=False)
    try:
        ev2.hn_items()
        hn_raises = False
    except RuntimeError:
        hn_raises = True
    pd, ps = ev.probe_items("heldout_disjoint"), ev.probe_items("heldout_strict")
    qid = {}
    for r in H.rows:
        qid.setdefault(ev._ann[(r["probe_idx"], r["ptype"])]["query_id"], set()).add((r["ptype"], r["prompt"], r["image"]))
    out["evaluator"] = {"hn_items_without_hard_negatives_raises": hn_raises, "n_hard_negatives": len(ev.hn_items()),
                        "n_heldout_disjoint": len(pd), "n_heldout_strict": len(ps), "strict_subset_of_disjoint": {p["sid"] for p in ps} <= {p["sid"] for p in pd},
                        "strict_rows_question_in_repair": sum(p["question_in_repair"] for p in ps), "one_query_per_query_id": all(len(v) == 1 for v in qid.values())}
    rl, ag, da, evo = out["rel"], out["add_guards"], dec["cap_with_B1_and_decode_guards"], out["evaluator"]
    log_fn(f"    own: r {out['r_own_B1']}; cos {out['cos_B1_vs_stored']}; rel {dict((k, (v['own_fired_last'], v['exact'])) for k, v in rl.items())}; {ag}; decode {dec}")
    out["pass"] = bool(out["edits_in_order"] and rl["cap_with_B1_guards"]["own_fired_last"] == T and rl["conformal_capped_with_B1_guards"]["own_fired_last"] == T
                       and ag["before_calibration"] == {"n_added": T, "n_firing_now": T}
                       and (ag["on_capped_log_refused"] or ag["on_capped_log_allow_nonfiring"]["n_not_firing_now"] == 0)
                       and ag["on_capped_log_allow_nonfiring"]["n_firing_now"] == rl["cap_stored_keys"]["own_fired_last"]
                       and rc["n_detectors_capped"] == T == rcg["n_detectors_capped"] and out["conformal"]["with_B1_guards"]["all_guards_fire"]
                       and da["n_edits_own_candidate_in_every_forward"] == da["n_edits"] > 0 and da["n_forwards_own_candidate"] == da["n_forwards"]
                       and da["hook_n_fire_calls_last"] == da["hook_n_calls_last"] == da["n_forwards"] and dec["n_firing_when_added"] == dec["n_guard_keys"]
                       and evo["hn_items_without_hard_negatives_raises"] and evo["n_hard_negatives"] > 0 and evo["strict_subset_of_disjoint"]
                       and 0 < evo["n_heldout_strict"] < evo["n_heldout_disjoint"] and evo["strict_rows_question_in_repair"] == 0 and evo["one_query_per_query_id"])
    out["criterion"] = b10_own.__doc__
    return out


# ----------------------------------------------------------------------------- the documented example
def b8_usage(ev, log, log_fn=print):
    """Executes the fenced python example of docs/USAGE.md on a subsample of the items (same calls, fewer rows)."""
    import rescope
    H = ev.H
    doc = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "USAGE.md")
    code, on = [], False
    with open(doc) as f:
        for line in f:
            if line.startswith("```"):
                if on:
                    break
                on = line.startswith("```python")
                continue
            if on:
                code.append(line)

    class SmallEvaluator(rescope.Evaluator):
        def edit_items(self, metrics=EV.EDIT_METRICS, edits=None):
            return super().edit_items(metrics, edits)[::12]

        def probe_items(self, *a, **k):
            return super().probe_items(*a, **k)[::25]

        def hn_items(self):
            return super().hn_items()[::40]

    class SmallSplits:                                                  # splits.sample on the subsampled pools: at most what is there
        def __getattr__(self, name):
            return getattr(rescope.splits, name)

        @staticmethod
        def sample(items, n, seed):
            return rescope.splits.sample(items, min(n, len(items)), seed)

    ns = {}
    exec("from rescope import *", ns)
    ns.update(load=lambda *a, **k: H, Evaluator=SmallEvaluator, splits=SmallSplits())
    src = "".join(code)
    assert "from rescope import *" in src
    t0 = time.time()
    exec(compile(src.replace("from rescope import *", "pass"), doc, "exec"), ns)
    out, rows = json.loads(ns["rows_json"]), ns["rows"]
    names = sorted(ns["methods"])
    res = {"n_code_lines": sum(bool(x.strip()) for x in code), "seconds": time.time() - t0, "methods": names,
           "n_rows": {k: {m: len(v) for m, v in d.items()} for k, d in out.items()},
           "all_methods_in_every_output": all(sorted(d) == names for d in out.values()),
           "pre_edit_D1": sum(r["D1"] for r in rows["pre-edit"]), "deployed_D1_equals_stored": sum(r["D1"] == r["D1_WE"] for r in rows["deployed"]),
           "n_probe_rows": len(rows["deployed"]), "all_disjoint": all(r["disjoint"] for r in rows["deployed"]),
           "lp_prompt_rows": [sum(r["lp"] == "prompt" for r in ns["rows_p"]), len(ns["rows_p"])],
           "lp_prompt_rows_equal_where_no_shift": sum(a["exact"] == b["exact"] and a["fired"] == b["fired"]
                                                      for a, b in zip(rows["scoped"], ns["rows_p"]) if a["lp_shift"] == 0),
           "n_rows_without_shift": sum(a["lp_shift"] == 0 for a in rows["scoped"]),
           "guards": {"n": int(ns["ses"].S.guard_key.shape[0]), "n_firing": int(ns["ses"].S.guard_ok().sum())},
           "rollback_all_D1": sum(r["D1"] for r in out["probes"]["ROLLBACK_ALL"]), "rollback_all_D2": sum(r["D2"] for r in out["d2"]["ROLLBACK_ALL"]),
           "rollback_all_hn_D1": sum(r["D1"] for r in out["hard_negatives"]["ROLLBACK_ALL"]),
           "session": {"n_ops": len(ns["ses"].history), "ops": sorted({h["op"] for h in ns["ses"].history}), "n_benign": len(ns["ses"].benign),
                       "n_detectors": ns["ses"].S.D, "own_keys_fire": bool(ns["ses"].S.guard_ok()[: H.T].all())},
           "summary": {m: {k: rescope.summarise(out[k][m]) for k in ("probes", "edits", "hard_negatives", "d2")} for m in names}}
    log_fn(f"    usage example: {res['n_rows']}")
    res["pass"] = bool(res["all_methods_in_every_output"] and res["pre_edit_D1"] == 0 and res["deployed_D1_equals_stored"] == res["n_probe_rows"] > 0
                       and res["all_disjoint"] and res["rollback_all_D1"] == 0 and res["rollback_all_D2"] == 0 and res["rollback_all_hn_D1"] == 0
                       and res["lp_prompt_rows"][0] == res["lp_prompt_rows"][1] == res["n_probe_rows"]
                       and res["lp_prompt_rows_equal_where_no_shift"] == res["n_rows_without_shift"]
                       and res["session"]["own_keys_fire"] and all(min(v.values()) > 0 for v in res["n_rows"].values()))
    return res


# ----------------------------------------------------------------------------- (c)
def _gpu_load():
    """Utilisation / memory of the card this process runs on, and the number of compute processes on it (context for timings)."""
    import subprocess
    try:
        idx = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        q = subprocess.run(["nvidia-smi", "-i", idx, "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20).stdout.strip().split(",")
        uuid = subprocess.run(["nvidia-smi", "-i", idx, "--query-gpu=uuid", "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.split()
        return {"gpu": idx, "utilization_pct": float(q[0]), "memory_used_MiB": float(q[1]), "n_compute_processes": sum(a == uuid for a in apps)}
    except Exception as e:                                             # timing context only
        return {"error": repr(e)}


def c_timing(ev, log, reps=7, n_seq=30):
    """Per-forward cost of the hook against the plain W_0 forward (B = 1 full forwards of 30 sequences).
    The GPU is SHARED with other jobs, so two measurements are reported:
    (1) gate only: decision + response on the captured module input of each sequence, minimum of 10 runs (the work the
        hook adds; short, so the minimum is nearly contention-free), relative to the plain forward;
    (2) end to end: every sequence is timed under every mode back to back (paired), the mode order rotates, repetition 0
        is a warm-up; per-sequence minimum over the repetitions and the median of the paired ratios (noisy under load).
    Gate modes run with record=False (no evaluation bookkeeping), '+rec' modes with it."""
    from rescope.scoped import Hook
    H = ev.H
    items = seq30(ev)[:: max(1, 30 // n_seq)]
    batches = [ev.batch(it) for it in items]
    S = ScopedLog.from_log(log, tau=0.35, W0=H.W0)
    gates = {"last_fp64": S, "last_module": S.replace(precision="module"), "all_fp64": S.replace(role="all"),
             "all_module": S.replace(role="all", precision="module"), "all_fp32": S.replace(role="all", precision="fp32"),
             "all_fp64_silent": S.replace(role="all", det_tau=torch.full((log.T,), INF, dtype=torch.float64)),
             "all_lin_nothreshold_fp64": ScopedLog.from_log(log, role="all", fire="all", response="lin", W0=H.W0)}
    gates = {k: ev._prepare(g) for k, g in gates.items()}
    modes = [("plain_W0", None, False)] + [(k, g, False) for k, g in gates.items()] + [("last_fp64+rec", gates["last_fp64"], True),
                                                                                         ("all_fp64+rec", gates["all_fp64"], True)]
    names = [m[0] for m in modes]
    T = np.zeros((reps + 1, len(modes), len(batches)))
    fired = {k: [None] * len(batches) for k in names}
    with torch.no_grad(), H.at(H.W0):                                  # module weight W_0 throughout
        for rep in range(reps + 1):
            for bi, b in enumerate(batches):
                for o in range(len(modes)):
                    mi = (o + rep + bi) % len(modes)
                    name, g, rec = modes[mi]
                    hk = None if g is None else Hook(H.mod, g, set_weight=False, record=rec)
                    if hk is not None:
                        hk.__enter__()
                        hk.begin(b["last_prompt_idx"])
                    try:
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        H.vlm.forward(b, logits_to_keep="answer")
                        torch.cuda.synchronize()
                        T[rep, mi, bi] = time.perf_counter() - t0
                    finally:
                        if hk is not None:
                            hk.__exit__(None, None, None)
                    if hk is not None:
                        f = hk.last
                        fired[name][bi] = None if f is None else (f.idx.tolist(), f.top.tolist(), int(f.cand.sum()))
    best = T[1:].min(axis=0)                                           # (modes, sequences)
    ntok = np.array([int(b["seq_len"][0]) for b in batches])
    # the added work alone: gate decision + response on the captured module input (min of 10 runs per sequence)
    G = np.zeros((len(gates) + 1, len(batches)))
    with torch.no_grad():
        for bi, (it, b) in enumerate(zip(items, batches)):
            x, lp = ev.capture(it, batch=b)
            x, lpt = x[None], torch.tensor([lp], device=x.device)
            out0 = torch.nn.functional.linear(x, H.W0)
            for gi, g in enumerate([None] + list(gates.values())):
                ts = []
                for _ in range(10):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    if g is None:
                        torch.nn.functional.linear(x, H.W0)             # the edited module itself, for scale
                    else:
                        f = g.firing(x, lpt, 0, {})
                        if f is not None:
                            g.respond(x, out0, f)
                    torch.cuda.synchronize()
                    ts.append(time.perf_counter() - t0)
                G[gi, bi] = min(ts)
            del x, out0
    smi = _gpu_load()
    out = {"n_sequences": len(batches), "reps": reps, "mean_seq_len": float(ntok.mean()), "device": torch.cuda.get_device_name(0),
           "gpu_load_during_test": smi, "note": c_timing.__doc__, "module_linear_ms": float(G[0].mean() * 1e3), "modes": {}}
    for mi, k in enumerate(names):
        pr = (T[1:, mi] / T[1:, 0]).ravel()
        out["modes"][k] = {"ms_per_forward": float(best[mi].mean() * 1e3), "ratio_to_plain": float(best[mi].sum() / best[0].sum()),
                           "overhead_ms_per_forward": float((best[mi] - best[0]).mean() * 1e3),
                           "overhead_us_per_token": float((best[mi] - best[0]).sum() / ntok.sum() * 1e6),
                           "paired_ratio_median": float(np.median(pr)), "paired_ratio_q25_q75": [float(np.quantile(pr, 0.25)), float(np.quantile(pr, 0.75))],
                           "n_sequences_firing": None if k == "plain_W0" else sum(f is not None for f in fired[k])}
        if k in gates:
            gi = 1 + list(gates).index(k)
            out["modes"][k].update(gate_only_ms=float(G[gi].mean() * 1e3), gate_only_over_plain_forward=float(G[gi].sum() / best[0].sum()),
                                   gate_only_us_per_token=float(G[gi].sum() / ntok.sum() * 1e6))
    out["module_dtype_gate_same_decisions"] = {"last": sum(a == b for a, b in zip(fired["last_fp64"], fired["last_module"])),
                                               "all": sum(a == b for a, b in zip(fired["all_fp64"], fired["all_module"]))}
    out["fp32_gate_same_decisions_all"] = sum(a == b for a, b in zip(fired["all_fp64"], fired["all_fp32"]))
    out["module_dtype_gate_same_firing_positions"] = {
        "last": sum((a is None and b is None) or (a is not None and b is not None and a[0] == b[0]) for a, b in zip(fired["last_fp64"], fired["last_module"])),
        "all": sum((a is None and b is None) or (a is not None and b is not None and a[0] == b[0]) for a, b in zip(fired["all_fp64"], fired["all_module"]))}
    out["pass"] = bool(all(np.isfinite(v["ms_per_forward"]) for v in out["modes"].values()) and out["modes"]["all_fp64_silent"]["n_sequences_firing"] == 0
                       and out["modes"]["all_lin_nothreshold_fp64"]["n_sequences_firing"] == len(batches))
    return out
