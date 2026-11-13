"""GPU tests of the additions for the final experiments (run through run_tests.py gpu --log <name>; one JSON each in tests/out/).

  b11_dual     baseline DUAL in free generation (probes, hard negatives, Rel / T-Gen / I-Gen prompts): the returned answer for
               the variants 'eq' / 'contains' (and 'eq' on the 10-token answer) against an independent re-implementation
               of the rule; two generations per query; refused in teacher-forced functions.
  b12_tokloc   token-level locality at B = 1: per-row values against an independent recomputation (argmax under the method
               against the argmax under W_0, same sequence), rows with an invalid y0, T-Loc* / I-Loc*, and the deployed log
               against its stored benchmark locality (difference reported).
  b13_sparse   SparseEvaluator rows == Evaluator rows on ALL edit sequences, ALL valid probe rows and the hard negatives for
               two gates calibrated by the default two-block rule (ReScope and the cosine gate): local store, lp='prompt',
               and, if it exists, the per-backbone key cache of the final experiments (keys bitwise on a sample).
  b14_usage2   the second (and, with a key cache, the third) python example of docs/USAGE.md executes on a subsample.
  d1_scale     (run_tests.py scale) scale-safe use on the T = 2000 log in a fresh process: registries, lean loading, peak RAM,
               the refused harness calls, stored metrics against B = 1 forwards on a sample, lean == standard loader.
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch

from rescope import baselines as BL, calibrate as CA, evaluate as EV, keycache as KC, loglib as LL, repair as RP, splits as SP
from rescope.sparse import SparseEvaluator


def _pick(rng, xs, n):
    return [xs[int(i)] for i in sorted(rng.choice(len(xs), min(n, len(xs)), replace=False))]


def default_gate(ev, log, feature="ncoord", alpha=0.05, n=200, seed=0, guards=True):
    """The default method of the protocol on this log: feature gate (role 'last', top-1, REPL, rebased values), the B = 1 keys
    of the edits' own Rel prompts as guards, two-block conformal calibration on n benign REPAIR queries (seeded draw).
    -> (gate, record, the calibration items)."""
    S = BL.feature_gate(log, feature, W0=ev.H.W0)
    if guards:
        S, _ = RP.add_guards(S, *ev.own_keys("rel"))
    pool = ev.probe_items("repair")
    benign = SP.sample(pool, min(n, len(pool)), seed)
    S, rec = CA.calibrate_two_block(S, [ev.eligible_keys(it, S) for it in benign], alpha, seed=seed)
    return S, rec, benign


# ----------------------------------------------------------------------------- b11: DUAL
def b11_dual(ev, log, log_fn=print, n=12, n_edit=10, smoke=False):
    """Pre-specified criterion (fixed before the first run): PASS iff
       (1) every DUAL row equals, in every field of a decode row, the row of the model that the rule selects, the selection
           being recomputed here from the deployed model's row and the list of edit targets with the testbed's own functions
           (norm_answer equality of the first line / of the 10-token answer; contains_ref of the 48-token answer);
       (2) every DUAL row records two generations, and the five methods together decode each item exactly twice;
       (3) DUAL is refused by the teacher-forced functions (TypeError);
       (4) both branches occur for each variant (some item returns the deployed answer, some the pre-edit answer);
       (5) Evaluator.dual_tf_rows (the labelled teacher-forced composition) returns, for every third item, the teacher-forced row of
           the model that the generation rows say DUAL selected.
       Reported, not criteria: release rates per item type, eq / contains of the edit prompts, agreement with y0 on probes."""
    H, TB = ev.H, ev.TB
    if smoke:
        n, n_edit = 3, 3
    rng = np.random.default_rng(11)
    pr = ev.probe_items("heldout_disjoint")
    items = (_pick(rng, [p for p in pr if p["ptype"] == "image"], n) + _pick(rng, [p for p in pr if p["ptype"] == "text"], n)
             + (_pick(rng, ev.hn_items(), n) if ev._hn_error is None else [])
             + ev.edit_items(edits=[int(i) for i in sorted(rng.choice(H.T, n_edit, replace=False))]))
    methods = {"deployed": None, "pre-edit": H.W0, "dual_eq": ev.dual("eq"), "dual_contains": ev.dual("contains"), "dual_eq_y": ev.dual("eq", on="y")}
    calls = {"n": 0}
    orig = H.prober.decode_v2

    def counted(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)
    H.prober.decode_v2 = counted
    t0 = time.time()
    try:
        rows = ev.generate(methods, items)
    finally:
        del H.prober.decode_v2                                           # back to the class method
    targets = [d["alt"] for d in H.items]
    nt = [TB.norm_answer(t) for t in targets]
    book = ("fired", "dual", "n_generations", "n_new_tokens")
    out = {"n_items": len(items), "n_decodes": calls["n"], "seconds": time.time() - t0, "n_targets": len(targets),
           "n_targets_empty_after_normalisation": sum(not t for t in nt), "variants": {}}
    kind = lambda it: it["kind"] + ":" + str(it.get("ptype", it.get("metric", it.get("hn_type", ""))))        # noqa: E731
    for name, sel in (("dual_eq", lambda r: [i for i, t in enumerate(nt) if t and TB.norm_answer(r["line"]) == t]),
                      ("dual_contains", lambda r: [i for i, t in enumerate(targets) if TB.contains_ref(r["long"], t)]),
                      ("dual_eq_y", lambda r: [i for i, t in enumerate(nt) if t and TB.norm_answer(r["y"]) == t])):
        v = {"n": len(items), "rows_equal_selected_model": 0, "match_equal": 0, "matching_edits_equal": 0, "own_flag_equal": 0, "two_generations": 0,
             "n_released_deployed": 0, "n_returned_pre_edit": 0, "by_kind": {}, "differing": []}
        for it, rE, r0, rd in zip(items, rows["deployed"], rows["pre-edit"], rows[name]):
            m = sel(rE)
            want = rE if m else r0
            same = all(rd[k] == want[k] for k in want if k not in book) and set(rd) - set(book) == set(want) - set(book)
            v["rows_equal_selected_model"] += int(same)
            v["match_equal"] += int(rd["dual"]["match"] == bool(m) == rd["fired"] and rd["dual"]["source"] == ("deployed" if m else "pre-edit"))
            v["matching_edits_equal"] += int(rd["dual"]["n_matching_edits"] == len(m) and rd["dual"]["edits"] == m[:8])
            v["own_flag_equal"] += int(rd["dual"]["own"] == ((it["edit"] in m) if it["kind"] == "edit" else None))
            v["two_generations"] += int(rd["n_generations"] == 2 and rd["n_new_tokens"] == [rE["n_long"], r0["n_long"]])
            v["n_released_deployed"] += int(bool(m))
            v["n_returned_pre_edit"] += int(not m)
            if not same:
                v["differing"].append(it["sid"])
            b = v["by_kind"].setdefault(kind(it), {"n": 0, "released": 0, "own_released": 0, "eq": 0, "eq_deployed": 0, "eq_pre_edit": 0, "contains": 0,
                                                   "contains_deployed": 0, "agree_y0": 0, "agree_y0_deployed": 0, "degenerate": 0})
            b["n"] += 1
            b["released"] += int(bool(m))
            b["degenerate"] += int(any(rd["degenerate"].values()))
            if it["kind"] == "edit":
                b["own_released"] += int(it["edit"] in m)
                b["eq"] += int(rd["eq"])
                b["eq_deployed"] += int(rE["eq"])
                b["eq_pre_edit"] += int(r0["eq"])
                b["contains"] += int(rd["contains"])
                b["contains_deployed"] += int(rE["contains"])
            else:
                b["agree_y0"] += int(bool(rd["agree"]))
                b["agree_y0_deployed"] += int(bool(rE["agree"]))
        v["summary"] = EV.summarise(rows[name]).get("dual")
        out["variants"][name] = v
        log_fn(f"    dual {name}: " + str({k: x for k, x in v.items() if k != 'by_kind'}))
    # the teacher-forced composition (dual_tf_rows): the row of the model that DUAL selects, for D1 / token locality at B = 1
    sub = items[::3]
    dr = ev.dual_tf_rows(ev.dual("contains"), sub)
    ref = ev.tf_rows({"deployed": None, "pre-edit": H.W0}, sub)
    sel = [r["fired"] for r in rows["dual_contains"]][::3]
    comp = {"n": len(sub), "selection_equals_generation_rows": sum(r["fired"] == s for r, s in zip(dr, sel)),
            "rows_equal_selected_model": sum(all(r[k] == (ref["deployed"] if s else ref["pre-edit"])[i][k] for k in ref["deployed"][i] if k != "fired")
                                             for i, (r, s) in enumerate(zip(dr, sel))),
            "n_deployed_selected": sum(sel), "locality": EV.locality(dr), "locality_deployed": EV.locality(ref["deployed"]),
            "D1_probe_rows": [sum(r["D1"] for r in dr if r["kind"] == "probe"), sum(r["kind"] == "probe" for r in dr)],
            "D1_probe_rows_deployed": sum(r["D1"] for r in ref["deployed"] if r["kind"] == "probe")}
    out["tf_composition"] = comp
    log_fn(f"    dual tf composition: {comp}")
    refused = {}
    for nm, fn in (("edits", lambda: ev.edits(ev.dual(), ("rel",), [0])), ("probes_in_dict", lambda: ev.tf_rows({"a": None, "d": ev.dual()}, items[:1])),
                   ("applied", lambda: ev.applied(ev.dual()).__enter__())):
        try:
            fn()
            refused[nm] = False
        except TypeError:
            refused[nm] = True
    H.reset()
    out["refused_in_teacher_forcing"] = refused
    out["module_weight_is_WE_after"] = bool(torch.equal(H.mod.weight.data, H.WE))
    vs = out["variants"].values()
    out["pass"] = bool(all(v["rows_equal_selected_model"] == v["n"] == v["match_equal"] == v["matching_edits_equal"] == v["own_flag_equal"] == v["two_generations"]
                           for v in vs)
                       and out["n_decodes"] == 2 * len(items) and all(refused.values()) and out["module_weight_is_WE_after"]
                       and comp["selection_equals_generation_rows"] == comp["n"] == comp["rows_equal_selected_model"]
                       and all(v["n_released_deployed"] > 0 and v["n_returned_pre_edit"] > 0 for v in vs))
    out["criterion"] = b11_dual.__doc__
    return out


# ----------------------------------------------------------------------------- b12: token-level locality
def b12_tokloc(ev, log, log_fn=print, n_check=40, step=1, smoke=False):
    """Pre-specified criterion (fixed before the first run): PASS iff
       (1) on a seeded sample of valid probe rows and on every probe row with an invalid but scorable y0, for the deployed
           log, the pre-edit weight and the default gate, the row's tok_loc equals the share of answer positions at which
           the argmax under the method equals the argmax under W_0, both recomputed here by plain forwards;
       (2) under the pre-edit weight tok_loc = 1 on every row (also where token accuracy against y0 is below 1: invalid y0);
       (3) T-Loc* / I-Loc* equal the means of the per-row values of the text / image rows;
       (4) every row on which the gate does not fire has tok_loc = 1;
       (5) for valid rows tok_loc == token accuracy (the pre-edit argmax is y0);
       (6) Evaluator.token_locality(W_0, pool, valid_only=False) includes the rows with an invalid y0 and gives tok_loc = 1 on every row.
       Reported, not a criterion: the deployed log's T-Loc* / I-Loc* against its stored benchmark locality (signed difference;
       the two are different quantities: decoder-v2 answer at B = 1 on valid rows vs benchmark reference answer in padded
       batches on all 500 items)."""
    H = ev.H
    if smoke:
        n_check, step = 6, 25
    gate, rec, _ = default_gate(ev, log, n=200 if not smoke else 20)
    methods = {"deployed": None, "pre-edit": H.W0, "rescope": ev._prepare(gate)}
    items = ev.probe_items("all")[::step]
    t0 = time.time()
    rows = ev.tf_rows(methods, items)
    inv = []
    for it in ev.probe_items("all", valid_only=False):
        if not it["y0_valid"]:
            try:
                if int(ev.batch(it)["answer_mask"][0].sum()) > 0:
                    inv.append(it)
            except (RuntimeError, ValueError):
                pass
    inv = inv[: (4 if smoke else len(inv))]
    rows_inv = ev.tf_rows(methods, inv) if inv else {k: [] for k in methods}
    rng = np.random.default_rng(12)
    idx = sorted(set(int(i) for i in rng.choice(len(items), min(n_check, len(items)), replace=False)))
    chk = {k: {"n": 0, "equal": 0, "n_invalid": 0, "equal_invalid": 0, "max_abs_diff": 0.0} for k in methods}
    w0_reproduces_valid = 0
    with torch.no_grad(), ev.hz.HE.vision_cache(H):
        for src, its, rws in (("valid", [items[i] for i in idx], {k: [rows[k][i] for i in idx] for k in methods}), ("invalid", inv, rows_inv)):
            for j, it in enumerate(its):
                ev.trim_vcache()
                b = ev.batch(it)
                nt = int(b["answer_mask"][0].sum())
                with ev.applied(H.W0):
                    a0 = H.vlm.forward(b, logits_to_keep="answer").logits[0, :nt].argmax(-1)
                if src == "valid":
                    w0_reproduces_valid += int(bool((a0 == b["answer_ids"][0, :nt]).all()))
                for k, m in methods.items():
                    with ev.applied(m) as hk:
                        if hk is not None:
                            hk.begin(b["last_prompt_idx"])               # the gate sits where the evaluation puts it (lp = 'tf')
                        lg = H.vlm.forward(b, logits_to_keep="answer").logits[0, :nt]
                    ref = float((lg.argmax(-1) == a0).float().mean()) if bool(torch.isfinite(lg).all()) else 0.0
                    got = rws[k][j]["tok_loc"]
                    c = chk[k]
                    c["n" if src == "valid" else "n_invalid"] += 1
                    c["equal" if src == "valid" else "equal_invalid"] += int(got == ref)
                    c["max_abs_diff"] = max(c["max_abs_diff"], abs((got if got is not None else -1.0) - ref))
    out = {"n_rows": len(items), "n_invalid_scorable_rows": len(inv), "seconds": time.time() - t0, "recomputed": chk,
           "w0_reproduces_y0_on_sampled_valid_rows": [w0_reproduces_valid, len(idx)], "calibration": {k: rec[k] for k in ("eta", "n1", "n2", "rank", "n_edits_capped")},
           "methods": {}}
    for k in methods:
        loc = EV.locality(rows[k])
        own = {}
        for nm, pt in (("T-Loc*", "text"), ("I-Loc*", "image")):
            v = [r["tok_loc"] for r in rows[k] if r["ptype"] == pt]
            own[nm] = 100.0 * float(np.mean(v)) if v else None
        fired = [r for r in rows[k] if r.get("fired")]
        silent = [r for r in rows[k] if r.get("fired") is False]
        out["methods"][k] = {"locality": loc, "own_mean": own, "aggregate_equals_mean_of_rows": all(
            (loc[nm] is None and own[nm] is None) or abs(loc[nm] - own[nm]) < 1e-9 for nm in own),
            "tok_loc_equals_token_acc_on_valid_rows": sum(r["tok_loc"] == r["token_acc"] for r in rows[k]), "n": len(rows[k]),
            "n_fired": len(fired), "n_silent": len(silent), "silent_rows_tok_loc_1": sum(r["tok_loc"] == 1.0 for r in silent),
            "D1_rate": 100.0 * float(np.mean([r["D1"] for r in rows[k]])),
            "invalid_rows": {"n": len(rows_inv[k]), "tok_loc_1": sum(r["tok_loc"] == 1.0 for r in rows_inv[k]),
                             "token_acc_below_1": sum(r["token_acc"] < 1.0 for r in rows_inv[k]),
                             "mean_tok_loc": float(np.mean([r["tok_loc"] for r in rows_inv[k]])) if rows_inv[k] else None}}
    if step == 1:                                                       # the public function with the rows of an invalid y0 included
        full = ev.token_locality(H.W0, "heldout_strict", valid_only=False)
        nv = len(ev.probe_items("heldout_strict"))
        out["token_locality_valid_only_false"] = {"pool": "heldout_strict", "n_valid_rows": nv, "n_rows": len(full["per_row"]),
                                                  "n_invalid_rows_included": sum(not r["y0_valid"] for r in full["per_row"]),
                                                  "n_rows_without_scorable_answer": full["n_rows_without_scorable_answer"],
                                                  "T-Loc*": full["T-Loc*"], "I-Loc*": full["I-Loc*"],
                                                  "all_tok_loc_1_under_pre_edit_weight": all(r["tok_loc"] == 1.0 for r in full["per_row"])}
    if step == 1:                                                       # the deployed log against its stored benchmark locality
        tl = ev.token_locality(None, "all", rows=rows["deployed"])
        dj = ev.token_locality(None, "heldout_disjoint", rows=[r for r in rows["deployed"] if r["disjoint"]])
        out["deployed_vs_stored"] = {"pool_all": {k: tl[k] for k in tl if k != "per_row"},
                                     "pool_heldout_disjoint": {k: dj[k] for k in dj if k != "per_row"}, "n_per_row": len(tl["per_row"]),
                                     "gate_heldout_disjoint": EV.locality([r for r in rows["rescope"] if r["disjoint"]])}
        log_fn(f"    tokloc deployed vs stored: {out['deployed_vs_stored']['pool_all']}")
    H.reset()
    m = out["methods"]
    out["pass"] = bool(all(c["equal"] == c["n"] > 0 and c["equal_invalid"] == c["n_invalid"] for c in chk.values())
                       and w0_reproduces_valid == len(idx)
                       and m["pre-edit"]["locality"]["T-Loc*"] == 100.0 == m["pre-edit"]["locality"]["I-Loc*"]
                       and m["pre-edit"]["invalid_rows"]["tok_loc_1"] == len(inv)
                       and (not inv or m["pre-edit"]["invalid_rows"]["token_acc_below_1"] == len(inv))
                       and all(v["aggregate_equals_mean_of_rows"] and v["tok_loc_equals_token_acc_on_valid_rows"] == v["n"] for v in m.values())
                       and m["rescope"]["silent_rows_tok_loc_1"] == m["rescope"]["n_silent"] > 0
                       and (step != 1 or all(np.isfinite(v) for v in out["deployed_vs_stored"]["pool_all"]["diff"].values() if v is not None))
                       and (step != 1 or (out["token_locality_valid_only_false"]["all_tok_loc_1_under_pre_edit_weight"]
                                          and out["token_locality_valid_only_false"]["n_rows"] >= out["token_locality_valid_only_false"]["n_valid_rows"])))
    out["criterion"] = b12_tokloc.__doc__
    return out


# ----------------------------------------------------------------------------- b13: SparseEvaluator == Evaluator
def _cmp(a, b):
    """(number of equal rows, sids of differing rows) of two row lists (full dict equality: every field)."""
    eq = [x == y for x, y in zip(a, b)]
    return int(sum(eq)) if len(a) == len(b) else -1, [x["sid"] for x, e in zip(a, eq) if not e][:20]


def b13_sparse(ev, log, log_fn=print, step=1, smoke=False):
    """Pre-specified criterion (fixed before the first run): PASS iff, for the two gates of the default calibration (ReScope =
    normalised coordinate, and the cosine gate; two-block rule, alpha = 0.05, n = 200, own Rel prompts as guards),
       (1) the rows of SparseEvaluator.tf_rows (local store: keys and pre-edit outcomes captured on first use) equal the rows of
           Evaluator.tf_rows in EVERY field on ALL edit sequences (Rel, T-Gen, I-Gen of every edit), ALL valid probe rows and all
           hard negatives; the same for lp = 'prompt' on all probe rows (first gate);
       (2) the public per-type functions (edits / probes / hard_negatives) of the SparseEvaluator return those rows;
       (3) both kinds of row occur for each gate: rows served without the hook, and rows that fire;
       (4) a method that is not sparse (role 'all', the deployed log) gives the Evaluator's rows through the SparseEvaluator;
       (5) if a per-backbone key cache exists under the cache root when the test runs: its teacher-forced keys and pre-edit
           outcomes are bitwise / exactly those of a fresh capture on a seeded sample, the rows served from it equal the
           Evaluator's rows as in (1) for every item it holds (and every 40th other item), and a cache of the other backbone is
           refused; if the final hard-negative set exists: a seeded sample of it gives equal rows through both paths
           (reported as 'not available' otherwise);
       (6) probe rows with an invalid pre-edit answer give equal rows through both paths.
       Reported: forwards run by each path, sequences inside the decision band."""
    H = ev.H
    if smoke:
        step = 12
    n_cal = 200 if not smoke else 20
    g1, r1, _ = default_gate(ev, log, "ncoord", n=n_cal)
    g2, r2, _ = default_gate(ev, log, "cos", n=n_cal)
    gates = {"rescope": g1, "cosine": g2}
    groups = {"edit": ev.edit_items()[::step], "probe": ev.probe_items("all")[::step], "hn": (ev.hn_items()[::step] if ev._hn_error is None else [])}
    items = groups["edit"] + groups["probe"] + groups["hn"]
    out = {"n_items": {k: len(v) for k, v in groups.items()}, "calibration": {k: {a: r[a] for a in ("eta", "n1", "n2", "rank", "n_edits_capped", "guarantee",
                                                                                              "n_guards_released_by_own_edit_top1", "n_guards")}
                                                                        for k, r in (("rescope", r1), ("cosine", r2))}}
    t0 = time.time()
    rows_e = ev.tf_rows(gates, items)
    out["seconds_evaluator"] = time.time() - t0
    log_fn(f"    sparse: evaluator rows done ({out['seconds_evaluator']:.0f}s)")
    sp = SparseEvaluator(H, cache=None)
    t0 = time.time()
    rows_s = sp.tf_rows(gates, items)
    out["seconds_sparse_first_call"] = time.time() - t0
    st1 = dict(sp.stats)
    t0 = time.time()
    rows_s2 = sp.tf_rows(gates, items)                                  # second call: everything silent comes from the local store
    out["seconds_sparse_second_call"] = time.time() - t0
    st2 = {k: sp.stats[k] - st1[k] for k in st1}
    res = {}
    off = {"edit": 0, "probe": len(groups["edit"]), "hn": len(groups["edit"]) + len(groups["probe"])}
    for k in gates:
        r = {}
        for kind, its in groups.items():
            a, b, c = (x[k][off[kind]:off[kind] + len(its)] for x in (rows_e, rows_s, rows_s2))
            n_eq, bad = _cmp(a, b)
            r[kind] = {"n": len(its), "equal": n_eq, "equal_second_call": _cmp(a, c)[0], "differing": bad, "n_fired": sum(bool(x["fired"]) for x in a)}
        res[k] = r
    out["local"] = {"rows": res, "stats_first_call": st1, "stats_second_call": st2, "n_local_entries": len(sp._local)}
    log_fn(f"    sparse: local store {res}; first call {st1}; second call {st2}")
    # lp = 'prompt' (the decode path's token), all probe rows, first gate
    pe, ps = ev.tf_rows(g1, groups["probe"], lp="prompt"), sp.tf_rows(g1, groups["probe"], lp="prompt")
    out["lp_prompt"] = {"n": len(pe), "equal": _cmp(pe, ps)[0], "differing": _cmp(pe, ps)[1], "n_shifted": sum(bool(x["lp_shift"]) for x in pe),
                        "n_fired": sum(bool(x["fired"]) for x in pe)}
    # probe rows with an INVALID pre-edit answer (token locality from the pre-edit argmax): the same rows through both paths
    inv = []
    for it in ev.probe_items("all", valid_only=False):
        if not it["y0_valid"]:
            try:
                if int(ev.batch(it)["answer_mask"][0].sum()) > 0:
                    inv.append(it)
            except (RuntimeError, ValueError):
                pass
    inv = inv[:4] if smoke else inv
    ie, is_ = (ev.tf_rows(g1, inv), sp.tf_rows(g1, inv)) if inv else ([], [])
    out["invalid_y0_rows"] = {"n": len(inv), "equal": _cmp(ie, is_)[0], "n_fired": sum(bool(r["fired"]) for r in ie),
                              "tok_loc_1_on_silent_rows": sum(r["tok_loc"] == 1.0 for r in ie if not r["fired"])}
    # the public per-type functions
    if step == 1:
        pub = {"edits": _cmp(rows_e["rescope"][:len(groups["edit"])], sp.edits(g1))[0] == len(groups["edit"]),
               "probes_heldout_disjoint": sp.probes(g1, pool="heldout_disjoint") == [x for x in rows_e["rescope"][off["probe"]:off["hn"]] if x["disjoint"]]}
        if groups["hn"]:
            pub["hard_negatives"] = sp.hard_negatives(g2) == rows_e["cosine"][off["hn"]:]
    else:
        pub = {"edits_subset": sp.tf_rows(g1, groups["edit"]) == rows_e["rescope"][:len(groups["edit"])]}
    out["public_functions_equal"] = pub
    # methods that are not sparse take the Evaluator's own path
    other = {"role_all": ev._prepare(g1.replace(role="all")), "deployed": None, "rescope": g1}
    few = groups["edit"][:3] + groups["probe"][:3]
    out["not_sparse_methods_equal"] = sp.tf_rows(other, few) == ev.tf_rows(other, few)
    # decision band: how many sequences were run only because a coordinate was within the band of a threshold
    ents = [sp._entry(it, "tf", capture=False) for it in items]
    K = torch.stack([e["key"] for e in ents])
    band = {}
    for k, g in gates.items():
        gc = g if g.device.type == "cpu" else g.to("cpu")
        r = torch.cat([gc.scores(K[a:a + 512]) for a in range(0, K.shape[0], 512)])
        fired = (r >= gc.theta()[None, :]).any(1)
        near = torch.cat([KC.near_threshold(gc, K[a:a + 512], r[a:a + 512]) for a in range(0, K.shape[0], 512)])
        hook_fired = torch.tensor([bool(x["fired"]) for x in rows_e[k]])
        band[k] = {"n": int(K.shape[0]), "cpu_fired": int(fired.sum()), "hook_fired": int(hook_fired.sum()), "cpu_equals_hook": int((fired == hook_fired).sum()),
                   "in_band": int(near.sum()), "in_band_and_silent": int((near & ~fired).sum())}
    out["decision_band"] = band
    # the per-backbone key cache of the final experiments, if it exists now
    if KC.KeyCache.available(H.backbone):
        spc = SparseEvaluator(H, cache="auto")
        pos = [i for i, it in enumerate(items) if spc.cache_row(it) is not None]
        pos = sorted(set(pos) | set(range(0, len(items), 40)))         # every item the cache holds + every 40th item (local store)
        sub = [items[i] for i in pos]
        n_held = sum(spc.cache_row(it) is not None for it in sub)
        ver = spc.verify_cache(items, n=40 if not smoke else 6, seed=13)
        t0 = time.time()
        rows_c = spc.tf_rows(gates, sub)
        cres = {k: {"n": len(sub), "equal": _cmp([rows_e[k][i] for i in pos], rows_c[k])[0], "differing": _cmp([rows_e[k][i] for i in pos], rows_c[k])[1]}
                for k in gates}
        out["key_cache"] = {"available": True, "dir": spc.cache.dir, "is_final": spc.cache.is_final,
                            "complete": spc.cache.meta.get("complete"), "n_sequences": spc.cache.n,
                            "n_items_held": n_held, "n_items": len(items), "verify": ver, "rows": cres, "stats": dict(spc.stats), "seconds": time.time() - t0}
        ok_cache = bool(ver["key_bitwise"] == ver["n_checked"] == ver["pre_equal"] and ver["key_prompt_bitwise"] == ver["key_prompt_checked"]
                        and all(c["equal"] == c["n"] for c in cres.values()) and spc.stats["silent_from_cache"] > 0)
        # the final hard-negative set (existing + new rows, cross rows), if it exists for this backbone: a seeded sample through both paths
        if os.path.exists(os.path.join(EV.HN_DIR_FINAL, H.backbone, "rows.jsonl")):
            evf, spf = EV.Evaluator(H, hn_dir=EV.HN_DIR_FINAL), SparseEvaluator(H, cache=spc.cache, hn_dir=EV.HN_DIR_FINAL)
            hf = evf.hn_items()
            rng = np.random.default_rng(13)
            hs = [hf[int(i)] for i in sorted(rng.choice(len(hf), min(60 if not smoke else 6, len(hf)), replace=False))]
            a, b = evf.tf_rows(g1, hs), spf.tf_rows(g1, hs)
            types_ = {}
            for h in hf:
                types_[h["hn_type"]] = types_.get(h["hn_type"], 0) + 1
            out["final_hard_negatives"] = {"available": True, "n_valid_rows": len(hf), "types": types_, "n_sample": len(hs), "rows_equal": _cmp(a, b)[0],
                                           "sample_types": sorted({h["hn_type"] for h in hs}), "n_fired": sum(bool(r["fired"]) for r in a),
                                           "n_D1": sum(r["D1"] for r in a), "cross_rows_have_both_edits": all(0 <= h["edit_b"] < H.T for h in hf if "edit_b" in h),
                                           "n_cross_rows": sum("edit_b" in h for h in hf), "n_src_related_to_log": sum(bool(h.get("src_related_to_log")) for h in hf),
                                           "n_from_cache": spf.stats["silent_from_cache"]}
            ok_cache = ok_cache and out["final_hard_negatives"]["rows_equal"] == len(hs) and out["final_hard_negatives"]["cross_rows_have_both_edits"]
        else:
            out["final_hard_negatives"] = {"available": False}
        log_fn(f"    sparse: key cache {out['key_cache']}")
        other = {"qwen25vl": "llava", "llava": "qwen25vl"}[H.backbone]          # a cache of the other backbone (same key width) must be refused
        if KC.KeyCache.available(other):
            try:
                SparseEvaluator(H, cache=KC.KeyCache(other))
                out["key_cache"]["other_backbone_refused"] = False
            except ValueError:
                out["key_cache"]["other_backbone_refused"] = True
            ok_cache = ok_cache and out["key_cache"]["other_backbone_refused"]
    else:
        out["key_cache"] = {"available": False, "note": f"no key cache for {H.backbone} under {KC.CACHE_ROOT} when the test ran"}
        ok_cache = True
    H.reset()
    out["module_weight_is_WE_after"] = bool(torch.equal(H.mod.weight.data, H.WE))
    out["pass"] = bool(all(v["equal"] == v["n"] == v["equal_second_call"] for r in res.values() for v in r.values())
                       and all(r["edit"]["n_fired"] > 0 for r in res.values())
                       and st1["silent_from_local"] > 0 and st2["local_captures"] == 0 and st2["forwards"] < st1["forwards"]
                       and out["lp_prompt"]["equal"] == out["lp_prompt"]["n"] and all(pub.values()) and out["not_sparse_methods_equal"]
                       and out["invalid_y0_rows"]["equal"] == out["invalid_y0_rows"]["n"]
                       and ok_cache and out["module_weight_is_WE_after"])
    out["criterion"] = b13_sparse.__doc__
    return out


# ----------------------------------------------------------------------------- d1: scale-safe use (fresh process)
def d1_scale(name, device="cuda:0", mem_gb=11.0, log_fn=print, small="TB-Q200-ae-l2_3", limit_gb=40.0):
    """Pre-specified criterion (fixed before the first run): PASS iff, in ONE fresh process,
       (1) register_logs() reports the scale registry and the text registry as loaded, and tolerates the state of the part-2
           registry (its logs are registered if their directories are complete, listed as incomplete otherwise);
       (2) the T = 2000 log loads through rescope.load (lean), every load-time check of the harness holds (W_E == W_0 +
           stored composite, factors match the stream, rollback(empty) == W_E), and the peak resident memory of the process
           stays below 40 GB after loading AND at the end of the test;
       (3) harness.baseline, harness.evaluate, the harness evalcache and HLog.edit_keys are refused for this log;
       (4) the lean loader gives bitwise the tensors of the harness loader on a T = 200 log (W_0, W_E, C, U, W, stored keys,
           batches, checks), reusing the loaded model;
       (5) stored_metrics(name) equals the testbed files (edit flags, probe flags), and on a seeded sample B = 1 forwards of
           the deployed log reproduce the stored D1 flags of the probe rows;
       (6) the default gate can be built, calibrated and evaluated at T = 2000 with last-position keys only.
       Reported: peak RAM after each step, seconds, the stored metrics next to B = 1 numbers of the sample."""
    import json
    hz = EV.harness()
    out = {"name": name, "limit_gb": limit_gb, "rss": {"start": EV.peak_rss_gb()}}
    reg = EV.register_logs()
    by = {}
    for n, v in reg["logs"].items():
        by.setdefault(v["source"], []).append(n)
    out["registry"] = {"sources": reg["sources"], "by_source": by, "n_logs": len(reg["logs"]), "n_complete": sum(v["complete"] for v in reg["logs"].values())}
    t0 = time.time()
    H = EV.load(name, device=device, mem_gb=mem_gb)
    out["load"] = dict(H.load_info, checks=dict(H.checks), n_batches=len(H.batches), n_probe_rows=len(H.rows), dtype=str(H.dtype),
                       gpu_peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30,
                       # arithmetic, not a measurement: what the harness loader keeps in RAM (one fp32 delta per batch)
                       batch_deltas_gb_not_held=len(H.batches) * H.m_out * H.d_in * 4 / 2 ** 30)
    out["rss"]["after_load"] = EV.peak_rss_gb()
    log_fn(f"    scale: loaded {name} in {time.time() - t0:.0f}s, lean {H.load_info['lean']}, peak RSS {out['rss']['after_load']:.1f} GB, checks {H.checks}")
    ev = EV.Evaluator(H, hard_negatives=False)
    refused = {}
    for nm, fn in (("harness.baseline", lambda: hz.pkg.baseline(H)), ("evaluation.baseline", lambda: hz.HE.baseline(H)),
                   ("evaluation.evalcache", lambda: hz.HE.evalcache(H)), ("HLog.edit_keys", lambda: H.edit_keys()),
                   ("harness.evaluate", lambda: hz.pkg.evaluate(H, hz.pkg.repair("NONE", H, [])[0], []))):
        try:
            fn()
            refused[nm] = False
        except RuntimeError as e:
            refused[nm] = "is refused for a log with T" in str(e)
    out["refused"] = refused
    # stored metrics against the files and against B = 1 forwards on a sample
    sm = EV.stored_metrics(name)
    d = os.path.join(EV.EXP, "06_testbed_v3", name)
    with open(os.path.join(d, "per_edit_metrics.jsonl")) as f:
        pe = [json.loads(x) for x in f if x.strip()]
    with open(os.path.join(d, "metrics.json")) as f:
        met = json.load(f)
    st = {"T": sm["T"], "edits": sm["edits"], "locality": sm["locality"], "probes": sm["probes"],
          "rel_exact_equals_mean_of_per_edit_flags": bool(abs(sm["edits"]["rel"]["exact"] - 100.0 * np.mean([bool(r["post_rel_exact"]) for r in pe])) < 1e-9),
          "per_edit_flags_equal_file": all(a["rel_exact"] == b["post_rel_exact"] and a["tgen_exact"] == b["post_tgen_exact"] and a["uid"] == b["uid"]
                                           for a, b in zip(sm["per_edit"], pe)) and len(sm["per_edit"]) == len(pe) == H.T,
          "probe_flags_equal_rows": all(a["D1"] == bool(r["is_complaint_D1"]) and a["D2"] == bool(r["is_complaint_D2"]) and a["group"] == r["group"]
                                        for a, r in zip(sm["probe_rows"], H.rows)) and len(sm["probe_rows"]) == len(H.rows),
          "loc_equals_metrics_json": sm["locality"]["T-Loc"] == met["TLoc_probes"] and sm["locality"]["I-Loc"] == met["ILoc_probes"]}
    rng = np.random.default_rng(14)
    pr = ev.probe_items("heldout_disjoint")
    sample = _pick(rng, pr, 30)
    rows = ev.tf_rows({"deployed": None, "pre-edit": H.W0}, sample)
    st["sample"] = {"n": len(sample), "deployed_D1_equals_stored": sum(r["D1"] == r["D1_WE"] for r in rows["deployed"]),
                    "pre_edit_D1": sum(r["D1"] for r in rows["pre-edit"]), "deployed_D1": sum(r["D1"] for r in rows["deployed"])}
    eds = [int(i) for i in sorted(rng.choice(H.T, 20, replace=False))]
    er = ev.edits(None, ("rel",), eds)
    st["sample"]["rel_B1_exact"] = sum(r["exact"] for r in er)
    st["sample"]["rel_stored_exact_same_edits"] = sum(bool(sm["per_edit"][i]["rel_exact"]) for i in eds)
    st["sample"]["rel_B1_flags_equal_stored_vendor_pass"] = sum(r["exact"] == bool(sm["per_edit"][r["edit"]]["rel_exact"]) for r in er)
    out["stored"] = st
    out["rss"]["after_stored_and_sample"] = EV.peak_rss_gb()
    # the default gate at T = 2000, last-position keys only
    t0 = time.time()
    log = LL.from_harness(H)
    S = BL.feature_gate(log, "ncoord", W0=H.W0)
    keys, edits = ev.own_keys("rel", edits=eds)
    S, _ = RP.add_guards(S, keys, edits)
    benign = SP.sample(ev.probe_items("repair"), 60, 0)
    S, rec = CA.calibrate_two_block(S, [ev.eligible_keys(it, S) for it in benign], 0.05, seed=0)
    gr = ev.edits(S, ("rel", "tgen"), eds)
    gp = ev.tf_rows(S, sample)
    out["gate"] = {"T": S.T, "seconds": time.time() - t0, "calibration": {k: rec[k] for k in ("eta", "n1", "n2", "rank", "n_edits_capped", "eta_infinite", "guarantee")},
                   "rel_own_fired_last": sum(bool(r["own_fired_last"]) for r in gr if r["metric"] == "rel"), "n_rel": len(eds),
                   "rel_exact": sum(r["exact"] for r in gr if r["metric"] == "rel"), "tgen_exact": sum(r["exact"] for r in gr if r["metric"] == "tgen"),
                   "probe_fired": sum(bool(r["fired"]) for r in gp), "probe_D1": sum(r["D1"] for r in gp), "n_probe": len(gp),
                   "rebase_values_finite": bool(torch.isfinite(S.values).all()), "guards_fire": bool(S.guard_ok().all())}
    out["rss"]["after_gate"] = EV.peak_rss_gb()
    del S, log
    # lean loader == harness loader (T = 200 log, the loaded model is reused)
    cmp_ = {"log": small}
    try:
        H.set_weight(H.W0)
        Hs = hz.pkg.load(small, device=device, vlm=H.vlm)
        ref = {k: getattr(Hs, k).clone() for k in ("W0", "WE", "C", "U", "Wfac", "Kp_log")}
        ref_meta = (list(Hs.batches), dict(Hs.checks), Hs.T)
        a0 = Hs.A[3].clone()
        del Hs
        torch.cuda.empty_cache()
        H.set_weight(H.W0)
        Hl = EV.load(small, device=device, lean=True, keep_batch_deltas=True, reuse=H)
        cmp_.update({k: bool(torch.equal(getattr(Hl, k), v)) for k, v in ref.items()})
        cmp_.update(batches_equal=list(Hl.batches) == ref_meta[0], checks_equal=dict(Hl.checks) == ref_meta[1], T=Hl.T, checks=dict(Hl.checks),
                    lazy_batch_delta_equal=bool(torch.equal(Hl.A[3], a0)), n_lazy_deltas=len(Hl.A), reused_model=Hl.load_info["reused_model_of"])
        cmp_["rollback_one_batch_equal_formula"] = bool(torch.equal(Hl.rollback_weight(Hl.batches[3]),
                                                                    (Hl.W0.float() + (Hl.C - a0.to(Hl.dev))).to(Hl.dtype)))
        del Hl, ref
    except Exception as e:                                               # reported; the criterion then fails
        cmp_["error"] = f"{type(e).__name__}: {e}"
    out["lean_vs_harness_loader"] = cmp_
    out["rss"]["end"] = EV.peak_rss_gb()
    out["gpu_peak_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
    src = reg["sources"]
    ok4 = all(cmp_.get(k) is True for k in ("W0", "WE", "C", "U", "Wfac", "Kp_log", "batches_equal", "checks_equal", "lazy_batch_delta_equal",
                                               "rollback_one_batch_equal_formula")) and "error" not in cmp_
    out["pass"] = bool(src["scale"]["ok"] and src["text"]["ok"] and name in reg["logs"]
                       and out["load"]["lean"] and all(v for k, v in H.checks.items() if k != "collapsed_working_point")
                       and out["rss"]["after_load"] < limit_gb and out["rss"]["end"] < limit_gb
                       and all(v is True for v in refused.values()) and ok4
                       and st["rel_exact_equals_mean_of_per_edit_flags"] and st["per_edit_flags_equal_file"] and st["probe_flags_equal_rows"]
                       and st["loc_equals_metrics_json"] and st["sample"]["deployed_D1_equals_stored"] == st["sample"]["n"] and st["sample"]["pre_edit_D1"] == 0
                       and out["gate"]["rebase_values_finite"] and out["gate"]["guards_fire"] and out["gate"]["rel_own_fired_last"] == out["gate"]["n_rel"])
    out["criterion"] = d1_scale.__doc__
    return out


# ----------------------------------------------------------------------------- b14: the second documented example
def _python_blocks():
    """The fenced python blocks of docs/USAGE.md, in order."""
    doc = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "USAGE.md")
    blocks, cur, other = [], None, False
    with open(doc) as f:
        for line in f:
            if line.startswith("```"):
                if cur is not None:                                      # closing fence of a python block
                    blocks.append("".join(cur))
                    cur = None
                elif other:                                              # closing fence of another block
                    other = False
                elif line.startswith("```python"):
                    cur = []
                else:
                    other = True
                continue
            if cur is not None:
                cur.append(line)
    return doc, blocks


def b14_usage2(ev, log, log_fn=print):
    """Executes the SECOND fenced python example of docs/USAGE.md (the default method of the final experiments: registries,
    SparseEvaluator, two-block calibration of ReScope and of the cosine gate, shared-scale ablation, token locality, DUAL,
    stored metrics) on a subsample of the items, and the THIRD one (KeyCache, no model) if the backbone's key cache exists.
    PASS iff the code runs, its output is JSON-serialisable, every method has a row for every item, the deployed rows
    reproduce the stored D1 flags, the sparse rows of the calibrated gate equal the rows of a plain Evaluator, DUAL rows record
    two generations, and stored_metrics agrees with metrics.json."""
    import json

    import rescope
    H = ev.H
    doc, blocks = _python_blocks()

    class SmallSparse(rescope.SparseEvaluator):
        def edit_items(self, metrics=EV.EDIT_METRICS, edits=None):
            return super().edit_items(metrics, edits)[::12]

        def probe_items(self, *a, **k):
            return super().probe_items(*a, **k)[::8]

    class SmallSplits:                                                  # splits.sample on the subsampled pools: at most what is there
        def __getattr__(self, name):
            return getattr(rescope.splits, name)

        @staticmethod
        def sample(items, n, seed):
            return rescope.splits.sample(items, min(n, len(items)), seed)

    ns = {}
    exec("from rescope import *", ns)
    ns.update(load=lambda *a, **k: H, SparseEvaluator=SmallSparse, splits=SmallSplits())
    src = blocks[1]
    assert "calibrate_two_block" in src and "from rescope import *" in src
    t0 = time.time()
    exec(compile(src.replace("from rescope import *", "pass"), doc, "exec"), ns)
    rows, rec, rec_c, rec_a, gen, sp = ns["rows"], ns["rec"], ns["rec_c"], ns["rec_a"], ns["gen"], ns["ev"]
    names = sorted(ns["methods"])
    plain = rescope.Evaluator(H, hard_negatives=False)
    items = sp.probe_items("heldout_disjoint")
    with open(os.path.join(EV.EXP, "06_testbed_v3", H.name, "metrics.json")) as f:
        met = json.load(f)
    res = {"n_code_lines": sum(bool(x.strip()) for x in src.splitlines()), "seconds": time.time() - t0, "methods": names,
           "json_chars": len(json.dumps(ns["out2"])), "n_probe_rows": {k: len(v) for k, v in rows.items()}, "n_items": len(items),
           "deployed_D1_equals_stored": sum(r["D1"] == r["D1_WE"] for r in rows["deployed"]),
           "calibration": {k: {a: r[a] for a in ("op", "eta", "n1", "n2", "rank", "n_edits_capped", "bound_false_firing", "eta_infinite")}
                           for k, r in (("rescope", rec), ("cosine", rec_c), ("shared", rec_a))},
           "sparse_rows_equal_plain_evaluator": {k: plain.tf_rows(ns["methods"][k], items) == rows[k] for k in ("rescope", "cosine", "shared")},
           "sparse_stats": dict(sp.stats), "loc": ns["loc"], "deployed_token_locality": {k: v for k, v in ns["dep"].items() if k != "per_row"},
           "dual": {"n": len(gen["dual"]), "two_generations": sum(r["n_generations"] == 2 for r in gen["dual"]), "released": sum(r["fired"] for r in gen["dual"]),
                    "released_contains": sum(r["fired"] for r in gen["dual_contains"]), "summary": ns["out2"]["dual"]},
           "stored_rel_exact_equals_metrics_json": ns["stored"]["edits"]["rel"]["exact"] == met["Rel_exact"],
           "edits": ns["out2"]["edits"]}
    log_fn(f"    usage2: {res['n_probe_rows']}, calibration {res['calibration']}, sparse {res['sparse_stats']}")
    # the third block: the same gate from the key cache alone (if the cache of this backbone exists now)
    if KC.KeyCache.available(H.backbone) and len(blocks) > 2:
        src3 = blocks[2].replace('"qwen25vl"', repr(H.backbone)).replace('"TB-Q2000-ae-l2_3"', repr(H.name))
        ns3 = {}
        exec("from rescope import *", ns3)
        ns3.update(splits=SmallSplits())                                # a partial cache may hold fewer benign rows than the example draws
        try:
            exec(compile(src3, doc, "exec"), ns3)
            dec = ns3["dec"]
            res["key_cache_block"] = {"ran": True, "cache_dir": ns3["kc"].dir, "is_final": ns3["kc"].is_final, "complete": ns3["kc"].meta.get("complete"),
                                      "n_sequences": len(dec["rows"]), "n_fired": int(dec["fired"].sum()), "n_near": int(dec["near"].sum()),
                                      "n_own_top": int(dec["own_top"].sum()), "eta": ns3["rec"]["eta"], "n_edits_capped": ns3["rec"]["n_edits_capped"]}
        except LookupError as e:                                        # a partial cache that does not hold this log yet
            res["key_cache_block"] = {"ran": False, "note": f"{type(e).__name__}: {e}"}
    else:
        res["key_cache_block"] = {"ran": False, "note": "no key cache for this backbone when the test ran"}
    H.reset()
    res["pass"] = bool(all(n == res["n_items"] > 0 for n in res["n_probe_rows"].values()) and sorted(rows) == names
                       and res["deployed_D1_equals_stored"] == res["n_items"] and all(res["sparse_rows_equal_plain_evaluator"].values())
                       and res["dual"]["two_generations"] == res["dual"]["n"] > 0 and res["stored_rel_exact_equals_metrics_json"]
                       and res["calibration"]["shared"]["n1"] == 0 and res["calibration"]["rescope"]["op"] == "calibrate_two_block"
                       and res["sparse_stats"]["silent_from_local"] + res["sparse_stats"]["silent_from_cache"] > 0
                       and res["deployed_token_locality"]["diff"] is not None)
    return res
