"""Check 6 (E2 on prefix logs) + check 8 (e, i): my forwards on the 10- and 50-edit prefixes (deployed prefix weight rebuilt from the stored batch updates, one saved linear
repair, the thresholds-only repair rebuilt) against the experiment's rows; H1 recomputed by the registered rule from the per-row files; best linear repair selected by the two-sided rule."""
from vcommon import *
import gzip
def rd(p):
    with gzip.open(p, "rt") as f: return [json.loads(l) for l in f]
PRE = f"{FINAL}/e2_prefix/out"
LIN = {"COMP_B": "comp_b", "COMP_BP": "comp_bp", "REEDIT_B": "reedit_b", "RB_TOT": "rb_tot", "RB_UNION": "rb_union"}
D1 = lambda rs: 100 * float(np.mean([r["D1"] for r in rs if r["kind"] == "probe" and r["pool"] != "repair"]))
REL = lambda rs: 100 * float(np.mean([r["exact"] for r in rs if r["kind"] == "edit" and r["metric"] == "rel"]))
res = {"rescored": {}, "h1": {}}
# ---------------------------------------------------------------- my forwards
for par in (QDEV, LDEV):
    for k, lin in ((10, "REEDIT_B__n10__s0"), (50, "RB_UNION__n10__s0")):
        pn = f"{par}-p{k}"
        if not os.path.exists(f"{OUT}/g_pre_{pn}.json"): continue
        J = jload(f"{OUT}/g_pre_{pn}.json"); rows = rd(f"{OUT}/g_pre_{pn}_rows.jsonl.gz")
        ncomp = 10
        main, comp = rows[:-ncomp], rows[-ncomp:]
        pr = [r for r in main if r["kind"] == "probe"]; ed = [r for r in main if r["kind"] == "edit"]
        out = {"prefix_weight_equals_stored_composite": J["prefix_weight_equals_stored_composite"], "theta_max_abs_diff_to_experiment": J["theta_max_abs_diff"], "n_refused": J["n_refused"],
               "n_thresholds_raised": J["n_thresholds_raised"], "n_capped_logonly": J["n_capped_logonly"], "linear_config": lin, "n_heldout_disjoint": len(pr)}
        for m, cfg in (("deployed_prefix", "deployed"), ("linear", lin), ("where_least", "WHERE_LEAST__n10__s0")):
            E = {r["sid"]: r for r in rd(f"{PRE}/rows/{pn}__{cfg}.jsonl.gz")}
            mine = {"D1_heldout_disjoint": 100 * float(np.mean([r[m]["D1"] for r in pr])), **{mm: 100 * float(np.mean([r[m]["exact"] for r in ed if r["metric"] == mm])) for mm in ("rel", "tgen", "igen")},
                    "complaints_D1": int(sum(bool(r[m]["D1"]) for r in comp))}
            if m == "where_least":
                mine["fired_heldout_disjoint"] = 100 * float(np.mean([r[m]["fired"] for r in pr])); mine["complaints_released"] = int(sum(bool(r[m]["fired"]) for r in comp))
                mine["D1_on_rows_without_release"] = int(sum(bool(r[m]["D1"]) for r in pr if not r[m]["fired"]))
            ex = [E[r["sid"]] for r in main if r["sid"] in E]
            exp = {"D1_heldout_disjoint": D1(ex), "rel": REL(ex)}
            diff = {"exact": sum(r[m]["exact"] != E[r["sid"]]["exact"] for r in main if r["sid"] in E), "D1": sum(r[m]["D1"] != E[r["sid"]]["D1"] for r in pr),
                    "n_compared": sum(r["sid"] in E for r in main)}
            if m == "where_least": diff["fired"] = sum(bool(r[m]["fired"]) != bool(E[r["sid"]]["fired"]) for r in main if r["sid"] in E)
            out[m] = {"mine": mine, "experiment_rows": exp, "rows_differing": diff}
        d, w = out["deployed_prefix"]["mine"], out["where_least"]["mine"]
        out["h1_seed0_from_my_forwards"] = {"D1_at_most_10": w["D1_heldout_disjoint"] <= 10, "rel_diff": w["rel"] - d["rel"], "two_sided": abs(w["rel"] - d["rel"]) <= 2 and w["D1_heldout_disjoint"] <= 10,
                                            "one_sided": w["rel"] - d["rel"] >= -2 and w["D1_heldout_disjoint"] <= 10}
        res["rescored"][pn] = out
        print(pn, "comp eq", out["prefix_weight_equals_stored_composite"], "theta diff", out["theta_max_abs_diff_to_experiment"])
        for m in ("deployed_prefix", "linear", "where_least"): print("   ", m, {a: (round(b, 2) if isinstance(b, float) else b) for a, b in out[m]["mine"].items()}, "| exp", {a: round(b, 2) for a, b in out[m]["experiment_rows"].items()}, "| diff", out[m]["rows_differing"])
# ---------------------------------------------------------------- H1 on all six prefixes from the experiment's per-row files (3 seeds), registered rule; selection of the best linear repair
H = jload(f"{PRE}/h1_prefix.json")
for par in (QDEV, LDEV):
    for k in (10, 20, 50):
        pn = f"{par}-p{k}"
        R = lambda c: rd(f"{PRE}/rows/{pn}__{c}.jsonl.gz")
        dep = R("deployed"); d1d, reld = D1(dep), REL(dep)
        wl = [R(f"WHERE_LEAST__n10__s{s}") for s in range(3)]
        d1w, relw = float(np.mean([D1(x) for x in wl])), float(np.mean([REL(x) for x in wl]))
        lin = {}
        for M in LIN:
            rs = [R(f"{M}__n10__s{s}") for s in range(3)]
            lin[M] = {"D1": float(np.mean([D1(x) for x in rs])), "rel": float(np.mean([REL(x) for x in rs])), "D1_per_seed": [D1(x) for x in rs]}
        pick = lambda ok: min([m for m in LIN if ok(lin[m]["rel"])], key=lambda m: (lin[m]["D1"], list(LIN).index(m)), default=None)
        one = pick(lambda r: r >= reld - 2); two = pick(lambda r: abs(r - reld) <= 2); anyb = pick(lambda r: True)
        def diff(m): return None if m is None else {"method": m, "D1_linear": lin[m]["D1"], "rel_linear": lin[m]["rel"], "difference": d1w - lin[m]["D1"],
                                                    "per_seed_difference": [D1(wl[s]) - lin[m]["D1_per_seed"][s] for s in range(3)]}
        st = H["cells"][f"{pn}|n10"]
        v = {"D1_repair": d1w, "rel_repair": relw, "D1_deployed": d1d, "rel_deployed": reld, "rel_diff": relw - reld, "D1_at_most_10": d1w <= 10,
             "two_sided_met": bool(d1w <= 10 and abs(relw - reld) <= 2), "one_sided_met": bool(d1w <= 10 and relw - reld >= -2),
             "stored_verdict": {"helps_on_this_log": st["verdict"]["helps_on_this_log"], "one_sided": st["verdict"]["one_sided_reading_met_on_this_log"], "D1": st["where_least"]["hd_D1"]["mean"], "rel": st["where_least"]["rel"]["mean"]},
             "best_linear_any": diff(anyb), "best_linear_one_sided_selection": diff(one), "best_linear_two_sided_selection": diff(two),
             "stored_best_rel": st["best_rel"]["method"], "stored_best_any": st["best_any"]["method"], "linear": lin}
        v["agrees_with_stored"] = bool(v["two_sided_met"] == st["verdict"]["helps_on_this_log"] and v["one_sided_met"] == st["verdict"]["one_sided_reading_met_on_this_log"]
                                       and abs(d1w - st["where_least"]["hd_D1"]["mean"]) < 1e-9 and abs(relw - st["where_least"]["rel"]["mean"]) < 1e-9
                                       and LIN[one] == st["best_rel"]["method"] and LIN[anyb] == st["best_any"]["method"])
        v["selection_changes_under_two_sided_rule"] = one != two
        v["difference_sign_changes"] = None if two is None else bool(np.sign(diff(one)["difference"]) != np.sign(diff(two)["difference"]))
        res["h1"][pn] = v
        print(pn, "D1 %.2f->%.2f Rel %.0f->%.0f | two-sided %s one-sided %s | stored agrees %s | best one-sided %s (%.2f) two-sided %s (%s) changes %s sign-change %s" % (
            d1d, d1w, reld, relw, v["two_sided_met"], v["one_sided_met"], v["agrees_with_stored"], one, diff(one)["difference"], two, None if two is None else round(diff(two)["difference"], 2),
            v["selection_changes_under_two_sided_rule"], v["difference_sign_changes"]))
# development logs (e2_curves rows) for the full H1 rule
dev = {}
for par, sh in ((QDEV, "Q"), (LDEV, "L")):
    RR = lambda c: rd(f"{FINAL}/e2_curves/out/rows/{c}.jsonl.gz")
    dep = RR(f"base__{sh}__deployed")
    wl = [RR(f"b__{sh}__where_least__logonly__n10__s{s}") for s in range(3)]
    D1h = lambda rs: 100 * float(np.mean([r["D1"] for r in rs if r["kind"] == "probe" and "D1" in r and r.get("pool") != "repair" and r.get("disjoint", True)]))
    d1w, relw, d1d, reld = float(np.mean([D1h(x) for x in wl])), float(np.mean([REL(x) for x in wl])), D1h(dep), REL(dep)
    dev[par] = {"D1_repair": d1w, "rel_repair": relw, "D1_deployed": d1d, "rel_deployed": reld, "two_sided_met": bool(d1w <= 10 and abs(relw - reld) <= 2), "one_sided_met": bool(d1w <= 10 and relw - reld >= -2)}
    print(par, dev[par])
res["h1_development_logs_from_rows"] = dev
res["h1_final"] = {"two_sided": {"development_logs": all(v["two_sided_met"] for v in dev.values()), "prefixes_met": [p for p, v in res["h1"].items() if v["two_sided_met"]]},
                   "one_sided": {"development_logs": all(v["one_sided_met"] for v in dev.values()), "prefixes_met": [p for p, v in res["h1"].items() if v["one_sided_met"]]}}
res["h1_final"]["two_sided"]["rule_met"] = bool(res["h1_final"]["two_sided"]["development_logs"] and res["h1_final"]["two_sided"]["prefixes_met"])
res["h1_final"]["one_sided"]["rule_met"] = bool(res["h1_final"]["one_sided"]["development_logs"] and res["h1_final"]["one_sided"]["prefixes_met"])
print(res["h1_final"])
jdump(res, OUT + "/c6_prefix.json")
