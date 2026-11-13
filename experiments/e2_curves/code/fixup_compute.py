"""E2 fix-up (amendments of 2026-11-12 01:00 and 02:55 UTC): ADDITIONS only, from stored files; no model run, no existing file is
changed.  Writes into out/fixup/:

  h1_registered.json           item 5: the rule of H1 (R6) as registered, its outcome under both readings of 'within 2 points', the
                               numbers behind them, and the prefix part copied from e2_prefix/out/h1_prefix.json with its time stamp
  reporting.json               item 6: (a) edits still released after a least repair, (b) false firing of the label-free migration on
                               MME-T by n, (c) capped edits of part c' with two decimals
  running_example_flag.json    item 6 (d): the complaint's edit and the why-not rephrase of the running example are two edits
  paper_numbers_fixup.json     flat {name: value}; every name starts with 'EtwoFx' (no name of out/paper_numbers.json is reused)
  paper_numbers_fixup_doc.json one line per name
  checks.json                  every check with its outcome
(item 4, out/fixup/cw_check_200.json, is written by fixup_cw200.py, which must have run before: its paper numbers are added here)

    PYTHONDONTWRITEBYTECODE=1 python fixup_compute.py        (CPU; then fixup_render.py)
"""
from __future__ import annotations
import os as _go, sys as _gs  # ARTIFACT: a failed run leaves the stored output files untouched (rescope_paths.guard_outputs)
_gd = _go.environ.get("RESCOPE_HOME") or _go.path.dirname(_go.path.abspath(__file__))
while not _go.path.exists(_go.path.join(_gd, "rescope_paths.py")) and _go.path.dirname(_gd) != _gd: _gd = _go.path.dirname(_gd)
_gs.path.insert(0, _gd); import rescope_paths as _RPG; _RPG.guard_outputs(__file__, __name__)

import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2c  # noqa: E402
from e2c import fx  # noqa: E402

FIX = os.path.join(e2c.OUT, "fixup")
PREFIX = os.path.join(e2c.FIN, "e2_prefix", "out")
HN = ("HN-I", "HN-Q", "HN-X")
KW = {"HN-I": "HNI", "HN-Q": "HNQ", "HN-X": "HNX"}
SW = {0: "Zero", 1: "One", 2: "Two"}
NW = {"0": "Zero", "1": "One", "2": "Two", "5": "Five", "10": "Ten", "20": "Twenty", "40": "Forty", "50": "Fifty", "100": "Hundred", "200": "TwoHundred", "ALL": "All"}
N_BOOT = fx.stats.N_BOOT
PN, DOC, CHECKS, ITEM_OF, CUR = {}, {}, [], {}, [0]
KEY = ("log", "part", "method", "variant", "n_label", "draw", "pool", "metric")
jload = fx.io.jload


def put(name, value, text):
    if value is None:
        return
    if name in PN:
        raise RuntimeError("paper number defined twice: " + name)
    PN[name], DOC[name], ITEM_OF[name] = value, text, CUR[0]


def check(item, what, ok, detail=None):
    CHECKS.append({"item": item, "check": what, "ok": bool(ok), "detail": detail})
    print(f"  [{'ok' if ok else 'FAILED'}] item {item}: {what}" + ("" if detail is None else f"  ({str(detail)[:200]})"), flush=True)


def close(a, b, tol=1e-9):
    return a is not None and b is not None and abs(float(a) - float(b)) <= tol


class Rec:
    def __init__(self):
        self.J = {}
        for r in jload(os.path.join(e2c.OUT, "results.json")):
            self.J[tuple(str(r.get(k)) if k == "n_label" else r.get(k) for k in KEY)] = r

    def __call__(self, log, part, method, variant, nl, metric, pool, draw=None):
        return self.J.get((log, part, method, variant, str(nl), draw, pool, metric))


def slim(r):
    return None if r is None else {k: r.get(k) for k in ("value", "ci_lo", "ci_hi", "n_units", "n_draws", "per_draw") if r.get(k) is not None or k == "value"}


# ----------------------------------------------------------------------------- item 4: paper numbers of cw_check_200.json
def item4():
    p = os.path.join(FIX, "cw_check_200.json")
    if not os.path.exists(p):
        check(4, "out/fixup/cw_check_200.json exists (fixup_cw200.py has run)", False)
        return None
    cw = jload(p)
    for name, c in cw["logs"].items():
        sh = e2c.SHORT[name]
        check(4, f"{sh}: the first 20 of the 200 seeded draws reproduce the stored draws of out/parts/cw_check_{name}.json (per-draw rates, fresh rows, bound, means)",
              c["reproduction_of_the_stored_20_draws"]["first_20_draws_reproduce_the_stored_draws"],
              {k: c["reproduction_of_the_stored_20_draws"][k] for k in ("max_abs_diff_fresh_repair", "max_abs_diff_heldout_disjoint", "mean_first_20_fresh_repair", "stored_mean_fresh_repair")})
        check(4, f"{sh}: the bound is the same for every draw and the statistic is the mean of the per-draw rates", c["bound_is_the_same_for_every_draw"]
              and c["fresh_repair"]["mean_of_per_draw_rates_equals_statistic"] and c["heldout_disjoint"]["mean_of_per_draw_rates_equals_statistic"])
        ph = "POST HOC read-out of part c' (literal two-block rule of ReScope-W without guards, alpha 0.05, n 200; gate level, captured vectors; out/fixup/cw_check_200.json)"
        put(f"EtwoFx{sh}WCalBound", c["bound"], f"{name}: bound of the two-block rule on the expected false-firing rate (%), {ph}")
        put(f"EtwoFx{sh}WCalNDraws", c["n_draws"], f"{name}: number of seeded calibration draws (seeds 0 to 199) of the {ph}")
        for key, w, pool in (("fresh_repair", "Fresh", "the valid REPAIR rows whose query is not a calibration query of the draw"), ("heldout_disjoint", "Held", "heldout_disjoint")):
            e = c[key]
            put(f"EtwoFx{sh}WCal{w}TwoHundredMean", e["mean_over_draws"], f"{name}: false firing (%) on {pool}, mean over 200 seeded draws; {ph}")
            put(f"EtwoFx{sh}WCal{w}TwoHundredLo", e["ci_lo"], f"{name}: lower end of the 95 % interval of that mean (bootstrap over query clusters, the 200 draws nested: the same resampled clusters under every draw; 2000 resamples)")
            put(f"EtwoFx{sh}WCal{w}TwoHundredHi", e["ci_hi"], f"{name}: upper end of that interval")
            put(f"EtwoFx{sh}WCal{w}TwoHundredContainsBound", e["interval_contains_the_bound"], f"{name}: true if that interval contains the bound ({c['bound']:.2f} %); {ph}")
            put(f"EtwoFx{sh}WCal{w}TwoHundredMinusBound", e["mean_minus_bound"], f"{name}: that mean minus the bound (points)")
            put(f"EtwoFx{sh}WCal{w}TwoHundredCrossedLo", e["crossed_sensitivity"]["ci_lo"], f"{name}: sensitivity: lower end of the interval when the 200 draws are resampled as well (crossed); not the reported interval")
            put(f"EtwoFx{sh}WCal{w}TwoHundredCrossedHi", e["crossed_sensitivity"]["ci_hi"], f"{name}: sensitivity: upper end of the crossed interval")
            put(f"EtwoFx{sh}WCal{w}TwoHundredDrawsAbove", 100.0 * e["over_draws"]["share_of_draws_above_the_bound"], f"{name}: share of the 200 draws (%) whose rate on {pool} is above the bound")
            put(f"EtwoFx{sh}WCal{w}TwoHundredPFive", e["over_draws"]["p05"], f"{name}: 5th percentile over the 200 draws of the rate on {pool}")
            put(f"EtwoFx{sh}WCal{w}TwoHundredPNinetyFive", e["over_draws"]["p95"], f"{name}: 95th percentile over the 200 draws of the rate on {pool}")
    return cw


# ----------------------------------------------------------------------------- item 5: H1 as registered
def holm_bounds(ps, floor):
    keys = sorted(ps, key=lambda k: ps[k])
    m, run, src, out = len(keys), 0.0, False, {}
    for rank, k in enumerate(keys):
        term = min(1.0, (m - rank) * ps[k])
        at = ps[k] <= floor * (1 + 1e-12)
        if term > run:
            run, src = term, at
        elif term == run and at:
            src = True
        out[k] = {"p": ps[k], "p_holm": run, "at_resolution": bool(at), "p_holm_is_upper_bound": bool(src)}
    return out


def read_prefix():
    """h1_prefix.json as it is at the time of reading (the run may still be in progress): read once, with its time stamp."""
    p = os.path.join(PREFIX, "h1_prefix.json")
    m0 = os.path.getmtime(p)
    h = jload(p)
    m1 = os.path.getmtime(p)
    st = jload(os.path.join(PREFIX, "status.json")) if os.path.exists(os.path.join(PREFIX, "status.json")) else {}
    ts = lambda t: time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(t))        # noqa: E731
    return h, {"file": p, "modified": ts(m0), "unchanged_while_read": m0 == m1, "read_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}, st


def item5(R, ver):
    h1 = jload(os.path.join(e2c.OUT, "h1.json"))
    proto = open(os.path.join(e2c.FIN, "PROTOCOL.md")).read()
    m = re.search(r"\*\*R6 Endpoints\.\*\*\s+(H1 \(repair at equal evidence.*?mild prefix \(R12\)\.)", proto, re.S)
    quote = None if m is None else re.sub(r"\s+", " ", m.group(1))
    check(5, "the rule of R6 is found in PROTOCOL.md and quoted from the file", quote is not None)
    floor = 1.0 / N_BOOT
    ps = {f"{e2c.SHORT[n]}/{k}": v["d1_diff"]["p_two_sided"] for n, hb in h1["logs"].items() for k, v in hb["comparisons"].items()}
    hb_ = holm_bounds(ps, floor)
    check(5, "Holm values of the two differences to linear repairs, recomputed, equal out/h1.json", all(close(hb_[k]["p_holm"], h1["holm"]["adjusted"][k], 1e-12) for k in ps))
    logs, okv, okf = {}, True, True
    for name in e2c.LOGS:
        sh, hb = e2c.SHORT[name], h1["logs"][name]
        a = (name, "b", "where_least", "logonly", "10")
        g = lambda metric, pool: R(*a, metric + "@mean", pool)                 # noqa: E731
        dep_d1 = R(name, "base", "deployed", "", "0", "probe.D1", "heldout_disjoint")
        dep_rel = R(name, "base", "deployed", "", "0", "edit.rel.exact", "edits")
        d1, rel = g("probe.D1", "heldout_disjoint"), g("edit.rel.exact", "edits")
        okv = okv and close(d1["value"], hb["d1_where"]) and close(rel["value"], hb["rel_where"]) and close(dep_rel["value"], hb["rel_deployed"]) \
            and all(close(x, y) for x, y in zip(d1["per_draw"], next(iter(hb["comparisons"].values()))["d1_where_per_seed"]))
        d1v, relv, depv = hb["d1_where"], hb["rel_where"], hb["rel_deployed"]
        c1, two, one = d1v <= 10.0, abs(relv - depv) <= 2.0, relv >= depv - 2.0
        okf = okf and c1 == hb["d1_at_most_10"] and two == hb["rel_within_2"] and one == hb["rel_not_more_than_2_below_deployed"] and (c1 and two) == hb["helps_on_this_log"] \
            and (c1 and one) == hb["one_sided_reading_met_on_this_log"]
        kg = {f: slim(g(f"kg.rel.{f}", "edits")) for f in ("kept", "gained", "lost")}
        chk = {"lost": slim(g("edit.checks_broken", "edits")), "gained": slim(g("edit.checks_fixed", "edits")), "n_checks": slim(g("edit.n_checks", "edits"))}
        fire = slim(g("probe.fired", "heldout_disjoint"))
        hn = {k: slim(g(f"gate.hn.{k}.fired", "hard_negatives")) for k in HN}
        comps = {}
        for k, v in hb["comparisons"].items():
            b = hb_[f"{sh}/{k}"]
            pr = ("<= " if b["p_holm_is_upper_bound"] else "") + f"{b['p_holm']:.3f}"
            comps[k] = {"linear_repair": v["linear"], "d1_repair": v["d1_where"], "d1_linear": v["d1_linear"], "d1_linear_per_stream": v["d1_linear_per_seed"],
                        "difference_points": v["d1_diff"]["value"], "ci_lo": v["d1_diff"]["ci_lo"], "ci_hi": v["d1_diff"]["ci_hi"], "p_two_sided": v["d1_diff"]["p_two_sided"],
                        "p_at_resolution": b["at_resolution"], "p_holm": b["p_holm"], "p_holm_is_upper_bound": b["p_holm_is_upper_bound"], "p_holm_print": pr,
                        "rel_linear": v["rel_linear"], "interval": "paired crossed bootstrap over the 3 complaint streams and the query clusters, 2000 resamples",
                        "selection": {"best_lowest_d1": "the linear repair with the lowest mean D1 at n = 10", "best_rel_within_2": hb["selection_of_best_rel_within_2"]}[k]}
            w = {"best_lowest_d1": "Best", "best_rel_within_2": "BestRel"}[k]
            put(f"EtwoFx{sh}Hone{w}PHolmPrint", pr, f"{name}: H1, D1 difference ReScope-W minus {v['linear']} ({k}): Holm-adjusted p as to be printed ('<=': upper bound, the bootstrap p sits at its resolution 1 / {N_BOOT}; 4 tests)")
            put(f"EtwoFx{sh}Hone{w}PHolmIsBound", bool(b["p_holm_is_upper_bound"]), f"{name}: H1, the same comparison: true if the Holm-adjusted p is an upper bound")
            put(f"EtwoFx{sh}Hone{w}PHolm", b["p_holm"], f"{name}: H1, the same comparison: Holm-adjusted p (an upper bound where ...PHolmIsBound is true; out/h1.json, holm.adjusted)")
        logs[name] = {"n_complaints": hb["n"], "complaint_streams": hb["seeds"],
                      "repair": {"what": "ReScope-W, least repair of the n = 10 reported complaints from the log-only thresholds",
                                 "D1_heldout_disjoint": {"mean": d1v, "per_stream": d1["per_draw"], "ci_lo": d1["ci_lo"], "ci_hi": d1["ci_hi"]},
                                 "rel_exact": {"mean": relv, "per_stream": rel["per_draw"]}},
                      "deployed_log": {"D1_heldout_disjoint": dep_d1["value"], "rel_exact": depv},
                      "rel_exact_repair_minus_deployed_points": relv - depv,
                      "D1_at_most_10_percent": bool(c1),
                      "two_sided_reading": {"condition": "|Rel exact - Rel exact of the deployed log| <= 2 points", "rel_condition_met": bool(two), "met_on_this_log": bool(c1 and two)},
                      "one_sided_reading": {"condition": "Rel exact not more than 2 points below the deployed log", "rel_condition_met": bool(one), "met_on_this_log": bool(c1 and one)},
                      "why_the_readings_differ": None if two == one else f"Rel exact of the repair is {relv - depv:+.1f} points ABOVE the deployed log (the repair answers more requests than the deployed log)",
                      "rel_edits_against_the_deployed_log": {"what": "edits (of 100) whose Rel check is exact under both (kept), only under the repair (gained), only under the deployed log (lost)",
                                                             "kept": kg["kept"], "gained": kg["gained"], "lost": kg["lost"]},
                      "edit_checks_of_300_against_the_deployed_log": {"what": "Rel, T-Gen and I-Gen checks (3 x 100) exact under the deployed log and not under the repair (lost), and the reverse (gained)",
                                                                      "lost": chk["lost"], "gained": chk["gained"], "n_checks": chk["n_checks"]},
                      "edits_still_released": {"what": "share of rows on which the repaired log still releases an edit: heldout_disjoint (hook, behaviour) and near-scope negatives per kind "
                                                       "(gate level, captured all-position vectors); mean over the three complaint streams",
                                               "heldout_disjoint_percent": fire, "near_scope_percent": hn},
                      "differences_to_linear_repairs": comps}
        put(f"EtwoFx{sh}HoneTwoSided", bool(c1 and two), f"{name}: H1 (R6) on this log, two-sided reading: D1 <= 10 % and |Rel exact - Rel exact deployed| <= 2 points (out/fixup/h1_registered.json; numbers of out/h1.json)")
        put(f"EtwoFx{sh}HoneOneSided", bool(c1 and one), f"{name}: H1 (R6) on this log, one-sided reading: D1 <= 10 % and Rel exact not more than 2 points below the deployed log")
        put(f"EtwoFx{sh}HoneDepDone", dep_d1["value"], f"{name}: H1, D1 % on heldout_disjoint of the deployed log (out/results.json, part base)")
        put(f"EtwoFx{sh}HoneDepRel", depv, f"{name}: H1, Rel exact % of the deployed log")
        for s in (0, 1, 2):
            put(f"EtwoFx{sh}HoneWhereDoneStream{SW[s]}", d1["per_draw"][s], f"{name}: H1, ReScope-W (least repair of 10 complaints, log-only start), D1 % on heldout_disjoint, complaint stream {s} (out/results.json, part b, probe.D1@mean per_draw)")
            put(f"EtwoFx{sh}HoneWhereRelStream{SW[s]}", rel["per_draw"][s], f"{name}: H1, the same repair, Rel exact %, complaint stream {s}")
        put(f"EtwoFx{sh}HoneWhereDoneLo", d1["ci_lo"], f"{name}: H1, lower end of the 95 % interval of the repair's D1 (crossed bootstrap, 3 streams x query clusters)")
        put(f"EtwoFx{sh}HoneWhereDoneHi", d1["ci_hi"], f"{name}: H1, upper end of that interval")
        for f, w in (("kept", "Kept"), ("gained", "Gained"), ("lost", "Lost")):
            put(f"EtwoFx{sh}HoneRel{w}", kg[f]["value"], f"{name}: H1 row, edits (of 100) whose Rel check is {f} against the deployed log, mean over the three complaint streams (out/results.json, part b, kg.rel.{f}@mean)")
        put(f"EtwoFx{sh}HoneChecksLost", chk["lost"]["value"], f"{name}: H1 row, edit checks (of 300) exact under the deployed log and not under the repair, mean over the three streams (edit.checks_broken@mean)")
        put(f"EtwoFx{sh}HoneChecksGained", chk["gained"]["value"], f"{name}: H1 row, edit checks (of 300) exact under the repair and not under the deployed log, mean over the three streams (edit.checks_fixed@mean)")
        v = ver["numbers"][sh]
        check(5, f"{sh}: stream 0 of the kept / gained / lost and lost / gained counts equals the verification's numbers (verify/verdicts.json)",
              [kg[f]["per_draw"][0] for f in ("kept", "gained", "lost")] == v["h1_row_rel_kept_gained_lost_seed0"] and
              [chk["lost"]["per_draw"][0], chk["gained"]["per_draw"][0]] == v["h1_row_edit_checks_lost_gained_of_300_seed0"])
        check(5, f"{sh}: firing on heldout_disjoint and near-scope firing of the H1 row equal the verification's numbers", close(fire["value"], v["h1_row_fires_on_heldout_disjoint_percent_mean"])
              and all(close(hn[k]["value"], v["h1_row_near_scope_firing_percent_gate_mean"][k]) for k in HN))
    check(5, "the numbers of out/h1.json equal the records of out/results.json (D1, Rel, per stream)", okv)
    check(5, "both readings, recomputed from the numbers, equal the flags of out/h1.json", okf)
    # prefix part: copied as it is at the time of reading
    hp, stamp, st = read_prefix()
    cells = {}
    okp = True
    for key, c in hp["cells"].items():
        if not key.endswith("|n10"):
            continue
        e = {"log": c["log"], "T": c["T"], "status": c["status"], "complete": c["status"] == "complete", "deployed": c.get("deployed"),
             "n_cells_missing_in_status_json": len(st.get(c["log"], {}).get("missing", [])) if c["log"] in st else None}
        if e["complete"]:
            v = c["verdict"]
            dv, rv, dp = c["where_least"]["hd_D1"]["mean"], c["where_least"]["rel"]["mean"], c["deployed"]["rel"]
            e.update(D1_heldout_disjoint=c["where_least"]["hd_D1"], rel_exact=c["where_least"]["rel"], rel_exact_repair_minus_deployed_points=v["rel_signed_diff_to_deployed"],
                     D1_at_most_10_percent=v["D1_at_most_10"], two_sided_reading_met=v["helps_on_this_log"], one_sided_reading_met=v["one_sided_reading_met_on_this_log"],
                     counts_for_the_endpoint=v["counts_for_the_endpoint"])
            okp = okp and (dv <= 10.0 and abs(rv - dp) <= 2.0) == v["helps_on_this_log"] and (dv <= 10.0 and rv >= dp - 2.0) == v["one_sided_reading_met_on_this_log"]
        cells[c["log"]] = e
    check(5, "prefix cells: both readings, recomputed from the copied numbers, equal the flags of h1_prefix.json", okp)
    check(5, "h1_prefix.json did not change while it was read", stamp["unchanged_while_read"], stamp)
    incomplete = [k for k, e in cells.items() if not e["complete"]]
    two_p = [k for k, e in cells.items() if e["complete"] and e["two_sided_reading_met"]]
    one_p = [k for k, e in cells.items() if e["complete"] and e["one_sided_reading_met"]]
    dev_two = all(v["two_sided_reading"]["met_on_this_log"] for v in logs.values())
    dev_one = all(v["one_sided_reading"]["met_on_this_log"] for v in logs.values())
    label = "prefix part: provisional" if incomplete else "prefix part: complete"

    def verdict(dev, pre):
        if not dev:
            return "not met (the development-log part fails; the prefix part cannot change this)"
        if pre:
            return "met" + (" (prefix part provisional: one complete prefix already meets it)" if incomplete else "")
        return "open (no complete prefix meets it yet)" if incomplete else "not met (no prefix meets it)"
    out = {"what": "H1 (repair at equal evidence) as registered, with its outcome under both readings of 'within 2 points' and the numbers behind them",
           "rule_as_registered": {"source": "PROTOCOL.md, section 6, amendment 'Review round 1' (2026-11-11 06:14 UTC, before any final run), R6", "quote": quote},
           "readings": {"two_sided": "Rel exact of the repair differs from Rel exact of the deployed log by at most 2 points in either direction",
                        "one_sided": "Rel exact of the repair is not more than 2 points below Rel exact of the deployed log",
                        "status": "the rule is not re-worded; both readings are reported (PROTOCOL.md, amendments of 2026-11-12 01:00 and 02:55 UTC); the one-sided reading was adopted after the run"},
           "development_logs": logs,
           "development_part": {"two_sided_reading_met_on_both_logs": bool(dev_two), "one_sided_reading_met_on_both_logs": bool(dev_one)},
           "holm": {"family": "the two differences to linear repairs on the two development logs (4 tests: backbones x the two choices of the best linear repair)",
                    "resolution_of_the_bootstrap_p": floor, "smallest_possible_adjusted_p": len(ps) * floor,
                    "note": "a bootstrap p at 1 / n_boot means that no resample crossed 0: the Holm-adjusted value is then an upper bound"},
           "prefix_part": {"label": label, "source": stamp, "copied_from": "e2_prefix/out/h1_prefix.json (cells at n = 10 reported complaints; R12)", "prefix_logs": cells,
                           "n_prefix_logs": len(cells), "n_complete": len(cells) - len(incomplete), "incomplete": incomplete,
                           "two_sided_reading_met_on": two_p, "one_sided_reading_met_on": one_p,
                           "fields_of_the_file": {k: hp.get(k) for k in ("n_prefix_cells_complete_n10", "prefixes_that_help_n10", "at_least_one_mild_prefix_helps")},
                           "at_least_one_prefix_meets_the_two_sided_reading": bool(two_p), "at_least_one_prefix_meets_the_one_sided_reading": bool(one_p)},
           "rule_outcome": {"two_sided_reading": verdict(dev_two, bool(two_p)), "one_sided_reading": verdict(dev_one, bool(one_p)),
                            "parts": "the rule needs both development logs AND at least one mild prefix"},
           "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    fx.io.jdump(out, os.path.join(FIX, "h1_registered.json"), indent=1)
    put("EtwoFxHoneDevTwoSided", bool(dev_two), "H1 (R6): the two-sided reading is met on both development logs (out/fixup/h1_registered.json)")
    put("EtwoFxHoneDevOneSided", bool(dev_one), "H1 (R6): the one-sided reading is met on both development logs")
    put("EtwoFxHoneRuleTwoSided", out["rule_outcome"]["two_sided_reading"], "H1 (R6): outcome of the registered rule (development logs and at least one mild prefix) under the two-sided reading")
    put("EtwoFxHoneRuleOneSided", out["rule_outcome"]["one_sided_reading"], "H1 (R6): outcome of the registered rule under the one-sided reading (adopted after the run)")
    put("EtwoFxHonePrefixLabel", label, f"H1: state of the prefix part as copied from e2_prefix/out/h1_prefix.json (file modified {stamp['modified']})")
    put("EtwoFxHonePrefixFileTime", stamp["modified"], "H1: time stamp of e2_prefix/out/h1_prefix.json when it was read")
    put("EtwoFxHonePrefixN", len(cells), "H1: number of prefix logs (10, 20 and 50 edits of the two development streams)")
    put("EtwoFxHonePrefixNComplete", len(cells) - len(incomplete), "H1: number of prefix logs whose n = 10 cell is complete in h1_prefix.json at the time of reading")
    put("EtwoFxHonePrefixNTwoSided", len(two_p), "H1: number of complete prefix logs on which the two-sided reading is met")
    put("EtwoFxHonePrefixNOneSided", len(one_p), "H1: number of complete prefix logs on which the one-sided reading is met")
    for k, e in cells.items():
        m2 = re.match(r"TB-([QL])100-ae-l2_3-p(\d+)$", k)
        nm = f"EtwoFxPre{m2.group(1)}P{NW[m2.group(2)]}" if m2 else None
        if nm is None:
            continue
        put(nm + "Complete", bool(e["complete"]), f"{k}: H1 prefix cell (n = 10) complete in h1_prefix.json at the time of reading ({stamp['modified']})")
        if e["complete"]:
            put(nm + "Done", e["D1_heldout_disjoint"]["mean"], f"{k}: H1, ReScope-W least repair of 10 complaints, D1 % on heldout_disjoint, mean over 3 streams (copied from h1_prefix.json)")
            put(nm + "Rel", e["rel_exact"]["mean"], f"{k}: H1, the same repair, Rel exact % (copied from h1_prefix.json)")
            put(nm + "RelMinusDeployed", e["rel_exact_repair_minus_deployed_points"], f"{k}: H1, Rel exact of the repair minus the deployed log (points; copied)")
            put(nm + "TwoSided", bool(e["two_sided_reading_met"]), f"{k}: H1, two-sided reading met on this prefix log (copied)")
            put(nm + "OneSided", bool(e["one_sided_reading_met"]), f"{k}: H1, one-sided reading met on this prefix log (copied)")
    return out


# ----------------------------------------------------------------------------- item 6: reporting
def item6(R, ver):
    out = {"edits_still_released": [], "mme_t": {}, "capped": {}}
    e2pn = jload(os.path.join(e2c.OUT, "paper_numbers.json"))
    for name in e2c.LOGS:
        sh = e2c.SHORT[name]
        # (a)
        for start, nl, w, title in (("logonly", "10", "WLogTen", "H1 row: least repair of 10 complaints from the log-only thresholds"),
                                    ("deployed", "10", "WDepTen", "least repair of 10 complaints from the deployed start (all thresholds -inf)"),
                                    ("deployed", "100", "WDepHundred", "least repair of 100 complaints from the deployed start (all thresholds -inf)")):
            a = (name, "b", "where_least", start, nl)
            fire, d1 = R(*a, "probe.fired@mean", "heldout_disjoint"), R(*a, "probe.D1@mean", "heldout_disjoint")
            hn = {k: R(*a, f"gate.hn.{k}.fired@mean", "hard_negatives") for k in HN}
            out["edits_still_released"].append({"log": name, "row": title, "start": start, "n": nl, "n_streams": fire.get("n_draws"), "heldout_disjoint_fired_percent": slim(fire),
                                                "heldout_disjoint_D1_percent": slim(d1), "near_scope_fired_percent": {k: slim(hn[k]) for k in HN}})
            src = f"out/results.json, part b, where_least / {start}, n = {nl}"
            put(f"EtwoFx{sh}{w}FireHeld", fire["value"], f"{name}: {title}: share of heldout_disjoint rows (%) on which the repaired log still releases an edit, mean over {fire.get('n_draws')} complaint stream(s) ({src}, probe.fired@mean)")
            put(f"EtwoFx{sh}{w}FireHeldLo", fire["ci_lo"], "lower end of its 95 % interval (crossed bootstrap over streams and query clusters)")
            put(f"EtwoFx{sh}{w}FireHeldHi", fire["ci_hi"], "upper end of its 95 % interval")
            for k in HN:
                put(f"EtwoFx{sh}{w}{KW[k]}Fire", hn[k]["value"], f"{name}: {title}: near-scope negatives {k} (%) on which an edit is still released, gate level, mean over the stream(s) ({src}, gate.hn.{k}.fired@mean)")
        v = ver["numbers"][sh]
        e = next(x for x in out["edits_still_released"] if x["log"] == name and x["start"] == "deployed" and x["n"] == "10")
        check(6, f"(a) {sh}: deployed start, n = 10: firing on heldout_disjoint and near-scope firing equal the verification's numbers", close(e["heldout_disjoint_fired_percent"]["value"], v["deployed_start_n10_fires_on_heldout_percent_mean"])
              and all(close(e["near_scope_fired_percent"][k]["value"], v["deployed_start_n10_near_scope_firing_percent"][k]) for k in HN))
        # (b) MME-T by n
        cg = jload(os.path.join(e2c.OUT, "parts", f"c_gate_{name}.json"))
        rows, okm, above = [], True, []
        for nl in ("0", "1", "2", "5", "10", "20", "40", "50", "100", "200", "ALL"):
            var = "logonly" if nl == "0" else ("logonly+least" if nl in ("1", "2", "5", "10", "20") else "")
            r = R(name, "c", "rescope", var, nl, "gate.MME-T.fired@mean", "MME-T")
            if r is None:
                continue
            cs = [c for c in cg["cells"] if c["method"] == "rescope" and str(c["n_label"]) == nl]
            vals = [c["st"]["MME-T"]["fired"] for c in cs]
            mean, p05, p95 = float(np.mean(vals)), float(np.percentile(vals, 5)), float(np.percentile(vals, 95))
            okm = okm and close(mean, r["value"]) and close(p05, r["ci_lo"]) and close(p95, r["ci_hi"]) and len(vals) == r["n_units"]
            level = None if r["alpha"] is None else 100.0 * r["alpha"]
            row = {"n_label": nl, "n": r["n"], "rule": cs[0]["rule"], "alpha": r["alpha"], "level_percent": level, "n_draws": len(vals), "mean": mean, "p05": p05, "p95": p95,
                   "min": float(min(vals)), "max": float(max(vals)), "mean_above_the_level": None if level is None else bool(mean > level),
                   "share_of_draws_above_the_level": None if level is None else float(np.mean([x > level for x in vals])), "n_rows": cs[0]["st"]["MME-T"]["n"],
                   "key_field_of_the_cell": cs[0]["st"]["MME-T"].get("which")}
            rows.append(row)
            if row["mean_above_the_level"]:
                above.append(nl)
            what = f"{name}: label-free migration (ReScope, {cs[0]['rule']}) on n = {nl} benign REPAIR queries: false firing (%) on the out-of-benchmark workload MME-T ({cs[0]['st']['MME-T']['n']} rows; keys as the part file describes them: prompt-only; cell field which = '{cs[0]['st']['MME-T'].get('which')}'), gate level"
            put(f"EtwoFx{sh}RescopeMmeT{NW[nl]}Mean", mean, f"{what}, mean over {len(vals)} draw(s) (out/results.json, part c, gate.MME-T.fired@mean; out/parts/c_gate_{name}.json)")
            put(f"EtwoFx{sh}RescopeMmeT{NW[nl]}PFive", p05, f"{what}: 5th percentile over the {len(vals)} draw(s)")
            put(f"EtwoFx{sh}RescopeMmeT{NW[nl]}PNinetyFive", p95, f"{what}: 95th percentile over the {len(vals)} draw(s)")
            if level is not None:
                put(f"EtwoFx{sh}RescopeMmeT{NW[nl]}AboveLevel", bool(mean > level), f"{what}: true if the mean exceeds the level {level:.0f} %")
        out["mme_t"][name] = {"rows": rows, "n_labels_with_mean_above_the_level": above, "keys": cg.get("pools"),
                              "note": "the level applies to the two-block rule (n >= 40); at n <= 20 the thresholds are log-only plus a least repair (no level).  The cell field 'which' of "
                                      "MME-T is copied as stored; the description of the keys is the one of the part file and of PROTOCOL.md (R25 (5))"}
        put(f"EtwoFx{sh}RescopeMmeTAboveLevelAt", ", ".join(above) if above else "none", f"{name}: the n at which the mean false firing of the label-free migration on MME-T exceeds the level 5 % (two-block rule, n >= 40)")
        check(6, f"(b) {sh}: mean, 5th and 95th percentile of MME-T firing recomputed from out/parts/c_gate_{name}.json equal the records of out/results.json", okm and len(rows) > 0, f"{len(rows)} n")
        check(6, f"(b) {sh}: the means at n = 40, 50, 100, 200, ALL equal the verification's numbers", all(close(next(x["mean"] for x in rows if x["n_label"] == k), val)
                                                                                                         for k, val in v["rescope_MME-T_false_firing_percent_mean_by_n"].items()))
        # (c) capped edits of c' with two decimals
        r = R(name, "cw", "where_cal", "two_block", "200", "gate.n_edits_capped@mean", "edits")
        out["capped"][name] = {"mean": r["value"], "p05": r["ci_lo"], "p95": r["ci_hi"], "n_draws": r["n_units"], "print_two_decimals": f"{r['value']:.2f}", "print_one_decimal_as_in_table_txt": f"{r['value']:.1f}"}
        put(f"EtwoFx{sh}WCalCappedTwoDec", f"{r['value']:.2f}", f"{name}: ReScope-W with two-block thresholds (part c'): capped edits, mean over {r['n_units']} draws, printed with two decimals (out/results.json, gate.n_edits_capped@mean)")
        put(f"EtwoFx{sh}WCalCappedPFive", r["ci_lo"], f"{name}: part c', capped edits: 5th percentile over the {r['n_units']} draws")
        put(f"EtwoFx{sh}WCalCappedPNinetyFive", r["ci_hi"], f"{name}: part c', capped edits: 95th percentile over the {r['n_units']} draws")
        check(6, f"(c) {sh}: the capped-edits record equals the paper number Etwo{sh}WCalCapped of E2", e2pn.get(f"Etwo{sh}WCalCapped") == r["value"], r["value"])
    fx.io.jdump(out, os.path.join(FIX, "reporting.json"), indent=1)
    # (d) running example
    ex = jload(os.path.join(e2c.OUT, "running_example.json"))
    sel, wn = ex["_selection"], ex["_details"]["whynot"]
    two = sel["edit_index"] != wn["edit"]
    flag = {"what": "flag for out/running_example.json (that file is not changed): the edit of the example's complaint and the why-not rephrase belong to two different edits",
            "two_different_edits": bool(two), "log": ex["log"],
            "complaint_edit": {"stream_position_0_based": sel["edit_index"], "stream_position_1_based": sel["edit_index"] + 1, "chosen_by": sel["edit_chosen_by"],
                               "question": ex["edit"]["question"], "target": ex["edit"]["target"], "coordinate_all_positions": sel["coordinate_of_the_example_edit_all_positions"]},
            "whynot_edit": {"stream_position_0_based": wn["edit"], "stream_position_1_based": wn["edit"] + 1, "is_the_example_edit": wn["is_the_example_edit"],
                            "question": wn["edit_question"], "target": wn["edit_target"], "rephrase": ex["whynot"]["query"], "coordinate": ex["whynot"]["coordinate"], "threshold": ex["whynot"]["threshold"]},
            "stream_position": "index of the edit in the edit stream of the log (0 = the first edit)",
            "for_the_text": "present them as two edits: the complaint is attributed to the edit with the largest coordinate (the default method releases no edit on the complaint); "
                            "the why-not question is asked about another edit of the same log",
            "source": "out/running_example.json: _selection.edit_index, _selection.edit_chosen_by, _details.whynot", "running_example_written": ex.get("written")}
    fx.io.jdump(flag, os.path.join(FIX, "running_example_flag.json"), indent=1)
    check(6, "(d) the two edits of the running example differ, and the record's own field is_the_example_edit says so", two and wn["is_the_example_edit"] is False, [sel["edit_index"], wn["edit"]])
    put("EtwoFxExampleTwoEdits", bool(two), "running example: true = the complaint's edit (largest coordinate) and the why-not rephrase belong to two different edits (out/fixup/running_example_flag.json)")
    put("EtwoFxExampleEditPosition", sel["edit_index"], "running example: stream position (0-based index in the edit stream) of the edit the complaint is attributed to (largest coordinate)")
    put("EtwoFxExampleEditPositionOneBased", sel["edit_index"] + 1, "running example: the same position counted from 1")
    put("EtwoFxExampleWhynotEditPosition", wn["edit"], "running example: stream position (0-based) of the edit whose text rephrase is the why-not query")
    put("EtwoFxExampleWhynotEditPositionOneBased", wn["edit"] + 1, "running example: the same position counted from 1")
    return out


def main():
    os.makedirs(FIX, exist_ok=True)
    R = Rec()
    ver = jload(os.path.join(e2c.EXP, "verify", "verdicts.json"))["unfavourable_items_in_the_files_missing_from_the_summary"]
    for i, fn in ((4, item4), (5, lambda: item5(R, ver)), (6, lambda: item6(R, ver))):
        print("item", i, flush=True)
        CUR[0] = i
        fn()
    old = jload(os.path.join(e2c.OUT, "paper_numbers.json"))
    coll = sorted(set(PN) & set(old))
    check(0, "no name of out/fixup/paper_numbers_fixup.json exists in out/paper_numbers.json", not coll, f"collisions: {len(coll)}")
    check(0, "every paper number has a documentation line", set(PN) == set(DOC))
    fx.io.jdump(dict(sorted(PN.items())), os.path.join(FIX, "paper_numbers_fixup.json"), indent=1)
    fx.io.jdump(dict(sorted(DOC.items())), os.path.join(FIX, "paper_numbers_fixup_doc.json"), indent=1)
    fx.io.jdump({"written": time.strftime("%Y-%m-%d %H:%M:%S"), "code": fx.code_fingerprint(), "n_paper_numbers": len(PN),
                 "n_paper_numbers_by_item": {str(i): sum(v == i for v in ITEM_OF.values()) for i in (4, 5, 6)}, "n_collisions_with_paper_numbers_json": len(coll),
                 "n_checks": len(CHECKS), "n_checks_failed": sum(not c["ok"] for c in CHECKS), "checks": CHECKS}, os.path.join(FIX, "checks.json"), indent=1)
    print(f"paper numbers: {len(PN)}; collisions with out/paper_numbers.json: {len(coll)}; checks failed: {sum(not c['ok'] for c in CHECKS)} of {len(CHECKS)}")


if __name__ == "__main__":
    main()
