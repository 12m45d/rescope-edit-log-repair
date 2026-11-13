"""Verification of E3, checks 2-4: the numbers of the experiment agents' two summaries against my own recomputation (out/mine_<bb>.json from
v_internal.py; H2 / H3 differences and their intervals recomputed here from the per-row files with my own crossed bootstrap).

    python v_claims.py      -> out/claims.json, out/claims_table.txt
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v_internal as VI  # noqa: E402

V = VI.V
MQ, ML = (json.load(open(f"{V}/out/mine_{bb}.json")) for bb in ("qwen", "llava"))
ROWS = []


def add(cid, quantity, exp, mine, tol, note=""):
    def close(a, b):
        if a is None or b is None:
            return a is None and b is None
        return abs(float(a) - float(b)) <= tol
    ok = all(close(a, b) for a, b in zip(exp, mine)) if isinstance(exp, (list, tuple)) else close(exp, mine)
    ROWS.append({"claim": cid, "quantity": quantity, "experiment": exp, "mine": mine, "agree": bool(ok), "tolerance": tol, "note": note})


def r1(x):
    return None if x is None else round(float(x), 1)


def Q(log):
    return "TB-Q" + log


def L(log):
    return "TB-L" + log


# ------------------------------------------------------------------ Qwen: the compact main table of the summary (Rel / T-Gen / I-Gen; D1 hd; D1 HN)
QT = {"100-ae-l2_3": {"deployed": (78, 9, 22, 54.2, 77.9), "where_logonly": (78, 7, 26, 1.1, 7.7), "rescope": (100, 56, 68, 3.9, 29.6), "white_0.1_shared": (100, 60, 79, 6.9, 39.4), "retr_sum": (100, 86, 95, 4.5, 36.6)},
      "200-ae-l2_3": {"deployed": (66, 5, 23, 77.4, 89.3), "where_logonly": (74, 4, 25, 0.9, 7.5), "rescope": (99.5, 40, 61, 4.0, 20.4), "white_0.1_shared": (99.5, 30, 65, 3.4, 20.4), "retr_sum": (99.5, 76, 91, 4.7, 32.5)},
      "500-ae-l2_3": {"deployed": (55, 4, 16, 89.0, 94.0), "where_logonly": (69, 4, 21, 1.1, 6.4), "rescope": (99.8, 26, 57, 1.8, 10.6), "white_0.1_shared": (99.8, 23, 60, 2.8, 15.0), "retr_sum": (99.8, 74, 89, 7.0, 27.1)},
      "100-memitw": {"deployed": (53, 11, 19, 89.5, 99.9), "where_logonly": (26, 5, 12, 7.7, 27.8), "rescope": (99.7, 18, 52, 4.2, 13.0), "white_0.1_shared": (100, 60, 79, 6.9, 39.3), "retr_sum": (100, 86, 95, 4.5, 36.5)},
      "500-memitw": {"deployed": (25, 6, 10, 96.6, 99.8), "where_logonly": (11, 1, 3, 11.2, 17.4), "rescope": (99.7, 11, 43, 1.8, 5.2), "white_0.1_shared": (99.8, 23, 60, 2.8, 15.0), "retr_sum": (99.8, 74, 89, 7.0, 27.0)},
      "1000-memitw": {"deployed": (15, 5, 7, 98.9, 99.7), "where_logonly": (7, 1, 2, 9.0, 13.6), "rescope": (99.9, 7, 34, 1.5, 4.2), "white_0.1_shared": (99.9, 14, 47, 4.5, 11.5), "retr_sum": (99.9, 59, 77, 5.8, 16.6)},
      "2000-memitw": {"deployed": (10, 4, 5, 97.6, 99.8), "where_logonly": (5, 0.3, 1, 4.7, 10.8), "rescope": (100, 5, 31, 2.8, 4.2), "white_0.1_shared": (99.9, 10, 42, 5.8, 12.1), "retr_sum": (100, 54, 74, 8.0, 14.3)}}
nq = 0
for log, tab in QT.items():
    for lab, e in tab.items():
        m = MQ[Q(log)]["beh"][lab]
        mine = [m["rel"], m["tgen"], m["igen"], m["hd_D1"], m["hn_D1"]]
        add("Q main table", f"Q{log} {lab}: Rel/T-Gen/I-Gen exact; D1 hd; D1 HN", list(e), [r1(x) for x in mine], 0.51)
        nq += 1

# ------------------------------------------------------------------ Qwen: statements
unseen = [g for g in VI.BB["qwen"]["logs"] if g != VI.BB["qwen"]["dev"]]
fr = [MQ[g]["gate"]["rescope"]["fire"] for g in unseen]
add("Q far locality", "rescope, gate firing on heldout_disjoint, 20 draws: min / max over the six unseen logs (%)", [2.3, 4.5], [r1(min(fr)), r1(max(fr))], 0.051,
    f"development log: {MQ[VI.BB['qwen']['dev']]['gate']['rescope']['fire']:.2f}; draws above 4.95 % per log: " + ", ".join(f"{g[3:]} {MQ[g]['gate']['rescope']['n_draws_above_bound']}/20" for g in VI.BB["qwen"]["logs"]))
add("Q far locality", "rescope, smallest Rel exact over the seven logs (%)", 99.5, r1(min(MQ[g]["beh"]["rescope"]["rel"] for g in VI.BB["qwen"]["logs"])), 0.051)


def h2(bb, log, other, sub=None):
    A, B = VI.reph_grid(bb, log, "rescope", sub), VI.reph_grid(bb, log, other, sub)
    return VI.my_crossed_diff(A, B)


for log, e in (("200-ae-l2_3", [2.8, -4.0, 10.8]), ("500-ae-l2_3", [-0.4, -3.0, 2.1])):
    add("Q H2", f"Q{log}: rescope minus white_0.1_shared, rephrase exact, all edits: diff [interval]", e, [r1(x) for x in h2("qwen", Q(log), "white_0.1_shared")], 0.45,
        "interval: my own crossed bootstrap (other random numbers)")
for log, e in (("100-memitw", -34.5), ("500-memitw", -14.6), ("1000-memitw", -10.0), ("2000-memitw", -8.1)):
    add("Q H2", f"Q{log}: rescope minus white_0.1_shared, rephrase exact, all edits", e, r1(h2("qwen", Q(log), "white_0.1_shared")[0]), 0.051)
add("Q H2", "Q1000-memitw, later edits: diff", -8.6, r1(h2("qwen", Q("1000-memitw"), "white_0.1_shared", np.arange(500, 1000))[0]), 0.051)
add("Q H2", "Q2000-memitw, later edits: diff [interval] (one draw)", [-7.3, -8.5, -6.2], [r1(x) for x in h2("qwen", Q("2000-memitw"), "white_0.1_shared", np.arange(500, 2000))], 0.3)
for log, e in (("1000-memitw", -3.7), ("2000-memitw", -1.9)):
    a, b = MQ[Q(log)]["gate"]["rescope"], MQ[Q(log)]["gate"]["white_0.1_shared"]
    add("Q H2", f"Q{log}, later edits: gate-level coverage form over 20 draws, rescope minus white_0.1_shared", e, r1(a["cov_reph_later"] - b["cov_reph_later"]), 0.051)


def h3b(bb, log, a, b):
    M = MQ if bb == "qwen" else ML
    return M[log]["beh"][a]["h3_exact"] - M[log]["beh"][b]["h3_exact"]


add("Q H3", "pair 1 (rescope minus white_1_shared), behaviour, Q200-ae / Q500-ae", [1.2, 5.1], [r1(h3b("qwen", Q("200-ae-l2_3"), "rescope", "white_1_shared")), r1(h3b("qwen", Q("500-ae-l2_3"), "rescope", "white_1_shared"))], 0.051)
me = [h3b("qwen", Q(g), "rescope", "white_1_shared") for g in ("100-memitw", "500-memitw", "1000-memitw", "2000-memitw")]
add("Q H3", "pair 1 on the four MEMIT-style logs: most negative / least negative", [-11.3, -4.9], [r1(min(me)), r1(max(me))], 0.051)
d = [MQ[g]["beh"]["retr_sum"]["reph"] - MQ[g]["beh"]["rescope"]["reph"] for g in VI.BB["qwen"]["logs"]]
add("Q retr", "retr_sum minus rescope, rephrase exact: min / max over the seven logs (points)", [28, 55], [round(min(d)), round(max(d))], 0.5, "summary: '28-55 points on every log'")
add("Q retr", "retr_sum fires on MORE hard negatives than rescope on every log (1 = yes)", 1, int(all(MQ[g]["beh"]["retr_sum"]["hn_fired"] > MQ[g]["beh"]["rescope"]["hn_fired"] for g in VI.BB["qwen"]["logs"])), 0)
ff = [MQ[Q(g)]["beh"]["retr_sum"]["hd_fired"] for g in ("500-ae-l2_3", "500-memitw", "1000-memitw", "2000-memitw")]
add("Q retr", "retr_sum, heldout_disjoint firing on the logs with T >= 500: min / max (%)", [6, 8], [round(min(ff)), round(max(ff))], 0.5)
add("Q retention", "rescope, T-Gen exact of the first 100 edits: ae T=100 -> 500", [56, 22], [round(MQ[Q("100-ae-l2_3")]["beh"]["rescope"]["tgen_first"]), round(MQ[Q("500-ae-l2_3")]["beh"]["rescope"]["tgen_first"])], 0.5)
add("Q retention", "rescope, T-Gen exact of the first 100 edits: memitw T=100 -> 2000", [18, 3], [round(MQ[Q("100-memitw")]["beh"]["rescope"]["tgen_first"]), round(MQ[Q("2000-memitw")]["beh"]["rescope"]["tgen_first"])], 0.5)
add("Q capped", "rescope, capped edits (mean of 20 draws): Q100-memitw -> Q2000-memitw", [5, 60], [round(MQ[Q("100-memitw")]["gate"]["rescope"]["capped"]), round(MQ[Q("2000-memitw")]["gate"]["rescope"]["capped"])], 0.5)
add("Q capped", "rescope_x, capped edits at Q2000-memitw (mean of 20 draws)", 220, round(MQ[Q("2000-memitw")]["gate"]["rescope_x"]["capped"]), 0.5)
add("Q capped", "literal rule without guards = rule as run on heldout_disjoint: largest |difference| of the 20-draw means, rescope, all logs (points)", 0.0,
    r1(max(abs(MQ[g]["gate"]["rescope"]["fire_lit"] - MQ[g]["gate"]["rescope"]["fire"]) for g in VI.BB["qwen"]["logs"])), 0.001)
add("Q where", "where_logonly vs deployed, Rel exact at Q100-memitw", [26, 53], [round(MQ[Q("100-memitw")]["beh"]["where_logonly"]["rel"]), round(MQ[Q("100-memitw")]["beh"]["deployed"]["rel"])], 0.5)
wf = [MQ[Q(g)]["beh"]["where_logonly"]["hd_fired"] for g in ("100-memitw", "500-memitw", "1000-memitw", "2000-memitw")]
add("Q where", "where_logonly, heldout_disjoint firing on the MEMIT-style logs: min / max (%)", [26, 46], [round(min(wf)), round(max(wf))], 0.5)
add("Q MME-T", "rescope_null on MME-T, development log (20 draws, %)", 23, round(MQ[Q("100-ae-l2_3")]["gate"]["rescope_null"]["mme"]), 0.5,
    "by log: " + ", ".join(f"{g[3:]} {MQ[g]['gate']['rescope_null']['mme']:.1f}" for g in VI.BB["qwen"]["logs"]))
add("Q MME-T", "rescope_nullW on MME-T, Q100-ae / Q200-ae (%)", [15, 18], [round(MQ[Q("100-ae-l2_3")]["gate"]["rescope_nullW"]["mme"]), round(MQ[Q("200-ae-l2_3")]["gate"]["rescope_nullW"]["mme"])], 0.5,
    "by log: " + ", ".join(f"{g[3:]} {MQ[g]['gate']['rescope_nullW']['mme']:.1f}" for g in VI.BB["qwen"]["logs"]))
add("Q margin", "smallest self-separation margin at T >= 1000 / edits with a non-positive margin", [-0.012, 1], [round(MQ[Q("2000-memitw")]["_margin"]["min"], 3), MQ[Q("2000-memitw")]["_margin"]["n_nonpositive"]], 0.0006)

# ------------------------------------------------------------------ LLaVA: the compact main table (Rel/T-Gen/I-Gen; gate fire hd; D1 hd; D1 HN; capped)
LT = {"100-ae-l2_3": {"deployed": (88, 29, 66, None, 62, 65, None), "rescope": (98, 38, 90, 3.1, 3.1, 29, 0.2), "rescope_x": (98, 15, 70, 3.5, 2.3, 8, 1), "white_1_shared": (98, 29, 93, 3.3, 3.6, 35, 0),
                      "retr_sum": (98, 89, 97, 7.2, 6.7, 42, 0.3), "where_logonly": (92, 28, 68, None, 2.8, None, None)},
      "200-ae-l2_3": {"deployed": (70, 22, 51, None, 76, 71, None), "rescope": (98, 33, 85, 2.8, 2.9, 26, 0.2), "rescope_x": (98, 19, 68, 4.2, 4.0, 6, 4), "white_1_shared": (98, 23, 89, 3.3, 3.4, 31, 0),
                      "retr_sum": (98, 83, 96, 5.8, 4.2, 35, 0.5), "where_logonly": (89, 26, 66, None, 3.4, None, None)},
      "500-ae-l2_3": {"deployed": (41, 11, 33, None, 92, 82, None), "rescope": (99, 18, 79, 1.9, 1.8, 17, 1.4), "rescope_x": (99, 15, 63, 2.4, 2.8, 4, 35), "white_1_shared": (99, 14, 83, 2.3, 2.9, 21, 0),
                      "retr_sum": (99, 72, 91, 5.9, 4.9, 26, 5), "where_logonly": (89, 20, 63, None, 10.4, None, None)},
      "100-memitw": {"deployed": (78, 26, 65, None, 78, 98, None), "rescope": (98, 11, 59, 4.1, 2.4, 28, 19), "rescope_x": (96, 4, 54, 4.4, 3.0, 6, 24), "white_1_shared": (98, 29, 93, 3.3, 3.6, 35, 0),
                     "retr_sum": (98, 89, 97, 7.2, 6.7, 42, 0.3), "where_logonly": (81, 17, 62, None, 17.5, None, None)},
      "500-memitw": {"deployed": (46, 13, 38, None, 87, 98, None), "rescope": (98, 8, 56, 2.8, 3.1, 22, 91), "rescope_x": (95, 4, 39, 3.3, 3.0, 5, 171), "white_1_shared": (99, 14, 83, 2.3, 2.9, 21, 0),
                     "retr_sum": (99, 72, 91, 5.9, 4.9, 26, 5), "where_logonly": (55, 10, 39, None, 14.7, None, None)},
      "1000-memitw": {"deployed": (37, 11, 30, None, 86, 98, None), "rescope": (99, 5, 55, 3.3, 2.8, 14, 149), "rescope_x": (97, 4, 38, 3.9, 4.7, 5, 411), "white_1_shared": (99, 8, 77, 3.0, 2.8, 18, 0),
                      "retr_sum": (99, 58, 79, 6.9, 3.8, 16, 12.5), "where_logonly": (48, 7, 34, None, 11.7, None, None)}}
for log, tab in LT.items():
    for lab, e in tab.items():
        m, g = ML[L(log)]["beh"][lab], (ML[L(log)]["gate"].get(lab) or {})
        mine = [m["rel"], m["tgen"], m["igen"], g.get("fire"), m["hd_D1"], m["hn_D1"], g.get("capped")]
        keep = [i for i, x in enumerate(e) if x is not None]
        add("L main table", f"L{log} {lab}: Rel/T-Gen/I-Gen exact; gate firing hd; D1 hd; D1 HN; capped", [e[i] for i in keep], [r1(mine[i]) for i in keep], 0.55)

LL_ = VI.BB["llava"]["logs"]
unl = [g for g in LL_ if g != VI.BB["llava"]["dev"]]
add("L locality", "rescope: Rel exact min / max; D1 hd min / max over the six logs", [98, 99, 2, 3],
    [round(min(ML[g]["beh"]["rescope"]["rel"] for g in LL_)), round(max(ML[g]["beh"]["rescope"]["rel"] for g in LL_)), round(min(ML[g]["beh"]["rescope"]["hd_D1"] for g in LL_)),
     round(max(ML[g]["beh"]["rescope"]["hd_D1"] for g in LL_))], 0.5)
add("L locality", "rescope: largest mean false firing over the six logs (gate, 20 draws; bound 4.95)", 4.1, r1(max(ML[g]["gate"]["rescope"]["fire"] for g in LL_)), 0.051,
    "draws above 4.95 % per log: " + ", ".join(f"{g[3:]} {ML[g]['gate']['rescope']['n_draws_above_bound']}/20" for g in LL_))
add("L locality", "L100-memitw, rescope: draws above 4.95 % / mean firing", [9, 4.1], [ML[L("100-memitw")]["gate"]["rescope"]["n_draws_above_bound"], r1(ML[L("100-memitw")]["gate"]["rescope"]["fire"])], 0.051)
add("L rephrases", "rescope, T-Gen exact on later edits: L200-ae, L500-ae, L500-memitw, L1000-memitw", [34, 18, 8, 6],
    [round(ML[L(g)]["beh"]["rescope"]["tgen_later"]) for g in ("200-ae-l2_3", "500-ae-l2_3", "500-memitw", "1000-memitw")], 0.5)
add("L rephrases", "L1000-memitw: T-Gen exact, rescope vs deployed", [5, 11], [round(ML[L("1000-memitw")]["beh"]["rescope"]["tgen"]), round(ML[L("1000-memitw")]["beh"]["deployed"]["tgen"])], 0.5)
ig = [ML[L(g)]["beh"]["white_1_shared"]["igen"] for g in ("500-memitw", "1000-memitw")] + [ML[L(g)]["beh"]["rescope"]["igen"] for g in ("500-memitw", "1000-memitw")]
add("L rephrases", "I-Gen exact on L500-memitw / L1000-memitw: white_1_shared (83, 77) and rescope (56, 55)", [83, 77, 56, 55], [round(x) for x in ig], 0.5)
H3T = {"200-ae-l2_3": (34.8, 43.5, 39.5, 39.3, 50.8), "500-ae-l2_3": (35.0, 37.3, 37.2, 31.2, 53.1), "500-memitw": (22.8, 23.2, 37.2, 31.2, 53.1), "1000-memitw": (24.8, 20.2, 35.8, 24.1, 52.3)}
for log, e in H3T.items():
    g = ML[L(log)]["gate"]
    add("L H3", f"L{log}: H3 coverage, gate level, 20 draws: rescope, rescope_x, white_1_shared, white_1_x, retr_sum", list(e), [r1(g[k]["h3cov"]) for k in VI.H3L], 0.051,
        "unreachable draws (count 0): " + ", ".join(f"{k} {g[k]['h3_n_unreachable']}" for k in VI.H3L) + "; edits at cap (mean): " + ", ".join(f"{k} {g[k]['h3_n_at_cap']:.1f}" for k in VI.H3L))
add("L capped", "rescope, capped edits (20 draws): L100-memitw, L500-memitw, L1000-memitw", [19, 91, 149], [round(ML[L(g)]["gate"]["rescope"]["capped"]) for g in ("100-memitw", "500-memitw", "1000-memitw")], 0.5)
add("L capped", "rescope_x, capped edits: L100-memitw, L500-memitw, L1000-memitw", [24, 171, 411], [round(ML[L(g)]["gate"]["rescope_x"]["capped"]) for g in ("100-memitw", "500-memitw", "1000-memitw")], 0.5)
ae_ = [ML[L(g)]["_margin"]["min"] for g in ("100-ae-l2_3", "200-ae-l2_3", "500-ae-l2_3")]
add("L capped", "smallest self-separation margin: memitw (min over logs); ae (min / max over logs)", [0.02, 0.39, 0.53],
    [round(min(ML[L(g)]["_margin"]["min"] for g in ("100-memitw", "500-memitw", "1000-memitw")), 2), round(min(ae_), 2), round(max(ae_), 2)], 0.0051)
hx = [ML[g]["beh"]["rescope_x"]["hn_D1"] for g in LL_]
dx = [ML[g]["beh"]["rescope"]["igen"] - ML[g]["beh"]["rescope_x"]["igen"] for g in LL_]
add("L near scope", "rescope_x: HN D1 min / max (%); I-Gen cost against rescope min / max (points)", [4, 8, 15, 20], [round(min(hx)), round(max(hx)), round(min(dx)), round(max(dx))], 0.5)
a, b = ML[L("100-ae-l2_3")]["beh"], ML[L("500-ae-l2_3")]["beh"]
add("L retention", "first 100 edits, ae T=100 -> 500: deployed Rel; where_logonly Rel; rescope Rel; rescope T-Gen", [88, 48, 92, 92, 98, 98, 38, 18],
    [round(x) for x in (a["deployed"]["rel_first"], b["deployed"]["rel_first"], a["where_logonly"]["rel_first"], b["where_logonly"]["rel_first"], a["rescope"]["rel_first"], b["rescope"]["rel_first"],
                        a["rescope"]["tgen_first"], b["rescope"]["tgen_first"])], 0.5)
g2, g5 = ML[L("200-ae-l2_3")]["gate"], ML[L("500-ae-l2_3")]["gate"]
add("L MME-T", "rescope on MME-T: L200-ae mean, min, max over draws; L500-ae mean", [14.9, 6.6, 28.2, 6.7],
    [r1(g2["rescope"]["mme"]), r1(min(g2["rescope"]["mme_by_draw"])), r1(max(g2["rescope"]["mme_by_draw"])), r1(g5["rescope"]["mme"])], 0.051)
add("L MME-T", "rescope_x on MME-T: L200-ae, L500-ae", [23.9, 14.3], [r1(g2["rescope_x"]["mme"]), r1(g5["rescope_x"]["mme"])], 0.051)
add("L counts", "non-oracle targets at T = 100, 200, 500, 1000", [2, 4, 6, 10], [len(ML[L(g)]["_stored"]["non_oracle"]) for g in ("100-ae-l2_3", "200-ae-l2_3", "500-ae-l2_3", "1000-memitw")], 0)
add("L counts", "invalid probe rows excluded (every log)", 12, ML[L("500-ae-l2_3")]["_ctx"]["n_probe_rows"] - ML[L("500-ae-l2_3")]["_ctx"]["n_probe_valid"], 0)
ns, nf, nall = [], [], []
for g in LL_:
    per = []
    for d in VI.draws_of(g):
        tp = VI.rows("llava", g, "rescope", d, "tfp")
        tf = {r["sid"]: r for r in VI.rows("llava", g, "rescope", d, "tf")}
        ns.append(len(tp))
        per.append(sum(bool(r["fired"]) != bool(tf[r["sid"]]["fired"]) for r in tp))
    nf.append(float(np.mean(per)))
    nall += per
add("L shifted", "sequences with a shifted decode position per log: min / max", [28, 52], [min(ns), max(ns)], 0)
add("L shifted", "rescope: firing decisions that differ at the decode's position, mean over the draws of a log: min / max over logs", [6, 20], [round(min(nf)), round(max(nf))], 0.5,
    f"single draws: {min(nall)} to {max(nall)}")
hi = [ML[g]["gate"]["retr_sum"]["hn_HN-I"] for g in LL_]
add("L retr", "retr_sum: T-Gen exact min / max over the five unseen logs; gate firing hd min / max; HN-I firing min / max (%)", [58, 83, 6, 7, 40, 60],
    [round(min(ML[g]["beh"]["retr_sum"]["tgen"] for g in unl)), round(max(ML[g]["beh"]["retr_sum"]["tgen"] for g in unl)), round(min(ML[g]["gate"]["retr_sum"]["fire"] for g in LL_)),
     round(max(ML[g]["gate"]["retr_sum"]["fire"] for g in LL_)), round(min(hi)), round(max(hi))], 0.5, "T-Gen is 89 on L100-memitw (same row as the development log)")

# ------------------------------------------------------------------ not in the summaries (computed here from the per-row files; no experiment value to compare)
EXTRA = {"LLaVA H2 (rescope minus white_0.1_shared, rephrase exact, all edits; my bootstrap)": {g[3:]: [r1(x) for x in h2("llava", g, "white_0.1_shared")] for g in unl},
         "LLaVA H2, later edits": {g[3:]: [r1(x) for x in h2("llava", g, "white_0.1_shared", np.arange(100, VI.T_of(g)))] for g in unl if VI.T_of(g) > 100},
         "LLaVA: rescope minus DEPLOYED, rephrase exact": {g[3:]: r1(ML[g]["beh"]["rescope"]["reph"] - ML[g]["beh"]["deployed"]["reph"]) for g in LL_},
         "Qwen H3 pair 2 (rescope_x minus white_1_x), behaviour": {g[3:]: r1(h3b("qwen", g, "rescope_x", "white_1_x")) for g in unseen},
         "Qwen H3 against retr_sum (rescope minus retr_sum), behaviour": {g[3:]: r1(h3b("qwen", g, "rescope", "retr_sum")) for g in unseen},
         "LLaVA H3 pair 1 / pair 2 / retr, behaviour": {g[3:]: [r1(h3b("llava", g, "rescope", "white_1_shared")), r1(h3b("llava", g, "rescope_x", "white_1_x")), r1(h3b("llava", g, "rescope", "retr_sum"))] for g in unl},
         "Qwen: rescope, near-scope held-out probes released (behaviour %) / gate level": {g[3:]: [r1(MQ[g]["beh"]["rescope"]["near_fired"]), r1(MQ[g]["gate"]["rescope"]["fire_near"])] for g in VI.BB["qwen"]["logs"]},
         "LLaVA: rescope, near-scope held-out probes released (behaviour %) / gate level": {g[3:]: [r1(ML[g]["beh"]["rescope"]["near_fired"]), r1(ML[g]["gate"]["rescope"]["fire_near"])] for g in LL_},
         "Qwen: rescope_null minus rescope, rephrase exact": {g[3:]: r1(MQ[g]["beh"]["rescope_null"]["reph"] - MQ[g]["beh"]["rescope"]["reph"]) for g in unseen},
         "LLaVA: rescope_null minus rescope, rephrase exact": {g[3:]: r1(ML[g]["beh"]["rescope_null"]["reph"] - ML[g]["beh"]["rescope"]["reph"]) for g in unl},
         "Qwen: rescope_shared minus rescope, rephrase exact": {g[3:]: r1(MQ[g]["beh"]["rescope_shared"]["reph"] - MQ[g]["beh"]["rescope"]["reph"]) for g in unseen},
         "Qwen: draws above 4.95 % of 20, rescope": {g[3:]: MQ[g]["gate"]["rescope"]["n_draws_above_bound"] for g in VI.BB["qwen"]["logs"]},
         "LLaVA: self-separation margin min / edits with a non-positive margin": {g[3:]: [round(ML[g]["_margin"]["min"], 3), ML[g]["_margin"]["n_nonpositive"]] for g in LL_},
         "LLaVA: BEN-T far firing (gate, 20 draws): rescope / rescope_shared / rescope_x": {g[3:]: [r1(ML[g]["gate"][k]["ben_far"]) for k in ("rescope", "rescope_shared", "rescope_x")] for g in LL_}}
json.dump(EXTRA, open(V + "/out/claims_extra.json", "w"), indent=1)

json.dump(ROWS, open(V + "/out/claims.json", "w"), indent=1)
lines = [f"{'claim':14s} {'agree':5s}  quantity | experiment | mine"]
for r in ROWS:
    lines.append(f"{r['claim']:14s} {'yes' if r['agree'] else 'NO':5s}  {r['quantity']} | {r['experiment']} | {r['mine']}" + (f"   ({r['note']})" if r["note"] else ""))
lines.append(f"\n{sum(r['agree'] for r in ROWS)} of {len(ROWS)} rows agree")
open(V + "/out/claims_table.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(l[:330] for l in lines if " NO " in l or "rows agree" in l))
for k, v in EXTRA.items():
    print(k, "::", v)
