#!/usr/bin/env python3
"""Statements of Sec. 5 about experiment E10 (public text logs), asserted against the experiment's JSON.

    python3 scripts/check_e10_statements.py        exit 1 if a statement does not hold (part of `make checks`)

Every statement of direction or size that the text of Sec. 5.2 (paragraph "Text logs"), Sec. 5.5 and the first paragraph
of Sec. 5.7 makes, and that the captions of Fig. fig:evidence, Table tab:scale (b) and Table tab:maint (a) make, is
listed below with the sentence it backs. The numbers themselves are macros (scripts/numbers_manifest.json) or cells
of generated tables (scripts/gen_tab_e10.py); this script checks the WORDS around them ("add little", "close
together", "best", "no row", "only", "as registered", ...). Sources: 02_research/experiments/10_final/e10_text/run
(out/summary, out/gate, out/diag_eta.json, out/screening.json, out/cert, verify/out/screening_T2000.json,
verify/verdicts.json) and the audit of the logs (09_testbed_scale/text_public/audit/out/misc.json).
"""
import json
import re
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen_tab_e10 as g                                                      # noqa: E402

ROOT = g.find_root()
RUN = os.path.join(ROOT, g.E10)
D = g.load(ROOT)
F = g.facts(D)
fails, n = [], 0


def ok(cond, sentence, detail=""):
    global n
    n += 1
    print(("  ok    " if cond else "  FAIL  ") + sentence + (f"   [{detail}]" if detail else ""))
    if not cond:
        fails.append(sentence)


def bm(tag, row, key):
    return D[tag]["behave"][row]["metrics"][key]["value"]


def eq(tag, key, metric):
    return D[tag]["equal"][key]["metrics"][metric]["value"]


def r22(tag, label, m=None):
    return g.r22row(D, tag, label, m)


ALPHA = 5.0
print("RQ1, text logs at equal evidence (Sec. 5.2; Fig. fig:evidence (c))")
dep, lo = bm("TOne", "deployed", "heldout.D1_tf"), bm("TOne", "where_logonly", "heldout.D1_tf")
ls = [eq("TOne", f"least_sacrificing|n{k}", "heldout.D1_tf") for k in g.NS]
ok(ls[0] <= lo + 1e-9 and ls[0] >= ls[1] >= ls[2], "complaints lower D1 monotonically from the log-only value", f"{lo:.1f} -> {ls}")
ok((dep - lo) / (dep - ls[2]) > 0.9, "'complaints add little to the log-only thresholds': the log-only thresholds give more than 90 % of the "
   "reduction reached after 100 complaints", f"{(dep - lo) / (dep - ls[2]):.3f}")
ok(all(g.eq_lost(D["TOne"]["equal"][f"least_sacrificing|n{k}"]) == 0 for k in g.NS), "'no edit lost' (clean log, least repair, every n)")
lin = [eq("TOne", f"{r}|n{k}", "heldout.D1_tf") for r in ("comp_min", "rb_lineage") for k in g.NS]
ok(min(lin) > 3 * max(ls), "compensation and lineage rollback leave far more prompts changed than the least repair at every n",
   f"{min(lin):.1f}..{max(lin):.1f} against {max(ls):.1f}")
pl = [g.eq_lost(D["TOne"]["equal"][f"rb_suffix|n{k}"]) for k in g.NS]
ok(min(pl) > 69 and max(pl) == 100, "'the resolving prefix deletes 70 to 100 of the 100 edits'", str(pl))
ok(F["equal_TFive_protecting_refused"]["value"] == F["equal_TFive_protecting_reported"]["value"] > 0,
   "caption: the protecting form refuses all complaints on the collapsed log")
for tag in g.LOGS:
    for k in (10, 100):
        c, r = F[f"equal_{tag}_least_n{k}_certified"]["value"], F[f"equal_{tag}_least_n{k}_reported"]["value"]
        ok(0 < c <= r, f"caption: certified as served {c} of {r} ({tag}, n = {k}): a count, not 'every'")

print("RQ4 (Sec. 5.5): the stream around the split (audit)")
P = json.load(open(os.path.join(ROOT, "02_research/experiments/09_testbed_scale/text_public/audit/out/misc.json")))["K"]["points"]
split = json.load(open(os.path.join(RUN, "out/split.json")))["split_position"]
ok(P[2]["n_requests"] == split and P[3]["n_requests"] == split + 1, "the audit's points 2 and 3 are the prefixes of 173 and 174 requests")
ok(P[2]["audit_first_100_requests_exact_pct"] > 90 and P[3]["audit_first_100_requests_exact_pct"] < 5, "'collapses the deployed model': first 100 requests")
ok(P[2]["audit_200_prompts"]["D1_tf"] >= 95, "'a model that has already drifted': D1 before the collapse", str(P[2]["audit_200_prompts"]))
ok(D["screen"]["logs"][g.LOGS["TFive"]]["flagged_detail"][1]["edit"] == split and D["screen"]["logs"][g.LOGS["TFive"]]["flagged_detail"][1]["solver_pos"] == 0,
   "update 173 is a request whose subject is the first prompt token (solver position 0)")

print("RQ4: clean log")
ok(lo < dep / 4, "'log-only thresholds lower D1'", f"{dep:.1f} -> {lo:.1f}")
ff_lo = g.ff(D, "TOne", "where_logonly")
ok(ff_lo > 50, "'although an edit still fires on ... of benign prompts' (more than half)", f"{ff_lo:.1f}")
ff_tb = g.ff(D, "TOne", "where_twoblock")
ok(ff_tb < ALPHA and bm("TOne", "where_twoblock", "heldout.D1_tf") < lo, "'at false firing ..., below alpha'", f"{ff_tb:.2f}")
ok(bm("TOne", "where_twoblock", "edit.paraphrase.exact") < bm("TOne", "deployed", "edit.paraphrase.exact") - 10, "'the repair loses paraphrases'")
close = ["rescope", "cosine", "whitecos_0.1", "cosine_published"]
ok(all(bm("TOne", r, "edit.request.exact") == 100 for r in close)
   and F["clean_close_par_max"]["value"] - F["clean_close_par_min"]["value"] < 3
   and F["clean_close_done_max"]["value"] - F["clean_close_done_min"]["value"] < 4,
   "'ReScope and the cosine gates lie close together' (all requests; paraphrases within 3 points; D1 within 4 points)",
   f"par {F['clean_close_par_min']['value']:.1f}..{F['clean_close_par_max']['value']:.1f}, D1 "
   f"{F['clean_close_done_min']['value']:.1f}..{F['clean_close_done_max']['value']:.1f}")
rows = [r for r in D["TOne"]["behave"] if r not in ("deployed", "preedit")]
lk = "lookup_subject"
ok(all(bm("TOne", lk, "edit.request.exact") >= bm("TOne", r, "edit.request.exact") for r in rows)
   and all(bm("TOne", lk, "edit.paraphrase.exact") >= bm("TOne", r, "edit.paraphrase.exact") for r in rows)
   and all(bm("TOne", lk, "heldout.D1_tf") <= bm("TOne", r, "heldout.D1_tf") for r in rows if bm("TOne", r, "edit.paraphrase.exact") > 50),
   "'the keyed store ... is best' (requests and paraphrases at least those of every row; lowest D1 among rows that answer paraphrases)")

print("RQ4: collapsed log")
gon = D["TFive"]["gate"]
ok(all(gon[r]["heldout.fired"]["mean"] == 100 for r in ("where_twoblock", "rescope", "where_logonly")),
   "'with guards the frozen default fails ...: fire on every benign prompt' (false firing 100 %, every draw)")
first = set(D["screen"]["logs"][g.LOGS["TFive"]]["against"]["first_token_subject"]["reference"])
cal = D["TFive"]["behave"]["rescope"]["cal"]
ok(all(n_c == 500 or first <= set(c) for n_c, c in zip(cal["n_edits_capped"], cal["capped_edits"])),
   "'the first-token updates are capped in every draw' (behavioural draws 0-2 of ReScope with guards)",
   f"first-token subjects {sorted(first)}; capped per draw {cal['n_edits_capped']}")
ok(F["cert_TFive_rows_certifying_nothing"]["value"] == ["rescope", "where_twoblock"], "caption: with guards, ReScope-W and ReScope certify no prompt")
pdr = D["TFive"]["behave"]["rescope@goff"]["per_draw"]["edit.request.exact.before"]
v = [pdr[k] for k in sorted(pdr, key=int)]
ok(min(v) < 1 and sorted(v)[1] > 90, "'bimodal': one behavioural draw near 0, two above 90", str([round(x, 1) for x in v]))
sil = F["goff_silenced_sorted"]["value"]
ok(F["goff_draws_majority_silenced"]["value"] == 3 and sil[2] >= 311 and sil[3] <= 27, "'in 3 of 20 gate-level draws ... silences at least 311 of 500'", str(sil[:5]))
eta = json.load(open(os.path.join(RUN, "out/diag_eta.json")))["logs"][g.LOGS["TFive"]]["per_draw"]
gd = D["TFive"]["gate_draws"]["rescope"]["goff"]["per_draw"]
bad = [d for d in gd if round(500 * (100 - gd[d]["request"]["own_released"]) / 100) > 250]
by = {str(e["draw"]): e for e in eta}
expl = set(D["screen"]["logs"][g.LOGS["TFive"]]["against"]["exploding_norm"]["reference"])
ok(len(bad) == 3 and all(by[d]["eta_without_exploding"] < 0.5 * by[d]["eta"] and expl & set(by[d]["detectors_at_or_above_eta"]) for d in bad),
   "'exploding detectors raise the common scale so far that ...': in these three draws the scale is more than twice what it would be "
   "without the exploding detectors, and an exploding detector is among those that set it",
   str({d: (round(by[d]['eta'], 3), round(by[d]['eta_without_exploding'], 3)) for d in bad}))
ok(bm("TFive", "where_logonly@goff", "edit.request.exact.before") > 80 and 35 < bm("TFive", "where_logonly@goff", "heldout.D1_tf") < 50,
   "'log-only thresholds without guards keep ... of these requests, but ... of held-out continuations still change'")
for r in ("cosine", "whitecos_0.1", "lookup_subject"):
    ok(bm("TFive", r, "edit.request.exact.before") > 98 and bm("TFive", r, "heldout.D1_tf") < 8,
       f"'{r}' keeps nearly all requests before the split at a few percent of changed continuations",
       f"{bm('TFive', r, 'edit.request.exact.before'):.1f} / {bm('TFive', r, 'heldout.D1_tf'):.1f}")
ok(F["from_split_max"]["value"] < 25 and "rb_" in F["from_split_max_row"]["value"], "'no row recovers the requests from the split on (at most ..., a rollback)'",
   f"{F['from_split_max']['value']:.1f}: {F['from_split_max_row']['value']}")
# final revision of 2026-11-12 (TRIAGE_FINAL.txt, T33): 'certificate and bound are stated for either detector; here the two cosine
# rows fire on 6.7 and 6.6 % of benign prompts, above the level' (Table tab:scale (b), column FF: gate level, mean of 20 draws)
_ffc = [gon[r]["heldout.fired"]["mean"] for r in ("cosine", "whitecos_0.1")]
ok(all(ALPHA < x < 10 for x in _ffc), "'here the two cosine rows fire on ... % of benign prompts, above the level' (collapsed log, with guards)", str([round(x, 2) for x in _ffc]))
_t5 = " ".join("\n".join(re.split(r"(?<!\\)%", l)[0] for l in open(os.path.join(os.path.dirname(HERE), "sec_experiments.tex"))).split())
ok("certificate and bound are stated for either detector" in _t5 and "calibration bound hold for either" not in _t5, "Sec. 5.5 no longer says that the bound 'holds' for the cosine detectors on the collapsed log")
# T10 (Sec. 1): 'where a request has a string key [the keyed store] is the best row of our tables (Secs. 5.5, 5.7)'. Clean log:
# the two checks 'is best' above and below. Collapsed log: lowest D1 and most paraphrases among the rows that keep more than
# half of the requests before the split; in the relational workload the highest reference preservation among the rows that
# retain more than half of the earlier requests.
_rows5 = [r for r in D["TFive"]["behave"] if r not in ("deployed", "preedit") and bm("TFive", r, "edit.request.exact.before") > 50]
ok(lk in _rows5 and all(bm("TFive", lk, "heldout.D1_tf") <= bm("TFive", r, "heldout.D1_tf") and bm("TFive", lk, "edit.paraphrase.exact.before") >= bm("TFive", r, "edit.paraphrase.exact.before") for r in _rows5),
   "Sec. 1 (T10): on the collapsed log the keyed store has the lowest D1 and the most paraphrases among the rows that keep more than half of the requests before the split", str(len(_rows5)) + " rows")
_good5 = [r for r in D["TFive"]["r22"]["rows"] if r["retention_before"]["value"] >= 50]
ok(all(r22("TFive", lk)["reference_preservation"]["value"] >= r["reference_preservation"]["value"] for r in _good5) and len(_good5) >= 5,
   "Sec. 1 (T10): relational workload, collapsed log: the keyed store has the highest reference preservation among the rows that retain at least half of the earlier requests")
ok(F["collapsed_cosine_goff_max_change"]["value"] < 1.0, "caption: the calibrated cosine rows move by less than one point without guards",
   f"{F['collapsed_cosine_goff_max_change']['value']:.2f}")

print("RQ4: screening and certificate")
ok(F["screen_TOne_flagged"]["value"] == 0 and F["screen_TFive_ties"]["value"] == 0, "none flagged on the clean log; no tie among the flagged updates of the collapsed log")
ok(F["screen_TFive_first_tp"]["value"] == F["screen_TFive_first_ref"]["value"], "'all ... first-token subjects' (collapsed log)")
ok(F["screen_TTwoThousand_negative"]["value"] + 8 == len(D["vscreen"]["logs"]["TX-G2XL-rome-T2000"]["flagged"]),
   "2000 requests: the verifier's 31 flagged edits = negative margins + 8 flagged ties")
ok(F["cert_violations_total"]["value"] == 0, "certificate: no violation on any certified decode")

print("RQ6, relational workload (Sec. 5.7; Table tab:maint (a))")
q = lambda tag, lab, m=None: r22(tag, lab, m)["queries_exact_pct"]["value"]
ok(q("TOne", "deployed") == 0, "'deployed: none'")
a, b3, b10, c = q("TOne", "ReScope-W log-only"), q("TOne", "ReScope-W least repair (sacrificing)", 3), \
    q("TOne", "ReScope-W least repair (sacrificing)", 10), q("TOne", "ReScope-W two-block")
ok(a < b3 < b10 < c and (b10 - a) < (c - b10), "'again complaints add little': 10 complained queries add less than the benign queries add on top",
   f"{a:.1f} < {b3:.1f} < {b10:.1f} < {c:.1f}")
rows1 = D["TOne"]["r22"]["rows"]
ok(rows1[13]["label"] == rows1[19]["label"] == "ReScope-W least repair (sacrificing)" and (rows1[13]["m"], rows1[19]["m"]) == (3, 10),
   "the manifest's rows 13 and 19 of the clean log are the sacrificing least repair after 3 and 10 complained queries")
ok(F["rel_TOne_linear_q_max"]["value"] < a, "compensation, lineage rollback and blocklist stay below the log-only thresholds", f"{F['rel_TOne_linear_q_max']['value']:.1f}")
ok(all(q("TOne", lk) >= r["queries_exact_pct"]["value"] for r in D["TOne"]["r22"]["rows"]), "'the keyed store on the requests is best' (clean log)")
good = [r for r in D["TFive"]["r22"]["rows"] if r["reference_preservation"]["value"] >= 90 and r["retention_before"]["value"] >= 50]
ok(all(r["guards"] is False or "sacrificing" in r["label"] or r["label"] == lk for r in good) and len(good) >= 5,
   "'only rows without guards and the keyed store restore the reference and retain earlier requests' "
   "(reference preservation >= 90 and retention before the split >= 50)", str(sorted({r["label"] for r in good})))
ok(r22("TFive", "oracle: delete update 173")["reference_preservation"]["value"] < 5 and r22("TFive", "rb_screen")["reference_preservation"]["value"] < 30,
   "'deleting update 173 or the screened updates does not'")
ok(F["rel_TFive_q_max"]["value"] == 0, "caption: no query result is exactly the intended one in any row of the collapsed log")

print(f"{n} statements checked, {len(fails)} failed")
sys.exit(1 if fails else 0)
