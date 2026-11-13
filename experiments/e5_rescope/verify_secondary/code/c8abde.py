"""Check 8 (a) three exact McNemar tests, (b) five added paper numbers, (d) modification times and name collisions, (e) the H1 record of the fix-up."""
from vcommon import *
import gzip, re
from math import comb
from fractions import Fraction
def rd(p):
    with gzip.open(p, "rt") as f: return [json.loads(l) for l in f]
E1 = f"{FINAL}/e1_main"; E2 = f"{FINAL}/e2_curves"
res = {}
# ------------------------------------------------------------------ (a)
def mcnemar(b, c):
    n, k = b + c, min(b, c)
    if n == 0: return 1.0
    return float(min(Fraction(1), 2 * sum(Fraction(comb(n, i), 2 ** n) for i in range(k + 1))))
def holm(ps):
    o = np.argsort(ps, kind="stable"); m = len(ps); adj = np.zeros(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * ps[i])); adj[i] = run
    return adj
MC = jload(f"{E1}/out/fixup/mcnemar.json")
def flags(log, label, d, outcome):
    p = f"{E1}/out/rows/{log}__{label}__d{d}__tf.jsonl.gz"
    if not os.path.exists(p): p = f"{E1}/out/rows/{log}__{label}__dx__tf.jsonl.gz"
    rows = rd(p)
    if outcome == "D1": return {r["csid"]: bool(r["D1"]) for r in rows if r["kind"] == "probe" and r.get("disjoint")}
    return {r["csid"]: bool(r["exact"]) for r in rows if r["kind"] == "edit" and r["metric"] == "rel"}
tests = []
for log, row, d, outcome in ((QDEV, "deployed", 0, "D1"), (LDEV, "retr_sum", 1, "D1"), (LDEV, "deployed", 0, "Rel")):
    a, o = flags(log, "rescope", d, outcome), flags(log, row, d, outcome)
    assert set(a) == set(o)
    b = sum(a[k] and not o[k] for k in a); c = sum(o[k] and not a[k] for k in a)
    p = mcnemar(b, c)
    fam = [(x["log"], x["row"], y["draw"], y["p"]) for x in MC["comparisons"] if x["outcome"] == outcome for y in x["per_draw"]]
    ps = np.array([p if (l, r, dd) == (log, row, d) else q for l, r, dd, q in fam])
    i = [j for j, (l, r, dd, q) in enumerate(fam) if (l, r, dd) == (log, row, d)][0]
    st = [y for x in MC["comparisons"] if x["log"] == log and x["row"] == row and x["outcome"] == outcome for y in x["per_draw"] if y["draw"] == d][0]
    t = {"log": log, "row": row, "draw": d, "outcome": outcome, "n_pairs": len(a), "mine": {"b": b, "c": c, "p": p, "p_holm_all_tests": float(holm(ps)[i]), "family_size": len(ps)},
         "stored": {k: st[k] for k in ("b", "c", "n", "p", "p_holm_all_tests")}}
    t["ok"] = bool(b == st["b"] and c == st["c"] and len(a) == st["n"] and abs(p - st["p"]) <= 1e-12 * max(1, st["p"]) + 1e-300 and abs(t["mine"]["p_holm_all_tests"] - st["p_holm_all_tests"]) <= 1e-9 * max(st["p_holm_all_tests"], 1e-300) + 1e-300)
    tests.append(t); print("(a)", t)
res["a_mcnemar"] = tests
# ------------------------------------------------------------------ (b)
PF = jload(f"{E1}/out/fixup/paper_numbers_fixup.json"); nums = []
g = rd(f"{E1}/out/rows/{LDEV}__white_1_shared__d0__gen.jsonl.gz")
hn = [r for r in g if r["kind"] == "hn" and r["hn_type"] == "HN-I"]
nums.append(("EoneFxLBestCosHNIWrongTarget (wrong-target rate, HN-I, white_1_shared, draw 0)", 100 * float(np.mean([bool(r["wrong_target"]) for r in hn])), PF["EoneFxLBestCosHNIWrongTarget"], len(hn)))
gd = rd(f"{E1}/out/rows/{LDEV}__deployed__dx__gen.jsonl.gz")
pc = [r for r in gd if r["kind"] == "probe" and r.get("disjoint") and r.get("pre_correct")]
nums.append(("EoneFxLDeployedDtwo (D2 rate of pre-correct heldout_disjoint rows, deployed)", 100 * float(np.mean([bool(r["D2"]) for r in pc])), PF["EoneFxLDeployedDtwo"], len(pc)))
tg = [r for r in g if r["kind"] == "edit" and r["metric"] == "tgen"]
nums.append(("EoneFxLBestCosGenTGenEq (generated T-Gen answer equals the target, white_1_shared, draw 0)", 100 * float(np.mean([bool(r["eq"]) for r in tg])), PF["EoneFxLBestCosGenTGenEq"], len(tg)))
bl = [r for r in rd(f"{E1}/out/rows/{LDEV}__ball__d1__tf.jsonl.gz") if r["kind"] == "hn" and r["hn_type"] == "HN-Q"]
nums.append(("EoneFxLBallHNQFireDOne (HN-Q firing, ball, calibration draw 1)", 100 * float(np.mean([bool(r["fired"]) for r in bl])), PF["EoneFxLBallHNQFireDOne"], len(bl)))
e2g = [r for r in rd(f"{E2}/out/rows/gen__L__d__COMP_B__nALL__s0__gen.jsonl.gz") if r["kind"] == "edit" and r["metric"] == "tgen"]
nums.append(("EoneFxLRowCompBAllGenTGenEq (row taken from E2: COMP-B at all complaints, generated T-Gen equals target)", 100 * float(np.mean([bool(r["eq"]) for r in e2g])), PF["EoneFxLRowCompBAllGenTGenEq"], len(e2g)))
res["b_paper_numbers"] = [{"name": n, "mine": m, "stored": s, "n_rows": k, "ok": bool(abs(m - s) < 1e-9)} for n, m, s, k in nums]
for x in res["b_paper_numbers"]: print("(b)", x)
# ------------------------------------------------------------------ (d)
def walk(root):
    for dp, dn, fn in os.walk(root):
        for f in fn:
            p = os.path.join(dp, f)
            try: yield p, os.path.getmtime(p)
            except OSError: pass
dres = {}
for exp, root in (("E1", E1), ("E2", E2)):
    fix = f"{root}/out/fixup"
    t0 = min(os.path.getmtime(os.path.join(fix, f)) for f in os.listdir(fix))
    own = lambda p: p.startswith(fix + "/") or re.search(r"/code/(__pycache__/)?fixup_", p)
    code_t = [os.path.getmtime(os.path.join(root, "code", f)) for f in os.listdir(f"{root}/code") if f.startswith("fixup_")]
    t_first = min(code_t + [os.path.getctime(fix)])
    files = list(walk(root))
    later = sorted([(p, t) for p, t in files if t > t0 and not own(p)], key=lambda x: x[1])
    later_first = sorted([(p, t) for p, t in files if t > t_first and not own(p)], key=lambda x: x[1])
    fmt = lambda t: time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t))
    a, b = jload(f"{root}/out/paper_numbers.json"), jload(f"{fix}/paper_numbers_fixup.json")
    dres[exp] = {"oldest_file_of_fixup_directory_utc": fmt(t0), "earliest_trace_of_the_fixup_utc (fix-up code / directory)": fmt(t_first), "n_files_checked": len(files),
                 "existing_files_modified_after_the_oldest_fixup_file": [(p.replace(FINAL + "/", ""), fmt(t)) for p, t in later],
                 "existing_files_modified_after_the_earliest_trace": [(p.replace(FINAL + "/", ""), fmt(t)) for p, t in later_first][:30],
                 "newest_non_fixup_file_utc": fmt(max(t for p, t in files if not own(p))), "n_paper_numbers": len(a), "n_fixup_numbers": len(b), "name_collisions": sorted(set(a) & set(b))}
    print("(d)", exp, {k: (v if not isinstance(v, list) else (len(v), v[:6])) for k, v in dres[exp].items()})
res["d_mtimes"] = dres
# ------------------------------------------------------------------ (e)
h = jload(f"{E2}/out/fixup/h1_registered.json")
prot = re.sub(r"\s+", " ", open(f"{FINAL}/PROTOCOL.md").read())
q = re.sub(r"\s+", " ", h["rule_as_registered"]["quote"])
c6 = jload(f"{OUT}/c6_prefix.json")
dev_ok = {}
for lg, v in h["development_logs"].items():
    m = c6["h1_development_logs_from_rows"][lg]
    dev_ok[lg] = bool(abs(v["repair"]["D1_heldout_disjoint"]["mean"] - m["D1_repair"]) < 1e-9 and abs(v["repair"]["rel_exact"]["mean"] - m["rel_repair"]) < 1e-9 and abs(v["deployed_log"]["rel_exact"] - m["rel_deployed"]) < 1e-9
                       and abs(v["deployed_log"]["D1_heldout_disjoint"] - m["D1_deployed"]) < 1e-9 and v["two_sided_reading"]["met_on_this_log"] == m["two_sided_met"] and v["one_sided_reading"]["met_on_this_log"] == m["one_sided_met"])
pf = f"{FINAL}/e2_prefix/out/h1_prefix.json"
res["e_h1"] = {"rule_quoted_verbatim_from_PROTOCOL": q in prot, "development_log_numbers_and_readings_equal_my_recomputation_from_rows": dev_ok,
               "h1_json_agrees": {lg: bool(abs(jload(f"{E2}/out/h1.json")["logs"][lg]["d1_where"] - c6["h1_development_logs_from_rows"][lg]["D1_repair"]) < 1e-9 and abs(jload(f"{E2}/out/h1.json")["logs"][lg]["rel_where"] - c6["h1_development_logs_from_rows"][lg]["rel_repair"]) < 1e-9
                                             and jload(f"{E2}/out/h1.json")["logs"][lg]["helps_on_this_log"] == c6["h1_development_logs_from_rows"][lg]["two_sided_met"]
                                             and jload(f"{E2}/out/h1.json")["logs"][lg]["one_sided_reading_met_on_this_log"] == c6["h1_development_logs_from_rows"][lg]["one_sided_met"]) for lg in dev_ok},
               "prefix_part_of_record": {"source_file_modified": h["prefix_part"]["source"]["modified"], "n_complete": h["prefix_part"]["n_complete"], "label": h["prefix_part"]["label"]},
               "final_h1_prefix_json_modified_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(os.path.getmtime(pf))), "record_older_than_final_prefix_file": True,
               "final_outcome_from_rows": c6["h1_final"],
               "prefix_details": {p: {k: v[k] for k in ("D1_repair", "rel_repair", "D1_deployed", "rel_deployed", "rel_diff", "two_sided_met", "one_sided_met")} for p, v in c6["h1"].items()}}
print("(e)", json.dumps(res["e_h1"], indent=0)[:1500])
jdump(res, OUT + "/c8abde.json")
