#!/usr/bin/env python3
"""Final check of 2026-11-11, task 3: Table tab:logs (tab_logs.tex, macros \\lg<tag>... of numbers.tex) against the stored
files of the testbed, recomputed with code that shares nothing with scripts/collect_setup.py.

    python3 checks/check_tab_logs_rows.py            writes checks/check_tab_logs_rows.json; exit 1 on a mismatch

For EVERY row of the table (the task asks for three; all 18 are cheap) the script reads
  02_research/experiments/06_testbed_v3/<log>/per_edit_metrics.jsonl   T = number of edits; Rel / T-Gen / I-Gen = share of
                                                                       the per-edit flags post_{rel,tgen,igen}_exact,
                                                                       as an exact fraction (NOT the summary of
                                                                       metrics.json, which the collector copies)
  .../<log>/metrics.json                                               T-Loc / I-Loc: TLoc_probes, ILoc_probes (the testbed
                                                                       stores no per-probe locality, so these two cells
                                                                       can only be compared with the stored aggregate;
                                                                       cross-check: the line "Loc probes T=.. I=.." of
                                                                       run.log, written by the build itself)
  .../<log>/probe_records_v2.jsonl                                     D1 on Held-out: own implementation of the split
and compares each value, rounded to one decimal on the EXACT value (a tie is rounded away from zero, as by hand), with
the text that the macro prints.

Split (own code, from the definitions of 07_repair_v3/harness/sampler.py and rescope/splits.py, not imported):
  probe ITEMS = sorted distinct uids of the probe rows; permutation numpy.random.default_rng(20261109); the first half
  is the REPAIR pool; a held-out row is DISJOINT if its query (ptype, prompt, image) is the query of no REPAIR row;
  D1 rate = rows with is_complaint_D1 / rows with y0_valid, among the disjoint held-out rows (the table's column D1).
Rounding: a value such as 91 / 2000 = 4.55 % is a tie. Python's '{:.1f}'.format(4.55) prints 4.5 (the binary double is
below 4.55); the exact value rounds to 4.6. The first run of this check (2026-11-11) found exactly this cell
(\\lgQWmmIGen, log TB-Q2000-memitw) printed as 4.5; scripts/gen_numbers.py now rounds on the decimal value
(function fmt_number), and the script reports every cell where printed and exact value differ.
"""
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP
import json
import os
import re
import sys
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
ROOT = _RP.ROOT
TB = os.path.join(ROOT, "02_research/experiments/06_testbed_v3")
SEED = 20261109


def macros():
    out = {}
    for m in re.finditer(r"(?m)^\\newcommand\{\\([A-Za-z]+)\}\{(.*?)\}  % ", open(os.path.join(PAPER, "numbers.tex")).read()):
        out[m.group(1)] = m.group(2).replace("\\xspace", "")
    return out


def table_rows():
    """(tag, testbed name) of every row of tab_logs.tex, in order."""
    rows = []
    for line in open(os.path.join(PAPER, "tab_logs.tex")):
        m = re.search(r"\{\\lg([A-Za-z]+)Name\}.*%\s*(TB-\S+)\s*$", line)
        if m:
            rows.append((m.group(1), m.group(2)))
    return rows


def one_decimal(fr):
    """Exact fraction (percent) -> string with one decimal, ties rounded away from zero on the EXACT value."""
    scaled = fr * 10                                           # exact: floor(10 x + 1/2) for x >= 0
    assert scaled >= 0
    tenths = (2 * scaled.numerator + scaled.denominator) // (2 * scaled.denominator)
    return str(Decimal(tenths) / Decimal(10)) if tenths % 10 else f"{tenths // 10}.0"


def jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


FINAL = os.path.join(ROOT, "02_research/experiments/10_final")


def final_rows(log):
    """Per-row flags of the DEPLOYED log as evaluated in the final experiments (experiment E3; one sequence per forward
    pass): e3_qwen / e3_llava, out/rows/<log>__deployed__dx__tf.jsonl.gz. None if the log was not evaluated there."""
    import gzip
    for exp in ("e3_qwen", "e3_llava"):
        f = os.path.join(FINAL, exp, "out", "rows", f"{log}__deployed__dx__tf.jsonl.gz")
        if os.path.exists(f):
            with gzip.open(f, "rt") as fh:
                return [json.loads(l) for l in fh]
    return None


def recompute(log):
    d = os.path.join(TB, log)
    edits = jsonl(os.path.join(d, "per_edit_metrics.jsonl"))
    T = len(edits)
    out = {"T": str(T)}
    exact = {}
    # Since 2026-11-12 (PROTOCOL.md, amendment of 06:16 UTC, decision 7; scripts/gen_vlm.py, facts logs_b1.*) the table
    # prints Rel / T-Gen / I-Gen of the deployed log AS EVALUATED IN THE FINAL EXPERIMENTS (one sequence per forward
    # pass). They are recomputed here from the per-row flags of experiment E3 (own count; not from x_release/out/
    # logs_b1.json, which the generator reads). The flags stored by the testbed at build time (padded batches) differ
    # on a few edits per metric; the difference is recorded per cell ("testbed_count") and is not a mismatch.
    fin = final_rows(log)
    for cell, key in (("Rel", "post_rel_exact"), ("TGen", "post_tgen_exact"), ("IGen", "post_igen_exact")):
        n_tb = sum(1 for e in edits if e[key] is True)
        assert all(isinstance(e[key], bool) for e in edits), (log, key)
        out[cell + "_testbed_count"] = f"{n_tb}/{T}"
        if fin is None:
            n, den = n_tb, T
        else:
            rows = [r for r in fin if r.get("kind") == "edit" and r.get("metric") == cell.lower()]
            assert len(rows) == T and all(isinstance(r["exact"], bool) for r in rows), (log, cell, len(rows))
            n, den = sum(1 for r in rows if r["exact"]), T
        exact[cell] = Fraction(100 * n, den)
        out[cell] = one_decimal(exact[cell])
        out[cell + "_count"] = f"{n}/{den}"
        out[cell + "_source"] = "testbed flags" if fin is None else "experiment E3, deployed rows (one sequence per forward pass)"
    met = json.load(open(os.path.join(d, "metrics.json")))
    assert met["n_edits"] == T, (log, met["n_edits"], T)
    for cell, key in (("TLoc", "TLoc_probes"), ("ILoc", "ILoc_probes")):
        out[cell] = one_decimal(Fraction(repr(met[key])))
        out[cell + "_stored"] = met[key]
    runlog = open(os.path.join(d, "run.log"), errors="replace").read()
    m = re.findall(r"Loc probes T=([0-9.]+) I=([0-9.]+)", runlog)
    out["run_log_loc"] = list(m[-1]) if m else None
    rows = jsonl(os.path.join(d, "probe_records_v2.jsonl"))
    uids = sorted({r["uid"] for r in rows})
    perm = np.random.default_rng(SEED).permutation(len(uids))
    repair = {uids[i] for i in perm[: len(uids) // 2]}
    qkey = lambda r: (r["ptype"], r["prompt"], r["image"])
    repair_queries = {qkey(r) for r in rows if r["uid"] in repair}
    held = [r for r in rows if r["uid"] not in repair]
    disjoint = [r for r in held if qkey(r) not in repair_queries]
    valid = [r for r in disjoint if r["y0_valid"]]
    d1 = [r for r in valid if r["is_complaint_D1"]]
    out["Cmp"] = one_decimal(Fraction(100 * len(d1), len(valid)))
    out["Cmp_count"] = f"{len(d1)}/{len(valid)}"
    out["pool"] = {"items": len(uids), "repair_items": len(repair), "heldout_rows": len(held),
                   "heldout_rows_with_a_repair_query": len(held) - len(disjoint),
                   "heldout_valid_rows_with_a_repair_query": sum(1 for r in held if qkey(r) in repair_queries and r["y0_valid"]),
                   "heldout_disjoint_valid": len(valid)}
    out["backbone"] = met["config"]["backbone"]
    return out, exact


def main():
    mac = macros()
    res = {"what": "Table tab:logs against the stored files of the testbed (own recomputation)", "rows": {}, "mismatches": [],
           "rounding_ties": []}
    for tag, log in table_rows():
        rec, exact = recompute(log)
        row = {"log": log, "paper_name": mac.get(f"lg{tag}Name"), "recomputed": rec, "printed": {}, "ok": True}
        for cell in ("T", "Rel", "TGen", "IGen", "TLoc", "ILoc", "Cmp"):
            printed = mac.get(f"lg{tag}{cell}")
            row["printed"][cell] = printed
            if printed != rec[cell]:
                row["ok"] = False
                tie = cell in exact and (exact[cell] * 10).denominator == 2
                item = {"log": log, "macro": f"\\lg{tag}{cell}", "printed": printed, "recomputed": rec[cell],
                        "exact": rec.get(cell + "_count"), "tie_rounded_by_the_binary_double": tie}
                (res["rounding_ties"] if tie else res["mismatches"]).append(item)
        # cross-check of the two locality cells with the line that the build itself wrote into run.log (two decimals)
        rl = rec.get("run_log_loc")
        row["run_log_locality_agrees"] = (None if not rl else all(
            abs(float(row["printed"][c]) - float(v)) <= 0.05 + 1e-9 for c, v in zip(("TLoc", "ILoc"), rl)))
        if row["run_log_locality_agrees"] is False:
            res["mismatches"].append({"log": log, "macro": f"\\lg{tag}TLoc / ILoc", "printed": [row["printed"]["TLoc"],
                                      row["printed"]["ILoc"]], "run_log": rl})
        res["rows"][tag] = row
    # counts that the text of Sec. 5.1 quotes: probe items, twins, valid held-out queries per backbone
    pools = {}
    for tag, row in res["rows"].items():
        b = row["recomputed"]["backbone"]
        pools.setdefault(b, set()).add(json.dumps(row["recomputed"]["pool"], sort_keys=True))
    res["pool_counts_per_backbone"] = {b: [json.loads(s) for s in sorted(v)] for b, v in pools.items()}
    res["text_macros"] = {k: mac.get(k) for k in ("prItems", "prTwins", "prQHeldValid", "prLHeldValid")}
    checks = []
    for b, name in (("qwen25vl", "prQHeldValid"), ("llava", "prLHeldValid")):
        vals = {p["heldout_disjoint_valid"] for p in res["pool_counts_per_backbone"].get(b, [])}
        checks.append({"macro": "\\" + name, "printed": mac.get(name), "recomputed_over_the_logs": sorted(vals),
                       "ok": vals == {int(mac.get(name))}})
    twins = {p["heldout_valid_rows_with_a_repair_query"] for v in res["pool_counts_per_backbone"].values() for p in v}
    checks.append({"macro": "\\prTwins", "printed": mac.get("prTwins"), "recomputed_over_the_logs": sorted(twins),
                   "ok": twins == {int(mac.get("prTwins"))}})
    items = {p["items"] for v in res["pool_counts_per_backbone"].values() for p in v}
    checks.append({"macro": "\\prItems", "printed": mac.get("prItems"), "recomputed_over_the_logs": sorted(items),
                   "ok": items == {int(mac.get("prItems"))}})
    res["text_checks"] = checks
    # Rows of the public text logs (added 2026-11-11 with the setup of the reduced programme): the printed cells come
    # from the BUILD's description of the deployed logs (text_logs.json); they are compared here with the numbers that
    # the INDEPENDENT AUDIT recomputed (text_public/audit/verdicts.json, per_log.<log>.deployed, one decimal), and the
    # row must be declared safe there.
    audit_path = os.path.join(ROOT, "02_research/experiments/09_testbed_scale/text_public/audit/verdicts.json")
    res["text_log_rows"] = {}
    trows = []
    for line in open(os.path.join(PAPER, "tab_logs.tex")):
        m = re.search(r"\{\\tx([A-Za-z]+)Name\}.*%\s*(TX-\S+)", line)
        if m:
            trows.append((m.group(1), m.group(2)))
    if trows:
        per_log = json.load(open(audit_path))["per_log"]
        for tag, log in trows:
            a = per_log[log]
            want = {"T": str(a["T"]), "Rel": f"{a['deployed']['request_paraphrase_exact_pct'][0]:.1f}",
                    "TGen": f"{a['deployed']['request_paraphrase_exact_pct'][1]:.1f}",
                    "TLoc": f"{a['deployed']['tokens_kept_pct']:.1f}", "Cmp": f"{a['deployed']['D1_heldout_pct']:.1f}"}
            got = {c: mac.get(f"tx{tag}{c}") for c in want}
            ok = a.get("verdict") == "safe" and got == want
            res["text_log_rows"][tag] = {"log": log, "audit_verdict": a.get("verdict"), "audit_class": a.get("class"),
                                         "printed": got, "audit": want, "ok": ok}
            if not ok:
                res["mismatches"].append({"log": log, "macro": f"\\tx{tag}*", "printed": got, "audit": want,
                                          "audit_verdict": a.get("verdict")})
    res["n_rows"] = len(res["rows"])
    res["n_rows_with_locality_confirmed_by_run_log"] = sum(1 for r in res["rows"].values() if r["run_log_locality_agrees"])
    res["n_rows_identical"] = sum(1 for r in res["rows"].values() if r["ok"])
    res["status"] = "ALL_MATCH" if not res["mismatches"] and not res["rounding_ties"] and all(c["ok"] for c in checks) else \
        ("ONLY_ROUNDING_TIES" if not res["mismatches"] and all(c["ok"] for c in checks) else "MISMATCH")
    json.dump(res, open(os.path.join(HERE, "check_tab_logs_rows.json"), "w"), indent=1)
    for tag, row in res["rows"].items():
        r, p = row["recomputed"], row["printed"]
        print(f"{row['paper_name']:14s} {row['log']:18s} " + " ".join(f"{c}={p[c]}" + ("" if p[c] == r[c] else f"(!{r[c]})")
                                                                      for c in ("T", "Rel", "TGen", "IGen", "TLoc", "ILoc", "Cmp"))
              + f"  [{r['Rel_count']}, {r['TGen_count']}, {r['IGen_count']}, D1 {r['Cmp_count']}]")
    for tag, r in res["text_log_rows"].items():
        print(f"{mac.get('tx' + tag + 'Name'):14s} {r['log']:18s} " + " ".join(f"{c}={v}" for c, v in r["printed"].items())
              + f"  [audit: {r['audit_verdict']}, {r['audit_class']}; cells {'equal' if r['ok'] else 'DIFFER from'} the audit's]")
    for c in checks:
        print(("ok  " if c["ok"] else "BAD ") + f"{c['macro']}: printed {c['printed']}, recomputed {c['recomputed_over_the_logs']}")
    print(f"T-Loc / I-Loc also equal to the line 'Loc probes' of run.log (two decimals) for "
          f"{res['n_rows_with_locality_confirmed_by_run_log']} of {res['n_rows']} rows")
    print("mismatches:", res["mismatches"])
    print("rounding ties:", res["rounding_ties"])
    print("status:", res["status"])
    return 0 if res["status"] != "MISMATCH" else 1


if __name__ == "__main__":
    sys.exit(main())
