"""E5 (e), reduced by R19: nothing new is run.  The verified record of 08_rescope/p11_scopecut (results.md) is re-reported with file and
line: exactness of the minimum cut against brute force, run times on the real and synthetic instances, and the out-of-sample comparison
of the minimum cut with independent per-edit thresholds and with global thresholds at 70 % test-witness coverage."""
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP
import re
from e5common import *

SRC = f"{_RP.ROOT}/02_research/experiments/08_rescope/p11_scopecut/results.md"

def cells(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]

def num(x):
    try:
        return float(x)
    except ValueError:
        return None

def main():
    lines = open(SRC).read().split("\n")
    out = {"source": SRC, "note": __doc__, "exactness": None, "run_time": [], "cut_pass": {}, "literal_table": [], "quotes": []}
    for i, ln in enumerate(lines, 1):
        if ln.startswith("Exactness: min cut == brute force"):
            m = re.search(r"brute force on (\d+)/(\d+) instances", ln)
            m2 = re.search(r"INDEP <= optimum on (\d+)/(\d+) \(equal on (\d+)\)", ln)
            out["exactness"] = {"line": i, "n_equal_brute_force": int(m.group(1)), "n_instances": int(m.group(2)),
                                "indep_at_most_optimum": int(m2.group(1)) if m2 else None, "indep_equal_optimum": int(m2.group(3)) if m2 else None, "text": ln[:600]}
        if ln.startswith("- **CUT-PASS:"):
            out["cut_pass"] = {"line": i, "verdict": "FAIL" if "FAIL" in ln[:30] else "PASS", "text": ln}
        if ln.startswith("  - Ratio SCOPE-CUT / INDEP at the same point"):
            out["quotes"].append({"line": i, "what": "minimum cut against independent per-edit thresholds", "text": ln.strip()})
        if "why the exact in-sample optimum does not transfer" in ln:
            out["quotes"].append({"line": i, "what": "why the in-sample optimum does not transfer", "text": ln.strip()[:900]})
        if ln.startswith("### Graph sizes and run times"):
            hdr = cells(lines[i + 1])
            j = i + 4
            while j <= len(lines) and lines[j - 1].startswith("|"):
                c = cells(lines[j - 1])
                r = dict(zip(hdr, c))
                out["run_time"].append({"line": j, "instance": r["instance"], "split": r["split"], "rows_kind": r["rows"], "edits": int(r["edits"]), "rows": int(r["rows N"]),
                                        "witness_nodes": int(r["witness nodes"]), "graph_nodes": int(r["graph nodes"]), "graph_edges": int(r["graph edges"]),
                                        "build_ms": num(r["build ms"]), "one_cut_ms_median": num(r["one cut ms (median)"]), "one_cut_ms_max": num(r["one cut ms (max)"]),
                                        "grid_61_lambda_ms": num(r["61-lambda grid ms"])})
                j += 1
            out["run_time_header"] = {"line": i, "text": ln}
        if ln.startswith("**literal** - path `CUT`"):
            hdr = cells(lines[i + 1])
            j = i + 4
            while lines[j - 1].startswith("|"):
                r = dict(zip(hdr, cells(lines[j - 1])))
                out["literal_table"].append({"line": j, "log": r["log"], "split": r["split"], "scope_cut_firing_pct": num(r["SCOPE-CUT firing %"]), "at_coverage": num(r["at coverage"]),
                                             "G-ABS": num(r["G-ABS"]), "G-REL": num(r["G-REL"]), "G-COS": num(r["G-COS"]), "best_global": r["best G"], "ratio_cut_over_best_global": num(r["ratio"]),
                                             "ratio_ci": r["bootstrap 95 % CI of the ratio"], "indep_firing_pct": num(r["INDEP firing %"]), "cut_over_indep": num(r["CUT / INDEP"])})
                j += 1
    out["not_in_the_record"] = "the out-of-sample learning curve over n in {50, 100, 200, all} fit rows and the 50 / 90 % coverage read-outs of the task's original text were not part of P11b " \
                               "and are not run (R19: E5 (e) runs nothing new); the record holds the 70 % coverage read-out on the full fit pool (589 / 594 rows), splits S2 and S3"
    jdump(out, f"{OUT}/e/rereport.json")
    print(out["exactness"], len(out["run_time"]), len(out["literal_table"]), out["cut_pass"].get("verdict"))

if __name__ == "__main__":
    main()
