"""Check 3a: the experiments' own analysis + rendering re-run in my directory (code/rerun_<bb>: their scripts with ROOT redirected; inputs are
links to their gate / behave / rows / factor directories) against the files they delivered.  Writes out/rerender.json."""
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP
import json, os, sys, difflib
FINAL = f"{_RP.ROOT}/02_research/experiments/10_final"
V = FINAL + "/e3_qwen/verify"
out = {}
for bb in ("qwen", "llava"):
    a, b = f"{FINAL}/e3_{bb}/out", f"{V}/out/rerun_{bb}/out"
    o = {}
    if not os.path.exists(b + "/table.txt"):
        out[bb] = {"missing": True}
        continue
    ra, rb = json.load(open(a + "/results.json")), json.load(open(b + "/results.json"))
    key = lambda r: json.dumps({k: v for k, v in r.items() if k not in ("value", "ci_lo", "ci_hi", "by_draw", "timing", "where_cal", "gate_coverage_form_20_draws", "p_two_sided", "vmin", "vmax")}, sort_keys=True)
    da, db = {}, {}
    for r in ra: da.setdefault(key(r), []).append(r)
    for r in rb: db.setdefault(key(r), []).append(r)
    nd, worst = 0, 0.0
    for k in da:
        if k not in db or len(da[k]) != len(db[k]):
            nd += 1; continue
        for x, y in zip(da[k], db[k]):
            for f in ("value", "ci_lo", "ci_hi"):
                u, w = x.get(f), y.get(f)
                if isinstance(u, (int, float)) and isinstance(w, (int, float)):
                    worst = max(worst, abs(u - w))
                    nd += abs(u - w) > 1e-9
                elif u != w:
                    nd += 1
    o["results"] = {"n_records_delivered": len(ra), "n_records_rerun": len(rb), "n_keys_only_in_one": len(set(da) ^ set(db)), "n_values_differ_above_1e-9": int(nd), "largest_abs_diff": worst}
    pa, pb = json.load(open(a + "/paper_numbers.json")), json.load(open(b + "/paper_numbers.json"))
    o["paper_numbers"] = {"n_delivered": len(pa), "n_rerun": len(pb), "n_names_only_in_one": len(set(pa) ^ set(pb)),
                          "n_differ": sum(1 for k in pa if k in pb and (abs(pa[k] - pb[k]) > 1e-9 if isinstance(pa[k], (int, float)) and isinstance(pb[k], (int, float)) else pa[k] != pb[k]))}
    ta, tb = open(a + "/table.txt").read().splitlines(), open(b + "/table.txt").read().splitlines()
    stamp = lambda s: ("rendered" in s and "by e3_render.py" in s) or ("written 20" in s) or s.startswith("driver {")
    df = [l for l in difflib.unified_diff(ta, tb, lineterm="", n=0) if l[:1] in "+-" and not l.startswith(("+++", "---")) and not stamp(l)]
    o["table"] = {"n_lines_delivered": len(ta), "n_lines_rerun": len(tb), "n_diff_lines_other_than_time_stamps": len(df), "first_diff_lines": [l[:300] for l in df[:8]]}
    out[bb] = o
json.dump(out, open(V + "/out/rerender.json", "w"), indent=1)
print(json.dumps(out, indent=1)[:3000])
