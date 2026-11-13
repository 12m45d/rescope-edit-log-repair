"""Regression unchanged?  (python tests/extra/regression_check.py, after the GPU suite and `run_tests.py table`.)
(1) the regression section of tests/out/table.md against the table before this work (tests/out_v1/table.md);
(2) the b345 pass of both logs against the previous outputs (tests/out_v1/gpu_<log>_b345_pass.json): regression numbers, agreement
counts and every field of every previous per-row record.  Writes tests/out/regression_unchanged.json."""
import json, os, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))                 # tests/
new_d, old_d = os.path.join(HERE, "out"), os.path.join(HERE, "out_v1")


def section(path):
    with open(path) as f:
        txt = f.read()
    return [l for l in txt[txt.index("## Regression"):].splitlines() if l.startswith("| `")]


res = {"table_regression_rows_identical": section(os.path.join(new_d, "table.md")) == section(os.path.join(old_d, "table.md")),
       "n_table_regression_rows": len(section(os.path.join(new_d, "table.md"))), "logs": {}}
for lg in ("TB-Q100-ae-l2_3", "TB-L100-ae-l2_3"):
    pn, po = os.path.join(new_d, f"gpu_{lg}_b345_pass.json"), os.path.join(old_d, f"gpu_{lg}_b345_pass.json")
    if not (os.path.exists(pn) and os.path.exists(po)):
        res["logs"][lg] = {"available": False}
        continue
    with open(pn) as f:
        new = json.load(f)
    with open(po) as f:
        old = json.load(f)
    n = same = 0
    new_fields = set()
    for c, rows in old["rows"].items():
        by = {r["sid"]: r for r in new["rows"][c]}
        for r in rows:
            q = by.get(r["sid"])
            n += 1
            same += int(q is not None and all(q.get(k) == v for k, v in r.items()))
            if q is not None:
                new_fields |= set(q) - set(r)
    res["logs"][lg] = {"available": True, "pass_new": new["pass"], "code_old": old.get("code"), "code_new": new.get("code"),
                       "regression_numbers_identical": all(old["regression"][c]["mine"] == new["regression"][c]["mine"] and
                                                           old["regression"][c]["theirs"] == new["regression"][c]["theirs"] for c in old["regression"]),
                       "agreement_identical": all(old["regression"][c]["agreement"] == new["regression"][c]["agreement"] for c in old["regression"]),
                       "iii_iv_identical": old["iii_total"] == new["iii_total"] and old["iv_total"] == new["iv_total"],
                       "n_rows": n, "rows_identical_in_every_previous_field": same, "fields_added_to_rows": sorted(new_fields),
                       "n_configs": len(old["rows"])}
res["unchanged"] = bool(res["table_regression_rows_identical"] and all(v.get("available") and v["regression_numbers_identical"] and v["agreement_identical"]
                                                                       and v["n_rows"] == v["rows_identical_in_every_previous_field"] for v in res["logs"].values()))
with open(os.path.join(new_d, "regression_unchanged.json"), "w") as f:
    json.dump(res, f, indent=1)
print(json.dumps(res, indent=1))
