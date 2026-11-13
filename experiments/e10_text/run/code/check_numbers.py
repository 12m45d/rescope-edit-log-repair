"""E10: every entry of out/paper_numbers.json is resolved again from its file and key and compared with its value (paper numbers are rounded to at most one decimal: tolerance 0.05); the flat
records of out/results.json are checked for their fields.  -> out/tests_numbers.json
Key syntax: dotted path; a path element may itself contain dots (metric names) or '|' and is matched greedily; 'rows[3]' indexes a list."""
import os
import re
import sys

import xcommon as X


def resolve(obj, path):
    if path == "":
        return [obj]
    out = []
    if isinstance(obj, dict):
        parts = path.split(".")
        for n in range(len(parts), 0, -1):
            head, rest = ".".join(parts[:n]), ".".join(parts[n:])
            m = re.fullmatch(r"(.+)\[(\d+)\]", head)
            if head in obj:
                out += resolve(obj[head], rest)
            elif m and m.group(1) in obj and isinstance(obj[m.group(1)], list):
                out += resolve(obj[m.group(1)][int(m.group(2))], rest)
    return out


pn = X.jload(X.OUT + "/paper_numbers.json")
bad, cache = [], {}
for name, e in pn.items():
    f = os.path.join(X.RUN, e["file"])
    if f not in cache:
        cache[f] = X.jload(f)
    got = resolve(cache[f], e["key"])
    ok = len(got) >= 1 and all((g == e["value"]) or (isinstance(g, (int, float)) and isinstance(e["value"], (int, float)) and abs(g - e["value"]) <= 0.0501) for g in got[:1])
    if not ok:
        bad.append([name, e["key"], e["value"], got[:2]])
recs = X.jload(X.OUT + "/results.json")
bad_rec = [r for r in recs if tuple(r.keys()) != X.FIELDS]
res = {"n_paper_numbers": len(pn), "n_not_reproduced": len(bad), "examples": bad[:10], "n_records": len(recs), "n_records_with_wrong_fields": len(bad_rec),
       "n_records_per_log": {lg: sum(r["log"] == lg for r in recs) for lg in sorted({r["log"] for r in recs})}, "all_ok": not bad and not bad_rec}
X.jdump(res, X.OUT + "/tests_numbers.json")
print(res)
sys.exit(0 if res["all_ok"] else 1)
