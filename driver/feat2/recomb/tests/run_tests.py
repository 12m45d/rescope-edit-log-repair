"""Run the tests of the recombination layer and render tests/out/table.txt.

    python tests/run_tests.py [core] [store] [real] [select]          (default: all four)

core    t_core.py    synthetic logs, fake store (CPU, 5 s)
store   t_store.py   capture bookkeeping and assembly with fake keys (CPU, 20 s)
real    t_real.py    the real store; gates on the development logs only (CPU, 1 min)
select  rc_check.py --fast for both backbones: reproduction byte for byte, disjointness, rules (CPU, 1 min; the full check with the
        image md5 recomputed from the files is `python rc_check.py <backbone>`)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
FEAT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "out")
DATA = os.path.join(os.path.dirname(os.path.dirname(FEAT)), "cache2", "recomb")
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    ENV.setdefault(v, "4")
os.makedirs(OUT, exist_ok=True)
want = [a for a in sys.argv[1:] if a in ("core", "store", "real", "select")] or ["core", "store", "real", "select"]
rc, lines = 0, []
for g in want:
    t0 = time.time()
    if g == "select":
        res = {}
        for bb in ("qwen25vl", "llava"):
            p = subprocess.run([sys.executable, os.path.join(FEAT, "rc_check.py"), bb, "--fast"], env=ENV, capture_output=True, text=True)
            with open(os.path.join(OUT, f"select_{bb}.log"), "w") as f:
                f.write(p.stdout + p.stderr)
            ck = json.load(open(os.path.join(DATA, bb, "check_select_fast.json")))
            for name, c in ck["checks"].items():
                res[f"{name}[{bb}]"] = {"ok": bool(c["ok"]) and p.returncode == 0, "seconds": None,
                                        "info": {k: v for k, v in c.items() if k not in ("ok", "runs", "examples", "note", "rules", "n_test_rows_by_source", "usable")}}
        with open(os.path.join(OUT, "select.json"), "w") as f:
            json.dump({"group": "select", "tests": res}, f, indent=1, default=str)
    else:
        p = subprocess.run([sys.executable, os.path.join(HERE, f"t_{g}.py")], env=ENV, capture_output=True, text=True)
        with open(os.path.join(OUT, f"t_{g}.log"), "w") as f:
            f.write(p.stdout + p.stderr)
        if not os.path.exists(os.path.join(OUT, f"t_{g}.json")):
            print(p.stdout[-2000:], p.stderr[-2000:])
            rc = 1
            continue
        res = json.load(open(os.path.join(OUT, f"t_{g}.json")))["tests"]
    n_ok = sum(bool(r["ok"]) for r in res.values())
    rc = rc or int(n_ok != len(res))
    lines.append(f"== {g}: {n_ok} of {len(res)} passed ({time.time() - t0:.0f} s)")
    for name, r in res.items():
        info = json.dumps(r.get("info"), default=str) if r["ok"] else r.get("error", "")
        lines.append(f"   [{'ok' if r['ok'] else 'FAIL'}] {name}: {info[:420]}")
have = {}
for f in sorted(os.listdir(OUT)):                                      # the table lists every group that has results, also those not run now
    if f.endswith(".json") and f[:-5].replace("t_", "") in ("core", "store", "real", "select"):
        t = json.load(open(os.path.join(OUT, f)))["tests"]
        have[f[:-5].replace("t_", "")] = (sum(bool(r["ok"]) for r in t.values()), len(t), time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(os.path.join(OUT, f)))))
head = ["TESTS of feat2/recomb (python tests/run_tests.py; details: tests/out/<group>.json, logs: tests/out/*.log)",
        "  " + "; ".join(f"{g}: {a}/{b} passed ({t})" for g, (a, b, t) in have.items()), ""]
txt = "\n".join(head + lines) + "\n"
if set(want) == {"core", "store", "real", "select"}:
    with open(os.path.join(OUT, "table.txt"), "w") as f:
        f.write(txt)
print(txt)
sys.exit(rc)
