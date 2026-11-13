"""Test runner of white (10_final/feat2/white).  Every test writes tests/out/<test>.json; the table is rendered from these files.

  python tests/run_tests.py                     all CPU tests (about 7 minutes, 3 GB of RAM in the naive tests), then the table
  python tests/run_tests.py parts stats         the tests of some groups (parts | stats | scores | mechanics | usage | gpu)
  python tests/run_tests.py c5_naive_qwen ...   single tests
  python tests/run_tests.py table               tests/out/table.txt from the JSON files (also printed)
  $GQ run --mem 1500 --prio 4 --name white_gpu -- python tests/run_tests.py gpu      the optional GPU test (no model; through the queue only)
Run it with $RESCOPE_PYTHON.  Nothing is written outside 10_final/feat2/white/tests.
The mechanics tests read feat2/white/out/mechanics_<log>.json: run `python -m white.mechanics run` (cwd = feat2) first.
A result is STALE (and the table says so) if the library, this package or the tests changed after it was produced.  The shared driver fx is
under review and edited by others: a changed fx fingerprint is reported on its own line (only the mechanics check and the draw function read fx).
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback

import tcommon as T
from tcommon import C

import t_parts  # noqa: E402,F401
import t_stats  # noqa: E402,F401
import t_scores  # noqa: E402,F401
import t_mech  # noqa: E402,F401
import t_usage  # noqa: E402,F401
import t_gpu  # noqa: E402,F401

GROUPS = ("parts", "stats", "scores", "mechanics", "usage")                # the default (CPU) run
OPTIONAL = ("gpu",)                                                        # through the GPU queue only
RESERVED = ("test", "group", "what", "seconds", "code")                    # fields the runner adds to every result


def fingerprints() -> dict:
    from white import mechanics
    fp = mechanics.code_fingerprint()
    import hashlib
    h = hashlib.sha1()
    for f in sorted(os.listdir(T.HERE)):
        if f.endswith(".py") and f != "run_tests.py":
            h.update(f.encode())
            with open(os.path.join(T.HERE, f), "rb") as fh:
                h.update(fh.read())
    fp["tests"] = h.hexdigest()[:12]
    return fp


def run(names):
    from white import mechanics
    os.makedirs(T.OUT, exist_ok=True)
    fp = fingerprints()
    for nm in names:
        fn, group, what = T.TESTS[nm]
        t0 = time.time()
        try:
            res = fn()
            clash = sorted(set(res) & set(RESERVED))
            if clash or "pass" not in res or "note" not in res:           # a malformed result is a failed test, never a crash of the run
                res = {"pass": False, "note": f"malformed test result: reserved keys {clash}, has pass / note: {'pass' in res} / {'note' in res}"}
        except Exception as e:                                             # a test that raises fails, with its traceback
            res = {"pass": False, "note": f"{type(e).__name__}: {e}"[:300], "traceback": traceback.format_exc()}
        res = dict(res, test=nm, group=group, what=what, seconds=time.time() - t0, code=fp)
        res["pass"] = bool(res["pass"])
        C.write_atomic(os.path.join(T.OUT, f"{nm}.json"), json.dumps(mechanics._clean(res), indent=1, allow_nan=False) + "\n")
        print(f"[{time.strftime('%H:%M:%S')}] {'PASS' if res['pass'] else 'FAIL'}  {nm:32s} {res['seconds']:6.1f}s  {res['note']}", flush=True)


def table() -> str:
    rows, missing, optional_missing = [], [], []
    for nm, (_, group, what) in T.TESTS.items():
        p = os.path.join(T.OUT, f"{nm}.json")
        if not os.path.exists(p):
            (optional_missing if group in OPTIONAL else missing).append(nm)
            continue
        with open(p) as f:
            rows.append(json.load(f))
    now = fingerprints()
    own = ("rescope", "white", "tests")
    stale = [r["test"] for r in rows if any(r["code"].get(k) != now[k] for k in own)]
    fx_other = sorted({str(r["code"].get("fx")) for r in rows if r["code"].get("fx") != now.get("fx")})
    L = ["TESTS of white (10_final/feat2/white): parts of the benign pool, BEN-W statistics, score functions, mechanics check",
         "rendered by tests/run_tests.py from tests/out/*.json; code now: " + ", ".join(f"{k} {v}" for k, v in now.items()),
         f"{sum(r['pass'] for r in rows)} of {len(rows)} tests pass" + (f"; NOT RUN: {', '.join(missing)}" if missing else "")
         + (f"; STALE results (library, package or tests changed since; re-run them): {', '.join(stale)}" if stale
            else "; every result was produced by the library, package and tests fingerprinted above")]
    if fx_other:
        L.append(f"the shared driver fx changed after some results were produced (then {', '.join(fx_other)}, now {now.get('fx')}): it is under review; only the "
                 "mechanics check and the draw function read it")
    if optional_missing:
        L.append(f"optional GPU test not run: {', '.join(optional_missing)} (gq run --mem 1500 --prio 4 -- python tests/run_tests.py gpu)")
    L.append("")
    for g in GROUPS + OPTIONAL:
        sub = [r for r in rows if r["group"] == g]
        if not sub:
            continue
        L.append(f"[{g}]")
        w = max(len(r["test"]) for r in sub)
        for r in sub:
            L.append(f"  {'PASS' if r['pass'] else 'FAIL'}  {r['test'].ljust(w)}  {r['seconds']:6.1f}s  {r['what']}")
            L.append(f"        {' ' * w}           -> {r['note']}")
        L.append("")
    return "\n".join(L) + "\n"


def main(argv):
    args = argv[1:]
    if args != ["table"]:
        names = []
        for a in (args or list(GROUPS)):
            if a in GROUPS + OPTIONAL:
                names += [n for n, (_, g, _) in T.TESTS.items() if g == a]
            elif a in T.TESTS:
                names.append(a)
            else:
                raise SystemExit(f"{a!r}: a group {GROUPS + OPTIONAL}, a test name ({', '.join(T.TESTS)}) or 'table'")
        run(names)
    txt = table()
    C.write_atomic(os.path.join(T.OUT, "table.txt"), txt)
    print(txt)
    bad = "  FAIL  " in txt or "; NOT RUN: " in txt or "; STALE results" in txt
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
