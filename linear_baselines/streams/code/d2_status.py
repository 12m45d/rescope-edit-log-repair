"""P12 -- which D2 decodes are finished?  (CPU only; the D2 counterpart of status.py)

usage: python d2_status.py --log NAME [--set core|all] [--quiet]
exit code 0 = every decode of the set that can be run has a record in out/<log>/d2/.

A decode can be run when its correction is available: the rule of run_d2.get_corr (an end point, a saved
correction, an alias of a saved correction, or a repair that changed nothing).  Configs whose repair failed are
not counted (run_d2.py skips them).  Nothing is written.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p12_common import OUT, jload  # noqa: E402
from run_d2 import d2_plan  # noqa: E402


def available(odir, name):
    if name in ("NONE", "ALL"):
        return True
    p = os.path.join(odir, "corr", f"{name}.pt")
    al = os.path.join(odir, "corr", f"{name}.alias")
    if not os.path.exists(p) and os.path.exists(al):
        p = os.path.join(odir, "corr", open(al).read().strip() + ".pt")
    if os.path.exists(p):
        return True
    rec = os.path.join(odir, f"{name}.json")
    return os.path.exists(rec) and jload(rec)["eval"]["fro"] == 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--set", default="all", choices=("core", "all"))
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    odir = os.path.join(OUT, a.log)
    names = d2_plan(a.set)
    done = [n for n in names if os.path.exists(os.path.join(odir, "d2", f"{n}.json"))]
    failed = [n for n in names if n not in done and os.path.exists(os.path.join(odir, f"{n}.failed.json"))]
    todo = [n for n in names if n not in done and n not in failed and available(odir, n)]
    nocorr = [n for n in names if n not in done and n not in failed and n not in todo]
    if not a.quiet:
        print(f"{a.log}: D2 set {a.set}: {len(done)} done, {len(todo)} to do {todo[:12]}, {len(nocorr)} without a saved correction "
              f"{nocorr}, {len(failed)} with a failed repair {failed}")
    sys.exit(0 if not todo else 1)


if __name__ == "__main__":
    main()
