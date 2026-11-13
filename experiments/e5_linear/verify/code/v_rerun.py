"""Verification, task (4): every value of out/paper_numbers.json is derived from out/results.json by the script.

usage: python v_rerun.py        -> verify/out/v_rerun.json        (CPU, seconds)

Nothing of e5_linear/out or e5_linear/code is changed: the measuring agent's scripts are copied, unmodified, into the
sandbox verify/rerun/code and their RAW inputs (per-complaint rows, optimum rows, capacity files, self-test, the agent's
own verification) into verify/rerun/out; e5_common.py derives its output directory from its own location, so the copies
read and write the sandbox only.  Three checks:
  (a) DERIVATION.  results.json (the stored file) -> e5_aggregate.paper_numbers() -> is_core filter -> compared value by
      value with the stored paper_numbers.json; the unfiltered dictionary with paper_numbers_all.json; DOC with
      paper_numbers_doc.json.
  (b) RE-RUN.  `python e5_aggregate.py` in the sandbox regenerates results.json and the three paper_numbers files from
      the raw inputs; compared with the stored files (results.json: every entry but the time stamps 'generated', 'sec').
  (c) RENDER.  `python e5_render.py` in the sandbox regenerates table.txt; compared line by line with the stored table
      (the second line carries two time stamps).
Also recorded: md5 sums of the measuring agent's files before and after (nothing changed), and of the copied scripts
against the originals.
"""
from __future__ import annotations

import v_common as C                    # first: it fixes the number of BLAS threads before numpy is loaded

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time

SRC_CODE = os.path.join(C.E5, "code")
SRC_OUT = C.REPORT_OUT
SB = os.path.join(C.VER, "rerun")
SB_CODE, SB_OUT = os.path.join(SB, "code"), os.path.join(SB, "out")
GENERATED = ("results.json", "paper_numbers.json", "paper_numbers_all.json", "paper_numbers_doc.json", "table.txt")
PY = sys.executable


def md5dir(d):
    return {f: C.md5(os.path.join(d, f)) for f in sorted(os.listdir(d)) if os.path.isfile(os.path.join(d, f))}


def diff(a, b, path="", out=None, limit=50):
    """Paths at which two JSON values differ (exact comparison; 1 == 1.0)."""
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: only in {'stored' if k in a else 'recomputed'}")
            else:
                diff(a[k], b[k], f"{path}/{k}", out, limit)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: lengths {len(a)} != {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                diff(x, y, f"{path}[{i}]", out, limit)
    else:
        same = (a == b) and (isinstance(a, bool) == isinstance(b, bool))
        if not same:
            out.append(f"{path}: {a!r} != {b!r}")
    return out


def count_leaves(o):
    if isinstance(o, dict):
        return sum(count_leaves(v) for v in o.values())
    if isinstance(o, list):
        return sum(count_leaves(v) for v in o)
    return 1


def main():
    t0 = time.time()
    before_out, before_code = md5dir(SRC_OUT), md5dir(SRC_CODE)
    if os.path.isdir(SB):
        shutil.rmtree(SB)
    os.makedirs(SB_CODE)
    os.makedirs(SB_OUT)
    os.makedirs(os.path.join(SB, "logs"))
    for f in sorted(os.listdir(SRC_CODE)):
        if f.endswith(".py"):
            shutil.copy2(os.path.join(SRC_CODE, f), os.path.join(SB_CODE, f))
    raw = [f for f in sorted(os.listdir(SRC_OUT)) if f not in GENERATED]
    for f in raw:
        shutil.copy2(os.path.join(SRC_OUT, f), os.path.join(SB_OUT, f))
    copied_equal = all(C.md5(os.path.join(SB_CODE, f)) == before_code[f] for f in os.listdir(SB_CODE))
    res = {"definition": __doc__, "sandbox": SB, "raw_inputs_copied": raw, "scripts_copied_bitwise": bool(copied_equal)}
    # ---------------------------------------------------------------- (a) derivation from the STORED results.json
    spec = importlib.util.spec_from_file_location("e5_aggregate", os.path.join(SB_CODE, "e5_aggregate.py"))
    sys.path.insert(0, SB_CODE)
    agg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agg)
    if os.path.realpath(agg.C.OUT) != os.path.realpath(SB_OUT):
        raise RuntimeError("the copied scripts do not point to the sandbox")
    stored_res = C.jload(os.path.join(SRC_OUT, "results.json"))
    P = json.loads(json.dumps(agg.C.clean(agg.paper_numbers(stored_res))))
    core = {k: v for k, v in P.items() if agg.is_core(k)}
    stored = {f: C.jload(os.path.join(SRC_OUT, f)) for f in GENERATED if f.endswith(".json")}
    d_core = diff(stored["paper_numbers.json"], core)
    d_all = diff(stored["paper_numbers_all.json"], P)
    d_doc = diff(stored["paper_numbers_doc.json"], json.loads(json.dumps(agg.DOC)))
    res["derivation"] = {"n_values_paper_numbers": len(stored["paper_numbers.json"]), "n_values_recomputed": len(core),
                         "n_differences_paper_numbers": len(d_core), "differences_paper_numbers": d_core,
                         "n_values_paper_numbers_all": len(stored["paper_numbers_all.json"]), "n_differences_paper_numbers_all": len(d_all),
                         "differences_paper_numbers_all": d_all, "n_differences_doc": len(d_doc),
                         "n_null_values_paper_numbers": int(sum(v is None for v in stored["paper_numbers.json"].values())),
                         "null_names": [k for k, v in stored["paper_numbers.json"].items() if v is None]}
    # ---------------------------------------------------------------- (b), (c) re-run of the unmodified scripts in the sandbox
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
    runs = {}
    for script in ("e5_aggregate.py", "e5_render.py"):
        p = subprocess.run([PY, script], cwd=SB_CODE, env=env, capture_output=True, text=True)
        runs[script] = {"returncode": p.returncode, "stdout": p.stdout.strip()[-400:], "stderr": p.stderr.strip()[-2000:]}
        if p.returncode != 0:
            raise RuntimeError(f"{script} failed in the sandbox: {p.stderr[-2000:]}")
    res["runs"] = runs
    new = {f: C.jload(os.path.join(SB_OUT, f)) for f in GENERATED if f.endswith(".json")}
    a, b = dict(stored["results.json"]), dict(new["results.json"])
    stamps = {k: [a.pop(k, None), b.pop(k, None)] for k in ("generated", "sec")}
    d_res = diff(a, b)
    res["rerun"] = {"results_json": {"n_leaf_values": count_leaves(a), "n_differences": len(d_res), "differences": d_res, "time_stamps_ignored": stamps},
                    **{f: {"n_values": len(stored[f]), "n_differences": len(diff(stored[f], new[f])), "differences": diff(stored[f], new[f]),
                           "bytes_equal": C.md5(os.path.join(SRC_OUT, f)) == C.md5(os.path.join(SB_OUT, f))}
                       for f in ("paper_numbers.json", "paper_numbers_all.json", "paper_numbers_doc.json")}}
    with open(os.path.join(SRC_OUT, "table.txt")) as f:
        ta = f.read().split("\n")
    with open(os.path.join(SB_OUT, "table.txt")) as f:
        tb = f.read().split("\n")
    dl = [i for i in range(max(len(ta), len(tb))) if i >= len(ta) or i >= len(tb) or ta[i] != tb[i]]
    res["render"] = {"n_lines_stored": len(ta), "n_lines_rerun": len(tb), "lines_differing": [i + 1 for i in dl],
                     "only_the_time_stamp_line_differs": bool(dl in ([], [1]) and len(ta) == len(tb)),
                     "differing_lines": [{"line": i + 1, "stored": ta[i][:200] if i < len(ta) else None, "rerun": tb[i][:200] if i < len(tb) else None} for i in dl[:5]]}
    after_out, after_code = md5dir(SRC_OUT), md5dir(SRC_CODE)
    res["untouched"] = {"out_files": len(before_out), "out_unchanged": before_out == after_out, "code_files": len(before_code),
                        "code_unchanged": before_code == after_code, "no_pycache_in_code": not os.path.exists(os.path.join(SRC_CODE, "__pycache__")),
                        "md5_out": after_out}
    res["all_pass"] = bool(not d_core and not d_all and not d_doc and not d_res and res["render"]["only_the_time_stamp_line_differs"]
                           and all(res["rerun"][f]["n_differences"] == 0 for f in ("paper_numbers.json", "paper_numbers_all.json", "paper_numbers_doc.json"))
                           and res["untouched"]["out_unchanged"] and res["untouched"]["code_unchanged"])
    res["sec"] = time.time() - t0
    C.jdump(res, os.path.join(C.OUT, "v_rerun.json"))
    print(json.dumps({k: res[k] for k in ("all_pass",)}), "| derivation diffs", len(d_core), len(d_all), len(d_doc), "| rerun results diffs", len(d_res),
          "| paper_numbers bytes equal", res["rerun"]["paper_numbers.json"]["bytes_equal"], "| table lines differing", res["render"]["lines_differing"],
          "| untouched", res["untouched"]["out_unchanged"], res["untouched"]["code_unchanged"])


if __name__ == "__main__":
    main()
