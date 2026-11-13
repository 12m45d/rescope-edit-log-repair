#!/bin/bash
eval "$(python3 "${RESCOPE_HOME:?set RESCOPE_HOME to the clone of the repository}/rescope_paths.py" --sh)"   # locations: the one configuration point
# P12 verifier, check (4): is results.md the render of the JSON records?
# Copies of the experiment's aggregate.py / render.py run in verify/sandbox (their paths are derived from the script
# location, so every output lands in the sandbox; the experiment's raw records are symlinked file by file, read-only
# use; _leak.json is NOT linked, so the twin map is recomputed).  Then: sandbox results.json vs out/results.json and
# sandbox results.md vs results.md.
cd "$(dirname "$0")" || exit 1
P12=$(cd .. && pwd); SB=$PWD/sandbox
PY=${RESCOPE_PYTHON}
rm -rf "$SB"; mkdir -p "$SB/code" "$SB/out"
for f in aggregate.py render.py p12_common.py gpuq.py; do cp "$P12/code/$f" "$SB/code/$f"; done
for lg in TB-Q100-ae-l2_3 TB-L100-ae-l2_3; do
  mkdir -p "$SB/out/$lg"
  for f in "$P12/out/$lg"/*.json; do b=$(basename "$f"); [ "$b" = "_leak.json" ] && continue; ln -s "$f" "$SB/out/$lg/$b"; done
  for sub in d2 rho superseded_zfallback; do
    [ -d "$P12/out/$lg/$sub" ] || continue
    mkdir -p "$SB/out/$lg/$sub"
    for f in "$P12/out/$lg/$sub"/*.json; do [ -e "$f" ] && ln -s "$f" "$SB/out/$lg/$sub/$(basename "$f")"; done
  done
done
[ -e "$P12/out/_incidents.json" ] && ln -s "$P12/out/_incidents.json" "$SB/out/_incidents.json"
[ -d "$P12/out/text_h2h" ] && { mkdir -p "$SB/out/text_h2h"; for f in "$P12/out/text_h2h"/*.json; do [ -e "$f" ] && ln -s "$f" "$SB/out/text_h2h/$(basename "$f")"; done; }
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
( cd "$SB/code" && $PY aggregate.py 2>&1 | grep -v Warning | tail -2 && $PY render.py 2>&1 | tail -2 )
$PY - <<PYEOF
import json, hashlib, os, sys
sys.path.insert(0, "$PWD")
import vcommon as VC
a = json.load(open("$P12/out/results.json")); b = json.load(open("$SB/out/results.json"))
def diff(x, y, path="", out=None, lim=40):
    out = [] if out is None else out
    if len(out) >= lim: return out
    if type(x) != type(y): out.append((path, "type")); return out
    if isinstance(x, dict):
        for k in sorted(set(x) | set(y)):
            if k not in x or k not in y: out.append((path + "/" + k, "missing")); continue
            diff(x[k], y[k], path + "/" + k, out, lim)
    elif isinstance(x, list):
        if len(x) != len(y): out.append((path, "len", len(x), len(y))); return out
        for i, (u, v) in enumerate(zip(x, y)): diff(u, v, f"{path}[{i}]", out, lim)
    elif x != y:
        out.append((path, x, y))
    return out
d = diff(a, b)
md_a = open("$P12/results.md").read(); md_b = open("$SB/results.md").read()
la, lb = md_a.split("\n"), md_b.split("\n")
dl = [i + 1 for i, (u, v) in enumerate(zip(la, lb)) if u != v]
la_leak = json.load(open("$P12/out/TB-Q100-ae-l2_3/_leak.json")); lb_leak = json.load(open("$SB/out/TB-Q100-ae-l2_3/_leak.json"))
res = {"results_json_identical": a == b, "results_json_differences": d[:20], "n_json_differences_listed": len(d),
       "results_md_identical": md_a == md_b, "results_md_lines": [len(la), len(lb)], "results_md_differing_lines": dl[:30],
       "results_md_md5": [hashlib.md5(md_a.encode()).hexdigest(), hashlib.md5(md_b.encode()).hexdigest()],
       "leak_map_recomputed_identical": {lg: json.load(open(f"$P12/out/{lg}/_leak.json")) == json.load(open(f"$SB/out/{lg}/_leak.json")) for lg in VC.LOGS},
       "mtime_results_md": os.path.getmtime("$P12/results.md"), "mtime_results_json": os.path.getmtime("$P12/out/results.json"),
       "newest_raw_record_mtime": max(os.path.getmtime(os.path.join(r, f)) for lg in VC.LOGS for r, _, fs in os.walk(f"$P12/out/{lg}") for f in fs if f.endswith(".json"))}
res["raw_records_older_than_results_json"] = res["newest_raw_record_mtime"] <= res["mtime_results_json"] + 1
VC.jdump(res, os.path.join(VC.VOUT, "render_check.json"))
print(json.dumps(res, indent=1)[:3000])
PYEOF
