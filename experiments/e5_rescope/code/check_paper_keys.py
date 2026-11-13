"""Do the files of E5 and E8 hold every key that the paper's generated macro file asks of them?  (reads 01_paper/v4_rescope/numbers.tex, read-only)
    python check_paper_keys.py"""
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP
import json, re, sys
PAPER = f"{_RP.ROOT}/01_paper/v4_rescope/numbers.tex"
ROOT = f"{_RP.ROOT}/"
want = sorted(set(re.findall(r"(02_research/experiments/10_final/(?:e5_rescope|e8_cost)/out/[a-z_]+\.json) :: (\S+)", open(PAPER).read())))
cache, miss = {}, []
for path, key in want:
    try:
        d = cache.setdefault(path, json.load(open(ROOT + path)))
    except FileNotFoundError:
        miss.append((path, key, "file missing")); continue
    if key in d:
        v = d[key]
    else:
        v, ok = d, True
        parts = key.split(".")
        # nested layout: the log name contains no dot except in names like TB-Q100-ae-l2_3 (no dots): split on the first dot
        for i in range(1, len(parts) + 1):
            head, rest = ".".join(parts[:i]), ".".join(parts[i:])
            if head in d and isinstance(d[head], dict) and rest in d[head]:
                v = d[head][rest]; break
        else:
            miss.append((path, key, "key missing")); continue
    print(f"{path.split('10_final/')[1]} :: {key} = {v!r}")
print(f"\n{len(want)} keys asked, {len(miss)} missing")
for m in miss:
    print("MISSING", m)
