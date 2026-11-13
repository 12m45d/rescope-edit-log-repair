"""Register the 09_testbed_scale logs with the shared testbed registry (tb_common.LOGS), in-process.

    import sys
    sys.path.insert(0, "$RESCOPE_ROOT/02_research/experiments/07_repair_v3")
    sys.path.insert(0, "$RESCOPE_ROOT/02_research/experiments/09_testbed_scale/code")
    import scale_registry                      # adds the new names to tb_common.LOGS (no file is modified)
    from harness import load
    H = load("TB-Q1000-ae-l2_3")

Works before or after `import harness` (both import the same module object `tb_common` from
06_testbed_v3/code).  Existing names are never overwritten.  `scale_registry.NAMES` lists the new
names, `scale_registry.available()` those whose directory is complete.
"""
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP
import json
import os
import sys

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TB_CODE = f"{_RP.ROOT}/02_research/experiments/06_testbed_v3/code"
if _TB_CODE not in sys.path:
    sys.path.insert(0, _TB_CODE)
import tb_common as _TB          # noqa: E402
if os.path.join(_HERE, "code") not in sys.path:
    sys.path.insert(0, os.path.join(_HERE, "code"))
import scale_data as _SD         # noqa: E402
_SD.install(_TB)                 # VLKEB streams: the original functions; MMKE-Bench streams: see scale_data.py

REGISTRY = _RP.expand(json.load(open(os.path.join(_HERE, "registry.json"))))
NAMES = list(REGISTRY)
for _n, _cfg in REGISTRY.items():
    if _n in _TB.LOGS and _TB.LOGS[_n] != _cfg:
        raise RuntimeError(f"log name {_n} already registered with a different config")
    _TB.LOGS.setdefault(_n, dict(_cfg))

_NEED = ("edit_batch.jsonl", "metrics.json", "composite_delta.pt", "edit_factors.pt", 
         "per_edit_metrics.jsonl", "probe_records_v2.jsonl", "complaints_D1_v2.jsonl",
         "complaints_D2_v2.jsonl", "complaints_v2.json")


def available():
    """Names whose log directory holds every file of the testbed layout."""
    out = []
    for n in NAMES:
        d = os.path.join(_TB.OUT_ROOT, n)
        if all(os.path.exists(os.path.join(d, f)) for f in _NEED) and os.path.isdir(os.path.join(d, "batch_deltas")):
            out.append(n)
    return out
