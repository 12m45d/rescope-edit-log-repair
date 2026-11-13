"""Development helper: run named CPU tests of t_cal / t_core / t_ops without writing the test JSON (prints key numbers)."""
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.abspath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP
import json, sys, time
sys.path.insert(0, f"{_RP.ROOT}/02_research/code")
sys.path.insert(0, f"{_RP.ROOT}/02_research/code/rescope/tests")
import torch
torch.set_num_threads(4)
import importlib
mods = [importlib.import_module(m) for m in sys.argv[1].split(",")]
tests = {}
for m in mods:
    tests.update(m.TESTS)
for name in sys.argv[2].split(","):
    t0 = time.time()
    try:
        r = tests[name]()
    except Exception:
        import traceback
        traceback.print_exc()
        continue
    print(name, "PASS" if r["pass"] else "FAIL", f"{time.time() - t0:.1f}s", flush=True)
    print(json.dumps({k: v for k, v in r.items() if k not in ("criterion",)}, default=str)[:int(sys.argv[3]) if len(sys.argv) > 3 else 3000], flush=True)
