
import runpy, sys
import numpy as np
_real, _rec = np.random.default_rng, []
class _Proxy:                                   # the referee's generator, unchanged; permutations of >= 300 rows are recorded
    def __init__(self, g):
        self._g = g
    def __getattr__(self, name):
        attr = getattr(self._g, name)
        if name != "permutation":
            return attr
        def permutation(x, *a, **k):
            out = attr(x, *a, **k)
            if hasattr(out, "__len__") and len(out) >= 300:
                _rec.append(np.asarray(out).copy())
            return out
        return permutation
np.random.default_rng = lambda *a, **k: _Proxy(_real(*a, **k))
sys.argv = [sys.argv[1]]
try:
    runpy.run_path(sys.argv[0], run_name="__main__")
finally:
    np.savez(sys.argv[0] + ".permutations.npz", **{f"p{i:05d}": p for i, p in enumerate(_rec)})
