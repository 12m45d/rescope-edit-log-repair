"""white: data and feature layer of the gates added by PROTOCOL.md section 6, amendment "Review round 1", items R2 and R3.

    import sys; sys.path.insert(0, "$RESCOPE_ROOT/02_research/experiments/10_final/feat2")
    import white

    white.benparts   parts of the label-free benign pool: BEN-W / BEN-S / BEN-T, split by source item (R2)
    white.wstats     statistics of the BEN-W keys: mean key mu, uncentred second moment C_W through its N x N Gram matrix,
                     M = (C_W + rho I)^-1 by the Woodbury identity (never a d x d inverse)
    white.scores     score matrices [queries x edits]: cos_centred, white(kappa), whitecos(kappa), cos_null, and the helper of
                     rescope_nullW; feature_gate(...) returns each of them as the detectors of a rescope.ScopedLog, so that the
                     library's calibration, key cache and hook use them exactly like the solver coordinate
    white.mechanics  mechanics check on the two development logs (script-rendered out/mechanics.txt)

Everything is CPU, float64, seeded and deterministic.  See USAGE.txt; tests: tests/run_tests.py.
"""
from . import common  # noqa: F401  (sets the BLAS thread caps and switches bytecode writing off before the library is imported)

__all__ = ["common", "benparts", "wstats", "scores", "mechanics"]


def __getattr__(name):                                  # submodules are loaded on first use (wstats / scores import torch and the library)
    if name in ("benparts", "wstats", "scores", "mechanics"):
        import importlib
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(name)
