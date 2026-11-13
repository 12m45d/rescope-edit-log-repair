"""recomb: data and feature layer of the variant rescope_x / <gate>_x (PROTOCOL.md section 6, amendment "Review round 1", item R5:
log-derived near-scope calibration; no label, no complaint).

    import sys; sys.path.insert(0, "$RESCOPE_ROOT/02_research/experiments/10_final/feat2")
    import recomb

    recomb.log_rows(log, kinds)                 the recombination rows a log with T edits may use (owner < T; X-X partner < T)
    recomb.log_keys(log, kinds)                 their last-prompt keys (prompt only, pre-edit model), float32 (n, d)
    recomb.own_scores(log, score, kinds)        scores of every edit on ITS OWN rows: .x (the baseline), .value, .per_edit(kind)
    recomb.own_baseline(log, score, kinds)      (x (D,), record): 'maximum normalised coordinate of edit i on its own recombinations'
    recomb.calibrate_two_block_x(S, coords=Q, x=x, alpha=, seed=)     the library's two-block calibration with b = max(block 1, x)
    recomb.augment(Q, x, split, seed)           the same as plain library input: one more block-1 row + explicit blocks
    recomb.two_block_rule_x(Q, x, alpha, ...)   the rule on numbers, composed from the amendment's text (reference)
    recomb.decide_rows(S, log)                  a calibrated gate on the recombination rows themselves (mechanics; development logs)
    recomb.store(backbone)                      rows + keys of a backbone (Store);  recomb.available(backbone)
    recomb.wiring                               example of the wiring into the shared driver fx (imported on first use)

core.py is the reference text; USAGE.txt the manual; the build scripts are rc_select.py -> rc_check.py -> rc_capture.py ->
rc_assemble.py; rc_mechanics.py writes out/mechanics.txt (development logs only); tests: tests/run_tests.py.
"""
import os as _os
import sys as _sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):       # at most 4 BLAS threads per process (as the driver)
    _os.environ.setdefault(_v, "4")
_sys.dont_write_bytecode = True                         # the library, the driver and the caches' code are read-only: no bytecode files there

from .core import (BACKBONES, CHUNK, DEFAULT_DATA, KINDS, M, LogRows, OwnScores, Store, augment, available, calibrate_two_block_x,  # noqa: F401
                   data_root, decide_rows, identify, log_keys, log_rows, own_baseline, own_scores, single_thread, store, two_block_rule_x)

__all__ = ["BACKBONES", "CHUNK", "DEFAULT_DATA", "KINDS", "M", "LogRows", "OwnScores", "Store", "augment", "available", "calibrate_two_block_x",
           "data_root", "decide_rows", "identify", "log_keys", "log_rows", "own_baseline", "own_scores", "single_thread", "store", "two_block_rule_x", "wiring"]


def __getattr__(name):                                  # wiring imports the shared driver fx: loaded on first use only
    if name == "wiring":
        import importlib
        return importlib.import_module(f"{__name__}.wiring")
    raise AttributeError(name)
