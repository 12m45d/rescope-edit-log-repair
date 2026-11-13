"""rescope: scoped replay of an edit log (reference implementation for the final experiments).

    loglib     EditLog (U_i, W_i, k_i, batches), rebased values, loaders, synthetic logs
    scoped     ScopedLog (detectors, thresholds, values, eta, role, fire, response), Hook: the one forward code path
    calibrate  calibrate_two_block (default), calibrate_shared, calibrate_dense, calibrate_log_only, calibrate_conformal
               (+ two_block_rule, shared_rule, float_safe_threshold, conformal_rank_exact, benign_scores, own_caps)
    repair     least_repair, null_repair, witness_repair, scope_cut, add_guards (+ CutProblem, Session, DECODE_MARGIN)
    baselines  cosine_gate, exact_gate, ball_gate, router_gate, zscore_gate (same interface as a scoped log); feature_gate
    evaluate   Evaluator (edits, probes, hard negatives, D2, free generation; gates, weights, harness corrections, DUAL),
               token locality, scale-safe loading (register_logs, load(lean), stored_metrics)
    keycache   KeyCache: per-backbone cached keys -> firing decision of every cached sequence for any gate / log, on the CPU
    sparse     SparseEvaluator: the Evaluator's teacher-forced rows of scoped gates, model run only where something fires
    splits     REPAIR / HELD-OUT / HELD-OUT-DISJOINT / HELD-OUT-STRICT probe pools, seeded calibration draws, edit halves

See docs/USAGE.md; tests: `python tests/run_tests.py cpu | gpu --log <name> | table`.
"""
from . import splits  # noqa: F401
from .baselines import FEATURES, DenseGate, ball_gate, coachhook_alpha, cosine_gate, exact_gate, feature_gate, key_distances, router_gate, zscore_gate  # noqa: F401
from .calibrate import (as_fraction, benign_scores, block_scores, calibrate_conformal, calibrate_dense, calibrate_log_only, calibrate_shared,  # noqa: F401
                        calibrate_two_block, conformal_rank, conformal_rank_exact, float_safe_threshold, keep_guards, own_caps, pooled_crosstalk,
                        shared_rule, split_blocks, two_block_rule)
from .evaluate import (HN_DIR, HN_DIR_FINAL, Dual, Evaluator, ensure_registered, harness, load, load_hard_negatives, load_rows, locality,  # noqa: F401
                       n_edits, peak_rss_gb, reedit, register_logs, stored_metrics, summarise)
from .keycache import KeyCache, decision_band, near_threshold, write_cache  # noqa: F401
from .loglib import EditLog, batch_deltas, from_harness, from_testbed, rebase, rebase_error, rebase_exact, synthetic_log  # noqa: F401
from .repair import DECODE_MARGIN, CutProblem, Session, add_guards, brute_force, least_repair, null_repair, scope_cut, witness_repair  # noqa: F401
from .scoped import Firing, Hook, ScopedLog  # noqa: F401
from .sparse import SparseEvaluator  # noqa: F401
