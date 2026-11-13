"""E10 step 8: export every built log in the on-disk layout that the library's own constructor reads
(rescope.loglib.from_testbed(<directory>): edit_factors.pt with U, W, K1_solver, uids + metrics.json with per_batch,
config, dtype), and verify that the library constructor returns exactly the log of tx_public.from_public.

    logs_edit/library_format/<name>/edit_factors.pt   U (T, 1600), W (T, 6400), K1_solver (T, 6400) float64
                                                      (+ K_template_mean, left, right float32, case_ids, uids)
    logs_edit/library_format/<name>/metrics.json      per_batch = one batch per update; config; dtype; pointers to
                                                      the pre-edit and the deployed weight and their sha256

    U_i = right_i * s_i (residual target - output), W_i = left_i / s_i (read-out), s_i = <left_i, k_request_i>,
    K1_solver_i = k_request_i (the key of the solve; see tx_public.py, "TWO KEYS").
The library itself is not modified.  CPU only.  Writes out/library_format.json.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tp_common as C  # noqa: E402
import tx_public as TX  # noqa: E402

import torch  # noqa: E402

torch.set_num_threads(4)
sys.path.insert(0, TX.LIB)
from rescope.loglib import from_testbed  # noqa: E402

ROOT = os.path.join(C.LOGS_EDIT, "library_format")
out = {"written": C.now(), "root": ROOT, "logs": {}}
reg = TX.registry()
for name in TX.available():
    e = reg[name]
    d = os.path.join(ROOT, name)
    os.makedirs(d, exist_ok=True)
    log = TX.from_public(name)                       # key = 'request'
    f = TX.factors(name)
    T = log.T
    mp, fp = os.path.join(d, "metrics.json"), os.path.join(d, "edit_factors.pt")
    stale = True
    if os.path.exists(mp) and os.path.exists(fp):
        stale = C.jload(mp).get("deployed_sha256") != e["deployed_sha256"]
    if stale:
        C.tsave({"U": log.U, "W": log.W, "K1_solver": log.K, "K_template_mean": f["k_template_mean"], "left": f["left"],
                 "right": f["right"], "uids": list(log.meta["uids"]), "case_ids": list(f["case_ids"]),
                 "note": "E10 public text log in the layout of 06_testbed_v3 (rescope.loglib.from_testbed); K1_solver = key of the "
                         "request prompt alone; left / right = the factors returned by the public execute_rome"}, fp)
        C.jdump({"name": name, "T": T, "dtype": "float32",
                 "config": {"l2": None, "layer": 17, "backbone": TX.BACKBONE, "editor": "rome-public", "module": TX.MODULE,
                            "batch": 1, "dataset": "CounterFact", "modality": "text", "source": e["source"], "records": e["records"]},
                 "per_batch": [{"batch": i, "edit_ids": [i]} for i in range(T)],
                 "pre_edit_weight": e["pre_edit_weight"], "pre_edit_sha256": e["pre_edit_sha256"],
                 "deployed_weight": e["deployed_weight"], "deployed_sha256": e["deployed_sha256"],
                 "weight_orientation": "Conv1D (d_in = 6400, d_out = 1600): y = x @ W + b; the library's (m, d) = (1600, 6400)",
                 "development_log": e["development_log"]}, mp)
    L = from_testbed(d)                              # the library's own constructor
    same = bool(torch.equal(L.U, log.U) and torch.equal(L.W, log.W) and torch.equal(L.K, log.K) and torch.equal(L.batch, log.batch))
    W0, WE = TX.weights(name)
    dW = (WE.double() - W0.double()).T
    rel = float((L.delta() - dW).norm() / dW.norm())
    out["logs"][name] = {"dir": d, "T": T, "from_testbed_equals_from_public": same, "name_read_by_library": L.name,
                         "own_coordinate_max_abs_dev_from_1": float((L.own() - 1).abs().max()),
                         "delta_vs_deployed_minus_pre_edit_rel_fro": rel, "rewritten": stale,
                         "bytes": sum(os.path.getsize(os.path.join(d, x)) for x in os.listdir(d))}
    C.say(name, out["logs"][name])
C.jdump(out, os.path.join(C.OUT, "library_format.json"))
