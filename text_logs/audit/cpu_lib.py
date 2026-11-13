"""Audit: the library-format copies through the project library's own constructor (rescope.loglib.from_testbed), CPU.
A FORMAT check of the stored tensors (no ScopedLog, no Hook, no threshold, no model): for each of the five logs the
EditLog's T, shapes, batch = one edit per batch, the own coordinate <W_i, K_i> (library method own()) and
delta() = sum_i outer(U_i, W_i) against (W_E - W_0)^T, with the weights read by the audit.  Output: out/lib.json
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a_common as A  # noqa: E402

import torch  # noqa: E402

torch.set_num_threads(4)
sys.path.insert(0, os.path.join(A.ROOT, "code"))
from rescope.loglib import from_testbed  # noqa: E402

W0 = A.load_w0().double()
out = {"written": A.now(), "logs": {}}
for name, (src, T, first) in A.LOGS.items():
    L = from_testbed(os.path.join(A.B_LOGS_EDIT, "library_format", name))
    WE = A.load_snapshot(src, T).double()
    dW = (WE - W0).T
    D = L.delta()
    fac = A.load_factors(src)
    s = (fac["left"][:T].double() * fac["k_request"][:T].double()).sum(1)
    out["logs"][name] = {"T": int(L.T), "name_read": L.name, "shapes": {"U": list(L.U.shape), "W": list(L.W.shape), "K": list(L.K.shape)},
                         "batch_is_arange_T": bool(torch.equal(L.batch, torch.arange(T))),
                         "own_coordinate_max_abs_dev_from_1": float((L.own() - 1).abs().max()),
                         "delta_shape": list(D.shape), "delta_vs_(WE-W0)^T_rel_fro": float((D - dW).norm() / dW.norm()),
                         "U_is_right_times_s": bool(torch.equal(L.U, fac["right"][:T].double() * s[:, None])),
                         "W_is_left_over_s": bool(torch.equal(L.W, fac["left"][:T].double() / s[:, None])),
                         "K_is_k_request": bool(torch.equal(L.K.float(), fac["k_request"][:T])),
                         "readout_norm_median": float(L.W.norm(dim=1).median()), "readout_norm_max": float(L.W.norm(dim=1).max()),
                         "readout_norm_max_index": int(L.W.norm(dim=1).argmax()),
                         "residual_norm_median(U)": float(L.U.norm(dim=1).median()), "residual_norm_max(U)": float(L.U.norm(dim=1).max())}
    A.say(name, out["logs"][name])
out["finished"] = A.now()
A.jdump(out, os.path.join(A.AOUT, "lib.json"))
