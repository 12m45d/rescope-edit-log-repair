"""P12 verifier -- module-level numbers of the report's "why" paragraph and of Finding 5, recomputed on the CPU with my
own linear algebra.  NOT independent in one respect: the HELD-OUT answer keys are the ones the experiment captured
(cache/heldout_anskeys_<log>.pt); gpu_verify.py MODDIAG repeats the first part with my own key capture when a GPU
window exists.  Inputs otherwise: checkpoint weight (safetensors), the testbed's composite_delta.pt, the harness
edit-key cache (read-only), the experiment's saved COMP-B n=ALL correction.  -> out/cpu_moddiag.json
"""
import glob
import os
import sys

import numpy as np
import torch
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vcommon as VC  # noqa: E402

sys.path.insert(0, os.path.join(VC.ROOT, "02_research", "code"))
from vendor_v7.vlm import MODEL_DIRS  # noqa: E402

torch.set_num_threads(8)
CFG = {"TB-L100-ae-l2_3": ("llava", 16, "editkeys_llava_L16_*.pt"), "TB-Q100-ae-l2_3": ("qwen25vl", 20, "editkeys_qwen25vl_L20_*.pt")}
out = {}
for log, (bb, L, pat) in CFG.items():
    kp = os.path.join(VC.P12, "cache", f"heldout_anskeys_{log}.pt")
    if not os.path.exists(kp):
        out[log] = {"skipped": "the experiment captured no held-out answer keys for this log"}
        continue
    W0 = None
    for fn in sorted(glob.glob(os.path.join(MODEL_DIRS[bb], "*.safetensors"))):
        with safe_open(fn, framework="pt", device="cpu") as f:
            for k in f.keys():
                if k.endswith(f"layers.{L}.mlp.down_proj.weight") and "vision" not in k and "visual" not in k:
                    W0 = f.get_tensor(k)
    C = torch.load(os.path.join(VC.TB_ROOT, log, "composite_delta.pt"), map_location="cpu")
    D = (W0.float() + C).to(W0.dtype).double() - W0.double()
    ek = torch.load(sorted(glob.glob(os.path.join(VC.V3, "harness", "cache", pat)))[0])
    Ka = torch.cat([ek["K"][i][ek["pp"][i]] for i in range(len(ek["K"]))]).double().T.contiguous()
    del ek
    U, S, _ = torch.linalg.svd(Ka, full_matrices=False)
    rk = int((S > max(Ka.shape) * float(np.finfo(np.float64).eps) * float(S[0])).sum())
    Q = U[:, :rk].contiguous()
    hk = torch.load(kp)
    rows = VC.records(log)
    pool = VC.my_split(rows)
    ho = [r for r in rows if pool[r["probe_idx"]] == "heldout" and r["y0_valid"]]
    assert hk["groups"] == [r["group"] for r in ho] and hk["counts"] == [int(r["n_ans_tok"]) for r in ho]
    assert hk["stored_D1"] == [not bool(r["reproduced"]) for r in ho]
    key = lambda r: (r["ptype"], r["prompt"], r["image"])  # noqa: E731
    rep = {}
    for r in rows:
        if pool[r["probe_idx"]] == "repair":
            rep.setdefault(key(r), []).append(r["group"])
    twin = [key(r) in rep for r in ho]
    K = hk["K"].double().T.contiguous()
    DK, QK = D @ K, Q.T @ K
    DPK = (D @ Q) @ QK
    s, a, j = [], [], 0
    for A in hk["counts"]:
        s.append(float(QK[:, j:j + A].norm() / K[:, j:j + A].norm()))
        a.append(float(DPK[:, j:j + A].norm() / DK[:, j:j + A].norm()))
        j += A
    d1 = hk["stored_D1"]
    med = lambda v, c: float(np.median([x for x, y, t in zip(v, d1, twin) if y == c and not t]))  # noqa: E731
    res = {"keys_source": "experiment's capture (cache/heldout_anskeys)", "rank_K_ans": rk, "n_edit_answer_keys": int(Ka.shape[1]),
           "D_share_on_K_ans_span": float((D @ Q).norm() / D.norm()), "fro_D": float(D.norm()),
           "s_complaints": med(s, True), "s_noncomplaints": med(s, False), "a_complaints": med(a, True), "a_noncomplaints": med(a, False),
           "n_complaints": int(sum(y and not t for y, t in zip(d1, twin))), "n_noncomplaints": int(sum((not y) and not t for y, t in zip(d1, twin))),
           "frac_complaints_a_below_half": float(np.mean([x < 0.5 for x, y, t in zip(a, d1, twin) if y and not t]))}
    # the saved COMP-B n=ALL correction: defining properties + restoration ratio on the held-out keys
    cp = os.path.join(VC.P12, "out", log, "corr", "COMP_B__nALL__s0.pt")
    if os.path.exists(cp):
        b = torch.load(cp, map_location="cpu")
        Lf, Rf = b["Lf"], b["Rf"]
        rec = VC.jload(os.path.join(VC.P12, "out", log, "COMP_B__nALL__s0.json"))
        repd = set(rec["reported_groups"])
        XK = Lf @ (Rf.T @ K)
        rho, j = [], 0
        for A in hk["counts"]:
            rho.append(float((DK[:, j:j + A] + XK[:, j:j + A]).norm() / DK[:, j:j + A].norm()))
            j += A
        tw_rep = [t and bool(repd & set(rep[key(r)])) for r, t in zip(ho, twin)]
        res["comp_b_all"] = {"rank": int(Rf.shape[1]), "norm_over_log": float(Lf.norm() / D.norm()),
                             "R_orthonormal_err": float((Rf.T @ Rf - torch.eye(Rf.shape[1], dtype=Rf.dtype)).abs().max()),
                             "max_abs_Rt_Q_ans": float((Rf.T @ Q).abs().max()),
                             "X_on_edit_answer_keys_rel": float((Lf @ (Rf.T @ Ka)).norm() / (D @ Ka).norm()),
                             "n_twins_of_reported": int(sum(tw_rep)), "max_rho_twins_of_reported": max([x for x, t in zip(rho, tw_rep) if t], default=None),
                             "median_rho_complaints_disjoint": med(rho, True), "median_rho_noncomplaints_disjoint": med(rho, False)}
    out[log] = res
    print(log, res)
VC.jdump(out, os.path.join(VC.VOUT, "cpu_moddiag.json"))
