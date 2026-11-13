"""P12 -- where does the log's damage on a probe row come from?  (CPU only; no model, no GPU)

Every edit-preserving compensation of this experiment protects K_ans, the span of the logged edits' answer-position
keys (the edited module's inputs where Rel is checked): the correction X satisfies X P = 0, P = projector onto that
span, so on any key k it acts through (I - P) k only.  This script splits the log's change of the module output at
the answer keys K_r of every valid HELD-OUT row r,
    D K_r = D P K_r + D (I - P) K_r,        D = W_E - W_0,
and stores
    a_r = ||D P K_r||_F / ||D K_r||_F          the part that travels through the protected span
    b_r = ||D (I - P) K_r||_F / ||D K_r||_F    the part an edit-preserving correction can act on
    s_r = ||P K_r||_F / ||K_r||_F              share of the key norm inside the protected span
a_r is also exactly the module-restoration ratio rho_r (see rho_cpu.py) of the weight
    W_proj = W_0 + D P       ("the log restricted to the edits' answer-key span": complement rollback),
the complaint-free limit of an edit-preserving linear repair at the module: it keeps the module output at every
edit answer key and returns to W_0 on everything orthogonal to them.  (Module-level statement only: no behavioural
evaluation of W_proj was run.)
Inputs: cache/heldout_anskeys_<log>.pt (captured by run_d2.heldout_keys; diagnostic, read by no method), the
harness's edit-key cache (read-only), the checkpoint weight and the testbed's composite_delta.pt.
The basis of K_ans is built like harness.methods._orth (QR + SVD, rank tol max(shape)*eps*s0) and its rank is
checked against the protection rank recorded by the COMP-B runs.
Output: out/<log>/_protect_split.json

usage: CUDA_VISIBLE_DEVICES="" python protect_split_cpu.py
"""
from __future__ import annotations
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP

import glob
import json
import os

import numpy as np
import torch

import rho_cpu as RC

HCACHE = f"{_RP.ROOT}/02_research/experiments/07_repair_v3/harness/cache"
EDITKEYS = {"TB-L100-ae-l2_3": "editkeys_llava_L16_*.pt", "TB-Q100-ae-l2_3": "editkeys_qwen25vl_L20_*.pt"}
EPS64 = float(np.finfo(np.float64).eps)
VERSION = 1


def orth(Kc):
    """Orthonormal basis of span(Kc) (d, m) fp64 -- the harness's rule (methods._orth)."""
    Q1, R1 = torch.linalg.qr(Kc)
    Ur, s, _ = torch.linalg.svd(R1)
    r = int((s > max(Kc.shape) * EPS64 * float(s[0])).sum())
    return Q1 @ Ur[:, :r]


def main():
    torch.set_num_threads(int(os.environ.get("P12_THREADS", "4")))      # at most 4 BLAS threads per process
    for log in RC.CKPT:
        kp = os.path.join(RC.CACHE, f"heldout_anskeys_{log}.pt")
        op = os.path.join(RC.OUT, log, "_protect_split.json")
        if os.path.exists(op) and json.load(open(op)).get("version") == VERSION:
            print(f"{log}: done")
            continue
        if not os.path.exists(kp):
            print(f"{log}: held-out answer keys not captured yet (needs one GPU start of run_d2.py) -- skipped")
            continue
        ef = sorted(glob.glob(os.path.join(HCACHE, EDITKEYS[log])))
        assert len(ef) == 1, ef
        ek = torch.load(ef[0])
        T = len(ek["K"])
        Kans = torch.cat([ek["K"][i][ek["pp"][i]] for i in range(T)]).double().T.contiguous()       # (d, n_ans)
        Klast = torch.stack([ek["K"][i][ek["last"][i]] for i in range(T)]).double().T.contiguous()  # (d, T)
        del ek
        Q, Ql = orth(Kans), orth(Klast)
        rec = os.path.join(RC.OUT, log, "COMP_B__n1__s0.json")
        prot = json.load(open(rec))["rank"]["protection"] if os.path.exists(rec) else None
        hk = torch.load(kp)
        W0 = RC.load_w0(log)
        C = torch.load(os.path.join(RC.TB, log, "composite_delta.pt"))
        D = ((W0.float() + C).to(W0.dtype).double() - W0.double())
        K = hk["K"].double().T.contiguous()                         # (d, N)
        DK = D @ K
        out = {"version": VERSION, "log": log, "groups": hk["groups"], "stored_D1": hk["stored_D1"], "ptype": hk["ptype"],
               "n_ans_keys": int(Kans.shape[1]), "rank_K_ans": int(Q.shape[1]), "rank_K_last": int(Ql.shape[1]),
               "protect_rank_recorded_by_COMP_B": prot, "rank_matches_COMP_B": (None if prot is None else int(Q.shape[1]) == int(prot)),
               "d_in": int(K.shape[0]), "fro_D": float(D.norm()),
               "fro_D_on_K_ans_span": float((D @ Q).norm()), "fro_D_on_K_last_span": float((D @ Ql).norm())}
        for tag, B in (("ans", Q), ("last", Ql)):
            PK = B @ (B.T @ K)
            DPK = D @ PK
            a, b, s, j = [], [], [], 0
            for A in hk["counts"]:
                den = DK[:, j:j + A].norm().clamp_min(1e-30)
                a.append(float(DPK[:, j:j + A].norm() / den))
                b.append(float((DK[:, j:j + A] - DPK[:, j:j + A]).norm() / den))
                s.append(float(PK[:, j:j + A].norm() / K[:, j:j + A].norm().clamp_min(1e-30)))
                j += A
            out[tag] = {"a": a, "b": b, "s": s}
            del PK, DPK
        # size of the module-output change itself, relative to the pre-edit output: ||D K_r|| / ||W_0 K_r||
        W0K = W0.double() @ K
        rel, j = [], 0
        for A in hk["counts"]:
            rel.append(float(DK[:, j:j + A].norm() / W0K[:, j:j + A].norm().clamp_min(1e-30)))
            j += A
        out["rel_change"] = rel
        tmp = op + f".tmp{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(out, f)
        os.replace(tmp, op)
        d1 = np.array(hk["stored_D1"])
        for tag in ("ans", "last"):
            a, b, s = (np.array(out[tag][k]) for k in ("a", "b", "s"))
            print(f"{log} K_{tag} (rank {out['rank_K_' + tag]}): median a / b / s  complaints {np.median(a[d1]):.2f} / {np.median(b[d1]):.2f} / "
                  f"{np.median(s[d1]):.2f}; non-complaints {np.median(a[~d1]):.2f} / {np.median(b[~d1]):.2f} / {np.median(s[~d1]):.2f}")
        print(f"{log}: rank K_ans {out['rank_K_ans']} (COMP-B recorded {prot}); ||D P_ans||/||D|| {out['fro_D_on_K_ans_span'] / out['fro_D']:.3f}; "
              f"||D P_last||/||D|| {out['fro_D_on_K_last_span'] / out['fro_D']:.3f}; median ||D K|| / ||W_0 K||: complaints "
              f"{np.median(np.array(rel)[d1]):.3f}, non-complaints {np.median(np.array(rel)[~d1]):.3f}")


if __name__ == "__main__":
    main()
