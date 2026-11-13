"""repair(method, H, complaints, **cfg) -> (Correction, cost)

Methods (names accept '-' or '_'):
  NONE          no correction.
  COMP          compensation: keep every edit, add  X = V pinv((I-P) K_q)  at the edited module,
                V = -(W_E - W_0) K_q  (restores W_0's module output at the query positions; log-derived
                target, no gradients), P = projector onto the protection keys.  alpha is fixed to 1.
                cfg: protect  tuple of {"none","K_last","K_ans","K_seq:<n>","probe:<n>","probe_last:<n>",
                              "K_pc:<r>"} (several entries = union of the spans); default ("K_last",)
                              K_pc:<r> = top-r principal directions of K_all (every position of every
                              logged edit's rel sequence; uncentred second moment, fp64 eigh; RQ3)
                     query    "answer" (answer-predicting positions of the complaint's y0) | "all"
                     form     "adapter" (fp64 side path, gated) | "merged" (bf16(W_E + X))
                     gate     adapter gate (default 1e-8; None = ungated, as the old cb_run H1)
                     tol_rel  pinv cut, singular values < tol_rel * max_j ||k_q,j|| dropped (1e-9)
                Batch mode = pass several complaints: one joint solve.
  REEDIT        the log's own editor installs (q -> y0) on top of W_E (baselines/reedit.py, fixed):
                z* by vendor zopt on W_E (uncached); if a row's z* is not an oracle (vendor early-stop
                off-by-one, PREP_VERIFY B2) it is re-optimised with the fixed loop (early stop on the
                EVALUATED delta) at 2x steps, then at 200 steps / early_stop 0.01; a row that never
                reaches an oracle is NOT installed (recorded).  Install = AlphaEdit solve
                D^T = (P Kp Kp^T + P K1 K1^T + l2 I)^-1 P K1 R^T in the factorised rank-1 form
                (FactorTap), fp64 on the model's GPU, write W_E.add_(D.to(dtype)) exactly like
                Editor._write.  cfg: variant "k0+kp" (default) | "k0", l2 (default: the log's).
                RQ2 options: solve "exact" (z* target with COMP's exact K_ans null-space solve), form
                "adapter", targets= (reuse z*), return_targets; see m_reedit.  COMP: solve "alphaedit".
  ROLLBACK_FO   greedy first-order rollback (P8 protocol): per step d mean log p(y0)/dW at the
                current weights, roll back argmax_i -<g, dW_i>, stop when y0 is reproduced
                (teacher forced) or after `budget` (10) rollbacks.
  ROLLBACK_BEH  best-improvement greedy on the behavioural criterion: per step try every remaining
                edit, keep the one maximising (# y0 tokens reproduced, mean log p(y0)) under teacher
                forcing; stop when y0 is reproduced or after `budget` (20) rollbacks.
  ROLLBACK_ALL  reference: delete every logged edit (W_0).

All forwards/backwards are counted: full (HF forward of the whole model), prefix (layers 0..L),
fwd_bwd (vendor zopt: full forward + backward to the layer-L output), suffix / suffix_grad
(exact suffix engine, layers L+1..N; bit-identical to full forwards, see suffix.py).
Wall-clock is CUDA-synchronised on the same GPU; per-log setup (protection bases, [redacted: describes a related submission under double-blind review],
V^T Kp) is cached in H._setup OUTSIDE the per-complaint timer for every method.
"""
from __future__ import annotations

import os
import time
from dataclasses import replace

import numpy as np
import torch

from .common import CACHE, TB, Timer, sha, sync
from .hlog import Correction
from .suffix import Suffix, tf_score

EPS64 = float(np.finfo(np.float64).eps)


# =========================================================================== cost accounting
class Meter:
    """Counts vlm.forward calls by kind (+ tokens, image tokens) while active."""

    KINDS = ("full", "prefix", "fwd_bwd", "suffix", "suffix_grad")

    def __init__(self, H, suffix: Suffix | None = None):
        self.H, self.S = H, suffix
        self.n = {k: 0 for k in self.KINDS}
        self.tok = {k: 0 for k in self.KINDS}
        self.img = {k: 0 for k in self.KINDS}
        self.ans = {k: 0 for k in self.KINDS}

    def __enter__(self):
        vlm = self.H.vlm
        orig = vlm.forward

        def fwd(batch, logits_to_keep=None, stop_after_layer=None):
            if torch.is_grad_enabled():
                k = "fwd_bwd"
            elif stop_after_layer is not None:
                k = "prefix"
            else:
                k = "full"
            self.n[k] += 1
            self.tok[k] += int(batch["inputs"]["attention_mask"].sum())
            self.img[k] += int(batch["n_image_tokens"].sum())
            self.ans[k] += int(batch["answer_mask"].sum()) if logits_to_keep == "answer" else int(batch["B"])
            return orig(batch, logits_to_keep=logits_to_keep, stop_after_layer=stop_after_layer)
        vlm.forward = fwd
        if self.S is not None:
            self.S.reset_counts()
        return self

    def __exit__(self, *exc):
        del self.H.vlm.forward
        if self.S is not None:
            for k in ("suffix", "suffix_grad"):
                self.n[k] += self.S.n_pass[k]
                self.tok[k] += self.S.tokens[k]
        return False

    def record(self):
        return {"passes": dict(self.n), "tokens": dict(self.tok), "image_tokens": dict(self.img),
                "flops_model": model_flops(self.H, self)}


def _arch(H):
    a = H._setup.get("arch")
    if a is None:
        v = H.vlm
        lay = v._layers()
        p_layer = sum(p.numel() for p in lay[0].parameters())
        if H.backbone == "qwen25vl":
            p_vis = sum(p.numel() for p in v.m.model.visual.parameters())
            vis_per_img_tok = v.merge ** 2           # ViT patches per LM image token
        else:
            p_vis = sum(p.numel() for p in v.m.model.vision_tower.parameters()) + \
                sum(p.numel() for p in v.m.model.multi_modal_projector.parameters())
            vis_per_img_tok = 1
        cfgd = v.m.config.get_text_config()
        a = {"p_layer": p_layer, "n_layers": len(lay), "d_model": cfgd.hidden_size,
             "vocab": cfgd.vocab_size, "p_vis": p_vis, "vis_per_img_tok": vis_per_img_tok}
        H._setup["arch"] = a
    return a


def model_flops(H, m: Meter) -> dict:
    """FLOPs ESTIMATE: 2*P_layer per token per decoder block (+4*T*d attention), 2*P_vis per
    ViT patch, 2*d*V per kept LM-head position; backward = 2x the forward of the blocks it
    traverses (above L).  Uses mean sequence length per pass kind."""
    a = _arch(H)
    N, L = a["n_layers"], H.L
    out = {}
    for k in m.KINDS:
        n, T = m.n[k], m.tok[k]
        if n == 0:
            out[k] = 0.0
            continue
        Tm = T / n
        per_layer = 2 * a["p_layer"] * T + 4 * Tm * a["d_model"] * T
        vis = 2 * a["p_vis"] * a["vis_per_img_tok"] * m.img[k]
        head = 2 * a["d_model"] * a["vocab"] * max(m.ans[k], n)
        mlp_L = 2 * H.m_out * H.d_in * T
        if k == "full":
            f = per_layer * N + vis + head
        elif k == "prefix":
            f = per_layer * (L + 1) + vis
        elif k == "fwd_bwd":
            f = per_layer * N + vis + head + 2 * (per_layer * (N - L - 1) + head)
        elif k == "suffix":
            f = per_layer * (N - L - 1) + mlp_L + head
        else:  # suffix_grad
            f = 3 * (per_layer * (N - L - 1) + head) + 2 * mlp_L
        out[k] = float(f)
    out["total"] = float(sum(out[k] for k in m.KINDS))
    return out


def _cost(method, cfg, meter, timers, solve_flops=0.0, extra=None):
    rec = {"method": method, "cfg": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
           **meter.record(), "flops_solve": float(solve_flops),
           "sec": {k: float(v) for k, v in timers.items()},
           "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
           "torch_threads": torch.get_num_threads()}
    rec["flops_total"] = rec["flops_model"]["total"] + rec["flops_solve"]
    rec["n_backward"] = meter.n["fwd_bwd"] + meter.n["suffix_grad"]
    rec["n_forward"] = sum(meter.n.values())
    if extra:
        rec.update(extra)
    return rec


def target_text(c, cfg):
    return c["pre_long_text"] if cfg.get("target") == "long" else c["y0_text"]


# =========================================================================== NONE / ALL
def m_none(H, complaints, **cfg):
    meter = Meter(H)
    with meter, Timer() as t:
        pass
    return Correction("none", meta={"method": "NONE"}), _cost("NONE", cfg, meter, {"total": t.sec})


def m_rollback_all(H, complaints, **cfg):
    meter = Meter(H)
    with meter, Timer() as t:
        W = H.W0.clone()
    return (Correction("weight", W=W, meta={"method": "ROLLBACK_ALL", "R": list(range(H.T)), "k": H.T}),
            _cost("ROLLBACK_ALL", cfg, meter, {"total": t.sec}))


# =========================================================================== COMP
def _orth(Kc):
    """Orthonormal basis of span(Kc) (d, m) fp64: QR + SVD of R, rank tol max(shape)*eps*s0."""
    Q1, R1 = torch.linalg.qr(Kc)
    Ur, s, _ = torch.linalg.svd(R1)
    tol = max(Kc.shape) * EPS64 * float(s[0])
    r = int((s > tol).sum())
    return Q1 @ Ur[:, :r], {"m": int(Kc.shape[1]), "rank": r, "sigma_min_over_max": float(s[-1] / s[0])}


def _probe_rows(H, n, seed=0):
    """n known-good probe rows: REPAIR pool, y0 valid, still reproduced at W_E (non-complaints)."""
    from .sampler import pool_of
    pl = pool_of(H)
    rows = sorted([r for r in H.rows if pl[r["probe_idx"]] == "repair" and r["y0_valid"]
                   and r["reproduced"] is True], key=lambda r: (r["probe_idx"], r["ptype"]))
    rng = np.random.default_rng(seed)
    n = min(n, len(rows))
    return [rows[int(i)] for i in sorted(rng.choice(len(rows), n, replace=False))]


def _probe_keys(H, rows, which):
    tag = sha([H.name, [(r["probe_idx"], r["ptype"], r["y0_text"]) for r in rows], which])
    p = os.path.join(CACHE, f"probekeys_{H.name}_{tag}.pt")
    if os.path.exists(p):
        return torch.load(p)
    ks = []
    for r in rows:
        b = H.complaint_batch(r)
        K = H.capture_inputs(b)
        A = int(b["answer_len"][0])
        pos = b["pred_pos"][0, :A] if which == "ans" else b["last_prompt_idx"]
        ks.append(K[pos].cpu())
    K = torch.cat(ks, 0)
    torch.save(K, p)
    return K


def pc_basis(H):
    """(U, evals): eigenvectors (d, d) fp64 on H.solve_dev of the uncentred second moment
    G = sum_i K_i^T K_i of K_all (every position of every logged edit's rel sequence, src + alt
    teacher forced = H.edit_keys()), sorted by decreasing eigenvalue (evals on the CPU).
    Cached on disk per (backbone, layer, stream) and in H._setup["pc"].  Used by "K_pc:<r>"."""
    if "pc" in H._setup:
        return H._setup["pc"]
    ek = H.edit_keys()
    tag = sha([H.backbone, H.L, [d["uid"] for d in H.items], "pc_v1"])
    p = os.path.join(CACHE, f"pcbasis_{H.backbone}_L{H.L}_{tag}.pt")
    if os.path.exists(p):
        d = torch.load(p)
    else:
        G = torch.zeros((H.d_in, H.d_in), dtype=torch.float64, device=H.solve_dev)
        for K in ek["K"]:
            Kd = K.to(H.solve_dev).double()
            G.addmm_(Kd.T, Kd)
        ev, U = torch.linalg.eigh(G)
        del G
        d = {"U": U.flip(1).contiguous().cpu(), "evals": ev.flip(0).cpu(),
             "n_keys": int(sum(K.shape[0] for K in ek["K"]))}
        del U
        torch.save(d, p)
    H._setup["pc"] = (d["U"].to(H.solve_dev), d["evals"])
    return H._setup["pc"]


def protection_basis(H, protect):
    """Orthonormal basis (d, r) fp64 on the GPU of the union of the protection spans (cached)."""
    protect = tuple(protect)
    key = ("basis",) + protect
    if key in H._setup:
        return H._setup[key]
    with Timer() as t:
        if protect in ((), ("none",)):
            H._setup[key] = (None, {"rank": 0, "sets": protect, "sec_setup": 0.0})
            return H._setup[key]
        if len(protect) == 1 and protect[0].startswith("K_pc:"):
            # a single principal subspace: its eigenvector block is already an orthonormal basis
            r = int(protect[0].partition(":")[2])
            U, ev = pc_basis(H)
            if r <= 0:
                H._setup[key] = (None, {"rank": 0, "sets": protect, "sec_setup": 0.0})
                return H._setup[key]
            r = min(r, H.d_in)
            Q = U[:, :r].clone()
            st = {"m": r, "rank": r, "sigma_min_over_max": float((ev[r - 1] / ev[0]).clamp(min=0) ** 0.5),
                  "energy_frac": float(ev[:r].sum() / ev.sum())}
            st.update({"sets": protect, "sec_setup": 0.0, "full_rank": r >= H.d_in})
            H._setup[key] = (Q, st)
            return H._setup[key]
        ek = None
        cols = []
        for spec in protect:
            name, _, arg = spec.partition(":")
            if name in ("K_last", "K_ans", "K_seq"):
                ek = ek or H.edit_keys()
            if name == "K_last":
                cols.append(torch.stack([ek["K"][i][ek["last"][i]] for i in range(H.T)]))
            elif name == "K_ans":
                cols.append(torch.cat([ek["K"][i][ek["pp"][i]] for i in range(H.T)]))
            elif name == "K_seq":
                n = int(arg)
                order = sorted(range(H.T), key=lambda i: (ek["len"][i], i))[:n]
                cols.append(torch.cat([ek["K"][i] for i in order]))
            elif name in ("probe", "probe_last"):
                rows = _probe_rows(H, int(arg))
                cols.append(_probe_keys(H, rows, "ans" if name == "probe" else "last"))
            elif name == "K_pc":
                r = min(int(arg), H.d_in)
                if r > 0:
                    cols.append(pc_basis(H)[0][:, :r].T)        # (r, d) fp64 on solve_dev
            elif name == "none":
                continue
            else:
                raise ValueError(spec)
        # (cast piece by piece: values identical to casting after the concatenation)
        if not cols:
            H._setup[key] = (None, {"rank": 0, "sets": protect, "sec_setup": 0.0})
            return H._setup[key]
        Kc = torch.cat([c.to(H.solve_dev).double() for c in cols], 0).T.contiguous()
        Q, st = _orth(Kc)
        del Kc
    st.update({"sets": protect, "sec_setup": t.sec, "full_rank": st["rank"] >= H.d_in})
    H._setup[key] = (Q, st)
    return H._setup[key]


def _comp_query_keys(H, complaints, query, cfg):
    """Module-input keys of the complaints at the query positions, (d, n) fp64 on H.solve_dev."""
    ks = []
    for c in complaints:
        b = H.complaint_batch(c, target_text(c, cfg))
        K = H.capture_inputs(b)
        if query == "answer":
            A = int(b["answer_len"][0])
            K = K[b["pred_pos"][0, :A]]
        elif query == "all":
            K = K[: int(b["seq_len"][0])]
        else:
            raise ValueError(query)
        ks.append(K)
    return torch.cat(ks, 0).to(H.solve_dev).double().T.contiguous()          # (d, n)


def _adapter_from_factors(Lraw, Wfac):
    """D = Lraw Wfac^T as an adapter (Lf, Rf) with orthonormal Rf (QR of Wfac)."""
    Q1, R1 = torch.linalg.qr(Wfac)
    return Lraw @ R1.T, Q1


# [not released: code of the editor that wrote the vision-language logs; it is described in a related submission under double-blind review (README, section "What can be reproduced")]
# Removed here: the functions behind the entry point below.
def _comp_alphaedit(H, complaints, **cfg):
    raise NotImplementedError("COMP with solve=\"alphaedit\" uses the solve and the statistics of the editor that wrote the vision-language "
                              "logs; the editor is not released (README, section \"What can be reproduced\")")


def m_comp(H, complaints, protect=("K_last",), query="answer", form="adapter", gate=1e-8,
           tol_rel=1e-9, solve="exact", variant="k0+kp", l2=None, check_vendor=False,
           prior=None, extra_basis=None, keep_keys=False, **cfg):
    """solve "exact" (default): X = V pinv((I-P) K_q), P = projector onto `protect`.
    RQ4 options (exact solve only; defaults = unchanged):
      prior        an adapter Correction already in place (sequential repair): the target becomes
                   V = (W_0 - (W_E + X_prior)) K_q and the returned adapter is X_prior + X (one
                   adapter, orthonormal R re-derived by QR of [R_prior, R]).
      extra_basis  (d, r2) fp64 orthonormal columns, orthogonal to the `protect` basis (caller's
                   duty), added to the protected span (e.g. keys of earlier repaired complaints).
      keep_keys    put the query keys K_q (d, n) fp64 in corr.meta["_Kq"] (pop before dumping).
    solve "alphaedit" (RQ2 mixed cell): the log-derived target (K_q, V) installed with the log's own
    AlphaEdit solve  X^T = (Pt Kp Kp^T + Pt K_q K_q^T + l2 I)^-1 Pt K_q V^T  (Pt = [redacted: describes a related submission under double-blind review],
    Kp = the log's solver keys if variant "k0+kp", l2 default the log's); `protect` is ignored."""
    if solve not in ("exact", "alphaedit"):
        raise ValueError(solve)
    if solve == "alphaedit":
        return _comp_alphaedit(H, complaints, query=query, form=form, gate=gate, variant=variant, l2=l2,
                               check_vendor=check_vendor, **cfg)
    cfg = dict(cfg, protect=tuple(protect), query=query, form=form, gate=gate, tol_rel=tol_rel)
    if prior is not None or extra_basis is not None:
        if prior is not None and prior.kind not in ("adapter", "none"):
            raise ValueError("prior must be an adapter correction")
        cfg.update(prior_rank=(0 if prior is None or prior.kind == "none" else int(prior.Rf.shape[1])),
                   extra_rank=(0 if extra_basis is None else int(extra_basis.shape[1])))
    if prior is not None and prior.kind == "none":
        prior = None
    Q, bst = protection_basis(H, protect)          # setup: outside the timer
    D_act = H.D_act
    meter = Meter(H)
    tm = {}
    with meter, Timer() as t_all:
        with Timer() as t_cap:
            Kq = _comp_query_keys(H, complaints, query, cfg)
        with Timer() as t_solve:
            V = -(D_act @ Kq)                                       # (m, n)
            if prior is not None:
                V = V - prior.Lf.to(H.solve_dev) @ (prior.Rf.to(H.solve_dev).T @ Kq)
            Kp = Kq if Q is None else Kq - Q @ (Q.T @ Kq)
            if extra_basis is not None:
                E = extra_basis.to(H.solve_dev, torch.float64)
                Kp = Kp - E @ (E.T @ Kp)
            Q1, R1 = torch.linalg.qr(Kp)
            Ur, s, Vh = torch.linalg.svd(R1)
            kqn = Kq.norm(dim=0)
            keep = s > tol_rel * float(kqn.max())
            nk = int(keep.sum())
            if nk:
                Rf = Q1 @ Ur[:, keep]
                Lf = V @ (Vh[keep].T / s[keep])
            else:
                Rf = torch.zeros((H.d_in, 1), dtype=torch.float64, device=H.solve_dev)
                Lf = torch.zeros((H.m_out, 1), dtype=torch.float64, device=H.solve_dev)
            Ls, Rs = Lf, Rf                                       # solve-device copies (diagnostics)
            if prior is not None and nk:                        # X_prior + X as one adapter
                Rc = torch.cat([prior.Rf.to(H.solve_dev), Rs], 1)
                Lc = torch.cat([prior.Lf.to(H.solve_dev), Ls], 1)
                Qc, Tc = torch.linalg.qr(Rc)
                Lf, Rf = Lc @ Tc.T, Qc
            elif prior is not None:
                Lf, Rf = prior.Lf.to(H.solve_dev), prior.Rf.to(H.solve_dev)
            Lf, Rf = Lf.to(H.dev), Rf.to(H.dev)                 # the correction lives with the model
            if form == "merged":
                Wm = (H.WE.double() + Lf @ Rf.T).to(H.dtype)
        tm.update({"capture": t_cap.sec, "solve": t_solve.sec})
    tm["total"] = t_all.sec
    n = Kq.shape[1]
    r = (0 if Q is None else Q.shape[1]) + (0 if extra_basis is None else extra_basis.shape[1])
    d, m = H.d_in, H.m_out
    fl = 2 * m * d * n + 4 * d * r * n + 4 * d * n * n + 10 * n ** 3 + 2 * d * n * nk + 2 * m * n * nk
    if form == "merged":
        fl += 2 * m * d * nk
    # diagnostics (outside the timer)
    Vn = max(float(V.norm()), 1e-30)
    diag = {"n_query": n, "rank_kept": nk, "n_dropped": n - nk, "protect_rank": r,
            "protect_stats": {k: v for k, v in bst.items() if k != "sets"},
            "cond": float(s[0] / s[keep][-1]) if nk else float("inf"),
            "e_mean": float((Kp.norm(dim=0) / kqn).mean()),
            "P1_rel": float((Ls @ (Rs.T @ Kp) - V).norm()) / Vn if nk else 1.0,
            "P2_RtQ_max": float((Rs.T @ Q).abs().max()) if (Q is not None and nk) else 0.0,
            "solve_device": str(H.solve_dev),
            "infeasible": nk == 0, "V_fro": float(V.norm())}
    if prior is not None or extra_basis is not None:
        diag.update({"new_fro": float(Ls.norm()) if nk else 0.0, "total_rank": int(Rf.shape[1]),
                     "P2_RtE_max": (float((Rs.T @ E).abs().max()) if (extra_basis is not None and nk) else 0.0)})
    meta = {"method": "COMP", **diag}
    if keep_keys:
        meta["_Kq"] = Kq
    if form == "adapter":
        corr = Correction("adapter", Lf=Lf, Rf=Rf, gate=gate, meta=meta)
    elif form == "merged":
        corr = Correction("weight", W=Wm, meta=dict(meta, adapter_fro=float(Lf.norm())))
    else:
        raise ValueError(form)
    return corr, _cost("COMP", cfg, meter, tm, fl, {"setup_sec_excluded": bst.get("sec_setup", 0.0),
                                                    "solve_device": str(H.solve_dev)})


# =========================================================================== REEDIT
# [not released: code of the editor that wrote the vision-language logs; it is described in a related submission under double-blind review (README, section "What can be reproduced")]
# Removed here: the functions behind the entry point below and their helpers.
def m_reedit(H, complaints, **cfg):
    raise NotImplementedError("REEDIT re-runs the editor that wrote the vision-language logs; the editor is not released (README, section "
                              "\"What can be reproduced\").  The per-row records and repaired-weight hashes of the REEDIT rows are in "
                              "linear_baselines/streams/out.")


# =========================================================================== rollbacks
def _suffix(H):
    if "suffix" not in H._setup:
        H._setup["suffix"] = Suffix(H)
    return H._setup["suffix"]


def _joint_score(S, states, W, dev=None):
    if dev is None:
        sc = [tf_score(S.logits(s, W=W), s.ans) for s in states]
    else:                                   # offloaded states (RQ4 option of m_rollback_fo)
        sc = []
        for s in states:
            sd = dev(s)
            sc.append(tf_score(S.logits(sd, W=W), sd.ans))
    return (sum(x[0] for x in sc), sum(x[1] for x in sc), float(np.mean([x[2] for x in sc]))), sc


def m_rollback_fo(H, complaints, budget=10, offload=False, record_flags=False, **cfg):
    """RQ4 options (defaults = unchanged): offload keeps the suffix states on the CPU (moved to the
    GPU per use; many complaints, e.g. greedy on a total reported loss); record_flags stores the
    per-complaint teacher-forced reproduced flags after every step in the trajectory."""
    cfg = dict(cfg, budget=budget)
    if offload:
        cfg["offload"] = True
    S = _suffix(H)
    meter = Meter(H, S)
    traj = []
    with meter, Timer() as t_all:
        if offload:
            from .evaluation import _state_cpu, _state_dev
            states, sc = [], []
            for c in complaints:
                st0 = S.prepare(H.complaint_batch(c, target_text(c, cfg)))
                sc.append(tf_score(st0.logits_ref, st0.ans))
                states.append(_state_cpu(st0))
                del st0

            def dev(st):
                return _state_dev(st, H.dev)
        else:
            states = [S.prepare(H.complaint_batch(c, target_text(c, cfg))) for c in complaints]
            sc = [tf_score(s.logits_ref, s.ans) for s in states]
            dev = None
        R, W = [], H.WE
        done = all(x[0] for x in sc)
        flags0 = [bool(x[0]) for x in sc]
        t_score = 0.0
        while not done and len(R) < budget:
            g = None
            for s in states:
                gi, _ = S.grad(s if dev is None else dev(s), W)
                g = gi.float() if g is None else g + gi.float()
            rem = [i for i in range(H.T) if i not in R]
            with Timer() as ts:
                fs = H.fo_scores(g, rem)
            t_score += ts.sec
            i = rem[int(np.argmax(fs))]
            R.append(i)
            W = H.rollback_weight(R)
            (nres, ncor, mlp), sc = _joint_score(S, states, W, dev)
            done = nres == len(states)
            traj.append({"k": len(R), "removed": i, "n_resolved": nres, "n_correct": ncor, "mean_lp": mlp})
            if record_flags:
                traj[-1]["flags"] = [bool(x[0]) for x in sc]
    meta = {"method": "ROLLBACK_FO", "R": R, "k": len(R), "resolved_internal": done, "trajectory": traj}
    if record_flags:
        meta["flags0"] = flags0
    corr = Correction("weight", W=W, meta=meta) if R else Correction("none", meta=meta)
    fl = 4.0 * H.m_out * H.d_in * H.T * len(traj)          # scoring <g, dW_i> for every edit, per step
    return corr, _cost("ROLLBACK_FO", cfg, meter, {"total": t_all.sec, "scoring": t_score}, fl)


def m_rollback_beh(H, complaints, budget=20, **cfg):
    cfg = dict(cfg, budget=budget)
    S = _suffix(H)
    meter = Meter(H, S)
    traj = []
    with meter, Timer() as t_all:
        states = [S.prepare(H.complaint_batch(c, target_text(c, cfg))) for c in complaints]
        R, W = [], H.WE
        sc = [tf_score(s.logits_ref, s.ans) for s in states]
        done = all(x[0] for x in sc)
        n_eval = 0
        while not done and len(R) < budget:
            best, best_key = None, None
            for i in range(H.T):
                if i in R:
                    continue
                key, _ = _joint_score(S, states, H.rollback_weight(R + [i]))
                n_eval += 1
                if best_key is None or key > best_key:
                    best, best_key = i, key
            R.append(best)
            W = H.rollback_weight(R)
            done = best_key[0] == len(states)
            traj.append({"k": len(R), "removed": best, "n_resolved": best_key[0], "n_correct": best_key[1],
                         "mean_lp": best_key[2]})
    meta = {"method": "ROLLBACK_BEH", "R": R, "k": len(R), "resolved_internal": done, "trajectory": traj,
            "n_candidate_evals": n_eval}
    corr = Correction("weight", W=W, meta=meta) if R else Correction("none", meta=meta)
    return corr, _cost("ROLLBACK_BEH", cfg, meter, {"total": t_all.sec})


METHODS = {"NONE": m_none, "COMP": m_comp, "REEDIT": m_reedit, "ROLLBACK_FO": m_rollback_fo,
           "ROLLBACK_BEH": m_rollback_beh, "ROLLBACK_ALL": m_rollback_all}


def repair(method, H, complaints, **cfg):
    """Uniform entry point. Returns (Correction, cost record); the model is at W_E on return."""
    key = method.upper().replace("-", "_")
    if key not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {list(METHODS)}")
    if isinstance(complaints, dict):
        complaints = [complaints]
    H.reset()
    corr, cost = METHODS[key](H, list(complaints), **cfg)
    H.reset()
    corr.meta.setdefault("method", key)
    cost["n_complaints"] = len(complaints)
    cost["correction_fro"] = corr.fro(H)
    return corr, cost
