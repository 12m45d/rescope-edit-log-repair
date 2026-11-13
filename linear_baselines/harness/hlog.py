"""load(log) -> HLog: the model at W_E with exact W_0 / W_E, the edit log's per-edit terms,
cached keys, and the one way every method applies / removes a correction.

Testbed v3 logs (06_testbed_v3/<name>):
  W_0  = the checkpoint weight of the edited module (bf16 / fp16)
  W_E  = (W_0.float() + sum_b A_b).to(dtype), A_b = stored exact applied batch delta (fp32),
         summed in batch order -- bit-identical to tb_load.TestbedLog.rebuild_WE; checked against
         the stored composite_delta.pt (W_E.float() - W_0.float()) at load time.
  rollback of an edit set R (order irrelevant):
         W(R) = (W_0.float() + (C - X(R))).to(dtype),  C = W_E.float() - W_0.float() (fp32, == the stored
         composite_delta.pt), X(R) = sum of the exact applied batch deltas A_b of the batches entirely
         in R + sum_{other i in R} outer(U_i, W_i) (the rank-1 solver term of edit i, fp64 -> fp32;
         NO re-solve of the remaining edits of a batch) -- the convention of PREP_VERIFY's vc_behav.
         W(empty) == W_E bitwise (checked at load); R = all edits is defined as W_0 exactly.
old_W2 (the OLD weak testbed, P8 build, B=1): W_E = W_ckpt += bf16(total fp32), W_0 = W_E - bf16(total)
  (P8's M(0), which differs from the checkpoint), rollback = P8's in-place sequential
  W -= bf16(delta_i) in the given order (dense fp32 deltas from w2/editlog.pkl, loaded on demand).

Protocol notes (PREP_VERIFY fix 2): complaints are decoder-v2 rows; D1 teacher-forces the RE-TOKENISED
stripped y0 (for LLaVA the leading SentencePiece space before a digit is attributed to the prompt, so
the first generated token is not checked in 14/26/34 D1 complaints of TB-L100-ae l2=10/3/1); y0_valid
requires W_0 to reproduce y0, so D1 stays self-consistent.  D2 is too sparse per small subset: report
it pooled.

Corrections (``Correction``):
  kind "none"
  kind "adapter": fp64 side path at the edited module, out <- dtype(out(fp64) + gate(x R) L^T)
                  (R orthonormal => ||correction||_F = ||L||_F); gate zeroes |x R| < gate*||x||
                  (fp64 round-off of keys inside the protected span) -- gate=None disables it.
  kind "weight":  a full replacement weight (dtype), e.g. merged bf16 / rollback / re-edit.
"""
from __future__ import annotations

import contextlib
import os
import pickle
from dataclasses import dataclass, field

import numpy as np
import torch

from .common import CACHE, EXP, TB, TB_ROOT, TestbedLog, sha

W2_DIR = os.path.join(EXP, "01_v1_attribution", "w2")
CB_CACHE = os.path.join(EXP, "05_zeroloss_repair", "cb_run", "cache")      # read-only use
GPU_DELTA_BUDGET = 1.0e9          # keep the exact batch deltas on the GPU only below this size


# --------------------------------------------------------------------------- corrections
@dataclass
class Correction:
    kind: str = "none"                     # none | adapter | weight
    Lf: torch.Tensor | None = None         # (m, r) fp64
    Rf: torch.Tensor | None = None         # (d, r) fp64, orthonormal columns
    gate: float | None = None
    W: torch.Tensor | None = None          # (m, d) module dtype, full replacement weight
    meta: dict = field(default_factory=dict)

    def fro(self, H: "HLog") -> float:
        """||effective weight change||_F relative to W_E (fp64)."""
        if self.kind == "none":
            return 0.0
        if self.kind == "adapter":
            return float(self.Lf.norm())
        return float((self.W.double() - H.WE.double()).norm())


def adapter_apply(x, out, Lf, Rf, gate=None):
    xd = x.double()
    z = xd @ Rf
    if gate is not None:
        z = torch.where(z.abs() < gate * xd.norm(dim=-1, keepdim=True), torch.zeros_like(z), z)
    return (out.double() + z @ Lf.T).to(out.dtype)


# --------------------------------------------------------------------------- log handle
class HLog:
    def __init__(self, name: str, device: str = "cuda:0", vlm=None, old_w2_dense: bool = False,
                 solve_device: str | None = None):
        self.name = name
        self.old_w2 = name == "old_W2"
        self.cfg = dict(TB.LOGS["[redacted: describes a related submission under double-blind review]"] if self.old_w2 else TB.LOGS[name])
        self.backbone, self.L = self.cfg["backbone"], int(self.cfg["layer"])
        self.path = os.path.join(TB_ROOT, name)
        self.checks = {}
        self.vlm = vlm if vlm is not None else TB.VLM(self.backbone, device=device)
        self.vlm.m.eval()
        self.dev = self.vlm.device
        # device of the closed-form solves of COMP and REEDIT (always the SAME for both methods):
        # the GPU for Qwen; the CPU (fp64, like the vendor editor) for LLaVA-7B, whose fp16 weights
        # alone take ~14 GB, so the fp64 [redacted: describes a related submission under double-blind review] (1 GB) + W_E - W_0 (0.36 GB) would push the
        # process past the 16 GB budget.
        self.solve_dev = torch.device(solve_device or ("cpu" if self.backbone == "llava" else str(self.dev)))
        self.mod = self.vlm.edit_module(self.L)
        self.module_name = self.vlm.edit_module_name(self.L)
        self.W_ckpt = self.mod.weight.detach().clone()
        self.dtype = self.W_ckpt.dtype
        self.m_out, self.d_in = self.W_ckpt.shape
        self.items, _ = TB.load_stream(self.cfg["stream"])
        self.T = len(self.items)
        self.probes, self.probe_info = TB.select_probes(self.items, key_exclude=(self.backbone == "llava"))
        self.rows = TB.jsonl_read(os.path.join(self.path, "probe_records_v2.jsonl"))
        self.row = {(r["probe_idx"], r["ptype"]): r for r in self.rows}
        self.checks["probe_uids_match_records"] = all(
            self.probes[r["probe_idx"]]["uid"] == r["uid"] for r in self.rows)
        # PREP_VERIFY A4: LLaVA l2=1 is a collapsed working point (T-/I-Loc 56.3/59.8, Rel exact 72 < 88
        # at l2=3) -- exclude from main results, report only as a degenerate stress point.
        self.checks["collapsed_working_point"] = name == "TB-L100-ae-l2_1"
        if self.old_w2:
            self._init_old_w2(old_w2_dense)
        else:
            self._init_tb()
        self.prober = TB.Prober(self.vlm)
        self._D_act = None
        self._edit_keys = None
        self._setup = {}                    # per-log method setup (protection bases, spectra, ...)
        self.reset()
        if not self.old_w2:
            self.checks["rollback_empty_equals_WE"] = bool(torch.equal(self.rollback_weight([]), self.WE))

    # ------------------------------------------------------------------ init
    def _init_tb(self):
        tl = TestbedLog(self.path)
        self.tl = tl
        self.batches = [list(b["edit_ids"]) for b in tl.per_batch]
        self.batch_of = {i: bi for bi, ids in enumerate(self.batches) for i in ids}
        A = [tl.batch_delta(b) for b in range(len(self.batches))]
        on_gpu = sum(a.numel() * 4 for a in A) <= GPU_DELTA_BUDGET
        self.A = [a.to(self.dev) if on_gpu else a for a in A]
        self.W0 = self.W_ckpt
        S = torch.zeros(self.W0.shape, dtype=torch.float32, device=self.dev)
        for a in self.A:
            S += a.to(self.dev)
        self.WE = (self.W0.float() + S).to(self.dtype)
        comp = tl.composite_delta().to(self.dev)
        self.checks["WE_float_minus_W0_equals_stored_composite"] = bool(
            torch.equal(self.WE.float() - self.W0.float(), comp))
        self.C = self.WE.float() - self.W0.float()
        self.checks["C_equals_stored_composite"] = bool(torch.equal(self.C, comp))
        del comp, S
        fac = tl.factors
        self.U = fac["U"].to(self.dev, torch.float64)          # (T, m)
        self.Wfac = fac["W"].to(self.dev, torch.float64)       # (T, d)
        self.Kp_log = fac["K1_solver"].float().T.contiguous()  # (d, T) stored solver keys
        self.checks["factor_uids_match_stream"] = list(fac["uids"]) == [d["uid"] for d in self.items]
        tl._fac = None

    def _init_old_w2(self, dense: bool):
        self.batches = [[i] for i in range(self.T)]
        self.batch_of = {i: i for i in range(self.T)}
        self.dense = None
        if dense:
            with open(os.path.join(W2_DIR, "editlog.pkl"), "rb") as fh:
                el = pickle.load(fh)
            assert [e.fact_id for e in el.entries] == [d["uid"] for d in self.items]
            m, d = self.W_ckpt.shape
            self.dense = torch.empty((self.T, m * d), dtype=torch.float32)
            tot = torch.zeros((m, d), dtype=torch.float32, device=self.dev)
            for i, e in enumerate(el.entries):            # P8 order, fp32 on the GPU
                di = e.delta_W.to(torch.float32)
                self.dense[i] = di.reshape(-1)
                tot += di.to(self.dev)
            del el
            total = tot
            cached = torch.load(os.path.join(CB_CACHE, "total_delta_fp32.pt")).to(self.dev)
            self.checks["old_w2_total_equals_cb_cache"] = bool(torch.equal(total, cached))
            del cached
        else:
            total = torch.load(os.path.join(CB_CACHE, "total_delta_fp32.pt")).to(self.dev)
        WE = self.W_ckpt.clone()
        WE += total.to(self.dtype)                         # exactly P8: weight.data += total.to(bf16)
        self.WE = WE
        self.W0 = WE - total.to(self.dtype)                # P8's M(0)
        self.checks["old_w2_W0_entries_differing_from_checkpoint"] = int((self.W0 != self.W_ckpt).sum())
        del total

    # ------------------------------------------------------------------ weights
    def set_weight(self, W):
        with torch.no_grad():
            self.mod.weight.data.copy_(W)

    def reset(self):
        self.set_weight(self.WE)
        if not torch.equal(self.mod.weight.data, self.WE):
            raise RuntimeError("reset to W_E is not bitwise")

    @property
    def D_act(self):
        """W_E - W_0 in fp64 (the weight difference actually in use)."""
        if self._D_act is None:
            self._D_act = (self.WE.double() - self.W0.double()).to(self.solve_dev)
        return self._D_act

    @contextlib.contextmanager
    def at(self, W):
        self.set_weight(W)
        try:
            yield
        finally:
            self.reset()

    @contextlib.contextmanager
    def applied(self, corr: Correction | None):
        if corr is None or corr.kind == "none":
            yield
            return
        if corr.kind == "weight":
            with self.at(corr.W):
                yield
            return
        if corr.kind != "adapter":
            raise ValueError(corr.kind)
        Lf, Rf, g = corr.Lf, corr.Rf, corr.gate

        def hook(mod, args, out):
            return adapter_apply(args[0], out, Lf, Rf, g)
        h = self.mod.register_forward_hook(hook)
        try:
            yield
        finally:
            h.remove()

    # ------------------------------------------------------------------ rollback
    def rollback_weight(self, R) -> torch.Tensor:
        """Weights with the edits in R rolled back (see module doc). R: iterable of edit ids
        (an ORDERED list for old_W2, whose P8 semantics is sequential bf16 subtraction)."""
        R = [int(i) for i in R]
        if self.old_w2:
            if self.dense is None:
                raise RuntimeError("old_W2 rollback needs load('old_W2', old_w2_dense=True)")
            W = self.WE.clone()
            m, d = W.shape
            with torch.no_grad():
                for i in R:
                    W -= self.dense[i].view(m, d).to(self.dev).to(self.dtype)
            return W
        Rs = set(R)
        if len(Rs) == self.T:
            return self.W0.clone()
        if not Rs:
            return (self.W0.float() + self.C).to(self.dtype)
        X = torch.zeros(self.W0.shape, dtype=torch.float32, device=self.dev)
        part = []
        for b, ids in enumerate(self.batches):
            inR = [i for i in ids if i in Rs]
            if inR and len(inR) == len(ids):
                X += self.A[b].to(self.dev)
            else:
                part += inR
        if part:
            idx = torch.tensor(sorted(part), device=self.dev)
            X += (self.U[idx].T @ self.Wfac[idx]).float()
        return (self.W0.float() + (self.C - X)).to(self.dtype)

    def fo_scores(self, g: torch.Tensor, remaining) -> np.ndarray:
        """First-order benefit of rolling back each remaining edit: -<g, dW_i>
        (g = d mean log p(y0) / dW).  testbed v3: dW_i = rank-1 solver term (fp64);
        old_W2: P8's dense fp32 applied delta, scored on the CPU exactly as run_p8.py."""
        rem = [int(i) for i in remaining]
        if self.old_w2:
            gc = g.detach().float().cpu().reshape(-1)
            sub = self.dense[rem]
            return (-(gc @ sub.T)).numpy().astype(np.float64)
        idx = torch.tensor(rem, device=self.dev)
        GW = g.detach().double() @ self.Wfac[idx].T              # (m, n)
        return (-(self.U[idx].T * GW).sum(0)).cpu().numpy()

    # ------------------------------------------------------------------ batches
    def probe_item(self, c):
        return self.probes[c["probe_idx"]]

    def complaint_batch(self, c, answer=None):
        """Teacher-forced batch of a probe row / complaint (Prober.batch: the D1 definition)."""
        ans = c["y0_text"] if answer is None else answer
        return self.prober.batch(self.probe_item(c), c["ptype"], answer=ans)

    def edit_batch(self, i, metric="rel"):
        it = self.items[i]
        img = it["image_rephrase"] if metric == "igen" else it["image"]
        q = it["rephrase"] if metric == "tgen" else it["src"]
        return self.vlm.build_batch([img], [q], answers=[it["alt"]], fmt="vqa")

    def complaint_item(self, c, answer=None):
        """The complaint as an editor item (image, src, alt): what REEDIT installs."""
        it = self.probe_item(c)
        if c["ptype"] == "image":
            img, src = it["m_loc"], it["m_loc_q"]
        else:
            img, src = None, it["loc"]
        if self.backbone != "qwen25vl" and c["ptype"] == "text" and TB.TEXT_FMT[self.backbone] != "vqa":
            raise NotImplementedError("editor uses fmt 'vqa'")
        return {"uid": f"complaint:{self.name}:{c['ptype']}_{c['probe_idx']}", "image": img, "src": src,
                "alt": c["y0_text"] if answer is None else answer}

    # ------------------------------------------------------------------ keys
    @torch.no_grad()
    def capture_inputs(self, batch) -> torch.Tensor:
        """Edited-module input at every position (T, d_in), module dtype; one truncated forward
        (layers 0..L).  It does not depend on the edited weight (single edited module)."""
        with self.vlm.capture(batch, [self.L], positions="all", module="edit", io=("in",), dtype=None) as cap:
            self.vlm.forward(batch, stop_after_layer=self.L)
        return cap.inp[self.L][0]

    def edit_keys(self):
        """Per edit sequence (src + alt, teacher forced): K (T_i, d_in) CPU, last prompt index,
        answer-predicting positions, image-token mask.  Cached per (backbone, layer, stream)."""
        if self._edit_keys is not None:
            return self._edit_keys
        tag = sha([self.backbone, self.L, [d["uid"] for d in self.items]])
        p = os.path.join(CACHE, f"editkeys_{self.backbone}_L{self.L}_{tag}.pt")
        if os.path.exists(p):
            self._edit_keys = torch.load(p)
            return self._edit_keys
        out = {"K": [], "last": [], "pp": [], "img": [], "len": []}
        for i in range(self.T):
            b = self.edit_batch(i, "rel")
            K = self.capture_inputs(b).cpu()
            A = int(b["answer_len"][0])
            out["K"].append(K)
            out["last"].append(int(b["last_prompt_idx"][0]))
            out["pp"].append(b["pred_pos"][0, :A].cpu())
            out["img"].append((b["inputs"]["input_ids"][0] == self.vlm.image_token_id).cpu())
            out["len"].append(int(b["seq_len"][0]))
        torch.save(out, p)
        self._edit_keys = out
        return out


def load(name: str, device: str = "cuda:0", vlm=None, **kw) -> HLog:
    """kw: old_w2_dense (old_W2 rollbacks), solve_device (override of the COMP/REEDIT solve device)."""
    return HLog(name, device=device, vlm=vlm, **kw)
