"""Exact suffix forward: re-run only the decoder blocks after the edited layer L.

For a fixed teacher-forced sequence, everything up to the edited module's input is independent
of the edited weight (single edited module).  ``prepare`` runs ONE full forward and keeps
  mid  = residual stream entering layer L's MLP residual add (input of post_attention_layernorm)
  x    = input of the edited module (down_proj) at every position
  kw_l = the exact kwargs every decoder block l > L received (mask, position embeddings, ...)
``logits(state, W, corr)`` then recomputes  h_L = mid + F.linear(x, W)  (optionally + adapter),
blocks L+1..N-1, the final norm and the LM head at the answer-predicting positions -- the same
ops on the same shapes as the full forward, so the logits are BIT-IDENTICAL to a full forward at
weight W (asserted at prepare time for the current weight; tests/test_units.py checks other W).
Batch size 1 only.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as Fnn

from .hlog import adapter_apply


@dataclass
class SuffixState:
    mid: torch.Tensor
    x: torch.Tensor
    kws: list
    gather: torch.Tensor          # (1, A, d) gather index for the LM-head input
    ans: torch.Tensor             # (A,) answer ids
    A: int
    T: int
    logits_ref: torch.Tensor      # (A, V) logits of the preparing full forward


class Suffix:
    def __init__(self, H):
        self.H = H
        v = H.vlm
        self.layers = v._layers()
        self.n = len(self.layers)
        self.L = H.L
        lm = v.m.model.language_model
        self.norm = lm.norm
        self.lm_head = v.lm_head
        self.n_pass = {"full": 0, "suffix": 0, "suffix_grad": 0}
        self.tokens = {"full": 0, "suffix": 0, "suffix_grad": 0}

    def reset_counts(self):
        for k in self.n_pass:
            self.n_pass[k] = 0
            self.tokens[k] = 0

    @torch.no_grad()
    def prepare(self, batch) -> SuffixState:
        H, L = self.H, self.L
        if batch["B"] != 1:
            raise ValueError("suffix engine is B=1 only")
        st = {}
        kws = [None] * self.n

        def pre_mid(mod, args):
            st["mid"] = args[0].detach()

        def post_mod(mod, args, out):
            st["x"] = args[0].detach()
            st["out"] = out.detach()

        def mk(l):
            def pre(mod, args, kwargs):
                kws[l] = dict(kwargs)
                if len(args) > 1:
                    raise RuntimeError("unexpected positional args to a decoder layer")
            return pre
        hs = [self.layers[L].post_attention_layernorm.register_forward_pre_hook(pre_mid),
              H.mod.register_forward_hook(post_mod)]
        hs += [self.layers[l].register_forward_pre_hook(mk(l), with_kwargs=True) for l in range(L + 1, self.n)]
        try:
            o = H.vlm.forward(batch, logits_to_keep="answer")
        finally:
            for h in hs:
                h.remove()
        self.n_pass["full"] += 1
        self.tokens["full"] += int(batch["seq_len"][0])
        A = int(batch["answer_len"][0])
        pp = batch["pred_pos"]
        gather = pp[:, :, None].expand(-1, -1, st["mid"].shape[-1])
        s = SuffixState(mid=st["mid"], x=st["x"], kws=kws, gather=gather,
                        ans=batch["answer_ids"][0, :A], A=A, T=int(batch["seq_len"][0]),
                        logits_ref=o.logits[0, :A])
        return s

    def _run(self, s: SuffixState, W, corr):
        out = Fnn.linear(s.x, W)
        if corr is not None and corr.kind == "adapter":
            out = adapter_apply(s.x, out, corr.Lf, corr.Rf, corr.gate)
        h = s.mid + out
        for l in range(self.L + 1, self.n):
            h = self.layers[l](h, **s.kws[l])
            if isinstance(h, tuple):
                h = h[0]
        h = self.norm(h)
        hp = torch.gather(h, 1, s.gather)
        return self.lm_head(hp)[0, :s.A]

    @torch.no_grad()
    def logits(self, s: SuffixState, W=None, corr=None):
        """Logits (A, V) at weight W (default: the module's current weight) [+ adapter corr]."""
        if corr is not None and corr.kind == "weight":
            W = corr.W
        W = self.H.mod.weight if W is None else W
        self.n_pass["suffix"] += 1
        self.tokens["suffix"] += s.T
        return self._run(s, W, corr)

    def grad(self, s: SuffixState, W=None):
        """(d mean log p(answer) / dW, mean log p) at weight W -- one suffix forward + backward."""
        W = (self.H.mod.weight if W is None else W).detach().clone().requires_grad_(True)
        with torch.enable_grad():
            lg = self._run(s, W, None)
            lp = torch.log_softmax(lg.float(), -1)
            mean_lp = lp[torch.arange(s.A, device=lp.device), s.ans].mean()
            g, = torch.autograd.grad(mean_lp, W)
        self.n_pass["suffix_grad"] += 1
        self.tokens["suffix_grad"] += s.T
        return g.detach(), float(mean_lp.detach())


def tf_score(logits, ans):
    """(resolved, n_correct, mean log p) of the answer under teacher forcing."""
    am = logits.argmax(-1)
    eq = am == ans
    lp = torch.log_softmax(logits.float(), -1)[torch.arange(len(ans), device=logits.device), ans]
    return bool(eq.all()), int(eq.sum()), float(lp.mean())
