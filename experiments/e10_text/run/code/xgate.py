"""E10 gate level (metric iv): decisions of many thresholdings of the same detectors on the cached keys, one pass per feature.

    lay = Layout(L, parts)                               # texts and rows of the parts of a log
    res = evaluate(L, lay, groups, device)               # groups: [(scoped log (its detectors = the feature), [Variant, ...])]
    st = stats(L, lay, res[tag])                         # the numbers of e10t.gate.gate_stats (+ the split at position 173)

Semantics = e10t.gate.decide / gate_stats (tested against them on the development log: t_gate.py): a query = the cached keys
of [prompt, 10-token pre-edit continuation]; a detector is a candidate at a position iff r >= theta; fire 'top1': per position
the candidate with the largest margin (r - b) / u (first maximum = lowest index) releases; fire 'all': every candidate edit.
  fired            some position has a candidate            own_cand      the row's own edit is a candidate somewhere
  own_released     the own edit releases somewhere (fire 'all': is a candidate);  other_released: another edit releases
  fired_prompt     a prompt position fires;   fired_last_only: fires only at the tenth continuation token
Differences from the adapter's code: the decision is taken per position on `device` (cuda: last-bit differences of the fp64
products; the adapter derives the per-position decision from the query maximum on the CPU); no decision band is reported.
One detector per edit is assumed (checked).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

import xcommon as X
import e10t
from e10t import gate as G

OWN_PARTS = ("request", "paraphrase", "neighborhood")


@dataclass
class Variant:
    tag: str
    theta: torch.Tensor                # (D,) fp64 effective thresholds
    b: torch.Tensor | None = None      # ranking baselines (two-block) or None
    u: torch.Tensor | None = None
    fire: str = "top1"


def variant_of(tag, S, fire=None) -> Variant:
    return Variant(tag, S.theta().clone(), None if S.det_b is None else S.det_b.clone(), None if S.det_u is None else S.det_u.clone(), fire or S.fire)


def light(S):
    """The gate with only what scoring needs (detectors, scales, kinds, geometry): small on the GPU."""
    if S.D != S.T:
        raise NotImplementedError("one detector per edit")
    z = torch.zeros(S.T, 1, dtype=torch.float64)
    return S.replace(guard_key=S.guard_key[:0], guard_edit=S.guard_edit[:0], det_key=S.det_w, lin_w=S.det_w, values=z, lin_u=z, det_b=None, det_u=None)


class Layout:
    def __init__(self, L, parts=("heldout", "paraphrase", "neighborhood", "request")):
        self.parts = tuple(parts)
        self.rows = {p: G.part_rows(L, p) for p in parts}
        self.tids = list(dict.fromkeys(int(r["text_id"]) for p in parts for r in self.rows[p]))
        tix = {t: i for i, t in enumerate(self.tids)}
        self.npr = np.array([L.kc.n_prompt(t) for t in self.tids], dtype=np.int64)
        self.npos = self.npr + e10t.N_DEC
        self.text = {p: np.array([tix[int(r["text_id"])] for r in self.rows[p]], dtype=np.int64) for p in parts}
        self.own = {p: np.array([int(r["pos"]) if p in OWN_PARTS else -1 for r in self.rows[p]], dtype=np.int64) for p in parts}
        self.clean = {p: np.array([bool(r.get("clean", True)) for r in self.rows[p]]) for p in parts}

    def chunks(self, max_pos):
        a, n = 0, len(self.tids)
        while a < n:
            b, tot = a, 0
            while b < n and (b == a or tot + self.npos[b] <= max_pos):
                tot += int(self.npos[b])
                b += 1
            yield a, b
            a = b


@torch.no_grad()
def evaluate(L, lay: Layout, groups, device="cuda", max_pos: int = 8192, progress=None) -> dict:
    """-> {tag: {'fired' (n_text,), 'fired_prompt', 'last_only', 'n_cand' (n_text,), part: {'own_cand', 'own_rel', 'oth_rel'} (rows)}}"""
    nt = len(lay.tids)
    dev = torch.device(device)
    gs = []
    for S, variants in groups:
        Sd = light(S).to(dev)
        gs.append((Sd, [(v, v.theta.to(dev), None if v.b is None else v.b.to(dev), None if v.u is None else v.u.to(dev)) for v in variants]))
    res = {}
    for _, vs in gs:
        for v, *_ in vs:
            if v.tag in res:
                raise ValueError(f"duplicate variant tag {v.tag}")
            res[v.tag] = {"fired": np.zeros(nt, bool), "fired_prompt": np.zeros(nt, bool), "last_only": np.zeros(nt, bool), "n_cand": np.zeros(nt, np.int32),
                          **{p: {k: np.zeros(len(lay.rows[p]), bool) for k in ("own_cand", "own_rel", "oth_rel")} for p in lay.parts if p in OWN_PARTS}}
    order = {p: np.argsort(lay.text[p], kind="stable") for p in lay.parts if p in OWN_PARTS}
    stext = {p: lay.text[p][order[p]] for p in order}
    npr_t = torch.from_numpy(lay.npr).to(dev)
    for ci, (a, b) in enumerate(lay.chunks(max_pos)):
        K, off = L.kc.stack(lay.tids[a:b])
        Kd = K.to(dev)
        cnt = (off[1:] - off[:-1]).to(dev)
        ptext = torch.repeat_interleave(torch.arange(b - a, device=dev), cnt)
        pin = torch.arange(Kd.shape[0], device=dev) - off[:-1].to(dev)[ptext]
        is_prompt = pin < npr_t[a:b][ptext]
        is_last = pin == npr_t[a:b][ptext] + e10t.N_DEC - 1
        sel = {}
        for p in order:                                               # rows of the part whose text is in this chunk
            lo, hi = np.searchsorted(stext[p], a, "left"), np.searchsorted(stext[p], b, "left")
            ridx = order[p][lo:hi]
            sel[p] = (ridx, torch.from_numpy(lay.text[p][ridx] - a).to(dev), torch.from_numpy(lay.own[p][ridx]).to(dev))
        for Sd, vs in gs:
            R = Sd.scores(Kd)
            for v, th, vb, vu in vs:
                cand = R >= th[None, :]
                fpos = cand.any(1)
                ctext = torch.zeros(b - a, R.shape[1], dtype=torch.float32, device=dev).index_add_(0, ptext, cand.float()) > 0
                seg = lambda m: torch.zeros(b - a, dtype=torch.float32, device=dev).index_add_(0, ptext, m.float()) > 0      # noqa: E731
                fired = ctext.any(1)
                out = res[v.tag]
                out["fired"][a:b] = fired.cpu().numpy()
                out["fired_prompt"][a:b] = seg(fpos & is_prompt).cpu().numpy()
                out["last_only"][a:b] = (fired & ~seg(fpos & ~is_last)).cpu().numpy()
                out["n_cand"][a:b] = ctext.sum(1).cpu().numpy()
                if v.fire == "top1":
                    rk = R
                    if vb is not None:
                        rk = rk - vb[None, :]
                    if vu is not None:
                        rk = rk / vu[None, :]
                    top = torch.where(cand, rk, torch.full_like(rk, -float("inf"))).argmax(1)
                    rel = torch.zeros(b - a, R.shape[1], dtype=torch.bool, device=dev)
                    rel[ptext[fpos], top[fpos]] = True
                else:
                    rel = ctext
                nrel = rel.sum(1)
                for p, (ridx, lt, own) in sel.items():
                    if ridx.size == 0:
                        continue
                    orel = rel[lt, own]
                    out[p]["own_rel"][ridx] = orel.cpu().numpy()
                    out[p]["oth_rel"][ridx] = ((nrel[lt] - orel.long()) > 0).cpu().numpy()
                    out[p]["own_cand"][ridx] = ctext[lt, own].cpu().numpy()
            del R
        if progress and ci % progress == 0:
            X.say(f"  gate level: chunk {ci} (texts {b}/{nt})")
    return res


def _pct(x):
    x = np.asarray(x)
    return float(100.0 * x.mean()) if x.size else None


def stats(L, lay: Layout, r: dict, split: int | None = None) -> dict:
    """The numbers of e10t.gate.gate_stats per part; split: also for the rows of the edits before / from that position."""
    out = {}
    for p in lay.parts:
        t = lay.text[p]
        fired = r["fired"][t]
        o = {"n": int(t.size), "fired": _pct(fired), "fired_prompt": _pct(r["fired_prompt"][t]), "fired_cont_only": _pct(fired & ~r["fired_prompt"][t]),
             "fired_last_only": _pct(r["last_only"][t]),
             "lineage_width_mean_if_fired": float(r["n_cand"][t][fired].mean()) if fired.any() else None,
             "lineage_width_max": int(r["n_cand"][t].max()) if t.size else None}
        if p in ("heldout", "repair", "neighborhood"):
            c = lay.clean[p]
            o.update(n_clean=int(c.sum()), fired_clean=_pct(fired[c]), n_near_scope=int((~c).sum()), fired_near_scope=_pct(fired[~c]))
        if p in OWN_PARTS:
            q = r[p]
            o.update(own_cand=_pct(q["own_cand"]), own_released=_pct(q["own_rel"]), other_released=_pct(q["oth_rel"]),
                     only_other_released=_pct(q["oth_rel"] & ~q["own_rel"]))
            if split is not None and L.T > split:
                for nm, m in (("before", lay.own[p] < split), ("from", lay.own[p] >= split)):
                    o[nm] = {"n": int(m.sum()), "fired": _pct(fired[m]), "own_released": _pct(q["own_rel"][m]), "other_released": _pct(q["oth_rel"][m]),
                             "only_other_released": _pct((q["oth_rel"] & ~q["own_rel"])[m])}
        out[p] = o
    return out
