"""Gate-level evaluation of a scoped gate on text prompts, from the key cache (no model).

A QUERY is a prompt text; its keys are the cached inputs of the edited module under the pre-edit model at the L positions of
the prompt and the N_DEC = 10 positions of the pre-edit continuation.  The role set is ALL of them (E10).

    F = e10t.gate.features(L, S, text_ids)          Q (n, D) fp64: largest feature of every detector over the query's positions
                                                    (= rescope ScopedLog.query_max of the cached keys; the `coords` of the library's
                                                    calibrate_two_block / calibrate_shared), kmax (n,): largest key norm of the query
    d = e10t.gate.decide(L, S, text_ids, own=None)  the gate's decision per query (arrays aligned with text_ids):
        fired          some detector is a candidate at some position        (Q[q, j] >= theta_j for some j)
        near           some |Q[q, j] - theta_j| lies inside the decision band BAND * (|theta_j| + |w_j| max|k| / s_j): another
                       batch layout of the forward may decide otherwise; behavioural code runs such prompts through the hook
        sure_silent    not fired and not near: the prompt is served bit for bit by the pre-edit model
        n_cand_edits   number of edits that are candidates somewhere in the query (lineage width)
        rel            per fired query the list of (position, releasing edit) under fire 'top1' (None under fire 'all')
        n_fire_pos, fired_prompt (a position of the prompt fires), fired_cont (a position of the continuation fires),
        fired_last_only (fires only at the 10th continuation token, whose module output cannot change the 10 measured tokens)
        own, own_cand, own_released, other_released     with own = the query's own edit (-1: none): it is a candidate somewhere /
                       it is the releasing edit at some position / another edit is released at some position
        margin         max_j (Q[q, j] - theta_j)   (>= 0 <=> fired)

ARITHMETIC.  Features are rescope's ScopedLog.scores of the cached float32 keys in fp64, in chunks of CHUNK_POS positions; the
per-position refinement of a fired query re-scores only the query's candidate detectors (same formulas) and treats the
position of the query maximum as a candidate, so that it agrees with the query-level decision.  Thresholds, ranking
(margin (r - b) / u after a two-block calibration) and ties (lowest detector index) are the library's.  Last-bit differences
between a feature computed here and in the hook are far inside the decision band.
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch

import e10t
from rescope.scoped import BALL, COS, INF, LIN

CHUNK_POS = 8192
_FEAT, _FEAT_MAX = {}, 4


def fkey(S) -> str:
    """Identity of a gate's FEATURE (detectors, scales, kinds, keys of ball detectors): sha1, cached on the object."""
    k = S._cache.get("e10t_fkey")
    if k is None:
        h = hashlib.sha1()
        for t in (S.det_w, S.det_s, S.det_kind):
            h.update(np.ascontiguousarray(t.detach().cpu().numpy()).tobytes())
        if bool((S.det_kind == BALL).any()):
            h.update(np.ascontiguousarray(S.det_key.detach().cpu().numpy()).tobytes())
        k = S._cache.setdefault("e10t_fkey", h.hexdigest()[:16])
    return k


def clear_cache():
    _FEAT.clear()


def features(L, S, tids, device: str = "cpu") -> dict:
    """Per-query maxima of the gate's feature over ALL cached positions of each text -> {'Q': (n, D) fp64 cpu, 'kmax': (n,) fp64,
    'n_pos': (n,) long}.  Cached per feature (detector set) and text; device: where the products are computed ('cpu': the
    reference, the same bits in every process; 'cuda': faster for long logs, last-bit differences)."""
    tids = [int(t) for t in tids]
    key = (fkey(S), str(device))
    ent = _FEAT.get(key)
    if ent is None:
        while len(_FEAT) >= _FEAT_MAX:
            _FEAT.pop(next(iter(_FEAT)))
        ent = _FEAT[key] = {"row": {}, "Q": [], "kmax": [], "n_pos": []}
    else:
        _FEAT[key] = _FEAT.pop(key)                                     # most recently used last
    miss = [t for t in dict.fromkeys(tids) if t not in ent["row"]]
    if miss:
        nb = L.kc.missing(miss)
        if nb:
            raise KeyError(f"{len(nb)} prompt texts are not in the key cache (parts {L.kc.parts}); e.g. text_id {nb[:5]}: build the part first")
        Sd = S if str(S.device) == str(torch.device(device)) else S.to(device)
        a = 0
        while a < len(miss):
            b, npos = a, 0
            while b < len(miss) and (b == a or npos + L.kc.n_pos(miss[b]) <= CHUNK_POS):
                npos += L.kc.n_pos(miss[b])
                b += 1
            K, off = L.kc.stack(miss[a:b])
            Kd = K.to(device)
            R = Sd.scores(Kd)                                           # (n_pos, D) fp64: the library's feature
            kn = Kd.double().norm(dim=1)
            Q = torch.stack([R[off[i]:off[i + 1]].max(0).values for i in range(b - a)]).cpu()
            km = torch.stack([kn[off[i]:off[i + 1]].max() for i in range(b - a)]).cpu()
            base = sum(q.shape[0] for q in ent["Q"])
            for i, t in enumerate(miss[a:b]):
                ent["row"][t] = base + i
            ent["Q"].append(Q)
            ent["kmax"].append(km)
            ent["n_pos"].append((off[1:] - off[:-1]).clone())
            del R, Kd, kn
            a = b
        ent["Q"], ent["kmax"], ent["n_pos"] = [torch.cat(ent["Q"])], [torch.cat(ent["kmax"])], [torch.cat(ent["n_pos"])]
    idx = torch.tensor([ent["row"][t] for t in tids], dtype=torch.long)
    if not len(tids):
        return {"Q": torch.zeros(0, S.D, dtype=torch.float64), "kmax": torch.zeros(0, dtype=torch.float64), "n_pos": torch.zeros(0, dtype=torch.long)}
    return {"Q": ent["Q"][0][idx], "kmax": ent["kmax"][0][idx], "n_pos": ent["n_pos"][0][idx]}


def band(S, Q, kmax, slack: float = e10t.BAND):
    """(n, D) half-width of the decision band around every threshold (as rescope.keycache.decision_band, per query: with the
    largest key norm of the query)."""
    wn = S._cache.get("wn")
    if wn is None:
        wn = S._cache.setdefault("wn", S.det_w.norm(dim=1) / S.det_s)
    wn = wn.cpu()
    scale = kmax[:, None] * wn[None, :]
    cos, ball = (S.det_kind == COS).cpu(), (S.det_kind == BALL).cpu()
    if bool(cos.any()):
        scale[:, cos] = wn[cos][None, :].expand(scale.shape[0], -1)
    if bool(ball.any()):
        scale[:, ball] = Q[:, ball].abs()
    th = S.theta().cpu()
    return slack * (torch.where(torch.isfinite(th), th.abs(), torch.zeros_like(th))[None, :] + scale)


def _scores_sub(S, k, J):
    """Features of the detectors J on keys k (n, d): the formulas of rescope ScopedLog._scores (fp64), restricted to J."""
    kd = k.double()
    r = (kd @ S.det_w[J].T) / S.det_s[J]
    kind = S.det_kind[J]
    cos, ball = kind == COS, kind == BALL
    if bool(cos.any()):
        r[:, cos] = r[:, cos] / kd.norm(dim=1, keepdim=True).clamp_min(1e-300)
    if bool(ball.any()):
        r[:, ball] = -torch.cdist(kd, S.det_key[J][ball], compute_mode="donot_use_mm_for_euclid_dist")
    return r


def positions(L, S, tid: int, J):
    """Per-position decision of ONE fired query whose candidate detectors are J (long tensor): -> list of (position, releasing
    edit, number of candidate edits at the position), fire 'top1' ranking of the library; the position of each candidate
    detector's maximum is a candidate by definition (agreement with the query-level decision)."""
    k = torch.from_numpy(np.array(L.kc.keys(tid)))
    r = _scores_sub(S, k, J)
    th = S.theta()[J]
    cand = r >= th[None, :]
    cand[r.argmax(0), torch.arange(J.numel())] = True
    rk = r
    if S.det_b is not None:
        rk = rk - S.det_b[J][None, :]
    if S.det_u is not None:
        rk = rk / S.det_u[J][None, :]
    ed = S.det_edit[J]
    out = []
    for p in cand.any(1).nonzero().flatten().tolist():
        c = cand[p]
        top = int(torch.where(c, rk[p], torch.full_like(rk[p], -INF)).argmax())      # the first maximum = the lowest detector index
        out.append((p, int(ed[top]), int(torch.unique(ed[c]).numel())))
    return out


def decide(L, S, tids, own=None, device: str = "cpu", slack: float = e10t.BAND, with_positions: bool = True) -> dict:
    """The gate's decision on the cached queries `tids` (see the module docstring).  own: per query its own edit or -1 / None."""
    tids = [int(t) for t in tids]
    n = len(tids)
    Sc = S if S.device.type == "cpu" else S.to("cpu")
    F = features(L, Sc, tids, device)
    Q, kmax = F["Q"], F["kmax"]
    th = Sc.theta()
    cand = Q >= th[None, :]
    fired = cand.any(1)
    near = (((Q - th[None, :]).abs() <= band(Sc, Q, kmax, slack)) | torch.isnan(Q)).any(1)
    ce = Sc._edit_any(cand)                                             # (n, T) candidate edits
    own_t = torch.full((n,), -1, dtype=torch.long) if own is None else torch.tensor([-1 if o is None else int(o) for o in own], dtype=torch.long)
    has = own_t >= 0
    own_cand = torch.zeros(n, dtype=torch.bool)
    if bool(has.any()):
        own_cand[has] = ce[has.nonzero().flatten(), own_t[has]]
    out = {"text_ids": tids, "fired": fired.numpy(), "near": near.numpy(), "sure_silent": (~fired & ~near).numpy(),
           "n_cand_edits": ce.sum(1).numpy(), "own": own_t.numpy(), "own_cand": own_cand.numpy(),
           "margin": (Q - th[None, :]).max(1).values.numpy() if n else np.zeros(0),
           "n_prompt": np.array([L.kc.n_prompt(t) for t in tids], dtype=np.int64)}
    top1 = Sc.fire == "top1"
    rel = [None] * n
    nfp = np.zeros(n, dtype=np.int64)
    fp, fc, fl = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    own_rel, oth_rel = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    if with_positions:
        for i in fired.nonzero().flatten().tolist():
            ps = positions(L, Sc, tids[i], cand[i].nonzero().flatten())
            nfp[i] = len(ps)
            npr = int(out["n_prompt"][i])
            fp[i], fc[i] = any(p < npr for p, _, _ in ps), any(p >= npr for p, _, _ in ps)
            fl[i] = all(p == npr + e10t.N_DEC - 1 for p, _, _ in ps)       # only at the last continuation token: cannot change the 10 tokens
            o = int(own_t[i])
            if top1:
                rel[i] = [(p, e) for p, e, _ in ps]
                eds = {e for _, e, _ in ps}
            else:
                eds = set(ce[i].nonzero().flatten().tolist())
            own_rel[i] = o >= 0 and o in eds
            oth_rel[i] = any(e != o for e in eds)
    out.update(rel=rel, n_fire_pos=nfp, fired_prompt=fp, fired_cont=fc, fired_last_only=fl, own_released=own_rel, other_released=oth_rel)
    return out


# ----------------------------------------------------------------------------- the standard row sets and their numbers
PARTS = ("request", "paraphrase", "neighborhood", "heldout", "repair")


def part_rows(L, part: str) -> list:
    return L.rows[part] if part in L.rows else L.benign(part)


def evaluate(L, M, parts=PARTS, device: str = "cpu") -> dict:
    """Decisions of a gate method on the standard row sets of a log -> {part: decision (+ 'rows')}.  own edit: the row's `pos`
    for request / paraphrase / neighbourhood prompts (the edit whose record the prompt belongs to), none for benign prompts."""
    S = M.obj if hasattr(M, "obj") else M
    out = {}
    for part in parts:
        rows = part_rows(L, part)
        own = [r["pos"] for r in rows] if part in L.rows else None
        d = decide(L, S, [r["text_id"] for r in rows], own, device)
        d["rows"] = rows
        out[part] = d
    return out


def _pct(x):
    x = np.asarray(x)
    return float(100.0 * x.mean()) if x.size else None


def gate_stats(L, ev: dict) -> dict:
    """The gate-level numbers of metric (iv) from evaluate(): false firing on benign prompts, coverage of paraphrases, firing
    on neighbourhood prompts (all in percent), lineage width, counts inside the decision band."""
    out = {}
    for part, d in ev.items():
        rows = d["rows"]
        clean = np.array([bool(r.get("clean", True)) for r in rows])
        o = {"n": len(rows), "fired": _pct(d["fired"]), "near_band": int(d["near"].sum()),
             "fired_prompt": _pct(d["fired_prompt"]), "fired_cont_only": _pct(d["fired"] & ~d["fired_prompt"]),
             "fired_last_only": _pct(d["fired_last_only"]),
             "mean_fire_positions_if_fired": float(d["n_fire_pos"][d["fired"]].mean()) if d["fired"].any() else None,
             "lineage_width_mean_if_fired": float(d["n_cand_edits"][d["fired"]].mean()) if d["fired"].any() else None,
             "lineage_width_max": int(d["n_cand_edits"].max()) if len(rows) else None}
        if part in ("heldout", "repair", "neighborhood"):
            o.update(n_clean=int(clean.sum()), fired_clean=_pct(d["fired"][clean]), n_near_scope=int((~clean).sum()),
                     fired_near_scope=_pct(d["fired"][~clean]))
        if part in ("request", "paraphrase", "neighborhood"):
            o.update(own_cand=_pct(d["own_cand"]), own_released=_pct(d["own_released"]), other_released=_pct(d["other_released"]),
                     only_other_released=_pct(d["other_released"] & ~d["own_released"]))
        out[part] = o
    return out
