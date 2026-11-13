"""evaluate(H, correction, complaints) -> uniform metric record (the same for every method/RQ).

resolution      per complaint, under the correction:
                  D1: the complaint's y0 reproduced under teacher forcing (all tokens argmax-equal,
                      Prober.batch / B=1 full forward: the D1 definition of 06_testbed_v3), token acc,
                      first mismatch; for complaints that are ALSO D2 (pre-edit 48-token answer
                      contains the reference, post-edit not): D2 resolved = decoder-v2 long answer
                      under the correction contains the reference (tb_common.contains_ref).
edit preservation  all T logged edits x {rel (src), tgen (rephrase), igen (image rephrase)},
                  teacher forced, B=1 full forward of each edit's own sequence: exact (all alt
                  tokens) and token accuracy; broken = exact at W_E and not after, fixed = the
                  reverse.  (The vendor batch-8 path that produced the logs' Rel is run once at W_E
                  as a check against the stored Rel; batch padding flips a few near-tie edits.)
collateral      HELD-OUT pool only (never used by any method): valid-y0 probe rows
                  non-complaints lost  (reproduced at W_E -> not reproduced),
                  other complaints fixed (D1 at W_E -> reproduced), per ptype, B=1 teacher forcing;
                  T-Loc / I-Loc of the held-out items vs the W_0 argmax snapshot (vendor Loc).
size            ||correction||_F (adapter: ||L||_F, R orthonormal; weights: ||W - W_E||_F), and
                relative to ||W_E - W_0||_F.
The baseline (W_E, and W_0 for Loc) is computed once per log and cached; at load time it is
checked against the testbed's stored numbers (Rel, per-row reproduced flags).
"""
from __future__ import annotations

import contextlib
import os

import numpy as np
import torch

from .common import CACHE, TB, VM, sha
from .sampler import pool_of, split_tag

EVAL_BS = 8
EDIT_METRICS = ("rel", "tgen", "igen")
_BKEYS = ("last_prompt_idx", "answer_start", "answer_len", "answer_ids", "answer_mask", "pred_pos",
          "labels", "seq_len", "n_image_tokens")


# --------------------------------------------------------------------------- speed-ups (exact)
def _to(b, dev):
    o = dict(b)
    o["inputs"] = {k: v.to(dev) for k, v in b["inputs"].items()}
    for k in _BKEYS:
        o[k] = b[k].to(dev)
    return o


def _fp(pv, thw):
    x = pv.double()
    return (tuple(pv.shape), None if thw is None else tuple(thw.flatten().tolist()), float(x.sum()),
            float(x.abs().sum()), float(x.flatten()[::9973].sum()), float((x * x).sum()))


@contextlib.contextmanager
def vision_cache(H):
    """Memoise the vision tower (+ projector) output per pixel_values fingerprint.  Only layer L
    of the LM is ever changed, so image features are identical in every forward; used ONLY inside
    evaluate() (never inside a timed method).  Exactness: tests/run_tests.py zero."""
    import types
    mdl = H.vlm.m.model
    if "get_image_features" in mdl.__dict__:          # already active (re-entrant use)
        yield
        return
    orig = mdl.get_image_features
    store = H._setup.setdefault("vcache", {})
    stat = H._setup.setdefault("vcache_stat", {"hit": 0, "miss": 0})

    def cached(pixel_values=None, *args, **kw):
        thw = args[0] if args else kw.get("image_grid_thw")
        key = _fp(pixel_values, thw)
        if key in store:
            stat["hit"] += 1
            return types.SimpleNamespace(pooler_output=tuple(t.to(H.dev) for t in store[key]))
        stat["miss"] += 1
        out = orig(pixel_values, *args, **kw)
        po = tuple(t.detach() for t in out.pooler_output)
        store[key] = tuple(t.cpu() for t in po)
        return types.SimpleNamespace(pooler_output=po)
    mdl.get_image_features = cached
    try:
        yield
    finally:
        del mdl.get_image_features


def _plan(H, probes):
    """vendor metrics.run_probes batching (same plan, same batches), built once, CPU."""
    vlm = H.vlm
    msl = 1024 if vlm.backbone != "llava" else 640
    out = []
    for with_image in (True, False):
        ids = [i for i, p in enumerate(probes) if (p["image"] is not None) == with_image]
        if not ids:
            continue
        rows = [(probes[i]["image"], probes[i]["prompt"], probes[i]["answer"], probes[i]["fmt"]) for i in ids]
        batches, _ = vlm.plan_batches(rows, msl, max_rows=EVAL_BS)
        for b in batches:
            gi = [ids[j] for j in b]
            sub = [probes[i] for i in gi]
            bb = vlm.build_batch([p["image"] for p in sub], [p["prompt"] for p in sub],
                                 [p["answer"] for p in sub], fmt=sub[0]["fmt"])
            out.append((gi, _to(bb, "cpu")))
    return out


@torch.no_grad()
def _run_planned(H, n, planned):
    """== vendor metrics.run_probes on the prebuilt batches: per probe (pred, em, acc, finite)."""
    res = [None] * n
    for gi, b in planned:
        b = _to(b, H.dev)
        logits = H.vlm.forward(b, logits_to_keep="answer").logits
        am, ai, al = b["answer_mask"], b["answer_ids"], b["answer_len"]
        finite = (torch.isfinite(logits).all(-1) | ~am).all(1)
        flat = logits.flatten(0, 1)
        keep = am.reshape(-1, 1)
        safe = torch.where(keep, torch.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0), torch.zeros_like(flat))
        pred = safe.argmax(-1).view(logits.shape[0], logits.shape[1])
        for k, i in enumerate(gi):
            m = int(al[k])
            ok = pred[k, :m] == ai[k, :m]
            res[i] = {"pred": pred[k, :m].cpu().numpy(), "acc": float(ok.float().mean()) if m else 0.0,
                      "em": float(ok.all()), "finite": bool(finite[k])}
    return res


# --------------------------------------------------------------------------- fast exact path
FAST_EVAL = True      # B=1 teacher-forced rows via the exact suffix engine (bit-identical logits)


def _mv(o, dev):
    if torch.is_tensor(o):
        return o.to(dev)
    if isinstance(o, tuple):
        return tuple(_mv(x, dev) for x in o)
    if isinstance(o, list):
        return [_mv(x, dev) for x in o]
    if isinstance(o, dict):
        return {k: _mv(v, dev) for k, v in o.items()}
    return o


def _state_cpu(st):
    from .suffix import SuffixState
    return SuffixState(mid=st.mid.cpu(), x=st.x.cpu(), kws=_mv(st.kws, "cpu"), gather=st.gather.cpu(),
                       ans=st.ans.cpu(), A=st.A, T=st.T, logits_ref=None)


def _state_dev(st, dev):
    from .suffix import SuffixState
    return SuffixState(mid=st.mid.to(dev, non_blocking=True), x=st.x.to(dev, non_blocking=True),
                       kws=_mv(st.kws, dev), gather=st.gather.to(dev), ans=st.ans.to(dev), A=st.A, T=st.T,
                       logits_ref=None)


def _suffix_engine(H):
    from .suffix import Suffix
    if "suffix_eval" not in H._setup:
        H._setup["suffix_eval"] = Suffix(H)
    return H._setup["suffix_eval"]


@torch.no_grad()
def _tf_state(H, st, corr):
    S = _suffix_engine(H)
    lg = S.logits(_state_dev(st, H.dev), W=H.mod.weight, corr=corr if (corr is not None and corr.kind == "adapter") else None)
    n = st.A
    if not torch.isfinite(lg).all():
        return None, 0.0, n, None
    eq = lg.argmax(-1) == st.ans.to(H.dev)
    mism = None if bool(eq.all()) else int((~eq).nonzero()[0, 0])
    return bool(eq.all()), float(eq.float().mean()), n, mism


def _states(H, batches):
    S = _suffix_engine(H)          # states (mid, x, block kwargs) do not depend on the edited weight
    with vision_cache(H):
        return [_state_cpu(S.prepare(_to(b, H.dev))) for b in batches]


def evalcache(H):
    ec = H._setup.get("evalcache")
    if ec is None:
        rows = heldout_rows(H)
        hi = heldout_items(H)
        ec = {"rows": rows, "row_batches": [_to(H.complaint_batch(r), "cpu") for r in rows],
              "edits": {m: [_to(H.edit_batch(i, m), "cpu") for i in range(H.T)] for m in EDIT_METRICS},
              "edits_b8": {m: _plan(H, VM.build_probes(H.items, m)) for m in EDIT_METRICS},
              "n_edit": {m: len(VM.build_probes(H.items, m)) for m in EDIT_METRICS},
              "loc_items": hi,
              "loc": {m: _plan(H, VM.build_probes(hi, m)) for m in ("tloc", "iloc")},
              "n_loc": {m: len(VM.build_probes(hi, m)) for m in ("tloc", "iloc")}}
        if FAST_EVAL:
            ec["row_states"] = _states(H, ec["row_batches"])
            ec["edit_states"] = {m: _states(H, ec["edits"][m]) for m in EDIT_METRICS}
        H._setup["evalcache"] = ec
    return ec


def _run_edits(H, corr=None, fast=None):
    """Every logged edit x {rel, tgen, igen}: B=1 teacher-forced full forward of the edit's own
    sequence (no padding; the same definition as the D1 check and the old cb_run evaluation).
    fast: the same logits through the exact suffix engine (corr must be the active correction)."""
    ec = evalcache(H)
    fast = FAST_EVAL if fast is None else fast
    out = {}
    for m in EDIT_METRICS:
        r = ([_tf_state(H, st, corr) for st in ec["edit_states"][m]] if fast
             else [_tf_batch(H, b) for b in ec["edits"][m]])
        out[m] = {"em": np.array([x[0] is True for x in r]), "acc": np.array([x[1] for x in r])}
    return out


def _run_edits_b8(H):
    """vendor metrics.run_probes(batch_size=8) path -- only to check W_E against the stored Rel."""
    ec = evalcache(H)
    out = {}
    for m in EDIT_METRICS:
        res = _run_planned(H, ec["n_edit"][m], ec["edits_b8"][m])
        out[m] = {"em": np.array([bool(r["em"]) and r["finite"] for r in res]),
                  "acc": np.array([r["acc"] if r["finite"] else 0.0 for r in res])}
    return out


@torch.no_grad()
def _tf_batch(H, b):
    b = _to(b, H.dev)
    o = H.vlm.forward(b, logits_to_keep="answer")
    am, ai = b["answer_mask"][0], b["answer_ids"][0]
    n = int(am.sum())
    lg = o.logits[0, :n]
    if not torch.isfinite(lg).all():
        return None, 0.0, n, None
    eq = lg.argmax(-1) == ai[:n]
    mism = None if bool(eq.all()) else int((~eq).nonzero()[0, 0])
    return bool(eq.all()), float(eq.float().mean()), n, mism


def _tf(H, row, answer=None):
    """(reproduced, token_acc, n_tok, first_mismatch) under the current weights / hooks."""
    return _tf_batch(H, H.complaint_batch(row, answer))


def heldout_rows(H):
    pl = pool_of(H)
    return [r for r in H.rows if pl[r["probe_idx"]] == "heldout" and r["y0_valid"]]


def heldout_items(H):
    pl = pool_of(H)
    return [p for i, p in enumerate(H.probes) if pl[i] == "heldout"]


def _loc(H, pre):
    ec = evalcache(H)
    out = {}
    for m in ("tloc", "iloc"):
        res = _run_planned(H, ec["n_loc"][m], ec["loc"][m])
        res = [dict(r, uid=it["uid"]) for r, it in zip(res, VM.build_probes(ec["loc_items"], m))]
        out[m] = np.array(TB.loc_rows(res, pre[m]))
    return out


def _heldout_tf(H, corr=None, fast=None):
    ec = evalcache(H)
    fast = FAST_EVAL if fast is None else fast
    if fast:
        return [_tf_state(H, st, corr)[0] for st in ec["row_states"]]
    return [_tf_batch(H, b)[0] for b in ec["row_batches"]]


def _check_fast(H, B):
    """The fast exact path must reproduce the full-forward W_E baseline flag for flag."""
    H.reset()
    ef = _run_edits(H, fast=True)
    rf = _heldout_tf(H, fast=True)
    ok = bool(all((ef[m]["em"] == B["edits"][m]["em"]).all() and (ef[m]["acc"] == B["edits"][m]["acc"]).all()
                  for m in EDIT_METRICS) and rf == B["reproduced"])
    B["checks"]["fast_eval_equals_full"] = ok
    if not ok:
        raise RuntimeError("fast evaluation path differs from the full forward path")


def baseline(H, force=False):
    """W_E (and W_0 Loc snapshot) reference numbers, cached per log."""
    if "baseline" in H._setup and not force:
        return H._setup["baseline"]
    tkey = [H.name, H.probe_info.get("probe_uids_sha"), EVAL_BS, "v3"]
    if split_tag(H) is not None:            # non-default split (sampler.set_split): own cache file
        tkey.append(["split", *split_tag(H)])
    tag = sha(tkey)
    p = os.path.join(CACHE, f"baseline_{H.name}_{tag}.pt")
    if os.path.exists(p) and not force:
        B = torch.load(p, weights_only=False)
        if FAST_EVAL and "fast_eval_equals_full" not in B["checks"]:
            _check_fast(H, B)
            torch.save(B, p)
        H._setup["baseline"] = B
        return B
    H.reset()
    ec = evalcache(H)
    with vision_cache(H):
        with H.at(H.W0):
            pre = {}
            for m in ("tloc", "iloc"):
                res = _run_planned(H, ec["n_loc"][m], ec["loc"][m])
                pre[m] = {it["uid"]: r["pred"] for r, it in zip(res, VM.build_probes(ec["loc_items"], m))}
            edits_W0 = _run_edits(H)
        edits = _run_edits(H, fast=False)
        edits_b8 = _run_edits_b8(H)
        loc = _loc(H, pre)
        rep = _heldout_tf(H, fast=False)
    rows = ec["rows"]
    B = {"edits": edits, "edits_W0": edits_W0, "pre_loc": pre, "loc": loc,
         "rows": [(r["probe_idx"], r["ptype"]) for r in rows], "reproduced": rep}
    st = H.tl.metrics if hasattr(H, "tl") else None
    chk = {"heldout_reproduced_equals_stored_v2": all(bool(x) == bool(r["reproduced"]) for x, r in zip(rep, rows)),
           "n_heldout_rows": len(rows), "n_heldout_items": len(ec["loc_items"])}
    if st is not None:
        chk["Rel_token_WE"] = float(edits_b8["rel"]["acc"].mean() * 100)
        chk["Rel_exact_WE"] = float(edits_b8["rel"]["em"].mean() * 100)
        chk["Rel_B1_token_WE"] = float(edits["rel"]["acc"].mean() * 100)
        chk["Rel_B1_exact_WE"] = float(edits["rel"]["em"].mean() * 100)
        chk["edits_B1_vs_b8_em_flags_differing"] = {m: int((edits[m]["em"] != edits_b8[m]["em"]).sum()) for m in EDIT_METRICS}
        chk["Rel_matches_stored"] = (abs(chk["Rel_token_WE"] - st["Rel_token"]) < 1e-6
                                     and abs(chk["Rel_exact_WE"] - st["Rel_exact"]) < 1e-6)
    B["checks"] = chk
    if FAST_EVAL:
        _check_fast(H, B)
    torch.save(B, p)
    H._setup["baseline"] = B
    return B


def evaluate(H, corr, complaints, d2=True, edits=True, collateral=True, loc=True, per_item=False):
    """per_item (RQ4 option, default off = unchanged record): also return the per-edit em/acc vectors,
    the per-held-out-row reproduced flags and the per-held-out-item Loc values (for uid bootstraps /
    paired tests)."""
    B = baseline(H)
    evalcache(H)                    # built at W_E, before any correction is applied
    rec = {"method": corr.meta.get("method"), "fro": corr.fro(H),
           "fro_rel_to_log": corr.fro(H) / float(H.D_act.norm())}
    H.reset()
    with H.applied(corr), vision_cache(H):
        res = []
        for c in complaints:
            ok, acc, n, mm = _tf(H, c)
            r = {"probe_idx": c["probe_idx"], "ptype": c["ptype"], "uid": c["uid"],
                 "D1_resolved": bool(ok), "tok_acc": acc, "n_tok": n, "first_mismatch": mm,
                 "is_D2": bool(c.get("is_complaint_D2"))}
            if d2 and c.get("is_complaint_D2"):
                y0, long, _ = H.prober.decode_v2(H.probe_item(c), c["ptype"])
                r["D2_resolved"] = bool(TB.contains_ref(long, c["ref_answer"]))
                r["long_after"] = long
            res.append(r)
        rec["complaints"] = res
        if edits:
            E = _run_edits(H, corr)
            ed = {}
            for m in EDIT_METRICS:
                e0, a0 = B["edits"][m]["em"], B["edits"][m]["acc"]
                e1, a1 = E[m]["em"], E[m]["acc"]
                ed[m] = {"exact": float(e1.mean() * 100), "token": float(a1.mean() * 100),
                         "exact_WE": float(e0.mean() * 100), "token_WE": float(a0.mean() * 100),
                         "n_broken": int((e0 & ~e1).sum()), "n_fixed": int((~e0 & e1).sum()),
                         "broken_ids": np.nonzero(e0 & ~e1)[0].tolist(),
                         "n_tok_decreased": int((a1 < a0 - 1e-9).sum()),
                         "identical_em_acc": bool((e0 == e1).all() and (a0 == a1).all())}
                if per_item:
                    ed[m]["em_list"] = [bool(x) for x in e1]
                    ed[m]["acc_list"] = [float(x) for x in a1]
            rec["edits"] = ed
            rec["edits_broken_total"] = sum(ed[m]["n_broken"] for m in EDIT_METRICS)
            rec["edits_fixed_total"] = sum(ed[m]["n_fixed"] for m in EDIT_METRICS)
        if collateral:
            rows = evalcache(H)["rows"]
            after = _heldout_tf(H, corr)
            col = {}
            for pt in ("image", "text", "all"):
                idx = [k for k, r in enumerate(rows) if pt == "all" or r["ptype"] == pt]
                nonc = [k for k in idx if B["reproduced"][k] is True]
                comp = [k for k in idx if B["reproduced"][k] is False]
                col[pt] = {"n_noncomplaints": len(nonc), "n_complaints": len(comp),
                           "noncomplaints_lost": sum(after[k] is not True for k in nonc),
                           "complaints_fixed": sum(after[k] is True for k in comp)}
            col["changed_rows"] = [rows[k]["group"] for k in range(len(rows))
                                   if bool(after[k]) != bool(B["reproduced"][k])]
            rec["collateral"] = col
            if per_item:
                rec["heldout_rows"] = [{"group": r["group"], "uid": r["uid"], "ptype": r["ptype"],
                                         "reproduced_WE": B["reproduced"][k] is True,
                                         "reproduced_after": after[k] is True} for k, r in enumerate(rows)]
        if loc:
            lc = _loc(H, B["pre_loc"])
            rec["loc"] = {m: {"after": float(lc[m].mean() * 100), "WE": float(B["loc"][m].mean() * 100),
                              "n_items_changed": int((lc[m] != B["loc"][m]).sum())} for m in ("tloc", "iloc")}
            if per_item:
                for m in ("tloc", "iloc"):
                    rec["loc"][m]["uids"] = [p["uid"] for p in VM.build_probes(evalcache(H)["loc_items"], m)]
                    rec["loc"][m]["per_item"] = [float(x) for x in lc[m]]
                    rec["loc"][m]["per_item_WE"] = [float(x) for x in B["loc"][m]]
    H.reset()
    return rec
