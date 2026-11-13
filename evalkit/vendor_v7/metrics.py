"""VLKEB-standard metrics on top of :mod:`v7.vlm` ([redacted: related project] §2.3).

Metric set (all on the *current* weights, teacher forced, greedy over the answer's own
tokenisation):

Rel      image+src          -> alt            token accuracy (mean over answer tokens, then
                                               over items); ``Rel_em`` = whole answer correct.
T-Gen    image+rephrase     -> alt
I-Gen    image_rephrase+src -> alt
Port     onehop: image+port_q -> port_a        (items without ``port_q`` are skipped)
T-Loc    text-only ``loc + " " + loc_ans``     agreement of the argmax at the answer positions
                                               before vs after editing ("m", plus exact-match
                                               "T-Loc_em": all positions agree).
I-Loc    image ``m_loc`` + m_loc_q -> m_loc_a  same, "m_loc" in the reference code.

Reference: code/VLKEB/easyeditor/evaluate/evaluate.py:471-500 (compute_multimodal_edit_quality)
and easyeditor/editors/multimodal_editor.py:267-283 (locality = np.mean(post == pre) over the
answer positions).  Two documented deviations: answers are teacher-forced with their own
tokenisation (VLKEB scores prompts_len-window logits, an off-by-one; see the report), and Loc is
reported as the raw agreement m (VLKEB's paper table implies rescaling to (m+1)/2, lower bound 50).

NaN guard: a non-finite logit anywhere in the answer window makes that metric's item count as
incorrect (0.0) and is counted in ``n_nonfinite``; such items should be filtered out with
``data.finite_mask`` before editing runs.
"""
from __future__ import annotations

import time
from typing import Iterable, Sequence

import numpy as np
import torch

from .vlm import QA_TEMPLATE

# (probe, metric) definitions: prompt field, answer field, image field or None, fmt
PROBES = {
    "rel":   ("src", "alt", "image", "vqa"),
    "tgen":  ("rephrase", "alt", "image", "vqa"),
    "igen":  ("src", "alt", "image_rephrase", "vqa"),
    "port":  ("port_q", "port_a", "image", "vqa"),
    "tloc":  ("loc", "loc_ans", None, "plain"),
    "iloc":  ("m_loc_q", "m_loc_a", "m_loc", "vqa"),
    # diagnostic only: same prompt as Rel but scored against the item's own pre-edit answer
    # (VLKEB's ``pred``, with "_" -> " ").  Not a reported metric.
    "rel_pred": ("src", "pred_text", "image", "vqa"),
}
ACC_METRICS = ("rel", "tgen", "igen", "port")     # token accuracy / EM
DIAG_METRICS = ("rel_pred",)                      # not part of the reported metric set
LOC_METRICS = ("tloc", "iloc")                    # argmax agreement before vs after
NO_IMAGE = {"tloc"}
QA_METRICS = ("rel", "tgen", "igen", "port")      # use the QA template; "tloc" is plain


def port_qa(d: dict, idx: int = 0) -> tuple[str, str] | None:
    """(port_q, port_a) for one VLKEB item, or None if this item has no usable one-hop entry.

    ``data.load_vlkeb`` already hoists ``port_new[idx]`` onto the item as top-level ``port_q`` /
    ``port_a``; the nested fallback keeps :func:`build_probes` correct for raw VLKEB records
    straight out of ``eval_edit_onehop.json`` (where the fields only ever exist nested) and for
    any item built by hand in a test.  ``port_new`` is a *list* -- one ``1-hop`` entry per record
    in every split we ship, but the reference code takes index 0 and asserts length 1, so we do
    the same and ignore the rest rather than silently averaging several questions together.
    """
    if "port_q" in d and "port_a" in d:
        return d["port_q"], d["port_a"]
    pn = d.get("port_new")
    if not isinstance(pn, (list, tuple)) or idx >= len(pn):
        return None
    entry = pn[idx]
    if not isinstance(entry, dict):
        return None
    if "port_q" not in entry or "port_a" not in entry:
        return None
    return entry["port_q"], entry["port_a"]


def build_probes(items: Sequence[dict], metric: str) -> list[dict]:
    """Per-item probe descriptors for one metric, skipping items without the required fields.

    Keys: item (index into ``items``), uid, metric, prompt, answer, image (None = text-only),
    fmt, plus ``port_idx`` for Port (which one-hop entry to use).

    Port is the one special case: its prompt/answer live under ``port_new[idx]`` in the raw
    VLKEB records (only ``data.load_vlkeb`` hoists them to the top level), so it is resolved
    through :func:`port_qa` instead of the flat ``pf in d`` test the other metrics use.
    """
    if metric not in PROBES:
        raise ValueError(f"unknown metric {metric!r}; choose from {list(PROBES)}")
    pf, af, imf, fmt = PROBES[metric]
    out = []
    for i, d in enumerate(items):
        if metric == "port":
            qa = port_qa(d)
            if qa is None:
                continue
            prompt, answer = qa
        else:
            if pf not in d or af not in d:
                continue
            prompt, answer = d[pf], d[af]
        img = None if imf is None else d[imf]
        rec = {"item": i, "uid": d.get("uid", str(i)), "metric": metric, "prompt": prompt,
               "answer": answer, "image": img, "fmt": fmt}
        if metric == "port":
            rec["port_idx"] = 0
        out.append(rec)
    return out


def merge_probes(items: Sequence[dict], metrics: Iterable[str]) -> tuple[list[dict], dict[str, list[int]]]:
    """Merge the probes of several metrics into one list (images last) -> (probes, index map)."""
    probes, index = [], {}
    for m in metrics:
        got = build_probes(items, m)
        index[m] = list(range(len(probes), len(probes) + len(got)))
        probes.extend(got)
    order = sorted(range(len(probes)), key=lambda i: (probes[i]["image"] is not None, i))
    remap = {old: new for new, old in enumerate(order)}
    probes = [probes[i] for i in order]
    index = {m: sorted(remap[i] for i in idx) for m, idx in index.items()}
    return probes, index


def run_probes(vlm, probes: Sequence[dict], max_seq_len: int | None = None,
               batch_size: int | None = None, verbose: bool = False,
               keep_logits: bool = False) -> list[dict]:
    """Teacher-forced forward for each probe.

    Returns one dict per probe: pred (answer_len,), target (answer_len,), acc, em, finite,
    top1_p (probability of the target token), seq_len (padded length of its batch).
    With ``keep_logits=True`` each dict also carries ``logits`` (A, V) at full precision.

    ``keep_logits`` defaults to False because those (A, V) tensors are the whole memory story
    on a large-vocabulary backbone: one probe's answer window at V=151,936 is ~0.1 GB in fp32
    for a 100-token free-form answer, and the list holds every probe until it returns.  On
    VLKEB (2-token answers, so ~4 MB per probe) either setting is fine; on MMKE-Bench's
    41-72-word answers a 300-item call exceeds a 32 GB card.  Nothing on the metric path
    (acc/em/Loc agreement) reads ``logits`` -- it is computed from ``pred``/``target`` -- so
    dropping it changes no number.
    """
    if max_seq_len is None:
        max_seq_len = 1024 if vlm.backbone != "llava" else 640
    results: list[dict | None] = [None] * len(probes)
    t0 = time.time()
    for with_image in (True, False):
        ids = [i for i, p in enumerate(probes) if (p["image"] is not None) == with_image]
        if not ids:
            continue
        rows = [(probes[i]["image"], probes[i]["prompt"], probes[i]["answer"], probes[i]["fmt"]) for i in ids]
        batches, lens = vlm.plan_batches(rows, max_seq_len, max_rows=batch_size)
        for b in batches:
            gi = [ids[j] for j in b]
            sub = [probes[i] for i in gi]
            batch = vlm.build_batch([p["image"] for p in sub], [p["prompt"] for p in sub],
                                    [p["answer"] for p in sub], fmt=sub[0]["fmt"])
            with torch.no_grad():
                out = vlm.forward(batch, logits_to_keep="answer")
            logits = out.logits  # (B, A, V)
            am, ai, al = batch["answer_mask"], batch["answer_ids"], batch["answer_len"]
            finite = (torch.isfinite(logits).all(-1) | ~am).all(1)   # padding columns masked out
            flat = logits.flatten(0, 1)
            keep = am.reshape(-1, 1)
            safe = torch.where(keep, torch.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0),
                               torch.zeros_like(flat))
            pred = safe.argmax(-1).view(logits.shape[0], logits.shape[1])
            lp = safe.log_softmax(-1).gather(-1, ai.clamp(min=0).reshape(-1, 1)).reshape(logits.shape[:2])
            for k, i in enumerate(gi):
                n = int(al[k])
                tgt = ai[k, :n]
                pr = pred[k, :n]
                ok = (pr == tgt)
                results[i] = {
                    "item": probes[i]["item"], "uid": probes[i]["uid"], "metric": probes[i]["metric"],
                    "pred": pr.cpu().numpy(), "target": tgt.cpu().numpy(),
                    "acc": float(ok.float().mean()) if n else 0.0, "em": float(ok.all()),
                    "finite": bool(finite[k]), "logp": float(lp[k, :n].float().mean()),
                    "seq_len": int(batch["seq_len"][k]),
                    "logits": logits[k, :n].float() if keep_logits else None,
                }
            if verbose:
                print(f"  [probes {'img' if with_image else 'txt'}] {len(batches)} batches, "
                      f"max len {max(lens[j] for j in b)}", flush=True)
    return results



def _acc(res: Sequence[dict], metric: str) -> dict:
    sel = [r for r in res if r["metric"] == metric]
    if not sel:
        return {"n": 0, "mean": None, "em": None, "n_nonfinite": 0, "per_item": []}
    acc = np.array([r["acc"] if r["finite"] else 0.0 for r in sel])
    return {"n": len(sel), "mean": float(acc.mean() * 100), "per_item": acc.tolist(),
            "em": float(np.mean([r["em"] if r["finite"] else 0.0 for r in sel]) * 100),
            "n_nonfinite": int(sum(not r["finite"] for r in sel))}


def _loc(res: Sequence[dict], metric: str, pre: dict | None) -> dict:
    sel = [r for r in res if r["metric"] == metric]
    if not sel or pre is None:
        return {"n": len(sel), "m": None}
    ms, ems, nonfin = [], [], 0
    for r in sel:
        p = pre.get(r["uid"])
        if p is None or not r["finite"]:
            nonfin += int(not r["finite"])
            ms.append(0.0)
            ems.append(0.0)
            continue
        n = min(len(p), len(r["pred"]))
        eq = (p[:n] == r["pred"][:n])
        ms.append(float(eq.mean()) if n else 0.0)
        ems.append(float(eq.all()) if n else 0.0)
    m = float(np.mean(ms) * 100) if ms else None
    return {"n": len(sel), "m": m, "m_rescaled": (m + 100.0) / 2 if m is not None else None,
            "em": float(np.mean(ems) * 100) if ems else None, "per_item": ms, "n_nonfinite": nonfin}


# [not released: code that no released code calls; it served the related project (README, "What can be reproduced")]
