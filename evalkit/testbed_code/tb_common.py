"""Shared helpers for the 06_testbed_v3 edit-log build ([redacted: describes a related submission under double-blind review] [redacted: describes a related submission under double-blind review] configuration).

Everything model-side goes through vendor_v7 (the read-only copy of [redacted: describes a related submission under double-blind review] code/v7);
nothing here changes the editor arithmetic.  Writes only under OUT_ROOT.
"""
from __future__ import annotations
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP

import json
import os
import re
import sys
import time
import unicodedata

CODE = f"{_RP.ROOT}/02_research/code"
if CODE not in sys.path:
    sys.path.insert(0, CODE)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

OUT_ROOT = f"{_RP.ROOT}/02_research/experiments/06_testbed_v3"
ZCACHE_DIR = os.path.join(OUT_ROOT, "_shared", "zcache")
FINITE_DIR = os.path.join(OUT_ROOT, "_shared", "finite_cache")
# [not released: location of the editor's statistics; the editor that wrote the vision-language logs; it is described in a related submission under double-blind review (README, section "What can be reproduced")]
EXP = f"{_RP.ROOT}/02_research/experiments"
DEVICE = "cuda:0"

# re-point the vendor module-level cache dirs to our own tree BEFORE anything can write
import vendor_v7.data as _D
_D.FINITE_CACHE_DIR = FINITE_DIR

from vendor_v7.vlm import VLM, load_image          # noqa: E402
from vendor_v7.data import load_vlkeb               # noqa: E402
from vendor_v7 import metrics as M   # noqa: E402


class _NotReleased:
    """Placeholder for a module of the editor that wrote the vision-language logs (its filters, its target optimisation and the cache of
    the optimised targets).  The editor is described in a related submission of the authors that is under review and is not part of
    this repository; code that needs it raises here."""

    def __init__(self, what):
        object.__setattr__(self, "_what", what)

    def __getattr__(self, name):
        raise NotImplementedError(f"{self._what}.{name}: the editor that wrote the vision-language logs and its cached targets are not released "
                                  "(README, section \"What can be reproduced\")")


F, Z, E = _NotReleased("filters"), _NotReleased("target optimisation"), _NotReleased("editor")
# [not released: the imports of the editor, its filters and its target optimisation stood here; the editor that wrote the vision-language
# logs is described in a related submission under double-blind review (README, section "What can be reproduced")]

# --------------------------------------------------------------------------- log configs
LOGS = {
    "TB-Q100-ae": dict(backbone="qwen25vl", layer=20, source="see README (related submission)", gamma="see README (related submission)", l2=10.0,
                       batch=10, stream=f"{EXP}/01_v1_attribution/w2/edit_batch.jsonl"),
    "[redacted: describes a related submission under double-blind review]": dict(backbone="qwen25vl", layer=20, source="see README (related submission)", gamma="see README (related submission)",
                         l2=10.0, batch=1,
                         stream=f"{EXP}/01_v1_attribution/w2/edit_batch.jsonl"),
    "TB-Q500-ae": dict(backbone="qwen25vl", layer=20, source="see README (related submission)", gamma="see README (related submission)", l2=10.0,
                       batch=10, stream=f"{EXP}/01_v1_attribution/w3b/edit_batch_500.jsonl"),
    "TB-L100-ae": dict(backbone="llava", layer=16, source="see README (related submission)", gamma="see README (related submission)", l2=10.0,
                       batch=10, stream=f"{EXP}/02_complaint_debugging/p5_probe/edit_batch.jsonl"),
}
# working-point grid (Track A): identical items / targets / rule / [redacted: describes a related submission under double-blind review] / B / layer as the l2=10
# parent log, only the AlphaEdit ridge l2 changes
for _par, _l2s in (("TB-Q100-ae", (1.0, 3.0)), ("TB-L100-ae", (1.0, 3.0))):
    for _l2 in _l2s:
        LOGS[f"{_par}-l2_{_l2:g}"] = dict(LOGS[_par], l2=_l2, parent=_par)
# text-probe prompt format: Qwen 'plain' (= chat template, P8); LLaVA 'vqa' (P5: 'plain' gives '?')
TEXT_FMT = {"qwen25vl": "plain", "llava": "vqa"}
N_PROBES, SEED_SEL, MAX_NEW = 500, 42, 10
MAX_NEW_LONG = 48


def log(msg, fh=None):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fh is not None:
        fh.write(line + "\n")
        fh.flush()


def jdump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def jsonl_write(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")


def jsonl_read(path):
    return [json.loads(l) for l in open(path) if l.strip()]


# --------------------------------------------------------------------------- data
def load_stream(path):
    """Edit stream (in edit order) as full VLKEB train items; verifies every field matches."""
    recs = jsonl_read(_RP.expand(path))
    tr = {d["uid"]: d for d in load_vlkeb("train")}
    items = []
    for r in recs:
        d = tr[r["uid"]]
        assert d["src"] == r["subject"] and d["alt"] == r["alt"] and d["image"] == r["image"], r
        items.append(d)
    assert [r["edit_id"] for r in recs] == list(range(len(recs)))
    return items, recs


def select_probes(edit_items, key_exclude=False):
    """P8 protocol: VLKEB eval minus the edit batch, items with m_loc and loc, seed 42, 500.

    key_exclude additionally drops eval items whose (image, src, alt) equals an edit item's
    (P5's LLaVA rule).  Returns (probes, info)."""
    ev = load_vlkeb("eval")
    uids = {d["uid"] for d in edit_items}
    keys = {(d["image"], d["src"], d["alt"]) for d in edit_items}
    elig = [d for d in ev if d["uid"] not in uids and d.get("m_loc") and d.get("loc")
            and (not key_exclude or (d["image"], d["src"], d["alt"]) not in keys)]
    elig_uidonly = [d for d in ev if d["uid"] not in uids and d.get("m_loc") and d.get("loc")]
    rng = np.random.default_rng(SEED_SEL)
    idx = sorted(rng.choice(len(elig), N_PROBES, replace=False))
    probes = [elig[i] for i in idx]
    return probes, {"n_eval": len(ev), "n_eligible": len(elig),
                    "n_eligible_uid_only": len(elig_uidonly), "key_exclude": key_exclude,
                    "probe_uids_sha": _sha([p["uid"] for p in probes])}


def _sha(xs):
    import hashlib
    return hashlib.sha1("|".join(xs).encode()).hexdigest()[:12]


# --------------------------------------------------------------------------- probes / decode
class Prober:
    """P8 probe helpers (greedy_multi / tf_state), with the per-probe image preprocessing
    computed once and passed to build_batch (identical tensors, fewer CPU decodes)."""

    def __init__(self, vlm):
        self.vlm = vlm
        self.text_fmt = TEXT_FMT[vlm.backbone]
        self._vis = {}

    def _vision(self, item):
        k = item["m_loc"]
        if k not in self._vis:
            if len(self._vis) > 64:
                self._vis.clear()
            self._vis[k] = self.vlm.process_images([load_image(k)])
        return self._vis[k]

    def batch(self, item, ptype, answer=None):
        v = self.vlm
        if ptype == "image":
            return v.build_batch([load_image(item["m_loc"])], [item["m_loc_q"]],
                                 answers=[answer] if answer else None, fmt="vqa",
                                 vision=self._vision(item))
        return v.build_batch([None], [item["loc"]], answers=[answer] if answer else None,
                             fmt=self.text_fmt)

    @torch.no_grad()
    def greedy(self, item, ptype, max_new=MAX_NEW):
        """P8 greedy_multi: re-tokenise the growing answer each step (the scored ids)."""
        v = self.vlm
        text = ""
        for _ in range(max_new):
            b = self.batch(item, ptype, answer=text if text else None)
            if not text:
                pos = int(b["last_prompt_idx"][0])
            else:
                pos = int(b["answer_start"][0]) + int(b["answer_len"][0]) - 1
            row = dict(b)
            row["last_prompt_idx"] = torch.tensor([pos], device=v.device, dtype=torch.long)
            o = v.forward(row, logits_to_keep="last")
            tk = int(o.logits[0, 0].argmax(-1))
            if tk == v.tok.eos_token_id:
                break
            cand = v.tok.decode(v._tok(text) + [tk], skip_special_tokens=True).strip()
            if not cand or cand == text:
                break
            text = cand
        return text

    def _eos_ids(self):
        v = self.vlm
        eos = v.m.generation_config.eos_token_id
        return sorted(set(eos if isinstance(eos, (list, tuple)) else [eos]) | {v.tok.eos_token_id})

    @staticmethod
    def _extend(inputs, ids):
        """model inputs of a B=1 prompt batch with token ids appended (no padding: B=1)."""
        if not ids:
            return dict(inputs)
        x = dict(inputs)
        dev = inputs["input_ids"].device
        t = torch.tensor([ids], dtype=torch.long, device=dev)
        x["input_ids"] = torch.cat([inputs["input_ids"], t], 1)
        x["attention_mask"] = torch.cat([inputs["attention_mask"], torch.ones_like(t)], 1)
        if "mm_token_type_ids" in inputs:
            x["mm_token_type_ids"] = torch.cat(
                [inputs["mm_token_type_ids"], torch.zeros_like(t).to(inputs["mm_token_type_ids"].dtype)], 1)
        return x

    @torch.no_grad()
    def greedy_ids_nocache(self, item, ptype, max_new=MAX_NEW):
        """Greedy over token ids with full (no-KV-cache) forwards, i.e. the same compute path as
        the teacher-forced D1 check; the generated ids are fed back as ids (never re-tokenised).
        Stops at EOS only.  Returns (ids, stop, prompt_batch)."""
        v = self.vlm
        eos = self._eos_ids()
        b = self.batch(item, ptype)
        gen, stop = [], "max_new"
        for _ in range(max_new):
            o = v.m(**self._extend(b["inputs"], gen), use_cache=False, logits_to_keep=1)
            tk = int(o.logits[0, -1].argmax(-1))
            if tk in eos:
                stop = "eos"
                break
            gen.append(tk)
        return gen, stop, b

    @torch.no_grad()
    def continue_plain(self, b, prefix_ids, n_more):
        """Plain greedy continuation (HF generate, KV cache, B=1) of prompt+prefix_ids with a
        FRESH GenerationConfig: nothing from the checkpoint's generation_config.json leaks in
        (Qwen2.5-VL ships repetition_penalty=1.05, which the old d2_long.gen_long inherited)."""
        from transformers import GenerationConfig
        v = self.vlm
        eos = self._eos_ids()
        gc = GenerationConfig(max_new_tokens=n_more, do_sample=False, num_beams=1,
                              repetition_penalty=1.0, eos_token_id=eos,
                              pad_token_id=v.tok.pad_token_id if v.tok.pad_token_id is not None
                              else eos[0])
        x = self._extend(b["inputs"], prefix_ids)
        out = v.m.generate(**x, generation_config=gc)
        new = out[0, x["input_ids"].shape[1]:].tolist()
        gen, stop = [], "max_new"
        for t in new:
            if t in eos:
                stop = "eos"
                break
            gen.append(t)
        return gen, stop

    @torch.no_grad()
    def decode_v2(self, item, ptype, max_new=MAX_NEW, max_new_long=MAX_NEW_LONG):
        """Decoder v2 (fixes the P8 y0 decoder, VERIFY.md s4 + s3).

        y0 (D1): first max_new greedy token ids from full no-cache forwards (TF compute path),
                 decoded, cut at the first newline, stripped.  Stops only at EOS / newline /
                 max_new -- whitespace-only tokens (the bare space Qwen and LLaMA emit before a
                 digit) no longer stop decoding, and ids are never re-tokenised mid-decode.
        long (D2): the same max_new ids + plain greedy KV-cache continuation to max_new_long
                 tokens (EOS only), so y0's ids are a prefix of the long answer's ids by
                 construction.  Returns (y0, long_text, info)."""
        tok = self.vlm.tok
        g10, stop10, b = self.greedy_ids_nocache(item, ptype, max_new)
        if stop10 == "eos" or max_new_long <= len(g10):
            more, lstop = [], stop10
        else:
            more, lstop = self.continue_plain(b, g10, max_new_long - len(g10))
        raw = tok.decode(g10, skip_special_tokens=True)
        body, stop = raw.lstrip(), stop10
        if "\n" in body:
            body, stop = body.split("\n", 1)[0], "newline"
        long_ids = g10 + more
        return body.strip(), tok.decode(long_ids, skip_special_tokens=True).strip(), {
            "gen_ids": g10, "raw": raw, "stop": stop, "long_ids": long_ids, "long_stop": lstop}

    @torch.no_grad()
    def tf_state(self, item, ptype, y0):
        """(reproduced, n_answer_tokens, first_mismatch) of y0 under teacher forcing."""
        if not y0:
            return None, 0, None
        b = self.batch(item, ptype, answer=y0)
        o = self.vlm.forward(b, logits_to_keep="answer")
        am, ai = b["answer_mask"][0], b["answer_ids"][0]
        n = int(am.sum())
        if n == 0:
            return None, 0, None
        eq = (o.logits[0].argmax(-1)[:n] == ai[:n])
        if not torch.isfinite(o.logits[0, :n]).all():
            return None, n, None
        mism = None if bool(eq.all()) else int((~eq).nonzero()[0, 0])
        return bool(eq.all()), n, mism


# --------------------------------------------------------------------------- correctness (D2)
_ART = re.compile(r"\b(a|an|the)\b")


def norm_answer(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"[^0-9a-z]+", " ", s)
    s = _ART.sub(" ", s)
    return " ".join(s.split())


_PAREN = re.compile(r"\s*\([^()]*\)\s*$")


def ref_core(ref: str) -> str:
    """Reference with ONE trailing parenthetical removed (Wikipedia disambiguators such as
    'Brazil (1985 film)' -> 'Brazil'); unchanged if nothing would remain."""
    core = _PAREN.sub("", ref or "")
    return core if core.strip() else (ref or "")


def contains_ref(pred: str, ref: str) -> bool:
    """Normalised word-boundary containment of the reference (trailing parenthetical
    disambiguator removed) in the decoded answer."""
    r, p = norm_answer(ref_core(ref)), norm_answer(pred)
    if not r:
        return False
    return f" {r} " in f" {p} "


# --------------------------------------------------------------------------- editing side
# [not released: code of the editor that wrote the vision-language logs; it is described in a related submission under double-blind review (README, section "What can be reproduced")]
# Removed here: the functions that loaded the editor's configuration and ran the editor.
# Everything below and above is the evaluation side: registry of the logs, stream and probe
# loaders, decoder v2, answer normalisation, metric helpers.


# --------------------------------------------------------------------------- metrics
def probe_metric_rows(vlm, items, metric, bs):
    res = M.run_probes(vlm, M.build_probes(items, metric), batch_size=bs)
    return res


def acc_summary(res):
    acc = [r["acc"] if r["finite"] else 0.0 for r in res]
    em = [r["em"] if r["finite"] else 0.0 for r in res]
    return {"n": len(res), "token": float(np.mean(acc) * 100), "exact": float(np.mean(em) * 100),
            "exact_n": int(sum(em)), "nonfinite": int(sum(not r["finite"] for r in res))}


def loc_rows(res, pre):
    """[redacted: describes a related submission under double-blind review] Loc per item: mean argmax agreement over the reference-answer positions."""
    out = []
    for r in res:
        p = pre.get(r["uid"])
        if p is None or not r["finite"]:
            out.append(0.0)
            continue
        n = min(len(p), len(r["pred"]))
        out.append(float((p[:n] == r["pred"][:n]).mean()) if n else 0.0)
    return out
