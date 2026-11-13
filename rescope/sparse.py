"""Sparse evaluation: the rows of the Evaluator for scoped gates, running the model only where something fires.

A sequence on which no edit fires is the pre-edit model, bit for bit (scoped.py: the hook returns the module's own output
tensor).  Its teacher-forced row therefore consists of
  * the outcome of the PRE-EDIT model on that sequence (exact, token accuracy, first mismatch): a property of the backbone
    and of the sequence, the same for every gate and every log; and
  * the firing flags of a hook that was called once and did not fire.
SparseEvaluator decides on the CPU, from the key of the sequence at the last prompt position, whether a gate is SURELY
silent; if so it assembles the row from the cached pre-edit outcome, otherwise it runs the ordinary forward through the
hook (which then decides by itself).  "Surely": no coordinate reaches its threshold AND none lies within the decision
band (keycache.decision_band: 1e-9 of the scale of the products), so the round-off between the CPU gate and the GPU hook
cannot change the decision; a sequence inside the band is simply run.  The rows are equal to Evaluator's rows (GPU test b13).

Sources of (key, pre-edit outcome) of an item, in this order:
  1. the per-backbone key cache of the final experiments (keycache.KeyCache: pre_exact / pre_token_acc / ... of index.jsonl;
     the item is found by CONTENT: format, question, image, teacher-forced text), so one cache serves every log;
  2. the local store of this object, filled on first use by ONE truncated forward (the key) and ONE forward at W_0
     (the outcome) per item: later gates and calls reuse it (save_local / load_local keep it on disk).
What is sparse: teacher-forced rows (edits, probes, hard_negatives, tf_rows; lp='tf' and lp='prompt') of scoped logs with
role 'last' and the fp64 gate, whatever their detector kind (ReScope, cosine / exact / ball baselines).  Everything else
(other roles, dense gates, weights, corrections, DUAL, free generation and D2) takes the Evaluator's own path unchanged.
own_keys / eligible_keys are served from the key cache where it holds the item (no forward).
"""
from __future__ import annotations

import os

import numpy as np
import torch

from .evaluate import Evaluator
from .keycache import SLACK, KeyCache, near_threshold
from .scoped import Hook, ScopedLog

PRE = ("exact", "token_acc", "n_tok", "finite", "first_mismatch")


class SparseEvaluator(Evaluator):
    def __init__(self, H, cache="auto", hard_negatives: bool = True, hn_dir: str | None = None, slack: float = SLACK):
        """cache: a KeyCache, None (local store only), or 'auto' = the backbone's cache of the final experiments if it exists.
        slack: relative half-width of the decision band inside which a sequence is run through the hook."""
        super().__init__(H, hard_negatives, hn_dir)
        if isinstance(cache, str):
            if cache != "auto":
                raise ValueError("cache: a KeyCache | None | 'auto'")
            cache = KeyCache(H.backbone) if KeyCache.available(H.backbone) else None
        if cache is not None:                                           # keys of another backbone / layer would be served silently: both have d = 11008
            if cache.meta.get("backbone") is not None and cache.backbone != H.backbone:
                raise ValueError(f"key cache of backbone {cache.backbone!r} ({cache.dir}) for a handle of {H.backbone!r}")
            if cache.layer is not None and int(cache.layer) != int(H.L):
                raise ValueError(f"key cache of layer {cache.layer} ({cache.dir}) for a log edited at layer {H.L}")
            if "tf" in cache._K and cache.d != H.d_in:
                raise ValueError(f"key cache with keys of width {cache.d} for a module of width {H.d_in}")
        self.cache, self.slack = cache, float(slack)
        self._cpu_gates = {}                # id(gate) -> (gate, its CPU copy): the CPU decision of a gate that lives on the GPU
        self._local = {}                    # content key -> {"k": (d,) float32, "kp": (d,) float32 | None, "pre": {...}}
        self.stats = {"rows": 0, "silent_from_cache": 0, "silent_from_local": 0, "run_fired_or_near": 0, "run_no_key": 0, "run_other_method": 0,
                      "local_captures": 0, "forwards": 0}

    # ------------------------------------------------------------------ items -> cached key and pre-edit outcome
    def content(self, it):
        """(format, question, image, teacher-forced text): the identity of the item's teacher-forced sequence."""
        q, ptype, ans = self._query(it)
        img = ptype == "image"
        return ("vqa" if img else self.H.prober.text_fmt, q["m_loc_q"] if img else q["loc"], q.get("m_loc") if img else None, ans)

    def cache_row(self, it):
        """Index position of the item in the key cache (None: not cached / no teacher-forced sequence there)."""
        if self.cache is None:
            return None
        ck = self.content(it)
        return self.cache.find(*ck) if ck[3] else None

    def _capture_local(self, it, ck):
        """One truncated forward (key at the last prompt position) + one forward at W_0 (the pre-edit outcome)."""
        b = self.batch(it)
        x, p = self.capture(it, batch=b)
        sh = self.lp_shift(it)
        with self.applied(self.H.W0):
            r0 = self.tf(it, b)
        self._local[ck] = {"k": x[p].float().cpu(), "kp": x[p - sh].float().cpu() if sh else None, "shift": sh, "pre": {k: r0[k] for k in PRE}}
        self.stats["local_captures"] += 1
        self.stats["forwards"] += 1
        return self._local[ck]

    def _entry(self, it, lp, capture=True):
        """{'key': (d,) float32, 'pre': flags, 'src': 'cache' | 'local'} of the item for the position `lp`, or None."""
        ck = self.content(it)
        if not ck[3]:
            return None
        j = self.cache.find(*ck) if self.cache is not None else None
        if j is not None:
            e = self.cache.index[j]
            if e.get("pre_exact") is not None and e.get("row_tf") is not None and (lp == "tf" or e.get("lp_shift") is not None):
                row = e["row_tf"] if (lp == "tf" or not e["lp_shift"]) else e.get("row_tf_prompt")
                if row is not None:
                    return {"key": self.cache.K("tf")[int(row)], "pre": {k: e["pre_" + k] for k in PRE}, "src": "cache"}
        ent = self._local.get(ck)
        if ent is None and capture:
            ent = self._capture_local(it, ck)
        if ent is None or (lp == "prompt" and ent["shift"] is None):
            return None
        return {"key": ent["k"] if (lp == "tf" or not ent["shift"]) else ent["kp"], "pre": ent["pre"], "src": "local"}

    # ------------------------------------------------------------------ rows
    @staticmethod
    def sparse_ok(method) -> bool:
        return isinstance(method, ScopedLog) and method.role == "last" and method.precision == "fp64"

    @staticmethod
    def _silent_flags(gate, own):
        """Firing flags of a hook that saw ONE forward containing the last prompt position and did not fire (the summary code
        of the real hook, so the fields follow scoped.Hook.summary)."""
        hk = Hook.__new__(Hook)
        hk.gate, hk.record = gate, True
        hk.begin(torch.zeros(1, dtype=torch.long))
        hk.calls = hk.calls_last = 1
        return hk.summary(own=own)

    def _silent_row(self, it, gate, pre, lp):
        """The Evaluator's row of a sequence on which the gate is silent = the pre-edit model's outcome + the flags of a silent
        hook.  Token locality: the logits ARE the pre-edit logits, so every argmax agrees (valid y0: = token accuracy = 1)."""
        loc, v = None, it.get("y0_valid")                               # as Evaluator.tf: None for items that do not declare a validity
        if it["kind"] != "edit" and v is not None:
            loc = float(pre["token_acc"]) if v is True else (1.0 if pre["finite"] else None)
        row = self._tf_row(it, bool(pre["exact"]), float(pre["token_acc"]), int(pre["n_tok"]), bool(pre["finite"]), pre["first_mismatch"], lp, loc)
        row.update(self._silent_flags(gate, it.get("edit")))
        return row

    def _cpu(self, gate: ScopedLog) -> ScopedLog:
        """The gate on the CPU (a copy, kept for the last few gates: T = 2000 detectors and guards are 0.5 GB)."""
        if gate.device.type == "cpu":
            return gate
        hit = self._cpu_gates.get(id(gate))
        if hit is None or hit[0] is not gate:
            if len(self._cpu_gates) >= 4:
                self._cpu_gates.clear()
            hit = self._cpu_gates[id(gate)] = (gate, gate.to("cpu"))
        return hit[1]

    def surely_silent(self, gate: ScopedLog, keys: torch.Tensor) -> torch.Tensor:
        """(n,) bool: on these keys (n, d) no detector is a candidate and none lies in the decision band of its threshold."""
        g = self._cpu(gate)
        th, out = g.theta(), []
        for a in range(0, keys.shape[0], 1024):
            k = keys[a:a + 1024]
            r = g.scores(k)
            out.append(~(r >= th[None, :]).any(1) & ~near_threshold(g, k, r, self.slack))
        return torch.cat(out) if out else torch.zeros(0, dtype=torch.bool)

    def _each(self, method, items, fn, verbose=0, lp="tf"):
        many = isinstance(method, dict)
        raw = dict(method.items() if many else [("_", method)])
        sparse = [k for k, m in raw.items() if self.sparse_ok(m)] if fn == "tf" else []
        if not sparse:
            return super()._each(method, items, fn, verbose, lp)
        if lp not in ("tf", "prompt"):
            raise ValueError("lp: 'tf' | 'prompt'")
        items = list(items)
        with torch.no_grad(), self.hz.HE.vision_cache(self.H):           # 1. key and pre-edit outcome of every item
            ents = []
            for it in items:
                self.trim_vcache()
                ents.append(self._entry(it, lp))
        have = [i for i, e in enumerate(ents) if e is not None]
        silent = {k: np.zeros(len(items), dtype=bool) for k in sparse}
        if have:                                                          # 2. the gates on the CPU
            K = torch.stack([ents[i]["key"] for i in have])
            for k in sparse:
                silent[k][have] = self.surely_silent(raw[k], K).numpy()
        prepared = {}
        out = {k: [None] * len(items) for k in raw}
        with torch.no_grad(), self.hz.HE.vision_cache(self.H):           # 3. rows: cached where surely silent, else the forward
            for n, it in enumerate(items):
                run = []
                for k, m in raw.items():
                    self.stats["rows"] += 1
                    if k in sparse and silent[k][n]:
                        out[k][n] = self._silent_row(it, m, ents[n]["pre"], lp)
                        self.stats["silent_from_" + ents[n]["src"]] += 1
                    else:
                        run.append(k)
                        self.stats["run_other_method" if k not in sparse else ("run_no_key" if ents[n] is None else "run_fired_or_near")] += 1
                if run:
                    self.trim_vcache()
                    b = self.batch(it)
                    pre = self.pre_ids(it, b)
                    for k in run:
                        if k not in prepared:
                            prepared[k] = self._prepare(raw[k])
                        with self.applied(prepared[k]):
                            out[k][n] = self.tf(it, b, lp=lp, pre_ids=pre)
                        self.stats["forwards"] += 1
                if verbose and (n + 1) % verbose == 0:
                    print(f"    sparse {fn} {n + 1}/{len(items)} ({self.stats})", flush=True)
        return out if many else out["_"]

    # ------------------------------------------------------------------ keys from the cache (no forward)
    def _cached(self, it):
        j = self.cache_row(it)
        return None if j is None else self.cache.index[j]

    def eligible_keys(self, it, gate, answer=True, lp: str = "tf"):
        """Evaluator.eligible_keys, served from the key cache for role 'last' where the cache holds the item (the same keys:
        float32 images of the module-dtype values, on the CPU); otherwise the Evaluator's forward."""
        e = self._cached(it) if getattr(gate, "role", None) == "last" else None
        kc = self.cache
        if e is not None and e.get("row_tf") is not None:
            tf = kc.K("tf")
            sh = e.get("lp_shift")
            if answer is True and (lp == "tf" or sh is not None):
                return tf[int(e["row_tf"] if (lp == "tf" or not sh) else e["row_tf_prompt"])][None].clone()
            if answer is False and e.get("row_decode") is not None:
                return kc.K("decode")[int(e["row_decode"])][None].clone()
            if answer == "both" and lp == "tf" and e.get("row_decode") is not None:
                return torch.stack([tf[int(e["row_tf"])], kc.K("decode")[int(e["row_decode"])]])
            if answer in ("decode", "all") and e.get("rows_decode_steps") is not None and kc.has("steps") and sh is not None:
                dk = kc.K("steps")[torch.tensor(e["rows_decode_steps"], dtype=torch.long)]
                if answer == "decode":
                    return dk
                return torch.cat([tf[int(e["row_tf"])][None]] + ([tf[int(e["row_tf_prompt"])][None]] if sh else []) + [dk])
        return super().eligible_keys(it, gate, answer, lp)

    def own_keys(self, metric: str = "rel", edits=None, answer=True):
        """Evaluator.own_keys from the key cache (float32, CPU) if the cache holds EVERY requested edit sequence (and the
        decode keys where they are asked for); otherwise the Evaluator's forwards."""
        if answer not in (True, False, "both", "all"):
            raise ValueError("answer: True | False | 'both' | 'all'")
        its = self.edit_items((metric,), edits)
        es = [self._cached(it) for it in its]
        kc = self.cache
        ok = kc is not None and all(e is not None and e.get("row_tf") is not None and e.get("lp_shift") is not None
                                    and (answer is True or e.get("row_decode") is not None) for e in es)
        if ok and answer == "all":
            ok = kc.has("steps") and all(e.get("rows_decode_steps") is not None for e in es)
        if not ok:
            return super().own_keys(metric, edits, answer)
        ks, ed = [], []
        for it, e in zip(its, es):
            new = []
            if answer is not False:
                new.append(kc.K("tf")[int(e["row_tf"])][None])
            if answer is not True and e["lp_shift"] == 0:
                new.append(kc.K("steps")[torch.tensor(e["rows_decode_steps"], dtype=torch.long)] if answer == "all"
                           else kc.K("decode")[int(e["row_decode"])][None])
            for k in new:
                ks.append(k)
                ed += [it["edit"]] * k.shape[0]
        return torch.cat(ks), ed

    # ------------------------------------------------------------------ checks and persistence
    @torch.no_grad()
    def verify_cache(self, items, n: int = 50, seed: int = 0) -> dict:
        """Re-capture a seeded sample of the items that the key cache holds and compare with it, bit for bit: the teacher-
        forced key (and the key at the decode's token where the positions differ) and the pre-edit outcome."""
        its = [it for it in items if self.cache_row(it) is not None]
        rng = np.random.default_rng(seed)
        pick = [its[int(i)] for i in sorted(rng.choice(len(its), min(n, len(its)), replace=False))] if its else []
        res = {"n_items": len(items), "n_in_cache": len(its), "n_checked": len(pick), "key_bitwise": 0, "key_prompt_checked": 0, "key_prompt_bitwise": 0,
               "pre_equal": 0, "differing": []}
        with self.hz.HE.vision_cache(self.H):
            for it in pick:
                self.trim_vcache()
                e = self.cache.index[self.cache_row(it)]
                b = self.batch(it)
                x, p = self.capture(it, batch=b)
                sh = self.lp_shift(it)
                with self.applied(self.H.W0):
                    r0 = self.tf(it, b)
                kb = bool(torch.equal(x[p].float().cpu(), self.cache.K("tf")[int(e["row_tf"])]))
                pe = all(r0[k] == e["pre_" + k] for k in PRE) and sh == e.get("lp_shift") and p == e.get("lp_tf")
                res["key_bitwise"] += int(kb)
                res["pre_equal"] += int(pe)
                if sh:
                    res["key_prompt_checked"] += 1
                    kpb = bool(torch.equal(x[p - sh].float().cpu(), self.cache.K("tf")[int(e["row_tf_prompt"])]))
                    res["key_prompt_bitwise"] += int(kpb)
                    kb = kb and kpb
                if not (kb and pe):
                    res["differing"].append(it["sid"])
        return res

    def save_local(self, path: str):
        """Write the local store (keys and pre-edit outcomes captured by this object) to `path` (torch file)."""
        tmp = path + f".tmp{os.getpid()}"
        torch.save({"backbone": self.H.backbone, "layer": self.H.L, "entries": self._local}, tmp)
        os.replace(tmp, path)

    def load_local(self, path: str) -> int:
        """Add a saved local store of the SAME backbone and layer; -> number of entries now held."""
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if blob["backbone"] != self.H.backbone or blob["layer"] != self.H.L:
            raise ValueError(f"local store of {blob['backbone']} L{blob['layer']}, this handle is {self.H.backbone} L{self.H.L}")
        self._local.update(blob["entries"])
        return len(self._local)
