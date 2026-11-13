"""fx.memo: the persistent memo of behavioural results, safe under concurrent writers.

WHAT IS STORED.  key = (sequence id in the backbone's key cache, mode, signature) -> outcome (a small JSON object).
  mode        'tf'      one teacher-forced forward, a gate decides at the harness index of the last prompt position
              'prompt'  the same sequence, a gate decides at the decode's token (only where lp_shift != 0; else stored as 'tf')
              'decode'  decoder-v2 free generation of the prompt (10 no-cache forwards + KV-cache continuation to 48 tokens)
  signature   what the model's output on that sequence depends on besides the sequence itself:
              'v:<sha1>'              scoped replay, response REPL, at the last prompt position: the fp64 value released
                                      ('decode': one entry per full forward of the decode, joined by ',', '-' = no release)
              'none'                  nothing is released / the pre-edit weight W_0 (decodes only: the teacher-forced pre-edit
                                      outcome of every sequence is in the key cache and is never stored here)
              'W:<sha1>'              a full replacement weight of the edited module (e.g. the deployed weight W_E)
              'RW:<sha1>'             the activation router with deployed weight W:<sha1>, on a sequence it routes ('tf' / 'prompt')
  outcome     the method-independent fields of the library's row (rescope.Evaluator.tf / decode), written by fx.behave only
              after the forward ran through the library's Evaluator.
The model's output is a deterministic function of (backbone, sequence, mode, signature); the directory is per backbone and the
file header records the library version (sha1 of rescope/*.py): shards written by ANOTHER version of the library are ignored and
counted (stats['shards_other_code']), i.e. any edit of the library empties the memo.  If the principal investigator has judged a
library change behaviour-neutral, the environment variable FX_MEMO_ACCEPT=<12-hex version>[,<version>...] admits the shards of the
named versions (Memo.summary()['accepted_library_versions'] records it; new shards always carry the current version).

CONCURRENCY.  Every process appends to its OWN shard file (<root>/<backbone>/shard_<host>_<pid>_<time>.jsonl, one JSON object per
line, flushed line by line); nobody ever rewrites a file.  Readers merge all shards (refresh() reads only the bytes appended since
the last look; an incomplete last line is left for the next look).  Two processes that compute the same key write the same
outcome; a key with two DIFFERENT outcomes is a conflict: it is counted, listed (Memo.conflicts) and the first one read is kept.
"""
from __future__ import annotations

import builtins
import hashlib
import json
import os
import socket
import time

import fx

MODES = ("tf", "prompt", "decode")
_MEMOS = {}


def value_sig(v) -> str:
    """Signature of a released value: 'v:' + sha1 of its fp64 bytes (a (m,) tensor / array)."""
    import numpy as np
    a = np.ascontiguousarray(v.detach().cpu().numpy() if hasattr(v, "detach") else v, dtype=np.float64)
    return "v:" + hashlib.sha1(a.tobytes()).hexdigest()[:20]


def weight_sig(W) -> str:
    """Signature of a full replacement weight: 'W:' + sha1 of its bytes in its own dtype (+ dtype and shape)."""
    import torch
    t = W.detach().contiguous().cpu()
    raw = t.view(torch.uint8).numpy().tobytes() if t.dtype != torch.bfloat16 else t.view(torch.int16).numpy().tobytes()
    h = hashlib.sha1(f"{t.dtype}{tuple(t.shape)}".encode())
    h.update(raw)
    return "W:" + h.hexdigest()[:20]


class Memo:
    def __init__(self, backbone: str, root: str | None = None, layer=None, readonly: bool = False):
        self.backbone, self.root = backbone, os.path.join(fx.MEMO_ROOT if root is None else root, backbone)
        os.makedirs(self.root, exist_ok=True)
        self.code = fx.code_fingerprint()["rescope"]
        self.accept = {self.code} | {c.strip() for c in os.environ.get("FX_MEMO_ACCEPT", "").split(",") if c.strip()}
        self.layer = layer
        self.readonly = readonly
        self.data, self.conflicts = {}, []
        self._off, self._hdr, self._fh, self._path = {}, {}, None, None
        self.stats = {"hits": 0, "misses": 0, "writes": 0, "shards": 0, "shards_other_code": 0, "entries": 0, "conflicts": 0}
        self.refresh()

    # ------------------------------------------------------------------ reading
    def refresh(self) -> int:
        """Read what other processes (and earlier runs) appended since the last look -> number of new entries."""
        new = 0
        for f in sorted(os.listdir(self.root)):
            if not (f.startswith("shard_") and f.endswith(".jsonl")):
                continue
            p = os.path.join(self.root, f)
            if p == self._path:                                         # my own shard: already in memory
                continue
            try:
                size = os.path.getsize(p)
            except OSError:
                continue
            off = self._off.get(p, 0)
            if size <= off:
                continue
            with builtins.open(p, "rb") as fh:
                fh.seek(off)
                buf = fh.read(size - off)
            end = buf.rfind(b"\n")
            if end < 0:
                continue                                                # an incomplete first line: next look
            self._off[p] = off + end + 1
            for line in buf[:end].split(b"\n"):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue                                            # a torn line (never produced by a healthy writer)
                if "header" in rec:
                    self._hdr[p] = rec["header"]
                    self.stats["shards"] += 1
                    ok = rec["header"].get("rescope") in self.accept and (self.layer is None or rec["header"].get("layer") in (None, self.layer))
                    self.stats["shards_other_code"] += int(not ok)
                    continue
                h = self._hdr.get(p)
                if h is None or h.get("rescope") not in self.accept or (self.layer is not None and h.get("layer") not in (None, self.layer)):
                    continue                                            # written by another version of the library / another layer
                new += self._put(rec["k"], rec["v"])
        self.stats["entries"] = len(self.data)
        return new

    def _put(self, k, v) -> int:
        old = self.data.get(k)
        if old is None:
            self.data[k] = v
            return 1
        if old != v:
            self.stats["conflicts"] += 1
            if len(self.conflicts) < 200:
                self.conflicts.append({"key": k, "kept": old, "other": v})
        return 0

    # ------------------------------------------------------------------ access
    @staticmethod
    def key(sid: str, mode: str, sig: str) -> str:
        if mode not in MODES:
            raise ValueError(f"mode: one of {MODES}")
        return f"{sid}|{mode}|{sig}"

    def get(self, sid: str, mode: str, sig: str):
        v = self.data.get(self.key(sid, mode, sig))
        self.stats["hits" if v is not None else "misses"] += 1
        return v

    def has(self, sid: str, mode: str, sig: str) -> bool:
        return self.key(sid, mode, sig) in self.data

    def put(self, sid: str, mode: str, sig: str, outcome: dict, by=None):
        """Store an outcome (appended to this process's shard at once).  A key that is already known with the same outcome is
        not written again; with another outcome it is a conflict (counted; the known outcome is kept)."""
        k = self.key(sid, mode, sig)
        v = json.loads(json.dumps(outcome, allow_nan=False))            # exactly what a reader will see
        if not self._put(k, v):
            return
        self.stats["entries"] = len(self.data)
        if self.readonly:
            return
        if self._fh is None:
            self._path = os.path.join(self.root, f"shard_{socket.gethostname()}_{os.getpid()}_{int(time.time() * 1000)}.jsonl")
            self._fh = builtins.open(self._path, "ab", buffering=0)
            self._fh.write((json.dumps({"header": {"backbone": self.backbone, "layer": self.layer, "rescope": self.code, "fx": fx.VERSION,
                                                   "pid": os.getpid(), "started": time.strftime("%Y-%m-%d %H:%M:%S")}}) + "\n").encode())
        rec = {"k": k, "v": v}
        if by:
            rec["by"] = by
        self._fh.write((json.dumps(rec, allow_nan=False) + "\n").encode())    # one unbuffered write per line
        self.stats["writes"] += 1

    def summary(self) -> dict:
        by = {}
        for k in self.data:
            m = k.split("|")[-2]
            by[m] = by.get(m, 0) + 1
        return dict(self.stats, root=self.root, by_mode=by, code=self.code, accepted_library_versions=sorted(self.accept))


def open(backbone: str, root: str | None = None, layer=None, readonly: bool = False) -> Memo:      # noqa: A001
    """The memo of a backbone (one object per process and root)."""
    key = (backbone, fx.MEMO_ROOT if root is None else root, layer, readonly)
    if key not in _MEMOS:
        _MEMOS[key] = Memo(backbone, root, layer, readonly)
    return _MEMOS[key]
