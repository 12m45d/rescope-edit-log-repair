"""Cached keys: the firing decision of every cached sequence, for any gate and any log, on the CPU (no model).

The input of the edited module at a token position depends on the backbone and on the token sequence, NOT on the edit
log (one edited module; everything below it is the pre-edit model).  The final experiments therefore capture the keys
once per backbone (02_research/experiments/10_final/cache/<backbone>/, built by 10_final/cache/code; described in the
FORMAT.txt next to the files) and every gate of every log is evaluated on them here.

The cache root is 10_final/cache (environment variable RESCOPE_KEYCACHE_ROOT overrides it, e.g. for a partial cache).
Index and key files are opened together at construction and checked against each other and against meta.json (md5 of the
index, shapes; verify=True: md5 of the key files): a directory that is being rewritten raises instead of serving wrong keys.

FILES READ (one directory per backbone: qwen25vl, llava)
  index.jsonl           one JSON object per cached sequence.  Fields used here (all others are passed through):
                          sid, kind (rel | tgen | igen | probe | hn_i | hn_q | hn_x | benign), family
                          fmt, prompt, image, answer        the model input and the teacher-forced text (edit target, or
                                                            the decoder-v2 pre-edit answer y0); `answer` None: no TF sequence
                          edit, edit_uid, metric            edits (stream position, item uid); hard negatives: their edit(s)
                          probe_idx, uid, ptype, pool, disjoint, strict, query_id      probe rows (harness / splits.annotate)
                          y0_text, y0_valid                 pre-edit answer and its validity (probes, hard negatives, benign subset)
                          pre_exact, pre_token_acc, pre_n_tok, pre_finite, pre_first_mismatch
                                                            Evaluator.tf of the teacher-forced sequence under the PRE-EDIT model
                          lp_tf, lp_decode, lp_shift        last prompt index: teacher-forced / prompt-only sequence; their shift
                          row_tf, row_tf_prompt             rows of keys_tf.pt: the teacher-forced key, and (lp_shift != 0 only) the
                                                            teacher-forced sequence's key at the decode's token
                          row_decode                        row of keys_decode.pt: the prompt-only key (first forward of a decode)
                          rows_decode_steps                 11 rows of keys_decode_steps.pt: prompt + k ids, k = 0..10 (decode_keys)
  keys_tf.pt, keys_decode.pt, keys_decode_steps.pt (optional)    float32 tensors (n, d): exact images of the module-dtype values
  meta.json (optional)

Correspondence with Evaluator.eligible_keys (role 'last'):   which='tf' <-> answer=True;  'prompt' <-> answer=True, lp='prompt';
'decode' <-> answer=False;  'steps' <-> answer='decode';  'all' <-> answer='all';  'both' <-> answer='both'.

PRECISION OF A CPU DECISION.  The gate is the package's fp64 gate (ScopedLog.scores), evaluated here on the CPU in batches.
The hook evaluates the same fp64 products on the GPU, one key at a time: the two agree to round-off (about 1e-13
relative), so a decision can differ only where a coordinate lies within round-off of its threshold.  `decide` flags
those sequences (`near`: within `slack` = 1e-9 relative of a threshold); SparseEvaluator runs them through the hook.
Only gates that decide at the last prompt position are supported: scoped logs with role 'last' and precision 'fp64'
(any detector kind: coordinate, cosine, ball) and the router (needs W_E - W_0).  Roles 'last+answer' / 'all' and the
z-score gate need every position of the sequence, which the cache does not hold.
"""
from __future__ import annotations
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP

import hashlib
import json
import os

import numpy as np
import torch

from .scoped import BALL, COS, INF, ScopedLog

DEFAULT_ROOT = f"{_RP.ROOT}/02_research/experiments/10_final/cache"
CACHE_ROOT = os.environ.get("RESCOPE_KEYCACHE_ROOT", DEFAULT_ROOT)
KINDS = ("rel", "tgen", "igen", "probe", "hn_i", "hn_q", "hn_x", "benign")
WHICH = ("tf", "prompt", "decode", "steps", "both", "all")
SLACK = 1e-9                    # relative width of the band around a threshold inside which a CPU decision is not trusted
_FILES = {"tf": "keys_tf.pt", "decode": "keys_decode.pt", "steps": "keys_decode_steps.pt"}


def decision_band(g: ScopedLog, k: torch.Tensor, r: torch.Tensor, slack: float = SLACK) -> torch.Tensor:
    """(n, D) fp64 half-width of the band around every threshold inside which another device or batch shape may decide
    differently: slack * (|theta_j| + scale_ij), scale = the magnitude of the terms behind the coordinate (|w_j| |k_i| / s_j
    for a linear coordinate, |w_j| / s_j for a cosine, the distance itself for a ball).  The round-off of the fp64 products
    is below 1e-12 of that scale (d * eps), so the default slack 1e-9 is a wide safety factor.  k: the keys (n, d), r: their
    scores (n, D)."""
    wn = g._cache.get("wn")
    if wn is None:
        wn = g._cache.setdefault("wn", g.det_w.norm(dim=1) / g.det_s)
    scale = k.double().norm(dim=1, keepdim=True) * wn[None, :]
    cos, ball = g.det_kind == COS, g.det_kind == BALL
    if bool(cos.any()):
        scale[:, cos] = wn[cos][None, :].expand(scale.shape[0], -1)
    if bool(ball.any()):
        scale[:, ball] = r[:, ball].abs()
    th = g.theta()
    return slack * (torch.where(torch.isfinite(th), th.abs(), torch.zeros_like(th))[None, :] + scale)


def near_threshold(g: ScopedLog, k: torch.Tensor, r: torch.Tensor, slack: float = SLACK) -> torch.Tensor:
    """(n,) bool: some coordinate of the key lies inside the decision band of its threshold (or is NaN)."""
    return (((r - g.theta()[None, :]).abs() <= decision_band(g, k, r, slack)) | torch.isnan(r)).any(1)


def content_key(fmt, prompt, image, answer=None):
    """Identity of a teacher-forced sequence: (format, question text, image path, teacher-forced text).  answer=None: the
    prompt-only sequence."""
    return (fmt, prompt, image, answer)


class KeyCache:
    """Reader of one backbone's key cache.  kc = KeyCache('qwen25vl')  (or KeyCache(directory)).

    kc.index                         list of the index rows (dicts), kc.n
    kc.rows(kind=, family=, T=, pool=, valid=, where=)     positions of index rows (T: edits / hard negatives of a log with T edits)
    kc.keys(which, rows)             float32 keys (n, d) ('steps': (n, 11, d))
    kc.coords(gate, rows, which)     (n, D) fp64 per-query maxima of the gate's feature (calibration: calibrate_two_block(coords=...))
    kc.decide(gate, which, rows, uids=)   firing decision of every sequence: fired, top edit, candidates, margin, near, own flags
    kc.find(fmt, prompt, image, answer)   index position of a sequence by content (None if it is not cached)
    kc.log_rows(uids, kind=...), kc.own_keys(uids)     the sequences of a log; the B = 1 own-prompt keys for repair.add_guards
    """

    def __init__(self, backbone_or_dir: str, root: str | None = None, mmap: bool = True, verify: bool = False):
        """verify=True also checks the md5 sums of the key files against meta.json (reads every file once)."""
        self.dir = backbone_or_dir if os.path.isdir(backbone_or_dir) else os.path.join(CACHE_ROOT if root is None else root, backbone_or_dir)
        p = os.path.join(self.dir, "index.jsonl")
        if not os.path.exists(p):
            raise FileNotFoundError(f"no key cache at {self.dir} (index.jsonl missing)")
        with open(p, "rb") as f:
            raw = f.read()
        self.index = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
        self.n = len(self.index)
        self.meta = {}
        if os.path.exists(os.path.join(self.dir, "meta.json")):
            with open(os.path.join(self.dir, "meta.json")) as f:
                self.meta = json.load(f)
        self.backbone = self.meta.get("backbone", os.path.basename(self.dir.rstrip("/")))
        self.layer = self.meta.get("layer")
        self._mmap, self._K = mmap, {}
        self._load_files(raw, verify)
        self._tf, self._q, self._sid = {}, {}, {}
        for j, e in enumerate(self.index):
            self._sid.setdefault(e["sid"], j)
            if e.get("row_tf") is not None and e.get("answer"):
                self._tf.setdefault(content_key(e.get("fmt"), e.get("prompt"), e.get("image"), e["answer"]), j)
            if e.get("row_decode") is not None:
                self._q.setdefault(content_key(e.get("fmt"), e.get("prompt"), e.get("image")), j)

    @property
    def is_final(self) -> bool:
        """This directory IS the cache of the final experiments (10_final/cache/<backbone>/, also through a symbolic link)."""
        return os.path.realpath(self.dir) == os.path.realpath(os.path.join(DEFAULT_ROOT, str(self.backbone)))

    @staticmethod
    def available(backbone: str, root: str | None = None) -> bool:
        d = os.path.join(CACHE_ROOT if root is None else root, backbone)
        return os.path.exists(os.path.join(d, "index.jsonl")) and os.path.exists(os.path.join(d, "keys_tf.pt"))

    # ------------------------------------------------------------------ files
    def has(self, name: str) -> bool:
        return os.path.exists(os.path.join(self.dir, _FILES[name]))

    def _load_files(self, raw: bytes, verify: bool):
        """Open every key file NOW (index and keys are then one state of the directory) and check that they belong together:
        rows addressed by the index exist; with a meta.json (written last by the builder): md5 of the index, shapes of the key
        files, and (verify) md5 of the key files.  A cache that is being rewritten, or files of two builds, raise here instead
        of serving the wrong keys."""
        def bad(msg):
            return RuntimeError(f"inconsistent key cache at {self.dir}: {msg} (is it being rewritten? files of two builds?)")
        for name, fname in _FILES.items():
            p = os.path.join(self.dir, fname)
            if not os.path.exists(p):
                continue
            try:
                t = torch.load(p, map_location="cpu", mmap=True) if self._mmap else torch.load(p, map_location="cpu")
            except (TypeError, RuntimeError, ValueError):                 # torch without mmap support / a legacy file
                t = torch.load(p, map_location="cpu")
            if isinstance(t, dict):                                       # tolerated: {'keys': tensor}
                t = next(v for k, v in t.items() if torch.is_tensor(v) and v.dim() == 2)
            if t.dtype != torch.float32 or t.dim() != 2:
                raise ValueError(f"{p}: expected a float32 (n, d) tensor, got {t.dtype} {tuple(t.shape)}")
            self._K[name] = t
            m = self.meta.get({"tf": "keys_tf", "decode": "keys_decode", "steps": "keys_decode_steps"}[name])
            if isinstance(m, dict) and m.get("shape") is not None and list(m["shape"]) != list(t.shape):
                raise bad(f"{fname} has shape {tuple(t.shape)}, meta.json says {tuple(m['shape'])}")
            if verify and isinstance(m, dict) and m.get("md5"):
                h = hashlib.md5()
                with open(p, "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 24), b""):
                        h.update(chunk)
                if h.hexdigest() != m["md5"]:
                    raise bad(f"md5 of {fname} differs from meta.json")
        if self.meta.get("index_md5") and hashlib.md5(raw).hexdigest() != self.meta["index_md5"]:
            raise bad("index.jsonl is not the index that meta.json describes")
        need = {"tf": -1, "decode": -1, "steps": -1}
        for e in self.index:
            for f_, name in (("row_tf", "tf"), ("row_tf_prompt", "tf"), ("row_decode", "decode")):
                if e.get(f_) is not None:
                    need[name] = max(need[name], int(e[f_]))
            if e.get("rows_decode_steps"):
                need["steps"] = max(need["steps"], max(int(r) for r in e["rows_decode_steps"]))
        for name, mx in need.items():
            if name in self._K and mx >= self._K[name].shape[0]:
                raise bad(f"the index addresses row {mx} of {_FILES[name]}, which has {self._K[name].shape[0]} rows")
        d = {int(t.shape[1]) for t in self._K.values()}
        if len(d) > 1:
            raise bad(f"key files with different widths {sorted(d)}")

    def K(self, name: str) -> torch.Tensor:
        """The key tensor 'tf' | 'decode' | 'steps' (float32, CPU; memory-mapped where torch allows it; opened at construction)."""
        if name not in self._K:
            raise FileNotFoundError(f"{os.path.join(self.dir, _FILES[name])}: this cache has no '{name}' keys")
        return self._K[name]

    @property
    def d(self) -> int:
        return int(self.K("tf").shape[1])

    # ------------------------------------------------------------------ selection
    def rows(self, kind=None, family=None, T=None, pool=None, valid=None, where=None) -> list:
        """Positions of the index rows with the given kind(s) / family / pool / validity of the pre-edit answer.
        T: keep the sequences of a log with T edits of the family (edits and hard negatives with edit < T; cross rows also
        edit_b < T); rows without an edit are kept.  pool: 'repair' | 'heldout' | 'heldout_disjoint' | 'heldout_strict' |
        'benign'.  where: a predicate on the index row."""
        kinds = None if kind is None else ({kind} if isinstance(kind, str) else set(kind))
        out = []
        for j, e in enumerate(self.index):
            if kinds is not None and e.get("kind") not in kinds:
                continue
            if family is not None and e.get("family") != family:
                continue
            if T is not None and e.get("edit") is not None and not (int(e["edit"]) < T and (e.get("edit_b") is None or int(e["edit_b"]) < T)):
                continue
            if pool is not None:
                ok = {"heldout_disjoint": bool(e.get("disjoint")), "heldout_strict": bool(e.get("strict"))}.get(pool, e.get("pool") == pool)
                if not ok:
                    continue
            if valid is not None and bool(e.get("y0_valid")) != valid:
                continue
            if where is not None and not where(e):
                continue
            out.append(j)
        return out

    def find(self, fmt, prompt, image, answer=None):
        """Index position of the sequence with this content (teacher-forced text `answer`; None: any sequence of the prompt)."""
        return (self._tf if answer is not None else self._q).get(content_key(fmt, prompt, image, answer))

    def by_sid(self, sid):
        return self._sid.get(sid)

    def family_of(self, uids):
        """The family whose edit stream starts with the edit uids of a log: every cached edit position < T carries the log's
        uid there, and at least one is cached (None if there is no such family; a partial cache may hold fewer than T edits)."""
        first = {}
        for e in self.index:
            if e.get("kind") == "rel" and e.get("edit") is not None:
                first.setdefault(e.get("family"), {})[int(e["edit"])] = e.get("edit_uid")
        for fam, m in first.items():
            hit = [m[i] == u for i, u in enumerate(uids) if i in m]
            if hit and all(hit):
                return fam
        return None

    def log_rows(self, uids, kind=None, **kw) -> list:
        """Positions of the cached sequences that belong to a log with these edit uids (EditLog.meta['uids']): the sequences
        of its family, edits and hard negatives restricted to the log's edits; kind / pool / valid / where as in rows()."""
        fam = self.family_of(uids)
        if fam is None:
            raise LookupError("the key cache holds no edit stream that starts with the edits of this log")
        return self.rows(kind=kind, family=fam, T=len(uids), **kw)

    # ------------------------------------------------------------------ keys
    def _row(self, j, field):
        v = self.index[j].get(field)
        if v is None:
            raise KeyError(f"sequence {self.index[j]['sid']} has no {field}")
        return v

    def key_rows(self, which: str, rows) -> list:
        """Rows of the key file behind `which` in ('tf', 'prompt', 'decode') for the index positions `rows`
        ('prompt': the teacher-forced sequence at the decode's token = row_tf_prompt where lp_shift != 0, else row_tf)."""
        if which == "tf":
            return [int(self._row(j, "row_tf")) for j in rows]
        if which == "prompt":
            return [int(self._row(j, "row_tf_prompt") if self.index[j].get("lp_shift") else self._row(j, "row_tf")) for j in rows]
        if which == "decode":
            return [int(self._row(j, "row_decode")) for j in rows]
        raise ValueError("which: 'tf' | 'prompt' | 'decode'")

    def keys(self, which: str, rows=None) -> torch.Tensor:
        """float32 keys of the index positions `rows` (default: every sequence that has the key): (n, d) for 'tf', 'prompt',
        'decode'; (n, 11, d) for 'steps' (the 11 full forwards of a decode, some identical)."""
        rows = self.having(which) if rows is None else list(rows)
        if which == "steps":
            idx = torch.tensor([self._row(j, "rows_decode_steps") for j in rows], dtype=torch.long).reshape(len(rows), -1)
            return self.K("steps")[idx.reshape(-1)].reshape(idx.shape[0], idx.shape[1], -1)
        return self.K("decode" if which == "decode" else "tf")[torch.tensor(self.key_rows(which, rows), dtype=torch.long)]

    def own_keys(self, uids, metric: str = "rel", answer=True):
        """(keys (n, d) float32, edit ids (n,)) for repair.add_guards, as Evaluator.own_keys but from the cache: the key at the
        last prompt position of every edit's own B = 1 sequence (`metric`), for the edits of a log with these uids.
        answer=True: the teacher-forced key; 'all': also the keys of the 11 full forwards of a decode, for edits whose decode
        sits on the same token (lp_shift == 0).  Raises LookupError if an edit of the log is not cached."""
        if answer not in (True, "all"):
            raise ValueError("answer: True | 'all'")
        pos = {}
        for j in self.log_rows(uids, kind=metric):
            pos.setdefault(int(self.index[j]["edit"]), j)
        miss = [i for i in range(len(uids)) if i not in pos or self.index[pos[i]].get("row_tf") is None]
        if miss:
            raise LookupError(f"{len(miss)} of {len(uids)} '{metric}' sequences of this log are not in the key cache (e.g. edits {miss[:5]})")
        ks, es = [], []
        for i in range(len(uids)):
            e = self.index[pos[i]]
            new = [self.K("tf")[int(e["row_tf"])][None]]
            if answer == "all" and e.get("lp_shift") == 0:
                new.append(self.K("steps")[torch.tensor(e["rows_decode_steps"], dtype=torch.long)])
            for k in new:
                ks.append(k)
                es += [i] * k.shape[0]
        return torch.cat(ks), es

    def having(self, which: str) -> list:
        """Index positions of the sequences for which `which` is defined."""
        f = {"tf": ("row_tf",), "prompt": ("row_tf",), "decode": ("row_decode",), "steps": ("rows_decode_steps",),
             "both": ("row_tf", "row_decode"), "all": ("row_tf", "rows_decode_steps")}[which]
        return [j for j, e in enumerate(self.index) if all(e.get(k) is not None for k in f)]

    # ------------------------------------------------------------------ gates on the CPU
    @staticmethod
    def _check(gate):
        if isinstance(gate, ScopedLog):
            if gate.role != "last":
                raise NotImplementedError(f"role {gate.role!r} needs every position of a sequence; the key cache holds the last prompt position")
            if gate.precision != "fp64":
                raise NotImplementedError("only the fp64 gate is reproducible from cached keys")
            return gate if gate.device.type == "cpu" else gate.to("cpu")
        if getattr(gate, "kind", None) == "router":
            return gate
        raise NotImplementedError(f"{type(gate).__name__} ({getattr(gate, 'kind', '')}): this gate does not decide at the last prompt position")

    def _parts(self, which, rows):
        """[(rows of a key file, key file name, positions in `rows`)]: the keys every query of `which` presents to the gate."""
        if which in ("tf", "prompt", "decode"):
            return [(self.key_rows(which, rows), "decode" if which == "decode" else "tf", list(range(len(rows))))]
        if which == "steps":
            st = [self._row(j, "rows_decode_steps") for j in rows]
            return [([int(r) for s in st for r in s], "steps", [i for i, s in enumerate(st) for _ in s])]
        if which == "both":
            return self._parts("tf", rows) + self._parts("decode", rows)
        if which == "all":
            sh = [i for i, j in enumerate(rows) if self.index[j].get("lp_shift")]
            extra = [([int(self._row(rows[i], "row_tf_prompt")) for i in sh], "tf", sh)] if sh else []
            return self._parts("tf", rows) + extra + self._parts("steps", rows)
        raise ValueError(f"which: one of {WHICH}")

    def coords(self, gate: ScopedLog, rows=None, which: str = "tf", chunk: int = 1024) -> torch.Tensor:
        """(n, D) fp64: the largest feature of every detector over the keys that each sequence presents to the gate
        (= ScopedLog.query_max of Evaluator.eligible_keys; 'tf': one key).  Pass it as `coords=` to
        calibrate.calibrate_two_block / calibrate_shared (rows = the benign calibration queries)."""
        g = self._check(gate)
        if not isinstance(g, ScopedLog):
            raise TypeError("coords needs a scoped log")
        rows = self.having(which) if rows is None else list(rows)
        out = torch.full((len(rows), g.D), -INF, dtype=torch.float64)
        for krows, name, pos in self._parts(which, rows):
            K = self.K(name)
            for a in range(0, len(krows), chunk):
                r = g.scores(K[torch.tensor(krows[a:a + chunk], dtype=torch.long)])
                p = torch.tensor(pos[a:a + chunk], dtype=torch.long)
                out.scatter_reduce_(0, p[:, None].expand_as(r), r, reduce="amax", include_self=True)     # several keys of a query: maximum
        return out

    def decide(self, gate, which: str = "tf", rows=None, uids=None, slack: float = SLACK, chunk: int = 1024, log=None) -> dict:
        """Firing decision of the gate on every sequence in `rows` (default: every cached sequence that has the key), CPU.
        log: the EditLog the gate was built from (shorthand for uids=log.meta['uids'] and, if rows is None, rows = the
        sequences of that log that have the key: log_rows).

        -> dict of arrays aligned with `rows`:
             rows        the index positions
             fired       some detector is a candidate on a key of the sequence
             near        some coordinate lies within `slack` (relative) of its threshold: the GPU hook may decide otherwise
             n_cand      number of candidate detectors (largest over the keys of the sequence)
             top_edit    the edit that releases under fire 'top1' at the first key that fires (-1: nothing fires)
             margin      largest r_j - theta_j over detectors and keys (>= 0 <=> fired)
           with uids (the edit uids of the gate's log, EditLog.meta['uids']) also, for sequences that belong to an edit of
           the log (index field edit_uid):
             own         that edit's index in the log (-1: none)
             own_cand    the own edit is a candidate; own_top: it is the edit that releases
        Router gates: fired / near / margin (activation - threshold) only."""
        g = self._check(gate)
        if log is not None:
            uids = log.meta["uids"] if uids is None else uids
            if rows is None:
                ok = set(self.having(which))
                rows = [j for j in self.log_rows(uids) if j in ok]
        rows = self.having(which) if rows is None else list(rows)
        n = len(rows)
        fired, near = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
        margin = np.full(n, -np.inf)
        if not isinstance(g, ScopedLog):                                  # router: |(W_E - W_0) k| >= eps at the last prompt position
            for krows, name, pos in self._parts(which, rows):
                K = self.K(name)
                for a in range(0, len(krows), chunk):
                    act = g.activation(K[torch.tensor(krows[a:a + chunk], dtype=torch.long)]).cpu().numpy()
                    p = np.asarray(pos[a:a + chunk])
                    np.maximum.at(margin, p, act - g.thr)
                    np.logical_or.at(near, p, np.abs(act - g.thr) <= slack * abs(g.thr) + 1e-300)
            return {"rows": rows, "which": which, "fired": margin >= 0, "near": near, "margin": margin}
        th = g.theta()
        n_cand, top_edit = np.zeros(n, dtype=np.int64), np.full(n, -1, dtype=np.int64)
        own = np.full(n, -1, dtype=np.int64)
        if uids is not None:
            pos_of = {u: i for i, u in enumerate(uids)}
            own = np.array([pos_of.get(self.index[j].get("edit_uid"), -1) for j in rows], dtype=np.int64)
        own_cand, own_top = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
        own_t = torch.from_numpy(own)
        for krows, name, pos in self._parts(which, rows):
            K = self.K(name)
            for a in range(0, len(krows), chunk):
                kc = K[torch.tensor(krows[a:a + chunk], dtype=torch.long)]
                r = g.scores(kc)
                p = np.asarray(pos[a:a + chunk])
                cand = r >= th[None, :]
                f = cand.any(1)
                d = r - th[None, :]
                m = torch.where(torch.isnan(d), torch.full_like(d, -INF), d).max(1).values.numpy()
                nr = near_threshold(g, kc, r, slack).numpy()
                top = g.det_edit[torch.where(cand, g.rank_key(r), torch.full_like(r, -INF)).argmax(1)]
                ce = g._edit_any(cand)
                o = own_t[torch.from_numpy(p)]
                oc = ce[torch.arange(ce.shape[0]), o.clamp(min=0)] & (o >= 0)
                for i in range(len(p)):                                   # several keys of a sequence: OR / max, first firing key decides top
                    q = int(p[i])
                    if bool(f[i]) and not fired[q]:
                        top_edit[q] = int(top[i])
                        own_top[q] = bool(o[i] >= 0 and int(top[i]) == int(o[i]))
                    fired[q] |= bool(f[i])
                    near[q] |= bool(nr[i])
                    margin[q] = max(margin[q], float(m[i]))
                    n_cand[q] = max(n_cand[q], int(cand[i].sum()))
                    own_cand[q] |= bool(oc[i])
        out = {"rows": rows, "which": which, "fired": fired, "near": near, "n_cand": n_cand, "top_edit": top_edit, "margin": margin}
        if uids is not None:
            out.update(own=own, own_cand=own_cand, own_top=own_top)
        return out

    def summary(self) -> dict:
        """Counts by family and kind, and which key files exist."""
        c = {}
        for e in self.index:
            k = f"{e.get('family')}|{e.get('kind')}"
            c[k] = c.get(k, 0) + 1
        return {"dir": self.dir, "realpath": os.path.realpath(self.dir), "is_final": self.is_final, "backbone": self.backbone, "layer": self.layer,
                "n": self.n, "by_family_kind": dict(sorted(c.items())), "files": {k: self.has(k) for k in _FILES}, "complete": self.meta.get("complete")}


def write_cache(directory: str, index: list, keys_tf: torch.Tensor, keys_decode: torch.Tensor | None = None, keys_steps: torch.Tensor | None = None,
                meta: dict | None = None):
    """Write a key cache in the format KeyCache reads (used for local caches and tests; the per-backbone caches of the
    final experiments are written by 10_final/cache/code).  Keys must be float32."""
    os.makedirs(directory, exist_ok=True)
    for name, t in (("tf", keys_tf), ("decode", keys_decode), ("steps", keys_steps)):
        if t is None:
            continue
        if t.dtype != torch.float32:
            raise ValueError("keys are stored as float32 (exact images of the module dtype)")
        tmp = os.path.join(directory, _FILES[name] + f".tmp{os.getpid()}")
        torch.save(t.contiguous().cpu(), tmp)
        os.replace(tmp, os.path.join(directory, _FILES[name]))
    tmp = os.path.join(directory, f"index.jsonl.tmp{os.getpid()}")
    with open(tmp, "w") as f:
        for e in index:
            f.write(json.dumps(e) + "\n")
    os.replace(tmp, os.path.join(directory, "index.jsonl"))
    with open(os.path.join(directory, "index.jsonl"), "rb") as f:
        meta = dict(meta or {}, index_md5=hashlib.md5(f.read()).hexdigest(), n_sequences=len(index))
    for name, t in (("keys_tf", keys_tf), ("keys_decode", keys_decode), ("keys_decode_steps", keys_steps)):
        if t is not None:
            meta[name] = dict(meta.get(name) or {}, shape=list(t.shape))
    with open(os.path.join(directory, "meta.json"), "w") as f:     # written last: the reader checks index and key files against it
        json.dump(meta, f, indent=1)
    with open(os.path.join(directory, "FORMAT.txt"), "w") as f:
        f.write("Key cache in the format of rescope/keycache.py (see its module docstring): index.jsonl (one JSON object per sequence), "
                "keys_tf.pt / keys_decode.pt / keys_decode_steps.pt (float32 (n, d) tensors addressed by row_tf / row_tf_prompt / row_decode / "
                "rows_decode_steps).\n")
