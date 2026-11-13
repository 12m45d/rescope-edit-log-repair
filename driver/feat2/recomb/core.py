"""recomb.core: rows, keys and per-edit baselines of the log-derived recombinations (variant rescope_x / <gate>_x).

PROTOCOL.md section 6, amendment "Review round 1", item R5:  for every edit i, m = 4 recombinations of each kind (X-I: its image +
the image-locality question of a partner item; X-Q: its question + the entity image of a partner item; X-X: its image + the question
of another edit j); only the last-prompt key of each recombination is captured (prompt only, pre-edit model); in the two-block rule
the baseline of edit i becomes the maximum of its block-1 baseline and of its coordinates on ITS OWN recombinations; block 2, the
quantile and the guards are unchanged.

DATA (built by rc_select.py / rc_capture.py / rc_assemble.py of this directory; README.txt next to the files)
    <DATA>/<backbone>/rows.jsonl      one row per recombination, 12 per edit of the backbone's long stream, in stream order
    <DATA>/<backbone>/keys.pt         {'rids', 'K_last' float32 (n, d), 'last_prompt_idx', ...}: row r of K_last = line r of rows.jsonl
    DATA = 10_final/cache2/recomb  (environment variable RECOMB_DATA overrides it: tests)

WHAT A LOG MAY USE.  A log whose edit uids are the first T edits of the backbone's long stream (every VLKEB log of the paper: the
development logs, working points, the T = 200 .. 2000 snapshots, the MEMIT-style logs, the prefix logs of 10 / 20 / 50 edits) uses the
rows with  owner position < T  and, for X-X,  partner position < T  (row field min_T <= T).  Any other log (other stream, other order,
MMKE-Bench, one-hop, BLIP2) raises LookupError: there are no recombinations for it.

THE BASELINE.  For a score function with one number per detector and key (S.scores(k) of a rescope.ScopedLog: the normalised solver
coordinate of feature_gate(log, 'ncoord'), a cosine-family gate, null-repaired detectors, ...; or any matrix builder keys -> (n, D)):
    x_j = max over the usable rows rho OWNED by the edit of detector j of score_j(key_rho)        (-inf if the edit has no usable row)
The owner of a row is the edit it was built for (X-I, X-X: the edit whose IMAGE the row shows; X-Q: the edit whose QUESTION it asks);
the edit j whose question an X-X row borrows is NOT an owner of that row.

HOW THE LIBRARY TAKES IT.  rescope.calibrate represents the per-detector baselines as a (D,) fp64 vector that it computes itself,
b = coords[block 1].max(0) (two_block_rule), keeps for the ranking (ScopedLog.det_b) and turns into floating-point-safe thresholds;
it has no argument for an external baseline.  A column-wise maximum over block 1 is all it does with block 1, so the baseline of R5,
max(b_j, x_j), is obtained by handing the library x as ONE MORE BLOCK-1 ROW of the calibration matrix with explicit blocks:
    coords' = [coords; x],   blocks = (block 1 + [n], block 2)        (augment)
    calibrate_two_block(S, coords=coords', alpha=alpha, blocks=blocks)                       (calibrate_two_block_x does exactly this)
The seeded split is taken on the n real queries (rescope.split_blocks(n, split, seed)), so block 2 is the block 2 of the default
method; scores, exact rank, eta, strict gate, floating-point-safe thresholds, ranking by the margin and the guards (keep_guards: capped
edits are counted) are the library's own code.  With x = -inf (no recombinations) the result is calibrate_two_block(S, coords=coords,
alpha, split=split, seed=seed) bit for bit (tests/t_core.py).  two_block_rule_x composes the same rule from the amendment's text
without the library's two_block_rule (reference for the tests; tests/tcommon.naive_rule is a second, plain-loop reference).

VALIDITY.  x depends on the log and on the recombination keys only: not on the calibration queries, not on any test query (the
recombination queries are disjoint from every test query: rc_check.py).  The score function s(q) = max_j (Q[q, j] - max(b_j, x_j)) / u_j
therefore still depends on block 1 and on fixed quantities only, the block-2 scores and the score of a fresh benign query stay
exchangeable, and the bound of the two-block rule holds as before.  It is a bound for benign queries; the recombinations make the
per-edit baselines harder, they do not make near-scope queries exchangeable with anything.

Everything is CPU, fp64, deterministic: rows in file order, scores in chunks of CHUNK = 1024 keys (the arithmetic of
ScopedLog.scores, as rescope.KeyCache.coords) with ONE BLAS thread (single_thread: the baseline x is then the same bits in every
process, whatever its thread setting; x_sha1 in the record identifies it).  No gate is evaluated here on any test row.
"""
from __future__ import annotations
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP

import contextlib
import hashlib
import json
import os
from dataclasses import dataclass, field

import numpy as np
import torch

ROOT = f"{_RP.ROOT}"
LIB = ROOT + "/02_research/code"
FINAL = ROOT + "/02_research/experiments/10_final"
DEFAULT_DATA = FINAL + "/cache2/recomb"
M = 4                                             # recombinations of each kind per edit
KINDS = ("X-I", "X-Q", "X-X")
BACKBONES = ("qwen25vl", "llava")
CHUNK = 1024
NEG = float("-inf")
_STORES = {}


def data_root() -> str:
    return os.environ.get("RECOMB_DATA", DEFAULT_DATA)


@contextlib.contextmanager
def single_thread():
    """One BLAS thread inside the block (restored afterwards).  MKL's matrix products depend on the number of threads in their last
    bits (measured here: four different baselines x under 1, 2, 4 and 8 threads); with one thread and the fixed chunks of CHUNK keys the
    scores of the recombination rows are the same bits in every process, whatever its thread setting (tests/t_real.py)."""
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(n)


def _lib():
    """The library, imported on first use (read-only)."""
    import sys
    if LIB not in sys.path:
        sys.path.insert(0, LIB)
    import rescope
    return rescope


def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


# ============================================================================= the store of one backbone
class Store:
    """Rows and keys of one backbone, opened together and checked against each other.

      backbone, dir, n, d, layer, T_stream, family
      rows        list of the row dicts of rows.jsonl (file order; fields: rc_select.py / README.txt)
      rids        list of row ids;  K  float32 (n, d) keys (memory-mapped), K[r] = key of rows[r];  lp  (n,) last prompt index
      kind        (n,) int: index into KINDS;  owner (n,) stream position of the owning edit;  partner (n,) X-X partner (-1: none)
      min_T       (n,) the smallest log length that may use the row (max(owner, partner) + 1)
      stream      edit uid at every stream position (T_stream uids)
      meta        capture_meta.json (md5 of keys.pt, bitwise verification of the capture path, ...) or {}
    verify=True also checks the md5 of keys.pt against capture_meta.json (reads the file once)."""

    def __init__(self, backbone: str, root: str | None = None, verify: bool = False):
        self.backbone = backbone
        self.dir = os.path.join(data_root() if root is None else root, backbone)
        rp, kp, mp = (os.path.join(self.dir, f) for f in ("rows.jsonl", "keys.pt", "capture_meta.json"))
        if not os.path.exists(rp) or not os.path.exists(kp):
            raise FileNotFoundError(f"no recombination store for backbone {backbone!r} at {self.dir} (rows.jsonl / keys.pt missing; "
                                    "built by rc_select.py, rc_capture.py, rc_assemble.py)")
        with open(rp, "rb") as f:
            raw = f.read()
        self.rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
        self.rows_sha256 = hashlib.sha256(raw).hexdigest()
        try:
            blob = torch.load(kp, map_location="cpu", mmap=True)
        except (TypeError, RuntimeError, ValueError):                   # a torch without mmap support
            blob = torch.load(kp, map_location="cpu")
        self.K = blob["K_last"]
        self.rids = list(blob["rids"])
        self.lp = np.asarray(blob["last_prompt_idx"], dtype=np.int64)
        self.layer = blob.get("layer")
        self.n, self.d = int(self.K.shape[0]), int(self.K.shape[1])
        self.meta = {}
        if os.path.exists(mp):
            with open(mp) as f:
                self.meta = json.load(f)

        def bad(msg):
            return RuntimeError(f"inconsistent recombination store at {self.dir}: {msg}")
        if self.K.dtype != torch.float32 or self.K.dim() != 2:
            raise bad(f"K_last is {self.K.dtype} {tuple(self.K.shape)}, expected float32 (n, d)")
        if blob.get("backbone") != backbone:
            raise bad(f"keys.pt belongs to backbone {blob.get('backbone')!r}")
        if self.rids != [r["rid"] for r in self.rows] or self.lp.shape[0] != self.n:
            raise bad("the row ids of keys.pt are not the rows of rows.jsonl, in order")
        if blob.get("rows_sha256") not in (None, self.rows_sha256):
            raise bad("keys.pt was captured for another rows.jsonl (sha256 differs)")
        if self.meta.get("keys_md5") and verify and _md5(kp) != self.meta["keys_md5"]:
            raise bad("md5 of keys.pt differs from capture_meta.json")
        kid = {k: i for i, k in enumerate(KINDS)}
        self.kind = np.array([kid[r["kind"]] for r in self.rows], dtype=np.int64)
        self.owner = np.array([r["edit_pos"] for r in self.rows], dtype=np.int64)
        self.partner = np.array([-1 if r["edit_b_pos"] is None else r["edit_b_pos"] for r in self.rows], dtype=np.int64)
        self.min_T = np.maximum(self.owner, self.partner) + 1
        if not np.array_equal(self.min_T, np.array([r["min_T"] for r in self.rows], dtype=np.int64)):
            raise bad("row field min_T is not max(owner, partner) + 1")
        self.T_stream = int(self.owner.max()) + 1
        self.family = self.rows[0]["family"]
        stream = {}
        for r in self.rows:
            if stream.setdefault(r["edit_pos"], r["edit_uid"]) != r["edit_uid"]:
                raise bad(f"two edit uids at stream position {r['edit_pos']}")
        if sorted(stream) != list(range(self.T_stream)):
            raise bad("stream positions without rows")
        self.stream = [stream[i] for i in range(self.T_stream)]
        for r in self.rows:
            j = r["edit_b_pos"]
            if (j is None) != (r["kind"] != "X-X") or (j is not None and (j == r["edit_pos"] or self.stream[j] != r["edit_b_uid"])):
                raise bad(f"partner of {r['rid']}: only X-X rows have one, another edit of the stream, with the uid at its position")
        per = np.bincount(self.owner * len(KINDS) + self.kind, minlength=self.T_stream * len(KINDS))
        if not bool((per == M).all()):
            raise bad(f"not {M} rows of every kind for every edit")

    def check_log(self, uids) -> int:
        """T of a log whose edit uids are the first T edits of this stream; LookupError otherwise."""
        uids = list(uids)
        T = len(uids)
        if T < 1 or T > self.T_stream or uids != self.stream[:T]:
            first = next((i for i, (a, b) in enumerate(zip(uids, self.stream)) if a != b), min(T, self.T_stream))
            raise LookupError(f"no recombinations for this log: its {T} edits are not the first {T} edits of the {self.backbone} stream of family "
                              f"{self.family!r} ({self.T_stream} edits; first difference at position {first})")
        return T

    def select(self, T: int, kinds=KINDS) -> np.ndarray:
        """Row numbers (file order) a log with the first T edits may use: owner position < T and (X-X) partner position < T."""
        ks = _kinds(kinds)
        return np.nonzero((self.min_T <= int(T)) & np.isin(self.kind, [KINDS.index(k) for k in ks]))[0]

    def summary(self) -> dict:
        return {"dir": self.dir, "backbone": self.backbone, "family": self.family, "layer": self.layer, "T_stream": self.T_stream, "n_rows": self.n, "d": self.d,
                "n_by_kind": {k: int((self.kind == i).sum()) for i, k in enumerate(KINDS)}, "rows_sha256": self.rows_sha256,
                "keys_md5": self.meta.get("keys_md5"), "capture_verified_bitwise": (self.meta.get("verify") or {}).get("ok")}


def _kinds(kinds) -> tuple:
    ks = (kinds,) if isinstance(kinds, str) else tuple(kinds)
    if not ks or any(k not in KINDS for k in ks) or len(set(ks)) != len(ks):
        raise ValueError(f"kinds: a non-empty subset of {KINDS}")
    return tuple(k for k in KINDS if k in ks)


def available(backbone: str, root: str | None = None) -> bool:
    d = os.path.join(data_root() if root is None else root, backbone)
    return os.path.exists(os.path.join(d, "rows.jsonl")) and os.path.exists(os.path.join(d, "keys.pt"))


def store(backbone: str, root: str | None = None, verify: bool = False) -> Store:
    """The store of a backbone (one object per process and directory)."""
    key = (os.path.join(data_root() if root is None else root, backbone))
    if key not in _STORES:
        _STORES[key] = Store(backbone, root, verify)
    return _STORES[key]


# ============================================================================= the rows of a log
def identify(log):
    """(backbone, edit uids, EditLog or None) of: a fx.logs.LogData, a rescope.EditLog (from_testbed / from_harness), the name of a
    testbed log, or a pair (backbone, uids)."""
    if isinstance(log, str):
        log = _lib().from_testbed(log)
    if isinstance(log, (tuple, list)) and len(log) == 2 and isinstance(log[0], str):
        return log[0], list(log[1]), None
    if hasattr(log, "uids") and hasattr(log, "backbone") and hasattr(log, "log"):          # fx.logs.LogData
        return log.backbone, list(log.uids), log.log
    meta = getattr(log, "meta", None)
    if isinstance(meta, dict) and meta.get("uids") is not None and meta.get("backbone"):  # rescope.EditLog
        return meta["backbone"], list(meta["uids"]), log
    raise TypeError("log: a fx.logs.LogData, a rescope.EditLog with meta['uids'] and meta['backbone'], a log name, or (backbone, uids)")


@dataclass
class LogRows:
    """The recombination rows a log may use, in file order (edit by edit: X-I, X-Q, X-X).
      idx      (n,) row numbers in the store          edit     (n,) the OWNING edit = its index in the log
      kind     (n,) 'X-I' | 'X-Q' | 'X-X'             partner  (n,) X-X: the edit whose question the row asks (-1: none)
      rids     row ids                                 rows     the row dicts (flags: q_in_edit_first_pos, same_category, ...)
      keys()   float32 (n, d) last-prompt keys         subset(mask)   the rows of a boolean mask"""
    store: Store
    T: int
    idx: np.ndarray
    kinds: tuple = KINDS

    @property
    def edit(self) -> np.ndarray:
        return self.store.owner[self.idx]

    @property
    def partner(self) -> np.ndarray:
        return self.store.partner[self.idx]

    @property
    def kind(self) -> np.ndarray:
        return np.array(KINDS)[self.store.kind[self.idx]]

    @property
    def rids(self) -> list:
        return [self.store.rids[int(i)] for i in self.idx]

    @property
    def rows(self) -> list:
        return [self.store.rows[int(i)] for i in self.idx]

    def __len__(self) -> int:
        return int(self.idx.shape[0])

    def keys(self) -> torch.Tensor:
        return self.store.K[torch.from_numpy(self.idx)]

    def subset(self, mask) -> "LogRows":
        return LogRows(self.store, self.T, self.idx[np.asarray(mask, dtype=bool)], self.kinds)

    def counts(self) -> dict:
        k = self.store.kind[self.idx]
        return {name: int((k == i).sum()) for i, name in enumerate(KINDS)}

    def per_edit(self) -> dict:
        """{kind: (T,) number of usable rows of every edit}."""
        k, e = self.store.kind[self.idx], self.edit
        return {name: np.bincount(e[k == i], minlength=self.T) for i, name in enumerate(KINDS)}


def log_rows(log, kinds=KINDS, root: str | None = None) -> LogRows:
    """The recombination rows a log may use (owner position < T; X-X partner position < T).  log: see identify()."""
    bb, uids, _ = identify(log)
    st = store(bb, root)
    T = st.check_log(uids)
    return LogRows(st, T, st.select(T, kinds), _kinds(kinds))


def log_keys(log, kinds=KINDS, root: str | None = None):
    """(keys float32 (n, d), LogRows): the last-prompt keys (prompt only, pre-edit model, B = 1) of the rows a log may use."""
    rows = log_rows(log, kinds, root)
    return rows.keys(), rows


# ============================================================================= scores of the edits on their own rows
def _scorer(log, score, T: int, det_edit):
    """-> (fn(keys float32 (n, d)) -> (n, D) fp64 tensor on the CPU, det_edit (D,) or None (known after the first chunk), label)."""
    rs = _lib()
    if score is None or isinstance(score, str):
        _, _, elog = identify(log)
        if elog is None:
            raise TypeError("a named feature needs the edit log: pass a fx.logs.LogData, a rescope.EditLog or a log name")
        score = rs.feature_gate(elog, score or "ncoord")
    if isinstance(score, rs.ScopedLog):
        if int(score.T) != T:
            raise ValueError(f"the gate has {int(score.T)} edits, the log {T}")
        de = score.det_edit.detach().cpu().numpy().astype(np.int64)
        if det_edit is not None and not np.array_equal(np.asarray(det_edit), de):
            raise ValueError("det_edit differs from the gate's det_edit")
        S = score
        return (lambda K: S.scores(K.to(S.device)).cpu()), de, f"ScopedLog:{S.meta.get('feature', S.meta.get('baseline', S.name))}"
    if callable(score):
        de = None if det_edit is None else np.asarray(det_edit, dtype=np.int64)
        return (lambda K: torch.as_tensor(score(K)).detach().to("cpu", torch.float64)), de, f"callable:{getattr(score, '__name__', type(score).__name__)}"
    raise TypeError("score: None | a feature name of rescope.feature_gate | a rescope.ScopedLog | a callable keys (n, d) -> (n, D) | an (n_rows, D) matrix")


@dataclass
class OwnScores:
    """Scores of the edits on their own recombination rows.
      rows      LogRows (n rows)                 value   (n,) fp64: score of the row's OWN edit (primary detector) at the row's key
      x         (D,) fp64 tensor: per detector the largest score over the usable rows of its edit (-inf: no usable row) = the baseline
                that calibrate_two_block_x combines with the block-1 baseline
      det_edit  (D,) edit of every detector      label   what was scored
      per_edit(kind=None)   (T,) largest `value` over the edit's rows (of one kind); -inf where it has none"""
    rows: LogRows
    value: np.ndarray
    x: torch.Tensor
    det_edit: np.ndarray
    label: str
    matrix: torch.Tensor | None = field(default=None, repr=False)

    def per_edit(self, kind=None) -> np.ndarray:
        out = np.full(self.rows.T, NEG)
        m = np.ones(len(self.rows), dtype=bool) if kind is None else (self.rows.kind == kind)
        np.maximum.at(out, self.rows.edit[m], self.value[m])
        return out

    def record(self) -> dict:
        x = self.x.numpy()
        fin = x[np.isfinite(x)]
        st = self.rows.store
        return {"what": "per-detector maximum of the score over the recombination rows owned by the detector's edit (amendment R5)",
                "backbone": st.backbone, "T": self.rows.T, "kinds": list(self.rows.kinds), "m": M, "n_rows": len(self.rows), "n_rows_by_kind": self.rows.counts(),
                "score": self.label, "n_detectors": int(x.shape[0]), "n_detectors_without_rows": int(np.isneginf(x).sum()),
                "x_min_median_max": [float(v) for v in np.quantile(fin, [0.0, 0.5, 1.0])] if fin.size else None,
                "x_sha1": hashlib.sha1(np.ascontiguousarray(x).tobytes()).hexdigest()[:12],
                "keys": "last prompt position of the prompt-only sequence, pre-edit model, B = 1", "rows_sha256": st.rows_sha256,
                "keys_md5": st.meta.get("keys_md5"), "store": st.dir}


def own_scores(log, score=None, kinds=KINDS, det_edit=None, rows: LogRows | None = None, chunk: int = CHUNK, full: bool = False,
               root: str | None = None) -> OwnScores:
    """Scores of every edit of a log on ITS OWN recombination rows.

    log     a fx.logs.LogData, a rescope.EditLog, a log name or (backbone, uids): see identify()
    score   None / 'ncoord' (the default: the normalised solver coordinate <W_i, k> / <W_i, k_i>) or another feature name of
            rescope.feature_gate ('coord', 'cos', 'negdist');
            a rescope.ScopedLog (ANY detectors: the un-thresholded base gate of a method, a cosine-family gate, null-repaired
            detectors; its S.scores are used, thresholds are ignored);
            a callable  keys float32 (n, d) -> (n, D) matrix  (a score-matrix builder of another feature package; one column per
            edit, or pass det_edit (D,) = the edit of every column);
            or the (n_rows, D) matrix itself, aligned with log_rows(log, kinds).
    kinds   the kinds to use (default: all three = the registered rule);  rows: an explicit LogRows (e.g. a subset) instead
    full    keep the whole (n, D) score matrix (.matrix)
    -> OwnScores (.x = the baseline vector, .value, .per_edit(kind))."""
    R_ = rows if rows is not None else log_rows(log, kinds, root)
    T, n = R_.T, len(R_)
    owner = torch.from_numpy(R_.edit)
    pre = None
    if torch.is_tensor(score) or isinstance(score, np.ndarray):
        pre = torch.as_tensor(score).detach().to("cpu", torch.float64)
        if pre.dim() != 2 or pre.shape[0] != n:
            raise ValueError(f"a score matrix must be (n_rows, D) = ({n}, D), aligned with log_rows(log, kinds)")
        fn, de, label = None, (None if det_edit is None else np.asarray(det_edit, dtype=np.int64)), "matrix"
    else:
        fn, de, label = _scorer(log, score, T, det_edit)
    xs, vals, mats = None, [], []
    for a in range(0, max(n, 1), chunk):
        o = owner[a:a + chunk]
        if pre is not None:
            Rm = pre[a:a + chunk]
        else:
            with single_thread():
                Rm = fn(R_.store.K[torch.from_numpy(R_.idx[a:a + chunk])])
        if Rm.dim() != 2 or Rm.shape[0] != o.shape[0]:
            raise ValueError(f"the score function returned shape {tuple(Rm.shape)} for {o.shape[0]} keys")
        if bool(torch.isnan(Rm).any()):
            raise ValueError("NaN score on a recombination key")
        D = int(Rm.shape[1])
        if de is None:
            if D != T:
                raise ValueError(f"the score matrix has {D} columns for {T} edits: pass det_edit (the edit of every column)")
            de = np.arange(T, dtype=np.int64)
        if de.shape != (D,) or de.min() < 0 or de.max() >= T:
            raise ValueError("det_edit must give an edit of the log for every column")
        if xs is None:
            xs = torch.full((D,), NEG, dtype=torch.float64)
            det = torch.from_numpy(de)
            prim = torch.full((T,), -1, dtype=torch.long)               # primary detector of every edit = its first column
            for j in range(D - 1, -1, -1):
                prim[int(de[j])] = j
            if int(prim.min()) < 0:
                raise ValueError("an edit of the log has no column in the score matrix")
        if o.shape[0]:
            mine = det[None, :] == o[:, None]
            xs = torch.maximum(xs, torch.where(mine, Rm, torch.full_like(Rm, NEG)).max(0).values)
            vals.append(Rm[torch.arange(o.shape[0]), prim[o]])
        if full:
            mats.append(Rm)
    if xs is None:                                                      # (no chunk at all cannot happen: the loop runs once for n = 0)
        raise RuntimeError("no score was computed")
    return OwnScores(rows=R_, value=(torch.cat(vals).numpy() if vals else np.zeros(0)), x=xs, det_edit=de, label=label,
                     matrix=torch.cat(mats) if full and mats else None)


def own_baseline(log, score=None, kinds=KINDS, **kw):
    """(x (D,) fp64 tensor, record): the per-edit baseline 'maximum (normalised) coordinate of edit i on its own recombinations' for
    any score function (own_scores: arguments), in the form calibrate_two_block_x / augment combine with the block-1 baseline."""
    o = own_scores(log, score, kinds, **kw)
    return o.x, o.record()


# ============================================================================= the two-block rule with an external baseline
def _vec(x, D: int) -> np.ndarray:
    v = np.full(D, NEG) if x is None else np.ascontiguousarray(torch.as_tensor(x).detach().cpu().numpy() if torch.is_tensor(x) else x, dtype=np.float64).reshape(-1)
    if v.shape != (D,):
        raise ValueError(f"the baseline x must have one entry per detector ({D}), got {v.shape}")
    if np.isnan(v).any() or np.isposinf(v).any():
        raise ValueError("the baseline x must not contain NaN or +inf (-inf = no recombination row)")
    return v


def augment(coords, x, split=(1, 1), seed: int = 0, blocks=None):
    """The library's calibration input with the external baseline as ONE MORE BLOCK-1 ROW.
    coords (n, D): per-query maxima of the n benign calibration queries (S.query_max / KeyCache.coords);  x (D,): own_baseline.
    -> (coords' (n + 1, D) fp64 tensor = [coords; x],  (idx1 + [n], idx2)) with (idx1, idx2) = rescope.split_blocks(n, split, seed)
    (or the given blocks): calibrate_two_block(S, coords=coords', alpha=alpha, blocks=(idx1 + [n], idx2)) then computes
    b = max(block-1 maximum, x) and everything else unchanged."""
    rs = _lib()
    Q = torch.as_tensor(coords).detach().to("cpu", torch.float64)
    if Q.dim() != 2:
        raise ValueError("coords must be an (n, D) matrix")
    n, D = Q.shape
    i1, i2 = rs.split_blocks(n, tuple(split), int(seed)) if blocks is None else (np.asarray(blocks[0], dtype=np.int64), np.asarray(blocks[1], dtype=np.int64))
    if len(i1) < 1 or len(i2) < 1 or len(np.intersect1d(i1, i2)) or max(int(i1.max()), int(i2.max())) >= n:
        raise ValueError("the two blocks must be non-empty, disjoint sets of the n calibration queries")
    Q2 = torch.cat([Q, torch.from_numpy(_vec(x, D))[None, :]])
    return Q2, (np.concatenate([i1, np.array([n], dtype=np.int64)]), i2)


def two_block_rule_x(coords, x, alpha, split=(1, 1), seed: int = 0, blocks=None, u=None) -> dict:
    """The rule of amendment R5 on numbers, composed from its text with the library's helpers (block_scores, conformal_rank_exact,
    float_safe_threshold) but NOT through rescope.two_block_rule; reference of the tests, next to the plain-loop tests/tcommon.naive_rule:
        block 1 -> b1_j = max_q coords[q, j];   b_j = max(b1_j, x_j);   block 2 -> s(q) = max_j (coords[q, j] - b_j) / u_j;
        eta = the k-th smallest s, k = ceil((1 - alpha)(n2 + 1)) exact (+inf if k > n2);   detector j is a candidate iff (r - b_j) / u_j > eta.
    -> dict(b1, x, b, eta, tau (floating-point-safe: r >= tau_j <=> candidate), rank, n1, n2, idx1, idx2, n_allowed, bound, scores2)."""
    rs = _lib()
    Q = np.asarray(torch.as_tensor(coords).detach().cpu().numpy() if torch.is_tensor(coords) else coords, dtype=np.float64)
    n, D = Q.shape
    i1, i2 = rs.split_blocks(n, tuple(split), int(seed)) if blocks is None else (np.asarray(blocks[0], dtype=np.int64), np.asarray(blocks[1], dtype=np.int64))
    xv = _vec(x, D)
    b1 = Q[i1].max(0)
    b = np.maximum(b1, xv)
    uu = None if u is None else np.broadcast_to(np.asarray(u, dtype=np.float64), (D,)).copy()
    sc = rs.block_scores(Q[i2], b, uu)
    n2 = int(sc.size)
    k = rs.conformal_rank_exact(n2, alpha)
    eta = float("inf") if k > n2 else float(np.sort(sc)[k - 1])
    tau = rs.float_safe_threshold(b, np.ones(D) if uu is None else uu, eta)
    allowed = 0 if k > n2 else n2 + 1 - k
    return {"b1": b1, "x": xv, "b": b, "u": uu, "eta": eta, "tau": tau, "rank": k, "n1": int(len(i1)), "n2": n2, "idx1": i1, "idx2": i2,
            "n_allowed": allowed, "bound": allowed / (n2 + 1), "scores2": sc}


def _quant(v):
    v = np.asarray(v, dtype=np.float64)
    f = v[np.isfinite(v)]
    return [float(q) for q in np.quantile(f, [0.0, 0.5, 1.0])] if f.size else None


def calibrate_two_block_x(S, queries=None, alpha=0.05, *, x, coords=None, split=(1, 1), seed: int = 0, blocks=None, u=None, rank: str = "excess",
                          allow_silence: bool = False):
    """Two-block calibration with the baseline of amendment R5: b_j = max(block-1 baseline, x_j).  Arguments and output as
    rescope.calibrate_two_block, plus
        x    (D,) per-detector baseline from the recombinations (own_baseline(log, S)[0]; None or -inf entries: no recombination).
    The library does all the work: x is handed to rescope.calibrate_two_block as one more block-1 row (augment), with the blocks of
    rescope.split_blocks(n, split, seed) on the n REAL queries.  Block 2, the exact rank, eta, the strict gate, the floating-point-safe
    thresholds, the ranking by the margin (det_b = the combined baseline) and the guards (capped edits: rec['n_edits_capped']) are
    unchanged; with x = -inf everywhere the scoped log equals calibrate_two_block(S, coords=coords, alpha, split=split, seed=seed) bit
    for bit.  x must not depend on the calibration queries (own_baseline depends on the log and the recombination keys only).
    -> (scoped log, record); record = the library's record with n / n1 / block1 / split / seed of the real queries, and
    record['recomb']: the block-1 baselines, x, how many baselines x raises, eta and thresholds of the rule WITHOUT x (same blocks)."""
    rs = _lib()
    if (queries is None) == (coords is None):
        raise ValueError("give exactly one of `queries` (keys) and `coords` (their per-detector maxima, S.query_max)")
    if coords is None:
        qs = [k[None] for k in queries.reshape(-1, S.d)] if torch.is_tensor(queries) else list(queries)
        coords = S.query_max(qs)
    Q = torch.as_tensor(coords).detach().to("cpu", torch.float64).reshape(-1, S.D)
    n, D = Q.shape
    Q2, bl = augment(Q, x, split, seed, blocks)
    i1, i2 = bl[0][:-1], bl[1]
    S2, rec = rs.calibrate_two_block(S, coords=Q2, alpha=alpha, blocks=bl, u=u, rank=rank, allow_silence=allow_silence)
    # the record speaks about the n real queries
    rec.update(n=int(n), n1=int(len(i1)), block1=[int(i) for i in i1], split=None if blocks is not None else [int(split[0]), int(split[1])],
               seed=None if blocks is not None else int(seed), op="calibrate_two_block_x")
    cal = dict(S2.meta.get("calibration") or {}, op="calibrate_two_block_x", n1=int(len(i1)))
    S2 = S2.replace(meta=dict(S2.meta, calibration=cal))
    # what x changed (numbers of the calibration only): the same rule without x on the same blocks
    Qn, xv = Q.numpy(), _vec(x, D)
    b1 = Qn[i1].max(0)
    ref = rs.two_block_rule(Qn, alpha, blocks=(i1, i2), u=u)
    lit = rs.two_block_rule(Q2.numpy(), alpha, blocks=bl, u=u)["tau"]                    # the literal (uncapped) thresholds with x
    raised = xv > b1
    rec["recomb"] = {"rule": "b_j = max(block-1 baseline b1_j, x_j); block 2, quantile, guards unchanged (PROTOCOL.md section 6, Review round 1, R5)",
                     "n_detectors": int(D), "n_detectors_with_x": int(np.isfinite(xv).sum()), "n_baselines_raised_by_x": int(raised.sum()),
                     "b1_min_median_max": _quant(b1), "x_min_median_max": _quant(xv), "raise_min_median_max": _quant((xv - b1)[raised]),
                     "eta_without_x": float(ref["eta"]), "eta_with_x": float(rec["eta"]),
                     "literal_threshold_min_median_max_without_x": _quant(ref["tau"]), "literal_threshold_min_median_max_with_x": _quant(lit),
                     "n_literal_thresholds_higher_with_x": int((lit > ref["tau"]).sum()), "n_literal_thresholds_lower_with_x": int((lit < ref["tau"]).sum()),
                     "n_literal_thresholds_equal": int((lit == ref["tau"]).sum())}
    return S2, rec


# ============================================================================= a calibrated gate on the recombination rows
def decide_rows(S, log=None, kinds=KINDS, rows: LogRows | None = None, chunk: int = CHUNK, root: str | None = None) -> dict:
    """Decision of a CALIBRATED scoped gate S (role 'last') on the recombination rows of its log, with the library's arithmetic
    (ScopedLog.scores / theta / rank_key, as KeyCache.decide): arrays aligned with the rows
        fired      some detector is a candidate            n_cand    number of candidate edits
        top_edit   the edit that releases under fire 'top1' (-1: nothing fires)
        own_cand   the row's OWN edit is a candidate       own_top   it is the edit that releases
        b_cand / b_top   X-X rows: the same for the edit whose question the row borrows (False elsewhere)
        own_margin  score of the own edit's primary detector minus its threshold
    Scores: the arithmetic of own_scores (one thread, chunks of `chunk` keys in row order): for the same rows the row that attains the
    baseline x_i of an uncapped edit has r_i == x_i bit for bit, so with eta >= 0 no uncapped edit is a candidate on its own rows.
    A mechanics read-out: these rows set the baselines of rescope_x, they are not test rows.  Under the rules of the final experiments a
    gate decision may only be looked at on the development logs."""
    R_ = rows if rows is not None else log_rows(log, kinds, root)
    if int(S.T) != R_.T:
        raise ValueError(f"the gate has {int(S.T)} edits, the log {R_.T}")
    if S.role != "last":
        raise NotImplementedError("the recombination keys are last-prompt keys: role 'last' only")
    n = len(R_)
    th = S.theta().cpu()
    de = S.det_edit.cpu()
    own, par = torch.from_numpy(R_.edit), torch.from_numpy(R_.partner)
    out = {k: np.zeros(n, dtype=bool) for k in ("fired", "own_cand", "own_top", "b_cand", "b_top")}
    out.update(n_cand=np.zeros(n, dtype=np.int64), top_edit=np.full(n, -1, dtype=np.int64), own_margin=np.full(n, NEG))
    for a in range(0, n, chunk):
        sl = slice(a, min(a + chunk, n))
        with single_thread():                                           # the bits of own_scores for the same rows and chunk
            r = S.scores(R_.store.K[torch.from_numpy(R_.idx[sl])].to(S.device)).cpu()
        cand = r >= th[None, :]
        f = cand.any(1)
        top = de[torch.where(cand, S.rank_key(r.to(S.device)).cpu(), torch.full_like(r, NEG)).argmax(1)]
        ce = S._edit_any(cand.to(S.device)).cpu()
        o, p = own[sl], par[sl]
        ar = torch.arange(o.shape[0])
        out["fired"][sl] = f.numpy()
        out["n_cand"][sl] = ce.sum(1).numpy()
        out["top_edit"][sl] = torch.where(f, top, torch.full_like(top, -1)).numpy()
        out["own_cand"][sl] = ce[ar, o].numpy()
        out["own_top"][sl] = (f & (top == o)).numpy()
        out["b_cand"][sl] = (ce[ar, p.clamp(min=0)] & (p >= 0)).numpy()
        out["b_top"][sl] = (f & (p >= 0) & (top == p)).numpy()
        out["own_margin"][sl] = (r[ar, o] - th[o]).numpy()
    out["rows"] = R_
    return out
