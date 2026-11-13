"""retr.py -- scores, calibration and decisions of the two reference systems that use the logged REQUESTS instead of the log's
factors (PROTOCOL.md, section 6, amendment "Review round 1", item R4).  CPU only; no model.

    import sys; sys.path.insert(0, "$RESCOPE_ROOT/02_research/experiments/10_final/feat2/retr"); import retr

  retr   edit i is a CANDIDATE on a query iff its image cosine AND its question cosine both pass thresholds calibrated per attribute
         by the library's two-block rule on the same calibration draw at the same alpha; queries without an image release nothing;
         the RELEASED edit is the candidate with the largest sum of the two margins (score - baseline); the behaviour is the
         library's scoped replay of that edit (the same rebased value at the last prompt position: forced_gate).
  lookup edit i is a candidate iff the query's image has the md5 of the edit's image AND the two question strings share at least
         one content word (stoplist.py); released = the candidate with the most shared content words; no calibration.

A LOG is anything that names the edits: a fx LogData (fx.logs.open), a rescope EditLog (meta['uids'], meta['backbone']), the name of
a testbed log, or a pair (cache, uids).  A SEQUENCE is a position in the index of the backbone's key cache (the driver's convention);
its attributes are line j of cache2/retr/<cache>/attributes.jsonl.

    st = retr.store("qwen25vl")                    attributes + embeddings of a key cache (one object per process)
    E = retr.edits(log)                            the requests of the log's edits (image md5, question, rows of the embedding matrices)
    SI, SQ = retr.scores(log, rows)                (n, T) float64: image cosine (-inf: the sequence has no image) and question cosine
    gI, gQ = retr.own_request_scores(log)          (T,) each edit's own request on itself (= 1 up to 1e-7): the guards
    thr = retr.calibrate(SI_cal, SQ_cal, alpha, seed, guards=(gI, gQ))      two-block rule per attribute -> Thresholds (.rec: record)
    thr = retr.calibrate_log(log, cal_rows, alpha, seed)                     the same from the calibration sequences of a log
    dec = retr.decide(SI, SQ, thr, own)            decision of every sequence (form of rescope.KeyCache.decide / fx.gate.decide)
    dec = retr.retr_decide(log, rows, thr)         scores + decide for sequences of the key cache
    dec = retr.lookup_decide(log, rows)            the lookup rule, same form
    g = retr.forced_gate(S, edit)                  the library gate (ScopedLog) that releases exactly `edit` at the last prompt position
    gs = retr.gates(S, dec)                        per sequence: None (pre-edit model) or the forced gate of its released edit
    plan = retr.release_plan(dec)                  {released edit: [positions in dec]}
    g = retr.request_gate(S, log, dec)             ONE gate object in the library's gate interface (firing / respond / apply) for a decision;
    rows = retr.gate_rows(ev, g, items)            its rows through rescope.Evaluator (g.select(sid) before each sequence)
    rows = retr.default_draw(log, n=200, draw=0)   the calibration draw of the library / driver (valid REPAIR probes, splits.sample)
    ev = retr.evaluate(log, thr | None)            {part: {which: decision}} over the standard row sets (form of fx.gate.evaluate; None: lookup)
    R = retr.reference("retr", log, cal_rows, alpha, seed) | retr.reference("lookup", log)      ONE handle: R.decide(rows, which), R.evaluate(),
                                                   R.gate(S, dec), R.rec, R.fingerprint

SCORES ARE EXACT.  The stored embeddings are unit-norm float32 vectors on the 2^-24 grid (encoders.finalize); the float64 dot
product of two such vectors is exact (every partial sum is a multiple of 2^-48 below 2), so a score is a pure function of the two
keys: the same bits in every process, batch shape and BLAS.  Identical questions / images give identical scores, and the strict
conformal comparison is decided the same way everywhere (no decision band: `near` is False for every sequence).

DECISION (arrays aligned with the sequences; the vocabulary of the library's gate decisions)
    fired      an edit is released                       top_edit   the released edit (-1: none)
    n_cand     number of candidate edits                 margin     max over edits of min(image score - threshold, question score -
                                                                    threshold)  (>= 0 <=> fired); lookup: shared content words - 1
    own        the sequence's own edit in the log (-1)   own_cand / own_top   it is a candidate / it is the released edit
    near       False (see above)                         rank       the ranking value of the released edit (sum of the two margins;
                                                                    lookup: number of shared content words)
    cand       (n, T) bool: the candidate edits
    retr only: n_img_pass, n_q_pass = number of edits that pass the image / the question threshold alone
    which='steps' adds step_fired / step_top / step_near (n, 11): the decision does not depend on the key, so the 11 forwards of a
    decode all decide alike.
"""
from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import dataclass, field

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import rcommon as C  # noqa: E402
import stoplist as SL  # noqa: E402

TIE = 1e-9                       # relative slack of "the own request must pass": rescope.scoped.TIE (checked in calibrate)
N_STEPS = 11                     # full forwards of a decode that contain the last prompt position
NEG = -np.inf
_STORES, _EDITS = {}, {}


# ============================================================================= attributes + embeddings of a key cache
class Store:
    """Attribute table and embeddings of one key cache.

      seq[j]            attribute row of the sequence at cache position j (dict; fields: build_attributes.py)
      requests[family]  the edit requests of a stream family, in stream order
      EI, EQ            the embedding matrices (float32, memory-mapped; shared by the backbones); img_row[md5], q_row[question_key]
      seq_img, seq_q    (n,) row of EI (-1: no image) / of EQ of every sequence
    verify=True also checks the md5 sums of the two matrices against emb/meta.json."""

    def __init__(self, cache: str, data_dir: str | None = None, verify: bool = False):
        root = C.DATA if data_dir is None else data_dir
        self.cache, self.dir, self.emb_dir = cache, os.path.join(root, cache), os.path.join(root, "emb")
        p = os.path.join(self.dir, "attributes.jsonl")
        if not os.path.exists(p):
            raise FileNotFoundError(f"no attribute table for the key cache {cache!r} ({p}); build_attributes.py builds it")
        self.meta = C.jload(os.path.join(self.dir, "attributes_meta.json"))
        if C.file_md5(p) != self.meta["md5"]:
            raise RuntimeError(f"{p} is not the table that attributes_meta.json describes")
        rows = C.jsonl_read(p)
        n = int(self.meta["n_sequences"])
        self.seq, self.n = rows[:n], n
        if any(r["table"] != "sequence" or r["row"] != j for j, r in enumerate(self.seq)) or any(r["table"] != "edit" for r in rows[n:]):
            raise RuntimeError(f"{p}: unexpected layout")
        self.requests, self.req_of_uid = {}, {}
        for r in rows[n:]:
            fam = self.requests.setdefault(r["family"], [])
            if r["edit"] != len(fam) or r["edit_uid"] in self.req_of_uid:
                raise RuntimeError(f"{p}: edit requests out of order or a uid twice ({r['sid']})")
            self.req_of_uid[r["edit_uid"]] = (r["family"], len(fam))
            fam.append(r)
        self.emb_meta = C.jload(os.path.join(self.emb_dir, "meta.json"))
        self.img_keys = C.jsonl_read(os.path.join(self.emb_dir, "image_keys.jsonl"))
        self.q_keys = C.jsonl_read(os.path.join(self.emb_dir, "question_keys.jsonl"))
        for kind, keys in (("image", self.img_keys), ("question", self.q_keys)):
            m = self.emb_meta[kind]
            if C.file_md5(os.path.join(self.emb_dir, m["keys_file"])) != m["keys_md5"] or m["shape"][0] != len(keys):
                raise RuntimeError(f"{self.emb_dir}: the {kind} key list is not the one the embeddings were assembled for")
            if verify and C.file_md5(os.path.join(self.emb_dir, m["file"])) != m["md5"]:
                raise RuntimeError(f"{self.emb_dir}/{m['file']}: md5 differs from meta.json")
        self.EI = np.load(os.path.join(self.emb_dir, "image_emb.npy"), mmap_mode="r")
        self.EQ = np.load(os.path.join(self.emb_dir, "question_emb.npy"), mmap_mode="r")
        if list(self.EI.shape) != self.emb_meta["image"]["shape"] or list(self.EQ.shape) != self.emb_meta["question"]["shape"] \
                or self.EI.dtype != np.float32 or self.EQ.dtype != np.float32:
            raise RuntimeError(f"{self.emb_dir}: embedding matrices do not have the shapes of meta.json")
        self.img_row = {r["md5"]: i for i, r in enumerate(self.img_keys)}
        self.q_row = {r["question_key"]: i for i, r in enumerate(self.q_keys)}
        try:
            self.seq_img = np.array([-1 if r["image_md5"] is None else self.img_row[r["image_md5"]] for r in self.seq], dtype=np.int64)
            self.seq_q = np.array([self.q_row[r["question_key"]] for r in self.seq], dtype=np.int64)
        except KeyError as e:
            raise RuntimeError(f"{cache}: a sequence has no embedding ({e}); rebuild the key lists and the embeddings") from None
        self._cw = {}

    def content_words(self, q_row: int) -> frozenset:
        """Content words of the question key at row q_row of the question matrix (stoplist.content_words; cached)."""
        w = self._cw.get(q_row)
        if w is None:
            w = self._cw[q_row] = SL.content_words(self.q_keys[q_row]["question_key"])
        return w

    def rows_of(self, rows) -> np.ndarray:
        rows = np.asarray(list(rows), dtype=np.int64).reshape(-1)
        if rows.size and (rows.min() < 0 or rows.max() >= self.n):
            raise IndexError(f"sequence positions must lie in [0, {self.n})")
        return rows


def store(cache: str, data_dir: str | None = None, verify: bool = False) -> Store:
    """The Store of a key cache ('qwen25vl' | 'llava'); one object per process."""
    key = (cache, data_dir)
    if key not in _STORES:
        _STORES[key] = Store(cache, data_dir, verify)
    return _STORES[key]


# ============================================================================= the edits of a log
@dataclass
class Edits:
    cache: str                      # key cache whose attribute table holds the requests
    family: str                     # stream family of the edits
    T: int
    uids: list                      # edit uids, in the log's order
    pos: np.ndarray                 # (T,) position of every edit in the family's stream
    img: np.ndarray                 # (T,) row of the image matrix of the edit's image
    q: np.ndarray                   # (T,) row of the question matrix of the edit's question
    md5: list                       # image content of every edit
    question: list                  # question string of every edit, as asked
    prefix: bool                    # the edits are the first T requests of the family, in order (true for every log the cache serves)
    name: str | None = None
    content_words: list = field(default_factory=list)       # frozenset per edit


def _resolve(log, cache=None):
    name = None
    if isinstance(log, str):
        name = log
        d = log if os.path.isdir(log) else os.path.join(C.TESTBED, log)
        uids = [r["uid"] for r in C.jsonl_read(os.path.join(d, "edit_batch.jsonl"))]
        if cache is None:
            cache = C.jload(os.path.join(d, "metrics.json"))["config"]["backbone"]
        name = os.path.basename(d.rstrip("/"))
    elif isinstance(log, (tuple, list)) and len(log) == 2 and isinstance(log[0], str):
        cache, uids = (cache or log[0]), list(log[1])
    elif hasattr(log, "uids") and hasattr(log, "index"):                # fx.logs.LogData
        uids, name = list(log.uids), getattr(log, "name", None)
        cache = cache or (getattr(log, "reg", None) or {}).get("cache") or getattr(log, "backbone", None)
    elif hasattr(log, "meta") and isinstance(log.meta, dict) and "uids" in log.meta:      # rescope.EditLog
        uids, name = list(log.meta["uids"]), getattr(log, "name", None)
        cache = cache or log.meta.get("backbone")
    else:
        raise TypeError("log: a fx LogData, a rescope EditLog, the name of a testbed log, or (cache, uids)")
    if cache is None:
        raise ValueError("the key cache of the log is not known: pass cache=")
    return cache, uids, name


def edits(log, cache: str | None = None) -> Edits:
    """The requests of the edits of a log (see the module docstring for what a log is)."""
    if isinstance(log, Edits):
        return log
    cache, uids, name = _resolve(log, cache)
    key = (cache, tuple(uids))
    if key in _EDITS:
        return _EDITS[key]
    st = store(cache)
    miss = [u for u in uids if u not in st.req_of_uid]
    if miss:
        raise LookupError(f"{len(miss)} of {len(uids)} edits of {name or 'the log'} have no request in the attribute table of {cache!r} (e.g. {miss[:3]})")
    if len(set(uids)) != len(uids):
        raise ValueError("an edit uid occurs twice in the log")
    fams = {st.req_of_uid[u][0] for u in uids}
    if len(fams) != 1:
        raise LookupError(f"the edits of {name or 'the log'} belong to several stream families: {sorted(fams)}")
    fam = fams.pop()
    pos = np.array([st.req_of_uid[u][1] for u in uids], dtype=np.int64)
    req = [st.requests[fam][int(p)] for p in pos]
    E = Edits(cache=cache, family=fam, T=len(uids), uids=list(uids), pos=pos,
              img=np.array([st.img_row[r["image_md5"]] for r in req], dtype=np.int64),
              q=np.array([st.q_row[r["question_key"]] for r in req], dtype=np.int64),
              md5=[r["image_md5"] for r in req], question=[r["question"] for r in req],
              prefix=bool((pos == np.arange(len(uids))).all()), name=name)
    E.content_words = [st.content_words(int(i)) for i in E.q]
    _EDITS[key] = E
    return E


def own(log, rows, cache: str | None = None) -> np.ndarray:
    """(n,) the own edit of every sequence in THIS log (-1: none): edit sequences -> their edit; hard negatives -> the edit whose
    image (HN-I, HN-X) or question (HN-Q) the row uses (the definition of KeyCache.decide(uids=) and of fx LogData.own)."""
    E = edits(log, cache)
    st = store(E.cache)
    idx = {u: i for i, u in enumerate(E.uids)}
    return np.array([idx.get(st.seq[int(j)]["edit_uid"], -1) for j in st.rows_of(rows)], dtype=np.int64)


# ============================================================================= scores
def _gram(M, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(len(a), len(b)) float64 dot products of rows a and rows b of an embedding matrix: exact (module docstring)."""
    ua, ia = np.unique(np.asarray(a, dtype=np.int64), return_inverse=True)
    ub, ib = np.unique(np.asarray(b, dtype=np.int64), return_inverse=True)
    A = np.asarray(M[ua], dtype=np.float64)
    B = np.asarray(M[ub], dtype=np.float64)
    return (A @ B.T)[ia.reshape(-1)][:, ib.reshape(-1)]


def score_matrices(st: Store, img_rows, q_rows, E: Edits):
    """(SI, SQ) for queries given by their rows of the image matrix (-1: no image) and of the question matrix."""
    img_rows, q_rows = np.asarray(img_rows, dtype=np.int64), np.asarray(q_rows, dtype=np.int64)
    n = img_rows.shape[0]
    SI = np.full((n, E.T), NEG, dtype=np.float64)
    has = img_rows >= 0
    if has.any():
        SI[has] = _gram(st.EI, img_rows[has], E.img)
    SQ = _gram(st.EQ, q_rows, E.q)
    return SI, SQ


def scores(log, rows, cache: str | None = None):
    """(SI, SQ): float64 matrices [sequences x edits] of the image cosine and of the question cosine between every sequence in
    `rows` (positions in the key-cache index) and the request of every edit of the log.  A sequence without an image has image
    score -inf for every edit, so nothing can be released on it."""
    E = edits(log, cache)
    st = store(E.cache)
    rows = st.rows_of(rows)
    return score_matrices(st, st.seq_img[rows], st.seq_q[rows], E)


def own_request_scores(log, cache: str | None = None):
    """(gI, gQ): (T,) scores of every edit's own request on itself = |e|^2 of its two embeddings (1 up to 1e-7).  These are the
    guards of the calibration: the own request of an edit must keep passing both thresholds."""
    E = edits(log, cache)
    st = store(E.cache)
    eI, eQ = np.asarray(st.EI[E.img], dtype=np.float64), np.asarray(st.EQ[E.q], dtype=np.float64)
    return (eI * eI).sum(1), (eQ * eQ).sum(1)


# ============================================================================= calibration: two-block rule per attribute, conjunction
@dataclass
class Thresholds:
    """Calibrated retr thresholds.  Edit i passes attribute a on a query iff score_a >= th_a[i]; it is a candidate iff it passes
    both; candidates are ranked by (SI - b_img) + (SQ - b_q)."""
    T: int
    th_img: np.ndarray              # (T,) installed thresholds on the image score (+inf: never)
    th_q: np.ndarray                # (T,) installed thresholds on the question score
    b_img: np.ndarray               # (T,) baselines of the ranking (two-block: block-1 maxima; shared rule: 0)
    b_q: np.ndarray
    eta_img: float                  # common scale of each attribute (the conformal quantile of the block-2 margins)
    eta_q: float
    tau_img: np.ndarray             # (T,) the literal (uncapped) floating-point-safe thresholds: s >= tau <=> fl(s - b) > eta
    tau_q: np.ndarray
    capped_img: np.ndarray          # (T,) bool: the threshold was lowered so that the edit's own request keeps passing
    capped_q: np.ndarray
    rec: dict = field(default_factory=dict)

    @property
    def capped(self) -> np.ndarray:
        return self.capped_img | self.capped_q


def _rule(Q, alpha, rule, split, seed, blocks):
    C.lib_path()
    from rescope import calibrate as LC
    from rescope import scoped as LS
    if LS.TIE != TIE:
        raise RuntimeError("the library's own-key slack changed")
    if rule == "two_block":
        return LC.two_block_rule(Q, alpha, split=tuple(split), seed=int(seed), blocks=blocks)
    if rule == "shared":
        return LC.shared_rule(Q, alpha)
    raise ValueError("rule: 'two_block' | 'shared'")


def _quant(v):
    v = np.asarray(v, dtype=np.float64)
    f = v[np.isfinite(v)]
    return None if f.size == 0 else [float(x) for x in np.quantile(f, [0.0, 0.5, 1.0])]


def calibrate(SI, SQ, alpha=0.05, seed: int = 0, split=(1, 1), blocks=None, guards=None, allow_silence: bool = False, rule: str = "two_block") -> Thresholds:
    """The library's two-block rule applied PER ATTRIBUTE on the same calibration draw; the release rule is the conjunction.

    SI, SQ   (n, T) score matrices of the n benign calibration queries (rows in the order of the draw; scores()).
    alpha    miscoverage of EACH attribute (float / Fraction / 'p/q': the rank is exact integer arithmetic, as in the library).
    seed, split, blocks   the split of the n queries into block 1 and block 2: rescope.split_blocks(n, split, seed), the SAME split
             for both attributes (and the split every other gate of the driver uses for that draw); blocks = (idx1, idx2) overrides.
    guards   (gI, gQ): (T,) scores of every edit's own request on itself (own_request_scores).  An edit whose own request the
             literal rule would silence on an attribute gets that threshold capped at (1 - 1e-9) x its own score (the library's
             own-key rule) and is counted; guards=None or allow_silence=True: the literal (uncapped) rule.
    rule     'two_block' (registered in R4) | 'shared' (one conformal threshold per attribute, no baselines: not registered for
             retr, provided for symmetry with the cosine gate).

    Per attribute a (library: rescope.two_block_rule): baseline b_a[i] = largest score of edit i over block 1; margin of a query
    = max_i (score_a[i] - b_a[i]); eta_a = the k-th smallest block-2 margin, k = ceil((1 - alpha)(n2 + 1)); edit i PASSES
    attribute a iff score_a[i] - b_a[i] > eta_a (strict; stored as the floating-point-safe threshold tau_a[i]).  Queries without
    an image have image score -inf: they never raise a baseline and their image margin is -inf.
    BOUND.  For a fresh benign query exchangeable with the calibration queries, P(some edit passes attribute a) <=
    floor(alpha (n2 + 1)) / (n2 + 1) <= alpha for each attribute (the calibration theorem, uncapped rule).  A release needs ONE edit
    that passes BOTH attributes, an event contained in each of the two events, hence
        P(retr releases an edit) <= min(bound_img, bound_q) <= alpha      (rec['bound_false_firing']).
    CAPS.  A capped threshold is below the conformal level of ITS attribute; capped edits are counted (rec['n_edits_capped'],
    rec['capped_edits'], per attribute rec['n_edits_capped_image' / '_question' / '_both']).  The conjunction keeps the bound as long
    as one attribute has no capped edit: every release still needs that attribute's LITERAL event.  rec['guarantee'] says which case
    holds, rec['bound_false_firing_with_caps'] is the bound that holds with the caps (None: void for edits capped on both)."""
    SI, SQ = np.asarray(SI, dtype=np.float64), np.asarray(SQ, dtype=np.float64)
    if SI.shape != SQ.shape or SI.ndim != 2:
        raise ValueError("SI and SQ must be (n, T) matrices of the same shape")
    n, T = SI.shape
    if np.isnan(SI).any() or np.isnan(SQ).any() or np.isinf(SQ).any() or np.isposinf(SI).any():
        raise ValueError("scores must be finite (image scores may be -inf for queries without an image)")
    if rule == "two_block":
        C.lib_path()
        from rescope.calibrate import split_blocks
        i1 = split_blocks(n, tuple(split), int(seed))[0] if blocks is None else np.asarray(blocks[0], dtype=np.int64)
        if not np.isfinite(SI[i1]).any():
            raise ValueError("block 1 holds no query with an image: the image baselines are undefined")
    out, recs = {}, {}
    for a, Q in (("img", SI), ("q", SQ)):
        r = out[a] = _rule(Q, alpha, rule, split, seed, blocks)
        sc2 = np.asarray(r["scores2"], dtype=np.float64)
        recs[a] = {"eta": float(r["eta"]), "eta_infinite": bool(np.isposinf(r["eta"])), "rank": int(r["rank"]), "n1": int(r["n1"]), "n2": int(r["n2"]),
                   "n_allowed_above": int(r["n_allowed"]), "bound_false_firing": float(r["bound"]),
                   "n_block2_queries_with_finite_margin": int(np.isfinite(sc2).sum()),
                   "baseline_min_median_max": _quant(r["b"]) if r["has_baseline"] else None, "literal_threshold_min_median_max": _quant(r["tau"])}
    if rule == "two_block" and not (np.array_equal(out["img"]["idx1"], out["q"]["idx1"]) and np.array_equal(out["img"]["idx2"], out["q"]["idx2"])):
        raise RuntimeError("the two attributes were not calibrated on the same blocks")
    tau = {a: np.asarray(out[a]["tau"], dtype=np.float64).copy() for a in out}
    th = {a: tau[a].copy() for a in out}
    capped = {a: np.zeros(T, dtype=bool) for a in out}
    sil = np.zeros(T, dtype=bool)
    if guards is not None:
        g = {"img": np.asarray(guards[0], dtype=np.float64), "q": np.asarray(guards[1], dtype=np.float64)}
        if g["img"].shape != (T,) or g["q"].shape != (T,) or not (np.isfinite(g["img"]).all() and np.isfinite(g["q"]).all()):
            raise ValueError("guards = (gI, gQ): two finite (T,) vectors")
        for a in out:
            s = g[a] < tau[a]                                           # the literal rule silences the edit's own request on this attribute
            sil |= s
            if not allow_silence:
                cap = g[a] - TIE * np.abs(g[a])
                th[a] = np.where(s, np.minimum(tau[a], cap), tau[a])
                capped[a] = s
    al = out["q"]["alpha"]
    cap_any, cap_both = capped["img"] | capped["q"], capped["img"] & capped["q"]
    bI, bQ = float(recs["img"]["bound_false_firing"]), float(recs["q"]["bound_false_firing"])
    if not cap_any.any():
        bound_caps, guar = min(bI, bQ), "exact"
    elif not capped["img"].any():                                       # a release needs the LITERAL image event, whatever the question caps
        bound_caps, guar = bI, (f"holds through the image attribute, on which no edit is capped: P(release) <= {bI:.6g}; "
                                f"{int(capped['q'].sum())} edit(s) are capped on the question attribute (their own question keeps passing)")
    elif not capped["q"].any():
        bound_caps, guar = bQ, (f"holds through the question attribute, on which no edit is capped: P(release) <= {bQ:.6g}; "
                                f"{int(capped['img'].sum())} edit(s) are capped on the image attribute (their own image keeps passing)")
    elif not cap_both.any():                                            # every release passes a literal threshold of one of the two attributes
        bound_caps, guar = min(1.0, bI + bQ), (f"no edit is capped on both attributes: P(release) <= {bI:.6g} + {bQ:.6g} (union of the two literal events); "
                                               f"capped: {int(capped['img'].sum())} image, {int(capped['q'].sum())} question")
    else:
        bound_caps, guar = None, (f"void for {int(cap_both.sum())} edit(s) capped on both attributes (for the other edits P(release) <= {bI:.6g} + {bQ:.6g})")
    rec = {"op": "retr.calibrate", "rule": rule, "alpha": float(al), "alpha_exact": f"{al.numerator}/{al.denominator}", "n": int(n), "T": int(T),
           "split": None if (blocks is not None or rule != "two_block") else [int(split[0]), int(split[1])],
           "seed": None if (blocks is not None or rule != "two_block") else int(seed),
           "block1": [int(i) for i in out["q"]["idx1"]], "block2": [int(i) for i in out["q"]["idx2"]],
           "n_queries_with_image": int(np.isfinite(SI).any(axis=1).sum()),
           "n_block1_queries_with_image": int(np.isfinite(SI[out["q"]["idx1"]]).any(axis=1).sum()) if len(out["q"]["idx1"]) else 0,
           "n_block2_queries_with_image": int(np.isfinite(SI[out["q"]["idx2"]]).any(axis=1).sum()),
           "image": recs["img"], "question": recs["q"], "strict": True,
           "bound_false_firing": min(bI, bQ),
           "bound": "P(a fresh benign query releases an edit) <= min over the two attributes of floor(alpha (n2 + 1)) / (n2 + 1) (conjunction; uncapped rule)",
           "bound_false_firing_with_caps": bound_caps,
           "guards": "the own request of every edit" if guards is not None else None, "allow_silence": bool(allow_silence),
           "n_own_requests_silenced_by_literal_rule": int(sil.sum()), "own_requests_silenced_by_literal_rule": [int(i) for i in np.nonzero(sil)[0]],
           "capped": bool(cap_any.any()), "n_edits_capped": int(cap_any.sum()), "capped_edits": [int(i) for i in np.nonzero(cap_any)[0]],
           "n_edits_capped_image": int(capped["img"].sum()), "n_edits_capped_question": int(capped["q"].sum()), "n_edits_capped_both": int(cap_both.sum()),
           "guarantee": guar,
           "top1": "largest (image score - image baseline) + (question score - question baseline); an exact tie goes to the lowest edit index"}
    if np.isneginf(out["img"]["eta"]):                                  # so few block-2 queries have an image that the conformal rank falls on a text query
        rec["note_image_scale"] = ("eta_img = -inf: at most floor(alpha (n2 + 1)) block-2 queries have an image, so the image attribute passes EVERY query "
                                   "that has an image (the rule as defined; the bound still holds for the pool the queries were drawn from)")
    thr = Thresholds(T=T, th_img=th["img"], th_q=th["q"], b_img=np.asarray(out["img"]["b"], dtype=np.float64), b_q=np.asarray(out["q"]["b"], dtype=np.float64),
                     eta_img=float(out["img"]["eta"]), eta_q=float(out["q"]["eta"]), tau_img=tau["img"], tau_q=tau["q"],
                     capped_img=capped["img"], capped_q=capped["q"], rec=rec)
    if guards is not None:
        rec["n_own_requests_passing"] = int(((g["img"] >= th["img"]) & (g["q"] >= th["q"])).sum())
    h = hashlib.sha1()
    for v in (th["img"], th["q"], thr.b_img, thr.b_q):
        h.update(np.ascontiguousarray(v, dtype=np.float64).tobytes())
    rec["fingerprint"] = h.hexdigest()[:12]                             # of the installed thresholds and ranking baselines
    return thr


def calibrate_log(log, cal_rows, alpha=0.05, seed: int = 0, split=(1, 1), allow_silence: bool = False, rule: str = "two_block", cache: str | None = None) -> Thresholds:
    """calibrate() for a log and its calibration sequences (positions in the key-cache index, IN THE ORDER OF THE DRAW: the split
    into blocks depends on it; fx.draws.calibration(L, n, draw).rows with seed = draw is the driver's convention).  Guards = the own
    request of every edit.  The record also says how many own requests are released by their own edit when the WHOLE log competes."""
    E = edits(log, cache)
    SI, SQ = scores(E, cal_rows)
    gI, gQ = own_request_scores(E)
    thr = calibrate(SI, SQ, alpha, seed, split, None, (gI, gQ), allow_silence, rule)
    st = store(E.cache)
    d = decide(*score_matrices(st, E.img, E.q, E), thr, own=np.arange(E.T))
    thr.rec.update(log=E.name, cache=E.cache, family=E.family, n_own_requests_passing=int(d["own_cand"].sum()),
                   n_own_requests_released_by_own_edit=int(d["own_top"].sum()),
                   own_requests_released_by_another_edit=[int(i) for i in np.nonzero(d["fired"] & ~d["own_top"])[0]])
    return thr


# ============================================================================= decisions
def _finish(dec: dict, own_, rows, which: str) -> dict:
    n = dec["fired"].shape[0]
    o = np.full(n, -1, dtype=np.int64) if own_ is None else np.asarray(own_, dtype=np.int64)
    if o.shape != (n,):
        raise ValueError("own: one edit index (or -1) per sequence")
    has = o >= 0
    cand = dec.pop("_cand")
    oc = np.zeros(n, dtype=bool)
    oc[has] = cand[np.nonzero(has)[0], o[has]]
    dec.update(cand=cand, own=o, own_cand=oc, own_top=dec["fired"] & has & (dec["top_edit"] == o), near=np.zeros(n, dtype=bool),
               rows=None if rows is None else [int(j) for j in rows], which=which, resolves_edits=True)
    if which == "steps":                                                # the decision does not depend on the key of a forward
        dec.update(step_fired=np.repeat(dec["fired"][:, None], N_STEPS, 1), step_top=np.repeat(dec["top_edit"][:, None], N_STEPS, 1),
                   step_near=np.zeros((n, N_STEPS), dtype=bool))
    return dec


def decide(SI, SQ, thr: Thresholds, own=None, rows=None, which: str = "tf") -> dict:
    """Apply the two per-attribute thresholds: per sequence the candidate edits (both attributes pass) and the RELEASED edit = the
    candidate with the largest sum of the two margins (SI - b_img) + (SQ - b_q); an exact tie goes to the lowest edit index.
    -> decision dict (module docstring).  own: (n,) own edit of every sequence (-1 / None: none); rows, which: labels only."""
    SI, SQ = np.asarray(SI, dtype=np.float64), np.asarray(SQ, dtype=np.float64)
    if SI.shape != SQ.shape or SI.ndim != 2 or SI.shape[1] != thr.T:
        raise ValueError(f"SI and SQ must be (n, {thr.T}) matrices")
    pI, pQ = SI >= thr.th_img[None, :], SQ >= thr.th_q[None, :]
    cand = pI & pQ
    with np.errstate(invalid="ignore"):
        rank = (SI - thr.b_img[None, :]) + (SQ - thr.b_q[None, :])
        slack = np.minimum(SI - thr.th_img[None, :], SQ - thr.th_q[None, :])
    slack = np.where(np.isnan(slack), NEG, slack)
    rk = np.where(cand, rank, NEG)
    top = rk.argmax(1) if thr.T else np.zeros(SI.shape[0], dtype=np.int64)        # first maximum = lowest edit index
    fired = cand.any(1)
    ar = np.arange(SI.shape[0])
    dec = {"fired": fired, "top_edit": np.where(fired, top, -1).astype(np.int64), "n_cand": cand.sum(1).astype(np.int64),
           "margin": slack.max(1) if thr.T else np.full(SI.shape[0], NEG), "rank": np.where(fired, rk[ar, top], NEG),
           "n_img_pass": pI.sum(1).astype(np.int64), "n_q_pass": pQ.sum(1).astype(np.int64), "_cand": cand}
    return _finish(dec, own, rows, which)


def retr_decide(log, rows, thr: Thresholds, which: str = "tf", cache: str | None = None) -> dict:
    """The retr decision of every sequence in `rows` (positions in the key-cache index) for a log and calibrated thresholds."""
    E = edits(log, cache)
    if thr.T != E.T:
        raise ValueError(f"thresholds of {thr.T} edits for a log of {E.T}")
    SI, SQ = scores(E, rows)
    return decide(SI, SQ, thr, own(E, rows), rows, which)


def lookup_shared(log, rows, cache: str | None = None) -> np.ndarray:
    """(n, T) int32: number of content words that the sequence's question shares with the edit's question, where the sequence has
    an image with the md5 of the edit's image; 0 elsewhere.  Edit i is a lookup candidate iff the entry is > 0."""
    E = edits(log, cache)
    st = store(E.cache)
    rows = st.rows_of(rows)
    by_img = {}
    for i, r in enumerate(E.img):
        by_img.setdefault(int(r), []).append(i)
    out = np.zeros((rows.shape[0], E.T), dtype=np.int32)
    for k, j in enumerate(rows):
        ir = int(st.seq_img[j])
        if ir < 0 or ir not in by_img:                                  # no image, or no edit with that image content
            continue
        w = st.content_words(int(st.seq_q[j]))
        if not w:
            continue
        for i in by_img[ir]:
            out[k, i] = len(w & E.content_words[i])
    return out


def lookup_decide(log, rows, which: str = "tf", cache: str | None = None) -> dict:
    """The lookup rule (no calibration): candidate = same image md5 AND at least one shared content word; released = the candidate
    with the most shared content words (tie: lowest edit index).  Same form as decide(); margin = shared words of the released
    edit - 1 (-1 where nothing is released); rank = that number of shared words."""
    E = edits(log, cache)
    sh = lookup_shared(E, rows)
    cand = sh > 0
    top = sh.argmax(1) if E.T else np.zeros(sh.shape[0], dtype=np.int64)
    fired = cand.any(1)
    best = sh.max(1) if E.T else np.zeros(sh.shape[0], dtype=np.int32)
    dec = {"fired": fired, "top_edit": np.where(fired, top, -1).astype(np.int64), "n_cand": cand.sum(1).astype(np.int64),
           "margin": best.astype(np.float64) - 1.0, "rank": np.where(fired, best, 0).astype(np.float64), "_cand": cand}
    return _finish(dec, own(E, rows), rows, which)


def lookup_record(log, cache: str | None = None) -> dict:
    """What determines the lookup rule on a log: the stop list's fingerprint, edits that can never be released (no content word),
    edits that share their image content with another edit."""
    E = edits(log, cache)
    cnt = {}
    for m in E.md5:
        cnt[m] = cnt.get(m, 0) + 1
    return {"op": "retr.lookup", "log": E.name, "T": E.T, "stop_list_sha1": SL.stop_list_sha1(), "stop_list_fixed_at": SL.FIXED_AT, "n_stop_words": len(SL.STOP),
            "rule": "same image md5 AND >= 1 shared content word; top-1 = most shared content words, tie -> lowest edit index; no calibration",
            "n_edits_without_content_word": sum(not w for w in E.content_words), "edits_without_content_word": [i for i, w in enumerate(E.content_words) if not w],
            "n_edits_sharing_their_image_with_another_edit": sum(cnt[m] > 1 for m in E.md5)}


# ============================================================================= the library gate of a decision, the default draw
def forced_gate(S, edit: int):
    """The library gate that RELEASES exactly `edit` at the last prompt position of whatever sequence it is run on: the scoped log S
    (a role-'last', top-1 gate of the log with the values to release, e.g. the driver's un-thresholded base gate) with threshold
    -inf for the edit's primary detector and +inf for every other detector.  Running the sequences on which retr / lookup release
    edit j through this gate (rescope.Evaluator / Hook, teacher forcing or a decode) IS the behaviour of the reference system there:
    module output W_0 k + V_j at the last prompt position, the pre-edit model everywhere else; a sequence on which nothing is
    released is the pre-edit model and needs no forward.  The released value is S.values[edit], so the outcome is the one of any
    scoped gate of the log that releases the same edit on the same sequence (same memo signature in the driver)."""
    import torch
    if S.role != "last" or S.fire != "top1":
        raise ValueError("forced_gate needs a role-'last', top-1 scoped log")
    edit = int(edit)
    if not 0 <= edit < S.T:
        raise IndexError(f"edit {edit} of a log with {S.T} edits")
    tau = torch.full_like(S.det_tau, float("inf"))
    tau[edit] = -float("inf")
    return S.replace(det_tau=tau, eta=1.0, det_b=None, det_u=None, name=f"release-{edit}:{S.name}", meta=dict(S.meta, forced_release=edit))


def gates(S, dec: dict) -> list:
    """Per sequence of a decision (decide / retr_decide / lookup_decide) the library gate that reproduces it: None where nothing is
    released (the sequence IS the pre-edit model: no forward is needed), else forced_gate(S, released edit).  One object per
    distinct edit (sequences that release the same edit share it)."""
    made = {}
    out = []
    for f, e in zip(dec["fired"], dec["top_edit"]):
        if not bool(f):
            out.append(None)
            continue
        e = int(e)
        if e not in made:
            made[e] = forced_gate(S, e)
        out.append(made[e])
    return out


def release_plan(dec: dict) -> dict:
    """{released edit: [positions i in the decision's arrays]} of the sequences on which an edit is released (the forwards a
    behavioural evaluation has to run, grouped by gate)."""
    plan = {}
    for i, (f, e) in enumerate(zip(dec["fired"], dec["top_edit"])):
        if bool(f):
            plan.setdefault(int(e), []).append(i)
    return plan


class RequestGate:
    """retr / lookup as a GATE of the library: the interface that baselines.py documents (firing / respond / apply, role, W0), so
    that rescope.Hook and rescope.Evaluator treat it like any other gate (it is not a ScopedLog: the library then takes the plain
    Evaluator path, one forward per item, and never decides it from a cached key).

    The decision of a reference system depends on the QUERY (image content, question string), not on the module input, so the gate
    is told which sequence the next forward(s) belong to:

        g = retr.RequestGate(S, dec, sids)       S: a role-'last', top-1, REPL scoped log of the log on the model's device (it supplies
                                                 the released values and W0, e.g. feature_gate(log, 'ncoord', W0=H.W0)); dec: a decision
                                                 (retr_decide / lookup_decide / decide); sids: one id per sequence of dec
        g = retr.request_gate(S, log, dec)       the same with sids = the library's item sids of dec['rows'] ('edit:rel:17', ...)
        g.select(sid)                            before the forward(s) of that sequence
        rows = retr.gate_rows(ev, g, items)      for it in items: g.select(it['sid']); ev.tf_rows(g, [it])       (fn='generate': decode)

    At the last prompt position of the selected sequence firing() returns the library's Firing: edit = the released edit, cand = the
    candidate edits of the decision (so the row flags of the Evaluator are the decision's: fired, top_last = top_edit, own_fired_last =
    own_cand, own_released_last = own_top, n_cand_edits = n_cand); respond() is S's (module output W_0 k + V_edit, one cast).  Where
    the decision releases nothing, firing() returns None and the hook leaves the module output untouched: the pre-edit model, bit for
    bit.  B = 1 (the evaluation's batch size)."""

    fire, role, precision = "top1", "last", "table"

    def __init__(self, S, dec: dict, sids, name: str = "request-gate", _shared=None):
        if S.role != "last" or S.fire != "top1" or S.response != "repl":
            raise ValueError("RequestGate needs a role-'last', top-1, REPL scoped log (released values)")
        self.S, self.name = S, name
        if _shared is not None:
            self._sh = _shared
            return
        sids = list(sids)
        n = len(dec["fired"])
        if len(sids) != n or len(set(sids)) != n:
            raise ValueError("sids: one distinct id per sequence of the decision")
        cand = dec.get("cand")
        table = {}
        for i, sid in enumerate(sids):
            e = int(dec["top_edit"][i]) if bool(dec["fired"][i]) else -1
            if e >= S.T:
                raise IndexError(f"released edit {e} of a log with {S.T} edits")
            c = [] if e < 0 else ([e] if cand is None else [int(x) for x in np.nonzero(np.asarray(cand[i]))[0]])
            if e >= 0 and e not in c:
                raise ValueError("the released edit must be one of the candidates")
            table[sid] = (e, c)
        self._sh = {"table": table, "cur": None, "copies": {}}

    # the attributes the hook and the evaluation read
    response = property(lambda self: self.S.response)
    W0 = property(lambda self: self.S.W0)
    det_edit = property(lambda self: self.S.det_edit)
    values = property(lambda self: self.S.values)
    T = property(lambda self: self.S.T)
    D = property(lambda self: self.S.D)
    device = property(lambda self: self.S.device)

    def to(self, device):
        """The gate with its scoped log on `device` (one copy per device is kept; the selection is shared by the copies)."""
        import torch
        device = torch.device(device)
        if device == self.S.device:
            return self
        key = str(device)
        if key not in self._sh["copies"]:
            self._sh["copies"][key] = RequestGate(self.S.to(device), None, None, self.name, _shared=self._sh)
        return self._sh["copies"][key]

    def select(self, sid):
        """The sequence of the next forward(s); raises KeyError for a sequence the decision does not hold."""
        if sid not in self._sh["table"]:
            raise KeyError(f"no decision for sequence {sid!r}")
        self._sh["cur"] = sid
        return self

    def decision(self, sid=None):
        """(released edit or -1, candidate edits) of a sequence (default: the selected one)."""
        sid = self._sh["cur"] if sid is None else sid
        if sid is None:
            raise RuntimeError("RequestGate.select(sid) must be called before a forward")
        return self._sh["table"][sid]

    def _edit_any(self, cand):
        return self.S._edit_any(cand)

    def firing(self, x, lp=None, pos0=0, state=None):
        import torch
        from rescope.scoped import INF, Firing
        e, c = self.decision()
        B, Tn = (x.shape[0], x.shape[1]) if x.dim() == 3 else (1, x.shape[0])
        if B != 1:
            raise ValueError("RequestGate decides one sequence per forward (B = 1)")
        if e < 0:
            return None
        idx = self.S.eligible(B, Tn, lp, pos0)                          # role 'last': the last prompt position, if this call holds it
        if idx is None or idx.numel() == 0:
            return None
        n, dev = idx.numel(), self.S.device
        cand = torch.zeros(n, self.S.D, dtype=torch.bool, device=dev)
        cand[:, torch.tensor(c, dtype=torch.long, device=dev)] = True     # primary detectors of the candidate edits
        r = torch.full((n, self.S.D), -INF, dtype=torch.float64, device=dev)
        r[cand] = 0.0
        r[:, e] = 1.0                                                   # ranking surrogate: the released edit first
        top = torch.full((n,), e, dtype=torch.long, device=dev)
        return Firing(idx=idx, row=idx // Tn, pos=idx % Tn + pos0, cand=cand, r=r, top=top, edit=self.S.det_edit[top])

    def respond(self, x, out, f):
        return self.S.respond(x, out, f)

    def apply(self, x, out, lp=None, pos0=0):
        f = self.firing(x, lp, pos0)
        return out if f is None else self.respond(x, out, f)


def request_gate(S, log, dec: dict, name: str | None = None, cache: str | None = None) -> RequestGate:
    """RequestGate for a decision over sequences of the key cache (dec['rows']): the sequence ids are the library's item sids
    (index field lib_sid: 'edit:rel:17', 'probe:image:3', a hard-negative id), i.e. it['sid'] of rescope.Evaluator's items."""
    E = edits(log, cache)
    if dec.get("rows") is None:
        raise ValueError("the decision carries no cache positions (rows)")
    idx = cache_index(E.cache)
    return RequestGate(S, dec, [idx[int(j)]["lib_sid"] for j in dec["rows"]], name or f"request-gate:{E.name}")


def gate_rows(ev, method, items, fn: str = "tf", lp: str = "tf"):
    """Rows of rescope.Evaluator, one item at a time, for a RequestGate or a dict {label: method} that holds RequestGates (other
    methods of the dict pass through unchanged): every RequestGate is told the item (select(it['sid'])) before its forward(s).
    fn='tf': teacher forcing (ev.tf_rows, lp as there); fn='generate': free generation (ev.generate; the gate decides alike in
    every forward of the decode).  -> list of rows, or {label: rows} for a dict."""
    many = isinstance(method, dict)
    ms = method if many else {"_": method}
    out = {k: [] for k in ms}
    for it in items:
        for m in ms.values():
            if isinstance(m, RequestGate):
                m.select(it["sid"])
        r = ev.tf_rows(ms, [it], lp=lp) if fn == "tf" else ev.generate(ms, [it])
        for k in ms:
            out[k] += r[k]
    return out if many else out["_"]


def default_draw(log, n: int = 200, draw: int = 0, cache: str | None = None) -> list:
    """The calibration draw of the library's usage and of the driver: rescope.splits.sample(valid REPAIR probe rows of the log's
    family, n, seed=draw), in index order.  For n = 200 this IS fx.draws.calibration(L, 200, draw).rows; the driver's draws of other
    sizes are nested around it and its other sources differ: take those from fx.draws.  Needs the key cache's index, not its keys."""
    C.lib_path()
    from rescope import splits
    E = edits(log, cache)
    idx = cache_index(E.cache)
    pool = [j for j, e in enumerate(idx) if e["kind"] == "probe" and e["family"] == E.family and e.get("pool") == "repair" and bool(e.get("y0_valid"))]
    return splits.sample(pool, int(n), int(draw))


_INDEX = {}


def cache_index(cache: str) -> list:
    """The index rows of a key cache (10_final/cache/<cache>/index.jsonl), read once."""
    if cache not in _INDEX:
        _INDEX[cache] = C.jsonl_read(os.path.join(C.CACHE, cache, "index.jsonl"))
    return _INDEX[cache]


def row_sets(log, cache: str | None = None) -> dict:
    """The standard row sets of a log, as fx.gate.row_sets: 'probe' (valid probe rows of every pool), 'edit' (Rel, T-Gen, I-Gen of
    every edit, metric-major), 'hn' (valid hard negatives of the log's edits; HN-X only with both edits in the log), 'benign', and
    the pools 'heldout_disjoint' / 'repair' (valid rows).  From the cache index; for logs whose edits are a prefix of a family."""
    E = edits(log, cache)
    if not E.prefix:
        raise LookupError("row_sets serves logs whose edits are the first T edits of a cached family")
    idx, T, fam = cache_index(E.cache), E.T, E.family
    out = {"probe": [], "edit": [], "hn": [], "benign": [], "heldout_disjoint": [], "repair": []}
    ed = {k: {} for k in C.EDIT_KINDS}
    for j, e in enumerate(idx):
        k = e["kind"]
        if k == "benign":
            out["benign"].append(j)
        if e["family"] != fam:
            continue
        if k == "probe" and bool(e.get("y0_valid")):
            out["probe"].append(j)
            if e.get("disjoint"):
                out["heldout_disjoint"].append(j)
            if e.get("pool") == "repair":
                out["repair"].append(j)
        elif k in C.EDIT_KINDS and int(e["edit"]) < T:
            ed[k].setdefault(int(e["edit"]), j)
        elif k in C.HN_KINDS and bool(e.get("y0_valid")) and int(e["edit"]) < T and (e.get("edit_b") is None or int(e["edit_b"]) < T):
            out["hn"].append(j)
    out["edit"] = [ed[k][i] for k in C.EDIT_KINDS for i in range(T)]
    return out


def evaluate(log, thr: Thresholds | None = None, positions=("tf",), parts=("probe", "edit", "hn", "benign"), rows: dict | None = None, cache: str | None = None) -> dict:
    """Gate-level pass of a reference system over the standard row sets of a log, in the form of fx.gate.evaluate:
    {part: {which: decision}} (the benign pool always under 'decode', as there), so that fx.metrics.gate_stats(L, ev, which) gives
    the numbers of every other gate.  thr: calibrated thresholds -> retr; None -> lookup.  rows: {part: positions} (default:
    row_sets(log), which equals fx.gate.row_sets(L)).  The decision of a sequence does not depend on the position `which`."""
    E = edits(log, cache)
    sets = row_sets(E) if rows is None else rows
    out = {}
    for p in parts:
        if p not in sets:
            continue
        out[p] = {}
        for w in (("decode",) if p == "benign" else tuple(positions)):
            out[p][w] = retr_decide(E, sets[p], thr, w) if thr is not None else lookup_decide(E, sets[p], w)
    return out


# ============================================================================= one handle per reference system and log
class Reference:
    """A reference system on a log as ONE object (what a driver can carry as the `obj` of a method):

        R = retr.reference("retr", log, cal_rows, alpha=0.05, seed=0)      calibrated retrieval reference (two-block rule per attribute)
        R = retr.reference("lookup", log)                                    the lookup rule (no calibration)
        R.kind ('retr' | 'lookup'), R.E (the log's Edits), R.thr (Thresholds | None), R.rec (record, JSON), R.fingerprint
        R.decide(rows, which="tf") -> decision          R.evaluate(positions=("tf",)) -> {part: {which: decision}}
        R.gate(S, dec) -> RequestGate                   R.forced(S, edit) -> the forced library gate"""

    def __init__(self, kind: str, log, thr: Thresholds | None = None, cache: str | None = None):
        if kind not in ("retr", "lookup"):
            raise ValueError("kind: 'retr' | 'lookup'")
        if (kind == "retr") != (thr is not None):
            raise ValueError("retr needs calibrated thresholds; lookup takes none")
        self.kind, self.E, self.thr = kind, edits(log, cache), thr
        self.rec = dict(thr.rec) if thr is not None else lookup_record(self.E)
        self.fingerprint = self.rec.get("fingerprint") or hashlib.sha1(f"lookup|{SL.stop_list_sha1()}|{'|'.join(self.E.uids)}".encode()).hexdigest()[:12]
        self.name = f"{kind}:{self.E.name}"

    def decide(self, rows, which: str = "tf") -> dict:
        return retr_decide(self.E, rows, self.thr, which) if self.kind == "retr" else lookup_decide(self.E, rows, which)

    def evaluate(self, positions=("tf",), parts=("probe", "edit", "hn", "benign"), rows: dict | None = None) -> dict:
        return evaluate(self.E, self.thr, positions, parts, rows)

    def gate(self, S, dec: dict) -> RequestGate:
        return request_gate(S, self.E, dec, name=f"{self.kind}-gate:{self.E.name}")

    @staticmethod
    def forced(S, edit: int):
        return forced_gate(S, edit)

    def __repr__(self):
        return f"Reference({self.kind!r}, log={self.E.name}, T={self.E.T}, fingerprint={self.fingerprint})"


def reference(kind: str, log, cal_rows=None, alpha=0.05, seed: int = 0, split=(1, 1), allow_silence: bool = False, cache: str | None = None) -> Reference:
    """Reference('retr', ...) calibrated on the sequences cal_rows (order of the draw; seed = seed of the two-block split = the draw
    index in the driver), or Reference('lookup', ...)."""
    if kind == "lookup":
        return Reference("lookup", log, None, cache)
    if cal_rows is None:
        raise ValueError("retr needs the calibration sequences (cal_rows)")
    return Reference("retr", log, calibrate_log(log, cal_rows, alpha, seed, split, allow_silence, cache=cache), cache)
