"""fx.draws: seeded calibration draws and reported-complaint sets.

CALIBRATION DRAWS (a draw = a list of sequences of the key cache + which of their keys the gate sees)
  calibration(L, n=200, draw=0)                           n benign queries from the VALID rows of the REPAIR pool of the log's probes
                                                          (teacher-forced key).  For n = 200 this is rescope.splits.sample(pool, 200,
                                                          seed=draw), the draw of docs/USAGE.md; the two-block split uses seed=draw too.
  calibration(L, n, draw, which="all")                    the same queries with EVERY key they can present to a role-'last' gate (teacher-
                                                          forced key, the decode's token, the 11 forwards of a decode): what the library asks
                                                          for if the conformal bound is to cover decode metrics (its rules 3 and 6)
  calibration(L, n, draw, source="benign", which=...)     from the label-free benign pool of 10_final/cache (a shifted source):
                                                          which='tf' (default) = the rows of the answer subset with a valid pre-edit
                                                          answer, teacher-forced key (the forward path of the evaluation);
                                                          which='decode' / 'steps' = all 3000 rows, prompt-only key / the 11 keys of a decode.
  calibration(L, n, draw, contamination=p)                round(p n) of the n queries are in-scope queries: text / image rephrase prompts
                                                          (T-Gen, I-Gen sequences) of edit half A = the first T // 2 edits; they replace the
                                                          LAST queries of the benign draw.  Report coverage on half B (fx.draws.half_b).
NESTED IN n.  For a fixed draw index and source the draws are nested: one seeded ranking of the pool (the 200 rows of the default
draw in a seeded order, then the other rows in a seeded order); the draw of size n is its first n rows, sorted in pool order.
The contaminating queries are nested in their number as well (a prefix of one seeded permutation).

DISJOINT PARTS.  partition(L, sizes, draw): consecutive slices of the same ranking (C0 / S / C1 of the detector-repair experiments),
disjoint as ROWS; null_sample(L, 100, draw): the benign sample of ReScope-N, separated from the calibration draw BY QUERY;
explicit(L, rows): any given sample.
QUERIES AND ROWS.  The probe pools repeat a few queries (two rows with the same question string and the same image content: one
query, one key; REPAIR pool of the Qwen stream: 489 valid rows, 474 queries).  query_ids(L, rows) = the key cache's content id.
A sample that CHANGES detectors must share no query with the calibration sample (the calibration theorem needs detectors that do
not depend on the calibration queries; protocol amendment of 2026-11-11 09:03): null_sample guarantees it, fx.methods checks it.

REPORTED COMPLAINTS (development logs; paired with 08_rescope/p12_linear_streams)
  complaint_stream(L, seed)       the nested stream of P12 (p12_common.build_stream): harness.sample_complaints(H, 100, seed) on the D1
                                  complaints of the REPAIR pool (50 image + 50 text), interleaved text / image, then the remaining REPAIR
                                  complaints in a seeded order -- recomputed here from the log's probe records, no model
  complaints(L, n, seed)          its first n complaints (n = 'ALL': all), as sequences of the key cache
  verify_p12(L)                   every P12 record of the log lists exactly these complaints (reported_groups)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

import fx

N_DEFAULT = 200
N_DRAWS = 20
ALPHAS = (0.01, 0.02, 0.05, 0.10, 0.20)
CONTAMINATION = (0.0, 0.01, 0.05, 0.20)
P12_SEEDS = (0, 1, 2)
P12_NS = (1, 2, 5, 10, 20, 50, 100, "ALL")
N_BASE = 100                              # p12_common.N_BASE
WHICH = ("tf", "prompt", "decode", "steps", "both", "all")      # keys of a query handed to the calibration (rescope.KeyCache)


@dataclass
class Draw:
    rows: list                            # sequences (positions in the key-cache index), in the order handed to the calibration rule
    which: str                            # keys of each query: 'tf' (default) | 'prompt' | 'decode' | 'steps' | 'both' | 'all' (KeyCache.coords)
    n: int
    draw: int
    source: str
    rec: dict = field(default_factory=dict)
    kc: object = None                     # the key cache the rows are positions of; None = the log's own (MME-C draws: the MME capture)

    @property
    def seed(self) -> int:                # seed of the two-block split
        return self.draw


SOURCES = ("repair", "BEN-S", "MME-C")     # calibration sources (the REPAIR pool unless an amendment names another: R2, R17)


def resolve_source(source: str, part=None) -> str:
    """The name of a calibration source.  'benign' (the WHOLE label-free pool, the source of the first driver) is refused
    unless a part is named (review finding 3: it holds the test part BEN-T): source='BEN-S', or source='benign', part='BEN-S'."""
    from . import pools as P
    if source == "benign":
        if part is None:
            raise ValueError("source='benign' drew from the whole benign pool, which contains the test part BEN-T: name the part, "
                             "source='BEN-S' (or source='benign', part='BEN-S'); the MME source is source='MME-C'")
        source = part
    elif part is not None and part != source:
        raise ValueError(f"source {source!r} with part {part!r}")
    if source in P.TEST_ONLY or source in P.STATISTICS_ONLY:
        raise P.PoolGuardError(f"{source} cannot calibrate: BEN-T and MME-T are test rows and are used for nothing else, BEN-W is the statistics "
                               f"part of the feature packages; calibration sources: {SOURCES}")
    if source not in SOURCES:
        raise ValueError(f"source: one of {SOURCES}")
    return source


def pool_rows(L, source: str = "repair", which: str = "tf", part=None, mask=None) -> list:
    """The rows a calibration draw is taken from, in index order.  source: 'repair' (valid REPAIR rows) | 'BEN-S' | 'MME-C'
    (key-level pools of fx.pools: which='tf' = rows with a valid pre-edit answer, teacher-forced key; 'decode' / 'steps' = every
    row, prompt-only key / the 11 keys of a decode; MME-C rows are positions in the MME capture's own index).
    mask: None | 'qdisjoint' (fx.masks.question_disjoint: R8) | a predicate on the index row -- rows to keep."""
    from . import masks as MK, pools as P
    source = resolve_source(source, part)
    if source == "repair":
        if which not in WHICH:
            raise ValueError(f"which: one of {WHICH}")
        rows, ix = L.pool("repair", valid=True), L.index
    else:
        if which not in ("tf", "decode", "steps"):
            raise ValueError(f"{source} draws: which = 'tf' | 'decode' | 'steps'")
        rows, _ = P.rows(L, source, which)
        kc = P.source_kc(L, source)
        ix = L.index if kc is None else kc.index
    if mask is not None:
        f = MK.question_disjoint(L) if mask == "qdisjoint" else mask
        if not callable(f):
            raise ValueError("mask: None | 'qdisjoint' | a predicate on the index row")
        rows = [j for j in rows if f(ix[j])]
    return rows


def ranking(n_pool: int, draw: int) -> np.ndarray:
    """Seeded ranking of range(n_pool): its first 200 entries are, as a set, rescope.splits.sample(range(n_pool), 200, draw)."""
    from rescope import splits
    base = np.array(splits.sample(list(range(n_pool)), min(N_DEFAULT, n_pool), int(draw)), dtype=np.int64)
    rest = np.setdiff1d(np.arange(n_pool), base)
    return np.concatenate([base[np.random.default_rng([int(draw), 1]).permutation(base.size)],
                           rest[np.random.default_rng([int(draw), 2]).permutation(rest.size)]])


def explicit(L, rows, which: str = "tf", seed: int = 0, source: str = "explicit", note: str = "", kc=None) -> Draw:
    """A calibration sample given as explicit sequences (e.g. a part of partition()); seed = seed of the two-block split.
    kc: the key cache the rows are positions of (None: the log's).  Test-only pool rows (BEN-T, MME-T) are refused."""
    from . import pools as P
    rows = [int(j) for j in rows]
    if len(set(rows)) != len(rows):
        raise ValueError("a calibration sample holds each query once")
    P.guard(L, rows, "calibration", kc)
    ix = L.index if kc is None else kc.index
    return Draw(rows=rows, which=which, n=len(rows), draw=int(seed), source=source,
                rec={"source": source, "which": which, "n": len(rows), "draw": int(seed), "split_seed": int(seed), "nested": False, "note": note,
                     "sids": [ix[j]["sid"] for j in rows]}, kc=kc)


def partition(L, sizes, draw: int = 0, source: str = "repair", which: str = "tf", names=None) -> list:
    """DISJOINT seeded subsets of a pool: consecutive slices of the draw's ranking (the first slice of size n is calibration(L, n,
    draw)), each sorted in pool order -> list of Draw.  Amendment of 2026-11-11 01:50 (independence of the recalibration sample):
    C0, S, C1 = partition(L, (160, 160, 160), draw) for E2 d; C0, C1 = partition(L, (200, 200), draw) for E2 e-f; the bound is
    claimed for the thresholds calibrated on C1 (fx.methods.recalibrate(..., cal=C1)) only.
    The parts are disjoint as ROWS, not as queries (module docstring): two parts can hold the same query (the registered sizes
    3 x 160 do not fit into the REPAIR pool otherwise).  Detector repairs driven by rows of one part and calibrated on another
    must drop the shared queries first (fx.methods refuses them); E2 d - g, the experiments this was written for, were cut by R19."""
    if resolve_source(source) != "repair":
        raise ValueError("partition: parts of the REPAIR pool")
    pool = pool_rows(L, source, which)
    sizes = [int(x) for x in sizes]
    if min(sizes) < 1 or sum(sizes) > len(pool):
        raise ValueError(f"cannot cut parts of sizes {sizes} from the {len(pool)} rows of the {source} pool of {L.name}")
    rk, out, a = ranking(len(pool), int(draw)), [], 0
    for i, n in enumerate(sizes):
        rows = [pool[int(k)] for k in np.sort(rk[a:a + n])]
        out.append(Draw(rows=rows, which=which, n=n, draw=int(draw), source=source,
                        rec={"source": source, "which": which, "n": n, "draw": int(draw), "n_pool": len(pool), "split_seed": int(draw), "nested": True,
                             "part": i, "part_name": None if names is None else names[i], "sizes": sizes, "offset_in_ranking": a, "sids": [L.sid(j) for j in rows]}))
        a += n
    return out


def query_ids(L, rows) -> list:
    """Query identity of sequences: the content id of the key cache (same kind of prompt, question string and image content -> same
    id, hence the same key at the last prompt position)."""
    return [L.index[int(j)]["query_id"] for j in rows]


def null_sample(L, n_a: int = 100, draw: int = 0, n: int = N_DEFAULT, source: str = "repair", mask=None) -> list:
    """The benign sample A of the registered variant ReScope-N (amendments of 2026-11-11 01:35 and 09:03): n_a benign QUERIES of
    the REPAIR pool, none of which is one of the n calibration queries of the same draw -> sequences, in pool order.
    The rows are taken in the order of the draw's seeded ranking from position n on (the calibration draw is its first n rows); a row
    is skipped if its query (query_ids: same question and image content) is a calibration query or is already in the sample.  Up
    to 09:03 the sample was positions n .. n + n_a - 1 of the ranking: disjoint from the calibration draw as rows, but 0 to 7 of
    the 200 calibration queries were null-sample queries, whose coordinates the null repair sets to 0.
    mask: the mask of the calibration draw (fx.draws.calibration(..., mask=)): the sample is then taken from the same filtered pool
    and ranking, so that it stays disjoint from that draw.  The source is a pool of the log's own key cache (REPAIR)."""
    n, n_a = int(n), int(n_a)
    if n_a == 0:
        return []
    if resolve_source(source) != "repair":
        raise ValueError("the null-repair sample comes from the REPAIR pool (PROTOCOL.md: calibration and null-repair queries come from the "
                         "REPAIR pool unless an amendment names another source); with another calibration source pass source='repair' here")
    pool = pool_rows(L, source, mask=mask)
    if n < 1 or n_a < 0 or n + n_a > len(pool):
        raise ValueError(f"cannot draw {n} calibration and {n_a} further rows from the {len(pool)} rows of the {source} pool of {L.name}")
    rk, qid = ranking(len(pool), int(draw)), query_ids(L, pool)
    seen, pick = {qid[int(k)] for k in rk[:n]}, []
    for k in rk[n:]:
        if len(pick) == n_a:
            break
        if qid[int(k)] not in seen:
            seen.add(qid[int(k)])
            pick.append(int(k))
    if len(pick) < n_a:
        raise ValueError(f"the {source} pool of {L.name} holds only {len(pick)} queries outside the calibration draw of {n} rows; {n_a} were asked for")
    return [pool[k] for k in sorted(pick)]


def scale_edits(L, n_first: int = 100, n_later: int = 200, seed: int = 0) -> list:
    """Edit subset for the expensive methods at scale (amendment of 22:10, item 6: DUAL): the first n_first edits + n_later later
    edits drawn with a seed (all later edits if there are fewer); sorted edit ids.  The draw is nested in n_later."""
    first = list(range(min(n_first, L.T)))
    later = np.arange(len(first), L.T)
    pick = later[np.random.default_rng([int(seed), 6]).permutation(later.size)[:n_later]]
    return first + sorted(int(i) for i in pick)


def half_a(L) -> list:
    return list(range(L.T // 2))


def half_b(L) -> list:
    return list(range(L.T // 2, L.T))


def even_edits(L) -> list:
    """Edits with an even stream index (the contaminating half of E6 (10); the fit half of read-out H3)."""
    return list(range(0, L.T, 2))


def odd_edits(L) -> list:
    return list(range(1, L.T, 2))


def calibration(L, n: int = N_DEFAULT, draw: int = 0, source: str = "repair", which: str = "tf", contamination: float = 0.0,
                contam_kinds=("tgen", "igen"), exclude=None, *, part=None, mask=None, contam_half: str = "first") -> Draw:
    """The calibration draw (see the module docstring).  -> Draw; Draw.rec lists the sequences.
    exclude: sequences that must not be calibration queries (reported complaints: amendment of 22:10, item 3); the draw is then the
    first n rows of the same seeded ranking that are not excluded (equal to the plain draw where nothing is excluded).  Excluded
    BY QUERY: a pool row with the question and image content of an excluded sequence (the same key) is left out as well.
    source: 'repair' (default) | 'BEN-S' | 'MME-C' (key-level pools, fx.pools; Draw.kc = the cache of the rows).  The whole benign
    pool (source='benign' without part=) and the test parts BEN-T / MME-T are refused.
    mask: None | 'qdisjoint' | predicate: the draw is taken from the FILTERED pool (its own seeded ranking; R8, second read-out).
    contam_half: which edits contaminate -- 'first' (edits 0 .. T // 2 - 1, kinds mixed at random: the first driver's draw) |
    'even' (E6 (10): the edits with an EVEN stream index, text and image rephrases half each: c - c // 2 T-Gen and c // 2 I-Gen
    sequences, each a prefix of its own seeded permutation, hence nested in the share; report coverage on the odd edits)."""
    from . import pools as P
    source = resolve_source(source, part)
    kc = P.source_kc(L, source) if source != "repair" else None
    ix = L.index if kc is None else kc.index
    pool = pool_rows(L, source, which, mask=mask)
    n, draw = int(n), int(draw)
    if not 0 <= contamination < 1:
        raise ValueError("contamination: a fraction in [0, 1)")
    c = int(round(contamination * n))
    rk = ranking(len(pool), draw)
    if exclude is not None and len(exclude):
        if kc is not None:
            raise ValueError("exclude= names sequences of the log's key cache; the source pool lives in another cache")
        P.guard(L, exclude, "stream")
        exq = set(query_ids(L, exclude))
        rk = np.array([i for i in rk if L.index[pool[int(i)]]["query_id"] not in exq], dtype=np.int64)
    if not 1 <= n - c <= len(rk):
        raise ValueError(f"cannot draw {n - c} of the {len(rk)} admissible rows of the {source} pool of {L.name}")
    rows = [pool[int(i)] for i in np.sort(rk[:n - c])]
    rec = {"source": source, "which": which, "n": n, "draw": draw, "n_pool": len(pool), "n_benign": n - c, "contamination": float(contamination),
           "n_contaminating": c, "split_seed": draw, "nested": True, "n_excluded": len(pool) - len(rk)}
    if mask is not None:
        rec["mask"] = mask if isinstance(mask, str) else "predicate"
    if c:
        if kc is not None:
            raise ValueError("contaminated draws mix edit sequences of the log's cache into the sample: not with a source of another key cache")
        if contam_half == "first":
            cand = [L.edit_rows[k][i] for k in contam_kinds for i in half_a(L)]
            pick = [cand[int(i)] for i in np.random.default_rng([draw, 3]).permutation(len(cand))[:c]]
            half = "A (edits 0 .. T // 2 - 1)"
        elif contam_half == "even":
            ev = even_edits(L)
            if tuple(contam_kinds) != ("tgen", "igen"):
                raise ValueError("contam_half='even': text and image rephrases, half each")
            nt, ni = c - c // 2, c // 2
            if max(nt, ni) > len(ev):
                raise ValueError(f"{c} contaminating queries need {max(nt, ni)} even edits; the log has {len(ev)}")
            pt = [L.edit_rows["tgen"][ev[int(i)]] for i in np.random.default_rng([draw, 31]).permutation(len(ev))[:nt]]
            pi = [L.edit_rows["igen"][ev[int(i)]] for i in np.random.default_rng([draw, 32]).permutation(len(ev))[:ni]]
            pick, half = pt + pi, "even stream index (c - c // 2 T-Gen + c // 2 I-Gen sequences)"
        else:
            raise ValueError("contam_half: 'first' | 'even'")
        rows = rows + pick
        rec.update(contam_kinds=list(contam_kinds), contam_half=half, contaminating_sids=[L.sid(j) for j in pick],
                   contaminating_edits=sorted({L.own(j) for j in pick}))
    rec["sids"] = [ix[j]["sid"] for j in rows]
    return Draw(rows=rows, which=which, n=n, draw=draw, source=source, rec=rec, kc=kc)


# ----------------------------------------------------------------------------- reported complaints (P12 pairing)
def _repair_complaints(L):
    """D1 complaints (under the DEPLOYED log, decoder v2) with a valid pre-edit answer in the REPAIR pool: (records sorted by
    (probe_idx, ptype), their sequences)."""
    out = []
    for j in L.rows("probe", pool="repair", valid=True):
        r = L.probe_rec(j)
        if r["is_complaint_D1"]:
            out.append((r["probe_idx"], r["ptype"], j))
    out.sort(key=lambda t: (t[0], t[1]))
    return out


def complaint_stream(L, seed: int, n_base: int = N_BASE) -> list:
    """The nested stream of reported complaints of (log, seed) as P12 built it -> sequences (positions in the key-cache index).
    n_base (default 100 = P12): the size of the stratified base draw = harness.sample_complaints(H, n_base, seed).  A log with
    fewer than n_base / 2 complaints of a modality (the prefix logs of R12) has no stream at the default; pass a smaller even
    n_base there (e.g. 20: the largest nested size the prefix experiments report below 'all') -- the first n_base complaints
    are then harness.sample_complaints(H, n_base, seed) as a set, interleaved text / image, and the rest follows in a seeded order."""
    comp = _repair_complaints(L)
    rng = np.random.default_rng(int(seed))                              # harness.sample_complaints(H, 100, seed): image first, then text
    base = {}
    N_BASE = int(n_base)                                                # noqa: N806  (shadows the module constant: the default is P12's)
    for pt, k in (("image", N_BASE // 2), ("text", N_BASE - N_BASE // 2)):
        rows = [t for t in comp if t[1] == pt]                          # sorted by probe_idx
        if k > len(rows):
            raise ValueError(f"only {len(rows)} D1 {pt} complaints in the REPAIR pool of {L.name}")
        base[pt] = [rows[int(i)] for i in rng.choice(len(rows), k, replace=False)]
    stream = [base["text"][k // 2] if k % 2 == 0 else base["image"][k // 2] for k in range(N_BASE)]
    used = {(t[0], t[1]) for t in stream}
    rest = [t for t in comp if (t[0], t[1]) not in used]
    stream += [rest[int(i)] for i in np.random.default_rng([int(seed), 12]).permutation(len(rest))]
    if len({t[2] for t in stream}) != len(comp):
        raise RuntimeError("complaint stream is not a permutation of the REPAIR complaints")
    return [t[2] for t in stream]


def complaints(L, n, seed: int = 0, n_base: int = N_BASE) -> list:
    """The first n complaints of the stream of (log, seed) (n = 'ALL': every REPAIR-pool D1 complaint); nested in n."""
    s = complaint_stream(L, seed, n_base)
    k = len(s) if n == "ALL" else int(n)
    if not 0 <= k <= len(s):
        raise ValueError(f"{L.name} has {len(s)} REPAIR-pool complaints")
    return s[:k]


def verify_p12(L, out_dir: str | None = None) -> dict:
    """Compare with the records of P12: every record <METHOD>__n<n>__s<seed>.json of the log must list the groups of
    complaints(L, n, seed) in the same order."""
    d = os.path.join(fx.P12, "out", L.name) if out_dir is None else out_dir
    res = {"dir": d, "n_records": 0, "n_equal": 0, "differing": [], "seeds": set(), "ns": set()}
    if not os.path.isdir(d):
        return dict(res, available=False, seeds=[], ns=[])
    streams = {}
    for f in sorted(os.listdir(d)):
        if not f.endswith(".json") or "__n" not in f:
            continue
        with open(os.path.join(d, f)) as fh:
            r = json.load(fh)
        if "reported_groups" not in r:
            continue
        s = int(r["seed"])
        if s not in streams:
            streams[s] = [L.index[j]["group"] for j in complaint_stream(L, s)]
        want = streams[s] if r["n"] == "ALL" else streams[s][:int(r["n"])]
        res["n_records"] += 1
        ok = list(r["reported_groups"]) == want
        res["n_equal"] += int(ok)
        res["seeds"].add(s)
        res["ns"].add(str(r["n"]))
        if not ok and len(res["differing"]) < 10:
            res["differing"].append(f)
    return dict(res, available=True, seeds=sorted(res["seeds"]), ns=sorted(res["ns"]), ok=res["n_records"] > 0 and res["n_equal"] == res["n_records"])
