"""Text logs, prompts, pools, the key cache for text and the seeded draws.

    L = e10t.open_log("TX-G2XL-rome-dev100")      # TextLog: L.log (rescope.EditLog, CPU fp64, K = the solver key k_request),
                                                   # L.rows[kind] (request | paraphrase | neighborhood rows of the logged records),
                                                   # L.benign("repair" | "heldout") (benign prompt rows with this log's flags),
                                                   # L.ids(text_id), L.y0(text_id), L.solver_pos, L.kc (KeyCache), L.weights()
    L50 = e10t.open_log("TX-G2XL-rome-dev100", T=50)     # a PREFIX of a log (first 50 updates; deployed weight by replay)
    dr = e10t.data.calibration(L, n=200, draw=0)   # the n = 200 benign prompts of draw 0 (REPAIR pool)

PROMPT ROWS are the rows of the build's out/prompts.jsonl (set, kind, pos = index of the edit in its log, k, text_id, ...);
rows with the same text share `text_id`, the unit of the key cache and of every forward.  Token ids are tok(text) (no BOS).

KEY CACHE (10_final/cache/text/gpt2-xl/<part>/): keys.npy (N_pos, 6400) float32 = the inputs of the edited module under the
PRE-EDIT model at every position of [prompt, y0] (L + 10 positions per text: prompt tokens, then the ten tokens of the
pre-edit greedy continuation, teacher forced), index.json (text_ids, offset, n_prompt), meta.json.  Parts: 'pools' (benign
prompts of the REPAIR and HELD-OUT pools: they do not depend on the log), 'dev' (request / paraphrase / neighbourhood
prompts of the development log), 'stream_<a>_<b>' (the same for the requests a .. b-1 of the reported stream; built by
scripts/build_keycache.py when the registered experiment starts).  A text is stored once (in the first part that has it).
float32, not float16: rounding the keys to float16 moves a normalised coordinate by up to 4.7e-5 of its scale |w||k|/s,
77 times the round-off between two batch layouts of the float32 forward (6.1e-7); the decision band (e10t.BAND = 1e-5)
covers the latter with a factor 16 and would have to be 30 times wider for float16 (tests c_cache / g_cache).

DRAWS (as fx.draws).  Calibration draw d of size n: the first n rows of fx.draws.ranking(|pool|, d) over the valid benign
prompts of the REPAIR pool in file order (n = 200: rescope.splits.sample(pool, 200, seed=d)); the two-block split uses
seed d.  null_sample: the next n_a rows of the same ranking whose text is no calibration text (disjoint by QUERY).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import torch

import e10t

D_IN, D_OUT = 6400, 1600
_G = {}


def _tx():
    import tx_public as TX
    return TX


def tokenizer():
    """The GPT-2 tokenizer as the build loaded it (public stack; safe to call before or after the model is loaded)."""
    if "tok" not in _G:
        import tp_common as C
        if "transformers" not in sys.modules:
            C.setup_public()
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(C.MODEL_DIR)
        tok.pad_token = tok.eos_token
        _G["tok"] = tok
    return _G["tok"]


def prompt_texts():
    if "PT" not in _G:
        _G["PT"] = _tx().prompt_texts()
    return _G["PT"]


def pre_continuations():
    if "PRE" not in _G:
        _G["PRE"] = _tx().pre_continuations()
        if int(_G["PRE"]["n_dec"]) != e10t.N_DEC:
            raise RuntimeError("the stored pre-edit continuations do not have N_DEC tokens")
    return _G["PRE"]


def all_prompts():
    if "ROWS" not in _G:
        _G["ROWS"] = _tx().prompts()
    return _G["ROWS"]


def counterfact():
    if "CF" not in _G:
        import tp_common as C
        _G["CF"] = C.load_counterfact()
    return _G["CF"]


def jdump(obj, path, indent=1):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=indent)
    os.replace(tmp, path)


def jload(path):
    with open(path) as f:
        return json.load(f)


# ----------------------------------------------------------------------------- key cache
class KeyCache:
    """Reader of the text key cache.  kc.has(text_id); kc.keys(text_id) -> (L + 10, 6400) float32 array (memory-mapped);
    kc.n_prompt(text_id); kc.stack(text_ids) -> (keys (N, 6400) float32 tensor, offsets (n + 1,)); kc.parts; kc.meta[part]."""

    def __init__(self, parts=None, root: str | None = None):
        self.dir = os.path.join(root or e10t.CACHE_ROOT, e10t.BACKBONE)
        have = sorted(p for p in (os.listdir(self.dir) if os.path.isdir(self.dir) else [])
                      if os.path.exists(os.path.join(self.dir, p, "meta.json")))
        self.parts = [p for p in (have if parts is None else parts)]
        self.where, self.arr, self.meta = {}, {}, {}
        for p in self.parts:
            d = os.path.join(self.dir, p)
            ix, self.meta[p] = jload(os.path.join(d, "index.json")), jload(os.path.join(d, "meta.json"))
            a = np.load(os.path.join(d, "keys.npy"), mmap_mode="r")
            if a.shape != (self.meta[p]["n_positions"], D_IN) or a.dtype != np.float32 or len(ix["text_ids"]) != self.meta[p]["n_texts"]:
                raise RuntimeError(f"key cache part {p}: files do not agree with meta.json")
            self.arr[p] = a
            for t, off, npr in zip(ix["text_ids"], ix["offset"], ix["n_prompt"]):
                self.where.setdefault(int(t), (p, int(off), int(npr)))
        w0 = {m["w0_sha256"] for m in self.meta.values()}
        if len(w0) > 1:
            raise RuntimeError("key cache parts were built with different pre-edit weights")

    def has(self, tid) -> bool:
        return int(tid) in self.where

    def n_prompt(self, tid) -> int:
        return self.where[int(tid)][2]

    def n_pos(self, tid) -> int:
        return self.where[int(tid)][2] + e10t.N_DEC

    def keys(self, tid) -> np.ndarray:
        p, off, npr = self.where[int(tid)]
        return self.arr[p][off:off + npr + e10t.N_DEC]

    def stack(self, tids):
        """(keys (N, 6400) float32 tensor, offsets (n + 1,) long): the cached positions of the texts, concatenated."""
        ks = [np.asarray(self.keys(t)) for t in tids]
        off = np.concatenate([[0], np.cumsum([k.shape[0] for k in ks])]).astype(np.int64)
        K = torch.from_numpy(np.concatenate(ks)) if ks else torch.zeros(0, D_IN)       # np.concatenate copies: a writable array
        return K, torch.from_numpy(off)

    def missing(self, tids) -> list:
        return [int(t) for t in tids if int(t) not in self.where]


def part_texts(part: str) -> list:
    """The text ids of a cache part, in storage order (see the module docstring)."""
    rows = all_prompts()
    if part == "pools":
        tids = [r["text_id"] for r in rows if r["set"] in ("repair", "heldout")]
    elif part == "dev":
        tids = [r["text_id"] for r in rows if r["set"] == "dev"]
    elif part.startswith("stream_"):
        a, b = (int(x) for x in part.split("_")[1:3])
        tids = [r["text_id"] for r in rows if r["set"] == "stream" and a <= r["pos"] < b]
    else:
        raise KeyError(f"cache part {part!r}: 'pools' | 'dev' | 'stream_<a>_<b>'")
    return list(dict.fromkeys(int(t) for t in tids))


STREAM_PARTS = ("stream_0_100", "stream_100_500", "stream_500_1000", "stream_1000_2000")


def parts_of(name: str) -> list:
    """Cache parts a log needs: the pools and the prompts of its own records."""
    src, T = _tx().PLAN[name]
    if src == "dev":
        return ["pools", "dev"]
    return ["pools"] + [p for p in STREAM_PARTS if int(p.split("_")[1]) < T]


def build_part(B, part: str, root: str | None = None, verbose: bool = True, tids=None) -> dict:
    """Build one part of the key cache with the model at the PRE-EDIT weight (resumable).  Texts already stored in another
    existing part are skipped.  B: e10t.Backbone.  -> meta."""
    root = os.path.join(root or e10t.CACHE_ROOT, e10t.BACKBONE)
    d = os.path.join(root, part)
    if os.path.exists(os.path.join(d, "meta.json")):
        return jload(os.path.join(d, "meta.json"))
    os.makedirs(d, exist_ok=True)
    B.check_w0()
    if B._served is not None:
        raise RuntimeError("build the key cache outside a served method")
    PT, PRE = prompt_texts(), pre_continuations()
    other = KeyCache(root=os.path.dirname(root)) if os.path.isdir(root) else None
    tids = [t for t in (part_texts(part) if tids is None else tids) if other is None or not other.has(t)]
    if not bool(PRE["valid"][torch.tensor(tids)].all()):
        raise RuntimeError("a text has no valid pre-edit continuation")
    seqs = [PT["input_ids"][t] + PRE["y0"][t].tolist() for t in tids]
    npr = [len(PT["input_ids"][t]) for t in tids]
    lens = [len(s) for s in seqs]
    off = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    N = int(off[-1])
    sig = {"part": part, "n_texts": len(tids), "n_positions": N, "tids_sha1": hashlib.sha1(json.dumps(tids).encode()).hexdigest(),
           "w0_sha256": PRE["w0"]["sha256"], "max_tokens": B.max_tokens, "max_rows": B.max_rows}
    kp, pp = os.path.join(d, "keys.npy"), os.path.join(d, "partial.json")
    bl = B.batches(lens, extra=0)
    done = 0
    if os.path.exists(pp) and os.path.exists(kp) and jload(pp)["sig"] == sig:
        done = int(jload(pp)["batches_done"])
        arr = np.load(kp, mmap_mode="r+")
    else:
        arr = np.lib.format.open_memmap(kp, mode="w+", dtype=np.float32, shape=(N, D_IN))
    t0 = time.time()
    for bi in range(done, len(bl)):
        b = bl[bi]
        xs = B.keys([seqs[i] for i in b])
        for i, x in zip(b, xs):
            arr[off[i]:off[i + 1]] = x.float().cpu().numpy()
        if (bi + 1) % 50 == 0 or bi + 1 == len(bl):
            arr.flush()
            jdump({"sig": sig, "batches_done": bi + 1}, pp)
            if verbose:
                print(f"  cache part {part}: batch {bi + 1}/{len(bl)} {time.time() - t0:.0f}s", flush=True)
    arr.flush()
    del arr
    h = hashlib.md5()
    with open(kp, "rb") as f:
        while True:
            blk = f.read(1 << 24)
            if not blk:
                break
            h.update(blk)
    jdump({"text_ids": tids, "offset": [int(x) for x in off[:-1]], "n_prompt": npr}, os.path.join(d, "index.json"), indent=None)
    meta = dict(sig, keys_md5=h.hexdigest(), dtype="float32", n_dec=e10t.N_DEC, backbone=e10t.BACKBONE, module=_tx().MODULE,
                layout="right-padded batches sorted by length (e10t.backbone.batches, extra=0), truncated forward of [prompt, y0]",
                n_batches=len(bl), seconds=time.time() - t0, written=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                code=e10t.code_fingerprint(), what="inputs of the edited module under the pre-edit model at every position of [prompt, y0]")
    jdump(meta, os.path.join(d, "meta.json"))
    if os.path.exists(pp):
        os.remove(pp)
    return meta


# ----------------------------------------------------------------------------- the log
def replay_weight(name: str, T: int | None = None) -> torch.Tensor:
    """Module weight (6400, 1600) float32 after the first T updates of a log: W_0 plus the public code's float32 in-place
    writes  W += left[:, None] @ right[None, :]  in order (bit for bit the deployed weight for T = the log's length)."""
    TX = _tx()
    f = TX.factors(name)
    T = f["left"].shape[0] if T is None else int(T)
    W = torch.load(TX.registry()[name]["pre_edit_weight"], map_location="cpu")["weight"].clone()
    for i in range(T):
        W += f["left"][i].unsqueeze(1) @ f["right"][i].unsqueeze(0)
    return W


class TextLog:
    def __init__(self, name: str, T: int | None = None, kc: KeyCache | None = None):
        TX = _tx()
        if name not in TX.PLAN:
            raise KeyError(f"unknown text log {name!r}: {TX.NAMES}")
        full = TX.from_public(name)
        self.base, self.T_full = name, full.T
        self.T = full.T if T is None else int(T)
        if not 1 <= self.T <= full.T:
            raise ValueError(f"prefix length {T} of a log with {full.T} updates")
        self.name = name if self.T == full.T else f"{name}@{self.T}"
        self.is_prefix = self.T != full.T
        self.development = name == TX.DEV_LOG
        self.src = TX.PLAN[name][0]
        if self.is_prefix:
            from rescope.loglib import EditLog
            meta = {k: (v[:self.T] if (torch.is_tensor(v) or isinstance(v, list)) and len(v) == full.T else v) for k, v in full.meta.items()}
            self.log = EditLog(full.U[:self.T], full.W[:self.T], full.K[:self.T], torch.arange(self.T), name=self.name, meta=meta)
        else:
            self.log = full
        fac = TX.factors(name)
        self.solver_pos = [int(s["lookup_idxs"][0]) for s in fac["scalars"][:self.T]]       # last subject token in the request prompt
        self.requests = list(fac["requests"][:self.T])
        self.records = [int(x) for x in fac["record_indices"][:self.T]]
        self.case_ids = [int(x) for x in fac["case_ids"][:self.T]]
        self.target = fac["target"][:self.T].double()                                       # module-output target of every update (logged)
        self.rows = {k: [] for k in ("request", "paraphrase", "neighborhood")}
        for r in all_prompts():
            if r["set"] == self.src and r["pos"] < self.T:
                r = dict(r)
                if r["kind"] == "neighborhood":
                    if self.src == "stream":
                        eq = r.get("text_first_stream_request_pos") is not None and r["text_first_stream_request_pos"] < self.T
                        men = r["mentions_stream_subject_pos"] is not None and r["mentions_stream_subject_pos"] < self.T
                    else:
                        eq = bool(r.get("text_is_dev_request"))
                        men = r["mentions_dev_subject_pos"] is not None and r["mentions_dev_subject_pos"] < self.T
                    r.update(near_text=bool(eq), near_mention=bool(men), clean=not (eq or men))
                self.rows[r["kind"]].append(r)
        for k in self.rows:
            self.rows[k].sort(key=lambda r: (r["pos"], r["k"]))
        if [r["pos"] for r in self.rows["request"]] != list(range(self.T)):
            raise RuntimeError("request rows do not match the log")
        self.PT, self.PRE = prompt_texts(), pre_continuations()
        self.kc = kc if kc is not None else KeyCache()
        self._benign, self._W, self._tgt = {}, None, None

    # ------------------------------------------------------------------ prompts
    def ids(self, tid) -> list:
        return self.PT["input_ids"][int(tid)]

    def y0(self, tid) -> list:
        return self.PRE["y0"][int(tid)].tolist()

    def valid(self, tid) -> bool:
        return bool(self.PRE["valid"][int(tid)])

    def text(self, tid) -> str:
        return self.PT["texts"][int(tid)]

    def benign(self, pool: str) -> list:
        """Benign prompt rows of a pool ('repair' | 'heldout') with THIS log's flags: near_subject (the record's subject is the
        subject of a logged request), near_mention (the prompt mentions a logged subject), near_text (the prompt is a logged
        prompt), clean = none of them.  Only rows with a valid pre-edit continuation (all rows of the build are valid)."""
        if pool not in self._benign:
            TX = _tx()
            out = []
            for r in all_prompts():
                if r["set"] != pool:
                    continue
                r = dict(r)
                if self.src == "stream":
                    ns = r["subject_first_stream_pos"] is not None and r["subject_first_stream_pos"] < self.T
                    nm = r["mentions_stream_subject_pos"] is not None and r["mentions_stream_subject_pos"] < self.T
                    nt = any(r[k] is not None and r[k] < self.T for k in ("text_first_stream_neighborhood_pos", "text_first_stream_request_pos"))
                else:                                               # development log (flags of the full log; a prefix inherits them)
                    ns = bool(r["subject_in_dev"])
                    nm = r["mentions_dev_subject_pos"] is not None
                    nt = bool(r["text_is_dev_request"] or r["text_is_dev_neighborhood"] or r["text_is_dev_paraphrase"])
                r.update(near_subject=bool(ns), near_mention=bool(nm), near_text=bool(nt), clean=not (ns or nm or nt))
                if self.valid(r["text_id"]):
                    out.append(r)
            self._benign[pool] = out
            del TX
        return self._benign[pool]

    def all_rows(self, sets=("request", "paraphrase", "neighborhood", "repair", "heldout")) -> list:
        out = []
        for s in sets:
            out += self.rows[s] if s in self.rows else self.benign(s)
        return out

    def target_ids(self, i: int, which: str = "new") -> list:
        """Token ids of the new (or true) object of request i as the public code tokenises it: tok(' ' + object)."""
        if self._tgt is None:
            tok = tokenizer()
            self._tgt = {w: [tok(" " + rq[f"target_{w}"]["str"].strip())["input_ids"] for rq in self.requests] for w in ("new", "true")}
        return self._tgt[which][int(i)]

    # ------------------------------------------------------------------ weights and keys
    def weights(self):
        """(W_0, W_E) in the module's orientation (6400, 1600) float32, CPU.  Prefix logs: W_E by replay (replay_weight)."""
        if self._W is None:
            TX = _tx()
            if self.is_prefix:
                W0 = torch.load(TX.registry()[self.base]["pre_edit_weight"], map_location="cpu")["weight"]
                self._W = (W0, replay_weight(self.base, self.T))
            else:
                self._W = TX.weights(self.base)
        return self._W

    def own_keys(self):
        """(keys (T, 6400) float32, edits (T,)): the key of every request AS SERVED = the cached module input at the solver
        position (last subject token) of its own request prompt: the guards of every calibration (repair.add_guards)."""
        ks = [torch.from_numpy(np.array(self.kc.keys(r["text_id"])[self.solver_pos[r["pos"]]])) for r in self.rows["request"]]
        return torch.stack(ks), list(range(self.T))

    def describe(self) -> dict:
        return {"name": self.name, "T": self.T, "development": self.development, "source": self.src, "prefix_of": self.base if self.is_prefix else None,
                "n_request": len(self.rows["request"]), "n_paraphrase": len(self.rows["paraphrase"]), "n_neighborhood": len(self.rows["neighborhood"]),
                "n_repair": len(self.benign("repair")), "n_heldout": len(self.benign("heldout")),
                "n_heldout_clean": sum(r["clean"] for r in self.benign("heldout")), "n_repair_clean": sum(r["clean"] for r in self.benign("repair")),
                "n_neighborhood_clean": sum(r["clean"] for r in self.rows["neighborhood"]), "cache_parts": list(self.kc.parts)}


_LOGS = {}


def open_log(name: str, T: int | None = None) -> TextLog:
    """The text log `name`, or (T given) its PREFIX of the first T updates.  A prefix that is itself a registered log of the same
    source (the stream's T = 100 / 500 / 1000 / 2000) is opened as that log (its stored, hash-verified deployed weight);
    any other prefix gets its deployed weight by replaying the logged float32 writes."""
    TX = _tx()
    if T is not None and name in TX.PLAN:
        src = TX.PLAN[name][0]
        same = [n for n, (s_, t_) in TX.PLAN.items() if s_ == src and t_ == int(T)]
        if same:
            name, T = same[0], None
    key = (name, T)
    if key not in _LOGS:
        _LOGS[key] = TextLog(name, T)
    return _LOGS[key]


# ----------------------------------------------------------------------------- draws
@dataclass
class Draw:
    rows: list                 # prompt rows of the REPAIR pool, in pool order
    n: int
    draw: int
    rec: dict = field(default_factory=dict)

    @property
    def seed(self) -> int:     # seed of the two-block split
        return self.draw

    @property
    def text_ids(self) -> list:
        return [r["text_id"] for r in self.rows]


def _ranking(n_pool: int, draw: int):
    import fx
    return fx.draws.ranking(n_pool, int(draw))


def calibration(L: TextLog, n: int = e10t.N_CAL, draw: int = 0, pool: str = "repair") -> Draw:
    """The n benign calibration prompts of draw `draw`: the first n rows of the draw's seeded ranking of the valid benign
    prompts of the REPAIR pool (all of them: flagged near-scope prompts are not removed, as in the registered experiments)."""
    rows = L.benign(pool)
    if not 1 <= n <= len(rows):
        raise ValueError(f"cannot draw {n} of the {len(rows)} benign prompts of the {pool} pool")
    idx = np.sort(_ranking(len(rows), draw)[:n])
    pick = [rows[int(i)] for i in idx]
    if len({r["text_id"] for r in pick}) != len(pick):
        raise RuntimeError("a calibration draw holds a prompt text twice")
    return Draw(pick, n, int(draw), {"pool": pool, "n": n, "draw": int(draw), "n_pool": len(rows), "split_seed": int(draw),
                                     "n_near": sum(not r["clean"] for r in pick), "prompt_ids": [r["prompt_id"] for r in pick]})


def null_sample(L: TextLog, n_a: int = e10t.N_NULL, draw: int = 0, n: int = e10t.N_CAL, pool: str = "repair") -> list:
    """The benign sample of the null-repair variant: n_a prompts of the REPAIR pool after position n of the draw's ranking
    whose TEXT is none of the n calibration texts and not yet in the sample (disjoint by query) -> rows in pool order."""
    if n_a == 0:
        return []
    rows = L.benign(pool)
    rk = _ranking(len(rows), draw)
    seen, pick = {rows[int(i)]["text_id"] for i in rk[:n]}, []
    for i in rk[n:]:
        if len(pick) == n_a:
            break
        t = rows[int(i)]["text_id"]
        if t not in seen:
            seen.add(t)
            pick.append(int(i))
    if len(pick) != n_a:
        raise ValueError(f"the {pool} pool has too few prompts for {n} calibration and {n_a} further prompts")
    return [rows[i] for i in sorted(pick)]
