"""mme_pool: reader of the MME benign workload (10_final/PROTOCOL.md, section 6, amendment "Clarification of H3 (R6) and R17",
item (2) = R17): the out-of-benchmark benign queries, captured under the PRE-EDIT model like the benign pool and split by
image into MME-C (calibration source) and MME-T (test rows).  A reader and nothing else: it builds no gate, calibrates
nothing and runs no model.

The capture of a backbone, 10_final/cache2/mme/<backbone>/, is a key cache in the file format of 10_final/cache/<backbone>/
(index.jsonl + keys_tf.pt / keys_decode.pt / keys_decode_steps.pt + meta.json), so rescope.keycache.KeyCache reads it as it
is; every row has kind 'benign' and pool 'benign', like the rows of the benign pool, plus `part` and `category`.  This
module is the glue around it: the KeyCache, the rows of a part, and the library items for a behavioural evaluation.

TEN-LINE EXAMPLE  (`gate`: a role-'last' rescope.ScopedLog of the backbone; H: the harness handle of its log)
    import sys; sys.path.insert(0, "$RESCOPE_ROOT/02_research/experiments/10_final/cache2/mme")
    import mme_pool, rescope                               # (mme_pool puts the library's directory on sys.path)
    P = mme_pool.load("qwen25vl")                          # KeyCache of cache2/mme/qwen25vl, md5-checked; P.kc, P.index
    cal = P.rows("MME-C", valid=True)                      # positions in P.index: calibration queries with a teacher-forced key
    test = P.rows("MME-T", valid=True)                     # test rows on which D1 is defined (pre-edit answer reproduced by W_0)
    coords = P.kc.coords(gate, cal, which="tf")            # (n, D) calibration scores -> calibrate.calibrate_two_block(coords=...)
    fire = P.kc.decide(gate, "tf", test)                   # false firing on MME-T: fire["fired"], fire["top_edit"], fire["near"]
    ev = rescope.SparseEvaluator(H, cache=P.kc, hard_negatives=False)      # behaviour: forwards only where the gate fires
    d1 = [r["D1"] for r in ev.tf_rows(gate, P.items(test))]                # D1 of every test row under the gate
    units = P.field(test, "uid")                           # the image of each row = the unit of a cluster bootstrap

KEYS OF A ROW (which = ...; rescope.keycache)   'decode' the prompt-only key = first forward of a decode (every row);
    'steps' the 11 keys of a decoder-v2 decode (every row); 'tf' the teacher-forced key of prompt + pre-edit answer (rows with a
    non-empty pre-edit answer); 'all' = tf + steps.  rows(part) without valid=True returns every row of the part: use
    which='decode' or 'steps' for them.
ROW FIELDS used here: part ('MME-C' | 'MME-T'), category, uid (the image: 'mme_img:<md5>'; rows of one image share the part),
    y0_text / y0_valid (decoder-v2 pre-edit answer and its validity), y0_yn ('yes' | 'no' | 'other'), ref_answer, query_id.
    NOTE deg_none_no is true for every pre-edit answer "No": on this yes / no workload it marks a legitimate answer.
    NOTE result rows of these items have kind 'benign': rescope.summarise / locality group by kind 'probe' / 'hn' and leave them
    out (compute rates from the row flags, as in the example), and the driver's fx.behave refuses benign rows by design.

API
    load(backbone, root=None, verify=True) -> MMEPool
    MMEPool.kc                       rescope.keycache.KeyCache of the directory     .index = kc.index     .meta     .dir
    MMEPool.rows(part=None, *, valid=None, category=None, where=None) -> [positions in .index]   (part: a name or a tuple)
    MMEPool.sids(part) -> [sid]      .part(sid) -> 'MME-C' | 'MME-T'      .field(rows, name) -> [values]
    MMEPool.items(rows) -> [library items]   for rescope.Evaluator / SparseEvaluator .tf_rows (D1, tok_loc) / .generate (agree,
                                     long_eq_pre): free image queries with their pre-edit answer; rows without a teacher-forced
                                     sequence are refused
    MMEPool.summary -> counts per part (rows, images, valid, yes / no / other)
Test: test_mme_pool.py next to this file (CPU; writes test_mme_pool.json).   Format and construction: <backbone>/README.txt.
"""
from __future__ import annotations
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP

import os
import sys

sys.dont_write_bytecode = True                                          # the library directory is read-only for this package
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

HERE = os.path.dirname(os.path.abspath(__file__))                       # 10_final/cache2/mme
LIB = f"{_RP.ROOT}/02_research/code"           # the library rescope
PARTS = ("MME-C", "MME-T")
BACKBONES = ("qwen25vl", "llava")
LAYER = {"qwen25vl": 20, "llava": 16}
if LIB not in sys.path:                                                 # `import rescope` works after `import mme_pool`
    sys.path.insert(0, LIB)


def directory(backbone: str, root: str | None = None) -> str:
    return os.path.join(HERE if root is None else root, backbone)


def keycache(backbone: str, root: str | None = None, verify: bool = False):
    """rescope.keycache.KeyCache of the MME capture of a backbone (verify: md5 of the key files against meta.json)."""
    from rescope.keycache import KeyCache
    return KeyCache(directory(backbone, root), verify=verify)


def item_of(e: dict) -> dict:
    """The library item of one index row (see MMEPool.items)."""
    if e.get("row_tf") is None:
        raise ValueError(f"{e['sid']}: no teacher-forced sequence (empty pre-edit answer); gate-level only (which='decode' | 'steps')")
    return {"kind": "benign", "sid": e["sid"], "y0_valid": bool(e["y0_valid"]), "part": e["part"], "category": e["category"],
            "uid": e["uid"], "query_id": e["query_id"], "ptype": "image", "source": "MME",
            "_row": {"image": e["image"], "prompt": e["prompt"], "y0_text": e["y0_text"] or "", "ptype": "image",
                     "long_text": e.get("long_text"), "edit_alt": None}}     # the two fields Evaluator.generate reads for a free query (no edit: leak = False)


class MMEPool:
    def __init__(self, backbone: str, root: str | None = None, verify: bool = True):
        if backbone not in BACKBONES:
            raise ValueError(f"backbone: one of {BACKBONES}")
        self.backbone, self.dir = backbone, directory(backbone, root)
        self.kc = keycache(backbone, root, verify)
        self.index, self.meta = self.kc.index, self.kc.meta
        self.parts = PARTS
        self._part = {e["sid"]: e["part"] for e in self.index}
        if verify:
            self.check()

    def check(self):
        """The directory is a complete MME capture of this backbone: raises otherwise."""
        m, ix = self.meta, self.index
        if m.get("backbone") != self.backbone or int(m.get("layer", -1)) != LAYER[self.backbone]:
            raise RuntimeError(f"{self.dir}: capture of {m.get('backbone')} layer {m.get('layer')}, expected {self.backbone} layer {LAYER[self.backbone]}")
        if not m.get("complete"):
            raise RuntimeError(f"{self.dir}: the capture is not complete ({m.get('n_missing_sequences')} rows missing)")
        if any(e.get("kind") != "benign" or e.get("part") not in PARTS or e.get("row_decode") is None or not e.get("rows_decode_steps") for e in ix):
            raise RuntimeError(f"{self.dir}: a row without part, prompt-only key or decode-step keys")
        if len(self._part) != len(ix):
            raise RuntimeError(f"{self.dir}: duplicate sids")
        img = {}
        for e in ix:
            if img.setdefault(e["uid"], e["part"]) != e["part"]:
                raise RuntimeError(f"{self.dir}: image {e['uid']} lies in two parts")
        if any(e.get("y0_valid") and e.get("row_tf") is None for e in ix):
            raise RuntimeError(f"{self.dir}: a valid row without a teacher-forced key")

    # ------------------------------------------------------------------ rows
    def rows(self, part=None, *, valid=None, category=None, where=None) -> list:
        """Positions in .index (= rows of kc).  part: 'MME-C' | 'MME-T' | a tuple | None (both); valid: True = rows whose pre-edit
        answer is reproduced by the pre-edit model (they have a teacher-forced key; D1 is defined), False = the others;
        category: an MME category or a tuple of them; where: a predicate on the index row."""
        want = None if part is None else ({part} if isinstance(part, str) else set(part))
        if want is not None and not want <= set(PARTS):
            raise ValueError(f"part: one of {PARTS}")
        cats = None if category is None else ({category} if isinstance(category, str) else set(category))
        out = []
        for j, e in enumerate(self.index):
            if want is not None and e["part"] not in want:
                continue
            if valid is not None and bool(e.get("y0_valid")) != valid:
                continue
            if cats is not None and e["category"] not in cats:
                continue
            if where is not None and not where(e):
                continue
            out.append(j)
        return out

    def sids(self, part) -> list:
        return [self.index[j]["sid"] for j in self.rows(part)]

    def part(self, sid: str) -> str:
        return self._part[sid]

    def field(self, rows, name: str) -> list:
        return [self.index[j].get(name) for j in rows]

    def items(self, rows) -> list:
        """Library items (rescope.Evaluator conventions) of rows that have a teacher-forced sequence: free image queries whose
        teacher-forced text is the pre-edit answer -- the item form of 10_final/cache/code/fc_capture.make_item for a benign row
        ('_row': image, prompt, y0_text, ptype; + long_text and edit_alt = None, which Evaluator.generate reads), plus public
        fields that the library copies into its result rows.  The teacher-forced row of such an item carries D1 (= the
        pre-edit answer is not reproduced) and tok_loc; its generation row carries agree (= the 10-token answer equals the
        pre-edit answer) and long_eq_pre."""
        return [item_of(self.index[j]) for j in rows]

    @property
    def summary(self) -> dict:
        out = {}
        for p in PARTS + ("all",):
            sub = [e for e in self.index if p == "all" or e["part"] == p]
            out[p] = {"rows": len(sub), "images": len({e["uid"] for e in sub}), "valid_y0": sum(bool(e["y0_valid"]) for e in sub),
                      "with_teacher_forced_key": sum(e["row_tf"] is not None for e in sub),
                      "y0": {k: sum(e.get("y0_yn") == k for e in sub) for k in ("yes", "no", "other")}}
        return out


def load(backbone: str, root: str | None = None, verify: bool = True) -> MMEPool:
    """The MME workload of a backbone ('qwen25vl' | 'llava').  root: another directory that holds <backbone>/ (tests).
    verify=True: md5 of the key files and of the index against meta.json, completeness, parts disjoint by image."""
    return MMEPool(backbone, root, verify)


if __name__ == "__main__":
    for bb in (sys.argv[1:] or BACKBONES):
        P = load(bb)
        print(bb, P.kc.summary()["n"], "rows;", {k: (v["rows"], v["images"], v["valid_y0"], v["y0"]) for k, v in P.summary.items()})
