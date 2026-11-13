"""fx.logs: the logs of the paper, their roles, and LogData = one log opened without a model.

REGISTRY (PROTOCOL.md sections 1 and 6; names are fixed here, a missing directory is tolerated and reported).  The registry states
what a log IS (role, backbone, editor, stream family, T, directory).  Which logs an experiment runs is the protocol's business:
at the time of writing (2026-11-11 09:03 UTC) amendment R19 runs nothing on the MMKE-Bench, BLIP2, one-hop and internal logs and
on the T = 200 logs of the MEMIT-style family (entry field cut_by_R19, shown by table(); no function refuses a log because of it).
Amendment R18 (testbed v4, "pilot logs") was withdrawn at 08:03 UTC and has left nothing here.
  development        TB-Q100-ae-l2_3, TB-L100-ae-l2_3                          (method development happened here)
  working_point      TB-Q100-ae, TB-Q100-ae-l2_1, TB-L100-ae                   (same streams, other ridge)
  unseen_alphaedit   TB-Q200-ae-l2_3, TB-Q500-ae-l2_3, TB-L200-ae-l2_3, TB-L500-ae-l2_3
  unseen_memit       TB-Q{100,200,500,1000,2000}-memitw, TB-L{100,200,500,1000}-memitw   (public configuration; being completed)
  internal           TB-Q1000-ae-l2_3, TB-Q2000-ae-l2_3, TB-L1000-ae-l2_3      (consistency check only; NEVER reported)
  stress             TB-Q100-mmke-l2_3, TB-Q500-mmke-l2_3                      (MMKE-Bench; separate stress test)
                     + TB-B{100,200,500}-ae (third backbone BLIP2-OPT-2.7B, layer 12: the editor did not install the edits; stress
                     logs of collateral damage, reported in the table of the MMKE-Bench logs; frozen row set of E3)
  unseen_onehop      TB-Q{100,500}-onehop-l2_3 (VLKEB one-hop split on Qwen): unseen; frozen row set of E3; calibration and null-repair
                     queries only from the REPAIR pool of their OWN probe rows (source='repair'; they have no benign pool and no
                     hard negatives); their portability queries are not sequences of the key cache
                     (both families: 09_testbed_scale/registry_part3.json, registered here if that file exists; protocol amendments of
                     2026-11-11 04:26 and 05:06; a log of that file that is neither gets the role 'scout')
  register(name, role, backbone=, layer=, family=, cache=, T=, ...)   adds any further log at run time (same fields)

    fx.logs.names(role=None)        registered names (of a role)
    fx.logs.info(name)              registry entry: role, backbone, layer, editor, family (stream family in the key cache), T, path, exists
    fx.logs.available()             {name: directory complete}   (which logs exist now)
    fx.logs.stored(name)            the deployed log's own stored metrics (rescope.stored_metrics; no model)
    fx.logs.open(name)              LogData: edit log (factors, own keys, batches), key cache, edits -> cache rows, pools, near-scope flags
    fx.logs.load_model(name, ...)   the harness handle through the library's scale-safe loader (lean, no harness baseline for T >= 500)

A SEQUENCE is one row of the backbone's key cache (rescope.KeyCache.index); everywhere in fx a sequence is addressed by its
POSITION j in that index (an int) and identified on disk by its cache sid (LogData.sid(j), unique within the backbone).
"""
from __future__ import annotations

import builtins
import json
import os
import re

import fx

ROLES = {
    "development": ["TB-Q100-ae-l2_3", "TB-L100-ae-l2_3"],
    "working_point": ["TB-Q100-ae", "TB-Q100-ae-l2_1", "TB-L100-ae"],
    "unseen_alphaedit": ["TB-Q200-ae-l2_3", "TB-Q500-ae-l2_3", "TB-L200-ae-l2_3", "TB-L500-ae-l2_3"],
    "unseen_memit": [f"TB-Q{t}-memitw" for t in (100, 200, 500, 1000, 2000)] + [f"TB-L{t}-memitw" for t in (100, 200, 500, 1000)],
    "internal": ["TB-Q1000-ae-l2_3", "TB-Q2000-ae-l2_3", "TB-L1000-ae-l2_3"],
    "stress": ["TB-Q100-mmke-l2_3", "TB-Q500-mmke-l2_3"],
}
ROLE_NOTE = {"development": "method development happened here; the only logs with numbers looked at before the final runs",
             "working_point": "working points of the development streams (ridge l2 = 10 / 1)",
             "unseen_alphaedit": "unseen AlphaEdit-style logs ([redacted: describes a related submission under double-blind review]), T <= 500",
             "unseen_memit": "unseen MEMIT-style logs in the public configuration (Wikipedia statistics, lam_hat = 100)",
             "internal": "AlphaEdit-style logs beyond T = 500: internal consistency check, never reported",
             "stress": "MMKE-Bench logs: Rel exact 0, irreproducible targets; separate stress test"}
BACKBONE = {"Q": ("qwen25vl", 20, 10500), "L": ("llava", 16, 17000)}          # letter -> (backbone, edited layer, gq --mem MiB)
NEED = ("edit_batch.jsonl", "metrics.json", "composite_delta.pt", "edit_factors.pt", "per_edit_metrics.jsonl",
        "probe_records_v2.jsonl", "complaints_D1_v2.jsonl", "complaints_D2_v2.jsonl", "complaints_v2.json")      # == scale_registry._NEED
EDIT_KINDS = ("rel", "tgen", "igen")
HN_KINDS = ("hn_i", "hn_q", "hn_x")
POOLS = ("repair", "heldout", "heldout_disjoint", "heldout_strict", "all")
_KC, _OPEN, _STORED = {}, {}, {}


def _entry(name: str, role: str) -> dict:
    m = re.match(r"^TB-([QL])(\d+)-(.+)$", name)
    bb, layer, mem = BACKBONE[m.group(1)]
    tail = m.group(3)
    e = {"name": name, "role": role, "backbone": bb, "layer": layer, "T": int(m.group(2)), "gq_mem": mem, "family": m.group(1), "dataset": "VLKEB",
         "reported": role != "internal", "unseen": role not in ("development", "working_point"), "path": os.path.join(fx.TESTBED, name)}
    if tail == "memitw":
        e.update(editor_family="memit", editor="MEMIT-style, public default configuration (Wikipedia-text statistics, lam_hat = 100)", statistics="see README (related submission)",
                 l2=None)
    else:
        l2 = re.search(r"l2_([0-9.]+)$", tail)
        e.update(editor_family="alphaedit", statistics="see README (related submission)", l2=float(l2.group(1)) if l2 else 10.0)
        e["editor"] = f"AlphaEdit-style, [redacted: describes a related submission under double-blind review], ridge l2 = {e['l2']:g}"
        if tail.startswith("mmke"):
            e.update(family="QK", dataset="MMKE-Bench")
    return e


def _cut_by_r19(name: str, role: str) -> bool:
    """Amendment R19 (2026-11-11 08:49, reduction of the programme): no final experiment runs on this log."""
    return role in ("internal", "stress", "unseen_onehop", "scout") or bool(re.match(r"^TB-[QL]200-memitw$", name))


REGISTRY = {n: _entry(n, role) for role, ns in ROLES.items() for n in ns}
for _e in REGISTRY.values():
    # cache = directory of the key cache under 10_final/cache; setup = registry to import first
    _e.update(cache=_e["backbone"], setup=None, cut_by_R19=_cut_by_r19(_e["name"], _e["role"]))
PART3 = fx.EXP + "/09_testbed_scale"                   # the scout's logs: registry_part3.json + code_part3/registry_part3.py


def register(name: str, role: str, backbone: str, layer: int, family: str, T: int, cache: str | None = None, dataset: str = "VLKEB",
             editor_family: str = "alphaedit", editor: str = "", l2=None, statistics=None, gq_mem: int = 10500, reported=None, setup=None, path=None,
             **extra) -> dict:
    """Add a log to the registry at run time (e.g. a log built after this package was written).  family: the stream family of its
    sequences in the key cache ('Q', 'L', 'QK', 'QO', 'B'); cache: directory name of that key cache under 10_final/cache (default: the
    backbone); path: its directory (default: 06_testbed_v3/<name>; the LIBRARY opens a log by its name, so the library must find it
    too); setup: None | 'part3' (import 09_testbed_scale/code_part3/registry_part3 before the library touches the log) | a callable
    without arguments (called before the library touches the log, e.g. the import of a further testbed registry).  An existing name
    must not change.  Further keyword arguments go into the entry."""
    e = dict(name=name, role=role, backbone=backbone, layer=int(layer), T=int(T), gq_mem=int(gq_mem), family=family, dataset=dataset, reported=reported,
             unseen=role not in ("development", "working_point"), path=path or os.path.join(fx.TESTBED, name), editor_family=editor_family, editor=editor,
             statistics=statistics, l2=l2, cache=cache or backbone, setup=setup, **extra)
    e.setdefault("cut_by_R19", _cut_by_r19(name, role))
    if name in REGISTRY and REGISTRY[name] != e:
        raise ValueError(f"{name} is already registered with other fields")
    REGISTRY[name] = e
    ROLES.setdefault(role, [])
    if name not in ROLES[role]:
        ROLES[role].append(name)
    return e


def _register_scout():
    p = os.path.join(PART3, "registry_part3.json")
    if not os.path.exists(p):
        return
    with builtins.open(p) as f:
        reg = json.load(f)
    for n, c in reg.items():
        m = re.match(r"^TB-[A-Z]+(\d+)-", n)
        if m is None:
            continue
        bb = c["backbone"]
        role = "stress" if bb == "blip2" else ("unseen_onehop" if "-onehop-" in n else "scout")
        register(n, role, bb, c["layer"], c["family"], int(m.group(1)), cache="blip2" if bb == "blip2" else str(c["family"]), dataset=c.get("dataset", "VLKEB"),
                 editor_family="memit" if c.get("editor") == "memit" else "alphaedit", l2=c.get("l2"), statistics=c.get("source"),
                 editor=f"AlphaEdit-style, {c.get('source')} statistics, ridge l2 = {c.get('l2')}", gq_mem=BACKBONE.get(c["family"][0], (None, None, 10500))[2] if bb != "blip2" else 10500,
                 reported=None, setup="part3")


ROLE_NOTE["stress"] += "; BLIP2 logs (TB-B*): the edits were written but did not take effect, stress logs of collateral damage"
ROLE_NOTE["unseen_onehop"] = "VLKEB one-hop split on Qwen: unseen, frozen row set of E3; calibration only from the REPAIR pool of their own probe rows"
ROLE_NOTE["scout"] = "further logs of the scout's registry: unseen, frozen row set at alpha = 0.05 only; reporting undecided"
for _r in ("internal", "stress", "unseen_onehop"):
    ROLE_NOTE[_r] += "; no final experiment runs on these logs (amendment R19 of 2026-11-11 08:49)"
_register_scout()


def setup(name: str):
    """Make a log known to the testbed registry the library reads (only the scout's logs need it: their registry module also
    installs the BLIP2 / one-hop support in-process; a log registered with a callable: that callable)."""
    if callable(REGISTRY[name].get("setup")):
        REGISTRY[name]["setup"]()
    elif REGISTRY[name].get("setup") == "part3":
        import sys
        for d in (fx.EXP + "/07_repair_v3", PART3 + "/code_part3"):
            if d not in sys.path:
                sys.path.insert(0, d)
        import rescope
        rescope.harness()                               # the library's private harness cache first
        import registry_part3  # noqa: F401


PROGRAMME = {"development": ("TB-Q100-ae-l2_3", "TB-L100-ae-l2_3"),
             "working_point": ("TB-Q100-ae", "TB-Q100-ae-l2_1", "TB-L100-ae"),
             "unseen": ("TB-Q200-ae-l2_3", "TB-Q500-ae-l2_3", "TB-Q100-memitw", "TB-Q500-memitw", "TB-Q1000-memitw", "TB-Q2000-memitw",
                        "TB-L200-ae-l2_3", "TB-L500-ae-l2_3", "TB-L100-memitw", "TB-L500-memitw", "TB-L1000-memitw")}


def programme(part=None) -> dict:
    """The logs of the REDUCED programme (amendment R19 b) with their roles: {name: registry role}; part: 'development' |
    'working_point' | 'unseen' (the reported unseen logs: T = 200 / 500 AlphaEdit-style, the MEMIT-style family without T = 200).
    The prefix logs of R12 are fx.logs.names('prefix')."""
    parts = PROGRAMME if part is None else {part: PROGRAMME[part]}
    return {n: REGISTRY[n]["role"] for ns in parts.values() for n in ns}


def names(role=None) -> list:
    """Registered log names; role: one of ROLES or a tuple of roles."""
    if role is None:
        return list(REGISTRY)
    roles = (role,) if isinstance(role, str) else tuple(role)
    for r in roles:
        if r not in ROLES:
            raise KeyError(f"role {r!r}: one of {list(ROLES)}")
    return [n for r in roles for n in ROLES[r]]


def exists(name: str) -> bool:
    """The log's directory holds every file of the testbed layout (the same list as scale_registry.available)."""
    d = REGISTRY[name]["path"] if name in REGISTRY else os.path.join(fx.TESTBED, name)
    need = REGISTRY[name].get("need", NEED) if name in REGISTRY else NEED          # entry field `need`: the files of a derived log (prefix logs)
    return all(os.path.exists(os.path.join(d, f)) for f in need) and os.path.isdir(os.path.join(d, "batch_deltas"))


def info(name: str) -> dict:
    """Registry entry of a log + `exists` + (if it exists) `T_stored` = n_edits of its metrics.json (must equal T)."""
    if name not in REGISTRY:
        raise KeyError(f"{name!r} is not a registered log (fx.logs.names(); fx.logs.register adds one); legacy / text logs are not registered on purpose")
    e = dict(REGISTRY[name], exists=exists(name))
    if e["exists"]:
        with builtins.open(os.path.join(e["path"], "metrics.json")) as f:
            e["T_stored"] = int(json.load(f)["n_edits"])
        if e["T_stored"] != e["T"]:
            raise RuntimeError(f"{name}: metrics.json says {e['T_stored']} edits, the name says {e['T']}")
    return e


def available() -> dict:
    """{name: bool} for every registered log (the MEMIT-style family is being completed: call again later)."""
    return {n: exists(n) for n in REGISTRY}


def table() -> str:
    """Plain-text table of the registry and of which logs exist now."""
    rows = [(n, e["role"], e["backbone"], e["editor_family"], e["family"], str(e["T"]), "yes" if exists(n) else "MISSING",
             "; ".join(x for x in ({True: "", False: "never reported", None: ""}[e["reported"]], "not run (R19)" if e.get("cut_by_R19") else "") if x))
            for n, e in REGISTRY.items()]
    head = ("log", "role", "backbone", "editor", "cache family", "T", "exists", "note")
    w = [max(len(r[i]) for r in rows + [head]) for i in range(len(head))]
    return "\n".join("  ".join(c.ljust(w[i]) for i, c in enumerate(r)).rstrip() for r in [head] + rows)


def stored(name: str, pools: bool = False) -> dict:
    """The deployed log's own stored metrics (rescope.stored_metrics: vendor pass of the testbed build for edits and benchmark
    locality, decoder-v2 probe records; no model, no harness baseline).  pools=True adds the per-pool probe counts (CPU: loads
    the stream and the harness split)."""
    info(name)
    key = (name, bool(pools))
    if key not in _STORED:
        import rescope
        if pools:
            setup(name)
        _STORED[key] = rescope.stored_metrics(name, pools=pools)
    return _STORED[key]


def keycache(cache: str):
    """The key cache 10_final/cache/<cache> (a backbone, or the family of a further benchmark; one object per process)."""
    if cache not in _KC:
        from rescope import KeyCache
        from rescope.keycache import DEFAULT_ROOT
        kc = KeyCache(cache)
        final = os.path.realpath(kc.dir) == os.path.realpath(os.path.join(DEFAULT_ROOT, cache))
        if not final or not kc.meta.get("complete"):
            raise RuntimeError(f"the key cache {cache} is not a complete final cache ({kc.dir})")
        _KC[cache] = kc
    return _KC[cache]


def norm_q(s) -> str:
    """Question-string identity (10_final/cache/code/fc_common.norm_q): stripped, lower case."""
    return (s or "").strip().lower()


def load_model(name: str, device: str = "cuda:0", mem_gb=None, reuse=None):
    """The harness handle of a log through the library's scale-safe loader (rescope.load: lean for T >= 500, the harness
    baseline / evalcache / all-position edit keys are refused there).  mem_gb=None: the cap of the GPU queue, (GQ_MEM - 600) MiB,
    if the process runs under `gq run`, else the queue's figure of the backbone."""
    import rescope
    e = info(name)
    if not e["exists"]:
        raise FileNotFoundError(f"log {name} does not exist yet ({e['path']})")
    if mem_gb is None:
        mem_gb = (int(os.environ.get("GQ_MEM", e["gq_mem"])) - 600) / 1024.0
    if callable(e.get("loader")):                                       # a log the library cannot find by name (prefix logs: fx.prefix)
        return e["loader"](name, device=device, mem_gb=mem_gb, reuse=reuse)
    setup(name)
    return rescope.load(name, device=device, mem_gb=mem_gb, reuse=reuse)


class LogData:
    """One log without a model: the edit log, the key cache of its backbone and the rows of that cache that belong to it.

      name, role, reg (registry entry), backbone, layer, family, T, uids
      log            rescope.EditLog (CPU, fp64): U (T, m), W (T, d), K (T, d) stored solver keys, batch (T,)
      kc, index      rescope.KeyCache of the backbone and its index (list of dicts); a sequence = a position j in it
      edit_rows      {'rel' | 'tgen' | 'igen': [position of the sequence of edit 0, 1, ..., T - 1]}      (edits -> cache rows)
      probe_rows     positions of the family's 1000 probe rows (valid or not), index order
      hn_rows        positions of the hard negatives of the log's edits (HN-I, HN-Q; HN-X with both edits in the log), valid or not
      benign_rows    positions of the backbone's label-free benign pool
      rows(kind, pool=, valid=, where=)   selection;  pool(name, valid=True)   probe rows of repair | heldout | heldout_disjoint |
                     heldout_strict | all
      sid(j), lib_sid(j), own(j)          cache sid, the library's sid of the same item, the row's own edit in THIS log (-1: none)
      near_edit(j)   first edit of THIS log whose question is the row's question string (None: none): probes that share a question
                     string with a logged edit are near-scope queries for the log (amendment of 22:10, item 5); rephrase=True: also
                     the edits' text rephrases
      probe_rec(j)   the log's own decoder-v2 record of a probe row (is_complaint_D1 / D2 under the DEPLOYED log, ...)
      own_keys()     (keys (T, d) float32, edits) of the edits' own B = 1 Rel prompts (teacher-forced), for repair.add_guards
      targets        the edits' targets, in edit order
      stored(pools)  the deployed log's own stored metrics
    """

    def __init__(self, name: str, verify: bool = True):
        from rescope import from_testbed
        self.reg = info(name)
        if not self.reg["exists"]:
            raise FileNotFoundError(f"log {name} does not exist yet ({self.reg['path']}); fx.logs.available() lists the logs that do")
        self.name, self.role, self.backbone, self.layer = name, self.reg["role"], self.reg["backbone"], self.reg["layer"]
        self.log = from_testbed(self.reg["path"])                       # the registered directory (default: 06_testbed_v3/<name>)
        self.uids, self.T = list(self.log.meta["uids"]), int(self.log.T)
        if self.T != self.reg["T"] or self.log.meta.get("backbone") != self.backbone or int(self.log.meta.get("layer")) != self.layer:
            raise RuntimeError(f"{name}: the stored log (T = {self.T}, {self.log.meta.get('backbone')} L{self.log.meta.get('layer')}) does not match the registry")
        self.kc = keycache(self.reg["cache"])
        self.index = self.kc.index
        if int(self.kc.layer) != self.layer or self.kc.backbone != self.backbone:
            raise RuntimeError(f"{name}: key cache of {self.kc.backbone} layer {self.kc.layer}, log of {self.backbone} edited at layer {self.layer}")
        self.family = self.kc.family_of(self.uids)
        if self.family != self.reg["family"]:
            raise LookupError(f"{name}: its edit stream is not the prefix of the cached family {self.reg['family']!r} (found: {self.family!r})")
        ix, T, fam = self.index, self.T, self.family
        self.edit_rows = {}
        for m in EDIT_KINDS:
            pos = {}
            for j in self.kc.rows(kind=m, family=fam, T=T):
                pos.setdefault(int(ix[j]["edit"]), j)
            miss = [i for i in range(T) if i not in pos or ix[pos[i]].get("edit_uid") != self.uids[i] or ix[pos[i]].get("row_tf") is None]
            if miss:
                raise LookupError(f"{name}: {len(miss)} '{m}' sequences are not in the key cache or carry another uid (e.g. edits {miss[:5]})")
            self.edit_rows[m] = [pos[i] for i in range(T)]
        self.probe_rows = self.kc.rows(kind="probe", family=fam)
        self.hn_rows = self.kc.rows(kind=HN_KINDS, family=fam, T=T)
        self.benign_rows = self.kc.rows(kind="benign")
        self._lib = {}
        for j in [j for m in EDIT_KINDS for j in self.edit_rows[m]] + self.probe_rows + self.hn_rows:
            if ix[j]["lib_sid"] in self._lib:
                raise RuntimeError(f"{name}: two cached sequences with the library sid {ix[j]['lib_sid']}")
            self._lib[ix[j]["lib_sid"]] = j
        if self.reg["cache"] != self.backbone:                          # a further cache of the same backbone shares the backbone's memo:
            bad = [ix[j]["sid"] for j in self._lib.values() if not ix[j]["sid"].startswith(fam + ":")]      # its sequence ids must be its own
            if bad:
                raise RuntimeError(f"{name}: sequence ids of cache {self.reg['cache']} without the family prefix (e.g. {bad[:3]}) could collide in the memo")
        self._q_edit, self._q_edit_reph = {}, {}
        for i in range(T):                                              # question strings of the log's edits: first edit that asks them
            self._q_edit.setdefault(norm_q(ix[self.edit_rows["rel"][i]]["prompt"]), i)           # the edit's own question
            for m in ("rel", "tgen"):                                   # its question or its text rephrase
                self._q_edit_reph.setdefault(norm_q(ix[self.edit_rows[m][i]]["prompt"]), i)
        self._own = {u: i for i, u in enumerate(self.uids)}
        self._probe_rec, self._own_keys, self.checks = None, None, {}
        if verify:
            self.verify()

    # ------------------------------------------------------------------ checks
    def _records(self) -> dict:
        if self._probe_rec is None:
            with builtins.open(os.path.join(self.reg["path"], "probe_records_v2.jsonl")) as f:
                recs = [json.loads(line) for line in f if line.strip()]
            self._probe_rec = {}
            for r in recs:
                j = self._lib.get(f"probe:{r['ptype']}:{r['probe_idx']}")
                if j is None:
                    raise LookupError(f"{self.name}: probe row {r['group']} is not in the key cache")
                self._probe_rec[j] = r
        return self._probe_rec

    def verify(self) -> dict:
        """The cache rows ARE the log's sequences: probe rows against the log's own probe_records_v2.jsonl (item, question, image,
        pre-edit answer and its validity, pre-edit 48-token correctness), edit rows against the stored uids (done at construction).
        Raises on any difference."""
        recs, ix = self._records(), self.index
        bad = []
        for j, r in recs.items():
            e = ix[j]
            same = (e["uid"] == r["uid"] and e["prompt"] == r["prompt"] and e["image"] == r["image"] and (e["y0_text"] or "") == (r["y0_text"] or "")
                    and bool(e["y0_valid"]) == bool(r["y0_valid"]) and bool(e.get("pre_correct_long")) == bool(r["pre_correct_long"]) and e["group"] == r["group"])
            if not same:
                bad.append(r["group"])
        if bad or len(recs) != len(self.probe_rows):
            raise RuntimeError(f"{self.name}: {len(bad)} probe rows of the log differ from the key cache (e.g. {bad[:5]}); rows {len(recs)} vs {len(self.probe_rows)}")
        self.checks = {"n_probe_rows_equal_to_cache": len(recs), "n_edit_sequences": {m: len(v) for m, v in self.edit_rows.items()},
                       "family": self.family, "cache_index_md5": self.kc.meta.get("index_md5")}
        return self.checks

    # ------------------------------------------------------------------ rows
    def seq(self, j: int) -> dict:
        return self.index[j]

    def sid(self, j: int) -> str:
        return self.index[j]["sid"]

    def lib_sid(self, j: int):
        return self.index[j].get("lib_sid")

    def by_lib_sid(self, sid: str) -> int:
        """Position of the sequence with the library sid ('edit:rel:17', 'probe:image:3', a hard-negative id) in this log."""
        return self._lib[sid]

    def own(self, j: int) -> int:
        """The row's own edit in this log (edit sequences: their edit; hard negatives: the edit whose image (HN-I, HN-X) or
        question (HN-Q) the row uses); -1 for probes and benign rows."""
        return self._own.get(self.index[j].get("edit_uid"), -1)

    def near_edit(self, j: int, rephrase: bool = False):
        """First edit of this log whose QUESTION equals the row's question string (stripped, lower case); None if none.  This is the
        near-scope flag of the protocol (amendment of 2026-11-10 22:10, item 5: 677 of the 2000 edits of the long Qwen stream ask a
        probe question; checked in tests/t_cpu.py).  rephrase=True: the edit's question OR its text rephrase (the definition of the
        key cache's flags q_in_edit / q_in_edit_first_pos of hard negatives and benign rows)."""
        return (self._q_edit_reph if rephrase else self._q_edit).get(norm_q(self.index[j].get("prompt")))

    def probe_rec(self, j: int) -> dict:
        return self._records()[j]

    def rows(self, kind, pool=None, valid=None, where=None) -> list:
        """Positions of the log's sequences.  kind: 'probe' | 'rel' | 'tgen' | 'igen' | 'edit' (the three, metric-major) | 'hn' | 'hn_i' |
        'hn_q' | 'hn_x' | 'benign', or a tuple of them.  pool (probes): repair | heldout | heldout_disjoint | heldout_strict | all.
        valid: True / False = rows whose pre-edit answer is / is not valid (probes, hard negatives, benign answer subset).
        where: predicate on the index row."""
        out = []
        for k in ((kind,) if isinstance(kind, str) else tuple(kind)):
            if k == "probe":
                sub = self.probe_rows
            elif k in EDIT_KINDS:
                sub = self.edit_rows[k]
            elif k == "edit":
                sub = [j for m in EDIT_KINDS for j in self.edit_rows[m]]
            elif k == "hn":
                sub = self.hn_rows
            elif k in HN_KINDS:
                sub = [j for j in self.hn_rows if self.index[j]["kind"] == k]
            elif k == "benign":
                sub = self.benign_rows
            else:
                raise ValueError(f"kind {k!r}")
            out += sub
        if pool is not None and pool != "all":
            if pool not in POOLS:
                raise ValueError(f"pool: one of {POOLS}")
            f = {"heldout_disjoint": lambda e: bool(e.get("disjoint")), "heldout_strict": lambda e: bool(e.get("strict"))}.get(pool, lambda e: e.get("pool") == pool)
            out = [j for j in out if self.index[j]["kind"] == "probe" and f(self.index[j])]
        if valid is not None:
            out = [j for j in out if bool(self.index[j].get("y0_valid")) == valid]
        if where is not None:
            out = [j for j in out if where(self.index[j])]
        return out

    def pool(self, pool: str = "heldout_disjoint", valid: bool = True) -> list:
        """Probe rows of a pool (default: rows with a valid pre-edit answer, the rows of every D1 / locality number)."""
        return self.rows("probe", pool=pool, valid=True if valid else None)

    def edits_of(self, metric: str, edits=None) -> list:
        """Sequences of a metric for a subset of edits (default all), in edit order."""
        r = self.edit_rows[metric]
        return list(r) if edits is None else [r[int(i)] for i in edits]

    # ------------------------------------------------------------------ log quantities
    def own_keys(self, answer=True):
        """(keys (n, d) float32, edit ids): the B = 1 teacher-forced key of every edit's own Rel prompt, from the key cache
        (KeyCache.own_keys == Evaluator.own_keys('rel')): the guards added before every calibration.  answer='all': also the keys
        of the 11 full forwards of a decode of the own prompt (edits whose decode sits on the same token), for gates whose own
        prompts must keep firing in a free generation."""
        if answer is not True:
            return self.kc.own_keys(self.uids, answer=answer)
        if self._own_keys is None:
            self._own_keys = self.kc.own_keys(self.uids)
        return self._own_keys

    @property
    def targets(self) -> list:
        return [self.index[j]["answer"] for j in self.edit_rows["rel"]]

    def stored(self, pools: bool = False) -> dict:
        return stored(self.name, pools)

    def describe(self) -> dict:
        ix = self.index
        hn = {}
        for j in self.hn_rows:
            k = ix[j]["hn_type"] + ("" if ix[j]["y0_valid"] else " (invalid y0)")
            hn[k] = hn.get(k, 0) + 1
        val = self.rows("probe", valid=True)
        pq = {norm_q(ix[j]["prompt"]) for j in self.probe_rows}
        return {"name": self.name, "role": self.role, "backbone": self.backbone, "family": self.family, "T": self.T, "editor": self.reg["editor"],
                "n_batches": int(self.log.batch.max()) + 1, "n_probe_rows": len(self.probe_rows), "n_probe_valid": len(val),
                "pools_valid": {p: len(self.pool(p)) for p in POOLS},
                "near_scope_valid": {p: sum(self.near_edit(j) is not None for j in self.pool(p)) for p in ("repair", "heldout_disjoint")},
                "near_scope_valid_incl_rephrase": {p: sum(self.near_edit(j, True) is not None for j in self.pool(p)) for p in ("repair", "heldout_disjoint")},
                "n_edits_asking_a_probe_question": sum(norm_q(ix[j]["prompt"]) in pq for j in self.edit_rows["rel"]),
                "hard_negatives": hn, "n_benign": len(self.benign_rows),
                "lp_shift_rows": {k: sum(bool(ix[j].get("lp_shift")) for j in self.rows(k)) for k in ("probe", "edit", "hn")}}


def open(name: str, verify: bool = True) -> LogData:        # noqa: A001  (fx.logs.open is the documented name)
    """LogData of a registered log (cached per process)."""
    if name not in _OPEN:
        _OPEN[name] = LogData(name, verify)
    return _OPEN[name]
