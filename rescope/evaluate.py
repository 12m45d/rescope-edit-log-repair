"""Unified evaluation of a method on an edit log: ONE hook code path, harness conventions, plain dict rows.

A METHOD is one of
  None                    the deployed log (module weight W_E)
  a torch.Tensor          a full replacement weight of the edited module (rollback, merged repair, W_0, ...)
  a harness Correction    07_repair_v3/harness (COMP adapter, REEDIT / ROLLBACK weight): applied by harness.applied.
                          REEDIT must be built with chunk=1 (evaluate.reedit does it): with the default chunk=None
                          the harness skips its z* oracle fallback and silently does not install some complaints
                          (P12 Anomaly 2).
  a gate                  a ScopedLog or a baseline gate: module weight W_0 + scoped.Hook
  a Dual                  baseline DUAL (answer match), FREE GENERATION ONLY: the deployed model's answer is returned only if,
                          after the testbed's answer normalisation, it equals (variant 'eq') / contains (variant 'contains')
                          the target of a logged edit; otherwise the pre-edit model's answer.  Two generations per query.
                          (Evaluator.dual_tf_rows: teacher-forced rows of the model that DUAL selects, a labelled composition.)
Every per-item function takes one method or a dict {name: method} (the items are then built once and every
method is run on the same batch); it returns the rows (or {name: rows}).

Conventions (07_repair_v3/harness, 06_testbed_v3):
  teacher forcing   B = 1 full forward of the item's own sequence, logits at the answer-predicting positions;
                    exact = every answer token is the argmax (False if a logit is not finite), token_acc = share.
  edits             Rel (image, src), T-Gen (image, rephrase), I-Gen (rephrased image, src); target = alt.
  probes            valid-y0 rows of probe_records_v2.jsonl (y0 = multi-token decoder-v2 answer of W_0, reproduced
                    by W_0); D1 = y0 no longer reproduced.  Pools: repair | heldout | heldout_disjoint |
                    heldout_strict | all (splits.py); rows carry query_id (identical queries) for cluster bootstraps.
  hard negatives    08_rescope/p11_retrofit/out/hard_negatives (valid rows of this log's edits); D1 as for probes.
                    A missing file, or a file without rows for this log's edits, is an error when hard negatives
                    are asked for (never an empty table).
  D2                rows whose pre-edit 48-token decoder-v2 answer contains the reference; D2 = the 48-token answer
                    under the method does not (tb_common.Prober.decode_v2 + contains_ref: the testbed's pipeline).
  free generation   decoder v2 greedy (10 ids from full no-cache forwards + KV-cache continuation to 48 tokens).
                    `agree` (y == y0) is None on rows without a valid y0.
  token locality    probe and hard-negative rows carry `tok_loc`: the share of answer tokens whose argmax equals the
                    pre-edit model's under teacher forcing of the pre-edit answer (B = 1).  For a valid y0 the pre-edit
                    argmax IS y0 (validity = reproduced by W_0), so tok_loc = token_acc and costs nothing; for a row
                    with an invalid y0 (valid_only=False) the pre-edit argmax comes from one forward at W_0 per item
                    (cached).  A row with a non-finite logit has tok_loc 0.  T-Loc* / I-Loc* = mean over text / image
                    probe rows (locality, Evaluator.token_locality).
Gate methods add firing flags to every row (scoped.Hook.summary); rows are JSON-serialisable.

SCALE.  load() registers the long logs on demand (register_logs: 09_testbed_scale registry, part-2 names, text registry) and
reads logs with T >= 500 without holding their applied batch deltas in memory (same W_0 / W_E bit for bit).  For such logs
the harness baseline / evalcache / all-position edit keys are refused (34 GB of RAM at T = 200): the deployed log's own
metrics come from the testbed's stored files (stored_metrics), every other number from this Evaluator at B = 1.

THE LAST PROMPT POSITION ON THE TWO PATHS.  Teacher forcing uses the harness index of the joint sequence
(answer_start - 1: the position that predicts the first scored answer token; also where the editor took the solver
key).  The decode path uses the last token of the prompt alone, and re-evaluates it in up to 11 full forwards of
growing length (10 no-cache forwards + the first forward of the KV-cache continuation); the low-precision forward
gives each length a slightly different key.  The two positions are the same token unless the tokenizer leaves a
bare space token between prompt and answer: LLaVA, answers that start with a digit (':' + '_' + '3'), 46 of the
1685 sequences of TB-L100-ae-l2_3, none on Qwen.  There the teacher-forced position is ONE TOKEN LATER than the
decode's, the keys differ (|dr| up to 0.8; GPU test b9), a role-'last' gate decides on different tokens on the two
paths, and an edit written at the later token (LLaVA edit 39) cannot fire under role 'last' in a decode.  Every
row carries `lp_shift` (teacher-forced index minus prompt-only index; tokenizer only).  Tools: tf_rows / edits /
probes / hard_negatives(..., lp="prompt") score the teacher-forced sequence with the gate at the DECODE's
position; eligible_keys(..., answer="all") returns every key the item can present to a role-'last' gate on either
path (complaints, calibration and guards for decode metrics; exact), answer="both" the keys of the two sequences
(approximate: use repair.DECODE_MARGIN).  See docs/USAGE.md, rule 3.
The harness is imported with a PRIVATE cache directory; nothing under 02_research/experiments is written.
"""
from __future__ import annotations
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP

import contextlib
import json
import os
import shutil
import sys
import types

import torch

from . import splits
from .scoped import Hook

ROOT = f"{_RP.ROOT}"
EXP = ROOT + "/02_research/experiments"
HARNESS_PARENT = EXP + "/07_repair_v3"
HN_DIR = EXP + "/08_rescope/p11_retrofit/out/hard_negatives"
CACHE = os.environ.get("RESCOPE_CACHE", ROOT + "/02_research/code/rescope/_hcache")
HN_DIR_FINAL = EXP + "/10_final/cache/hard_negatives"      # existing + new hard negatives (HN-I, HN-Q, HN-X), same file format
SCALE = EXP + "/09_testbed_scale"
SCALE_T = 500                      # from this many edits on: lean loading, no harness baseline / evalcache / all-position edit keys
ALLOW_HARNESS_AT_SCALE = False     # set True to lift that guard (34 GB of RAM at T = 200)
EDIT_METRICS = ("rel", "tgen", "igen")
MAX_NEW = 10                       # decoder v2: ids from full no-cache forwards before the KV-cache continuation (tb_common.MAX_NEW)
_HZ = None
_REG = {"logs": {}, "sources": {}}
_NEED = ("edit_batch.jsonl", "metrics.json", "composite_delta.pt", "edit_factors.pt", "per_edit_metrics.jsonl",
         "probe_records_v2.jsonl", "complaints_D1_v2.jsonl", "complaints_D2_v2.jsonl", "complaints_v2.json")      # == scale_registry._NEED


def _guard(fn, what):
    """In-process guard around a harness function that holds every position of every edit sequence in RAM."""
    def guarded(H, *a, **k):
        if not ALLOW_HARNESS_AT_SCALE and int(getattr(H, "T", 0)) >= SCALE_T:
            raise RuntimeError(f"{what} is refused for a log with T = {H.T} >= {SCALE_T} edits (it needs 34 GB of RAM at T = 200). Use "
                               "rescope.Evaluator / SparseEvaluator (B = 1, last-position keys only) and rescope.stored_metrics(name) for the "
                               "deployed log's own metrics; rescope.evaluate.ALLOW_HARNESS_AT_SCALE = True lifts the guard.")
        return fn(H, *a, **k)
    guarded.__wrapped__ = fn
    guarded.__doc__ = fn.__doc__
    return guarded


def _complete(TB, name):
    d = os.path.join(TB.OUT_ROOT, name)
    return all(os.path.exists(os.path.join(d, f)) for f in _NEED) and os.path.isdir(os.path.join(d, "batch_deltas"))


def _register(TB, text: bool = True):
    """Register the log families that exist on disk with tb_common.LOGS (in-process; see register_logs)."""
    src = _REG["sources"]
    known = set(TB.LOGS)

    def note(names, source):
        for n in names:
            _REG["logs"].setdefault(n, {"source": source})

    note([n for n in TB.LOGS if n not in _REG["logs"]], "testbed")
    d = SCALE + "/code"
    if os.path.exists(d + "/scale_registry.py"):                       # 1. the long logs of 09_testbed_scale (registry.json)
        try:
            if d not in sys.path:
                sys.path.insert(0, d)
            import scale_registry
            src["scale"] = {"ok": True, "names": list(scale_registry.NAMES)}
            note(scale_registry.NAMES, "scale")
        except Exception as e:                                          # tolerated: the other sources may still work
            src["scale"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    else:
        src["scale"] = {"ok": False, "error": "09_testbed_scale/code/scale_registry.py not found"}
    # 2. part 2: its registry script appends names to registry.json (read again: names appended after scale_registry was
    #    imported), and plan.json lists every part-2 log; a listed log whose directory is complete is registered as well
    p2 = {"ok": False, "appended": [], "pending": [], "incomplete": []}
    try:
        if os.path.exists(SCALE + "/registry.json"):
            with open(SCALE + "/registry.json") as f:
                reg = _RP.expand(json.load(f))
            for n, cfg in reg.items():
                if n not in TB.LOGS:
                    TB.LOGS[n] = dict(cfg)
                    p2["appended"].append(n)
            note(p2["appended"], "scale")
        plan = SCALE + "/out_part2/plan.json"
        if os.path.exists(plan):
            with open(plan) as f:
                entries = json.load(f).get("registry_entries", {})
            for n, cfg in entries.items():
                if n in TB.LOGS:
                    _REG["logs"].setdefault(n, {})["source"] = "part2" if _REG["logs"].get(n, {}).get("source") != "part2-pending" else "part2-pending"
                elif _complete(TB, n):
                    TB.LOGS[n] = dict(cfg)
                    p2["pending"].append(n)
                    _REG["logs"][n] = {"source": "part2-pending"}
                else:
                    p2["incomplete"].append(n)
            p2["ok"] = True
        else:
            p2["error"] = "09_testbed_scale/out_part2/plan.json not found (no part-2 logs yet)"
    except Exception as e:
        p2["error"] = f"{type(e).__name__}: {e}"
    src["part2"] = p2
    d = SCALE + "/code_text"
    if text:                                                            # 3. text-only logs (also installs the text-only prober)
        if os.path.exists(d + "/text_registry.py"):
            try:
                if d not in sys.path:
                    sys.path.insert(0, d)
                import text_registry
                src["text"] = {"ok": True, "names": list(text_registry.NAMES), "hn_dir": text_registry.HN_DIR}
                note(text_registry.NAMES, "text")
            except Exception as e:
                src["text"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        else:
            src["text"] = {"ok": False, "error": "09_testbed_scale/code_text/text_registry.py not found"}
    note([n for n in TB.LOGS if n not in _REG["logs"]], "other")
    for n, v in _REG["logs"].items():
        v["complete"] = _complete(TB, n)
    return sorted(set(TB.LOGS) - known)


def register_logs(text: bool = True) -> dict:
    """Register every log family with the shared testbed registry (tb_common.LOGS), IN-PROCESS: no file is modified.
      scale   09_testbed_scale/code/scale_registry.py (registry.json: long VLKEB logs, MEMIT-style logs, MMKE-Bench logs);
      part2   the public-default MEMIT family: names that its registry script has appended to registry.json (the file is
              read again at every call) and names of out_part2/plan.json whose log directory already holds every file of
              the testbed layout ('part2-pending': complete on disk, not yet appended by their builder);
      text    09_testbed_scale/code_text/text_registry.py (CounterFact logs; text=False skips it: importing it installs the
              text-only prober and loaders on top).
    Every source is optional: a missing or failing one is reported, not raised (the part-2 registry may not exist yet;
    call again later to pick it up).  load(name) / load_rows(name) / stored_metrics(name) call this on demand for a name
    that is not registered yet, so the default code path of the development logs is untouched.
    -> {'logs': {name: {'source', 'complete'}}, 'sources': {source: status}}."""
    _register(harness().TB, text)
    return _REG


def ensure_registered(name: str):
    """Make `name` known to tb_common.LOGS (registering the scale / part-2 logs, then the text logs, only if needed)."""
    TB = harness().TB
    if name in TB.LOGS or name == "old_W2":
        return
    _register(TB, text=False)
    if name not in TB.LOGS:
        _register(TB, text=True)
    if name not in TB.LOGS:
        raise KeyError(f"unknown log {name!r}: not in tb_common.LOGS, the 09_testbed_scale registry, the part-2 plan or the text registry "
                       f"(sources: {_REG['sources']})")


def harness():
    """The shared repair harness (namespace: pkg, HC, HL, HE, HS, HM, TB, VM), imported once with its cache
    directories re-pointed to CACHE (baseline files are copied, the large read-only caches are symlinked)."""
    global _HZ
    if _HZ is None:
        os.makedirs(CACHE, exist_ok=True)
        src = HARNESS_PARENT + "/harness/cache"
        for f in sorted(os.listdir(src)):
            dst = os.path.join(CACHE, f)
            if os.path.lexists(dst):
                continue
            try:
                if f.startswith("baseline_"):
                    tmp = dst + f".tmp{os.getpid()}"
                    shutil.copy2(os.path.join(src, f), tmp)
                    os.replace(tmp, dst)
                else:
                    os.symlink(os.path.join(src, f), dst)
            except OSError:
                pass
        if HARNESS_PARENT not in sys.path:
            sys.path.insert(0, HARNESS_PARENT)
        import harness as pkg
        from harness import common as HC, evaluation as HE, hlog as HL, methods as HM, sampler as HS
        for m in (HC, HL, HE, HM):
            m.CACHE = CACHE
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if not hasattr(HE.baseline, "__wrapped__"):                   # scale guard (in-process; no harness file is modified)
            HE.baseline = pkg.baseline = _guard(HE.baseline, "harness.baseline")
            HE.evalcache = _guard(HE.evalcache, "harness evalcache (every edit sequence as CPU suffix states)")
            HL.HLog.edit_keys = _guard(HL.HLog.edit_keys, "HLog.edit_keys (every position of every edit sequence)")
        _HZ = types.SimpleNamespace(pkg=pkg, HC=HC, HL=HL, HE=HE, HS=HS, HM=HM, TB=HC.TB, VM=HC.VM)
    return _HZ


class _LazyDeltas:
    """The applied batch deltas of a log read from disk on demand (H.A of a lean handle: rollbacks work, RAM stays flat)."""

    def __init__(self, tl, n):
        self.tl, self.n = tl, n

    def __len__(self):
        return self.n

    def __getitem__(self, b):
        return self.tl.batch_delta(int(b))

    def __iter__(self):
        return (self[b] for b in range(self.n))


def _lean_class(hz):
    """HLog subclass whose load never holds more than ONE applied batch delta (T = 2000: 200 deltas of 90 MB = 18 GB in
    the harness loader).  Same arithmetic as HLog._init_tb (fp32 sum on the model's device, in batch order), hence the same
    W_E bit for bit and the same load-time checks; H.A reads the deltas from disk on demand."""
    HL = hz.HL

    class LeanHLog(HL.HLog):
        def _init_tb(self):
            tl = HL.TestbedLog(self.path)
            self.tl = tl
            self.batches = [list(b["edit_ids"]) for b in tl.per_batch]
            self.batch_of = {i: bi for bi, ids in enumerate(self.batches) for i in ids}
            self.W0 = self.W_ckpt
            S = torch.zeros(self.W0.shape, dtype=torch.float32, device=self.dev)
            for b in range(len(self.batches)):
                S += tl.batch_delta(b).to(self.dev)
            self.A = _LazyDeltas(tl, len(self.batches))
            self.WE = (self.W0.float() + S).to(self.dtype)
            comp = tl.composite_delta().to(self.dev)
            self.checks["WE_float_minus_W0_equals_stored_composite"] = bool(torch.equal(self.WE.float() - self.W0.float(), comp))
            self.C = self.WE.float() - self.W0.float()
            self.checks["C_equals_stored_composite"] = bool(torch.equal(self.C, comp))
            del comp, S
            fac = tl.factors
            self.U = fac["U"].to(self.dev, torch.float64)
            self.Wfac = fac["W"].to(self.dev, torch.float64)
            self.Kp_log = fac["K1_solver"].float().T.contiguous()
            self.checks["factor_uids_match_stream"] = list(fac["uids"]) == [d["uid"] for d in self.items]
            tl._fac = None
    return LeanHLog


def n_edits(name: str) -> int:
    """Number of edits of a testbed log, from its metrics.json (no model)."""
    with open(os.path.join(ROOT, "02_research/experiments/06_testbed_v3", name, "metrics.json")) as f:
        return int(json.load(f)["n_edits"])


def peak_rss_gb() -> float:
    """Peak resident memory of this process so far (GB)."""
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20


def load(name: str, device: str = "cuda:0", mem_gb: float | None = None, keep_batch_deltas: bool = False, lean: bool | None = None, reuse=None):
    """harness.load(name) with a per-process GPU memory cap; the model is at W_E, H.W0 / H.WE are exact.
    Any registered log: the development logs, and (registered on demand, register_logs) the logs of the 09_testbed_scale
    registry, of its part 2 and of the text registry.
    keep_batch_deltas=True keeps the applied batch deltas (needed by the harness rollback methods only).
    lean: load without holding the applied batch deltas in memory (same W_0 / W_E / factors / checks; H.A then reads a delta
    from disk on demand).  Default None = lean for logs with T >= 500 (T = 2000: 18 GB of deltas otherwise).
    reuse: a handle of ANOTHER log of the same backbone and layer that is no longer needed: its model is reused (no second
    copy of the backbone in memory; the edited module is first set back to its W_0).  The old handle must not be used
    afterwards (its model now holds the new log's W_E).
    H.load_info records the mode, the seconds and the peak resident memory of the process after loading."""
    import time
    hz = harness()
    ensure_registered(name)
    if mem_gb is not None and device.startswith("cuda"):
        dev = torch.device(device)
        torch.cuda.set_per_process_memory_fraction(min(1.0, mem_gb * 2 ** 30 / torch.cuda.get_device_properties(dev).total_memory), dev)
    if lean is None:
        lean = name != "old_W2" and n_edits(name) >= SCALE_T
    t0, rss0 = time.time(), peak_rss_gb()
    vlm = None
    if reuse is not None:
        cfg = hz.TB.LOGS[name]
        if reuse.backbone != cfg["backbone"] or reuse.L != int(cfg["layer"]):
            raise ValueError(f"cannot reuse the model of {reuse.name} ({reuse.backbone} L{reuse.L}) for {name} ({cfg['backbone']} L{cfg['layer']})")
        reuse.set_weight(reuse.W0)                                       # the new handle reads W_0 from the module
        vlm = reuse.vlm
    H = _lean_class(hz)(name, device=device, vlm=vlm) if lean else hz.pkg.load(name, device=device, vlm=vlm)
    if reuse is not None and not torch.equal(H.W0, reuse.W0):
        raise RuntimeError("the reused model did not hold the pre-edit weight")
    if not keep_batch_deltas:
        H.A = None
        torch.cuda.empty_cache()
    H.load_info = {"lean": bool(lean), "seconds": time.time() - t0, "peak_rss_gb_before": rss0, "peak_rss_gb": peak_rss_gb(), "T": int(H.T),
                   "reused_model_of": None if reuse is None else reuse.name}
    return H


def load_rows(name: str):
    """(probe rows, item -> pool) of a log WITHOUT a model (CPU): for splits and bookkeeping."""
    hz = harness()
    ensure_registered(name)
    cfg = hz.TB.LOGS[name]
    items, _ = hz.TB.load_stream(cfg["stream"])
    stub = types.SimpleNamespace(_setup={})
    stub.probes, _ = hz.TB.select_probes(items, key_exclude=(cfg["backbone"] == "llava"))
    rows = hz.TB.jsonl_read(os.path.join(hz.HC.TB_ROOT, name, "probe_records_v2.jsonl"))
    return rows, hz.HS.pool_of(stub)


def stored_metrics(name: str, pools: bool = True) -> dict:
    """The DEPLOYED log's own metrics, read from the testbed's stored files (metrics.json, per_edit_metrics.jsonl,
    probe_records_v2.jsonl): no model and no harness baseline (which must not be called for T >= 500).
      edits     Rel / T-Gen / I-Gen exact and token accuracy (percent) and the per-edit flags: the testbed's VENDOR pass
                (teacher forcing in padded batches of 8; a few near-tie edits differ from B = 1), non-finite rows counted;
      locality  the stored benchmark locality T-Loc / I-Loc (percent; vendor pass: argmax agreement with the pre-edit model
                on the benchmark's reference answers, all 500 probe items);
      probes    decoder-v2 probe rows: valid y0, D1 complaints (among valid rows), D2 complaints, by ptype and (pools=True;
                needs the stream and the harness split, CPU) by pool: repair / heldout / heldout_disjoint.
    Per-row flags: 'per_edit' (one dict per edit) and 'probe_rows' (group, ptype, y0_valid, D1, D2, pool, disjoint)."""
    d = os.path.join(EXP, "06_testbed_v3", name)
    with open(os.path.join(d, "metrics.json")) as f:
        m = json.load(f)
    with open(os.path.join(d, "per_edit_metrics.jsonl")) as f:
        pe = [json.loads(line) for line in f if line.strip()]
    with open(os.path.join(d, "probe_records_v2.jsonl")) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    post = m.get("post") or {}
    key = {"rel": "Rel", "tgen": "TGen", "igen": "IGen"}
    edits = {k: {"exact": m.get(f"{v}_exact"), "token": m.get(f"{v}_token"), "n": (post.get(k) or {}).get("n"),
                 "exact_n": (post.get(k) or {}).get("exact_n"), "nonfinite": (post.get(k) or {}).get("nonfinite")} for k, v in key.items()}
    ann = None
    if pools:
        rws, pool = load_rows(name)
        ann = {a["group"]: a for a in splits.annotate(rws, pool)}

    def count(sub):
        v = [r for r in sub if r["y0_valid"]]
        return {"n_rows": len(sub), "n_valid": len(v), "n_D1": sum(bool(r["is_complaint_D1"]) for r in v),
                "D1_rate": (100.0 * sum(bool(r["is_complaint_D1"]) for r in v) / len(v)) if v else None,
                "n_pre_correct": sum(bool(r["pre_correct_long"]) for r in sub), "n_D2": sum(bool(r["is_complaint_D2"]) for r in sub)}
    probes = {"all": count(rows)}
    for pt in sorted({r["ptype"] for r in rows}):
        probes[pt] = count([r for r in rows if r["ptype"] == pt])
    if ann is not None:
        for pl in ("repair", "heldout", "heldout_disjoint"):
            probes[pl] = count([r for r in rows if splits.in_pool(ann[r["group"]], pl)])
    return {"name": name, "T": int(m["n_edits"]), "config": m.get("config"), "dtype": m.get("dtype"), "edits": edits,
            "pre_edits": m.get("pre"),
            "locality": {"T-Loc": m.get("TLoc_probes"), "I-Loc": m.get("ILoc_probes"), "T-Loc_em": m.get("TLoc_probes_em"),
                         "I-Loc_em": m.get("ILoc_probes_em"), "nonfinite": m.get("loc_nonfinite")},
            "probes": probes,
            "per_edit": [{"edit": int(r["edit_id"]), "uid": r["uid"], "batch": r.get("batch"),
                          **{f"{k}_exact": r.get(f"post_{k}_exact") for k in key}, **{f"{k}_token": r.get(f"post_{k}_token") for k in key},
                          "pre_rel_exact": r.get("pre_rel_exact")} for r in pe],
            "probe_rows": [{"group": r["group"], "ptype": r["ptype"], "y0_valid": bool(r["y0_valid"]), "D1": bool(r["is_complaint_D1"]),
                            "D2": bool(r["is_complaint_D2"]), "pre_correct": bool(r["pre_correct_long"]),
                            **({"pool": ann[r["group"]]["pool"], "disjoint": ann[r["group"]]["disjoint"]} if ann is not None else {})} for r in rows],
            "source": {"dir": d, "edits_and_locality": "vendor pass of the testbed build (padded batches of 8, reference answers for Loc)",
                       "probes": "decoder-v2 probe records (B = 1 teacher forcing of the multi-token pre-edit answer)"}}


def load_hard_negatives(H, hn_dir: str = HN_DIR, missing_ok: bool = False) -> list:
    """Valid-y0 hard-negative rows of this log's edits (+ 'edit' = edit index).  A missing file raises
    FileNotFoundError (missing_ok=True: []), so that a wrong path never looks like "no hard negatives".
    hn_dir: HN_DIR (the P11 rows: HN-I, HN-Q of the first 100 edits) or HN_DIR_FINAL (existing + new rows, with the cross
    rows HN-X: image of edit a, question of edit b; kept only if BOTH edits are in the log, 'edit_b' = index of b).
    Rows of the final set say how they relate to the long edit stream (src_related_edit_first_pos, q_in_edit_first_pos);
    Evaluator.hn_items turns that into per-log item fields (src_related_to_log, q_edit) and does NOT drop such rows."""
    p = os.path.join(hn_dir, H.backbone, "rows.jsonl")
    if not os.path.exists(p):
        if missing_ok:
            return []
        raise FileNotFoundError(f"no hard-negative file for backbone {H.backbone!r}: {p}")
    idx = {d["uid"]: i for i, d in enumerate(H.items)}
    out = []
    with open(p) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r["edit_uid"] in idx and r["y0_valid"]:
                    if r.get("edit_b_uid") is not None:                  # a cross row needs both of its edits in the log
                        if r["edit_b_uid"] not in idx:
                            continue
                        r = dict(r, edit_b=idx[r["edit_b_uid"]])
                    out.append(dict(r, edit=idx[r["edit_uid"]]))
    return out


def reedit(H, complaints, variant: str = "k0+kp", **kw):
    """The harness REEDIT correction of a list of complaints, built as P12 built it -> (Correction, info).
    1. z* of every complaint on its own (B = 1 z* optimisation: the memory of one sequence's backward pass), with
       chunk=1.  chunk=1 is REQUIRED: the harness' documented z* oracle fallback ladder only runs when `chunk` is given;
       with the default chunk=None a complaint whose first z* attempt is not an oracle is silently not installed
       (P12 Anomaly 2).  A direct harness.repair("REEDIT", H, complaints, chunk=1) is the same method, but its first
       z* attempt batches all complaints of a modality (several GB more GPU memory).
    2. ONE joint install of all targets (harness.repair("REEDIT", ..., targets=..., chunk=1)).
    info: targets that are oracles, attempts per complaint, complaints installed / not installed."""
    hz = harness()
    cs = list(complaints)
    tg, att = [], []
    for c in cs:
        one, _ = hz.pkg.repair("REEDIT", H, [c], variant=variant, return_targets=True, chunk=1, **kw)
        tg.append(one.meta.pop("_targets")[0])
        att.append(int(one.meta["zopt_attempts"][0]))
    corr, cost = hz.pkg.repair("REEDIT", H, cs, variant=variant, targets=tg, chunk=1, **kw)
    return corr, {"n": len(cs), "n_oracle_targets": sum(bool(t.oracle) for t in tg), "zopt_attempts": att,
                  "installed_rows": [int(j) for j in corr.meta.get("installed_rows", [])],
                  "not_installed_rows": [int(j) for j in corr.meta.get("not_installed_rows", [])], "variant": variant, "chunk": 1}


def degenerate(y: str, long_ids) -> dict:
    """Degenerate-output flags of a decode (counted in every generation table)."""
    ys, run, prev, rep = y.strip(), 0, None, False
    for t in long_ids:
        run = run + 1 if t == prev else 1
        prev = t
        if run >= 10:
            rep = True
            break
    return {"empty": not ys, "qmark": ys == "?", "The": ys == "The", "single_char": len(ys) == 1, "repeat": rep}


class Dual:
    """Baseline DUAL (answer match), a method for FREE GENERATION (Evaluator.generate / d2): run the deployed and the
    pre-edit model on the query (decoder v2, greedy) and return the deployed model's answer only if it matches the target
    of a logged edit, otherwise the pre-edit model's answer.
      variant 'eq'        norm_answer(deployed answer) == norm_answer(target of some edit), the answer being the first line of
                          the 48-token decode (on='line', default: the comparison behind the row field `eq`) or the
                          10-token decoder-v2 answer (on='y': the comparison behind `eq10`);
              'contains'  contains_ref(deployed 48-token answer, target of some edit): normalised word-boundary containment
                          (the comparison behind `contains`).
    targets: the edit targets in edit order (Dual.of(H): H.items[i]['alt']); targets that normalise to the empty string
    never match (counted).  Needs the list of targets (not only the factors) and TWO generations per query.
    Rows: the decode row of the RETURNED answer + `fired` (= the deployed answer was released), `dual` (match, matching
    edits, own = the item's own edit is among them, source), `n_generations` = 2, `n_new_tokens` = [deployed, pre-edit]."""

    def __init__(self, targets, variant: str = "eq", on: str = "line", name: str | None = None):
        if variant not in ("eq", "contains"):
            raise ValueError("variant: 'eq' | 'contains'")
        if on not in ("line", "y"):
            raise ValueError("on: 'line' | 'y'")
        self.targets, self.variant, self.on = list(targets), variant, on
        self.name = name or f"dual-{variant}" + ("" if on == "line" or variant != "eq" else "-y")
        self._norm = None

    @staticmethod
    def of(H, variant: str = "eq", on: str = "line"):
        return Dual([d["alt"] for d in H.items], variant, on)

    def match(self, TB, line: str, long: str, y: str | None = None) -> list:
        """Indices of the logged edits whose target the deployed answer matches (line / long / y: the fields of its decode row)."""
        if self._norm is None:
            self._norm = ([TB.norm_answer(t) for t in self.targets], [TB.norm_answer(TB.ref_core(t)) for t in self.targets])
        if self.variant == "eq":
            nl = TB.norm_answer(line if self.on == "line" else y)
            return [i for i, t in enumerate(self._norm[0]) if t and t == nl]
        p = f" {TB.norm_answer(long)} "                                  # == tb_common.contains_ref(long, target), target by target
        return [i for i, r in enumerate(self._norm[1]) if r and f" {r} " in p]

    def n_empty_targets(self, TB) -> int:
        self.match(TB, "", "")
        return sum(not t for t in self._norm[0 if self.variant == "eq" else 1])


class Evaluator:
    def __init__(self, H, hard_negatives: bool = True, hn_dir: str | None = None):
        """hard_negatives=False does not load them; a missing file is reported when they are first asked for.
        hn_dir: directory of the hard-negative files (default HN_DIR; HN_DIR_FINAL = existing + new rows incl. HN-X)."""
        self.hn_dir = HN_DIR if hn_dir is None else hn_dir
        self.H, self.hz = H, harness()
        self.TB = self.hz.TB
        self.pool = self.hz.HS.pool_of(H)
        self._ann = {(a["probe_idx"], a["ptype"]): a for a in splits.annotate(H.rows, self.pool)}
        self.hn, self._hn_error = [], None if hard_negatives else RuntimeError("hard negatives were not loaded: Evaluator(H, hard_negatives=False)")
        if hard_negatives:
            try:
                self.hn = load_hard_negatives(H, self.hn_dir)
                if not self.hn:                                         # a file without rows for THIS log's edits is no table either
                    self._hn_error = LookupError(f"no valid hard-negative rows for the edits of {H.name} under {self.hn_dir}/{H.backbone}")
            except FileNotFoundError as e:                              # logs without hard negatives stay usable for everything else
                self._hn_error = e
        self._hook = None
        self._shift = {}
        self._pre = {}                      # sid -> pre-edit argmax ids of the teacher-forced answer (rows with an invalid y0)

    # ------------------------------------------------------------------ items
    def edit_items(self, metrics=EDIT_METRICS, edits=None) -> list:
        ids = range(self.H.T) if edits is None else edits
        return [{"kind": "edit", "sid": f"edit:{m}:{i}", "metric": m, "edit": int(i), "uid": self.H.items[i]["uid"]}
                for m in metrics for i in ids]

    def probe_items(self, pool: str = "all", valid_only: bool = True, pre_correct: bool = False) -> list:
        out = []
        for r in self.H.rows:
            a = self._ann[(r["probe_idx"], r["ptype"])]
            if splits.in_pool(a, pool) and (r["y0_valid"] or not valid_only) and (r["pre_correct_long"] or not pre_correct):
                out.append({"kind": "probe", "sid": f"probe:{r['ptype']}:{r['probe_idx']}", "probe_idx": r["probe_idx"],
                            "ptype": r["ptype"], "uid": r["uid"], "group": r["group"], "pool": a["pool"], "disjoint": a["disjoint"],
                            "strict": a["strict"], "question_in_repair": a["question_in_repair"], "query_id": a["query_id"],
                            "y0_valid": bool(r["y0_valid"]), "D1_WE": bool(r["is_complaint_D1"]), "D2_WE": bool(r["is_complaint_D2"])})
        return out

    def hn_items(self) -> list:
        if self._hn_error is not None:
            raise self._hn_error
        T = self.H.T
        out = []
        for r in self.hn:
            it = {"kind": "hn", "sid": r["hn_id"], "hn_id": r["hn_id"], "hn_type": r["hn_type"], "edit": int(r["edit"]),
                  "related_flag": bool(r.get("related_flag")), "same_category": bool(r.get("same_category")), "y0_valid": True, "_row": r}
            if r.get("edit_b") is not None:
                it["edit_b"] = int(r["edit_b"])
            if "src_related_edit_first_pos" in r:                        # the final set (HN_DIR_FINAL): relation of the row to THIS log's edits
                rel, qe = r.get("src_related_edit_first_pos"), r.get("q_in_edit_first_pos")
                it.update(rule=r.get("rule"), group=r.get("group"), new=bool(r.get("new")),
                          src_related_to_log=rel is not None and int(rel) < T,     # the row's source item is related to an edit of this log
                          q_edit=int(qe) if qe is not None and int(qe) < T else None)   # edit of this log with the row's question (cross-like row)
            out.append(it)
        return out

    def _query(self, it):
        """(pseudo probe item, ptype, teacher-forced answer) of an item, in the testbed's Prober conventions."""
        H = self.H
        if it["kind"] == "edit":
            d = H.items[it["edit"]]
            return ({"m_loc": d["image_rephrase"] if it["metric"] == "igen" else d["image"],
                     "m_loc_q": d["rephrase"] if it["metric"] == "tgen" else d["src"]}, "image", d["alt"])
        if it["kind"] == "probe":
            return H.probes[it["probe_idx"]], it["ptype"], H.row[(it["probe_idx"], it["ptype"])]["y0_text"]
        r = it["_row"]
        return {"m_loc": r["image"], "m_loc_q": r["prompt"]}, "image", r["y0_text"]

    def batch(self, it, answer: bool = True):
        """The item's B = 1 batch: teacher-forced (default) or prompt only."""
        if it["kind"] == "edit" and answer:
            return self.H.edit_batch(it["edit"], it["metric"])
        q, ptype, ans = self._query(it)
        return self.H.prober.batch(q, ptype, answer=ans if answer else None)

    def lp_shift(self, it):
        """Teacher-forced last prompt index minus the prompt-only one (tokenizer only, cached per item).
        0: the same token on both paths.  1: the joint tokenisation leaves a bare space token after the prompt
        (LLaVA, answers that start with a digit), which the harness counts as the last prompt position.  None: the
        prompt's own tokens are not a prefix of the joint sequence, or the item has no answer text."""
        if it["sid"] not in self._shift:
            q, ptype, ans = self._query(it)
            v, img = self.H.vlm, ptype == "image"
            sh = None
            if ans:
                try:
                    _, text, sep = v.text_parts(q["m_loc_q"] if img else q["loc"], "vqa" if img else self.H.prober.text_fmt, img)
                    a, joint = v.answer_and_joint(text, ans, sep)
                    head, alone = joint[:len(joint) - len(a)], v._tok(text)
                    sh = len(head) - len(alone) if head[:len(alone)] == alone else None
                except RuntimeError:                                    # an answer text without a scorable tokenisation (invalid y0)
                    sh = None
            self._shift[it["sid"]] = sh
        return self._shift[it["sid"]]

    def _lp(self, it, b, lp: str):
        """Last prompt index (1,) handed to the hook for the teacher-forced batch b: the harness index ('tf') or the
        index of the prompt's own last token ('prompt' = where the decode path looks)."""
        if lp == "tf":
            return b["last_prompt_idx"]
        if lp != "prompt":
            raise ValueError("lp: 'tf' | 'prompt'")
        sh = self.lp_shift(it)
        if sh is None:
            raise ValueError(f"{it['sid']}: the prompt-only position is not defined in the teacher-forced sequence")
        return b["last_prompt_idx"] - sh

    @torch.no_grad()
    def capture(self, it, answer: bool = True, batch=None, lp: str = "tf"):
        """(x (Tn, d) module input at every position, module dtype; last prompt index): one truncated forward.
        lp='prompt' (teacher-forced sequence only) returns the index of the prompt's own last token instead."""
        b = self.batch(it, answer) if batch is None else batch
        return self.H.capture_inputs(b), int(self._lp(it, b, lp if answer else "tf")[0])

    @torch.no_grad()
    def decode_keys(self, it, n: int = MAX_NEW):
        """(n + 1, d) module inputs at the prompt's last token in the n + 1 FULL-SEQUENCE forwards that a decoder-v2
        decode of the item can run: prompt + k ids, k = 0 .. n (the MAX_NEW no-cache forwards and the first forward of
        the KV-cache continuation; later steps only see the new token).  The model is causal, so the key at that
        position does not depend on WHICH ids follow, but in the low-precision forward it does depend on HOW MANY
        (the last bits change with the sequence length, which moves a coordinate by up to a few per cent).  The
        filler id is the prompt's own last id; GPU test b9 checks that these keys are bitwise the keys of a real
        decode.  Costs n + 1 truncated forwards."""
        q, ptype, _ = self._query(it)
        b = self.H.prober.batch(q, ptype)
        lq = int(b["last_prompt_idx"][0])
        fill = int(b["inputs"]["input_ids"][0, lq])
        ks = []
        for k in range(n + 1):
            bk = dict(b, inputs=self.H.prober._extend(b["inputs"], [fill] * k), T=b["T"] + k)
            ks.append(self.H.capture_inputs(bk)[lq])
        return torch.stack(ks)

    def eligible_keys(self, it, gate, answer=True, lp: str = "tf"):
        """Keys of the item at the gate's eligible positions (for calibration and repairs), ONE query.
        answer=True   the teacher-forced sequence (what tf_rows evaluates; lp as in tf_rows);
               False  the prompt-only sequence (the first forward of the decode path);
               'both' the two above (cheap; a repair that must hold in a decode then also needs repair.DECODE_MARGIN);
               'decode'  every key a decode can present to a role-'last' gate (decode_keys: the prompt's last token at
                      the 11 sequence lengths of its full forwards): exact for D2 / free generation, role 'last' only;
               'all'  every position at which the evaluation can place a role-'last' gate for this item: the teacher-
                      forced key (lp='tf'), the teacher-forced sequence's key at the decode's token where that is
                      another token (lp='prompt', lp_shift != 0), and 'decode'.  What a complaint or a calibration
                      query must contain if the result is to hold on both paths."""
        if isinstance(answer, str):
            if answer == "both":
                return torch.cat([self.eligible_keys(it, gate, True, lp), self.eligible_keys(it, gate, False)])
            if answer not in ("decode", "all"):
                raise ValueError("answer: True | False | 'both' | 'decode' | 'all'")
            if gate.role != "last":
                raise NotImplementedError("the eligible positions of a decode depend on the generated tokens unless the role is 'last'")
            dk = self.decode_keys(it)
            if answer == "decode":
                return dk
            x, p = self.capture(it, True)
            ks = [x[p][None]] + ([x[p - self.lp_shift(it)][None]] if self.lp_shift(it) else []) + [dk]
            return torch.cat(ks)
        x, p = self.capture(it, answer, lp=lp)
        return gate.eligible_keys(x, torch.tensor([p]))

    def own_keys(self, metric: str = "rel", edits=None, answer=True):
        """(keys (n, d) module dtype, edit ids (n,)): the module input at the last prompt position of every edit's own
        B = 1 sequence, for repair.add_guards (the stored solver key is the B = 10 key of the same prompt; its B = 1
        coordinate is 0.95 - 1.02 on Qwen, so a threshold capped at the stored key can silence the edit's own prompt).
        answer=True: teacher-forced sequence; False / 'both': (also) the prompt-only sequence of the decode path;
        'all': teacher-forced + every full forward of a decode (decode_keys: the own prompt then keeps firing in every
        forward of a free generation).  Decode-path keys are skipped for edits whose prompt-only position is another
        token (lp_shift != 0: the edit was not written there, and as a guard that key would drag the threshold down)."""
        if answer not in (True, False, "both", "all"):
            raise ValueError("answer: True | False | 'both' | 'all'")
        ks, es = [], []
        for it in self.edit_items((metric,), edits):
            new = []
            if answer is not False:
                x, p = self.capture(it, True)
                new.append(x[p][None])
            if answer is not True and self.lp_shift(it) == 0:
                if answer == "all":
                    new.append(self.decode_keys(it))
                else:
                    x, p = self.capture(it, False)
                    new.append(x[p][None])
            for k in new:
                ks.append(k)
                es += [it["edit"]] * k.shape[0]
        return torch.cat(ks), es

    def predict(self, it, gate, answer: bool = True, batch=None, lp: str = "tf"):
        """Firing of a gate on the item, PREDICTED from the captured keys (no forward through the hook)."""
        x, p = self.capture(it, answer, batch, lp)
        return self._on_device(gate).firing(x[None], torch.tensor([p], device=x.device), 0, {})

    # ------------------------------------------------------------------ the one code path
    def _on_device(self, gate):
        dev = getattr(gate, "device", None)
        return gate if dev is None or dev == torch.device(self.H.dev) else gate.to(torch.device(self.H.dev))

    def _prepare(self, method):
        return self._on_device(method) if hasattr(method, "firing") else method

    @contextlib.contextmanager
    def applied(self, method):
        """Model under `method` (see the module docstring); yields the Hook of a gate method, else None.
        The model is back at W_E afterwards."""
        H = self.H
        H.reset()
        try:
            if method is None:
                yield None
            elif torch.is_tensor(method):
                if method.shape != H.W0.shape:
                    raise ValueError(f"replacement weight has shape {tuple(method.shape)}, the module {tuple(H.W0.shape)}")
                with H.at(method):                                      # cast to the module dtype by the copy, like a deployed weight
                    yield None
            elif isinstance(method, Dual):
                raise TypeError("DUAL is defined on generated answers: use Evaluator.generate / d2 (it has no teacher-forced forward)")
            elif hasattr(method, "firing"):
                with H.at(H.W0), Hook(H.mod, self._on_device(method), set_weight=False) as hk:
                    self._hook = hk
                    yield hk
            elif hasattr(method, "kind") and hasattr(method, "meta"):
                with H.applied(method):
                    yield None
            else:
                raise TypeError(f"not a method: {type(method)}")
        finally:
            self._hook = None
            H.reset()

    def trim_vcache(self, keep: int = 8):
        """Bound the harness' vision-feature memo (it is only useful across the methods of one item)."""
        vc = self.H._setup.get("vcache")
        if vc is not None and len(vc) > keep:
            vc.clear()

    def _public(self, it):
        return {k: v for k, v in it.items() if not k.startswith("_")}

    @torch.no_grad()
    def pre_ids(self, it, batch=None):
        """Pre-edit argmax ids of the item's teacher-forced answer, for token-level locality.  Only items that DECLARE an invalid
        pre-edit answer (it['y0_valid'] is False: probe_items(valid_only=False)) cost a forward at W_0 (cached per item; False
        if a pre-edit logit is not finite).  None otherwise: for a valid y0 the pre-edit argmax is the teacher-forced ids
        (W_0 reproduces y0); items without the field (edits, items built elsewhere) get no token locality.
        Call OUTSIDE `applied` (it switches the model to W_0 and back)."""
        if it["kind"] == "edit" or it.get("y0_valid") is not False:
            return None
        if it["sid"] not in self._pre:
            b = self.batch(it) if batch is None else batch
            n = int(b["answer_mask"][0].sum())
            if n == 0:
                raise ValueError(f"{it['sid']}: no answer tokens (invalid y0)")
            with self.applied(self.H.W0):
                lg = self.H.vlm.forward(b, logits_to_keep="answer").logits[0, :n]
            self._pre[it["sid"]] = lg.argmax(-1) if bool(torch.isfinite(lg).all()) else False
        return self._pre[it["sid"]]

    @torch.no_grad()
    def tf(self, it, batch=None, logits: bool = False, lp: str = "tf", pre_ids=None):
        """One teacher-forced forward of the item under the active method -> row (call inside `applied`).
        lp: where a gate's last prompt position is: 'tf' = the harness index of the teacher-forced sequence (default;
        P10 / P11 convention), 'prompt' = the prompt's own last token (the decode path's position; differs on rows with
        lp_shift != 0).  The scored answer tokens are the same.
        pre_ids: Evaluator.pre_ids(it) (token-level locality of rows with an invalid y0; the per-item functions pass it)."""
        b = self.batch(it) if batch is None else batch
        if self._hook is not None:
            self._hook.begin(self._lp(it, b, lp))
        n = int(b["answer_mask"][0].sum())
        if n == 0:
            raise ValueError(f"{it['sid']}: no answer tokens (invalid y0)")
        lg = self.H.vlm.forward(b, logits_to_keep="answer").logits[0, :n]
        finite = bool(torch.isfinite(lg).all())
        eq = (lg.argmax(-1) == b["answer_ids"][0, :n]) if finite else torch.zeros(n, dtype=torch.bool)
        exact = bool(eq.all())
        acc = float(eq.float().mean())
        loc = None
        if it["kind"] != "edit":                                        # token-level locality (module docstring)
            if pre_ids is None:
                loc = acc if it.get("y0_valid") is True else None        # valid y0: the pre-edit argmax is y0 itself
            elif pre_ids is not False:
                loc = float((lg.argmax(-1) == pre_ids).float().mean()) if finite else 0.0
        row = self._tf_row(it, exact, acc, n, finite, None if (exact or not finite) else int((~eq).nonzero()[0, 0]), lp, loc)
        row.update(self._flags(it))
        return (row, lg) if logits else row

    def _tf_row(self, it, exact, token_acc, n, finite, first_mismatch, lp, tok_loc=None):
        """The method-independent fields of a teacher-forced row, from the outcome of the forward (tf) or from the cached
        outcome of the pre-edit model (SparseEvaluator: a sequence on which nothing fires)."""
        row = dict(self._public(it), exact=exact, token_acc=token_acc, n_tok=n, finite=finite, first_mismatch=first_mismatch, lp=lp,
                   lp_shift=self.lp_shift(it))
        if it["kind"] != "edit":
            row["D1"] = not exact
            row["tok_loc"] = tok_loc
        return row

    def _flags(self, it):
        if self._hook is None:
            return {"fired": None}
        return self._hook.summary(own=it.get("edit"))

    @torch.no_grad()
    def decode(self, it):
        """Decoder-v2 free generation of the item's prompt under the active method -> row (inside `applied`)."""
        H, TB = self.H, self.TB
        q, ptype, ans = self._query(it)
        pb = H.prober.batch(q, ptype)
        if self._hook is not None:
            self._hook.begin(pb["last_prompt_idx"], gen=True)
        y, long, info = H.prober.decode_v2(q, ptype)
        line = long.split("\n")[0].strip()
        row = dict(self._public(it), y=y, long=long, line=line, stop=info["stop"], long_stop=info["long_stop"],
                   n_long=len(info["long_ids"]), degenerate=degenerate(y, info["long_ids"]),
                   lp="prompt", lp_shift=self.lp_shift(it))                # the decode's gate sits on the prompt's own last token
        if self._hook is not None:                 # position bookkeeping of the hook through the KV-cache decode
            n10, more = len(info["gen_ids"]), len(info["long_ids"]) - len(info["gen_ids"])
            eos = info["long_stop"] == "eos"
            last_len = int(pb["seq_len"][0]) + (n10 if (eos and not more and n10 < MAX_NEW) else n10 + more + int(eos) - 1 if more else n10)
            row["pos_ok"] = bool(self._hook.seen == last_len)
        if it["kind"] == "edit":
            na, nl = TB.norm_answer(ans), TB.norm_answer(line)
            row.update(eq=nl == na, eq10=TB.norm_answer(y) == na, contains=bool(TB.contains_ref(long, ans)),
                       starts=bool(na) and (nl + " ").startswith(na + " "))
        elif it["kind"] == "probe":
            r = H.row[(it["probe_idx"], it["ptype"])]
            pc = bool(r["pre_correct_long"])
            post = bool(TB.contains_ref(long, r["ref_answer"]))
            row.update(agree=(y == r["y0_text"]) if r["y0_valid"] else None, long_eq_pre=long == r["pre_long_text"],
                       pre_correct=pc, post_correct=post, D2=pc and not post)
        else:
            r = it["_row"]
            row.update(agree=y == r["y0_text"], long_eq_pre=long == r["long_text"], leak=bool(TB.contains_ref(long, r["edit_alt"])))
        row.update(self._flags(it))
        return row

    def _decode_plain(self, it, which, memo):
        """Decode row of the item under the deployed ('WE') or the pre-edit ('W0') model, computed once per item."""
        if which not in memo:
            with self.applied(None if which == "WE" else self.H.W0):
                memo[which] = self.decode(it)
        return memo[which]

    def _dual_row(self, it, dual, memo):
        """DUAL: the decode row of the returned answer (deployed if it matches a logged target, else pre-edit) + bookkeeping."""
        rE, r0 = self._decode_plain(it, "WE", memo), self._decode_plain(it, "W0", memo)
        m = dual.match(self.TB, rE["line"], rE["long"], rE["y"])
        row = dict(rE if m else r0)
        own = it.get("edit") if it["kind"] == "edit" else None
        row.update(fired=bool(m), n_generations=2, n_new_tokens=[rE["n_long"], r0["n_long"]],
                   dual={"variant": dual.variant, "match": bool(m), "n_matching_edits": len(m), "edits": m[:8],
                         "own": None if own is None else own in m, "source": "deployed" if m else "pre-edit",
                         "deployed_equals_pre_edit": rE["long"] == r0["long"]})
        return row

    def _each(self, method, items, fn, verbose=0, lp="tf"):
        many = isinstance(method, dict)
        methods = {k: self._prepare(m) for k, m in (method.items() if many else [("_", method)])}
        if fn == "tf" and any(isinstance(m, Dual) for m in methods.values()):
            raise TypeError("DUAL is defined on generated answers: use Evaluator.generate / d2 (it has no teacher-forced forward)")
        out = {k: [] for k in methods}
        with torch.no_grad(), self.hz.HE.vision_cache(self.H):
            for n, it in enumerate(items):
                self.trim_vcache()
                b = self.batch(it) if fn == "tf" else None
                pre = self.pre_ids(it, b) if fn == "tf" else None
                memo = {}                                               # decode rows of the deployed / pre-edit model of THIS item
                for k, m in methods.items():
                    if fn == "tf":
                        with self.applied(m):
                            out[k].append(self.tf(it, b, lp=lp, pre_ids=pre))
                    elif isinstance(m, Dual):
                        out[k].append(self._dual_row(it, m, memo))
                    elif m is None or m is self.H.W0:                    # shared with DUAL: one decode per model and item
                        out[k].append(dict(self._decode_plain(it, "WE" if m is None else "W0", memo)))
                    else:
                        with self.applied(m):
                            out[k].append(self.decode(it))
                if verbose and (n + 1) % verbose == 0:
                    print(f"    {fn} {n + 1}/{len(items)}", flush=True)
        return out if many else out["_"]

    # ------------------------------------------------------------------ one function per item type
    # lp (teacher-forced functions): position of a gate's "last prompt position", 'tf' (harness index, default) or
    # 'prompt' (the decode path's token); only rows with lp_shift != 0 differ (see the module docstring).
    def tf_rows(self, method, items, verbose=0, lp="tf"):
        """Teacher-forced rows of an explicit list of items (from edit_items / probe_items / hn_items)."""
        return self._each(method, items, "tf", verbose, lp)

    def edits(self, method, metrics=EDIT_METRICS, edits=None, verbose=0, lp="tf"):
        """Rel / T-Gen / I-Gen, teacher forced: rows with exact, token_acc (+ firing flags, own_* for gates)."""
        return self._each(method, self.edit_items(metrics, edits), "tf", verbose, lp)

    def probes(self, method, pool: str = "all", verbose=0, lp="tf"):
        """Valid-y0 probe rows of a pool, teacher forced: rows with D1 (= y0 not reproduced), pool, disjoint, strict,
        query_id."""
        return self._each(method, self.probe_items(pool), "tf", verbose, lp)

    def hard_negatives(self, method, verbose=0, lp="tf"):
        """Valid hard negatives of the log's edits, teacher forced: rows with D1, hn_type, edit, own_fired."""
        return self._each(method, self.hn_items(), "tf", verbose, lp)

    def d2(self, method, pool: str = "all", verbose=0):
        """Pre-correct probe rows (valid y0 or not) of a pool, 48-token decode: rows with D2."""
        return self._each(method, self.probe_items(pool, valid_only=False, pre_correct=True), "decode", verbose)

    def generate(self, method, items=None, verbose=0):
        """Free generation (decoder v2) of any items (default: every edit prompt): eq / eq10 / contains / starts for
        edits, agree for probes and hard negatives, degenerate-output flags.  Also the place of the DUAL baseline
        (method = Dual.of(H, 'eq' | 'contains')): rows of probes, hard negatives and edit prompts (Rel, T-Gen, I-Gen)."""
        return self._each(method, self.edit_items() if items is None else items, "decode", verbose)

    def dual(self, variant: str = "eq", on: str = "line") -> Dual:
        """The DUAL baseline of this log (targets = the logged edits' targets)."""
        return Dual.of(self.H, variant, on)

    def dual_tf_rows(self, dual: Dual, items, verbose=0, lp: str = "tf") -> list:
        """Teacher-forced rows of the DUAL COMPOSITION, so that D1 / exact / token-level locality at B = 1 can be reported for
        DUAL next to the gates.  This is NOT part of the baseline's definition (which is about generated answers): per item
        the query is decoded under the deployed and the pre-edit model, DUAL's rule selects one of the two MODELS, and the row
        is the teacher-forced row of the selected model (pre-edit model: D1 = False and tok_loc = 1 on a valid row).
        Rows carry `fired` (= the deployed model was selected), `dual`, `n_generations` = 2, `n_new_tokens`."""
        out = []
        with torch.no_grad(), self.hz.HE.vision_cache(self.H):
            for n, it in enumerate(items):
                self.trim_vcache()
                d = self._dual_row(it, dual, {})
                b = self.batch(it)
                pre = self.pre_ids(it, b)
                with self.applied(None if d["fired"] else self.H.W0):
                    row = self.tf(it, b, lp=lp, pre_ids=pre)
                row.update(fired=d["fired"], dual=d["dual"], n_generations=2, n_new_tokens=d["n_new_tokens"])
                out.append(row)
                if verbose and (n + 1) % verbose == 0:
                    print(f"    dual tf {n + 1}/{len(items)}", flush=True)
        return out

    def token_locality(self, method=None, pool: str = "heldout_disjoint", rows=None, valid_only: bool = True) -> dict:
        """Token-level locality at B = 1 of a method on the probe rows of a pool: T-Loc* / I-Loc* (locality) and the
        per-row values.  valid_only=True (default): the rows with a valid pre-edit answer (the rows of every D1 number);
        False: also the rows whose y0 the pre-edit model does not reproduce (their pre-edit argmax costs one forward at W_0
        per item; rows without a scorable answer are skipped and counted).
        For the deployed log (method None) the stored benchmark locality of the log (stored_metrics: vendor pass, the
        benchmark's reference answers, padded batches of 8, all 500 probe items) and the SIGNED differences T-Loc* - T-Loc,
        I-Loc* - I-Loc are added.  The two quantities differ by definition (teacher-forced text, batch shape, row set); the
        difference is reported as it is.  rows: teacher-forced probe rows already computed for the method."""
        skipped = 0
        if rows is None:
            if valid_only:
                rows = self.probes(method, pool)
            else:
                its = []
                for it in self.probe_items(pool, valid_only=False):
                    try:
                        ok = int(self.batch(it)["answer_mask"][0].sum()) > 0
                    except (RuntimeError, ValueError):                  # empty y0 / an answer without a scorable tokenisation
                        ok = False
                    its.append(it) if ok else None
                    skipped += int(not ok)
                rows = self.tf_rows(method, its)
        out = dict(locality(rows), pool=pool, valid_only=valid_only, n_rows_without_scorable_answer=skipped,
                   per_row=[{"sid": r["sid"], "ptype": r["ptype"], "tok_loc": r.get("tok_loc"), "finite": r["finite"], "y0_valid": r.get("y0_valid")}
                            for r in rows if r.get("kind") == "probe"])
        if method is None:
            st = stored_metrics(self.H.name, pools=False)["locality"]
            out["stored"] = st
            out["diff"] = {k: (None if out[k + "*"] is None or st.get(k) is None else out[k + "*"] - st[k]) for k in ("T-Loc", "I-Loc")}
        return out


def locality(rows) -> dict:
    """Token-level locality at B = 1 from teacher-forced probe rows of ONE method: T-Loc* / I-Loc* = mean (percent) over
    the text / image probe rows of `tok_loc` (share of answer tokens whose argmax equals the pre-edit model's under teacher
    forcing of the pre-edit answer).  A row with a non-finite logit counts as 0 agreement (as the benchmark's Loc does) and
    is counted (n_*_nonfinite); `*_finite` are the means without those rows; `*_em` = share of rows with every token equal
    (= 100 - D1 rate).  Rows without a known pre-edit validity (tok_loc None) are excluded and counted (n_*_undefined)."""
    out = {}
    pr = [r for r in rows if r.get("kind") == "probe" and "tok_loc" in r]
    for nm, pt in (("T-Loc*", "text"), ("I-Loc*", "image")):
        sub = [r for r in pr if r.get("ptype") == pt]
        ok = [r for r in sub if r["tok_loc"] is not None]
        fin = [r for r in ok if r["finite"]]
        out[nm] = (100.0 * sum(r["tok_loc"] for r in ok) / len(ok)) if ok else None
        out[nm + "_em"] = (100.0 * sum(r["tok_loc"] == 1.0 for r in ok) / len(ok)) if ok else None
        out[nm + "_finite"] = (100.0 * sum(r["tok_loc"] for r in fin) / len(fin)) if fin else None
        out.update({f"n_{pt}": len(ok), f"n_{pt}_nonfinite": len(ok) - len(fin), f"n_{pt}_undefined": len(sub) - len(ok)})
    return out


def summarise(rows) -> dict:
    """Rates (percent) of a list of rows of ONE method: per edit metric exact / token accuracy (teacher forcing) or eq /
    contains (free generation), D1 rate of probe and hard-negative rows, D2 count, firing rates (None for methods
    that are not gates); n per group.  Convenience only: every number is a mean of row flags."""
    def pct(v):
        if not v or any(x is None for x in v):
            return None
        return 100.0 * sum(v) / len(v)
    out = {}
    for m in EDIT_METRICS:
        sub = [r for r in rows if r.get("kind") == "edit" and r.get("metric") == m]
        if sub and "exact" in sub[0]:
            out[m] = {"n": len(sub), "exact": pct([r["exact"] for r in sub]), "token": pct([r["token_acc"] for r in sub]),
                      "own_fired_last": pct([r.get("own_fired_last") for r in sub])}
        elif sub:
            out[m] = {"n": len(sub), "eq": pct([r["eq"] for r in sub]), "contains": pct([r["contains"] for r in sub]),
                      "own_fired_last": pct([r.get("own_fired_last") for r in sub]),
                      "n_degenerate": sum(any(r["degenerate"].values()) for r in sub)}
    for kind in ("probe", "hn"):
        sub = [r for r in rows if r.get("kind") == kind and "D1" in r]
        if sub:
            out[kind] = {"n": len(sub), "D1": pct([r["D1"] for r in sub]), "fired": pct([r.get("fired") for r in sub])}
    d2 = [r for r in rows if "D2" in r and r.get("pre_correct")]
    if d2:
        out["D2"] = {"n_pre_correct": len(d2), "n_D2": sum(r["D2"] for r in d2), "fired": pct([r.get("fired") for r in d2])}
    if any(r.get("kind") == "probe" and r.get("tok_loc") is not None for r in rows):
        out["loc"] = locality(rows)
    du = [r for r in rows if "dual" in r]
    if du:
        out["dual"] = {"n": len(du), "released_deployed": pct([r["dual"]["match"] for r in du]), "generations_per_query": 2.0,
                       "n_new_tokens": sum(sum(r["n_new_tokens"]) for r in du)}
    return out
