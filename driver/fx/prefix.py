"""fx.prefix: prefix logs of the development streams (amendment R12, mild regime; R19 (d): the prefix logs only).

A PREFIX LOG is a development stream after its first k edits: the pre-edit weight plus the stored applied batch updates of the
first k / 10 batches (k = 10, 20, 50: 1, 2, 5 batches), with the stored factors of these edits.

    names        TB-Q100-ae-l2_3-p10 / -p20 / -p50, TB-L100-ae-l2_3-p10 / -p20 / -p50      (role 'prefix'; fx.logs.names('prefix'))
                 ...-p100 (role 'prefix_check'): the SAME construction with all 10 batches; it must reproduce the development
                 log (deployed weight and composite bitwise, factors, D1 flags of the stored records) and is used for nothing else
    directory    10_final/common/out/prefix_logs/<name>/  in the testbed layout, so that the UNMODIFIED harness loader builds it:
                   edit_batch.jsonl       the first k records of the stream
                   edit_factors.pt        rows 0 .. k - 1 of the parent's factors (U, W, K1_solver, k_t_single, uids): the same bits
                   batch_deltas/          links to the parent's exact applied batch updates batch_000 .. batch_(k/10 - 1)
                   composite_delta.pt     W_E.float() - W_0.float() with W_E = (W_0.float() + sum of these updates, in batch order,
                                          fp32, on the model's device).to(module dtype): the harness's rule (HLog._init_tb),
                                          which is bitwise the stored deployed weight when all batches are taken
                   metrics.json           config, dtype, n_edits = k, per_batch of the first batches, the editor's build record; and
                                          Rel / TGen / IGen _exact and _token of the PREFIX weight by the testbed's stored-metric
                                          rule (the vendor pass in padded batches of 8, run through the harness's own function),
                                          which the harness's baseline() compares itself with -- NOT a reported number (every
                                          reported number is B = 1: prefix.json); TLoc_probes / ILoc_probes are None (not computed)
                   probe_records_v2.jsonl the parent's probe records with the fields that depend on the deployed weight
                                          RECOMPUTED for the prefix weight (reproduced, first_mismatch_pos, is_complaint_D1: teacher
                                          forcing at B = 1 through the library's Evaluator) -- the D2 / decode fields are None
                                          (R12 registers D1 complaints only)
                   prefix.json            description, checks, completeness marker
    The intermediate weights of the original build were not stored; the prefix weight is the reconstruction rule above (for all
    10 batches it is the stored deployed weight bit for bit; the fp32 summation and the cast are the build's own).

    L = fx.logs.open(name)                               LogData: factors, key-cache rows of the first k edits; hard negatives =
                                                         those of its edits (HN-X: both edits in the prefix); pools as the parent
    H = fx.logs.load_model(name)                         the harness handle (= fx.prefix.load_model): driver, library Evaluator
    H = fx.prefix.load_model(name, keep_batch_deltas=True, cache_dir=<your dir>)     for the harness's linear repairs:
        hz = rescope.harness(); cs = hz.HS.sample_complaints(H, n, seed); corr, cost = hz.HM.repair("COMP", H, cs, protect=("K_ans",),
        query="answer", form="adapter") ...  (the calls of 08_rescope/p12_linear_streams/code/run_streams.py); harness cache files
        of prefix logs (W_E baseline, edit keys, probe keys) go to cache_dir (default out/prefix_logs/_hcache), never into the
        harness's or the library's own cache directory
    fx.prefix.edit_log(name)                             the rescope.EditLog (library object) of the prefix
    recombinations: recomb.log_rows(L) of the feature package serves any prefix of a cached stream.
    python -m fx.prefix build <parent> [k ...]           GPU job (queue): builds the prefix logs of a development log + description
    python -m fx.prefix table                            renders out/prefix_logs.txt from the prefix.json files
NOT available for prefix logs: rescope.load / rescope.stored_metrics / fx.logs.stored by name (the library looks under 06_testbed_v3),
P12 records, D2 complaints.
"""
from __future__ import annotations

import builtins
import json
import os
import re
import time

import torch

import fx
from . import logs as LG

ROOT = os.path.join(fx.COMMON, "out", "prefix_logs")
KS = (10, 20, 50)
CHECK_K = 100
BATCH = 10
NEED_BUILD = ("edit_batch.jsonl", "metrics.json", "composite_delta.pt", "edit_factors.pt", "probe_records_v2.jsonl")
NEED = NEED_BUILD + ("prefix.json",)
DEPLOYED_FIELDS = ("reproduced", "first_mismatch_pos", "is_complaint_D1", "is_complaint_D1_raw")
NOT_COMPUTED = ("yE_text", "yE_equals_y0", "post_long_text", "post_correct_at10", "post_correct_long", "is_complaint_D2_long", "is_complaint_D2_at10", "is_complaint_D2")
TABLE = os.path.join(fx.COMMON, "out", "prefix_logs.txt")


def name(parent: str, k: int) -> str:
    return f"{parent}-p{int(k)}"


def spec(nm: str) -> tuple:
    m = re.match(r"^(.+)-p(\d+)$", nm)
    if m is None or m.group(1) not in LG.ROLES["development"]:
        raise KeyError(f"{nm!r} is not a prefix log")
    return m.group(1), int(m.group(2))


def directory(nm: str) -> str:
    return os.path.join(ROOT, nm)


def _register():
    for parent in LG.ROLES["development"]:
        e = LG.REGISTRY[parent]
        for k in KS + (CHECK_K,):
            nm = name(parent, k)
            if nm in LG.REGISTRY:
                continue
            LG.register(nm, "prefix" if k != CHECK_K else "prefix_check", e["backbone"], e["layer"], e["family"], k, cache=e["cache"], dataset=e["dataset"],
                        editor_family=e["editor_family"], editor=e["editor"] + f"; the first {k} edits ({k // BATCH} batches) of {parent}", l2=e["l2"],
                        statistics=e["statistics"], gq_mem=e["gq_mem"], reported=None, path=directory(nm), need=NEED,
                        loader=load_model, prefix_of=parent, n_batches=k // BATCH)
    LG.ROLE_NOTE["prefix"] = "prefix logs of the development streams after 10 / 20 / 50 edits (R12, mild regime); built by fx.prefix"
    LG.ROLE_NOTE["prefix_check"] = "the prefix construction with all 10 batches: must reproduce the development log; a test object, never an experiment log"


def complete(nm: str) -> bool:
    p = os.path.join(directory(nm), "prefix.json")
    return os.path.exists(p) and bool(fx.io.jload(p).get("complete")) and all(os.path.exists(os.path.join(directory(nm), f)) for f in NEED)


def describe(nm: str) -> dict:
    return fx.io.jload(os.path.join(directory(nm), "prefix.json"))


def edit_log(nm: str):
    """The library's EditLog of a prefix log (CPU; factors of the first k edits, batches)."""
    from rescope import from_testbed
    return from_testbed(directory(nm))


# ----------------------------------------------------------------------------- the loader (unmodified harness on the prefix directory)
def redirect_cache(cache_dir: str | None = None) -> str:
    """Point the harness's cache directory of THIS process to cache_dir (default $FX_HCACHE or out/prefix_logs/_hcache); the
    read-only caches that exist (edit keys, probe keys, statistics bases) are linked, W_E baselines of other logs are not."""
    import rescope
    hz = rescope.harness()
    d = cache_dir or os.environ.get("FX_HCACHE") or os.path.join(ROOT, "_hcache")
    cur = hz.HC.CACHE
    if os.path.realpath(cur) == os.path.realpath(d):
        return d
    os.makedirs(d, exist_ok=True)
    for f in sorted(os.listdir(cur)):
        dst = os.path.join(d, f)
        if f.startswith("baseline_") or os.path.lexists(dst):
            continue
        try:
            os.symlink(os.path.realpath(os.path.join(cur, f)), dst)
        except OSError:
            pass
    for m in (hz.HC, hz.HL, hz.HE, hz.HM):
        m.CACHE = d
    return d


def load_model(nm: str, device: str = "cuda:0", mem_gb=None, reuse=None, keep_batch_deltas: bool = False, cache_dir: str | None = None, _building: bool = False):
    """The harness handle of a prefix log: harness.load on the prefix directory (the harness's own loader and load-time checks:
    W_E from the batch updates == the stored composite, rollback(empty) == W_E, factors and probes match the stream).  The model
    is at the prefix's W_E; H.W0 / H.WE are exact.  keep_batch_deltas=True keeps the applied batch updates (the harness's rollback
    methods); reuse: a handle of the same backbone that is no longer needed (its model is reused); cache_dir: see redirect_cache."""
    import rescope
    e = LG.info(nm)
    parent, k = spec(nm)
    if not _building and not complete(nm):
        raise FileNotFoundError(f"prefix log {nm} is not built (python -m fx.prefix build {parent}, a GPU job)")
    hz = rescope.harness()
    rescope.ensure_registered(parent)
    TB = hz.TB
    cfg = dict(TB.LOGS[parent], stream=os.path.join(e["path"], "edit_batch.jsonl"), prefix_of=parent, n_prefix=k)
    if nm in TB.LOGS and TB.LOGS[nm] != cfg:
        raise RuntimeError(f"{nm} is registered with the testbed under another configuration")
    TB.LOGS[nm] = cfg
    if mem_gb is None:
        mem_gb = (int(os.environ.get("GQ_MEM", e["gq_mem"])) - 600) / 1024.0
    if device.startswith("cuda"):
        dev = torch.device(device)
        torch.cuda.set_per_process_memory_fraction(min(1.0, mem_gb * 2 ** 30 / torch.cuda.get_device_properties(dev).total_memory), dev)
    cdir = redirect_cache(cache_dir)
    vlm = None
    if reuse is not None:
        if reuse.backbone != cfg["backbone"] or reuse.L != int(cfg["layer"]):
            raise ValueError(f"cannot reuse the model of {reuse.name} for {nm}")
        reuse.set_weight(reuse.W0)
        vlm = reuse.vlm
    t0 = time.time()
    old_root, old_budget = hz.HL.TB_ROOT, hz.HL.GPU_DELTA_BUDGET
    hz.HL.TB_ROOT, hz.HL.GPU_DELTA_BUDGET = ROOT, 0                      # the log's directory; applied updates stay on the CPU
    try:
        H = hz.pkg.load(nm, device=device, vlm=vlm)
    finally:
        hz.HL.TB_ROOT, hz.HL.GPU_DELTA_BUDGET = old_root, old_budget
    want = ("probe_uids_match_records", "WE_float_minus_W0_equals_stored_composite", "C_equals_stored_composite", "factor_uids_match_stream", "rollback_empty_equals_WE")
    bad = [c for c in want if H.checks.get(c) is not True]
    if bad or H.T != k:
        raise RuntimeError(f"{nm}: load-time checks of the harness failed: {bad} (T = {H.T})")
    if reuse is not None and not torch.equal(H.W0, reuse.W0):
        raise RuntimeError("the reused model did not hold the pre-edit weight")
    if not keep_batch_deltas:
        H.A = None
        torch.cuda.empty_cache()
    H.load_info = {"lean": False, "seconds": time.time() - t0, "T": int(H.T), "prefix_of": parent, "harness_cache": cdir, "reused_model_of": None if reuse is None else reuse.name}
    return H


# ----------------------------------------------------------------------------- construction
def materialise(parent: str, k: int) -> str:
    """The files of a prefix log that need no model (module docstring); the probe records are PROVISIONAL (deployed-weight fields
    None) until build() has evaluated the prefix weight.  -> the directory."""
    if k % BATCH or not 0 < k <= LG.REGISTRY[parent]["T"]:
        raise ValueError(f"k must be a multiple of the batch size {BATCH}")
    src, d, nb = LG.REGISTRY[parent]["path"], directory(name(parent, k)), k // BATCH
    os.makedirs(os.path.join(d, "batch_deltas"), exist_ok=True)
    for f in ("prefix.json", "composite_delta.pt"):                     # (metrics.json is rewritten below, without the stored-metric fields)
        if os.path.exists(os.path.join(d, f)):
            os.remove(os.path.join(d, f))
    with builtins.open(os.path.join(src, "edit_batch.jsonl")) as f:
        recs = [line for line in f if line.strip()][:k]
    if [json.loads(r)["edit_id"] for r in recs] != list(range(k)):
        raise RuntimeError("the stream file is not in edit order")
    with builtins.open(os.path.join(d, "edit_batch.jsonl"), "w") as f:
        f.writelines(recs)
    fac = torch.load(os.path.join(src, "edit_factors.pt"), map_location="cpu")
    out = {key: (v[:k].clone() if torch.is_tensor(v) else (list(v[:k]) if key == "uids" else v)) for key, v in fac.items()}
    out["doc"] = str(fac.get("doc", "")) + f" [prefix: rows 0 .. {k - 1} of {parent}/edit_factors.pt, fx.prefix]"
    torch.save(out, os.path.join(d, "edit_factors.pt"))
    with builtins.open(os.path.join(src, "metrics.json")) as f:
        met = json.load(f)
    pb = met["per_batch"][:nb]
    if [i for b in pb for i in b["edit_ids"]] != list(range(k)) or [b["batch"] for b in pb] != list(range(nb)):
        raise RuntimeError("the first batches of the parent do not hold exactly the first k edits")
    keep = {key: met[key] for key in ("config", "layer_name", "dtype", "zcfg", "probe_info", "text_probe_fmt") if key in met}
    if "build" in met:                                                  # the editor's configuration record (spectrum, rule, protected rank): read by P12's re-edit code
        keep["build"] = dict(met["build"], n_batches=nb, prefix_note="copied from the parent's build record: spectrum / rule / protected describe the editor "
                             "configuration, which the prefix shares; timings and reconstruction checks are the parent's")
    keep.update(name=name(parent, k), n_edits=k, per_batch=pb, prefix={"of": parent, "n_edits": k, "n_batches": nb, "built_by": "fx.prefix",
                                                                    "note": "edit and locality metrics of the parent's metrics.json describe the parent's deployed weight and are not copied"})
    fx.io.jdump(keep, os.path.join(d, "metrics.json"), indent=1)
    for b in range(nb):
        dst = os.path.join(d, "batch_deltas", f"batch_{b:03d}.pt")
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(os.path.realpath(os.path.join(src, "batch_deltas", f"batch_{b:03d}.pt")), dst)
    for f in os.listdir(os.path.join(d, "batch_deltas")):                # no update beyond the prefix
        if int(re.match(r"batch_(\d+)\.pt", f).group(1)) >= nb:
            os.remove(os.path.join(d, "batch_deltas", f))
    _write_records(parent, k, None)
    return d


def _write_records(parent: str, k: int, d1: dict | None):
    """probe_records_v2.jsonl of the prefix: the parent's records, deployed-weight fields from d1 = {(probe_idx, ptype): (reproduced,
    first_mismatch)} of the valid rows (None: provisional, all None)."""
    src, d = LG.REGISTRY[parent]["path"], directory(name(parent, k))
    with builtins.open(os.path.join(src, "probe_records_v2.jsonl")) as f:
        recs = [json.loads(line) for line in f if line.strip()]
    out = []
    for r in recs:
        r = dict(r)
        for key in NOT_COMPUTED:
            if key in r:
                r[key] = None
        got = None if d1 is None else d1.get((r["probe_idx"], r["ptype"]))
        if got is None:
            r.update(reproduced=None, first_mismatch_pos=None, is_complaint_D1=False if d1 is not None else None, is_complaint_D1_raw=None)
        else:
            r.update(reproduced=bool(got[0]), first_mismatch_pos=got[1], is_complaint_D1=bool(r["y0_valid"]) and not got[0], is_complaint_D1_raw=not got[0])
        r["prefix"] = {"of": parent, "n_edits": k, "provisional": d1 is None}
        out.append(r)
    tmp = os.path.join(d, "probe_records_v2.jsonl.tmp")
    with builtins.open(tmp, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, os.path.join(d, "probe_records_v2.jsonl"))


def _vendor_edit_metrics(H) -> dict:
    """Rel / T-Gen / I-Gen exact and token accuracy (percent) of the handle's deployed weight by the rule of the testbed's STORED
    metrics: the vendor pass in padded batches of 8, through the harness's own plan and runner (harness.evaluation._plan /
    _run_planned on vendor metrics.build_probes: exactly what harness.baseline() recomputes and compares with metrics.json)."""
    import rescope
    hz = rescope.harness()
    HE, VM = hz.HE, hz.VM
    H.reset()
    out = {}
    with torch.no_grad(), HE.vision_cache(H):
        for m, key in (("rel", "Rel"), ("tgen", "TGen"), ("igen", "IGen")):
            probes = VM.build_probes(H.items, m)
            res = HE._run_planned(H, len(probes), HE._plan(H, probes))
            em = [bool(r["em"]) and r["finite"] for r in res]
            acc = [r["acc"] if r["finite"] else 0.0 for r in res]
            out[f"{key}_exact"] = float(sum(em) / len(em) * 100)
            out[f"{key}_token"] = float(sum(acc) / len(acc) * 100)
    vc = H._setup.get("vcache")
    if vc is not None:
        vc.clear()
    H.reset()
    return out


def _forget(nm: str):
    from . import behave, methods
    LG._OPEN.pop(nm, None)
    methods._VALUES.pop(nm, None)
    for key in [key for key in methods._BASE if key[0] == nm]:
        methods._BASE.pop(key)
    for key in [key for key in behave._SES if key[0] == nm]:
        behave._SES.pop(key)


def build(parent: str, ks=KS + (CHECK_K,), verbose: bool = True) -> dict:
    """Build the prefix logs of a development log (GPU; one backbone = one process): materialise, composite by the harness's rule,
    load through the unmodified harness, evaluate the prefix's deployed weight through fx.behave (teacher forcing, B = 1: edits and
    every valid probe row), write the final probe records and prefix.json.  k = 100: the checks against the development log."""
    from . import behave, methods as MT, metrics
    import rescope
    if parent not in LG.ROLES["development"]:
        raise ValueError("prefix logs are built for the development logs only")
    say = (lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)) if verbose else (lambda m: None)
    Hp = LG.load_model(parent)
    hz = rescope.harness()
    tl = hz.HL.TestbedLog(LG.REGISTRY[parent]["path"])
    W0, WEp, dev = Hp.W0, Hp.WE, Hp.dev
    stored_comp = tl.composite_delta()
    pfac = torch.load(os.path.join(LG.REGISTRY[parent]["path"], "edit_factors.pt"), map_location="cpu")
    Lpar = LG.open(parent)
    prev, res = Hp, {}
    for k in ks:
        nm, nb = name(parent, k), k // BATCH
        say(f"{nm}: materialise")
        d = materialise(parent, k)
        S = torch.zeros(W0.shape, dtype=torch.float32, device=dev)
        for b in range(nb):                                             # HLog._init_tb: fp32 sum in batch order on the model's device
            S += tl.batch_delta(b).to(dev)
        WE = (W0.float() + S).to(W0.dtype)
        C = (WE.float() - W0.float()).cpu()
        torch.save(C, os.path.join(d, "composite_delta.pt"))
        del S
        LG.REGISTRY[nm]["need"] = NEED_BUILD
        try:
            _forget(nm)
            H = load_model(nm, reuse=prev, _building=True)
            prev = H
            L = LG.open(nm)
            ses = behave.session(L, H)
            dep = MT.deployed(L)
            t0 = time.time()
            ed = ses.edits(dep)
            pr = ses.tf(dep, L.rows("probe", valid=True))
            sec = time.time() - t0
            d1 = {(r["probe_idx"], r["ptype"]): (not r["D1"], r["first_mismatch"]) for r in pr}
            _write_records(parent, k, d1)
            vend = _vendor_edit_metrics(H)                               # the stored-metric fields the harness's baseline() reads
            mp = os.path.join(d, "metrics.json")
            met = fx.io.jload(mp)
            met.update(vend, TLoc_probes=None, ILoc_probes=None, Rel_exact_n=int(round(vend["Rel_exact"] * k / 100.0)),
                       stored_metric_note="Rel / TGen / IGen: vendor pass in padded batches of 8 at the prefix weight (harness plan); not a reported number")
            fx.io.jdump(met, mp, indent=1)
        finally:
            LG.REGISTRY[nm]["need"] = NEED
        es = metrics.edit_stats(ed)
        hd = [r for r in pr if r["disjoint"]]
        rp = [r for r in pr if r["pool"] == "repair"]
        hn = {}
        for j in L.rows("hn", valid=True):
            hn[L.index[j]["hn_type"]] = hn.get(L.index[j]["hn_type"], 0) + 1
        fac = torch.load(os.path.join(d, "edit_factors.pt"), map_location="cpu")
        desc = {"name": nm, "prefix_of": parent, "T": k, "n_batches": nb, "backbone": L.backbone, "complete": True, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "code": fx.code_fingerprint(), "weight_sig": ses.wsig(H.WE), "seconds_eval": sec,
                "deployed": {"rel": {"n": es["rel"]["n"], "exact": es["rel"]["exact"], "n_exact": sum(bool(r["exact"]) for r in ed if r["metric"] == "rel")},
                             "tgen": {"n": es["tgen"]["n"], "exact": es["tgen"]["exact"], "n_exact": sum(bool(r["exact"]) for r in ed if r["metric"] == "tgen")},
                             "igen": {"n": es["igen"]["n"], "exact": es["igen"]["exact"], "n_exact": sum(bool(r["exact"]) for r in ed if r["metric"] == "igen")},
                             "heldout_disjoint": {"n": len(hd), "n_D1": sum(bool(r["D1"]) for r in hd), "D1": metrics.pct(r["D1"] for r in hd)},
                             "repair": {"n": len(rp), "n_D1": sum(bool(r["D1"]) for r in rp), "D1": metrics.pct(r["D1"] for r in rp),
                                        "n_D1_image": sum(bool(r["D1"]) for r in rp if r["ptype"] == "image"), "n_D1_text": sum(bool(r["D1"]) for r in rp if r["ptype"] == "text")}},
                "hard_negatives_valid": hn, "n_probe_rows_valid": len(pr), "stored_metric_fields_batch8": vend,
                "checks": {"harness_load_checks": {c: bool(v) for c, v in H.checks.items()}, "WE_equals_rule": bool(torch.equal(H.WE, WE)),
                           "factors_equal_parent_rows": bool(all(torch.equal(fac[key], pfac[key][:k]) for key in ("U", "W", "K1_solver", "k_t_single")) and list(fac["uids"]) == list(pfac["uids"][:k])),
                           "hook_mismatch": int(ses.stats["hook_mismatch"]), "memo_conflicts": int(ses.memo.stats.get("conflicts", 0))}}
        if k == LG.REGISTRY[parent]["T"]:                               # all batches: the development log itself
            stored = {(r["probe_idx"], r["ptype"]): r for r in (Lpar.probe_rec(j) for j in Lpar.probe_rows)}
            diff = sum(bool(stored[key]["is_complaint_D1"]) != (bool(stored[key]["y0_valid"]) and not v[0]) for key, v in d1.items())
            pmet = fx.io.jload(os.path.join(LG.REGISTRY[parent]["path"], "metrics.json"))
            desc["checks"]["batch8_metrics_equal_parent_stored"] = {key: abs(vend[key] - float(pmet[key])) < 1e-6 for key in ("Rel_exact", "Rel_token", "TGen_exact", "TGen_token", "IGen_exact", "IGen_token")}
            desc["checks"].update(all_batches={"composite_equals_stored_composite_bitwise": bool(torch.equal(C, stored_comp.cpu())),
                                               "WE_equals_development_WE_bitwise": bool(torch.equal(WE, WEp)),
                                               "n_valid_probe_rows": len(d1), "n_D1_flags_differing_from_stored_records": int(diff),
                                               "n_stored_D1_complaints": sum(bool(r["is_complaint_D1"]) for r in stored.values()),
                                               "n_D1_complaints_here": sum(bool(stored[key]["y0_valid"]) and not v[0] for key, v in d1.items())})
        fx.io.jdump(desc, os.path.join(d, "prefix.json"), indent=1)
        _forget(nm)
        res[nm] = desc
        say(f"{nm}: Rel {desc['deployed']['rel']['exact']:.0f} T-Gen {desc['deployed']['tgen']['exact']:.0f} I-Gen {desc['deployed']['igen']['exact']:.0f} "
            f"D1 heldout_disjoint {desc['deployed']['heldout_disjoint']['D1']:.1f} ({sec:.0f}s); checks {desc['checks'].get('all_batches', '')}")
    return res


def table() -> str:
    """Render out/prefix_logs.txt from the prefix.json files (no number typed by hand)."""
    rows, notes = [], []
    for parent in LG.ROLES["development"]:
        for k in KS + (CHECK_K,):
            nm = name(parent, k)
            if not complete(nm):
                rows.append((nm, "NOT BUILT") + ("",) * 9)
                continue
            d = describe(nm)
            x = d["deployed"]
            hn = d["hard_negatives_valid"]
            rows.append((nm, str(d["T"]), str(d["n_batches"]), f"{x['rel']['exact']:.1f} ({x['rel']['n_exact']}/{x['rel']['n']})",
                         f"{x['tgen']['exact']:.1f} ({x['tgen']['n_exact']}/{x['tgen']['n']})", f"{x['igen']['exact']:.1f} ({x['igen']['n_exact']}/{x['igen']['n']})",
                         f"{x['heldout_disjoint']['D1']:.2f} ({x['heldout_disjoint']['n_D1']}/{x['heldout_disjoint']['n']})",
                         f"{x['repair']['n_D1']} ({x['repair']['n_D1_image']} image + {x['repair']['n_D1_text']} text) of {x['repair']['n']}",
                         " / ".join(str(hn.get(t, 0)) for t in ("HN-I", "HN-Q", "HN-X")),
                         "yes" if all(d["checks"]["harness_load_checks"].get(c) for c in ("WE_float_minus_W0_equals_stored_composite", "rollback_empty_equals_WE", "factor_uids_match_stream"))
                         and d["checks"]["WE_equals_rule"] and d["checks"]["factors_equal_parent_rows"] else "NO", d["code"]["fx"]))
            ab = d["checks"].get("all_batches")
            if ab:
                notes.append(f"  {nm} (all 10 batches = the development log): composite == stored composite bitwise {ab['composite_equals_stored_composite_bitwise']}; "
                             f"W_E == the development log's W_E bitwise {ab['WE_equals_development_WE_bitwise']}; D1 flags differing from the stored probe records "
                             f"{ab['n_D1_flags_differing_from_stored_records']} of {ab['n_valid_probe_rows']} valid rows ({ab['n_D1_complaints_here']} complaints here, "
                             f"{ab['n_stored_D1_complaints']} stored)")
    head = ("log", "T", "batches", "Rel exact % (n)", "T-Gen exact % (n)", "I-Gen exact % (n)", "D1 % heldout_disjoint (n)", "D1 complaints in REPAIR", "HN-I / HN-Q / HN-X valid",
            "load checks", "fx code")
    txt = [f"PREFIX LOGS OF THE DEVELOPMENT STREAMS (R12) -- rendered {time.strftime('%Y-%m-%d %H:%M:%S')} by `python -m fx.prefix table` from out/prefix_logs/<log>/prefix.json",
           "deployed weight of a prefix = pre-edit weight + the stored applied batch updates of its first batches (harness rule, fp32 sum, cast to the module dtype);",
           "Rel / T-Gen / I-Gen: exact, teacher forced, B = 1, over the prefix's edits; D1: valid rows whose pre-edit answer is not reproduced (teacher forcing, B = 1);",
           "hard negatives: valid rows of the prefix's edits (HN-X: both edits in the prefix); '-p100' rows are the construction check, not experiment logs.", "",
           fx.io.text_table(head, rows), ""] + notes
    out = "\n".join(txt) + "\n"
    os.makedirs(os.path.dirname(TABLE), exist_ok=True)
    with builtins.open(TABLE, "w") as f:
        f.write(out)
    return out


_register()

if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3 and sys.argv[1] == "build":
        ks = tuple(int(x) for x in sys.argv[3:]) or KS + (CHECK_K,)
        build(sys.argv[2], ks)
        print(table())
    elif len(sys.argv) >= 2 and sys.argv[1] == "table":
        print(table())
    else:
        print(__doc__)
