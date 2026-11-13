"""The row set of E10 on ONE text log, all metrics (i) - (vi): the mechanics run of the development log, and the template of
the registered experiment.

    $GQ run --mem 12000 --prio -8 --name e10t_dev -- $PY scripts/run_rowset.py                       # development log, draw 0
    $GQ run --mem 12000 --prio -8 --name e10t_gen -- $PY scripts/run_rowset.py --parts gen           # (parts may run as separate jobs)
    python scripts/run_rowset.py --parts gate                                                        # CPU only
    python scripts/render_table.py                                                                   # -> out/dev_table.txt

parts   gate       (iv) gate level over --n-draws calibration draws (CPU; --feature-device cuda for long logs)
        behave     (i) new object on request / paraphrase prompts, (iii) D1 on HELD-OUT benign and neighbourhood prompts, draw --draw
        public     (ii) public scores of every row (public evaluation code on the served model)
        gen        (v) n-gram entropy of the public generations: deployed, pre-edit, default method
        retention  (vi) the first --first requests under the deployed log, the default method and the two ReScope-W rows (--retention-rows) for the
                   prefixes --prefixes of the log (decode-based match, public scores, D1 on their neighbourhood prompts)
Outputs under out/<tag>/ (tag = 'dev' for the development log): parts/<part>.json (everything the table shows),
records/<parts>/results.json (flat records, fx.io) with rows/*.jsonl.gz (per-row flags), public/ (public-format case files),
gen/ (generated texts).
Resumable: behavioural outcomes are kept in the store, public scores in their case files, generations in their files.
A log other than the development log is refused unless --reported-ok is given (the registered experiment).
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "code"))
import e10t  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--log", default=e10t.DEV_LOG)
ap.add_argument("--parts", default="gate,behave,public,gen,retention")
ap.add_argument("--draw", type=int, default=0)
ap.add_argument("--n-draws", type=int, default=e10t.N_DRAWS)
ap.add_argument("--alpha", type=float, default=e10t.ALPHA)
ap.add_argument("--rows", default=",".join(e10t.methods.ROWSET_R14), help="default: the eight rows + the two ReScope-W rows of amendment R14")
ap.add_argument("--gen-rows", default="deployed,preedit,rescope")
ap.add_argument("--retention-rows", default="deployed,rescope,where_logonly,where_twoblock",
                help="E9 (c) as amended: the deployed log, the thresholds-only repair (ReScope-W) and the default method")
ap.add_argument("--prefixes", default=None, help="development log: 20,50,100; stream: 100,500,1000,2000 up to the log's length")
ap.add_argument("--first", type=int, default=None, help="development log: 20; stream: 100")
ap.add_argument("--feature-device", default="cpu")
ap.add_argument("--tag", default=None)
ap.add_argument("--reported-ok", action="store_true")
ap.add_argument("--fresh", action="store_true", help="do not read behavioural outcomes stored by earlier runs (recompute and store again)")
args = ap.parse_args()
PARTS = args.parts.split(",")
if args.log != e10t.DEV_LOG and not args.reported_ok:
    raise SystemExit(f"{args.log} is not the development log: methods run on it only in the registered experiment (--reported-ok)")
TAG = args.tag or ("dev" if args.log == e10t.DEV_LOG else args.log)
OUT = os.path.join(e10t.OUT, TAG)
os.makedirs(os.path.join(OUT, "parts"), exist_ok=True)
NEED_GPU = any(p in PARTS for p in ("behave", "public", "gen", "retention"))
B = e10t.Backbone() if NEED_GPU else None
if B is None and args.feature_device != "cpu":                         # gate level with features on the GPU: a queue job without the model (--mem 4000)
    import torch as _t
    if "GQ_MEM" in os.environ:
        _t.cuda.set_per_process_memory_fraction((int(os.environ["GQ_MEM"]) - 600) * 2 ** 20 / _t.cuda.get_device_properties(0).total_memory, 0)

import fx  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

L = e10t.open_log(args.log)
if args.first is None:
    args.first = 20 if L.development else 100
if args.prefixes is None:
    args.prefixes = "20,50,100" if L.development else ",".join(str(t) for t in (100, 500, 1000, 2000) if t <= L.T)
LABELS = args.rows.split(",")
dtag = lambda M: M.label + ("" if M.draw is None else f"__d{M.draw}")                              # noqa: E731  (file tag of a method)
R = fx.io.Results(f"E10-{TAG}", os.path.join(OUT, "records", "+".join(PARTS)), load=False)      # one record file per invocation (parts may run as separate jobs)
jd = lambda obj, name: e10t.data.jdump(fx.io.clean(obj), os.path.join(OUT, "parts", name))      # noqa: E731
say = lambda *a: print(f"[{time.strftime('%H:%M:%S', time.gmtime())}]", *a, flush=True)          # noqa: E731
meta = {"log": L.name, "describe": L.describe(), "alpha": args.alpha, "draw": args.draw, "code": e10t.code_fingerprint(), "args": vars(args),
        "undefined_rows": e10t.methods.UNDEFINED}


def slim(rec):
    """A method record without the long lists."""
    out = {}
    for k, v in rec.items():
        if isinstance(v, dict):
            v = slim(v)
        if isinstance(v, list) and len(v) > 20:
            continue
        out[k] = v
    return out


# ----------------------------------------------------------------------------- (iv) gate level over calibration draws
if "gate" in PARTS:
    t0 = time.time()
    per, recs = {}, {}
    gparts = ("heldout", "paraphrase", "neighborhood", "request")
    for d in range(args.n_draws):
        for M in e10t.methods.rowset(L, alpha=args.alpha, draw=d, labels=[x for x in LABELS if x not in ("deployed", "preedit")]):
            if M.label not in e10t.methods.DEPENDS_ON_DRAW and d > 0:
                continue
            st = e10t.gate.gate_stats(L, e10t.gate.evaluate(L, M, parts=gparts, device=args.feature_device))
            c = M.rec.get("cal", {})
            st["cal"] = {"eta": c.get("eta"), "n_edits_capped": c.get("n_edits_capped"), "bound": c.get("bound_false_firing"),
                         "n_own_not_candidate": M.rec["own_requests"]["n_not_candidate"], "n_own_not_released": M.rec["own_requests"]["n_not_released"],
                         "threshold_median": M.rec["thresholds"]["median"], "seconds_build": M.rec.get("seconds")}
            per.setdefault(M.label, {})[d] = st
        say(f"gate level: draw {d} done ({time.time() - t0:.0f}s)")
    summ = {}
    for lab, by in per.items():
        vals = {}
        for part in gparts:
            for k in ("fired", "fired_clean", "fired_near_scope", "own_released", "other_released", "only_other_released", "fired_prompt", "fired_last_only", "near_band",
                      "lineage_width_mean_if_fired", "lineage_width_max"):
                if k in by[next(iter(by))][part]:
                    vals[f"{part}.{k}"] = {d: by[d][part][k] for d in by}
        for k in ("eta", "n_edits_capped", "n_own_not_released", "threshold_median"):
            vals[f"cal.{k}"] = {d: by[d]["cal"][k] for d in by}
        summ[lab] = {}
        for k, v in vals.items():
            if lab in e10t.methods.DEPENDS_ON_DRAW:
                s = R.add_draws(L.name, lab, "gate." + k, v, alpha=args.alpha, n=e10t.N_CAL, pool=k.split(".")[0])
            else:
                R.add(L.name, lab, "gate." + k, v[0], pool=k.split(".")[0])
                s = fx.stats.over_draws([v[0]])
            summ[lab][k] = s
    jd({"meta": meta, "n_draws": args.n_draws, "per_draw": per, "summary": summ, "seconds": time.time() - t0,
        "n": {p: len(e10t.gate.part_rows(L, p)) for p in gparts}}, "gate.json")
    say(f"gate level done: {args.n_draws} draws, {time.time() - t0:.0f}s")

methods = e10t.methods.rowset(L, alpha=args.alpha, draw=args.draw, labels=LABELS) if NEED_GPU else []
by_label = {M.label: M for M in methods}
ses = e10t.behave.Session(L, B, fresh=args.fresh, device=args.feature_device) if NEED_GPU else None
kw = lambda M: dict(alpha=M.alpha, n=M.n, draw=M.draw, variant=M.variant)                         # noqa: E731

# ----------------------------------------------------------------------------- (i) + (iii) behaviour
if "behave" in PARTS:
    out = {"meta": meta, "methods": {}, "records": {M.label: slim(M.rec) for M in methods}}
    for M in methods:
        t0 = time.time()
        before = dict(ses.stats)
        e = ses.edits(M)
        p = ses.probes(M, "heldout")
        nb = ses.probes(M, "neighborhood")
        es, ps, ns = e10t.behave.edit_stats(e, ci=True), e10t.behave.probe_stats(p, ci=True), e10t.behave.probe_stats(nb, ci=True)
        cost = {k: ses.stats[k] - before[k] for k in ("sequences_run", "batches", "silent", "from_store", "near", "hook_mismatch", "hook_mismatch_dec", "seconds_model")}
        cost.update(seconds=time.time() - t0, n_prompts=len(e) + len(p) + len(nb), n_passes=2 * (len(e) + len(p) + len(nb)))
        out["methods"][M.label] = {"edits": es, "heldout": ps, "neighborhood": ns, "cost": cost, "kind": M.kind, "key": M.key}
        R.add_nested(L.name, M.label, es, prefix="edit.", pool="edits", **kw(M))
        R.add_nested(L.name, M.label, ps, prefix="heldout.", pool="heldout", **kw(M))
        R.add_nested(L.name, M.label, ns, prefix="neighborhood.", pool="neighborhood", **kw(M))
        R.rows(f"{L.name}__{M.label}__d{args.draw}", e + p + nb)
        say(f"behave {M.label}: {cost['sequences_run']} sequences run, {cost['silent']} silent, {cost['from_store']} from the store, {cost['seconds']:.0f}s")
        jd(out, "behave.json")
    assert ses.stats["hook_mismatch"] == 0, ses.stats["mismatches"]
    out["session"] = {k: v for k, v in ses.stats.items() if k != "mismatches"}
    out["peak_alloc_MiB"] = torch.cuda.max_memory_allocated() / 2 ** 20
    jd(out, "behave.json")

# ----------------------------------------------------------------------------- (ii) public scores
if "public" in PARTS:
    out = {"meta": meta, "methods": {}}
    for M in methods:
        sc = e10t.public.scores(ses, M, os.path.join(OUT, "public"), tag=dtag(M), verbose=50)
        out["methods"][M.label] = sc
        for k in ("ES", "PS", "NS", "EM", "PM", "NM", "S"):
            R.add(L.name, M.label, "public." + k, sc[k], pool="records", n_units=sc["n"], **kw(M))
        say(f"public {M.label}: ES {sc['ES']:.1f} PS {sc['PS']:.1f} NS {sc['NS']:.1f} ({sc['n_new']} new records, {sc['seconds']:.0f}s)")
        jd(out, "public.json")

# ----------------------------------------------------------------------------- (v) degeneration
if "gen" in PARTS:
    out = {"meta": meta, "methods": {}}
    os.makedirs(os.path.join(OUT, "gen"), exist_ok=True)
    order = [x for x in args.gen_rows.split(",") if x in by_label]
    order = sorted(order, key=lambda x: x != "preedit")                  # the pre-edit texts first: they are the reference
    ref = None
    for lab in order:
        M = by_label[lab]
        t0 = time.time()
        p = os.path.join(OUT, "gen", f"{L.name}__{dtag(M)}.json")
        g = e10t.public.generation(ses, M, p, verbose=10)
        if lab == "preedit":
            ref = g["per_text"]
        g = e10t.public.generation_summary(e10t.data.jload(p)["rows"], ref)
        g["seconds"], g["file"] = time.time() - t0, p
        out["methods"][lab] = {k: v for k, v in g.items() if k != "per_text"}
        R.add(L.name, lab, "gen.GE", g["GE"], pool="generation_first100", n_units=g["n_records"], **kw(M))
        say(f"generation {lab}: GE {g['GE']:.1f} ({g['seconds']:.0f}s)")
        jd(out, "gen.json")

# ----------------------------------------------------------------------------- (vi) retention
if "retention" in PARTS:
    out = {"meta": meta, "first": args.first, "by_T": {}}
    for T in [int(x) for x in args.prefixes.split(",")]:
        if T > L.T or T < args.first:
            continue
        LT = L if T == L.T else e10t.open_log(args.log, T=T)
        sT = ses if T == L.T else e10t.behave.Session(LT, B, fresh=args.fresh, device=args.feature_device)
        o = {}
        for M in e10t.methods.rowset(LT, alpha=args.alpha, draw=args.draw, labels=[x for x in args.retention_rows.split(",") if x in LABELS]):
            e = sT.edits(M, first=args.first)
            nb = sT.probes(M, "neighborhood", first=args.first)
            sc = e10t.public.scores(sT, M, os.path.join(OUT, "public"), first=args.first, tag=dtag(M))
            o[M.label] = {"edits": e10t.behave.edit_stats(e), "neighborhood": e10t.behave.probe_stats(nb), "key": M.key,
                          "public": {k: sc[k] for k in ("n", "ES", "PS", "NS", "EM", "PM", "NM", "S")},
                          "cal": {k: M.rec.get("cal", {}).get(k) for k in ("eta", "n_edits_capped")} if M.kind == "gate" else None}
            R.add_nested(L.name, M.label, o[M.label]["edits"], prefix=f"retention.T{T}.edit.", pool=f"first{args.first}", **kw(M))
            R.rows(f"{LT.name.replace('@', '_at_')}__{M.label}__d{args.draw}__first{args.first}", e + nb)
        assert sT.stats["hook_mismatch"] == 0
        out["by_T"][T] = o
        say(f"retention: T = {T} done")
    jd(out, "retention.json")

R.save(parts=PARTS, log=L.name, e10t=e10t.code_fingerprint())
if B is not None:
    B.check_w0()
    say("peak MiB", torch.cuda.max_memory_allocated() / 2 ** 20)
say("done", PARTS)
