"""Verification of E1, GPU part: the library's PLAIN Evaluator (no driver, no memo, no key cache); every row is a real B = 1 forward.

    gq run --mem 10500|17000 -- python v_gpu.py <development log>

Calibration draw 0, alpha = 0.05, n = 200.  Gates are built on the model's device from keys captured here:
    rescope         feature 'ncoord', guards (stored keys + B = 1 own Rel keys), two-block rule (seed 0)
    rescope_shared  the same detectors, shared-scale rule
    rescope_null    null repair (ridge 0) on the teacher-forced last-prompt keys of 100 REPAIR queries that are no calibration query
                    (sample: the rule of the driver's USAGE, re-implemented here from the library's rows), then the two-block rule
    cosine_shared   feature 'cos', guards, shared-scale rule
    deployed        W_E
Blocks (each writes out/rows/<log>__<block>.json.gz; a finished block is skipped on restart):
    gates, tf_edit, tf_probe, tf_hn, gen (seeded samples: 60 edit prompts, 60 heldout_disjoint probes, 60 hard negatives; deployed + rescope;
    DUAL on the 60 probes and the first 40 edit prompts), d2 (60 pre-correct heldout_disjoint rows; deployed + rescope), w0 (pre-edit
    teacher-forced rows of the DUAL items, for the labelled composition).
"""
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP
import gzip
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, f"{_RP.ROOT}/02_research/code")
import rescope  # noqa: E402
from rescope import splits  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "out")
ROWS = os.path.join(OUT, "rows")
DEV = ("TB-Q100-ae-l2_3", "TB-L100-ae-l2_3")
ALPHA, N, DRAW, N_NULL = 0.05, 200, 0, 100
SAMPLE_SEED = 20261112
T0 = time.time()


def say(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


def dump(o, p):
    tmp = p + ".tmp"
    with gzip.open(tmp, "wt") as f:
        json.dump(o, f, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    os.replace(tmp, p)


def pub(rows):
    return [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]


def seeded(items, n, salt):
    idx = np.sort(np.random.default_rng([SAMPLE_SEED, salt]).choice(len(items), n, replace=False))
    return [items[int(i)] for i in idx]


def main():
    name = sys.argv[1]
    assert name in DEV, "verification opens the two development logs only"
    os.makedirs(ROWS, exist_ok=True)
    path = lambda blk: os.path.join(ROWS, f"{name}__{blk}.json.gz")   # noqa: E731
    cap_gb = (int(os.environ.get("GQ_MEM", "10500")) - 600) / 1024.0
    H = rescope.load(name, device="cuda:0", mem_gb=cap_gb)
    ev = rescope.Evaluator(H, hn_dir=rescope.HN_DIR_FINAL)
    log = rescope.from_harness(H).to(H.dev)
    say("loaded", name, "T", H.T, "cap GB", round(cap_gb, 2))

    # ------------------------------------------------------------------ gates (always rebuilt: seconds)
    own_k, own_e = ev.own_keys("rel")
    pool = ev.probe_items("repair")                                     # valid REPAIR rows, file order
    n = len(pool)
    idx = np.sort(np.random.default_rng(DRAW).choice(n, N, replace=False))     # = rescope.splits.sample(pool, 200, 0)
    benign = [pool[int(i)] for i in idx]
    assert [b["sid"] for b in benign] == [b["sid"] for b in splits.sample(pool, N, DRAW)]
    base_idx = idx.copy()
    rest = np.setdiff1d(np.arange(n), base_idx)
    rk = np.concatenate([base_idx[np.random.default_rng([DRAW, 1]).permutation(base_idx.size)], rest[np.random.default_rng([DRAW, 2]).permutation(rest.size)]])
    rowof = {(r["probe_idx"], r["ptype"]): r for r in H.rows}
    qk = lambda it: (it["ptype"], rowof[(it["probe_idx"], it["ptype"])]["prompt"], rowof[(it["probe_idx"], it["ptype"])]["image"])   # noqa: E731
    seen, pick = {qk(pool[int(k)]) for k in rk[:N]}, []
    for k in rk[N:]:
        if len(pick) == N_NULL:
            break
        if qk(pool[int(k)]) not in seen:
            seen.add(qk(pool[int(k)]))
            pick.append(int(k))
    null_items = [pool[k] for k in sorted(pick)]
    assert len(null_items) == N_NULL and not ({qk(x) for x in null_items} & {qk(x) for x in benign})

    def base(feature):
        S = rescope.feature_gate(log, feature, role="last", fire="top1", response="repl", values="rebased", W0=H.W0)
        S, rec = rescope.add_guards(S, own_k, own_e)
        return S
    B = base("ncoord")
    keys = [ev.eligible_keys(it, B) for it in benign]
    kn = torch.cat([ev.eligible_keys(it, B) for it in null_items])
    assert all(k.shape[0] == 1 for k in keys) and kn.shape[0] == N_NULL
    G, recs = {}, {}
    G["rescope"], recs["rescope"] = rescope.calibrate_two_block(B, keys, ALPHA, seed=DRAW)
    G["rescope_shared"], recs["rescope_shared"] = rescope.calibrate_shared(B, keys, ALPHA)
    Bn, nrec = rescope.null_repair(B, kn, ridge=0.0)
    G["rescope_null"], recs["rescope_null"] = rescope.calibrate_two_block(Bn, keys, ALPHA, seed=DRAW)
    G["cosine_shared"], recs["cosine_shared"] = rescope.calibrate_shared(base("cos"), keys, ALPHA)
    slim = lambda r: {k: v for k, v in r.items() if isinstance(v, (int, float, str, bool)) or v is None}     # noqa: E731
    grec = {"log": name, "alpha": ALPHA, "n": N, "draw": DRAW, "calibration_sids": [b["sid"] for b in benign], "null_sids": [b["sid"] for b in null_items],
            "cal": {k: slim(v) for k, v in recs.items()}, "capped_edits": {k: [int(x) for x in v.get("capped_edits", [])] for k, v in recs.items()},
            "null_repair": slim(nrec),
            "eff_threshold": {k: [float(x) for x in (g.eta * g.det_tau).detach().cpu().tolist()] for k, g in G.items()},
            "n_hd": len(ev.probe_items("heldout_disjoint")), "n_hn": len(ev.hn_items()), "n_repair_valid": n}
    dump(grec, path("gates"))
    say("gates", {k: (v.get("eta"), v.get("n_edits_capped")) for k, v in recs.items()})
    methods = dict(G, deployed=None)

    # ------------------------------------------------------------------ teacher forcing (every row a forward)
    for blk, fn in (("tf_edit", lambda: ev.edits(methods, verbose=100)), ("tf_probe", lambda: ev.probes(methods, pool="heldout_disjoint", verbose=100)),
                    ("tf_hn", lambda: ev.hard_negatives(methods, verbose=200))):
        if os.path.exists(path(blk)):
            continue
        out = fn()
        dump({k: pub(v) for k, v in out.items()}, path(blk))
        say("done", blk, {k: len(v) for k, v in out.items()})

    # ------------------------------------------------------------------ generation (seeded samples)
    e_s = seeded(ev.edit_items(), 60, 1)
    order = np.random.default_rng([SAMPLE_SEED, 11]).permutation(60)
    e_dual = [e_s[int(i)] for i in sorted(order[:40])]
    e_rest = [it for it in e_s if it["sid"] not in {x["sid"] for x in e_dual}]
    p_s = seeded(ev.probe_items("heldout_disjoint"), 60, 2)
    h_s = seeded(ev.hn_items(), 60, 3)
    d_s = seeded(ev.probe_items("heldout_disjoint", valid_only=False, pre_correct=True), 60, 4)
    dual = ev.dual("eq")
    if not os.path.exists(path("gen")):
        a = ev.generate({"deployed": None, "preedit": H.W0, "rescope": G["rescope"], "dual": dual}, p_s + e_dual, verbose=20)
        say("gen: probes + dual edit prompts decoded")
        b = ev.generate({"deployed": None, "rescope": G["rescope"]}, e_rest + h_s, verbose=20)
        out = {k: pub(a[k]) + pub(b.get(k, [])) for k in a}
        dump(out, path("gen"))
        say("done gen", {k: len(v) for k, v in out.items()})
    if not os.path.exists(path("d2")):
        out = ev.generate({"deployed": None, "rescope": G["rescope"]}, d_s, verbose=20)
        dump({k: pub(v) for k, v in out.items()}, path("d2"))
        say("done d2", {k: len(v) for k, v in out.items()})
    if not os.path.exists(path("w0")):
        out = ev.tf_rows({"preedit": H.W0, "deployed": None}, p_s + e_dual)
        dump({k: pub(v) for k, v in out.items()}, path("w0"))
        say("done w0")
    say("JOB DONE", name)


if __name__ == "__main__":
    main()
