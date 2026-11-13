"""Independent verification of E2 a-c (+ running example): GPU job of ONE log.

    gq run --mem 10500 -- python vjob.py TB-Q100-ae-l2_3        (17000 for TB-L100-ae-l2_3)

No driver, no memo, no key cache: the model is loaded with rescope.load, every row comes from rescope.Evaluator.tf_rows /
generate (the library's plain evaluator), every threshold is computed HERE from coordinates captured HERE
(r_i(k) = <W_i, k> / <W_i, k_i>, fp64, on the model's device).  The library is used for: the handle, the Evaluator, the
ScopedLog / Hook objects that carry my thresholds, the seeded block split of the two-block rule (a draw definition) and the
shared harness for COMP / ROLLBACK_FO (the repair itself is not the object of this verification).
Resumable: captures, the rebuilt compensation, every chunk of the evaluation pass and every part of the example are files.
"""
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP
import gzip
import json
import math
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import numpy as np
import torch

sys.path.insert(0, f"{_RP.ROOT}/02_research/code")
import rescope
from rescope import Evaluator, ScopedLog, from_harness, harness, load, splits
from rescope.calibrate import split_blocks

NAME = sys.argv[1]
ONLY = sys.argv[2].split(",") if len(sys.argv) > 2 else None
FIN = f"{_RP.ROOT}/02_research/experiments/10_final"
VER = FIN + "/e2_curves/verify"
OUT = f"{VER}/out/{NAME}"
P12 = f"{_RP.ROOT}/02_research/experiments/08_rescope/p12_linear_streams"
os.makedirs(OUT, exist_ok=True)
IS_Q = NAME.startswith("TB-Q")
INF = float("inf")
T0 = time.time()


def say(*a):
    print(f"[{time.strftime('%H:%M:%S')}] {NAME[3]}", *a, flush=True)


def jdump(obj, path):
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items() if not str(k).startswith("_")}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            x = float(x)
        if isinstance(x, (np.bool_,)):
            return bool(x)
        if torch.is_tensor(x):
            return clean(x.tolist())
        if isinstance(x, float) and not math.isfinite(x):
            return "inf" if x > 0 else ("-inf" if x < 0 else "nan")
        return x
    tmp = path + ".tmp"
    op = gzip.open if path.endswith(".gz") else open
    with op(tmp, "wt") as fh:
        json.dump(clean(obj), fh)
    os.replace(tmp, path)


def jload(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as fh:
        return json.load(fh)


# ----------------------------------------------------------------------------- model, items, log
cap_gb = (int(os.environ.get("GQ_MEM", "10500")) - 600) / 1024.0
H = load(NAME, device="cuda:0", mem_gb=cap_gb, keep_batch_deltas=IS_Q)
ev = Evaluator(H, hard_negatives=False)
hz = harness()
if IS_Q or not os.path.exists(f"{OUT}/comp_b_all.pt"):
    H.solve_dev = torch.device("cpu")                              # as P12: COMP solves on the CPU (fp64); the first LLaVA build of this job ran on the GPU (recorded)
log = from_harness(H).to(H.dev)
T = log.T
Wd, Kd, Ud = log.W.double(), log.K.double(), log.U.double()
s_own = (Wd * Kd).sum(1)                                                # s_i = <W_i, k_i>
assert bool((s_own > 0).all())
say(f"loaded: T {T}, cap {cap_gb:.1f} GB, {time.time() - T0:.0f}s")


def coords(x):
    """(n, T) fp64 normalised solver coordinates of keys x (n, d)."""
    return (x.double() @ Wd.T) / s_own[None, :]


EDIT_ITEMS = ev.edit_items()
HO_ITEMS = ev.probe_items("heldout")
REP_ITEMS = ev.probe_items("repair")
HD_ITEMS = [it for it in HO_ITEMS if it["disjoint"]]
by_group_rep = {it["group"]: it for it in REP_ITEMS}
jdump({"log": NAME, "heldout": HO_ITEMS, "n_heldout_valid": len(HO_ITEMS), "n_heldout_disjoint": len(HD_ITEMS), "n_heldout_strict": sum(it["strict"] for it in HO_ITEMS),
       "n_repair_valid": len(REP_ITEMS), "n_probe_rows": len(H.rows), "n_probe_valid": sum(bool(r["y0_valid"]) for r in H.rows)}, f"{OUT}/pools.json")

# ----------------------------------------------------------------------------- draws (rebuilt here; checked against the prep file)
prep = jload(f"{VER}/out/prep/draws_{NAME}.json")


def build_stream(seed):
    """P12's nested complaint stream (p12_common.build_stream), re-implemented."""
    base = hz.HS.sample_complaints(H, 100, seed=seed)
    img, txt = [c for c in base if c["ptype"] == "image"], [c for c in base if c["ptype"] == "text"]
    stream = [txt[k // 2] if k % 2 == 0 else img[k // 2] for k in range(100)]
    used = {c["group"] for c in stream}
    rows = sorted([r for r in H.rows if ev.pool[r["probe_idx"]] == "repair" and r["y0_valid"]], key=lambda r: (r["probe_idx"], r["ptype"]))
    rest = [dict(r, complaint_kind="D1", sample_seed=seed) for r in rows if r["is_complaint_D1"] and r["group"] not in used]
    return stream + [rest[int(i)] for i in np.random.default_rng([seed, 12]).permutation(len(rest))]


STREAM = build_stream(0)
SG = [c["group"] for c in STREAM]
assert SG == prep["p12_reported_groups"]["ALL"], "complaint stream differs from the P12 record"
assert SG[:10] == prep["p12_reported_groups"]["10"] and SG[:100] == prep["p12_reported_groups"]["100"]
n_pool = len(REP_ITEMS)
base = np.sort(np.random.default_rng(0).choice(n_pool, 200, replace=False))
rest = np.setdiff1d(np.arange(n_pool), base)
rk = np.concatenate([base[np.random.default_rng([0, 1]).permutation(base.size)], rest[np.random.default_rng([0, 2]).permutation(rest.size)]])
CAL200 = [REP_ITEMS[int(i)] for i in base]
CAL10 = [REP_ITEMS[int(i)] for i in np.sort(rk[:10])]
assert [it["sid"] for it in CAL200] == [it["sid"] for it in splits.sample(REP_ITEMS, 200, 0)]
assert [it["sid"] for it in CAL200] == prep["driver"]["cal200"] and [it["sid"] for it in CAL10] == prep["driver"]["cal10"]
say(f"draws ok: stream {len(STREAM)}, cal200 / cal10 equal to the driver's draw 0")

# ----------------------------------------------------------------------------- the running example's complaint (rule of the task)
EX = None
if IS_Q:
    for pos, c in enumerate(STREAM):
        if c["ptype"] == "image" and c.get("pre_correct_at10") and len(str(c["y0_text"]).split()) <= 6:
            EX = {"pos": pos, "c": c, "item": by_group_rep[c["group"]]}
            break
    say(f"example complaint: {EX['c']['group']} at stream position {EX['pos']}")

# ----------------------------------------------------------------------------- captures
CAP_P = f"{OUT}/capture.pt"
REL_ITEMS = ev.edit_items(("rel",))


def capture_all():
    if os.path.exists(CAP_P):
        return torch.load(CAP_P)
    need = {}
    for c in STREAM[:100]:
        need[by_group_rep[c["group"]]["sid"]] = by_group_rep[c["group"]]
    for it in CAL200 + REL_ITEMS:
        need[it["sid"]] = it
    if EX is not None:
        need[EX["item"]["sid"]] = EX["item"]
    out = {}
    t0 = time.time()
    with torch.no_grad(), hz.HE.vision_cache(H):
        for n, (sid, it) in enumerate(need.items()):
            ev.trim_vcache()
            x, p = ev.capture(it)
            r = coords(x)                                               # (positions, T)
            rec = {"rmax": r.max(0).values.cpu(), "rlast": r[p].cpu(), "n_pos": int(x.shape[0]), "lp": int(p)}
            if it["kind"] == "edit":                                    # log-only negatives: the own column without its solver position
                r2 = r.clone()
                r2[p, it["edit"]] = -INF
                rec["colmax_excl_own_lp"] = r2.max(0).values.cpu()
            out[sid] = rec
            if (n + 1) % 100 == 0:
                say(f"capture {n + 1}/{len(need)} ({time.time() - t0:.0f}s)")
    torch.save(out, CAP_P + ".tmp")
    os.replace(CAP_P + ".tmp", CAP_P)
    return out


CAP = capture_all()
say(f"captures: {len(CAP)} sequences")

# ----------------------------------------------------------------------------- guards and thresholds (my rules)
MARGIN, ABS_MARGIN, TIE, SELF_MARGIN = 1e-6, 1e-12, 1e-9, 0.05
g_stored = torch.stack([coords(Kd[i:i + 1])[0, i] for i in range(T)]).cpu()                       # = 1 up to round-off
g_own = torch.stack([CAP[f"edit:rel:{i}"]["rlast"][i] for i in range(T)])                         # B = 1 key of the own Rel prompt
GUARDS = torch.stack([g_stored, g_own])                                                           # (2, T): coordinate of edit i on its guards


def above(m):
    b = torch.where(torch.isinf(m), m, m + MARGIN * m.abs() + ABS_MARGIN)
    return torch.where(m == -INF, m, torch.nextafter(b, torch.full_like(m, INF)))


def least(tau, queries):
    """Least repair, one complaint at a time; a complaint whose repair would silence a firing guard is refused."""
    tau = tau.clone()
    refused, raised = [], 0
    for k, m in enumerate(queries):
        new = torch.maximum(tau, above(m))
        before, after = GUARDS >= tau[None, :], GUARDS >= new[None, :]
        if bool((before & ~after).any()):
            refused.append(k)
            continue
        raised += int((new > tau).sum())
        tau = new
    return tau, refused, raised


def cap_guards(lit):
    """Thresholds lowered as little as needed for both guards of every edit to fire: (1 - 1e-9) x the guard coordinate."""
    capv = (GUARDS - TIE * GUARDS.abs()).min(0).values
    sil = (GUARDS < lit[None, :]).any(0)
    return torch.where(sil, torch.minimum(lit, capv), lit), sil


TH = {}
rmax_c = [CAP[by_group_rep[g]["sid"]]["rmax"] for g in SG[:100]]
neg = torch.full((T,), -INF, dtype=torch.float64)
for i in range(T):
    neg = torch.maximum(neg, CAP[f"edit:rel:{i}"]["colmax_excl_own_lp"])
for n in (10, 100):
    tau, ref, nr = least(torch.full((T,), -INF, dtype=torch.float64), rmax_c[:n])
    TH[f"wl_dep_n{n}"] = {"tau": tau, "n_refused": len(ref), "n_raised_total": nr, "still_firing": int(sum(bool((m >= tau).any()) for m in rmax_c[:n]))}
tau_log, sil_log = cap_guards(neg + SELF_MARGIN * neg.abs())
tau, ref, nr = least(tau_log, rmax_c[:10])
TH["wl_log_n10"] = {"tau": tau, "n_refused": len(ref), "n_raised_total": nr, "start_capped": int(sil_log.sum()), "still_firing": int(sum(bool((m >= tau).any()) for m in rmax_c[:10]))}
TH["wl_log_n0"] = {"tau": tau_log}
# migration (role last): log-only pooled cross-talk, + least repair of 10 benign queries, two-block rule on 200
R_kk = coords(Kd).cpu()                                                  # R[j, i] = r_i(k_j)
theta0 = float(R_kk[~torch.eye(T, dtype=torch.bool)].max())
tau0, sil0 = cap_guards(torch.full((T,), theta0, dtype=torch.float64))
TH["c_n0"] = {"tau": tau0, "theta": theta0, "n_capped": int(sil0.sum())}
tau, ref, nr = least(tau0, [CAP[it["sid"]]["rlast"] for it in CAL10])
TH["c_n10"] = {"tau": tau, "n_refused": len(ref), "n_raised_total": nr}
Q = torch.stack([CAP[it["sid"]]["rlast"] for it in CAL200]).numpy()      # (200, T)
i1, i2 = split_blocks(200, (1, 1), 0)
b = Q[i1].max(0)
sc = np.sort((Q[i2] - b[None, :]).max(1))
n2 = len(i2)
k_rank = -((-19 * (n2 + 1)) // 20)                                       # ceil((1 - 1/20)(n2 + 1)), integers
eta = float(sc[k_rank - 1]) if k_rank <= n2 else INF
lit = np.empty(T)
for i in range(T):                                                       # smallest fp64 r with fl(r - b_i) > eta
    t = b[i] + eta
    if not math.isfinite(eta):
        lit[i] = INF
        continue
    while (t - b[i]) > eta:
        t = np.nextafter(t, -np.inf)
    while not (t - b[i]) > eta:
        t = np.nextafter(t, np.inf)
    lit[i] = t
tau200, sil200 = cap_guards(torch.from_numpy(lit))
TH["c_n200"] = {"tau": tau200, "b": torch.from_numpy(b), "eta": eta, "k": int(k_rank), "n2": int(n2), "n_capped": int(sil200.sum()),
                "capped_edits": [int(i) for i in sil200.nonzero().flatten()], "tau_literal": torch.from_numpy(lit)}
jdump({"log": NAME, "guards": {"stored": g_stored, "own_rel_b1": g_own}, "thresholds": TH, "logonly_negatives": neg,
       "cal10": [it["sid"] for it in CAL10], "cal200": [it["sid"] for it in CAL200], "complaints100": SG[:100]}, f"{OUT}/thresholds.json")
say("thresholds: " + json.dumps({k: {kk: vv for kk, vv in v.items() if isinstance(vv, (int, float))} for k, v in TH.items()}))

# ----------------------------------------------------------------------------- methods
BASE_W = ScopedLog.from_log(log, role="all", fire="all", response="lin", values="stored", W0=H.W0)
BASE_M = ScopedLog.from_log(log, role="last", fire="top1", response="repl", values="rebased", W0=H.W0)
dev = lambda t: t.to(H.dev)                                             # noqa: E731
METHODS = {"deployed": None}
for k in ("wl_dep_n10", "wl_dep_n100", "wl_log_n10"):
    METHODS[k] = BASE_W.replace(det_tau=dev(TH[k]["tau"]))
for k in ("c_n0", "c_n10"):
    METHODS[k] = BASE_M.replace(det_tau=dev(TH[k]["tau"]))
METHODS["c_n200"] = BASE_M.replace(det_tau=dev(TH["c_n200"]["tau"]), det_b=dev(TH["c_n200"]["b"]))

COMP_P = f"{OUT}/comp_b_all.pt"
if os.path.exists(COMP_P):
    bb = torch.load(COMP_P, map_location="cpu")
    METHODS["comp_b_all"] = hz.HL.Correction("adapter", Lf=bb["Lf"].to(H.dev), Rf=bb["Rf"].to(H.dev), gate=bb["gate"], meta={})
else:
    t0 = time.time()
    corr, cost = hz.pkg.repair("COMP", H, STREAM, protect=("K_ans",), query="answer", form="adapter")
    torch.save({"Lf": corr.Lf.cpu(), "Rf": corr.Rf.cpu(), "gate": corr.gate, "meta": {k: v for k, v in corr.meta.items() if isinstance(v, (int, float, str, bool))},
                "seconds": time.time() - t0, "solve_dev": str(H.solve_dev)}, COMP_P)
    METHODS["comp_b_all"] = corr
    say(f"COMP-B on all {len(STREAM)} complaints rebuilt in {time.time() - t0:.0f}s")

# ----------------------------------------------------------------------------- the evaluation pass (plain Evaluator, B = 1)
ITEMS = EDIT_ITEMS + HO_ITEMS
CH = 64


def run_pass():
    nch = (len(ITEMS) + CH - 1) // CH
    for c in range(nch):
        p = f"{OUT}/tf_chunk_{c:03d}.json.gz"
        if os.path.exists(p):
            continue
        t0 = time.time()
        rows = ev.tf_rows(METHODS, ITEMS[c * CH:(c + 1) * CH])
        jdump(rows, p)
        say(f"tf chunk {c + 1}/{nch} ({time.time() - t0:.0f}s)")


if ONLY is None or "pass" in ONLY:
    run_pass()


# ----------------------------------------------------------------------------- the running example, from scratch (Qwen log)
def example():
    out_p = f"{OUT}/example.json"
    part_p = f"{OUT}/example_parts.json"
    P = jload(part_p) if os.path.exists(part_p) else {}

    def part(name, fn):
        if name not in P:
            t0 = time.time()
            P[name] = fn()
            jdump(P, part_p)
            P.update(jload(part_p))
            say(f"example part {name} ({time.time() - t0:.0f}s)")
        return P[name]

    c, it = EX["c"], EX["item"]
    cap = CAP[it["sid"]]
    rmax, rlast = cap["rmax"], cap["rlast"]
    t200, b200 = TH["c_n200"]["tau"], TH["c_n200"]["b"]
    fired0 = bool((rlast >= t200).any())
    edit = int(torch.where(rlast >= t200, rlast - b200, torch.full_like(rlast, -INF)).argmax()) if fired0 else int(rmax.argmax())
    e_item = {k: v for k, v in H.items[edit].items() if isinstance(v, (str, int, float))}
    # deployed answer (decoder v2)
    dep = part("deployed_decode", lambda: ev.generate(None, [it])[0])
    # rollback: the per-complaint greedy first-order search re-run here (harness ROLLBACK_FO, budget 10)

    def rb():
        corr, cost = hz.pkg.repair("ROLLBACK_FO", H, [c], budget=10)
        R = [int(i) for i in corr.meta["R"]]
        WR_h = H.rollback_weight(R)
        C = H.WE.float() - H.W0.float()
        X = (Ud[R].T @ Wd[R]).float()                                   # rank-one solver terms of the removed edits
        whole = [bi for bi, ids in enumerate(H.batches) if ids and all(i in set(R) for i in ids)]
        WR = (H.W0.float() + (C - X)).to(H.W0.dtype)
        rel = ev.tf_rows({"rb": WR, "dep": None}, REL_ITEMS + [it])
        d = {r["edit"]: bool(r["exact"]) for r in rel["dep"][:-1]}
        a = {r["edit"]: bool(r["exact"]) for r in rel["rb"][:-1]}
        kept = [i for i in range(T) if i not in set(R)]
        return {"R": R, "resolved_internal": bool(corr.meta["resolved_internal"]), "whole_batches_in_R": whole,
                "my_weight_equals_harness_rollback_weight": bool(torch.equal(WR, WR_h)), "max_abs_diff_to_harness_weight": float((WR.float() - WR_h.float()).abs().max()),
                "n_deleted": len(R), "edits_broken": sum(d[i] and not a[i] for i in kept), "n_kept": len(kept),
                "complaint_reproduced_after_rollback": not rel["rb"][-1]["D1"], "complaint_D1_deployed": bool(rel["dep"][-1]["D1"]),
                "rel_exact_deployed": sum(d.values()), "rel_exact_after": sum(a.values())}
    rbk = part("rollback", rb)
    # compensation of this one complaint (COMP-B options of P12): generated held-out answers against the deployed log's
    c1p = f"{OUT}/comp_example.pt"
    if os.path.exists(c1p):
        bb = torch.load(c1p, map_location="cpu")
        corr1 = hz.HL.Correction("adapter", Lf=bb["Lf"].to(H.dev), Rf=bb["Rf"].to(H.dev), gate=bb["gate"], meta={})
        rank1 = bb["rank"]
    else:
        corr1, _ = hz.pkg.repair("COMP", H, [c], protect=("K_ans",), query="answer", form="adapter")
        rank1 = corr1.meta.get("rank_kept")
        torch.save({"Lf": corr1.Lf.cpu(), "Rf": corr1.Rf.cpu(), "gate": corr1.gate, "rank": rank1}, c1p)
    gen = {"comp": [], "dep": []}
    for a in range(0, len(HD_ITEMS), 93):
        g = part(f"gen_{a}", lambda a=a: {k: [{"sid": r["sid"], "y": r["y"]} for r in v] for k, v in ev.generate({"comp": corr1, "dep": None}, HD_ITEMS[a:a + 93]).items()})
        gen["comp"] += g["comp"]
        gen["dep"] += g["dep"]
    assert [r["sid"] for r in gen["comp"]] == [x["sid"] for x in HD_ITEMS] == [r["sid"] for r in gen["dep"]]
    changed = sum(x["y"] != y["y"] for x, y in zip(gen["comp"], gen["dep"]))
    ctf = part("comp_tf_complaint", lambda: ev.tf_rows(corr1, [it])[0])
    # rescoping: least repair of this one complaint from the deployed scoping; own Rel prompts still released at the last prompt position
    tau1, ref1, nr1 = least(torch.full((T,), -INF, dtype=torch.float64), [rmax])
    S1 = BASE_W.replace(det_tau=dev(tau1))
    rw = part("rescope_rel", lambda: ev.tf_rows(S1, REL_ITEMS + [it]))
    own_fire = sum(bool(r["own_fired_last"]) for r in rw[:-1])
    own_gate = int((g_own >= tau1).sum())
    # why-not pair under the default calibrated method (two-block, n 200, draw 0, my thresholds)
    S200 = METHODS["c_n200"]

    def whynot():
        order = [edit] + [i for i in range(T) if i != edit]
        with torch.no_grad(), hz.HE.vision_cache(H):
            for i in order:
                ti = ev.edit_items(("tgen",), [i])[0]
                ev.trim_vcache()
                x, p = ev.capture(ti)
                r = coords(x[p][None])[0].cpu()
                if float(r[i]) < float(t200[i]) and not bool((r >= t200).any()):
                    g = ev.generate(S200, [ti])[0]
                    tfr = ev.tf_rows(S200, [ti])[0]
                    return {"edit": i, "coordinate": float(r[i]), "threshold": float(t200[i]), "answer": g["y"], "line": g["line"], "fired_in_decode": g.get("fired"),
                            "fired_tf_hook": tfr.get("fired"), "edit_item": {k: v for k, v in H.items[i].items() if isinstance(v, (str, int, float))},
                            "n_checked": order.index(i) + 1}
        return None
    wn = part("whynot", whynot)
    out = {"log": NAME, "T": T,
           "selection": {"group": c["group"], "sid": it["sid"], "stream_position_seed0": EX["pos"], "ptype": c["ptype"], "pre_correct_at10": bool(c["pre_correct_at10"]),
                         "n_words_pre_answer": len(str(c["y0_text"]).split()), "ref_answer": c.get("ref_answer")},
           "edit_index": edit, "edit_chosen_by": "released by the default method" if fired0 else "largest coordinate over all positions (default method releases nothing)",
           "edit_item": e_item,
           "complaint": {"query": c["prompt"], "pre_answer": c["y0_text"], "deployed_answer": dep["y"], "deployed_answer_record": c.get("yE_text"),
                         "max_coordinate": float(rmax.max()), "max_coordinate_last_prompt": float(rlast.max()), "n_pos": cap["n_pos"]},
           "rollback": rbk, "compensation": {"heldout_changed": int(changed), "rank": rank1, "complaint_reproduced": not ctf["D1"]},
           "heldout": {"n": len(HD_ITEMS)},
           "rescope": {"own_requests_firing": int(own_fire), "own_requests_firing_gate_level": own_gate, "refused": len(ref1), "n_raised": nr1,
                       "rel_exact_after": sum(bool(r["exact"]) for r in rw[:-1]), "complaint_fires_after_repair": bool(rw[-1]["fired"]),
                       "complaint_reproduced_after_repair": not rw[-1]["D1"]},
           "whynot": wn}
    jdump(out, out_p)
    say("example: " + json.dumps({k: out[k] for k in ("edit_index", "complaint", "compensation", "rescope", "whynot")})[:900])


if IS_Q and (ONLY is None or "example" in ONLY):
    example()
jdump({"log": NAME, "done": True, "seconds": time.time() - T0, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
       "library_files": "rescope (plain Evaluator); no fx, no memo, no key cache"}, f"{OUT}/DONE.json")
say(f"job done in {time.time() - T0:.0f}s")
