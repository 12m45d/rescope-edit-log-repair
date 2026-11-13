"""white.mechanics: mechanics check of the gates of R3 on the TWO DEVELOPMENT LOGS (and on no other log).

    python -m white.mechanics run      -> out/mechanics_<log>.json      (CPU, key cache; about 1 minute per log)
    python -m white.mechanics render   -> out/mechanics.txt             (rendered from the JSON files; no number is typed by hand)

For calibration draw 0 of the REPAIR pool (the shared driver's draw function fx.draws.calibration if the driver imports, else the
library's seeded split rescope.splits.sample: the output says which), alpha = 0.05, n = 200, and for BOTH calibrations of the library
(calibrate_two_block with split seed 0; calibrate_shared), every variant is built, given the guards of every gate (stored solver keys +
the B = 1 keys of the edits' own Rel prompts), calibrated on the teacher-forced keys of the 200 queries, and decided on the log's cached
sequences by rescope.KeyCache.decide:
    threshold summary (eta, thresholds, capped edits), false firing on the valid HELD-OUT-DISJOINT probe rows, coverage of text / image
    rephrases (the own edit releases on the T-Gen / I-Gen sequence), firing on the valid hard negatives HN-I / HN-Q / HN-X.
Rows: the registered default method (normalised solver coordinate) and the registered cosine gate, computed the same way, then
cos_centred, cos_null (the 100 keys of rescope_null in draw 0), white(kappa), whitecos(kappa), the registered rescope_null and its
equal-data companion rescope_nullW (white.scores.null_w).
Checks against the sanity experiment (e0_sanity): the draw, the thresholds and the per-row decisions of the registered rows against
e0's per-row file for draw 0; eta and capped edits of draw 0 against e0's table.txt; and the 20-draw cells of the registered rows of
table.txt (re-computed here for the registered rows only).
THIS IS A MECHANICS CHECK, NOT A RESULT: nothing is selected or tuned from it.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

import numpy as np

from . import common as C
from . import scores as WS
from . import wstats

torch = C.use_library()

ALPHA, N_CAL, DRAW, N_NULL, N_DRAWS_E0 = 0.05, 200, 0, 100, 20
CALS = ("two_block", "shared")
REGISTERED = ("rescope", "cosine")
E0 = C.FINAL + "/e0_sanity/out"
E0_ID = {("rescope", "two_block"): "M1", ("rescope", "shared"): "M2", ("cosine", "two_block"): "M3", ("cosine", "shared"): "M4"}
E0_LABEL = {"M1": "M1 ncoord two-block (default)", "M2": "M2 ncoord shared scale", "M3": "M3 cosine two-block", "M4": "M4 cosine shared scale"}
HN = (("HN-I", "hn_i"), ("HN-Q", "hn_q"), ("HN-X", "hn_x"))


def variant_list() -> list:
    """[(row label, what it is)] in table order."""
    out = [("rescope", "registered default method: normalised solver coordinate <W_i, k> / <W_i, k_i>"),
           ("cosine", "registered cosine gate: raw cosine to the stored solver key")]
    what = {"cos_centred": "cos(k - mu, k_i - mu), mu = mean BEN-W key",
            "cos_null": "cosine detector after the library's null repair (ridge 0) on the 100 keys of rescope_null (draw 0)",
            "white": "k_i' M k / k_i' M k_i, M = (C_W + rho I)^-1, rho = kappa tr(C_W) / d", "whitecos": "cosine in the whitened space"}
    out += [(lab, what[feat] + (f", kappa = {wstats.kappa_tag(k)}" if k is not None else "")) for lab, feat, k in WS.variants()]
    out += [("rescope_null", "registered variant: default detectors after null repair on 100 REPAIR keys (teacher-forced), disjoint from the draw"),
            ("rescope_nullW", "default detectors after null repair on ALL 600 BEN-W keys (prompt-only keys)")]
    return out


def pct(flags):
    """Percentage of true flags, computed as the sanity experiment does (100.0 * count / n: the same float, so that formatted cells agree)."""
    flags = [bool(x) for x in np.asarray(flags).tolist()]
    return 100.0 * sum(flags) / len(flags) if flags else None


def bits(v) -> str:
    return "".join("1" if bool(x) else "0" for x in v)


def unbits(s: str):
    return np.frombuffer(s.encode(), dtype=np.uint8) == ord("1")


def local_ranking(n_pool: int, draw: int) -> np.ndarray:
    """Copy of the shared driver's ranking rule (fx.draws.ranking), used only if the driver does not import: the 200 rows of
    rescope.splits.sample(pool, 200, draw) in a seeded order, then the other rows in a seeded order."""
    from rescope import splits
    base = np.array(splits.sample(list(range(n_pool)), min(N_CAL, n_pool), int(draw)), dtype=np.int64)
    rest = np.setdiff1d(np.arange(n_pool), base)
    return np.concatenate([base[np.random.default_rng([int(draw), 1]).permutation(base.size)],
                           rest[np.random.default_rng([int(draw), 2]).permutation(rest.size)]])


class Context:
    """One development log without a model: the edit log, the key cache, the row sets, the draw."""

    def __init__(self, name: str, driver: bool = True):
        from rescope import KeyCache, from_testbed, splits
        if name not in C.DEV_LOGS.values():
            raise SystemExit(f"{name}: the mechanics check touches only the two development logs {sorted(C.DEV_LOGS.values())}")
        self.name = name
        self.backbone = {v: k for k, v in C.DEV_LOGS.items()}[name]
        self.log = from_testbed(name)
        self.kc = KeyCache(self.backbone)
        self.uids, self.T = self.log.meta["uids"], self.log.T
        kc, u = self.kc, self.uids
        self.sets = {"repair": kc.log_rows(u, kind="probe", pool="repair", valid=True),
                     "heldout_disjoint": kc.log_rows(u, kind="probe", pool="heldout_disjoint", valid=True),
                     "rel": kc.log_rows(u, kind="rel"), "tgen": kc.log_rows(u, kind="tgen"), "igen": kc.log_rows(u, kind="igen")}
        for _, k in HN:
            self.sets[k] = kc.log_rows(u, kind=k, valid=True)
        self.eval_names = ("heldout_disjoint", "rel", "tgen", "igen") + tuple(k for _, k in HN)
        self.rows = [j for nm in self.eval_names for j in self.sets[nm]]
        a = 0
        self.span = {}
        for nm in self.eval_names:
            self.span[nm] = (a, a + len(self.sets[nm]))
            a += len(self.sets[nm])
        self.own = kc.own_keys(u)                                        # B = 1 keys of the edits' own Rel prompts (teacher-forced)
        self.stats = wstats.load(self.backbone)
        lib_draw = lambda d: splits.sample(self.sets["repair"], N_CAL, d)          # noqa: E731
        self.fx, self.L = None, None
        if driver:
            try:
                self.fx = C.use_driver()
                self.L = self.fx.logs.open(name)
            except Exception as e:                                          # the driver is under review: fall back to the library
                self.fx, self.driver_error = None, f"{type(e).__name__}: {e}"
        if self.fx is not None:
            if self.L.pool("repair") != self.sets["repair"] or self.L.pool("heldout_disjoint") != self.sets["heldout_disjoint"]:
                raise RuntimeError("the driver's pools are not the library's pools")
            self.draw_rows = lambda d: list(self.fx.draws.calibration(self.L, N_CAL, d).rows)          # noqa: E731
            self.null_rows = list(self.fx.draws.null_sample(self.L, N_NULL, DRAW))
            self.draw_source = "the shared driver: fx.draws.calibration(L, 200, draw) and fx.draws.null_sample(L, 100, draw)"
            self.draw_equals_library = self.draw_rows(DRAW) == lib_draw(DRAW)
            rk = local_ranking(len(self.sets["repair"]), DRAW)
            self.null_equals_local_rule = self.null_rows == [self.sets["repair"][int(k)] for k in np.sort(rk[N_CAL:N_CAL + N_NULL])]
        else:
            self.draw_rows = lib_draw
            rk = local_ranking(len(self.sets["repair"]), DRAW)
            self.null_rows = [self.sets["repair"][int(k)] for k in np.sort(rk[N_CAL:N_CAL + N_NULL])]
            self.draw_source = ("the library's seeded split rescope.splits.sample(valid REPAIR rows, 200, draw) (the driver did not import: "
                                f"{self.driver_error}); null sample by a local copy of the driver's ranking rule")
            self.draw_equals_library, self.null_equals_local_rule = True, True
        if set(self.null_rows) & set(self.draw_rows(DRAW)):
            raise RuntimeError("the null sample must be disjoint from the calibration draw")
        self.null_keys = kc.keys("tf", self.null_rows)                    # the 100 keys of rescope_null in draw 0 (teacher-forced)
        used = set(self.draw_rows(DRAW)) | set(self.null_rows)
        self.sets["repair_rest"] = [j for j in self.sets["repair"] if j not in used]      # exchangeable with the calibration rows by construction
        self.span["repair_rest"] = (len(self.rows), len(self.rows) + len(self.sets["repair_rest"]))
        self.eval_names = self.eval_names + ("repair_rest",)
        self.rows = self.rows + self.sets["repair_rest"]

    def base(self, variant: str):
        """(un-thresholded gate with all guards, record) of a row."""
        from rescope import add_guards, feature_gate, null_repair
        rec = {}
        if variant in ("rescope", "rescope_null", "rescope_nullW"):
            S, _ = add_guards(feature_gate(self.log, "ncoord"), *self.own)
            if variant == "rescope_null":                                  # as fx.methods.scoped: guards, then the library's null repair
                S0 = S
                S, r = null_repair(S0, self.null_keys, ridge=0.0)
                infl = S.det_w.norm(dim=1) / S0.det_w.norm(dim=1)
                rec["null"] = dict(_null_public(r), detector_norm_inflation={"median": float(infl.median()), "max": float(infl.max())})
            elif variant == "rescope_nullW":
                S, r = WS.null_w(S, stats=self.stats)
                rec["null"] = _null_public(r)
        elif variant == "cosine":
            S, _ = add_guards(feature_gate(self.log, "cos"), *self.own)
        else:
            feat, kappa = _parse(variant)
            S = WS.feature_gate(self.log, feat, stats=self.stats, kappa=kappa, null_keys=self.null_keys)
            w = S.meta["white"]
            rec["white"] = {k: v for k, v in w.items() if k != "null_repair"}
            if "null_repair" in w:
                rec["null"] = dict(_null_public(w["null_repair"]), detector_norm_inflation=w["detector_norm_inflation"])
            S, _ = add_guards(S, *self.own)
        Rg = S.guard_scores()
        ar = torch.arange(self.T)
        stored, b1 = Rg[:self.T][ar, ar], Rg[self.T:][ar, ar]
        rec.update(gate_class=type(S).__name__, n_guards=int(S.guard_key.shape[0]),
                   own_stored_key_max_abs_dev_from_1=float((stored - 1).abs().max()),
                   own_b1_rel_key_min_median_max=[float(b1.min()), float(b1.median()), float(b1.max())])
        return S, rec

    def calibrate(self, S, cal: str, draw: int, allow_silence: bool = False):
        """The library's calibration on the teacher-forced keys of the draw.  allow_silence=True: the literal (uncapped) rule, the
        event the conformal bound covers."""
        from rescope import calibrate_shared, calibrate_two_block
        Q = self.kc.coords(S, self.draw_rows(draw), "tf")
        if cal == "two_block":
            return calibrate_two_block(S, coords=Q, alpha=ALPHA, seed=int(draw), allow_silence=allow_silence)
        return calibrate_shared(S, coords=Q, alpha=ALPHA, allow_silence=allow_silence)

    def fire_disjoint(self, G) -> float:
        """% of the valid HELD-OUT-DISJOINT rows on which an edit fires."""
        return pct(self.kc.decide(G, "tf", rows=self.sets["heldout_disjoint"], uids=self.uids)["fired"])

    def read_out(self, G) -> dict:
        d = self.kc.decide(G, "tf", rows=self.rows, uids=self.uids)
        sl = lambda nm, k: np.asarray(d[k])[self.span[nm][0]:self.span[nm][1]]          # noqa: E731
        out = {"n": {nm: len(self.sets[nm]) for nm in self.eval_names},
               "fire_heldout_disjoint": pct(sl("heldout_disjoint", "fired")), "fire_repair_rest": pct(sl("repair_rest", "fired")),
               "rel_own_top": pct(sl("rel", "own_top")),
               "tgen_own_top": pct(sl("tgen", "own_top")), "tgen_own_cand": pct(sl("tgen", "own_cand")), "tgen_other_top": pct(sl("tgen", "fired") & ~sl("tgen", "own_top")),
               "igen_own_top": pct(sl("igen", "own_top")), "igen_own_cand": pct(sl("igen", "own_cand")), "igen_other_top": pct(sl("igen", "fired") & ~sl("igen", "own_top")),
               "hn_fired": {t: pct(sl(k, "fired")) for t, k in HN}, "hn_own_top": {t: pct(sl(k, "own_top")) for t, k in HN},
               "n_near_threshold": int(np.asarray(d["near"]).sum()),
               "flags": {"fired": bits(d["fired"]), "own_top": bits(d["own_top"])}}
        return out


def _parse(variant: str):
    for lab, feat, kappa in WS.variants():
        if lab == variant:
            return feat, kappa
    raise KeyError(variant)


def _null_public(r: dict) -> dict:
    keep = ("n_keys", "rank_benign", "ridge", "n_detectors", "infeasible", "rel_change_median", "rel_change_max", "max_abs_benign_r", "max_own_rel_err",
            "n_guards_lost", "n_a", "keys", "sample", "detector_norm_inflation")
    return {k: r[k] for k in keep if k in r}


def _cal_public(rec: dict, G) -> dict:
    th = G.theta().numpy()
    fin = th[np.isfinite(th)]
    return {"op": rec["op"], "n1": rec["n1"], "n2": rec["n2"], "rank": rec["rank"], "eta": rec["eta"], "bound_false_firing": rec["bound_false_firing"],
            "top1_rank": rec["top1_rank"], "n_edits_capped": rec["n_edits_capped"], "capped_edits": rec["capped_edits"],
            "n_guards_silenced_by_literal_rule": rec["n_guards_silenced_by_literal_rule"],
            "n_own_keys_silenced_by_literal_rule": rec["n_own_keys_silenced_by_literal_rule"], "n_guards_silenced": rec["n_guards_silenced"],
            "n_guards_released_by_own_edit_top1": rec["n_guards_released_by_own_edit_top1"], "guarantee": rec["guarantee"],
            "threshold_min_median_max": [float(fin.min()), float(np.median(fin)), float(fin.max())] if fin.size else None,
            "n_thresholds_infinite": int(np.isposinf(th).sum()),
            "baseline_min_median_max": rec.get("baseline_min_median_max"), "tau": [float(x) for x in th]}


def over_draws(values) -> dict:
    """mean, 5th and 95th percentile (numpy, linear), as e0_common.over_draws."""
    v = np.asarray(values, dtype=np.float64)
    return {"mean": float(v.mean()), "p05": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95))}


def run(name: str, driver: bool = True) -> dict:
    t0 = time.time()
    ctx = Context(name, driver)
    out = {"log": name, "backbone": ctx.backbone, "T": ctx.T, "alpha": ALPHA, "n_cal": N_CAL, "draw": DRAW, "n_null": N_NULL,
           "draw_source": ctx.draw_source, "driver_imported": ctx.fx is not None,
           "driver_draw_equals_library_split": ctx.draw_equals_library, "driver_null_sample_equals_local_ranking_rule": ctx.null_equals_local_rule,
           "calibration_sids": [ctx.kc.index[j]["lib_sid"] for j in ctx.draw_rows(DRAW)], "null_sids": [ctx.kc.index[j]["lib_sid"] for j in ctx.null_rows],
           "row_sids": [ctx.kc.index[j]["lib_sid"] for j in ctx.rows], "row_spans": {k: list(v) for k, v in ctx.span.items()},
           "n_rows": {k: len(v) for k, v in ctx.sets.items()},
           "stats": {"n": ctx.stats.n, "d": ctx.stats.d, "trace": ctx.stats.trace, "rho": {wstats.kappa_tag(k): ctx.stats.rho(k) for k in C.KAPPAS},
                     "stats_pt_md5": ctx.stats.meta.get("stats_pt_md5")},
           "code": code_fingerprint(), "variants": {}, "what": dict(variant_list())}
    gates = {}
    for variant, _ in variant_list():
        t = time.time()
        S, rec = ctx.base(variant)
        e = {"build": rec, "cal": {}}
        for cal in CALS:
            G, crec = ctx.calibrate(S, cal, DRAW)
            e["cal"][cal] = {"calibration": _cal_public(crec, G), "read_out": ctx.read_out(G)}
            Gu, _ = ctx.calibrate(S, cal, DRAW, allow_silence=True)     # the literal rule (no guard cap): what the bound is about
            e["cal"][cal]["read_out"]["fire_heldout_disjoint_uncapped"] = ctx.fire_disjoint(Gu)
            gates[(variant, cal)] = G
        out["variants"][variant] = e
        r = e["cal"]["two_block"]
        print(f"[{time.strftime('%H:%M:%S')}] {name} {variant:14s} two-block: eta {r['calibration']['eta']:.4f}, capped {r['calibration']['n_edits_capped']:3d}, "
              f"heldout_disjoint {r['read_out']['fire_heldout_disjoint']:.2f} %, T-Gen {r['read_out']['tgen_own_top']:.0f} %, I-Gen {r['read_out']['igen_own_top']:.0f} % "
              f"({time.time() - t:.1f}s)", flush=True)
    out["null_sample"] = null_sample_overlap(ctx)
    out["checks"] = checks(ctx, out, gates)
    print(f"[{time.strftime('%H:%M:%S')}] {name}: {time.time() - t0:.0f}s", flush=True)      # timings are printed, not stored: the files are reproducible
    return out


def null_sample_overlap(ctx: Context) -> dict:
    """The null sample of rescope_null / cos_null (100 REPAIR rows, disjoint from the calibration draw AS ROWS) against the other row
    sets AS QUERIES (query_id of the key cache: same image content and question string): rows whose query is a null-sample query have
    bitwise the null key, so every null-repaired detector scores them 0.  Counts are label-free; the scores of these rows after the
    repair are read from the three null-repaired gates (development log)."""
    from rescope import split_blocks
    ix, kc = ctx.kc.index, ctx.kc
    cal, nul = ctx.draw_rows(DRAW), ctx.null_rows
    nq = {ix[j]["query_id"] for j in nul}
    ns = {C.norm_q(ix[j]["prompt"]) for j in nul}
    twin = np.array([ix[j]["query_id"] in nq for j in cal])
    i1, i2 = split_blocks(N_CAL, (1, 1), DRAW)
    held = kc.log_rows(ctx.uids, kind="probe", pool="heldout", valid=True)
    key_of = {}
    for j in nul:
        key_of.setdefault(ix[j]["query_id"], kc.keys("tf", [j])[0])
    same_key = sum(bool(torch.equal(kc.keys("tf", [j])[0], key_of[ix[j]["query_id"]])) for j, t in zip(cal, twin) if t)
    cnt = lambda rows: {"n": len(rows), "same_query": sum(ix[j]["query_id"] in nq for j in rows),            # noqa: E731
                        "same_question_string": sum(C.norm_q(ix[j]["prompt"]) in ns for j in rows)}
    q1 = {ix[cal[int(i)]]["query_id"] for i in i1}                        # the same phenomenon inside the two-block rule, for EVERY two-block row:
    b2_twin = [int(i) for i in i2 if ix[cal[int(i)]]["query_id"] in q1]   # a block-2 query identical to a block-1 query has margin <= 0 everywhere
    out = {"n_null": len(nul), "n_distinct_queries_in_null_sample": len(nq), "n_distinct_queries_in_calibration_draw": len({ix[j]["query_id"] for j in cal}),
           "block2_queries_identical_to_a_block1_query": len(b2_twin),
           "calibration": dict(cnt(cal), same_query_block1=int(twin[i1].sum()), same_query_block2=int(twin[i2].sum()), same_query_key_bitwise_equal=int(same_key)),
           "repair_rest": cnt(ctx.sets["repair_rest"]), "heldout": cnt(held), "heldout_disjoint": cnt(ctx.sets["heldout_disjoint"]), "after_repair": {}}
    per = []
    if ctx.fx is not None:                                               # the same count for the other draws (label-free)
        for d in range(N_DRAWS_E0):
            q = {ix[j]["query_id"] for j in ctx.fx.draws.null_sample(ctx.L, N_NULL, d)}
            per.append(sum(ix[j]["query_id"] in q for j in ctx.draw_rows(d)))
        out["calibration_same_query_draws_0_19"] = {"values": per, "min": int(min(per)), "median": float(np.median(per)), "max": int(max(per))}
    for variant in ("rescope", "rescope_null", "cos_null", "rescope_nullW"):
        S, _ = ctx.base(variant)
        Q = kc.coords(S, cal, "tf").numpy()
        G, rec = ctx.calibrate(S, "two_block", DRAW, allow_silence=True)
        s2 = (Q[i2] - G.det_b.numpy()[None, :]).max(1)
        rank = np.argsort(np.argsort(s2, kind="stable"), kind="stable") + 1          # rank among the block-2 scores (1 = smallest)
        out["after_repair"][variant] = {"max_abs_score_of_same_query_calibration_rows": float(np.abs(Q[twin]).max()) if twin.any() else None,
                                        "block2_ranks_of_same_query_rows": sorted(int(x) for x in rank[twin[i2]]), "rank_of_eta": rec["rank"], "n2": rec["n2"]}
    return out


def code_fingerprint() -> dict:
    import hashlib

    def fp(folder):
        h = hashlib.sha1()
        for f in sorted(os.listdir(folder)):
            if f.endswith(".py"):
                h.update(f.encode())
                with open(os.path.join(folder, f), "rb") as fh:
                    h.update(fh.read())
        return h.hexdigest()[:12]
    out = {"rescope": fp(os.path.join(C.LIB, "rescope")), "white": fp(C.HERE)}
    if os.path.isdir(os.path.join(C.DRIVER, "fx")):
        out["fx"] = fp(os.path.join(C.DRIVER, "fx"))
    return out


# ----------------------------------------------------------------------------- checks against the sanity experiment and the driver
def checks(ctx: Context, out: dict, gates: dict) -> dict:
    res = {}
    # (a) e0's per-row file, draw 0: the draw, the thresholds, the decisions of the registered rows
    p = os.path.join(E0, f"gate_{ctx.name}.json")
    if os.path.exists(p):
        with open(p) as f:
            g = json.load(f)
        pos = {r["lib_sid"]: i for i, r in enumerate(g["rows"])}
        mine = out["row_sids"]
        common = [i for i, s in enumerate(mine) if s in pos]
        e0_idx = np.array([pos[mine[i]] for i in common], dtype=np.int64)
        res["e0_rows"] = {"file": os.path.relpath(p, C.FINAL), "n_rows_here": len(mine), "n_rows_also_in_e0": len(common), "e0_rescope_fingerprint": g["code"]["rescope"],
                          "rescope_fingerprint_now": out["code"]["rescope"],
                          "calibration_sids_equal": g["draws"][DRAW]["calibration_sids"] == out["calibration_sids"], "methods": {}}
        for (variant, cal), mid in E0_ID.items():
            e = g["methods"][mid]["draws"][DRAW]
            m = out["variants"][variant]["cal"][cal]
            tau0 = np.array([float(x) for x in e["tau"]], dtype=np.float64)
            tau1 = np.array(m["calibration"]["tau"], dtype=np.float64)
            f0, f1 = unbits(e["tf"]["fired"])[e0_idx], unbits(m["read_out"]["flags"]["fired"])[common]
            o0, o1 = unbits(e["tf"]["own_top"])[e0_idx], unbits(m["read_out"]["flags"]["own_top"])[common]
            res["e0_rows"]["methods"][mid] = {"variant": variant, "cal": cal, "eta_e0": e["rec"]["eta"], "eta_here": m["calibration"]["eta"],
                                              "eta_equal": float(e["rec"]["eta"]) == float(m["calibration"]["eta"]),
                                              "thresholds_bitwise_equal": bool(np.array_equal(tau0, tau1)), "max_abs_threshold_difference": float(np.abs(tau0 - tau1).max()),
                                              "n_capped_e0": e["rec"]["n_edits_capped"], "n_capped_here": m["calibration"]["n_edits_capped"],
                                              "n_rows_fired_differs": int((f0 != f1).sum()), "n_rows_own_top_differs": int((o0 != o1).sum())}
    else:
        res["e0_rows"] = {"file": p, "missing": True}
    # (b) e0's table.txt: eta / capped edits of draw 0 (its GPU-built gates), and the 20-draw gate-level cells of the registered rows
    tp = os.path.join(E0, "table.txt")
    if os.path.exists(tp):
        with open(tp) as f:
            txt = f.read()
        a = txt.index(f"\n{ctx.name}  (")
        nxt = [txt.index(f"\n{n}  (") for n in C.DEV_LOGS.values() if n != ctx.name and txt.index(f"\n{n}  (") > a]
        sec = txt[a:min(nxt)] if nxt else txt[a:]
        line = next(ln for ln in sec.splitlines() if "gates built on the GPU:" in ln)
        d0 = {}
        for mid in ("M1", "M2", "M3", "M4"):
            mm = re.search(mid + r": draw 0 eta ([0-9.]+|inf) capped (\d+)", line)
            variant, cal = next(k for k, v in E0_ID.items() if v == mid)
            c = out["variants"][variant]["cal"][cal]["calibration"]
            d0[mid] = {"eta_table": mm.group(1), "eta_here": f"{c['eta']:.4f}", "capped_table": int(mm.group(2)), "capped_here": c["n_edits_capped"],
                       "equal": mm.group(1) == f"{c['eta']:.4f}" and int(mm.group(2)) == c["n_edits_capped"]}
        res["e0_table_draw0"] = {"file": os.path.relpath(tp, C.FINAL), "what": "eta (4 decimals) and capped edits of draw 0, e0's gates built on the GPU", "methods": d0}
        # the 20 draws of the registered rows (already reported for the development logs in table.txt): cells mean [5th, 95th percentile]
        cells = {}
        for (variant, cal), mid in E0_ID.items():
            S, _ = ctx.base(variant)
            per = []
            for dd in range(N_DRAWS_E0):
                G, _ = ctx.calibrate(S, cal, dd)
                per.append(ctx.read_out(G))
            fmt = lambda v, nd: "{:.{nd}f} [{:.{nd}f}, {:.{nd}f}]".format(v["mean"], v["p05"], v["p95"], nd=nd)          # noqa: E731
            mine = {"fire % (as run)": fmt(over_draws([r["fire_heldout_disjoint"] for r in per]), 2),
                    "T-Gen own fires": fmt(over_draws([r["tgen_own_top"] for r in per]), 1), "T-Gen own cand": fmt(over_draws([r["tgen_own_cand"] for r in per]), 1),
                    "I-Gen own fires": fmt(over_draws([r["igen_own_top"] for r in per]), 1), "I-Gen own cand": fmt(over_draws([r["igen_own_cand"] for r in per]), 1),
                    "HN-I an edit": fmt(over_draws([r["hn_fired"]["HN-I"] for r in per]), 1), "HN-I own": fmt(over_draws([r["hn_own_top"]["HN-I"] for r in per]), 1),
                    "HN-Q an edit": fmt(over_draws([r["hn_fired"]["HN-Q"] for r in per]), 1), "HN-Q own": fmt(over_draws([r["hn_own_top"]["HN-Q"] for r in per]), 1),
                    "HN-X an edit": fmt(over_draws([r["hn_fired"]["HN-X"] for r in per]), 1), "HN-X own a": fmt(over_draws([r["hn_own_top"]["HN-X"] for r in per]), 1)}
            tab = _table_cells(sec, E0_LABEL[mid])
            cells[mid] = {k: {"here": v, "table": tab.get(k), "equal": tab.get(k) == v} for k, v in mine.items()}
        res["e0_table_20_draws"] = {"what": "gate-level cells of table.txt section (1), mean [5th, 95th percentile] over the 20 draws, registered rows only",
                                    "methods": cells, "n_cells": sum(len(c) for c in cells.values()),
                                    "n_equal": sum(v["equal"] for c in cells.values() for v in c.values())}
    else:
        res["e0_table_draw0"] = {"file": tp, "missing": True}
    # (c) the shared driver's own rows of the registered methods: the same thresholds bit for bit
    if ctx.fx is not None:
        drv = {}
        for lab, variant, cal, kw in (("rescope", "rescope", "two_block", {}), ("rescope_shared", "rescope", "shared", {}), ("cosine", "cosine", "two_block", {}),
                                      ("cosine_shared", "cosine", "shared", {"calibration": "shared"}), ("rescope_null", "rescope_null", "two_block", {})):
            try:
                name = "cosine" if lab.startswith("cosine") else lab
                M = ctx.fx.methods.build(name, ctx.L, alpha=ALPHA, n=N_CAL, draw=DRAW, **kw)
                th = M.obj.theta().numpy()
                mine = np.array(out["variants"][variant]["cal"][cal]["calibration"]["tau"], dtype=np.float64)
                drv[lab] = {"thresholds_bitwise_equal": bool(np.array_equal(th, mine)), "max_abs_threshold_difference": float(np.abs(th - mine).max())}
            except Exception as e:                                          # the driver is under review
                drv[lab] = {"error": f"{type(e).__name__}: {e}"}
        res["driver_rows"] = {"what": "fx.methods.build(<registered row>, L, alpha = 0.05, n = 200, draw = 0): thresholds against the rows computed here", "rows": drv}
        # the new gates through the driver's gate-level path: fx.gate.decide == rescope.KeyCache.decide
        gl = {}
        for (variant, cal), G in gates.items():
            if variant in REGISTERED or cal != "two_block":
                continue
            try:
                a = ctx.fx.gate.decide(ctx.L, G, ctx.rows, "tf")
                b = ctx.kc.decide(G, "tf", rows=ctx.rows, uids=ctx.uids)
                gl[variant] = {k: bool(np.array_equal(np.asarray(a[k]), np.asarray(b[k]))) for k in ("fired", "top_edit", "own_cand", "own_top", "near")}
            except Exception as e:
                gl[variant] = {"error": f"{type(e).__name__}: {e}"}
        res["driver_gate_level"] = {"what": "fx.gate.decide(L, gate, rows) == rescope.KeyCache.decide(gate, rows) for the new gates (two-block), per array", "rows": gl}
    return res


def _table_cells(section: str, label: str) -> dict:
    """Cells of the rows of e0's table.txt that start with `label` (first table of section (1): firing / coverage; second: hard negatives)."""
    lines = section.splitlines()
    out, seen = {}, 0
    for i, ln in enumerate(lines):
        if not ln.startswith("    " + label):
            continue
        seen += 1
        if seen > 2:
            break
        k = i
        while not lines[k].startswith("    method"):
            k -= 1
        head = re.split(r"\s{2,}", lines[k].strip())
        cells = re.split(r"\s{2,}", ln.strip())
        if len(head) == len(cells):
            for h, c in zip(head[1:], cells[1:]):
                out.setdefault(h, c)
    return out


# ----------------------------------------------------------------------------- rendering
def _f(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


def _tab(head, rows, left=1):
    w = [max(len(str(r[i])) for r in [head] + rows) for i in range(len(head))]
    line = lambda r: "  ".join(str(c).ljust(w[i]) if i < left else str(c).rjust(w[i]) for i, c in enumerate(r)).rstrip()          # noqa: E731
    return ["    " + line(head), "    " + "  ".join("-" * x for x in w)] + ["    " + line(r) for r in rows]


def render(paths=None) -> str:
    paths = paths or [os.path.join(C.OUT, f"mechanics_{n}.json") for n in C.DEV_LOGS.values()]
    res, missing = [], []
    for p in paths:
        if not os.path.exists(p):                                         # a log that has not been run yet is named, not silently dropped
            missing.append(p)
            continue
        with open(p) as f:
            res.append(_unclean(json.load(f)))
    L = ["MECHANICS CHECK of the gates of PROTOCOL.md section 6, amendment 'Review round 1', item R3      (rendered by feat2/white/mechanics.py from out/mechanics_<log>.json;",
         "no number is typed by hand)",
         "THIS TABLE IS A MECHANICS CHECK, NOT A RESULT: one calibration draw, development logs only; nothing is selected or tuned from it.",
         f"alpha = {ALPHA}; n = {N_CAL} calibration queries = calibration draw {DRAW} of the valid REPAIR probe rows; teacher-forced keys; every gate decides at the last",
         "prompt position (role 'last'), top-1, guards = stored solver keys + B = 1 keys of the edits' own Rel prompts (capped thresholds; a capped edit is outside the",
         "conformal bound).  Calibration through the library: calibrate_two_block (split seed = draw) and calibrate_shared; decisions by rescope.KeyCache.decide.",
         "fire = % of the valid HELD-OUT-DISJOINT probe rows on which an edit fires (false firing; 'uncapped' = under the literal rule without the guard cap, the event",
         "the bound covers -- marginally over calibration draws: ONE draw can exceed it); T-Gen / I-Gen = % of the text / image rephrase sequences on which their OWN",
         "edit releases (coverage; 'cand' = is a candidate); HN-I / HN-Q / HN-X = % of the valid hard negatives on which AN edit fires.",
         "rest = % of the valid REPAIR rows that are neither calibration queries nor in the null sample on which an edit fires: the only rows that are exchangeable with",
         "the calibration queries by construction (same pool, same seeded ranking; about 190 rows).",
         "capped = edits whose threshold was lowered to keep a guard firing; own B=1 = score of each edit's own B = 1 Rel key on its own detector (min / median).",
         ""]
    for r in res:
        n = r["n_rows"]
        L += ["#" * 150, f"{r['log']}  ({r['backbone']}, T = {r['T']} edits)", "#" * 150,
              f"draw: {r['draw_source']}",
              f"      driver's draw 0 == rescope.splits.sample(valid REPAIR rows, 200, 0): {r['driver_draw_equals_library_split']}; "
              f"null sample (rescope_null, cos_null) == positions 200 .. 299 of the draw's ranking: {r['driver_null_sample_equals_local_ranking_rule']}",
              f"rows: REPAIR {n['repair']} valid (calibration pool; {n['repair_rest']} of them outside the calibration draw and the null sample), "
              f"HELD-OUT-DISJOINT {n['heldout_disjoint']} valid, T-Gen {n['tgen']}, I-Gen {n['igen']}, "
              f"hard negatives HN-I {n['hn_i']} / HN-Q {n['hn_q']} / HN-X {n['hn_x']} (valid rows)",
              f"BEN-W statistics: N = {r['stats']['n']}, d = {r['stats']['d']}, tr(C_W) = {r['stats']['trace']:.4f}; rho = "
              + ", ".join(f"{v:.4e} (kappa {k})" for k, v in r["stats"]["rho"].items()),
              f"code: rescope {r['code']['rescope']}, white {r['code']['white']}" + (f", fx {r['code']['fx']}" if "fx" in r["code"] else ""), ""]
        for cal in CALS:
            L.append(f"  calibration: {'two-block (calibrate_two_block; the default calibration)' if cal == 'two_block' else 'shared scale (calibrate_shared)'}")
            rows = []
            for v, e in r["variants"].items():
                c, o, b = e["cal"][cal]["calibration"], e["cal"][cal]["read_out"], e["build"]
                th = c["threshold_min_median_max"]
                rows.append([v + (" *" if v in REGISTERED or v == "rescope_null" else ""), _f(c["eta"], 4), "-" if th is None else f"{th[0]:.3f} / {th[1]:.3f} / {th[2]:.3f}",
                             str(c["n_edits_capped"]), f"{b['own_b1_rel_key_min_median_max'][0]:.3f} / {b['own_b1_rel_key_min_median_max'][1]:.3f}",
                             _f(o["fire_heldout_disjoint"], 2), _f(o["fire_heldout_disjoint_uncapped"], 2), _f(o["fire_repair_rest"], 2), _f(o["tgen_own_top"]),
                             _f(o["tgen_own_cand"]), _f(o["igen_own_top"]), _f(o["igen_own_cand"]), _f(o["hn_fired"]["HN-I"]), _f(o["hn_fired"]["HN-Q"]),
                             _f(o["hn_fired"]["HN-X"]), _f(o["rel_own_top"], 0)])
            L += _tab(["variant", "eta", "thresholds min / median / max", "capped", "own B=1 min / med", "fire %", "uncapped", "rest", "T-Gen", "(cand)", "I-Gen", "(cand)",
                       "HN-I", "HN-Q", "HN-X", "Rel own"], rows)
            L.append(f"    bound of the uncapped rule: {100 * r['variants']['rescope']['cal'][cal]['calibration']['bound_false_firing']:.2f} %; * = registered before this amendment")
            L.append("")
        L.append("  detectors (label-free):")
        rows = []
        for v, e in r["variants"].items():
            b = e["build"]
            nul, wh = b.get("null"), b.get("white", {})
            rows.append([v, b["gate_class"], f"{b['own_stored_key_max_abs_dev_from_1']:.1e}",
                         "-" if nul is None else f"{nul['n_keys']} (rank {nul['rank_benign']})", "-" if nul is None else f"{nul['max_abs_benign_r']:.1e}",
                         "-" if nul is None else f"{nul['detector_norm_inflation']['median']:.3f} / {nul['detector_norm_inflation']['max']:.3f}",
                         "-" if nul is None else str(len(nul["infeasible"])),
                         "-" if "rho" not in wh else f"{wh['rho']:.3e}",
                         "-" if "rho_kMk_over_kk_min_median_max" not in wh else " / ".join(f"{x:.3f}" for x in wh["rho_kMk_over_kk_min_median_max"])])
        L += _tab(["variant", "gate class", "|own - 1| max", "null keys", "max |r| on them", "norm inflation med / max", "infeasible", "rho", "rho k_i'Mk_i / |k_i|^2 min / med / max"],
                  rows, left=2)
        L.append("    |own - 1| = deviation of the score at the edit's own stored solver key from 1; null keys (rank) = keys of the null repair and their numerical rank")
        L.append("    (the REPAIR pool repeats some queries: identical keys); max |r| on them = largest score of a null key after the repair; norm inflation = |repaired")
        L.append("    detector| / |detector|;")
        L.append("    rho k'Mk / |k|^2 = share of the solver key's squared norm that survives the whitening (1 = isotropic; small = the key lies in the BEN-W directions).")
        L.append("")
        ns = r["null_sample"]
        c, ar = ns["calibration"], ns["after_repair"]
        L.append(f"  null sample of rescope_null / cos_null (draw {r['draw']}): {ns['n_null']} REPAIR rows with {ns['n_distinct_queries_in_null_sample']} distinct queries; disjoint from the "
                 "calibration draw as ROWS, not as QUERIES:")
        L.append(f"    rows whose query (image content + question) is a null-sample query: calibration draw {c['same_query']} of {c['n']} (block 1: {c['same_query_block1']}, block 2: "
                 f"{c['same_query_block2']}; key bitwise the null key: {c['same_query_key_bitwise_equal']}), REPAIR rest {ns['repair_rest']['same_query']} of {ns['repair_rest']['n']}, "
                 f"all valid HELD-OUT {ns['heldout']['same_query']} of {ns['heldout']['n']}, HELD-OUT-DISJOINT {ns['heldout_disjoint']['same_query']} of {ns['heldout_disjoint']['n']}"
                 + (f"; calibration draws 0 .. 19: min {ns['calibration_same_query_draws_0_19']['min']}, median {ns['calibration_same_query_draws_0_19']['median']:.0f}, "
                    f"max {ns['calibration_same_query_draws_0_19']['max']}" if "calibration_same_query_draws_0_19" in ns else ""))
        L.append(f"    rows whose question STRING occurs in the null sample: calibration draw {c['same_question_string']} of {c['n']}, REPAIR rest "
                 f"{ns['repair_rest']['same_question_string']} of {ns['repair_rest']['n']}, HELD-OUT-DISJOINT {ns['heldout_disjoint']['same_question_string']} of {ns['heldout_disjoint']['n']}")
        L.append("    these calibration rows after the repair (largest |score| on any detector; their ranks among the block-2 scores, eta = rank "
                 f"{ar['rescope']['rank_of_eta']} of {ar['rescope']['n2']}): "
                 + "; ".join(f"{v} {'-' if e['max_abs_score_of_same_query_calibration_rows'] is None else format(e['max_abs_score_of_same_query_calibration_rows'], '.1e')} "
                             f"{e['block2_ranks_of_same_query_rows']}" for v, e in ar.items()))
        L.append(f"    the same inside the two-block rule, for every two-block row (registered ones included): the draw holds {ns['n_distinct_queries_in_calibration_draw']} distinct "
                 f"queries; block-2 queries identical to a block-1 query (margin <= 0 on every detector): {ns['block2_queries_identical_to_a_block1_query']} of {ar['rescope']['n2']}")
        L.append("    A HELD-OUT-DISJOINT row is by definition not a REPAIR query, so neither can happen to a reported row: the calibration queries are not exactly exchangeable")
        L.append("    with the reported rows, and rescope_null / cos_null add the null-sample rows to that (rescope_nullW adds none: no BEN-W query is a probe query of this")
        L.append("    family).  Effect on the firing rate over draws: NOT measured here.")
        L.append("")
        ck = r["checks"]
        L.append("  checks against the sanity experiment (e0_sanity) and the shared driver:")
        er = ck.get("e0_rows", {})
        if er.get("missing"):
            L.append(f"    e0 per-row file missing: {er['file']}")
        else:
            L.append(f"    {er['file']} (draw 0; {er['n_rows_also_in_e0']} of the {er['n_rows_here']} rows read here are rows of e0; library fingerprint then {er['e0_rescope_fingerprint']}, "
                     f"now {er['rescope_fingerprint_now']}): calibration queries equal: {er['calibration_sids_equal']}")
            rows = [[f"{mid} = {m['variant']} / {m['cal']}", f"{m['eta_e0']:.6f}", f"{m['eta_here']:.6f}", str(m["thresholds_bitwise_equal"]), f"{m['max_abs_threshold_difference']:.1e}",
                     f"{m['n_capped_e0']} / {m['n_capped_here']}", str(m["n_rows_fired_differs"]), str(m["n_rows_own_top_differs"])] for mid, m in er["methods"].items()]
            L += ["  " + x for x in _tab(["registered row", "eta e0", "eta here", "thresholds bitwise equal", "max |diff|", "capped e0 / here", "rows with other 'fired'",
                                           "rows with other 'own releases'"], rows)]
        td = ck.get("e0_table_draw0", {})
        if td.get("missing"):
            L.append(f"    e0 table missing: {td['file']}")
        else:
            L.append(f"    {td['file']}, line 'gates built on the GPU', draw 0 (eta, capped edits): "
                     + "; ".join(f"{mid} table {m['eta_table']} / {m['capped_table']}, here {m['eta_here']} / {m['capped_here']}" for mid, m in td["methods"].items())
                     + f" -> all equal: {all(m['equal'] for m in td['methods'].values())}")
            t20 = ck["e0_table_20_draws"]
            L.append(f"    table.txt section (1), 20-draw cells of the registered rows (mean [5th, 95th percentile]; fire, T-Gen / I-Gen own fires and cand, HN-I / HN-Q / HN-X an edit "
                     f"and own): {t20['n_equal']} of {t20['n_cells']} cells reproduced exactly")
            bad = [(mid, k, v) for mid, c in t20["methods"].items() for k, v in c.items() if not v["equal"]]
            for mid, k, v in bad[:12]:
                L.append(f"      DIFFERS {mid} {k}: here {v['here']}, table {v['table']}")
        if "driver_rows" in ck:
            L.append("    shared driver, fx.methods.build(row, alpha = 0.05, n = 200, draw = 0), thresholds bitwise equal to the rows here: "
                     + "; ".join(f"{k}: {v.get('thresholds_bitwise_equal', v.get('error'))}" for k, v in ck["driver_rows"]["rows"].items()))
            gl = ck["driver_gate_level"]["rows"]
            okk = [k for k, v in gl.items() if "error" not in v and all(v.values())]
            L.append(f"    new gates through the driver's gate-level path, fx.gate.decide == KeyCache.decide (fired, top edit, own candidate, own releases, near): "
                     f"{len(okk)} of {len(gl)} gates identical" + ("" if len(okk) == len(gl) else "; NOT identical: " + ", ".join(f"{k} {v}" for k, v in gl.items() if k not in okk)))
        else:
            L.append("    the shared driver did not import: no comparison with its rows")
        L.append("")
    for p in missing:
        L.append(f"NOT RUN YET: {os.path.relpath(p, C.FINAL)} does not exist (python -m white.mechanics run)")
    return "\n".join(L) + "\n"


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="mechanics check on the two development logs")
    ap.add_argument("cmd", choices=("run", "render", "check"))
    ap.add_argument("logs", nargs="*")
    ap.add_argument("--no-driver", action="store_true", help="do not import the shared driver (library split)")
    a = ap.parse_args(argv)
    os.makedirs(C.OUT, exist_ok=True)
    dump = lambda o: json.dumps(_clean(o), indent=1, allow_nan=False) + "\n"          # noqa: E731
    if a.cmd == "check":                                               # a re-run reproduces the stored files byte for byte (same thread setting)
        ok = True
        for name in (a.logs or list(C.DEV_LOGS.values())):
            with open(os.path.join(C.OUT, f"mechanics_{name}.json"), "rb") as f:
                stored = f.read()
            new = _clean(run(name, driver=not a.no_driver))
            same = stored == dump(new).encode("utf-8")
            if same:
                msg = "reproduced byte for byte"
            else:                                                        # the shared driver is under review: was only ITS fingerprint edited?
                old = json.loads(stored.decode("utf-8"))
                fx_old, fx_new = old.get("code", {}).pop("fx", None), new.get("code", {}).pop("fx", None)
                same = old == new and fx_old != fx_new
                msg = (f"reproduced: every number equal; only the fingerprint of the shared driver changed ({fx_old} -> {fx_new}: fx was edited after the stored run)"
                       if same else "DIFFERS from a re-run")
            ok &= same
            print(f"{name}: mechanics_{name}.json {msg}", flush=True)
        with open(os.path.join(C.OUT, "mechanics.txt"), "rb") as f:
            same = f.read() == render().encode("utf-8")
        ok &= same
        print(f"mechanics.txt {'is the rendering of the stored JSON files' if same else 'DIFFERS from the rendering of the stored JSON files'}")
        raise SystemExit(0 if ok else 1)
    if a.cmd == "run":
        for name in (a.logs or list(C.DEV_LOGS.values())):
            out = run(name, driver=not a.no_driver)
            C.write_atomic(os.path.join(C.OUT, f"mechanics_{name}.json"), dump(out))
            print(f"wrote {os.path.join(C.OUT, f'mechanics_{name}.json')}", flush=True)
    txt = render()
    C.write_atomic(os.path.join(C.OUT, "mechanics.txt"), txt)
    print(txt)


def _clean(x):
    """JSON-safe copy: numpy scalars / arrays -> Python; non-finite floats -> the strings 'inf', '-inf', 'nan' (read back by _unclean)."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        x = float(x)
    if isinstance(x, float) and not np.isfinite(x):
        return "nan" if np.isnan(x) else ("inf" if x > 0 else "-inf")
    return x


def _unclean(x):
    if isinstance(x, dict):
        return {k: _unclean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_unclean(v) for v in x]
    if isinstance(x, str) and x in ("inf", "-inf", "nan"):
        return float(x)
    return x


if __name__ == "__main__":
    main()
    sys.exit(0)
