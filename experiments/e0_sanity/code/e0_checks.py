"""E0 checks (CPU).  Every check is written to out/checks.json and rendered in out/table.txt; none of them changes a number.

  python e0_checks.py            all checks that the files on disk allow (the GPU checks need out/behaviour/<log>/)

  A  naive      an independent re-implementation of M1 - M6 at the gate level (plain numpy on the raw files: index.jsonl,
                keys_tf.pt, edit_factors.pt; no rescope code): calibration draw, two-block / shared rule from their
                definitions, guards and cap, top-1; its per-row decisions against the library's (out/gate_<log>.json), all 20 draws.
  B  ladder     the last step of the referee ladder (out/referee.json, matrix path) against the gate-level numbers (KeyCache.decide).
  C  hook       GPU: the firing flags of every evaluated sequence (hook, draws 0 - 2) against the CPU decisions from the key cache;
                thresholds built on the GPU against those built on the CPU; the calibration draws; the row sets.
  D  behaviour  GPU: deployed D1 flags against the testbed's stored D1 complaints; the pre-edit control (D1 = 0, token locality 1,
                edit exact = the cache's pre-edit outcome); silent rows of every gate equal the pre-edit rows; non-finite logits.
  E  seeds      draw d uses seed d for the draw AND for the two-block split: the same rule with an independently seeded split,
                6000 draws (no difference expected).
"""
from __future__ import annotations

import json
import math
import os

import e0_common as C

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(4)
CACHE = C.ROOT + "/02_research/experiments/10_final/cache"
TESTBED = C.ROOT + "/02_research/experiments/06_testbed_v3"
TIE = 1e-9


# ----------------------------------------------------------------------------- A: naive re-implementation
def naive_log(name):
    """Raw files -> row records (in the order of out/gate_<log>.json), feature matrices and guard features."""
    bb = C.LOGS[name]["backbone"]
    fac = torch.load(os.path.join(TESTBED, name, "edit_factors.pt"), map_location="cpu")
    W, Ks, uids = fac["W"].double().numpy(), fac["K1_solver"].double().numpy(), list(fac["uids"])
    T = W.shape[0]
    with open(os.path.join(CACHE, bb, "index.jsonl")) as f:
        index = [json.loads(line) for line in f if line.strip()]
    Ktf = torch.load(os.path.join(CACHE, bb, "keys_tf.pt"), map_location="cpu", mmap=True)
    fam = next(e["family"] for e in index if e["kind"] == "rel" and e["edit"] == 0 and e["edit_uid"] == uids[0])
    mine = [e for e in index if e["family"] == fam]
    assert all(e["edit_uid"] == uids[e["edit"]] for e in mine if e["kind"] == "rel" and e["edit"] < T)
    probes = [e for e in mine if e["kind"] == "probe" and e["y0_valid"]]
    edits = [e for e in mine if e["kind"] in ("rel", "tgen", "igen") and e["edit"] < T]
    hn = [e for e in mine if e["kind"] in ("hn_i", "hn_q", "hn_x") and e["y0_valid"] and e["edit"] < T and (e.get("edit_b") is None or e["edit_b"] < T)]
    rows = probes + edits + hn
    K = Ktf[torch.tensor([e["row_tf"] for e in rows])].double().numpy()
    rel = sorted([e for e in edits if e["kind"] == "rel"], key=lambda e: e["edit"])
    Kb1 = Ktf[torch.tensor([e["row_tf"] for e in rel])].double().numpy()       # B = 1 keys of the edits' own Rel prompts
    s = (W * Ks).sum(1)
    Kn = Ks / np.linalg.norm(Ks, axis=1, keepdims=True)

    def feat(keys, feature):
        if feature == "ncoord":
            return (keys @ W.T) / s[None, :]
        return (keys @ Kn.T) / np.linalg.norm(keys, axis=1, keepdims=True)

    own = np.array([e["edit"] if e["kind"] != "probe" else -1 for e in rows])
    rep = [i for i, e in enumerate(rows) if e["kind"] == "probe" and e["pool"] == "repair"]
    return {"T": T, "rows": rows, "own": own, "repair": np.array(rep), "feat": {f: feat(K, f) for f in ("ncoord", "cos")},
            "guard": {f: np.concatenate([np.diag(feat(Ks, f)), np.diag(feat(Kb1, f))]).reshape(2, T) for f in ("ncoord", "cos")}}


def naive_decide(N, feature, cal, seed, capped=True, gamma=None):
    """Decisions of one method on every row, from the definitions.  -> dict(fired, own_cand, own_top, top_edit, eta, n_capped)."""
    R, T, own = N["feat"][feature], N["T"], N["own"]
    ar = np.arange(T)
    cap = np.full(T, np.inf)
    if cal == "fixed":
        cand = R >= gamma
        key, eta = R, None
    else:
        pool = N["repair"]
        idx = pool[np.sort(np.random.default_rng(seed).choice(len(pool), C.N_CAL, replace=False))]      # the seeded draw of n = 200
        Q = R[idx]
        if cal == "two_block":
            perm = np.random.default_rng(seed).permutation(C.N_CAL)
            i1, i2 = perm[:C.N_CAL // 2], perm[C.N_CAL // 2:]
            b = Q[i1].max(0)
            sc = np.sort((Q[i2] - b[None, :]).max(1))
            n2 = len(i2)
        else:
            b = np.zeros(T)
            sc = np.sort(Q.max(1))
            n2 = C.N_CAL
        k = -((-19 * (n2 + 1)) // 20)                                 # ceil((1 - 1/20)(n2 + 1)) in integer arithmetic
        assert C.ALPHA == 0.05 and k == math.ceil(0.95 * (n2 + 1) - 1e-9)
        eta = sc[k - 1] if k <= n2 else np.inf
        key = R - b[None, :]                                          # margin; ranks the candidates (shared: r itself)
        cand = key > eta                                              # STRICT
        if capped:                                                    # guards: stored solver key and B = 1 Rel key of every edit
            g = N["guard"][feature]                                   # (2, T) feature of edit i's guard keys on detector i
            sil = ~((g - b[None, :]) > eta)
            low = np.where(sil, g - TIE * np.abs(g), np.inf).min(0)
            cap = low
            cand = cand | (R >= cap[None, :])
    fired = cand.any(1)
    top = np.where(cand, key, -np.inf).argmax(1)
    top_edit = np.where(fired, top, -1)
    has = own >= 0
    own_cand = np.zeros(len(own), dtype=bool)
    own_cand[has] = cand[has, own[has]]
    own_top = fired & has & (top == own)
    return {"fired": fired, "own_cand": own_cand, "own_top": own_top, "top_edit": top_edit, "eta": eta, "n_capped": int(np.isfinite(cap).sum())}


def check_naive(name):
    G = C.jload(os.path.join(C.OUT, f"gate_{name}.json"))
    N = naive_log(name)
    res = {"rows_equal": [e["sid"] for e in N["rows"]] == [r["sid"] for r in G["rows"]], "n_rows": len(N["rows"]), "methods": {}}
    total = bad = 0
    for mid, spec in C.METHODS.items():
        m = {"n_decisions": 0, "n_differing": 0, "differing": [], "max_abs_eta_diff": 0.0, "capped_counts_equal": True}
        if spec["cal"] == "fixed":
            cases = [(None, "tf", G["methods"][mid]["fixed"]["tf"], dict(gamma=spec["gamma"]), None)]
        else:
            cases = []
            for e in G["methods"][mid]["draws"]:
                cases.append((e["seed"], "tf", e["tf"], dict(capped=True), e["rec"]))
                cases.append((e["seed"], "tf_uncapped", e["tf_uncapped"], dict(capped=False), e["rec_uncapped"]))
        for seed, variant, lib, kw, rec in cases:
            d = naive_decide(N, spec["feature"], spec["cal"], seed, **kw)
            for fl in ("fired", "own_cand", "own_top"):
                diff = d[fl] != C.unbits(lib[fl])
                m["n_decisions"] += int(diff.size)
                m["n_differing"] += int(diff.sum())
                if diff.any() and len(m["differing"]) < 10:
                    m["differing"].append({"seed": seed, "variant": variant, "flag": fl, "sids": [N["rows"][i]["sid"] for i in np.nonzero(diff)[0][:5]]})
            dt = d["top_edit"] != np.array(lib["top_edit"])
            m["n_decisions"] += int(dt.size)
            m["n_differing"] += int(dt.sum())
            if rec is not None:
                m["max_abs_eta_diff"] = max(m["max_abs_eta_diff"], abs(float(rec["eta"]) - d["eta"]))
                if variant == "tf":
                    m["capped_counts_equal"] = m["capped_counts_equal"] and d["n_capped"] == rec["n_edits_capped"]
        total += m["n_decisions"]
        bad += m["n_differing"]
        res["methods"][mid] = m
    res.update(n_decisions=total, n_differing=bad,
               ok=bool(res["rows_equal"] and bad == 0 and all(m["capped_counts_equal"] and m["max_abs_eta_diff"] < 1e-9 for m in res["methods"].values())))
    return res


# ----------------------------------------------------------------------------- B: ladder end point == gate-level numbers
def gate_draw_stats(G, mid, variant="tf"):
    """Per-draw gate statistics of a method from the per-row flags of out/gate_<log>.json (a single 'draw' for M5 / M6)."""
    info = G["rows"]
    es = [G["methods"][mid]["fixed"]] if "fixed" in G["methods"][mid] else G["methods"][mid]["draws"]
    return [C.gate_stats(info, C.unbits(e[variant]["fired"]), C.unbits(e[variant]["own_cand"]), C.unbits(e[variant]["own_top"]), e[variant]["top_edit"])
            for e in es]


def check_ladder(name):
    p = os.path.join(C.OUT, "referee.json")
    if not os.path.exists(p):
        return {"available": False}
    lad = C.jload(p)["logs"][name]["ladder"]
    G = C.jload(os.path.join(C.OUT, f"gate_{name}.json"))
    out = {"available": True, "max_abs_diff": 0.0}
    for cal, mid in (("two_block", "M1"), ("shared", "M2")):
        st = gate_draw_stats(G, mid)
        mine = {"firing": np.mean([s["probe"]["heldout_disjoint"]["fired"] for s in st]), "cov_rel": np.mean([s["edit"]["rel"]["own_top"] for s in st]),
                "cov_tgen": np.mean([s["edit"]["tgen"]["own_top"] for s in st]), "cov_igen": np.mean([s["edit"]["igen"]["own_top"] for s in st])}
        d = {k: abs(float(v) - lad[cal]["R5"][k]["mean"]) for k, v in mine.items()}
        out[cal] = {"gate_table": {k: float(v) for k, v in mine.items()}, "abs_diff_to_ladder_R5": d}
        out["max_abs_diff"] = max(out["max_abs_diff"], max(d.values()))
    out["ok"] = bool(out["max_abs_diff"] < 1e-9)
    return out


# ----------------------------------------------------------------------------- C, D: GPU rows
def check_hook(name):
    B = C.load_behaviour(name)
    if B is None:
        return {"available": False}
    G = C.jload(os.path.join(C.OUT, f"gate_{name}.json"))
    d = os.path.join(C.OUT, "behaviour", name)
    pos = {r["lib_sid"]: i for i, r in enumerate(G["rows"])}
    res = {"available": True, "n_flags": 0, "n_differing": 0, "differing": [], "max_rel_tau_diff": 0.0, "max_abs_eta_diff": 0.0, "draws_equal": True,
           "row_sets": {}}
    for g in ("edits", "probes", "hn"):
        sids = [r["sid"] for r in B["static"][g]["deployed"]]
        want = [r["lib_sid"] for r in G["rows"] if (r["kind"] in ("rel", "tgen", "igen") if g == "edits" else
                                                    (r["kind"] == "probe" and r["disjoint"]) if g == "probes" else r["kind"].startswith("hn"))]
        res["row_sets"][g] = {"n_gpu": len(sids), "n_cpu": len(want), "same_set": set(sids) == set(want)}
    for sname in B:
        gates = C.jload(os.path.join(d, f"gates_{sname}.json"))
        for mid in gates:
            if sname == "static":
                cpu = G["methods"][mid]["fixed"]
            else:
                seed = int(sname[4:])
                cpu = G["methods"][mid]["draws"][seed]
                assert cpu["seed"] == seed
                res["draws_equal"] = res["draws_equal"] and gates[mid]["rec"]["calibration_sids"] == G["draws"][seed]["calibration_sids"]
                res["max_abs_eta_diff"] = max(res["max_abs_eta_diff"], abs(float(gates[mid]["rec"]["eta"]) - float(cpu["rec"]["eta"])))
            a, b = np.array(gates[mid]["tau"], dtype=np.float64), np.array(cpu["tau"], dtype=np.float64)
            res["max_rel_tau_diff"] = max(res["max_rel_tau_diff"], float((np.abs(a - b) / np.abs(b)).max()))
            fl = {k: C.unbits(cpu["tf"][k]) for k in ("fired", "own_cand", "own_top")}
            for g in ("edits", "probes", "hn"):
                for r in B[sname][g][mid]:
                    i = pos[r["sid"]]
                    pairs = [("fired", bool(r["fired"]), bool(fl["fired"][i])), ("top", int(r["top_last"]), int(cpu["tf"]["top_edit"][i]))]
                    if "edit" in r:
                        pairs += [("own_cand", bool(r["own_fired_last"]), bool(fl["own_cand"][i])), ("own_top", bool(r["own_released_last"]), bool(fl["own_top"][i]))]
                    for nm, x, y in pairs:
                        res["n_flags"] += 1
                        if x != y:
                            res["n_differing"] += 1
                            if len(res["differing"]) < 20:
                                res["differing"].append({"set": sname, "method": mid, "sid": r["sid"], "flag": nm, "hook": x, "cpu": y})
    res["ok"] = bool(res["n_differing"] == 0 and res["draws_equal"] and all(v["same_set"] for v in res["row_sets"].values()))
    return res


def check_behaviour(name):
    B = C.load_behaviour(name)
    if B is None:
        return {"available": False}
    G = C.jload(os.path.join(C.OUT, f"gate_{name}.json"))
    bb = C.LOGS[name]["backbone"]
    mine = {r["sid"] for r in G["rows"]}
    with open(os.path.join(CACHE, bb, "index.jsonl")) as f:
        pre = {}
        for line in f:
            e = json.loads(line)
            if e.get("lib_sid") is not None and e.get("pre_exact") is not None and e["sid"] in mine:
                pre[e["lib_sid"]] = e
    st = B["static"]
    dep, w0 = st["probes"]["deployed"], st["probes"]["pre-edit"]
    res = {"available": True,
           "deployed_D1_equals_stored_D1": {"n": len(dep), "equal": sum(r["D1"] == r["D1_WE"] for r in dep)},
           "pre_edit_probe_D1": sum(r["D1"] for r in w0), "pre_edit_hn_D1": sum(r["D1"] for r in st["hn"]["pre-edit"]),
           "pre_edit_tok_loc_all_1": all(r["tok_loc"] == 1.0 for r in w0 + st["hn"]["pre-edit"]),
           "pre_edit_edit_exact_equals_cache": {"n": len(st["edits"]["pre-edit"]),
                                                "equal": sum(r["exact"] == pre[r["sid"]]["pre_exact"] and abs(r["token_acc"] - pre[r["sid"]]["pre_token_acc"]) < 1e-12
                                                             for r in st["edits"]["pre-edit"])},
           "silent_rows": {"n": 0, "equal_to_pre_edit": 0}, "nonfinite": {}, "fired_none_for_non_gates": True}
    for sname, gs in B.items():
        for g, ms in gs.items():
            base = {r["sid"]: r for r in st[g]["pre-edit"]}
            for m, rows in ms.items():
                nf = sum(not r["finite"] for r in rows)
                if nf:
                    res["nonfinite"][f"{sname}/{m}/{g}"] = nf
                if m in ("deployed", "pre-edit"):
                    res["fired_none_for_non_gates"] = res["fired_none_for_non_gates"] and all(r["fired"] is None for r in rows)
                    continue
                for r in rows:
                    if not r["fired"]:
                        p = base[r["sid"]]
                        res["silent_rows"]["n"] += 1
                        res["silent_rows"]["equal_to_pre_edit"] += int(all(r[k] == p[k] for k in ("exact", "token_acc", "n_tok", "finite", "first_mismatch"))
                                                                       and r.get("D1") == p.get("D1") and r.get("tok_loc") == p.get("tok_loc"))
    res["n_nonfinite_rows"] = sum(res["nonfinite"].values())
    res["ok"] = bool(res["deployed_D1_equals_stored_D1"]["equal"] == res["deployed_D1_equals_stored_D1"]["n"] and res["pre_edit_probe_D1"] == 0
                     and res["pre_edit_hn_D1"] == 0 and res["pre_edit_tok_loc_all_1"]
                     and res["pre_edit_edit_exact_equals_cache"]["equal"] == res["pre_edit_edit_exact_equals_cache"]["n"]
                     and res["silent_rows"]["equal_to_pre_edit"] == res["silent_rows"]["n"] and res["fired_none_for_non_gates"])
    return res


# ----------------------------------------------------------------------------- E: one seed for the draw and for the split
def check_seed_coupling(name, n_draws=6000):
    """Draw d uses seed d both for the draw of the n calibration rows and for the two-block split (docs/USAGE.md).  Is the split
    then still unrelated to the drawn rows?  M1's rule (uncapped) with the split seeded by d against the split seeded independently
    (d + 1000003), same draws: rows firing on HELD-OUT-DISJOINT, T-Gen coverage (own edit a candidate), and the balance of image
    rows between the blocks.  Library calls only."""
    import rescope
    from rescope import KeyCache, from_testbed
    kc = KeyCache(C.LOGS[name]["backbone"])
    L = from_testbed(name)
    uids, T = L.meta["uids"], L.T
    probes = kc.log_rows(uids, kind="probe", valid=True)
    pinfo = [kc.index[j] for j in probes]
    rep = np.array([i for i, e in enumerate(pinfo) if e["pool"] == "repair"])
    disj = np.array([i for i, e in enumerate(pinfo) if e.get("disjoint")])
    img = np.array([e["ptype"] == "image" for e in pinfo])
    G0 = C.base_gate(L, "ncoord")
    Rp = G0.scores(kc.keys("tf", probes)).numpy()
    Rt = G0.scores(kc.keys("tf", kc.log_rows(uids, kind="tgen"))).numpy()
    ar = np.arange(T)
    res = {}
    for mode, off in (("same_seed", 0), ("independent_seed", 1000003)):
        fire, cov, bal = [], [], []
        for d in range(n_draws):
            idx = rep[np.sort(np.random.default_rng(d).choice(len(rep), C.N_CAL, replace=False))]
            G, rec = rescope.calibrate_two_block(G0, coords=Rp[idx], alpha=C.ALPHA, seed=d + off, allow_silence=True)
            th = G.theta().numpy()
            fire.append(100.0 * (Rp[disj] >= th[None, :]).any(1).mean())
            cov.append(100.0 * (Rt[ar, ar] >= th).mean())
            bal.append(int(img[idx][np.array(rec["block1"])].sum()) - int(img[idx][np.array(rec["block2"])].sum()))
        res[mode] = {k: {"mean": float(np.mean(v)), "se": float(np.std(v, ddof=1) / np.sqrt(len(v)))}
                     for k, v in (("firing", fire), ("cov_tgen", cov), ("image_rows_block1_minus_block2", bal))}
    out = {"n_draws": n_draws, **res}
    for k in ("firing", "cov_tgen", "image_rows_block1_minus_block2"):
        d = res["same_seed"][k]["mean"] - res["independent_seed"][k]["mean"]
        se = float(np.hypot(res["same_seed"][k]["se"], res["independent_seed"][k]["se"]))
        out["diff_" + k] = {"mean": d, "se": se, "z": d / se if se > 0 else 0.0}
    out["ok"] = bool(all(abs(out["diff_" + k]["z"]) < 3 for k in ("firing", "cov_tgen", "image_rows_block1_minus_block2")))
    return out


if __name__ == "__main__":
    out = {"code": C.code_fingerprint(), "logs": {}}
    for name in C.LOGS:
        r = {"A_naive": check_naive(name), "B_ladder": check_ladder(name), "C_hook": check_hook(name), "D_behaviour": check_behaviour(name),
             "E_seed_coupling": check_seed_coupling(name)}
        out["logs"][name] = r
        for k, v in r.items():
            C.say(f"{name} {k}: " + json.dumps({a: b for a, b in C._clean(v).items() if a not in ("methods", "differing", "nonfinite")})[:700])
    C.jdump(out, os.path.join(C.OUT, "checks.json"), indent=1)
    C.say(f"wrote {os.path.join(C.OUT, 'checks.json')}")
