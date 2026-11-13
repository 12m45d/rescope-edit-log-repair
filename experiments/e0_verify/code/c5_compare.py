"""Check 5, CPU part: one calibration draw of M1 and M3 per development log, gate level and behaviour, against the reported rows.

  python c5_compare.py Q|L        -> out/c5_<tag>.json

GATE LEVEL (my naive implementation on the cached keys; no rescope code):
  * per-sequence decisions of M1 - M4 for ALL 20 draws (capped and uncapped) and of M5 / M6 against the stored flags of
    e0_sanity/out/gate_<log>.json; the calibration rows of every draw;
  * the reported gate-level row of every method recomputed from MY decisions and compared with e0_sanity/out/summary.txt
    (draw 0 of M1 and M3 is listed on its own).
BEHAVIOUR (needs out/c5_<tag>_rows.json of c5_behaviour.py: my own model run of draw 0):
  * my rows of M1 / M3 against the stored rows of e0_sanity/out/behaviour/<log>/draw0__*.json, per sequence;
  * my thresholds (CPU, naive) against the stored thresholds of that run (GPU, library);
  * the reported behaviour row of every method (mean of draws 0 - 2) recomputed from the stored per-row flags and compared with
    summary.txt; draw 0 of M1 / M3 from my rows next to it.
"""
from __future__ import annotations

import glob
import os
import re
import sys

import numpy as np

import vcommon as V

import torch  # noqa: E402

METH = {"M1": ("ncoord", "two_block"), "M2": ("ncoord", "shared"), "M3": ("cos", "two_block"), "M4": ("cos", "shared")}
FIXED = {"M5": 0.65, "M6": 0.999}
ALPHA, N_CAL, N_DRAWS = "1/20", 200, 20


def bits(s):
    return np.frombuffer(s.encode(), dtype=np.uint8) == ord("1")


def pct(v):
    v = np.asarray(v, dtype=bool)
    return 100.0 * float(v.mean()) if v.size else None


def nums(s):
    return [float(x) for x in re.findall(r"\d+(?:\.\d+)?", s)]              # the printed values are non-negative ('-' separates a range)


def close(mine, theirs, digits):
    """my value against a number printed with `digits` decimals."""
    return abs(mine - theirs) <= 0.5 * 10 ** (-digits) + 1e-9


def parse_summary():
    """The reported rows of e0_sanity/out/summary.txt -> {('gate' | 'beh', log letter, method): [column strings]}."""
    out, mode = {}, None
    for line in open(f"{V.SANITY}/summary.txt"):
        if line.startswith("log  method  fire"):
            mode = "gate"
            continue
        if line.startswith("log  method    Rel/T/I"):
            mode = "beh"
            continue
        if not line.strip() or line.startswith(("REFEREE", "NOTES", "BEHAVIOUR", "GATE")):
            mode = None
            continue
        if mode:
            p = line.split()
            if len(p) >= 3 and p[0] in ("Q", "L"):
                out[(mode, p[0], p[1])] = p[2:]
    return out


def gate_level(tag, res):
    d = V.DEV[tag]
    name, bb, fam = d["log"], d["backbone"], d["family"]
    log = V.read_log(name)
    T, uids, W, K = log["T"], log["uids"], log["W"], log["K"]
    idx = V.read_index(bb)
    Ktf = V.read_keys(bb, "tf")
    pos = {u: i for i, u in enumerate(uids)}
    prow = [e for e in idx if e["family"] == fam and e["kind"] == "probe"]
    ann = V.annotate(prow)
    for e, a in zip(prow, ann):
        e["_ann"] = a
    in_log = lambda e: e["edit"] < T and (e.get("edit_b") is None or e["edit_b"] < T)        # noqa: E731
    rows = [e for e in prow if e["y0_valid"]] + [e for e in idx if e["family"] == fam and e["kind"] in ("rel", "tgen", "igen") and in_log(e)] + \
        [e for e in idx if e["family"] == fam and e["kind"] in ("hn_i", "hn_q", "hn_x") and e["y0_valid"] and in_log(e)]
    Gs = V.jload(f"{V.SANITY}/gate_{name}.json")
    g = res["gate"] = {"rows_equal_stored": [e["sid"] for e in rows] == [r["sid"] for r in Gs["rows"]], "n_rows": len(rows)}
    X = Ktf[torch.tensor([e["row_tf"] for e in rows])].double().numpy()
    own = np.array([pos.get(e.get("edit_uid"), -1) for e in rows])
    own_b = np.array([e["edit_b"] if e.get("edit_b") is not None else -1 for e in rows])
    kind = np.array([e["kind"] for e in rows])
    hd = np.array([e["kind"] == "probe" and e["_ann"]["disjoint"] for e in rows])
    rep = [i for i, e in enumerate(rows) if e["kind"] == "probe" and e["_ann"]["pool"] == "repair"]
    rel = sorted([e for e in rows if e["kind"] == "rel"], key=lambda e: e["edit"])
    Xb1 = Ktf[torch.tensor([e["row_tf"] for e in rel])].double().numpy()
    gedit = np.concatenate([np.arange(T), np.arange(T)])
    g["n"] = {"heldout_disjoint": int(hd.sum()), "repair": len(rep), **{k: int((kind == k).sum()) for k in ("rel", "tgen", "igen", "hn_i", "hn_q", "hn_x")}}

    def stats(fired, top, cand, cand_unc=None):
        has = own >= 0
        oc = np.zeros(len(rows), dtype=bool)
        oc[has] = cand[has, own[has]]
        ot = fired & has & (top == own)
        bt = fired & (own_b >= 0) & (top == own_b)
        s = {"fire": pct(fired[hd]), "cov_tgen": pct(ot[kind == "tgen"]), "cov_igen": pct(ot[kind == "igen"]), "sil_after": int((~oc[kind == "rel"]).sum()),
             "hn_any": [pct(fired[kind == k]) for k in ("hn_i", "hn_q", "hn_x")], "hn_own": [pct(ot[kind == "hn_i"]), pct(ot[kind == "hn_q"]), pct((ot | bt)[kind == "hn_x"])]}
        if cand_unc is not None:
            ocu = np.zeros(len(rows), dtype=bool)
            ocu[has] = cand_unc[has, own[has]]
            s["sil_before"] = int((~ocu[kind == "rel"]).sum())
        return s, oc, ot

    tot = {"n_flags": 0, "n_differing": 0, "n_draws_cal_rows_differing": 0, "differing": []}

    def cmp(mine, lib, what):
        fired, top, oc, ot = mine
        for nm, a, b in (("fired", fired, bits(lib["fired"])), ("own_cand", oc, bits(lib["own_cand"])), ("own_top", ot, bits(lib["own_top"])),
                         ("top_edit", top, np.array(lib["top_edit"]))):
            df = a != b
            tot["n_flags"] += int(df.size)
            tot["n_differing"] += int(df.sum())
            if df.any() and len(tot["differing"]) < 10:
                tot["differing"].append({"what": what, "flag": nm, "sids": [rows[i]["sid"] for i in np.nonzero(df)[0][:5]]})

    per = {}
    for mid, (feature, rule) in METH.items():
        F = V.FEATURES[feature](X, W, K)
        Gm = np.concatenate([V.FEATURES[feature](K, W, K), V.FEATURES[feature](Xb1, W, K)])
        ds = []
        for dr in range(N_DRAWS):
            cal = [rep[i] for i in V.draw(len(rep), N_CAL, dr)]
            if [rows[i]["lib_sid"] for i in cal] != Gs["draws"][dr]["calibration_sids"]:
                tot["n_draws_cal_rows_differing"] += 1
            if rule == "two_block":
                i1, i2 = V.halves(N_CAL, dr)
                b, eta, k = V.naive_two_block(F[cal], i1, i2, ALPHA)
            else:
                b, eta, k = V.naive_shared(F[cal], ALPHA)
            cap, sil = V.naive_caps(Gm, gedit, b, eta)
            fu, tu, cu = V.naive_decide(F, b, eta)
            fc, tc, cc = V.naive_decide(F, b, eta, cap)
            s, oc, ot = stats(fc, tc, cc, cu)
            su, ocu, otu = stats(fu, tu, cu)
            s.update(eta=eta, n_capped=int(np.isfinite(cap).sum()), fire_uncapped=su["fire"])
            st = Gs["methods"][mid]["draws"][dr]
            cmp((fc, tc, oc, ot), st["tf"], f"{mid} draw {dr} capped")
            cmp((fu, tu, ocu, otu), st["tf_uncapped"], f"{mid} draw {dr} uncapped")
            s["eta_abs_diff_stored"] = abs(eta - float(st["rec"]["eta"])) if np.isfinite(eta) else 0.0
            s["n_capped_equal_stored"] = s["n_capped"] == st["rec"]["n_edits_capped"]
            ds.append(s)
        per[mid] = ds
    for mid, gamma in FIXED.items():
        F = V.FEATURES["cos"](X, W, K)
        cand = F >= gamma
        fired = cand.any(1)
        top = np.where(fired, np.where(cand, F, -np.inf).argmax(1), -1)
        s, oc, ot = stats(fired, top, cand)
        cmp((fired, top, oc, ot), Gs["methods"][mid]["fixed"]["tf"], f"{mid} fixed")
        per[mid] = [s]
    g["decisions_vs_stored_flags"] = tot
    # the reported rows
    rep_rows = parse_summary()
    table = {}
    for mid, ds in per.items():
        mean = lambda f: float(np.mean([f(s) for s in ds]))                                 # noqa: E731
        fire = [s["fire"] for s in ds]
        m = {"fire_mean": float(np.mean(fire)), "fire_p05": float(np.percentile(fire, 5)), "fire_p95": float(np.percentile(fire, 95)),
             "cov_tgen": mean(lambda s: s["cov_tgen"]), "cov_igen": mean(lambda s: s["cov_igen"]),
             "sil_before": mean(lambda s: s.get("sil_before", s["sil_after"])), "sil_after": mean(lambda s: s["sil_after"]),
             "hn_any": [mean(lambda s, j=j: s["hn_any"][j]) for j in range(3)], "hn_own": [mean(lambda s, j=j: s["hn_own"][j]) for j in range(3)]}
        r = rep_rows[("gate", tag, mid)]
        f = nums(r[0])
        ok = close(m["fire_mean"], f[0], 2)
        if mid in METH:
            ok = ok and close(m["fire_p05"], f[1], 2) and close(m["fire_p95"], f[2], 2)
        c = nums(r[1])
        ok = ok and close(m["cov_tgen"], c[0], 1) and close(m["cov_igen"], c[1], 1)
        sl = nums(r[2])
        ok = ok and (close(m["sil_before"], sl[0], 1) and close(m["sil_after"], sl[1], 0) if mid in METH else close(m["sil_after"], sl[0], 0))
        ok = ok and all(close(a, b, 0) for a, b in zip(m["hn_any"], nums(r[3]))) and all(close(a, b, 0) for a, b in zip(m["hn_own"], nums(r[4])))
        m.update(reported=" ".join(r), equal_reported=bool(ok))
        if mid in ("M1", "M3"):
            m["draw0"] = ds[0]
        table[mid] = m
    g["table"] = table
    g["all_reported_rows_reproduced"] = all(m["equal_reported"] for m in table.values())
    V.say(f"{name} gate: rows equal {g['rows_equal_stored']}, decisions {tot['n_differing']}/{tot['n_flags']} differing, cal rows differing in {tot['n_draws_cal_rows_differing']} draws, "
          f"reported rows reproduced { {k: v['equal_reported'] for k, v in table.items()} }")
    return rep_rows


def load_stored(name, sname):
    out = {}
    for p in sorted(glob.glob(f"{V.SANITY}/behaviour/{name}/{sname}__*__*.json")):
        part = V.jload(p)
        for m, rows in part["rows"].items():
            out.setdefault(m, {}).update({r["sid"]: r for r in rows})
    return out


def beh_stats(rows):
    """rows: list of dicts with kind fields: metric (edits), ptype / strict (probes), hn_type (hn); exact, D1, tok_loc."""
    e = lambda m: pct([r["exact"] for r in rows if r.get("metric") == m])                    # noqa: E731
    pr = [r for r in rows if r.get("ptype") is not None and r.get("hn_type") is None]
    hn = lambda t: pct([not r["exact"] for r in rows if r.get("hn_type") == t])             # noqa: E731
    tl = lambda pt: 100.0 * float(np.mean([r["tok_loc"] for r in pr if r["ptype"] == pt]))  # noqa: E731
    return {"exact": [e("rel"), e("tgen"), e("igen")], "D1": [pct([not r["exact"] for r in pr]), pct([not r["exact"] for r in pr if r["strict"]])],
            "hn_D1": [hn("HN-I"), hn("HN-Q"), hn("HN-X")], "loc": [tl("text"), tl("image")], "n": {"probes": len(pr), "strict": sum(bool(r["strict"]) for r in pr)}}


def behaviour(tag, res, rep_rows):
    name = V.DEV[tag]["log"]
    b = res["behaviour"] = {}
    # ---- the reported rows from the stored per-row flags (all methods)
    stored = {s: load_stored(name, s) for s in ("static", "draw0", "draw1", "draw2")}

    def norm(r):
        return {"metric": r.get("metric") if r["kind"] == "edit" else None, "ptype": r.get("ptype") if r["kind"] == "probe" else None, "strict": r.get("strict"),
                "hn_type": r.get("hn_type"), "exact": r["exact"], "tok_loc": r.get("tok_loc")}
    table = {}
    for mid in ("M1", "M2", "M3", "M4", "M5", "M6", "deployed"):
        sets = ("draw0", "draw1", "draw2") if mid in METH else ("static",)
        st = [beh_stats([norm(r) for r in stored[s][mid].values()]) for s in sets]
        m = {k: [float(np.mean([s[k][j] for s in st])) for j in range(len(st[0][k]))] for k in ("exact", "D1", "hn_D1", "loc")}
        r = rep_rows[("beh", tag, mid)]
        ok = all(close(a, x, 0) for a, x in zip(m["exact"], nums(r[0]))) and all(close(a, x, 2) for a, x in zip(m["D1"], nums(r[1]))) and \
            all(close(a, x, 1) for a, x in zip(m["hn_D1"], nums(r[2]))) and all(close(a, x, 2) for a, x in zip(m["loc"], nums(r[3])))
        m.update(reported=" ".join(r), equal_reported=bool(ok), per_draw=st if mid in ("M1", "M3") else None, n=st[0]["n"])
        table[mid] = m
    b["table_from_stored_rows"] = table
    b["all_reported_rows_reproduced_from_stored_rows"] = all(m["equal_reported"] for m in table.values())
    # ---- my own run of draw 0
    p = f"{V.OUT}/c5_{tag}_rows.json"
    if not os.path.exists(p):
        b["my_run"] = None
        V.say(f"{name} behaviour: my run is not there yet; reported rows from stored rows reproduced { {k: v['equal_reported'] for k, v in table.items()} }")
        return
    mine = V.jload(p)
    rows, meta = mine["rows"], mine["meta"]
    gates = V.jload(f"{V.OUT}/c5_{tag}/gates.json")
    sg = V.jload(f"{V.SANITY}/behaviour/{name}/gates_draw0.json")
    my = {"n_rows": len(rows), "n_keys_in_cache": sum(r["key_in_cache"] for r in rows), "n_keys_bitwise_cache": sum(r["key_bitwise_cache"] for r in rows),
          "n_lp_equal_cache": sum(r["lp_equals_cache"] for r in rows), "methods": {}}
    for mid in ("M1", "M3"):
        g, s = gates["gates"][mid], sg[mid]
        f = lambda x: float(x) if not isinstance(x, str) else float(x)                      # noqa: E731
        bm, cm = np.array([f(x) for x in g["b"]]), np.array([f(x) for x in g["cap"]])
        tau_s, b_s = np.array([f(x) for x in s["tau"]]), np.array([f(x) for x in s["b"]])
        tau_m = np.minimum(bm + float(g["eta"]), cm)
        thr = {"eta_mine": g["eta"], "eta_stored": s["rec"]["eta"], "abs_eta_diff": abs(float(g["eta"]) - float(s["rec"]["eta"])),
               "max_abs_b_diff": float(np.abs(bm - b_s).max()), "max_abs_threshold_diff": float(np.abs(tau_m - tau_s).max()),
               "capped_mine": [int(i) for i in np.nonzero(np.isfinite(cm))[0]], "capped_stored": s["rec"]["capped_edits"],
               "calibration_sids_equal": gates["calibration_sids"] == s["rec"]["calibration_sids"]}
        st = stored["draw0"][mid]
        diff = {k: 0 for k in ("fired", "top", "own_cand", "own_top", "exact", "token_acc", "tok_loc", "n_tok", "finite")}
        bad, n, n_bad = [], 0, 0
        for r in rows:
            x, y = r[mid], st.get(r["sid"])
            if y is None:
                bad.append({"sid": r["sid"], "missing_in_stored": True})
                continue
            n += 1
            is_edit = y["kind"] == "edit"
            d_ = {"fired": x["fired"] != bool(y["fired"]), "top": x["top"] != y["top_last"], "exact": x["exact"] != y["exact"],
                  "token_acc": abs(x["token_acc"] - y["token_acc"]) > 1e-6, "n_tok": x["n_tok"] != y["n_tok"], "finite": x["finite"] != y["finite"],
                  "own_cand": (meta[r["sid"]]["own"] >= 0) and x["own_cand"] != bool(y.get("own_fired_last")),
                  "own_top": (meta[r["sid"]]["own"] >= 0) and x["own_top"] != bool(y.get("own_released_last")),
                  "tok_loc": (not is_edit) and abs(x["tok_loc"] - y["tok_loc"]) > 1e-6}
            for k, v in d_.items():
                diff[k] += int(bool(v))
            n_bad += int(any(d_.values()))
            if any(d_.values()) and len(bad) < 10:
                bad.append({"sid": r["sid"], "flags": [k for k, v in d_.items() if v], "mine": x, "stored": {k: y.get(k) for k in ("fired", "top_last", "exact", "token_acc", "tok_loc", "own_fired_last", "own_released_last")}})
        mr = []
        for r in rows:
            mm = meta[r["sid"]]
            mr.append({"metric": mm.get("metric"), "ptype": mm.get("ptype"), "strict": mm.get("strict"), "hn_type": mm.get("hn_type"), "exact": r[mid]["exact"],
                       "tok_loc": r[mid].get("tok_loc")})
        ms = beh_stats(mr)
        ss = table[mid]["per_draw"][0]
        my["methods"][mid] = {"thresholds": thr, "n_rows_compared": n, "n_stored_rows": len(st), "flags_differing": diff, "n_rows_with_a_difference": n_bad,
                              "differing": bad, "stats_mine_draw0": ms, "stats_stored_draw0": ss,
                              "stats_equal": all(abs(a - c) < 1e-9 for k in ("exact", "D1", "hn_D1", "loc") for a, c in zip(ms[k], ss[k])),
                              "n_fired_mine": sum(r[mid]["fired"] for r in rows)}
    b["my_run"] = my
    V.say(f"{name} behaviour: keys bitwise {my['n_keys_bitwise_cache']}/{my['n_rows']}; " +
          "; ".join(f"{m}: flags differing {v['flags_differing']}, stats equal {v['stats_equal']}, thresholds {v['thresholds']['max_abs_threshold_diff']:.1e}" for m, v in my["methods"].items()))
    V.say(f"  reported rows reproduced from stored rows { {k: v['equal_reported'] for k, v in table.items()} }")


def main(tag):
    res = {"log": V.DEV[tag]["log"]}
    rep_rows = gate_level(tag, res)
    behaviour(tag, res, rep_rows)
    V.jdump(res, f"{V.OUT}/c5_{tag}.json")


if __name__ == "__main__":
    main(sys.argv[1])
