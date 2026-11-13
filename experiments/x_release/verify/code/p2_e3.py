#!/usr/bin/env python3
"""Part 2: E3 fix-ups (both backbones) from per-row / per-draw files: paired differences H2 and H3 (point values exact, intervals and
two-sided values from MY resamples: 20000, own generator and seed), Holm over the unseen logs, and the reporting read-outs next to
far locality (near-scope probes released, calibration draws above the bound, capped edits, self-separation margin, edits at cap in H3).
-> out/mine_e3.json"""
import json
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vlib import *  # noqa: E402,F401,F403

SEL = {"cos_H2": "white_0.1_shared", "cos_H3": "white_1_shared", "cos_H3_x": "white_1_x", "retr": "retr_sum"}     # e1_main/out/best_cosine_gate.json, checked below


def edit_grid(log, method, part, edits=None):
    """(draws, edits) matrix of (T-Gen exact + I-Gen exact) / 2 per edit; the list of edits.  part 'h3': the stored rows hold the odd
    edits under the raised scale; an unreachable draw (cell record) counts 0."""
    mats, units = [], None
    for d in bdraws(log):
        acc = {}
        if part == "h3":
            c = jload(f"{FINAL}/{EXP[bb(log)]}/out/behave/{log}/{method}__d{d}__h3.json")
            if not c["reachable"]:
                acc = {e: [0.0, 2] for e in c["odd_edits"]}
        if not acc:
            for r in rows(e3_path(log, method, d, part)):
                if r["kind"] == "edit" and r["metric"] in ("tgen", "igen") and r.get("lp", "tf") == "tf":
                    a = acc.setdefault(r["edit"], [0.0, 0])
                    a[0] += float(bool(r["exact"]))
                    a[1] += 1
        if edits is not None:
            acc = {e: v for e, v in acc.items() if e in edits}
        assert all(v[1] == 2 for v in acc.values()), (log, method, part, d)
        if units is None:
            units = sorted(acc)
        assert sorted(acc) == units, (log, method, part, d)
        mats.append([acc[u][0] / 2.0 for u in units])
    return np.array(mats), units


def at_cap(log, method):
    v = [jload(f"{FINAL}/{EXP[bb(log)]}/out/behave/{log}/{method}__d{d}__h3.json") for d in bdraws(log)]
    return mean([x.get("n_edits_at_cap") for x in v]), sum(not x["reachable"] for x in v)


def gate_cells(log, method):
    b = jload(f"{FINAL}/{EXP[bb(log)]}/out/gate/{log}/{method}__reg.json")
    return [b["cells"][k] for k in sorted(b["cells"], key=int)]


def hd_flags(log, cells):
    """(draws, valid heldout_disjoint rows) firing flags of the gate-level cells, and the near-scope mask"""
    un = lambda s: np.array([c == "1" for c in s], dtype=bool)                                   # noqa: E731
    if bb(log) == "Q":
        c = jload(f"{FINAL}/e3_qwen/out/gate/{log}/_context.json")
        return np.array([un(x["hd_flags"]) for x in cells]), un(c["hd"]["near"])
    c = jload(f"{FINAL}/e3_llava/out/gate/{log}/_meta.json")
    hd = un(c["probe"]["masks"]["hd"])
    return np.array([un(x["probe_fired"])[hd] for x in cells]), un(c["probe"]["masks"]["hd_near"])[hd]


def main():
    best = jload(FINAL + "/e1_main/out/best_cosine_gate.json")
    out = {"selection_file_text_has": {k: (v in json.dumps(best)) for k, v in SEL.items()}, "logs": {}}
    for log in QLOGS + LLOGS:
        b, T = bb(log), T_of(log)
        o = {"T": T, "behavioural_draws": list(bdraws(log)), "H2": {}, "H3": {}}
        later = set(range(LATER_FROM[b], T))
        A, units = edit_grid(log, "rescope", "tf")
        assert units == list(range(T)), log
        for role, ref in (("Cos", SEL["cos_H2"]), ("Retr", SEL["retr"])):
            Bm, u2 = edit_grid(log, ref, "tf")
            assert u2 == units
            o["H2"][role + "All"] = dict(crossed_boot_diff(A, Bm), ref=ref, files=f"{rel(e3_path(log, 'rescope', 0))} | {ref}; draws {list(bdraws(log))}")
            if later:
                idx = np.array(sorted(later))
                o["H2"][role + "Later"] = dict(crossed_boot_diff(A[:, idx], Bm[:, idx]), ref=ref, n_later=len(idx))
        for pair, a, ref in (("PairOne", "rescope", SEL["cos_H3"]), ("PairTwo", "rescope_x", SEL["cos_H3_x"]), ("Retr", "rescope", SEL["retr"])):
            Ah, ua = edit_grid(log, a, "h3")
            Bh, ub = edit_grid(log, ref, "h3")
            assert ua == ub and all(e % 2 == 1 for e in ua), (log, pair)
            ca, na = at_cap(log, a)
            cb, nb = at_cap(log, ref)
            o["H3"][pair] = dict(crossed_boot_diff(Ah, Bh), a=a, b=ref, at_cap_a=ca, at_cap_b=cb, unreachable=na + nb,
                                 files=f"{rel(e3_path(log, a, 0, 'h3'))} | {ref}; cell records out/behave/{log}/<method>__d<draw>__h3.json")
        # ---- reporting next to far locality
        rep = {}
        per = []
        for d in bdraws(log):
            v = [bool(r["fired"]) for r in rows(e3_path(log, "rescope", d)) if is_hd(r) and r.get("near_edit") is not None]
            per.append(pct(v))
            rep["near_n"] = len(v)
        rep["near_rel_beh"] = mean(per)
        cells = gate_cells(log, "rescope")
        F, near = hd_flags(log, cells)
        rate = [100.0 * float(f.mean()) for f in F]
        rep.update(n_gate_draws=len(cells), n_hd=int(F.shape[1]), draws_above=int(sum(r > BOUND + 1e-12 for r in rate)), rates=rate, fire_mean=mean(rate),
                   near_rel_gate=mean([100.0 * float(f[near].mean()) for f in F]), capped_mean=mean([int(x["n_capped"]) for x in cells]),
                   capped_max=max(int(x["n_capped"]) for x in cells), bound_in_cells=sorted({round(100 * x["cal"]["bound_false_firing"], 9) for x in cells}))
        mf = f"{FINAL}/e3_qwen/out/factor/{log}.json" if b == "Q" else f"{FINAL}/e3_llava/out/gate/{log}/_factor.json"
        pe = jload(mf)["self_separation_margin"]["per_edit"]
        rep.update(margin_min=min(pe), margin_nonpositive=sum(1 for x in pe if x <= 0), margin_n=len(pe), margin_file=rel(mf))
        o["reporting"] = rep
        out["logs"][log] = o
        forget()
        print(log, "H2 cos %.2f [%.2f, %.2f] p %.4g | H3 p1 %.2f p2 %.2f retr %.2f | near %.2f above %d cap %.2f margin %.4f" % (
            o["H2"]["CosAll"]["value"], o["H2"]["CosAll"]["ci_lo"], o["H2"]["CosAll"]["ci_hi"], o["H2"]["CosAll"]["p"], o["H3"]["PairOne"]["value"],
            o["H3"]["PairTwo"]["value"], o["H3"]["Retr"]["value"], rep["near_rel_beh"], rep["draws_above"], rep["capped_mean"], rep["margin_min"]), flush=True)
    # ---- Holm over the unseen logs: H2 = 11 logs x 2 references, H3 = 11 logs x 3 comparisons (my p values, floor 1 / 20000)
    unseen = [l for l in QLOGS + LLOGS if l not in DEV.values()]
    fams = {"H2": {f"{l} | {r}": out["logs"][l]["H2"][r + "All"] for l in unseen for r in ("Cos", "Retr")},
            "H3": {f"{l} | {r}": out["logs"][l]["H3"][r] for l in unseen for r in ("PairOne", "PairTwo", "Retr")}}
    out["holm_mine"] = {}
    for h, mem in fams.items():
        adj = holm({k: v["p"] for k, v in mem.items()})
        # the stored values have resolution 1 / 2000: the same decision rule with my p floored at that resolution
        adj2 = holm({k: max(v["p"], 0.0005) for k, v in mem.items()})
        rej = [k for k in mem if adj[k] <= 0.05]
        rej2 = [k for k in mem if adj2[k] <= 0.05]
        out["holm_mine"][h] = {"family": len(mem), "adjusted": adj, "adjusted_with_floor_0_0005": adj2, "rejected": len(rej), "rejected_with_floor_0_0005": len(rej2),
                               "rejected_worse": sum(mem[k]["value"] < 0 for k in rej2), "rejected_better": sum(mem[k]["value"] > 0 for k in rej2),
                               "not_rejected": [k for k in mem if k not in rej2], "not_rejected_no_floor": [k for k in mem if k not in rej]}
        print(h, {k: v for k, v in out["holm_mine"][h].items() if not k.startswith("adjusted")})
    json.dump(out, open(OUT + "/mine_e3.json", "w"), indent=1)


if __name__ == "__main__":
    main()
