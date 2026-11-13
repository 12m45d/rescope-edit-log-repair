"""Mechanics check of rescope_x on the two development logs (the ONLY logs on which a gate decision may be looked at).

    python rc_mechanics.py            -> out/mechanics.txt, out/mechanics.json            (CPU; key cache + recombination keys; no model)

Calibration draw 0, alpha = 0.05, n = 200, the default method of the shared driver (fx.methods.base_gate / fx.draws.calibration: role
'last', normalised solver coordinate, top-1, REPL, rebased values, guards = stored solver keys + B = 1 own Rel keys; two-block rule,
seed 0).  'without' = that method; 'with' = the same with the per-edit baseline max(block 1, own recombinations) of amendment R5
(recomb.wiring.method).  Reported per log:
  (1) per kind the distribution of the edits' own-recombination coordinates next to their block-1 baselines;
  (2) eta, baselines and thresholds with and without the recombinations, capped edits;
  (3) gate level at the teacher-forced keys of the test rows (fx.gate.evaluate / fx.metrics.gate_stats): false firing on
      heldout_disjoint, coverage of text and image rephrases, firing on the test hard negatives per kind; the benign pool; the two gates
      on the recombination rows themselves.
A mechanics check, not a result: nothing is tuned, selected or compared with another method; m, the kinds and the rule are fixed by
the amendment.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rc_common as R  # noqa: E402

sys.path.insert(0, os.path.dirname(R.FEAT))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import recomb  # noqa: E402
from recomb import wiring  # noqa: E402

fx = wiring.use_fx()
ALPHA, N, DRAW = 0.05, 200, 0
QS = (0.0, 0.25, 0.5, 0.75, 1.0)


def q5(v):
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    return [float(x) for x in np.quantile(v, QS)] if v.size else [None] * 5


def fmt(v, w=7, p=3):
    return "-".rjust(w) if v is None else f"{v:{w}.{p}f}"


def pc(v, w=7):
    return "-".rjust(w) if v is None else f"{v:{w}.2f}"


def one(name: str) -> dict:
    L = fx.logs.open(name)
    out = {"log": name, "backbone": L.backbone, "T": L.T, "alpha": ALPHA, "n": N, "draw": DRAW}
    M_ref = fx.methods.rescope(L, alpha=ALPHA, n=N, draw=DRAW)           # the default method of the driver
    M0 = wiring.method(L, ALPHA, N, DRAW, x=False, label="rescope(without)")
    M1 = wiring.method(L, ALPHA, N, DRAW, x=True, label="rescope_x(with)")
    S0, S1 = M0.obj, M1.obj
    out["without_equals_fx_rescope_bitwise"] = bool(torch.equal(S0.det_tau, M_ref.obj.det_tau) and torch.equal(S0.det_b, M_ref.obj.det_b)
                                                    and S0.eta == M_ref.obj.eta)
    c0, c1, rr = M0.rec["cal"], M1.rec["cal"], M1.rec["cal"]["recomb"]
    # ---------------------------------------------------------------- (1) own-recombination coordinates next to the block-1 baselines
    base, _ = fx.methods.base_gate(L, "ncoord")
    dr = fx.draws.calibration(L, N, DRAW)
    Q = fx.methods.calib_coords(L, base, dr).numpy()
    b1 = Q[np.array(c0["block1"])].max(0)
    check_b1 = bool(np.array_equal(b1, S0.det_b.numpy()))
    o = recomb.own_scores(L, base)
    rows = o.rows
    x = o.x.numpy()
    kinds = {}
    for k in recomb.KINDS:
        m = o.per_edit(k)
        mk = rows.kind == k
        kinds[k] = {"n_rows": int(mk.sum()), "per_edit_max": q5(m), "per_row": q5(o.value[mk]), "n_edits_above_b1": int((m > b1).sum()),
                    "n_rows_above_b1_of_their_edit": int((o.value[mk] > b1[rows.edit[mk]]).sum()), "max_minus_b1": q5(m - b1)}
    arg = np.array([int(np.argmax([o.per_edit(k)[i] for k in recomb.KINDS])) for i in range(L.T)])
    d = fx.gate.features(L, base, L.rows("edit"), "tf")
    ek = np.array([L.index[j]["kind"] for j in L.rows("edit")])
    eo = np.array([L.own(j) for j in L.rows("edit")])
    own_edit = d[torch.arange(d.shape[0]), torch.from_numpy(eo)].numpy()
    out["coordinates"] = {"b1": q5(b1), "b1_is_det_b_of_the_default_method": check_b1, "kinds": kinds, "x": q5(x), "x_minus_b1": q5(x - b1),
                          "n_edits_x_above_b1": int((x > b1).sum()), "kind_attaining_x": {k: int((arg == i).sum()) for i, k in enumerate(recomb.KINDS)},
                          "own_coordinate_on_own_sequences(tf key)": {m: q5(own_edit[ek == m]) for m in ("rel", "tgen", "igen")},
                          "block1_n": len(c0["block1"]), "calibration_rows_max_over_edits": q5(Q.max(1))}
    # ---------------------------------------------------------------- (2) thresholds
    t0, t1 = S0.det_tau.numpy(), S1.det_tau.numpy()
    out["thresholds"] = {"eta": [c0["eta"], c1["eta"]], "baseline": [q5(S0.det_b.numpy()), q5(S1.det_b.numpy())],
                         "literal": [rr["literal_threshold_min_median_max_without_x"], rr["literal_threshold_min_median_max_with_x"]],
                         "installed": [q5(t0), q5(t1)], "n_installed_higher_with": int((t1 > t0).sum()), "n_installed_lower_with": int((t1 < t0).sum()),
                         "n_installed_equal": int((t1 == t0).sum()), "installed_change": q5(t1 - t0),
                         "n_literal_higher_with": rr["n_literal_thresholds_higher_with_x"], "n_literal_lower_with": rr["n_literal_thresholds_lower_with_x"],
                         "capped_edits": [c0["n_edits_capped"], c1["n_edits_capped"]], "capped_edit_ids": [c0["capped_edits"], c1["capped_edits"]],
                         "n_capped_in_both": len(set(c0["capped_edits"]) & set(c1["capped_edits"])),
                         "bound": [c0["bound_false_firing"], c1["bound_false_firing"]], "n1_n2_rank": [c1["n1"], c1["n2"], c1["rank"]],
                         "own_keys_silenced_by_literal_rule": [c0["n_own_keys_silenced_by_literal_rule"], c1["n_own_keys_silenced_by_literal_rule"]],
                         "guards_silenced_by_literal_rule": [c0["n_guards_silenced_by_literal_rule"], c1["n_guards_silenced_by_literal_rule"]]}
    # ---------------------------------------------------------------- (3) gate level
    st = [fx.metrics.gate_stats(L, fx.gate.evaluate(L, M, positions=("tf",)), "tf") for M in (M0, M1)]
    out["gate"] = [{"probe": s["probe"], "edit": s["edit"], "hn": s["hn"], "benign": s["benign"]} for s in st]
    rec = []
    for S in (S0, S1):
        dd = recomb.decide_rows(S, L)
        kk = dd["rows"].kind
        rec.append({k: {"n": int((kk == k).sum()), "fired": 100.0 * float(dd["fired"][kk == k].mean()), "own_cand": 100.0 * float(dd["own_cand"][kk == k].mean()),
                        "own_top": 100.0 * float(dd["own_top"][kk == k].mean()), "b_top": 100.0 * float(dd["b_top"][kk == k].mean())} for k in recomb.KINDS})
    out["recombination_rows"] = rec
    # ---------------------------------------------------------------- (4) calibration quantities over the registered draws (no test row)
    dd = {"eta_without": [], "eta_with": [], "capped_without": [], "capped_with": [], "raised": [], "own_rows_own_cand_with": []}
    for draw in range(fx.draws.N_DRAWS):
        A = wiring.method(L, ALPHA, N, draw, x=False)
        B = wiring.method(L, ALPHA, N, draw, x=True)
        dd["eta_without"].append(A.rec["cal"]["eta"])
        dd["eta_with"].append(B.rec["cal"]["eta"])
        dd["capped_without"].append(A.rec["cal"]["n_edits_capped"])
        dd["capped_with"].append(B.rec["cal"]["n_edits_capped"])
        dd["raised"].append(B.rec["cal"]["recomb"]["n_baselines_raised_by_x"])
        dd["own_rows_own_cand_with"].append(int(recomb.decide_rows(B.obj, L)["own_cand"].sum()))      # recombination rows, not test rows
    out["draws"] = dict(dd, n_draws=fx.draws.N_DRAWS, n_eta_with_negative=int(sum(e < 0 for e in dd["eta_with"])),
                        n_eta_without_negative=int(sum(e < 0 for e in dd["eta_without"])))
    out["records"] = {"recomb": M1.rec["recomb"], "cal_recomb": rr, "fingerprints": [M_ref.key, M0.key, M1.key]}
    return out


def render(res: dict) -> str:
    L = []
    c, t, g, rc = res["coordinates"], res["thresholds"], res["gate"], res["recombination_rows"]
    L.append(f"== {res['log']} ({res['backbone']}, T = {res['T']}) ==  calibration draw {res['draw']}, alpha = {res['alpha']}, n = {res['n']} "
             f"(block 1 = {t['n1_n2_rank'][0]} queries, block 2 = {t['n1_n2_rank'][1]}, rank {t['n1_n2_rank'][2]})")
    L.append(f"   'without' is fx.methods.rescope bit for bit: {res['without_equals_fx_rescope_bitwise']};   rows used: "
             + ", ".join(f"{k} {v['n_rows']}" for k, v in c["kinds"].items()) + "   (keys: prompt only, last prompt position)")
    L.append("")
    L.append("(1) own-recombination coordinates r_i(k) = <W_i, k> / s_i next to the block-1 baselines (quantiles over the edits)")
    hd = ("quantity", "min", "q25", "median", "q75", "max", "edits > b1", "rows > b1")
    rows = [("block-1 baseline b1_i (max over block 1)",) + tuple(fmt(v) for v in c["b1"]) + ("", "")]
    for k, v in c["kinds"].items():
        rows.append((f"{k}: max over the edit's rows",) + tuple(fmt(x) for x in v["per_edit_max"]) + (f"{v['n_edits_above_b1']} of {res['T']}", f"{v['n_rows_above_b1_of_their_edit']} of {v['n_rows']}"))
    rows.append(("all kinds: x_i",) + tuple(fmt(v) for v in c["x"]) + (f"{c['n_edits_x_above_b1']} of {res['T']}", ""))
    rows.append(("x_i - b1_i",) + tuple(fmt(v) for v in c["x_minus_b1"]) + ("", ""))
    for k, v in c["kinds"].items():
        rows.append((f"{k}: single rows",) + tuple(fmt(x) for x in v["per_row"]) + ("", ""))
    for m, lab in (("rel", "own prompt (Rel)"), ("tgen", "text rephrase (T-Gen)"), ("igen", "image rephrase (I-Gen)")):
        rows.append((f"for reference: own coordinate, {lab}",) + tuple(fmt(x) for x in c["own_coordinate_on_own_sequences(tf key)"][m]) + ("", ""))
    rows.append(("for reference: calibration queries, max over edits",) + tuple(fmt(x) for x in c["calibration_rows_max_over_edits"]) + ("", ""))
    L.append(R.text_table(rows, hd))
    L.append("    kind that attains x_i: " + ", ".join(f"{k} {n}" for k, n in c["kind_attaining_x"].items()))
    L.append("")
    L.append("(2) calibration                                        without      with recombinations")
    L.append(f"    eta (block-2 quantile of max_i (r_i - b_i))       {fmt(t['eta'][0], 9, 4)}    {fmt(t['eta'][1], 9, 4)}")
    L.append(f"    baseline b_i            min / median / max        " + " / ".join(fmt(t['baseline'][0][i], 6) for i in (0, 2, 4)) + "    "
             + " / ".join(fmt(t['baseline'][1][i], 6) for i in (0, 2, 4)))
    L.append(f"    literal threshold       min / median / max        " + " / ".join(fmt(v, 6) for v in t['literal'][0]) + "    " + " / ".join(fmt(v, 6) for v in t['literal'][1]))
    L.append(f"    installed threshold     min / median / max        " + " / ".join(fmt(t['installed'][0][i], 6) for i in (0, 2, 4)) + "    "
             + " / ".join(fmt(t['installed'][1][i], 6) for i in (0, 2, 4)))
    L.append(f"    literal thresholds with recombinations: higher for {t['n_literal_higher_with']} edits, lower for {t['n_literal_lower_with']}; "
             f"installed (after guards): higher {t['n_installed_higher_with']}, lower {t['n_installed_lower_with']}, equal {t['n_installed_equal']}; "
             f"change min / median / max " + " / ".join(fmt(t['installed_change'][i], 6) for i in (0, 2, 4)))
    L.append(f"    capped edits (guards; bound void for them)        {t['capped_edits'][0]:>9}    {t['capped_edits'][1]:>9}        (capped in both: {t['n_capped_in_both']}; "
             f"guards the literal rule would silence: {t['guards_silenced_by_literal_rule'][0]} / {t['guards_silenced_by_literal_rule'][1]})")
    L.append(f"    bound on benign false firing                      {100 * t['bound'][0]:8.2f} %   {100 * t['bound'][1]:8.2f} %")
    L.append("")
    L.append("(3) gate level, teacher-forced keys of the test rows (% of rows)        n    without       with")
    p0, p1 = g[0]["probe"], g[1]["probe"]
    for key, lab in (("heldout_disjoint", "false firing, heldout_disjoint (valid rows)"), ("heldout_disjoint_image", "    image rows"),
                     ("heldout_disjoint_text", "    text rows"), ("heldout_disjoint_near", "    rows whose question an edit of the log asks"),
                     ("heldout_disjoint_far", "    other rows"), ("repair", "for reference: REPAIR pool (calibration rows are drawn from it)")):
        L.append(f"    {lab:<62}{p0[key]['n']:>6}    {pc(p0[key]['fired'])}    {pc(p1[key]['fired'])}")
    e0, e1 = g[0]["edit"], g[1]["edit"]
    for m, lab in (("tgen", "coverage, text rephrases (T-Gen: the own edit releases)"), ("igen", "coverage, image rephrases (I-Gen: the own edit releases)"),
                   ("rel", "own prompts (Rel: the own edit releases)")):
        L.append(f"    {lab:<62}{e0[m]['n']:>6}    {pc(e0[m]['own_top'])}    {pc(e1[m]['own_top'])}")
        L.append(f"    {'    another edit releases':<62}{'':>6}    {pc(e0[m]['other_top'])}    {pc(e1[m]['other_top'])}")
    h0, h1 = g[0]["hn"], g[1]["hn"]
    for k in ("HN-I", "HN-Q", "HN-X"):
        L.append(f"    {'test hard negatives ' + k + ': an edit fires':<62}{h0[k]['n']:>6}    {pc(h0[k]['fired'])}    {pc(h1[k]['fired'])}")
        L.append(f"    {'    the own edit releases':<62}{'':>6}    {pc(h0[k]['own_top'])}    {pc(h1[k]['own_top'])}")
        if k == "HN-X":
            L.append(f"    {'    the edit whose question is asked releases':<62}{'':>6}    {pc(h0[k]['b_top'])}    {pc(h1[k]['b_top'])}")
    L.append(f"    {'test hard negatives, all kinds: an edit fires':<62}{h0['all']['n']:>6}    {pc(h0['all']['fired'])}    {pc(h1['all']['fired'])}")
    b0, b1_ = g[0]["benign"], g[1]["benign"]
    L.append(f"    {'for reference: benign pool, prompt-only keys':<62}{b0['n']:>6}    {pc(b0['fired'])}    {pc(b1_['fired'])}")
    for k in recomb.KINDS:
        L.append(f"    {'recombination rows ' + k + ' (prompt-only keys): an edit fires':<62}{rc[0][k]['n']:>6}    {pc(rc[0][k]['fired'])}    {pc(rc[1][k]['fired'])}")
        L.append(f"    {'    the own edit is a candidate':<62}{'':>6}    {pc(rc[0][k]['own_cand'])}    {pc(rc[1][k]['own_cand'])}")
        if k == "X-X":
            L.append(f"    {'    the edit whose question is asked releases':<62}{'':>6}    {pc(rc[0][k]['b_top'])}    {pc(rc[1][k]['b_top'])}")
    d = res["draws"]
    q3 = lambda v: " / ".join(fmt(x, 7, 4) for x in np.quantile(np.asarray(v, dtype=np.float64), [0.0, 0.5, 1.0]))          # noqa: E731
    i3 = lambda v: " / ".join(f"{int(x)}" for x in np.quantile(np.asarray(v), [0.0, 0.5, 1.0]))                             # noqa: E731
    L.append("")
    L.append(f"(4) calibration quantities over the {d['n_draws']} registered draws (calibration rows and guards only; no test row): min / median / max")
    L.append(f"    eta           without {q3(d['eta_without'])}      with {q3(d['eta_with'])}      draws with eta < 0: {d['n_eta_without_negative']} without, "
             f"{d['n_eta_with_negative']} with")
    L.append(f"    capped edits  without {i3(d['capped_without'])}      with {i3(d['capped_with'])}      baselines raised by the recombinations: {i3(d['raised'])} of {res['T']}")
    L.append(f"    recombination rows (of {sum(v['n_rows'] for v in c['kinds'].values())}) on which their OWN edit is a candidate, with: {i3(d['own_rows_own_cand_with'])}")
    L.append("    (eta < 0 means that an edit fires BELOW its baseline, hence possibly on its own recombinations; a capped edit can as well; the bound on")
    L.append("     benign queries holds either way)")
    return "\n".join(L)


def main():
    t0 = time.time()
    res = {bb: one(name) for bb, name in R.DEV.items()}
    head = ["MECHANICS CHECK of the variant rescope_x (PROTOCOL.md section 6, amendment \"Review round 1\", item R5: log-derived near-scope calibration)",
            "Development logs only (TB-Q100-ae-l2_3, TB-L100-ae-l2_3): the only logs on which a gate decision, a coverage or a firing rate may be looked at.",
            "A mechanics check, NOT a result: nothing was tuned or selected; m = 4, the three kinds and the rule are fixed by the amendment.  One calibration",
            "draw (draw 0): the numbers of section (3) carry the draw-to-draw spread of every calibrated gate (not shown here) and no interval.",
            "",
            "without = the default method (fx.methods.rescope: two-block rule; baseline b_i = largest coordinate of edit i over the 100 block-1 queries);",
            "with    = the same gate, same draw, same blocks, with b_i = max(block-1 baseline, x_i), x_i = the largest coordinate of edit i on ITS OWN",
            "          recombination rows (4 X-I + 4 X-Q + 4 X-X per edit; cache2/recomb/<backbone>); block 2, the quantile and the guards unchanged",
            "          (recomb.calibrate_two_block_x = rescope.calibrate_two_block with x as one more block-1 row).  An edit fires iff it is the top-1 edit by",
            "          r_i - b_i and r_i - b_i > eta: only the DIFFERENCES between the edits' baselines matter once eta is re-estimated on block 2.",
            ""]
    body = []
    for bb in R.DEV:
        body += [render(res[bb]), ""]
    foot = ["written by feat2/recomb/rc_mechanics.py (no time stamp: a re-run reproduces this file); numbers: out/mechanics.json; recombination store: "
            + "; ".join(f"{bb} rows sha256 {res[bb]['records']['recomb']['rows_sha256'][:12]} keys md5 {res[bb]['records']['recomb']['keys_md5']}" for bb in R.DEV)]
    txt = "\n".join(head + body + foot) + "\n"
    os.makedirs(os.path.join(R.FEAT, "out"), exist_ok=True)
    with open(os.path.join(R.FEAT, "out", "mechanics.txt"), "w") as f:
        f.write(txt)
    R.jdump(res, os.path.join(R.FEAT, "out", "mechanics.json"))
    print(txt)
    print(f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
