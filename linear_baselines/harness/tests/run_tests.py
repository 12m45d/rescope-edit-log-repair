"""Unit tests of the harness (GPU).  usage: python -m harness.tests.run_tests <test> [...]

  zero      zero correction (adapter gated/ungated, merged, W_E weight, empty rollback) is
            bit-identical to the unrepaired model (logits of complaint + edit sequences, and the
            full evaluate() record); suffix engine == full forward at W_E / W_0 / W(R) bitwise.
  comp_w2   COMP (K_last, answer positions, fp64 adapter, alpha 1) on the OLD W2 testbed reproduces
            cb_run's H1 per-complaint resolution on 20 random P8 complaints (10 image / 10 text).
  fo_w2     ROLLBACK-FO (budget 10) on the OLD W2 testbed reproduces P8's greedy_fo_v2 flag,
            removal sequence and pass counts on the same 20 complaints.
  reedit    REEDIT reproduces baselines/smoke (TB-Q100-ae, 8 complaints, variants k0 / k0+kp):
            resolution, Rel after, collateral on the smoke's 20 probes, dW/W0, z* steps; the
            factorised solve equals the vendor solve.
Each test writes out/tests/<test>.json; tests/render_tests.py renders TESTS.md from them.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from harness import load, repair, evaluate, Correction  # noqa: E402
from harness.common import EXP, HARNESS_DIR, TB, VM, jdump, log  # noqa: E402
from harness.suffix import Suffix  # noqa: E402

OUT = os.path.join(HARNESS_DIR, "out", "tests")
CB = os.path.join(EXP, "05_zeroloss_repair", "cb_run")
P8 = os.path.join(EXP, "03_algorithm_probes", "p8_probe")
BASE = os.path.join(EXP, "07_repair_v3", "baselines")


def p8_complaints():
    comp = [json.loads(l) for l in open(os.path.join(P8, "complaints_v2.jsonl"))]
    for ci, c in enumerate(comp):
        c["ci"] = ci
        c["y0_text"] = c["y0_text"].strip()
    rng = np.random.default_rng(2026)
    img = [c for c in comp if c["ptype"] == "image"]
    txt = [c for c in comp if c["ptype"] == "text"]
    sel = [img[int(i)] for i in sorted(rng.choice(len(img), 10, replace=False))] + \
          [txt[int(i)] for i in sorted(rng.choice(len(txt), 10, replace=False))]
    return sel


# =========================================================================== zero
def t_zero():
    from harness import sample_complaints
    H = load("TB-Q100-ae-l2_3")
    cs = sample_complaints(H, 4, seed=11)
    z = lambda: (torch.zeros((H.m_out, 3), dtype=torch.float64, device=H.dev),
                 torch.linalg.qr(torch.randn(H.d_in, 3, dtype=torch.float64, device=H.dev))[0])
    L0, R0 = z()
    corrs = {"adapter_gated": Correction("adapter", Lf=L0, Rf=R0, gate=1e-8),
             "adapter_ungated": Correction("adapter", Lf=L0, Rf=R0, gate=None),
             "merged": Correction("weight", W=(H.WE.double() + L0 @ R0.T).to(H.dtype)),
             "weight_WE": Correction("weight", W=H.WE.clone()),
             "rollback_empty": Correction("weight", W=H.rollback_weight([]))}
    seqs = [H.complaint_batch(c) for c in cs] + [H.edit_batch(i, m) for i in (0, 37, 81) for m in ("rel", "igen")]
    with torch.no_grad():
        ref = [H.vlm.forward(b, logits_to_keep="answer").logits.clone() for b in seqs]
    out = {"n_sequences": len(seqs), "logits_bitwise": {}, "eval_identical": {}}
    for k, c in corrs.items():
        with H.applied(c), torch.no_grad():
            out["logits_bitwise"][k] = all(torch.equal(H.vlm.forward(b, logits_to_keep="answer").logits, r)
                                           for b, r in zip(seqs, ref))
    r_none = evaluate(H, Correction("none", meta={"method": "NONE"}), cs)
    for k in ("adapter_gated", "merged"):
        r = evaluate(H, corrs[k], cs)
        same = (r["complaints"] == r_none["complaints"] and r["edits"] == r_none["edits"]
                and r["collateral"] == r_none["collateral"] and r["loc"] == r_none["loc"])
        out["eval_identical"][k] = same
    out["none_record_unchanged"] = {"edits_broken": r_none["edits_broken_total"],
                                    "edits_fixed": r_none["edits_fixed_total"],
                                    "collateral_changed_rows": len(r_none["collateral"]["changed_rows"]),
                                    "loc_items_changed": {m: r_none["loc"][m]["n_items_changed"] for m in r_none["loc"]},
                                    "edits_identical": all(v["identical_em_acc"] for v in r_none["edits"].values())}
    # suffix engine vs full forward at several weights
    S = Suffix(H)
    sw = {"WE": H.WE, "W0": H.W0, "W(R=3,17,55)": H.rollback_weight([3, 17, 55]),
          "W(batch 2 + edit 40)": H.rollback_weight(list(range(20, 30)) + [40])}
    sx = {}
    for nm, W in sw.items():
        okk = []
        for b in seqs[:6]:
            st = S.prepare(b)
            with H.at(W), torch.no_grad():
                o = H.vlm.forward(b, logits_to_keep="answer").logits[0, :st.A]
            okk.append(torch.equal(S.logits(st, W=W), o))
        sx[nm] = all(okk)
    out["suffix_equals_full"] = sx
    out["rollback_all_equals_W0"] = bool(torch.equal(H.rollback_weight(range(H.T)), H.W0))
    out["load_checks"] = H.checks
    # evaluation speed-ups are exact: vision-feature cache (miss, then hit) and the prebuilt
    # vendor-plan batches == vendor metrics.run_probes
    from harness.evaluation import vision_cache, _run_edits_b8 as _run_edits
    img = [b for b in seqs if not b["text_only"]]
    with torch.no_grad():
        r0 = [H.vlm.forward(b, logits_to_keep="answer").logits.clone() for b in img]
        with vision_cache(H):
            r1 = [H.vlm.forward(b, logits_to_keep="answer").logits.clone() for b in img]
            r2 = [H.vlm.forward(b, logits_to_keep="answer").logits.clone() for b in img]
    out["vision_cache_bitwise"] = all(torch.equal(a, b) and torch.equal(a, c) for a, b, c in zip(r0, r1, r2))
    E = _run_edits(H)
    vend = {}
    for m in ("rel", "tgen", "igen"):
        res = VM.run_probes(H.vlm, VM.build_probes(H.items, m), batch_size=8)
        vend[m] = bool(np.array_equal(E[m]["em"], np.array([bool(r["em"]) and r["finite"] for r in res]))
                       and np.array_equal(E[m]["acc"], np.array([r["acc"] for r in res])))
    out["prebuilt_batches_equal_vendor_run_probes"] = vend
    from harness.evaluation import baseline
    out["baseline_checks"] = baseline(H)["checks"]
    # keys at the edited module's input do not depend on the edited weight
    b = seqs[0]
    k_E = H.capture_inputs(b)
    with H.at(H.W0):
        k_0 = H.capture_inputs(b)
    out["keys_independent_of_edited_weight"] = bool(torch.equal(k_E, k_0))
    out["pass"] = (all(out["logits_bitwise"].values()) and all(out["eval_identical"].values())
                   and all(sx.values()) and out["rollback_all_equals_W0"]
                   and out["keys_independent_of_edited_weight"]
                   and out["none_record_unchanged"]["edits_broken"] == 0
                   and out["none_record_unchanged"]["collateral_changed_rows"] == 0
                   and out["none_record_unchanged"]["edits_identical"]
                   and out["vision_cache_bitwise"] and all(vend.values())
                   and out["baseline_checks"]["Rel_matches_stored"]
                   and out["baseline_checks"]["heldout_reproduced_equals_stored_v2"])
    return out


# =========================================================================== comp_w2
def t_comp_w2():
    H = load("old_W2")
    sel = p8_complaints()
    sweep = {}
    for l in open(os.path.join(CB, "sweep_rows.jsonl")):
        r = json.loads(l)
        if r["pset"] == "last" and r["qset"] == "A":
            sweep[r["ci"]] = r
    head = {}
    for l in open(os.path.join(CB, "head_rows.jsonl")):
        r = json.loads(l)
        if r["head"] == "H1_last_A_a1_adapter":
            head[r["ci"]] = r
    rows = []
    for c in sel:
        row = {"ci": c["ci"], "ptype": c["ptype"]}
        for tag, gate in (("ungated", None), ("gated", 1e-8)):
            corr, cost = repair("COMP", H, [c], protect=("K_last",), query="answer", form="adapter", gate=gate)
            rec = evaluate(H, corr, [c], d2=False, collateral=(tag == "ungated"), loc=False,
                           edits=(tag == "ungated"))
            row[tag] = {"resolved": rec["complaints"][0]["D1_resolved"], "dW_fro": corr.fro(H),
                        "edits_broken_rel": rec["edits"]["rel"]["n_broken"] if "edits" in rec else None,
                        "sec": cost["sec"], "passes": cost["passes"]}
        old = sweep[c["ci"]]
        row["old"] = {"resolved": old["res"]["1.0"]["cat"] == "resolved", "dW_fro": old["dW_fro"],
                      "edits_broken_rel": head[c["ci"]]["edits"]["n_broken"] if c["ci"] in head else None,
                      "t_solve": old.get("t_solve")}
        rows.append(row)
        log(f"comp_w2 ci={c['ci']} {c['ptype']} new={row['ungated']['resolved']} old={row['old']['resolved']} "
            f"dW {row['ungated']['dW_fro']:.4f}/{row['old']['dW_fro']:.4f}")
    agree = sum(r["ungated"]["resolved"] == r["old"]["resolved"] for r in rows)
    agree_g = sum(r["gated"]["resolved"] == r["old"]["resolved"] for r in rows)
    reld = [abs(r["ungated"]["dW_fro"] - r["old"]["dW_fro"]) / r["old"]["dW_fro"] for r in rows]
    out = {"n": len(rows), "complaints_ci": [r["ci"] for r in rows],
           "resolved_new_ungated": sum(r["ungated"]["resolved"] for r in rows),
           "resolved_new_gated": sum(r["gated"]["resolved"] for r in rows),
           "resolved_old": sum(r["old"]["resolved"] for r in rows),
           "flag_agreement_ungated": agree, "flag_agreement_gated": agree_g,
           "dW_fro_max_rel_diff": max(reld),
           "edits_broken_new": sum(r["ungated"]["edits_broken_rel"] for r in rows),
           "edits_broken_old": sum(r["old"]["edits_broken_rel"] for r in rows if r["old"]["edits_broken_rel"] is not None),
           "old_200_H1": 0.795, "load_checks": H.checks, "rows": rows}
    out["pass"] = agree >= 19 and out["dW_fro_max_rel_diff"] < 1e-6
    return out


# =========================================================================== fo_w2
def t_fo_w2():
    H = load("old_W2", old_w2_dense=True)
    sel = p8_complaints()
    g = {json.loads(l)["ci"]: json.loads(l) for l in open(os.path.join(P8, "greedy_fo_v2.jsonl"))}
    rows = []
    for c in sel:
        corr, cost = repair("ROLLBACK_FO", H, [c], budget=10)
        rec = evaluate(H, corr, [c], d2=False, edits=False, collateral=False, loc=False)
        old = g[c["ci"]]
        r = {"ci": c["ci"], "ptype": c["ptype"], "new": {"resolved": rec["complaints"][0]["D1_resolved"],
                                                         "resolved_internal": corr.meta["resolved_internal"],
                                                         "k": corr.meta["k"], "R": corr.meta["R"],
                                                         "passes": cost["passes"], "sec": cost["sec"]["total"]},
             "old": {"resolved": old["resolved"], "k": old["k_removed"], "R": old["removed_ids"],
                     "n_fwd": old["n_fwd"], "n_bwd": old["n_bwd"], "sec": old["t_wall"]}}
        r["same_flag"] = r["new"]["resolved"] == r["old"]["resolved"]
        r["same_R"] = r["new"]["R"] == r["old"]["R"]
        rows.append(r)
        log(f"fo_w2 ci={c['ci']} {c['ptype']} new {r['new']['resolved']} k={r['new']['k']} {r['new']['R']} | "
            f"old {r['old']['resolved']} k={r['old']['k']} {r['old']['R']}")
    out = {"n": len(rows), "flag_agreement": sum(r["same_flag"] for r in rows),
           "removal_sequence_identical": sum(r["same_R"] for r in rows),
           "internal_vs_evaluate_agree": sum(r["new"]["resolved"] == r["new"]["resolved_internal"] for r in rows),
           "resolved_new": sum(r["new"]["resolved"] for r in rows), "resolved_old": sum(r["old"]["resolved"] for r in rows),
           "mean_k_new": float(np.mean([r["new"]["k"] for r in rows])), "mean_k_old": float(np.mean([r["old"]["k"] for r in rows])),
           "mean_sec_new": float(np.mean([r["new"]["sec"] for r in rows])), "mean_sec_old": float(np.mean([r["old"]["sec"] for r in rows])),
           "load_checks": H.checks, "rows": rows}
    out["pass"] = out["flag_agreement"] >= 19
    return out


# =========================================================================== reedit
def t_reedit():
    sm = json.load(open(os.path.join(BASE, "smoke", "results.json")))
    H = load(sm["log"])
    v1 = {r["group"]: r for r in TB.jsonl_read(os.path.join(H.path, "complaints_D1.jsonl"))}
    allrec = {r["group"]: r for r in TB.jsonl_read(os.path.join(H.path, "probe_records.jsonl"))}
    coll = [allrec[g] for g in sm["collateral_probes"]]

    def rel():
        res = VM.run_probes(H.vlm, VM.build_probes(H.items, "rel"), batch_size=8)
        s = TB.acc_summary(res)
        return s["token"], s["exact"]

    def tfs(r):
        b = H.vlm.build_batch([r["image"]], [r["prompt"]], answers=[r["y0_text"]], fmt=r["fmt"])
        with torch.no_grad():
            o = H.vlm.forward(b, logits_to_keep="answer")
        n = int(b["answer_mask"][0].sum())
        return bool((o.logits[0, :n].argmax(-1) == b["answer_ids"][0, :n]).all())

    rows = []
    for sr in sm["rows"]:
        if sr.get("failed") or sr["variant"] not in ("k0", "k0+kp"):
            continue
        c = dict(v1[sr["group"]])
        corr, cost = repair("REEDIT", H, [c], variant=sr["variant"], check_vendor=True)
        with H.applied(corr):
            ok = tfs(c)
            rt, re_ = rel()
            cb = sum(not tfs(r) for r in coll)
        r = {"group": sr["group"], "variant": sr["variant"],
             "new": {"resolved": ok, "rel_token": rt, "rel_exact": re_, "coll_broken": cb,
                     "dW_over_W0": corr.meta.get("applied_dW_over_W0"), "dW_over_W0_solver": corr.meta.get("dW_over_W0"), "z_steps": corr.meta["targets"][0]["steps"],
                     "z_oracle": corr.meta["targets"][0]["oracle"], "attempts": corr.meta["zopt_attempts"][0],
                     "attainment": corr.meta.get("solve_attainment"), "vendor_check": corr.meta.get("vendor_check"),
                     "passes": cost["passes"], "sec": cost["sec"]},
             "old": {"resolved": sr["resolved"], "rel_token": sr["rel_token_after"], "rel_exact": sr["rel_exact_after"],
                     "coll_broken": sr["coll_broken"], "dW_over_W0": sr["delta_over_W0"],
                     "z_steps": sr["target"]["steps_used"], "attainment": sr["solve"]["attainment"],
                     "passes": sr["cost"]["passes"], "sec_total": sr["cost"]["sec_total"]}}
        rows.append(r)
        log(f"reedit {r['group']} {r['variant']} new {r['new']['resolved']} rel {rt:.2f}/{re_:.0f} cb {cb} "
            f"dW {r['new']['dW_over_W0']:.4e} | old {r['old']['resolved']} rel {r['old']['rel_token']:.2f}/"
            f"{r['old']['rel_exact']:.0f} cb {r['old']['coll_broken']} dW {r['old']['dW_over_W0']:.4e}")
    f = lambda k: sum(r["new"][k] == r["old"][k] for r in rows)
    out = {"n": len(rows), "resolved_agree": f("resolved"), "rel_exact_agree": f("rel_exact"),
           "coll_broken_agree": f("coll_broken"), "z_steps_agree": f("z_steps"),
           "rel_token_max_absdiff": max(abs(r["new"]["rel_token"] - r["old"]["rel_token"]) for r in rows),
           "dW_over_W0_max_reldiff": max(abs(r["new"]["dW_over_W0"] - r["old"]["dW_over_W0"]) / r["old"]["dW_over_W0"] for r in rows),
           "vendor_rel_diff_D_max": max(r["new"]["vendor_check"]["rel_diff_D"] for r in rows),
           "vendor_W_after_entries_differing_max": max(r["new"]["vendor_check"]["W_after_entries_differing"] for r in rows),
           "sec_solve_new_median": float(np.median([r["new"]["sec"]["solve"] for r in rows])),
           "sec_total_new_median": float(np.median([r["new"]["sec"]["total"] for r in rows])),
           "sec_total_old_median": float(np.median([r["old"]["sec_total"] for r in rows])),
           "rows": rows}
    out["pass"] = out["resolved_agree"] == out["n"] and out["vendor_rel_diff_D_max"] < 1e-10
    return out


# =========================================================================== kseq
def t_kseq(n_seq=26, n_c=6):
    """Protecting EVERY position of the n shortest edit sequences (K_seq:n) leaves those sequences
    bit-identical end to end (all positions, full logits) under the gated fp64 adapter."""
    from harness import sample_complaints
    H = load("TB-Q100-ae-l2_3")
    ek = H.edit_keys()
    order = sorted(range(H.T), key=lambda i: (ek["len"][i], i))[:n_seq]
    seqs = {i: H.edit_batch(i, "rel") for i in order}
    with torch.no_grad():
        ref = {i: H.vlm.forward(b).logits.cpu() for i, b in seqs.items()}      # CPU: keep the process < 16 GB
    cs = sample_complaints(H, n_c, seed=5)
    rows = []
    for c in cs:
        r = {"group": c["group"]}
        for tag, gate in (("gated", 1e-8), ("ungated", None)):
            corr, cost = repair("COMP", H, [c], protect=(f"K_seq:{n_seq}",), query="answer", gate=gate)
            with H.applied(corr), torch.no_grad():
                same = [torch.equal(H.vlm.forward(b).logits.cpu(), ref[i]) for i, b in seqs.items()]
            r[tag] = {"n_identical": int(sum(same)), "fro": corr.fro(H), "rank": corr.meta["protect_rank"],
                      "infeasible": corr.meta["infeasible"]}
        rec = evaluate(H, corr, [c], d2=False, edits=False, collateral=False, loc=False)
        r["resolved_ungated"] = rec["complaints"][0]["D1_resolved"]
        rows.append(r)
        log(f"kseq {c['group']} gated {r['gated']['n_identical']}/{n_seq} ungated {r['ungated']['n_identical']}/{n_seq} "
            f"rank {r['gated']['rank']} res {r['resolved_ungated']}")
    out = {"n_seq": n_seq, "n_positions": int(sum(ek["len"][i] for i in order)), "d_in": H.d_in,
           "n_star_d_in_over_mean_len": H.d_in / float(np.mean(ek["len"])),
           "repairs_all_identical_gated": sum(r["gated"]["n_identical"] == n_seq for r in rows),
           "repairs_all_identical_ungated": sum(r["ungated"]["n_identical"] == n_seq for r in rows),
           "n_repairs": len(rows), "resolved": sum(r["resolved_ungated"] for r in rows), "rows": rows}
    out["pass"] = out["repairs_all_identical_gated"] == len(rows)
    return out


# =========================================================================== fasteval
def t_fasteval():
    """evaluate() through the exact suffix engine == evaluate() with full forwards, for adapter and
    weight corrections (COMP, merged COMP, ROLLBACK-FO, REEDIT)."""
    import harness.evaluation as EV
    from harness import sample_complaints
    H = load("TB-Q100-ae-l2_3")
    cs = sample_complaints(H, 4, seed=21)
    rows = []
    for c in cs:
        for meth, cfg in (("COMP", {}), ("COMP", {"form": "merged"}), ("ROLLBACK_FO", {"budget": 10}), ("REEDIT", {})):
            corr, _ = repair(meth, H, [c], **cfg)
            t0 = time.time()
            EV.FAST_EVAL = True
            a = evaluate(H, corr, [c], d2=False)
            t1 = time.time()
            EV.FAST_EVAL = False
            b = evaluate(H, corr, [c], d2=False)
            t2 = time.time()
            EV.FAST_EVAL = True
            same = all(a[k] == b[k] for k in ("complaints", "edits", "collateral", "loc"))
            rows.append({"group": c["group"], "method": meth, "cfg": cfg, "identical": same,
                         "sec_fast": t1 - t0, "sec_full": t2 - t1, "broken": a["edits_broken_total"],
                         "lost": a["collateral"]["all"]["noncomplaints_lost"]})
            log(f"fasteval {c['group']} {meth} {cfg} identical={same} fast {t1-t0:.1f}s full {t2-t1:.1f}s")
    out = {"n": len(rows), "n_identical": sum(r["identical"] for r in rows),
           "sec_fast_median": float(np.median([r["sec_fast"] for r in rows])),
           "sec_full_median": float(np.median([r["sec_full"] for r in rows])),
           "baseline_checks": EV.baseline(H)["checks"], "rows": rows}
    out["pass"] = out["n_identical"] == out["n"]
    return out


# =========================================================================== llava
def t_llava(name="TB-L100-ae-l2_3"):
    """Every method + the evaluation runs on the LLaVA backbone (fp16, L16); suffix == full bitwise;
    GPU memory of the process."""
    from harness import sample_complaints
    torch.cuda.reset_peak_memory_stats()
    H = load(name)
    cs = sample_complaints(H, 2, seed=0)
    S = Suffix(H)
    sx = []
    for c in cs:
        b = H.complaint_batch(c)
        st = S.prepare(b)
        sx.append(torch.equal(S.logits(st), st.logits_ref))
        W = H.rollback_weight([1, 2, 50])
        with H.at(W), torch.no_grad():
            o = H.vlm.forward(b, logits_to_keep="answer").logits[0, :st.A]
        sx.append(torch.equal(S.logits(st, W=W), o))
    rows = []
    for meth, cfg in (("NONE", {}), ("COMP", {}), ("COMP", {"form": "merged", "query": "all"}), ("REEDIT", {}),
                      ("ROLLBACK_FO", {"budget": 10}), ("ROLLBACK_BEH", {"budget": 3}), ("ROLLBACK_ALL", {})):
        corr, cost = repair(meth, H, cs[:1], **cfg)
        rec = evaluate(H, corr, cs[:1])
        rows.append({"method": meth, "cfg": cfg, "resolved": rec["complaints"][0]["D1_resolved"],
                     "edits_broken": rec["edits_broken_total"], "lost": rec["collateral"]["all"]["noncomplaints_lost"],
                     "fro": rec["fro"], "passes": cost["passes"], "sec": cost["sec"]["total"]})
        log(f"llava {meth} {cfg} res {rows[-1]['resolved']} broken {rows[-1]['edits_broken']} lost {rows[-1]['lost']} "
            f"passes {cost['passes']} {cost['sec']['total']:.2f}s")
    from harness.evaluation import baseline
    out = {"log": name, "suffix_equals_full": sx, "load_checks": H.checks, "baseline_checks": baseline(H)["checks"],
           "rows": rows, "max_gpu_mem_GB": torch.cuda.max_memory_allocated() / 1e9}
    out["pass"] = all(sx) and out["baseline_checks"]["Rel_matches_stored"] and out["baseline_checks"]["heldout_reproduced_equals_stored_v2"]
    return out


# =========================================================================== reedit_noise
def t_reedit_noise(groups=("image_427", "text_191"), reps=2):
    """Run-to-run noise of z*: the ORIGINAL baselines/reedit.py twice and the harness REEDIT twice on
    the same complaint (TB-Q100-ae, variant k0); vendor zopt's GPU backward is not deterministic."""
    sys.path.insert(0, BASE)
    import reedit as RE
    H = load("TB-Q100-ae")
    v1 = {r["group"]: r for r in TB.jsonl_read(os.path.join(H.path, "complaints_D1.jsonl"))}
    H.set_weight(H.W0)
    rl = RE.RepairLog("TB-Q100-ae", vlm=H.vlm)          # reads W_0 from the module, installs W_E
    ok = torch.equal(rl.W0, H.W0) and torch.equal(rl.WE, H.WE)
    rows = []
    for g in groups:
        c = v1[g]
        for k in range(reps):
            co, _ = RE.repair_reedit(rl, c, "k0")
            W_old = (rl.WE.float() + co["delta"].to(H.dev)).to(H.dtype)
            rows.append({"group": g, "impl": "baselines/reedit.py", "rep": k, "applied_dW_over_W0": co["delta_over_W0"],
                         "applied_dW_over_W0_fp64": float((W_old.double() - H.WE.double()).norm() / H.W0.double().norm()),
                         "z_delta_over_h": co["target"]["delta_over_h"], "z_nll": co["target"]["final_nll"],
                         "z_steps": co["target"]["steps_used"]})
            cn, _ = repair("REEDIT", H, [c], variant="k0")
            rows[-1]["W_after_bitwise_equal_harness_same_rep"] = bool(torch.equal(W_old, cn.W))
            rows.append({"group": g, "impl": "harness", "rep": k, "applied_dW_over_W0": cn.meta["applied_dW_over_W0"],
                         "applied_dW_over_W0_fp64": cn.meta["applied_dW_over_W0"],
                         "z_delta_over_h": cn.meta["targets"][0]["delta_over_h"], "z_nll": cn.meta["targets"][0]["nll"],
                         "z_steps": cn.meta["targets"][0]["steps"]})
            log(f"reedit_noise {rows[-2]} | {rows[-1]}")
    spread = {}
    for g in groups:
        v = [r["applied_dW_over_W0_fp64"] for r in rows if r["group"] == g]
        spread[g] = (max(v) - min(v)) / min(v)
    out = {"W_E_identical_to_reedit_py": ok, "rows": rows, "rel_spread_applied_dW": spread, "pass": ok}
    return out


# =========================================================================== rq2cells
def t_rq2cells(name="TB-Q100-ae-l2_3", n=4, seed=12):
    """RQ2 options of COMP / REEDIT (backwards compatible):
    (a) defaults unchanged: COMP() == COMP(solve="exact") bitwise; REEDIT() still equals the vendor
        solve (W_after bitwise) and REEDIT(targets=<its own z*>) reproduces its W_after bitwise;
    (b) REEDIT(form="adapter") = the same D as the merged re-edit (fp64, vs the vendor solve);
    (c) REEDIT(solve="exact"): X k1 = z* - h exactly, R^T Q = 0 (protected K_ans keys get a zero gated
        adapter output, bitwise);
    (d) COMP(solve="alphaedit") == vendor Filters.solve on (K_q, V) with the log's rule and Kp."""
    from harness import sample_complaints
    from harness.hlog import adapter_apply
    from harness.methods import protection_basis
    H = load(name)
    cs = sample_complaints(H, n, seed=seed)
    ek = H.edit_keys()
    Kans = torch.cat([ek["K"][i][ek["pp"][i]] for i in range(H.T)]).to(H.dev)        # (N, d) module dtype
    rows = []
    for c in cs:
        r = {"group": c["group"]}
        a1, _ = repair("COMP", H, [c], protect=("K_ans",))
        a2, _ = repair("COMP", H, [c], protect=("K_ans",), solve="exact")
        r["comp_default_bitwise"] = bool(torch.equal(a1.Lf, a2.Lf) and torch.equal(a1.Rf, a2.Rf))
        ca, cost_a = repair("COMP", H, [c], solve="alphaedit", check_vendor=True)
        r["comp_ae_vendor_rel_diff_D"] = ca.meta["vendor_check"]["rel_diff_D"]
        r["comp_ae_attainment"] = ca.meta["solve_attainment"]
        r["comp_ae_Rf_orthonormal_err"] = float((ca.Rf.T @ ca.Rf - torch.eye(ca.Rf.shape[1], dtype=torch.float64,
                                                                            device=ca.Rf.device)).abs().max())
        e0, cost0 = repair("REEDIT", H, [c], check_vendor=True, return_targets=True)
        tg = e0.meta.pop("_targets")
        r["reedit_oracle"] = bool(tg[0].oracle)
        if e0.kind != "weight":
            r["skipped"] = "no oracle z*"
            rows.append(r)
            continue
        r["reedit_default_vendor_W_after_differing"] = e0.meta["vendor_check"]["W_after_entries_differing"]
        e1, cost1 = repair("REEDIT", H, [c], targets=tg)
        r["reedit_reused_targets_W_after_bitwise"] = bool(torch.equal(e0.W, e1.W))
        r["reused_cost_has_no_zopt"] = cost1["passes"]["fwd_bwd"] == 0 and cost0["passes_zopt"]["fwd_bwd"] > 0
        e2, _ = repair("REEDIT", H, [c], targets=tg, form="adapter", check_vendor=True)
        r["reedit_adapter_vendor_rel_diff_D"] = e2.meta["vendor_check"]["rel_diff_D"]
        r["reedit_adapter_attainment_minus_merged"] = e2.meta["solve_attainment"] - e0.meta["solve_attainment"]
        e3, _ = repair("REEDIT", H, [c], targets=tg, solve="exact", form="adapter")
        r["reedit_exact_attainment"] = e3.meta["solve_attainment"]
        r["reedit_exact_RtQ_max"] = e3.meta["P2_RtQ_max"]
        with torch.no_grad():
            out0 = torch.randn(Kans.shape[0], H.m_out, device=H.dev).to(H.dtype)
            out1 = adapter_apply(Kans, out0, e3.Lf, e3.Rf, e3.gate)
        r["reedit_exact_protected_outputs_bitwise"] = bool(torch.equal(out0, out1))
        rows.append(r)
        log(f"rq2cells {r}")
    Q, _ = protection_basis(H, ("K_ans",))
    ok = [x for x in rows if "skipped" not in x]
    res = {"log": name, "n": len(rows), "n_checked_reedit": len(ok), "rows": rows, "protect_rank_K_ans": int(Q.shape[1]),
           "pass": bool(all(x["comp_default_bitwise"] and x["comp_ae_vendor_rel_diff_D"] < 1e-9 for x in rows)
                        and len(ok) > 0
                        and all(x["reedit_default_vendor_W_after_differing"] == 0
                                and x["reedit_reused_targets_W_after_bitwise"] and x["reused_cost_has_no_zopt"]
                                and x["reedit_adapter_vendor_rel_diff_D"] < 1e-9
                                and abs(x["reedit_exact_attainment"] - 1) < 1e-9
                                and x["reedit_exact_RtQ_max"] < 1e-12
                                and x["reedit_exact_protected_outputs_bitwise"] for x in ok))}
    return res


# =========================================================================== kpc (RQ3 extension)
def t_kpc(name="TB-Q100-ae-l2_3", n_c=3, seed=7):
    """protect "K_pc:<r>" (top-r principal directions of K_all, added for RQ3).  Checks: the
    eigendecomposition of the uncentred second moment G = sum_i K_i^T K_i (orthonormality,
    eigen-residual, descending order, energy captured by U_64 recomputed from the keys == top-64
    eigenvalues / trace); K_pc:0 == none (identical correction); K_pc:d_in -> full rank, every complaint
    infeasible, zero correction, module output bitwise W_E; K_pc:(d_in - A) still restores exactly (P1)
    with the correction orthogonal to the protected span (P2); union rank (K_last + K_pc:64) == an
    independent matrix_rank; the pre-existing specs (K_last, K_ans, K_seq:3, K_last+probe:20) give the
    bitwise same basis as the pre-extension code path (concatenate, then cast)."""
    import torch.nn.functional as Fnn
    from harness import sample_complaints
    from harness.hlog import adapter_apply
    from harness.methods import protection_basis, pc_basis, _orth, _probe_rows, _probe_keys
    H = load(name)
    ek = H.edit_keys()
    U, ev = pc_basis(H)
    dev = U.device
    out = {"log": name, "d_in": H.d_in, "n_keys_K_all": int(sum(K.shape[0] for K in ek["K"]))}
    I = torch.eye(H.d_in, dtype=torch.float64, device=dev)
    out["orthonormal_err"] = float((U.T @ U - I).abs().max())
    del I
    G = torch.zeros((H.d_in, H.d_in), dtype=torch.float64, device=dev)
    for K in ek["K"]:
        Kd = K.to(dev).double()
        G.addmm_(Kd.T, Kd)
    idx = [0, 63, 1000, H.d_in - 1]
    evd = ev.to(dev)
    out["eig_residual_rel_max"] = max(float((G @ U[:, j] - evd[j] * U[:, j]).norm()) / float(evd[0]) for j in idx)
    del G
    out["evals_descending"] = bool((ev[:-1] >= ev[1:]).all())
    out["eval_min_over_max"] = float(ev[-1] / ev[0])
    num = sum(float((K.to(dev).double() @ U[:, :64]).pow(2).sum()) for K in ek["K"])
    den = sum(float(K.to(dev).double().pow(2).sum()) for K in ek["K"])
    out["energy64_from_keys"] = num / den
    out["energy64_from_evals"] = float(ev[:64].sum() / ev.sum())
    cs = sample_complaints(H, n_c, seed=seed)
    rows = []
    for c in cs:
        c0, _ = repair("COMP", H, [c], protect=("K_pc:0",))
        cn, _ = repair("COMP", H, [c], protect=("none",))
        cf, _ = repair("COMP", H, [c], protect=(f"K_pc:{H.d_in}",))
        x = H.capture_inputs(H.complaint_batch(c))[None]
        oE = Fnn.linear(x, H.WE)
        oc = adapter_apply(x, oE, cf.Lf, cf.Rf, cf.gate)
        A = int(cf.meta["n_query"])
        cr, _ = repair("COMP", H, [c], protect=(f"K_pc:{H.d_in - A}",))
        rows.append({"group": c["group"], "A": A,
                     "pc0_equals_none": bool(torch.equal(c0.Lf, cn.Lf) and torch.equal(c0.Rf, cn.Rf)),
                     "full_infeasible": bool(cf.meta["infeasible"]), "full_fro": cf.fro(H),
                     "full_module_out_bitwise_WE": bool(torch.equal(oE, oc)),
                     "r_d_minus_A_rank_kept": int(cr.meta["rank_kept"]), "r_d_minus_A_P1_rel": cr.meta["P1_rel"],
                     "r_d_minus_A_P2": cr.meta["P2_RtQ_max"], "r_d_minus_A_fro": cr.fro(H), "none_fro": cn.fro(H)})
        for k in [k for k in H._setup if isinstance(k, tuple) and k[0] == "basis" and str(k[1]).startswith("K_pc")]:
            del H._setup[k]
        log(f"kpc {rows[-1]}")
    out["rows"] = rows
    Q, st = protection_basis(H, ("K_last", "K_pc:64"))
    KL = torch.stack([ek["K"][i][ek["last"][i]] for i in range(H.T)]).to(dev).double()
    out["union_rank"] = int(st["rank"])
    out["union_rank_independent"] = int(torch.linalg.matrix_rank(torch.cat([KL, U[:, :64].T], 0)))
    compat = {}
    for spec in (("K_last",), ("K_ans",), ("K_seq:3",), ("K_last", "probe:20")):
        H._setup.pop(("basis",) + spec, None)
        Qn, _ = protection_basis(H, spec)
        cols = []
        for s in spec:
            nm, _, arg = s.partition(":")
            if nm == "K_last":
                cols.append(torch.stack([ek["K"][i][ek["last"][i]] for i in range(H.T)]))
            elif nm == "K_ans":
                cols.append(torch.cat([ek["K"][i][ek["pp"][i]] for i in range(H.T)]))
            elif nm == "K_seq":
                order = sorted(range(H.T), key=lambda i: (ek["len"][i], i))[:int(arg)]
                cols.append(torch.cat([ek["K"][i] for i in order]))
            elif nm == "probe":
                cols.append(_probe_keys(H, _probe_rows(H, int(arg)), "ans"))
        Qo, _ = _orth(torch.cat(cols, 0).to(H.solve_dev).double().T.contiguous())
        compat["+".join(spec)] = bool(torch.equal(Qn, Qo))
        H._setup.pop(("basis",) + spec, None)
    out["compat_bitwise_basis"] = compat
    out["pass"] = (out["orthonormal_err"] < 1e-10 and out["eig_residual_rel_max"] < 1e-10 and out["evals_descending"]
                   and abs(out["energy64_from_keys"] - out["energy64_from_evals"]) < 1e-9
                   and all(r["pc0_equals_none"] and r["full_infeasible"] and r["full_module_out_bitwise_WE"]
                           and r["full_fro"] == 0.0 and r["r_d_minus_A_P2"] < 1e-8 for r in rows)
                   and out["union_rank"] == out["union_rank_independent"] == H.T + 64
                   and all(compat.values()))
    return out


TESTS = {"kpc": t_kpc, "rq2cells": t_rq2cells, "reedit_noise": t_reedit_noise, "llava": t_llava, "fasteval": t_fasteval, "kseq": t_kseq, "zero": t_zero, "comp_w2": t_comp_w2, "fo_w2": t_fo_w2, "reedit": t_reedit}

if __name__ == "__main__":
    torch.set_num_threads(8)
    for name in sys.argv[1:]:
        t0 = time.time()
        res = TESTS[name]()
        res["seconds"] = time.time() - t0
        res["gpu"] = torch.cuda.get_device_name(0)
        jdump(res, os.path.join(OUT, f"{name}.json"))
        log(f"TEST {name}: pass={res['pass']} ({res['seconds']:.0f}s)")
        torch.cuda.empty_cache()
