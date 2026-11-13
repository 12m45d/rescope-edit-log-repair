"""Unit test of the RQ4 harness extensions (all backwards compatible; defaults unchanged).

  sampler.split_pools(frac) / set_split   default split unchanged; 25/75 split by item, nested in 50/50;
                                          set_split(H) restores the default (pools, baseline cache tag)
  evaluate(per_item=True)                 same summary record as per_item=False; per-item values
                                          average to the summary numbers
  COMP prior / extra_basis / keep_keys    defaults bitwise unchanged; sequential COMP with
                                          accumulating protection: the final adapter restores W_0's
                                          module output at EVERY repaired complaint's query keys and
                                          vanishes on the protected span; ||X_batch|| <= ||X_seq||
                                          (batch = min-norm solution of the same constraints)
  ROLLBACK_FO offload / record_flags      identical removal sequence, trajectory and weights
  REEDIT chunk                            homogeneous rows, chunk >= n: bitwise identical W_after (same z*);
                                          mixed image+text rows (old path raises) are grouped by modality and
                                          put back in order: every installed complaint resolved with chunk 8
                                          and chunk 1; rel. diff of the two updates reported (batch-shape
                                          sensitivity of the editor's bf16 capture, informative only)

usage: CUDA_VISIBLE_DEVICES=1 python -m harness.tests.test_rq4_ext   -> out/tests/rq4_ext.json
"""
from __future__ import annotations

import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from harness import load, repair, evaluate, baseline, sample_complaints  # noqa: E402
from harness.common import CACHE, HARNESS_DIR, jdump, log, sha  # noqa: E402
from harness import sampler as SM  # noqa: E402
from harness import evaluation as EV  # noqa: E402
from harness.methods import protection_basis, _orth  # noqa: E402

OUT = os.path.join(HARNESS_DIR, "out", "tests")


def orth_against(Kq, Q, E):
    """Orthonormal basis of the part of span(Kq) outside span([Q, E]) (two passes of block GS)."""
    X = Kq
    for _ in range(2):
        if Q is not None:
            X = X - Q @ (Q.T @ X)
        if E is not None:
            X = X - E @ (E.T @ X)
    B, st = _orth(X)
    return B


def main(name="TB-Q100-ae"):
    t0 = time.time()
    res = {"log": name}
    H = load(name)
    # ---------------------------------------------------------------- split
    p_def = SM.pool_of(H)
    p_half = SM.split_pools(H, SM.SPLIT_SEED, 0.5)
    tag_def = sha([H.name, H.probe_info.get("probe_uids_sha"), EV.EVAL_BS, "v3"])
    p25 = SM.set_split(H, 0.25)
    rep = lambda p: {H.probes[i]["uid"] for i, v in p.items() if v == "repair"}  # noqa: E731
    uid_consistent = all(len({p25[i] for i, q in enumerate(H.probes) if q["uid"] == u}) == 1
                         for u in {q["uid"] for q in H.probes})
    tag_25 = SM.split_tag(H)
    SM.set_split(H)
    p_back = SM.pool_of(H)
    res["split"] = {"default_equals_frac_0.5": p_def == p_half, "n_items": len({q["uid"] for q in H.probes}),
                    "n_repair_50": len(rep(p_def)), "n_repair_25": len(rep(p25)),
                    "nested_25_in_50": rep(p25) <= rep(p_def), "item_level_25": uid_consistent,
                    "split_tag_25": tag_25, "restored_equals_default": p_back == p_def,
                    "split_tag_restored_none": SM.split_tag(H) is None}
    B = baseline(H)
    tag_now = sha([H.name, H.probe_info.get("probe_uids_sha"), EV.EVAL_BS, "v3"])
    res["split"]["baseline_tag_default_unchanged"] = tag_now == tag_def and os.path.exists(
        os.path.join(CACHE, f"baseline_{H.name}_{tag_def}.pt"))
    cs = sample_complaints(H, 4, seed=123)
    # ---------------------------------------------------------------- COMP defaults
    c0, _ = repair("COMP", H, cs[:2], protect=("K_ans",))
    c1, _ = repair("COMP", H, cs[:2], protect=("K_ans",), prior=None, extra_basis=None, keep_keys=False)
    res["comp_defaults_bitwise"] = bool(torch.equal(c0.Lf, c1.Lf) and torch.equal(c0.Rf, c1.Rf))
    # ---------------------------------------------------------------- evaluate per_item
    r0 = evaluate(H, c0, cs[:2], d2=False)
    r1 = evaluate(H, c0, cs[:2], d2=False, per_item=True)
    strip = lambda r: {k: v for k, v in r.items() if k != "heldout_rows"}  # noqa: E731

    def strip_deep(r):
        r = dict(strip(r))
        r["edits"] = {m: {k: v for k, v in d.items() if k not in ("em_list", "acc_list")} for m, d in r["edits"].items()}
        r["loc"] = {m: {k: v for k, v in d.items() if k not in ("uids", "per_item", "per_item_WE")}
                    for m, d in r["loc"].items()}
        return r
    rows = r1["heldout_rows"]
    lost = sum(1 for x in rows if x["reproduced_WE"] and not x["reproduced_after"])
    res["per_item"] = {
        "summary_identical": strip_deep(r1) == r0,
        "rel_exact_from_list": abs(100 * sum(r1["edits"]["rel"]["em_list"]) / len(r1["edits"]["rel"]["em_list"])
                                   - r1["edits"]["rel"]["exact"]) < 1e-9,
        "lost_from_rows": lost == r1["collateral"]["all"]["noncomplaints_lost"],
        "tloc_from_list": abs(100 * sum(r1["loc"]["tloc"]["per_item"]) / len(r1["loc"]["tloc"]["per_item"])
                              - r1["loc"]["tloc"]["after"]) < 1e-9}
    # ---------------------------------------------------------------- sequential COMP
    Q, _ = protection_basis(H, ("K_ans",))
    prior, E, Ks = None, None, []
    for c in cs[:3]:
        corr, cost = repair("COMP", H, [c], protect=("K_ans",), prior=prior, extra_basis=E, keep_keys=True)
        Kq = corr.meta.pop("_Kq")
        Ks.append(Kq)
        Eb = orth_against(Kq, Q, E)
        E = Eb if E is None else torch.cat([E, Eb], 1)
        prior = corr
    X = (prior.Lf @ prior.Rf.T).to(H.solve_dev)
    D = H.D_act
    restore = [float((X @ K + D @ K).norm() / (D @ K).norm()) for K in Ks]
    leak = float((X @ Q).norm() / X.norm())
    Rf = prior.Rf
    ortho = float((Rf.T @ Rf - torch.eye(Rf.shape[1], dtype=Rf.dtype, device=Rf.device)).abs().max())
    cb, _ = repair("COMP", H, cs[:3], protect=("K_ans",))
    Xb = (cb.Lf @ cb.Rf.T).to(H.solve_dev)
    restore_b = [float((Xb @ K + D @ K).norm() / (D @ K).norm()) for K in Ks]
    res["sequential"] = {"restore_rel_err_seq": restore, "restore_rel_err_batch": restore_b,
                         "protect_leak_rel": leak, "R_orthonormality_err": ortho,
                         "fro_seq": float(X.norm()), "fro_batch": float(Xb.norm()),
                         "batch_le_seq": float(Xb.norm()) <= float(X.norm()) * (1 + 1e-9),
                         "rank_seq": int(Rf.shape[1]), "rank_batch": int(cb.Rf.shape[1]),
                         "fro_meta_equals": abs(prior.fro(H) - float(X.norm())) < 1e-9 * float(X.norm())}
    del X, Xb
    # ---------------------------------------------------------------- rollback offload
    a, _ = repair("ROLLBACK_FO", H, cs[:3], budget=3)
    b, _ = repair("ROLLBACK_FO", H, cs[:3], budget=3, offload=True, record_flags=True)
    ta = [{k: v for k, v in t.items()} for t in a.meta["trajectory"]]
    tb = [{k: v for k, v in t.items() if k != "flags"} for t in b.meta["trajectory"]]
    flags_ok = all(sum(t["flags"]) == t["n_resolved"] for t in b.meta["trajectory"])
    res["rollback_offload"] = {"same_R": a.meta["R"] == b.meta["R"], "same_trajectory": ta == tb,
                               "same_W": (a.kind == b.kind) and (a.kind == "none" or bool(torch.equal(a.W, b.W))),
                               "flags_consistent": flags_ok, "R": a.meta["R"]}
    # ---------------------------------------------------------------- REEDIT chunk
    # homogeneous (text-only) complaints: chunk=None (the old path) vs chunk >= n -> bitwise
    txt = sample_complaints(H, 6, seed=321)[3:]
    r_a, _ = repair("REEDIT", H, txt, return_targets=True)
    tg = r_a.meta.pop("_targets")
    r_n, _ = repair("REEDIT", H, txt, targets=tg)
    r_8, _ = repair("REEDIT", H, txt, targets=tg, chunk=8)
    # mixed image + text complaints (the old path raises: vendor batching needs one modality)
    mixed = cs[:4]
    m_8, _ = repair("REEDIT", H, mixed, chunk=8, return_targets=True)
    tgm = m_8.meta.pop("_targets")
    m_8b, _ = repair("REEDIT", H, mixed, targets=tgm, chunk=8)
    m_1, _ = repair("REEDIT", H, mixed, targets=tgm, chunk=1)
    ok_kind = r_n.kind == "weight" and m_8b.kind == "weight"
    rc = {"text_only_chunk8_bitwise": bool(ok_kind and torch.equal(r_n.W, r_8.W)),
          "mixed_installed": m_8b.meta["installed_rows"], "mixed_zopt_attempts": m_8.meta["zopt_attempts"]}
    if ok_kind:
        dn = m_8b.W.double() - H.WE.double()
        d1 = m_1.W.double() - H.WE.double()
        rc["mixed_chunk1_vs_chunk8_rel_diff_update"] = float((d1 - dn).norm() / dn.norm())
        rc["mixed_solve_attainment"] = m_8b.meta["solve_attainment"]
        rm = evaluate(H, m_8b, mixed, d2=False, edits=False, collateral=False, loc=False)
        rc["mixed_resolved"] = [r["D1_resolved"] for r in rm["complaints"]]
        rm1 = evaluate(H, m_1, mixed, d2=False, edits=False, collateral=False, loc=False)
        rc["mixed_chunk1_resolved"] = [r["D1_resolved"] for r in rm1["complaints"]]
    res["reedit_chunk"] = rc
    s = res
    res["pass"] = bool(
        s["split"]["default_equals_frac_0.5"] and s["split"]["nested_25_in_50"] and s["split"]["item_level_25"]
        and s["split"]["restored_equals_default"] and s["split"]["split_tag_restored_none"]
        and s["split"]["baseline_tag_default_unchanged"]
        and s["comp_defaults_bitwise"] and all(s["per_item"].values())
        and max(s["sequential"]["restore_rel_err_seq"]) < 1e-8 and s["sequential"]["protect_leak_rel"] < 1e-8
        and s["sequential"]["R_orthonormality_err"] < 1e-10 and s["sequential"]["batch_le_seq"]
        and s["sequential"]["fro_meta_equals"]
        and s["rollback_offload"]["same_R"] and s["rollback_offload"]["same_trajectory"]
        and s["rollback_offload"]["same_W"] and s["rollback_offload"]["flags_consistent"]
        and s["reedit_chunk"]["text_only_chunk8_bitwise"]
        and all(s["reedit_chunk"].get("mixed_resolved", [False]))
        and s["reedit_chunk"].get("mixed_chunk1_resolved") == s["reedit_chunk"].get("mixed_resolved"))
    # mixed_chunk1_vs_chunk8_rel_diff_update is informative only: the bf16 capture forward of the
    # editor depends on the batch composition (padding / batch shape), so different chunkings give
    # slightly different keys K1 / current outputs h and the AlphaEdit solve amplifies them; the
    # row -> key mapping is checked behaviourally (every installed complaint resolved in both).
    res["seconds"] = time.time() - t0
    res["gpu"] = torch.cuda.get_device_name(0)
    jdump(res, os.path.join(OUT, "rq4_ext.json"))
    log(f"TEST rq4_ext: pass={res['pass']} ({res['seconds']:.0f}s) {res}")


if __name__ == "__main__":
    torch.set_num_threads(8)
    main()
