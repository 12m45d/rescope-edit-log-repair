"""E0 (1): gate-level decisions from the key caches, 20 seeded calibration draws.  CPU only, no model.

  python e0_gate.py TB-Q100-ae-l2_3      ->  out/gate_TB-Q100-ae-l2_3.json
  python e0_gate.py TB-L100-ae-l2_3      ->  out/gate_TB-L100-ae-l2_3.json

For each log: the cached sequences of the log (valid probe rows, Rel / T-Gen / I-Gen of the 100 edits, valid hard negatives of the
first 100 edits), and for every method and draw the firing decision of every sequence (rescope.KeyCache.decide):
  variants per calibrated method (M1 - M4) and draw
     tf           the method as defined: guards on (capped thresholds), gate at the teacher-forced last prompt position
     tf_uncapped  the literal conformal rule (allow_silence=True): the event the conformal bound covers
     prompt       the capped gate at the decode's token of the teacher-forced sequence (differs where lp_shift != 0: LLaVA)
     decode       the capped gate on the prompt-only key (first forward of a decode)
  M5 / M6 (fixed thresholds): tf, prompt, decode.
The JSON keeps the static row records, the calibration draws, thresholds, calibration records and per-row flags; every
statistic is computed from them by e0_render.py (definitions: e0_common.gate_stats).
"""
from __future__ import annotations

import sys
import time

import e0_common as C

import torch  # noqa: E402

torch.set_num_threads(4)
import rescope  # noqa: E402
from rescope import KeyCache, add_guards, from_testbed, splits  # noqa: E402


def row_record(e, T):
    """Static record of one cached sequence (what the statistics need)."""
    r = {"sid": e["sid"], "lib_sid": e.get("lib_sid"), "kind": e["kind"], "edit": e.get("edit"), "lp_shift": e.get("lp_shift")}
    if e["kind"] == "probe" or e["kind"].startswith("hn"):           # degenerate pre-edit answers (counted in every table)
        r.update(y0_deg={k: bool(e.get("deg_" + k)) for k in ("empty", "single_char", "qmark", "none_no")}, n_ans_tok=e.get("pre_n_tok"))
    if e["kind"] == "probe":
        r.update(ptype=e["ptype"], pool=e["pool"], disjoint=bool(e.get("disjoint")), strict=bool(e.get("strict")),
                 question_in_repair=bool(e.get("question_in_repair")), probe_idx=e["probe_idx"], y0_valid=bool(e["y0_valid"]))
    elif e["kind"].startswith("hn"):
        rel, qe = e.get("src_related_edit_first_pos"), e.get("q_in_edit_first_pos")
        r.update(hn_type=e["hn_type"], edit_b=e.get("edit_b"), group=e.get("hn_group"), new=bool(e.get("new")), rule=e.get("rule"),
                 related_flag=bool(e.get("related_flag")), same_category=bool(e.get("same_category")), y0_valid=bool(e["y0_valid"]),
                 src_related_to_log=rel is not None and int(rel) < T, q_edit=int(qe) if qe is not None and int(qe) < T else None,
                 deg_none_no=bool(e.get("deg_none_no")))
    return r


def flags(dec, full=True):
    out = {"fired": C.bits(dec["fired"]), "own_cand": C.bits(dec["own_cand"]), "own_top": C.bits(dec["own_top"]),
           "top_edit": [int(x) for x in dec["top_edit"]]}
    if full:
        out.update(near=C.bits(dec["near"]), n_cand_max=int(dec["n_cand"].max()), n_near=int(dec["near"].sum()))
    return out


def run(name: str):
    C.check_log(name)
    t0 = time.time()
    bb = C.LOGS[name]["backbone"]
    kc = KeyCache(bb, verify=True)                                    # md5 of the key files against meta.json
    L = from_testbed(name)
    uids, T = L.meta["uids"], L.T
    probes = kc.log_rows(uids, kind="probe", valid=True)
    edits = kc.log_rows(uids, kind=("rel", "tgen", "igen"))
    hn = kc.log_rows(uids, kind=("hn_i", "hn_q", "hn_x"), valid=True)
    repair = kc.log_rows(uids, kind="probe", pool="repair", valid=True)
    rows = probes + edits + hn
    info = [row_record(kc.index[j], T) for j in rows]
    assert all(r["group"] == "first100" for r in info if r["kind"].startswith("hn")), "a T = 100 log has first-100 hard negatives only"
    own = kc.own_keys(uids)                                           # B = 1 keys of the edits' own Rel prompts (teacher-forced)
    C.say(f"{name}: cache {kc.dir} (final {kc.is_final}, complete {kc.meta.get('complete')}), T = {T}, rows: {len(probes)} valid probes "
          f"({len(repair)} REPAIR), {len(edits)} edit sequences, {len(hn)} valid hard negatives; own keys {tuple(own[0].shape)}")
    out = {"log": name, "backbone": bb, "T": T, "alpha": C.ALPHA, "n_cal": C.N_CAL, "n_draws": C.N_DRAWS,
           "cache": {"dir": kc.dir, "is_final": kc.is_final, "complete": kc.meta.get("complete"), "index_md5": kc.meta.get("index_md5"),
                     "md5_verified": True},
           "code": C.code_fingerprint(), "rows": info, "n_repair_pool": len(repair), "draws": [], "features": {}, "methods": {}}
    for d in range(C.N_DRAWS):
        ben = splits.sample(repair, C.N_CAL, d)
        out["draws"].append({"seed": d, "calibration_sids": [kc.index[j]["lib_sid"] for j in ben]})
    for feature in ("ncoord", "cos"):
        G0 = C.base_gate(L, feature)
        Gg, grec = add_guards(G0, *own)
        Rg = Gg.guard_scores()                                        # (2T, T): stored solver keys, then the B = 1 Rel keys
        ar = torch.arange(T)
        Qrep = kc.coords(Gg, repair)                                  # (n REPAIR, T): the whole calibration pool on every detector
        top = Qrep.argmax(0)
        relq = {int(kc.index[j]["edit"]): kc.index[j]["prompt"] for j in edits if kc.index[j]["kind"] == "rel"}
        out["features"][feature] = {"add_guards": grec, "n_guards": int(Gg.guard_key.shape[0]),
                                    "own_feature_stored_key": [float(x) for x in Rg[:T][ar, ar]],
                                    "own_feature_b1_rel_key": [float(x) for x in Rg[T:][ar, ar]],
                                    # the REPAIR row with the largest feature on each edit's detector (a near-scope benign query raises b_i)
                                    "repair_max": [float(x) for x in Qrep.max(0).values],
                                    "repair_argmax_sid": [kc.index[repair[int(i)]]["lib_sid"] for i in top],
                                    "repair_argmax_same_question_as_edit": [kc.index[repair[int(i)]]["prompt"] == relq[e] for e, i in enumerate(top)]}
        for mid in C.CALIBRATED:
            if C.METHODS[mid]["feature"] != feature:
                continue
            per = []
            for d in range(C.N_DRAWS):
                ben = splits.sample(repair, C.N_CAL, d)
                Q = kc.coords(Gg, ben)                                # (200, T) feature of every detector at the teacher-forced key
                G, rec = C.calibrate(Gg, mid, d, coords=Q)
                Gu, recu = C.calibrate(Gg, mid, d, coords=Q, allow_silence=True)
                e = {"seed": d, "rec": C.rec_public(rec), "rec_uncapped": C.rec_public(recu),
                     "tau": [float(x) for x in G.det_tau], "tau_uncapped": [float(x) for x in Gu.det_tau],
                     "b": None if G.det_b is None else [float(x) for x in G.det_b],
                     "tf": flags(kc.decide(G, "tf", rows=rows, uids=uids)),
                     "tf_uncapped": flags(kc.decide(Gu, "tf", rows=rows, uids=uids)),
                     "prompt": flags(kc.decide(G, "prompt", rows=rows, uids=uids), full=False),
                     "decode": flags(kc.decide(G, "decode", rows=rows, uids=uids), full=False)}
                per.append(e)
                s = C.gate_stats(info, C.unbits(e["tf"]["fired"]), C.unbits(e["tf"]["own_cand"]), C.unbits(e["tf"]["own_top"]), e["tf"]["top_edit"])
                C.say(f"  {mid} draw {d:2d}: eta {rec['eta']:.4f}, capped edits {rec['n_edits_capped']:3d}; heldout_disjoint firing "
                      f"{s['probe']['heldout_disjoint']['fired']:.2f} %, T-Gen own top-1 {s['edit']['tgen']['own_top']:.0f} %, I-Gen "
                      f"{s['edit']['igen']['own_top']:.0f} %, HN fired {s['hn']['all']['fired']:.1f} % ({time.time() - t0:.0f}s)")
            out["methods"][mid] = {"spec": C.METHODS[mid], "draws": per}
    for mid in C.FIXED:
        G = C.fixed_gate(L, mid)
        e = {"tau": [float(x) for x in G.theta()], "tf": flags(kc.decide(G, "tf", rows=rows, uids=uids)),
             "prompt": flags(kc.decide(G, "prompt", rows=rows, uids=uids), full=False),
             "decode": flags(kc.decide(G, "decode", rows=rows, uids=uids), full=False)}
        out["methods"][mid] = {"spec": C.METHODS[mid], "fixed": e}
        s = C.gate_stats(info, C.unbits(e["tf"]["fired"]), C.unbits(e["tf"]["own_cand"]), C.unbits(e["tf"]["own_top"]), e["tf"]["top_edit"])
        C.say(f"  {mid}: heldout_disjoint firing {s['probe']['heldout_disjoint']['fired']:.2f} %, Rel own top-1 {s['edit']['rel']['own_top']:.0f} %, "
              f"T-Gen {s['edit']['tgen']['own_top']:.0f} %, I-Gen {s['edit']['igen']['own_top']:.0f} %, HN fired {s['hn']['all']['fired']:.1f} %")
    out["seconds"] = time.time() - t0
    out["rescope_version"] = getattr(rescope, "__file__", "")
    path = f"{C.OUT}/gate_{name}.json"
    C.jdump(out, path)
    C.say(f"wrote {path} ({out['seconds']:.0f}s)")


if __name__ == "__main__":
    run(sys.argv[1])
