"""E2 part e: the running example of Section 2 (Example ex:running), written to 10_final/out/running_example.json.

Runs inside job.py (block 'e', Qwen development log): J is the job module (model handle, session, capture, harness).
RULES, fixed before any outcome was looked at (task text):
  complaint   the first IMAGE-probe complaint in the seeded complaint order of part a (seed 0) whose cached pre-edit answer is
              correct with respect to the benchmark reference (record field pre_correct_at10: the 10-token decoder-v2 answer
              contains the reference) and at most six words long;
  edit        the edit that the default calibrated method (rescope, alpha 0.05, n 200, draw 0) releases on it, if any; otherwise
              the edit with the largest normalised coordinate on it (all positions of the teacher-forced pre-edit trace);
  why-not     the text rephrase (T-Gen prompt) of the first edit, starting with the example's edit and then in stream order,
              whose own coordinate at the last prompt position is below its calibrated threshold under the same default
              method and on which no edit is released (the answer is then the pre-edit model's).
A key that cannot be produced as defined is null, with the reason under _null.
"""
from __future__ import annotations

import json
import os
import time

import torch

import e2c
from e2c import fx

PATH = os.path.join(e2c.FIN, "out", "running_example.json")
MODEL = {"qwen25vl": "Qwen2.5-VL-3B", "llava": "LLaVA-1.5-7B"}


def run(J):
    L, H, ses, hz = J.L, J.H, J.ses, J.hz
    if L.name != "TB-Q100-ae-l2_3":
        J.say("running example: defined on TB-Q100-ae-l2_3 only; skipped")
        return
    if J.SM:
        J.say("running example: not run in a smoke job")
        return
    null, extra = {}, {}
    stream = fx.draws.complaint_stream(L, 0)
    j = pos = None
    for k, jj in enumerate(stream):
        e, r = L.index[jj], L.probe_rec(jj)
        if e["ptype"] == "image" and r.get("pre_correct_at10") and len(str(e["y0_text"]).split()) <= 6:
            j, pos = jj, k
            break
    if j is None:
        raise RuntimeError("no complaint satisfies the selection rule")
    e, prec = L.index[j], L.probe_rec(j)
    cap = J.cap_ix()
    rmax = cap["c"]["Rmax"][cap["ix"][j]]
    rlast = cap["c"]["Rlast"][cap["ix"][j]]
    # the edit
    M0 = fx.methods.rescope(L, alpha=0.05, n=200, draw=0)
    d0 = ses.decide(M0, [j], "tf")
    fired0 = bool(d0["fired"][0])
    edit = int(d0["top_edit"][0]) if fired0 else int(rmax.argmax())
    je = L.edit_rows["rel"][edit]
    # answers (driver: decoder-v2 generation of the deployed log; the pre-edit answer is the cached one)
    gen_dep = ses.generate(J.DEP, [j])[0]
    dep_answer = gen_dep.get("y")
    extra["deployed_answer_record_yE_text"] = prec.get("yE_text")
    extra["deployed_answer_48_tokens_first_line"] = gen_dep.get("line")
    # rollback: the per-complaint greedy first-order search of the existing records (P12 cache, budget 10)
    rb = fx.io.jload(os.path.join(e2c.P12, "cache", f"rbfo_{L.name}.json")).get(e["group"])
    n_del = broken = None
    if rb is None:
        null["rollback"] = "the existing records hold no per-complaint rollback set for this complaint"
    else:
        R = [int(i) for i in rb["R"]]
        extra["rollback"] = {"R": R, "resolved_internal_in_record": bool(rb["resolved_internal"]), "source": "p12_linear_streams/cache/rbfo_<log>.json (harness ROLLBACK_FO, budget 10)"}
        Mr = fx.methods.rollback(L, H, R)
        rc = ses.tf(Mr, [j])[0]
        extra["rollback"]["complaint_reproduced_after_rollback_driver"] = not rc["D1"]
        er, ed = ses.edits(Mr), ses.edits(J.DEP)
        dep = {(r["metric"], r["edit"]): bool(r["exact"]) for r in ed}
        kept = [r for r in er if r["edit"] not in set(R)]
        b_rel = sum(dep[(r["metric"], r["edit"])] and not r["exact"] for r in kept if r["metric"] == "rel")
        extra["rollback"]["kept_edit_checks_broken_rel_tgen_igen"] = sum(dep[(r["metric"], r["edit"])] and not r["exact"] for r in kept)
        extra["rollback"]["n_kept_edits"] = L.T - len(R)
        extra["rollback"]["n_in_removal_set"], extra["rollback"]["kept_edits_broken_rel"] = len(R), int(b_rel)
        if rb["resolved_internal"]:
            n_del, broken = len(R), int(b_rel)
        else:
            null["rollback"] = ("the greedy search of the existing record did not resolve this complaint within its budget of 10 edits: no rollback set was found; "
                                "_details.rollback holds the set at the budget and what it breaks")
    # compensation: COMP-B of this one complaint (options of P12)
    cs = [c for c in J.p12_stream(0) if c["group"] == e["group"]]
    corr, _ = hz.pkg.repair("COMP", H, cs, protect=("K_ans",), query="answer", form="adapter")
    Mc = fx.methods.wrap(L, corr, "comp_example", {"harness": "COMP", "options": {"protect": ["K_ans"], "query": "answer", "form": "adapter"}}, name="COMP_B", n=1)
    hd = L.pool("heldout_disjoint")
    rc, rd = ses.tf(Mc, hd + [j]), ses.tf(J.DEP, hd + [j])
    ch_flags = sum(bool(a["D1"]) != bool(b["D1"]) for a, b in zip(rc[:-1], rd[:-1]))
    ch_tok = sum((a["exact"], a["first_mismatch"], round(a["token_acc"], 6)) != (b["exact"], b["first_mismatch"], round(b["token_acc"], 6)) for a, b in zip(rc[:-1], rd[:-1]))
    extra["compensation"] = {"complaint_reproduced_after_compensation": not rc[-1]["D1"], "heldout_D1_flags_changed": int(ch_flags),
                             "heldout_rows_with_changed_token_outcome": int(ch_tok), "fixed": sum(b["D1"] and not a["D1"] for a, b in zip(rc[:-1], rd[:-1])),
                             "lost": sum(a["D1"] and not b["D1"] for a, b in zip(rc[:-1], rd[:-1])), "rank": corr.meta.get("rank_kept"),
                             "definition": "heldout_changed = heldout_disjoint rows whose teacher-forced outcome on the pre-edit answer (exact flag, first mismatch, token accuracy) "
                                           "differs between the deployed log and the compensated model"}
    # "answers changed": the generated answers (decoder v2, greedy, 10 tokens) of the held-out queries under the compensated model
    # against those under the deployed log
    gc_, gd_ = ses.generate(Mc, hd), ses.generate(J.DEP, hd)
    ch_gen = sum(a.get("y") != b.get("y") for a, b in zip(gc_, gd_))
    extra["compensation"]["heldout_generated_answers_changed"] = int(ch_gen)
    extra["compensation"]["definition"] = ("heldout_changed = heldout_disjoint rows whose generated answer (decoder v2, greedy, 10 tokens) under the compensated model differs "
                                           "from the answer under the deployed log; teacher-forced counts are listed next to it")
    del corr
    # the same for the compensation that restores the complaint at ALL positions of its pre-edit trace (COMP-B-ALLPOS of P12: query='all')
    corr2, _ = hz.pkg.repair("COMP", H, cs, protect=("K_ans",), query="all", form="adapter")
    Mc2 = fx.methods.wrap(L, corr2, "comp_allpos_example", {"harness": "COMP", "options": {"protect": ["K_ans"], "query": "all", "form": "adapter"}}, name="COMP_B_ALLPOS", n=1)
    r2 = ses.tf(Mc2, hd + [j])
    g2 = ses.generate(Mc2, hd)
    extra["compensation_all_positions"] = {"complaint_reproduced_after_compensation": not r2[-1]["D1"], "rank": corr2.meta.get("rank_kept"),
                                           "heldout_generated_answers_changed": sum(a.get("y") != b.get("y") for a, b in zip(g2, gd_)),
                                           "heldout_D1_flags_changed": sum(bool(a["D1"]) != bool(b["D1"]) for a, b in zip(r2[:-1], rd[:-1])),
                                           "what": "the compensation with query='all' (restores the module output at every position of the complaint's pre-edit trace); "
                                                   "compensation.heldout_changed refers to COMP-B (query='answer', K_ans protected: the registered compensation, which restores "
                                                   "the answer positions only)"}
    del corr2
    # rescoping: least repair of this one complaint from the deployed scoping
    Mw = fx.methods.where_least(L, [j], H, start="deployed", seed=0, label="where_least_example")
    ew = ses.edits(Mw, metrics=("rel",))
    rw = ses.tf(Mw, [j])[0]
    own = sum(bool(r["own_fired_last"]) for r in ew)
    extra["rescope"] = {"refused": Mw.rec["least_repair"]["n_refused"], "complaint_fires_after_repair": bool(rw["fired"]), "complaint_reproduced_after_repair": not rw["D1"],
                        "n_thresholds_raised": Mw.rec["least_repair"]["n_thresholds_raised_total"], "rel_exact_after": sum(bool(r["exact"]) for r in ew),
                        "n_edits_candidate_on_query_before_repair": int(torch.isfinite(rmax).sum())}
    if Mw.rec["least_repair"]["n_refused"]:
        null["rescope.note"] = "the library refused the least repair of this complaint (it would silence an own request); own_requests_firing is the unrepaired count"
    # why-not pair under the default calibrated method
    th = M0.obj.theta()
    tg = L.edit_rows["tgen"]
    order = [edit] + [i for i in range(L.T) if i != edit]
    dd = ses.decide(M0, [tg[i] for i in order], "tf")
    F = fx.gate.features(L, M0, [tg[i] for i in order], "tf")
    wn = None
    for k, i in enumerate(order):
        if not bool(dd["own_cand"][k]) and not bool(dd["fired"][k]):
            wn = (i, float(F[k, i]), float(th[i]))
            break
    why = {"query": None, "answer": None, "coordinate": None, "threshold": None}
    if wn is None:
        null["whynot"] = "no text rephrase with the own coordinate below its threshold and no released edit"
    else:
        i, co, t = wn
        g = ses.generate(M0, [tg[i]])[0]
        why = {"query": L.index[tg[i]]["prompt"], "answer": g.get("y"), "coordinate": co, "threshold": t}
        extra["whynot"] = {"edit": i, "is_the_example_edit": i == edit, "edit_question": L.index[L.edit_rows["rel"][i]]["prompt"], "edit_target": L.targets[i],
                           "answer_first_line_48_tokens": g.get("line"), "fired_in_decode": g.get("fired"), "pre_edit_answer_of_the_prompt": L.index[tg[i]].get("prompt_y0_text")}
    out = {"log": L.name, "model": MODEL[L.backbone], "T": L.T,
           "edit": {"question": L.index[je]["prompt"], "target": L.targets[edit]},
           "complaint": {"query": e["prompt"], "pre_answer": e["y0_text"], "deployed_answer": dep_answer, "max_coordinate": float(rmax.max())},
           "rollback": {"n_deleted": n_del, "edits_broken": broken},
           "compensation": {"heldout_changed": int(ch_gen)},
           "heldout": {"n": len(hd)},
           "rescope": {"own_requests_firing": int(own)},
           "whynot": why,
           "_null": null,
           "_selection": {"rule": __doc__, "complaint_sid": e["sid"], "group": e["group"], "stream_position_seed0": pos, "reference_answer": prec.get("ref_answer"),
                          "image": e.get("image"), "edit_index": edit, "edit_chosen_by": "released by the default calibrated method (draw 0)" if fired0 else
                          "largest normalised coordinate over all positions (the default calibrated method releases no edit on the complaint)",
                          "max_coordinate_at_last_prompt_position": float(rlast.max()), "coordinate_of_the_example_edit_all_positions": float(rmax[edit]),
                          "n_positions": cap["c"]["n_pos"][cap["ix"][j]]},
           "_details": extra,
           "_definitions": {"complaint.max_coordinate": "largest normalised coordinate of any edit at any position of the teacher-forced pre-edit trace of the query",
                            "rollback.n_deleted": "size of the removal set of the per-complaint greedy first-order rollback of the existing records (budget 10)",
                            "rollback.edits_broken": "kept edits (not in the removal set) whose Rel check is exact under the deployed log and not after the rollback (driver, B = 1)",
                            "compensation.heldout_changed": extra["compensation"]["definition"], "heldout.n": "valid heldout_disjoint rows",
                            "rescope.own_requests_firing": "edits whose own Rel prompt still releases the edit at the last prompt position after the least repair of this one complaint "
                                                           "from the deployed scoping (hook flags)",
                            "whynot.coordinate / threshold": "own normalised coordinate of the edit at the last prompt position of its text rephrase and the edit's calibrated "
                                                             "threshold (default method, alpha 0.05, n 200, draw 0); whynot.answer: the answer generated under that method"},
           "written": time.strftime("%Y-%m-%d %H:%M:%S"), "code": fx.code_fingerprint()}
    fx.io.jdump(out, PATH, indent=1)
    fx.io.jdump(out, os.path.join(e2c.OUT, "running_example.json"), indent=1)
    J.say("running example written: " + json.dumps({k: out[k] for k in ("complaint", "rollback", "compensation", "rescope", "whynot")})[:600])
