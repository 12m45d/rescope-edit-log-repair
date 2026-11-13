"""Extra check X5 (CPU): questions left open by checks 1 - 5.

  python x5_more.py        -> out/x5_more.json

  A  two readings of 'top-1' when edits are CAPPED.  PROTOCOL.md: "edit i fires iff it is the top-1 edit by r_i - b_i and
     r_i - b_i > eta" (argmax over ALL edits, then the threshold).  Library: the candidates (r_i - b_i > eta, or r_i >= cap_i for a
     capped edit) are ranked by the margin and the best CANDIDATE releases.  Without caps the two are the same rule; with caps a
     capped edit can release although an uncapped non-candidate has the larger margin.  Counted here for M1 - M4, 20 draws.
  B  LLaVA edits whose teacher-forced last prompt position is one token after the decode's: at which of the two tokens was the
     stored solver key taken (cosine of the stored key to the cached B = 1 key at either position)?
  C  the counts of the cache summary recounted from the index / rows files with my own definitions.
  D  the source pool of the hard negatives re-derived; every hard-negative row's source against it; `src_related_edit_first_pos`
     of every existing row recomputed.
"""
from __future__ import annotations

import re

import numpy as np

import vcommon as V

import torch  # noqa: E402

METH = {"M1": ("ncoord", "two_block"), "M2": ("ncoord", "shared"), "M3": ("cos", "two_block"), "M4": ("cos", "shared")}
STREAMS = {"qwen25vl": "TB-Q2000-ae-l2_3", "llava": "TB-L1000-ae-l2_3"}


def mid(path):
    return path.split("/")[0] if path else None


def low(s):
    return " ".join((s or "").lower().split())


def part_a(res):
    for tag, d in V.DEV.items():
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
        in_log = lambda e: e["edit"] < T and (e.get("edit_b") is None or e["edit_b"] < T)          # noqa: E731
        rows = [e for e in prow if e["y0_valid"]] + [e for e in idx if e["family"] == fam and e["kind"] in ("rel", "tgen", "igen") and in_log(e)] + \
            [e for e in idx if e["family"] == fam and e["kind"] in ("hn_i", "hn_q", "hn_x") and e["y0_valid"] and in_log(e)]
        X = Ktf[torch.tensor([e["row_tf"] for e in rows])].double().numpy()
        kind = np.array([e["kind"] for e in rows])
        own = np.array([pos.get(e.get("edit_uid"), -1) for e in rows])
        hd = np.array([e["kind"] == "probe" and e["_ann"]["disjoint"] for e in rows])
        rep = [i for i, e in enumerate(rows) if e["kind"] == "probe" and e["_ann"]["pool"] == "repair"]
        rel = sorted([e for e in rows if e["kind"] == "rel"], key=lambda e: e["edit"])
        Xb1 = Ktf[torch.tensor([e["row_tf"] for e in rel])].double().numpy()
        gedit = np.concatenate([np.arange(T), np.arange(T)])
        out = {}
        for mid_, (feature, rule) in METH.items():
            F = V.FEATURES[feature](X, W, K)
            Gm = np.concatenate([V.FEATURES[feature](K, W, K), V.FEATURES[feature](Xb1, W, K)])
            acc = {k: [] for k in ("n_capped", "rows_differing", "heldout_fire_lib", "heldout_fire_literal", "tgen_cov_lib", "tgen_cov_literal", "igen_cov_lib", "igen_cov_literal",
                                   "rel_own_released_lib", "rel_own_released_literal", "hn_fire_lib", "hn_fire_literal")}
            for dr in range(20):
                cal = [rep[i] for i in V.draw(len(rep), 200, dr)]
                if rule == "two_block":
                    i1, i2 = V.halves(200, dr)
                    b, eta, _ = V.naive_two_block(F[cal], i1, i2, "1/20")
                else:
                    b, eta, _ = V.naive_shared(F[cal], "1/20")
                cap, _ = V.naive_caps(Gm, gedit, b, eta)
                fired, top, cand = V.naive_decide(F, b, eta, cap)                       # library reading: best candidate
                M = F - b[None, :]
                j = M.argmax(1)                                                         # literal reading: argmax over all edits, then its own threshold
                ar = np.arange(len(rows))
                fl = cand[ar, j]
                tl = np.where(fl, j, -1)
                has = own >= 0
                ot, otl = fired & has & (top == own), fl & has & (tl == own)
                acc["n_capped"].append(int(np.isfinite(cap).sum()))
                acc["rows_differing"].append(int(((fired != fl) | (top != tl)).sum()))
                for nm, f_, t_ in (("lib", fired, ot), ("literal", fl, otl)):
                    acc[f"heldout_fire_{nm}"].append(100.0 * f_[hd].mean())
                    acc[f"tgen_cov_{nm}"].append(100.0 * t_[kind == "tgen"].mean())
                    acc[f"igen_cov_{nm}"].append(100.0 * t_[kind == "igen"].mean())
                    acc[f"rel_own_released_{nm}"].append(100.0 * t_[kind == "rel"].mean())
                    acc[f"hn_fire_{nm}"].append(100.0 * f_[np.isin(kind, ("hn_i", "hn_q", "hn_x"))].mean())
            out[mid_] = {k: float(np.mean(v)) for k, v in acc.items()}
            out[mid_]["n_rows"] = len(rows)
            out[mid_]["max_rows_differing_in_a_draw"] = int(max(acc["rows_differing"]))
        res.setdefault("A_top1_readings", {})[tag] = out
        V.say(f"A {name}: " + "; ".join(f"{m}: capped {v['n_capped']:.1f}, rows differing {v['rows_differing']:.1f}, held-out fire {v['heldout_fire_lib']:.2f} vs {v['heldout_fire_literal']:.2f}, "
                                         f"T-Gen {v['tgen_cov_lib']:.1f} vs {v['tgen_cov_literal']:.1f}, Rel own released {v['rel_own_released_lib']:.1f} vs {v['rel_own_released_literal']:.1f}" for m, v in out.items()))


def part_b(res):
    idx = V.read_index("llava")
    Ktf = V.read_keys("llava", "tf")
    log = V.read_log(STREAMS["llava"])
    out = []
    for e in idx:
        if e["kind"] == "rel" and e.get("lp_shift"):
            k = log["K"][e["edit"]]
            c = lambda r: float(np.dot(k, Ktf[r].double().numpy()) / (np.linalg.norm(k) * np.linalg.norm(Ktf[r].double().numpy())))      # noqa: E731
            out.append({"edit": e["edit"], "target": e["answer"], "cos_stored_key_vs_key_at_lp_tf": c(e["row_tf"]), "cos_stored_key_vs_key_at_decode_token": c(e["row_tf_prompt"])})
    res["B_shifted_edits_solver_key_position"] = {"edits": out, "all_written_at_the_teacher_forced_position": all(x["cos_stored_key_vs_key_at_lp_tf"] > 0.99 > x["cos_stored_key_vs_key_at_decode_token"] for x in out)}
    V.say(f"B: {res['B_shifted_edits_solver_key_position']}")


def bare_none_no(s):
    return re.sub(r"[^a-z]", "", (s or "").lower()) in ("none", "no")


def part_c(res):
    out = {}
    for bb in ("qwen25vl", "llava"):
        idx = V.read_index(bb)
        tab = {}
        for kind in ("rel", "tgen", "igen", "probe", "hn_i", "hn_q", "hn_x", "benign"):
            sub = [e for e in idx if e["kind"] == kind]
            fams = {}
            for e in sub:
                fams[e["family"]] = fams.get(e["family"], 0) + 1
            wa = [e for e in sub if e.get("y0_text") is not None or e.get("y0_valid") is not None]
            tab[kind] = {"rows_by_family": fams, "with_y0": len(wa), "valid": sum(bool(e["y0_valid"]) for e in wa), "dup": sum(e["n_same_query"] > 1 for e in sub),
                         "shifted": sum(bool(e.get("lp_shift")) for e in sub), "bare_none_no": sum(bare_none_no(e.get("y0_text")) for e in wa if e.get("y0_text")),
                         "empty": sum(e.get("y0_text") is not None and not e["y0_text"].strip() for e in wa), "single_char": sum(len((e.get("y0_text") or "").strip()) == 1 for e in wa)}
        tab["all"] = {"rows": len(idx), "with_y0": sum(v["with_y0"] for v in tab.values()), "valid": sum(v["valid"] for v in tab.values()), "dup": sum(v["dup"] for v in tab.values()),
                      "shifted": sum(v["shifted"] for v in tab.values())}
        hn = V.jsonl(f"{V.CACHE}/hard_negatives/{bb}/rows.jsonl")
        new_i = [r for r in hn if r["hn_type"] == "HN-I" and r.get("new")]
        old_i = [r for r in hn if r["hn_type"] == "HN-I" and not r.get("new")]
        tab["hn_file"] = {"n": len(hn), "existing": sum(not r.get("new") for r in hn), "new": sum(bool(r.get("new")) for r in hn),
                          "new_HN-I_same_category": [sum(bool(r["same_category"]) for r in new_i), len(new_i)], "new_HN-I_related_flag": [sum(bool(r["related_flag"]) for r in new_i), len(new_i)],
                          "existing_HN-I_related_flag": [sum(bool(r["related_flag"]) for r in old_i), len(old_i)],
                          "existing_rows_with_source_related_to_a_stream_edit": sum(r.get("src_related_edit_first_pos") is not None for r in hn if not r.get("new")),
                          "smallest_such_position": min([r["src_related_edit_first_pos"] for r in hn if not r.get("new") and r.get("src_related_edit_first_pos") is not None] or [None])}
        out[bb] = tab
        V.say(f"C {bb}: " + "; ".join(f"{k} {v}" for k, v in tab.items()))
    res["C_cache_counts"] = out


def part_d(res):
    train = {d["uid"]: d for d in V.vlkeb("train")}
    evl = V.vlkeb("eval")
    ev = {d["uid"]: d for d in evl}
    # every edit of the project's VLKEB streams used by the caches (both backbones)
    streams = {bb: [train[r["uid"]] for r in V.stream_items(nm)] for bb, nm in STREAMS.items()}
    allitems = [d for its in streams.values() for d in its]
    keys = {(d["image"], d["src"], d["alt"]) for d in allitems}
    mids = {m for d in allitems for m in (mid(d["image"]), mid(d["image_rephrase"]))}
    names = {n for d in allitems for n in (low(d["alt"]), low(d["pred"].replace("_", " ")))}
    out = {}
    for bb in ("qwen25vl", "llava"):
        idx = V.read_index(bb)
        probe_uids = {e["uid"] for e in idx if e["kind"] == "probe" and e["family"] in ("Q", "L")}
        pool = [d for d in evl if d["uid"] not in probe_uids and (d["image"], d["src"], d["alt"]) not in keys
                and not ({mid(d["image"]), mid(d["image_rephrase"])} & mids) and not ({low(d["alt"]), low(d["pred"].replace("_", " "))} & names)]
        pu = {d["uid"] for d in pool}
        hn = V.jsonl(f"{V.CACHE}/hard_negatives/{bb}/rows.jsonl")
        new = [r for r in hn if r.get("new") and r["hn_type"] != "HN-X"]
        its = streams[bb]
        first_mid, first_name, first_key = {}, {}, {}
        for p, d in enumerate(its):
            for m in (mid(d["image"]), mid(d["image_rephrase"])):
                first_mid.setdefault(m, p)
            for n in (low(d["alt"]), low(d["pred"].replace("_", " "))):
                first_name.setdefault(n, p)
            first_key.setdefault((d["image"], d["src"], d["alt"]), p)
        diff, n_old, rel_n = [], 0, 0
        for r in hn:
            if r.get("new") or r["hn_type"] == "HN-X":
                continue
            n_old += 1
            s = ev[r["src_uid"]]
            c = [first_mid[m] for m in (mid(s["image"]), mid(s["image_rephrase"])) if m in first_mid] + [first_name[n] for n in (low(s["alt"]), low(s["pred"].replace("_", " "))) if n in first_name]
            if (s["image"], s["src"], s["alt"]) in first_key:
                c.append(first_key[(s["image"], s["src"], s["alt"])])
            mine = min(c) if c else None
            rel_n += int(mine is not None)
            if mine != r.get("src_related_edit_first_pos"):
                diff.append([r["hn_id"], mine, r.get("src_related_edit_first_pos")])
        out[bb] = {"my_source_pool_size": len(pool), "n_new_rows": len(new), "new_rows_whose_source_is_outside_my_pool": sum(r["src_uid"] not in pu for r in new),
                   "n_distinct_sources_of_new_rows": len({r["src_uid"] for r in new}), "n_existing_rows": n_old, "existing_rows_related_to_a_stream_edit_mine": rel_n,
                   "existing_rows_first_related_position_differing_from_file": len(diff), "differing": diff[:5],
                   "new_rows_related_to_an_edit_of_the_own_stream_mine": sum(bool(({mid(ev[r["src_uid"]]["image"]), mid(ev[r["src_uid"]]["image_rephrase"])} & set(first_mid))
                                                                                  or ({low(ev[r["src_uid"]]["alt"]), low(ev[r["src_uid"]]["pred"].replace("_", " "))} & set(first_name))) for r in new)}
        V.say(f"D {bb}: {out[bb]}")
    res["D_hard_negative_sources"] = out


def main():
    res = {}
    part_a(res)
    part_b(res)
    part_c(res)
    part_d(res)
    V.jdump(res, f"{V.OUT}/x5_more.json")


if __name__ == "__main__":
    main()
