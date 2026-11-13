"""Lean review, further CPU checks (development logs only for every number):
  a. hn_unseen_foreign (clarification (2) of 09:03) against an own count from the raw recombination rows
  b. query-level overlap of BEN-W (statistics / null-repair keys) with BEN-T, BEN-S, REPAIR, HELD-OUT, MME (a nulled key is a test key?)
  c. the MME split by image
  d. recombination partners: X-I / X-Q source items are neither edits nor probe items of the cache
  e. mean false firing on heldout_disjoint over the 20 draws, as run and literal (no cap), for the default method and the gates whose
     detectors depend on a sample (rescope_null, cos_null, rescope_nullW) and the _x form: against the bound 4.95 %
    python rl_cpu3.py <development log>     -> out/cpu3_<log>.json
"""
import json
import sys

import rl_common as C
import numpy as np
import fx

NAME = sys.argv[1]
L = fx.logs.open(NAME)
out = {"log": NAME}
nq = lambda s: (s or "").strip().lower()          # noqa: E731
# ---- a
rc = [json.loads(x) for x in open(f"{C.FIN}/cache2/recomb/{L.backbone}/rows.jsonl") if x.strip()]
pos = {u: i for i, u in enumerate(L.uids)}
use = [r for r in rc if r["edit_uid"] in pos and (r["kind"] != "X-X" or r["edit_b_uid"] in pos)]
qs, ims = {nq(r["prompt"]) for r in use}, {r["image_md5"] for r in use}
partners = {pos[r["edit_b_uid"]] for r in use if r["kind"] == "X-X"}
hn = [int(j) for j in L.rows("hn", valid=True)]
mine = {"HN-I": [0, 0], "HN-Q": [0, 0], "HN-X": [0, 0]}
keep_m = {}
for j in hn:
    e = L.index[j]
    t = e["hn_type"]
    ok = nq(e["prompt"]) not in qs if t == "HN-I" else (e["image_md5"] not in ims if t == "HN-Q" else pos.get(e.get("edit_b_uid"), -1) not in partners)
    keep_m[j] = ok
    mine[t][0] += 1
    mine[t][1] += int(ok)
d = fx.review1.hn_unseen_foreign(L)
out["a_unseen_foreign"] = {"mine_all_kept": mine, "driver": d["n"], "rows_differ": sum(keep_m[j] != d["keep"][j] for j in hn), "n_recombination_rows_used": len(use),
                           "driver_n_rows": d["n_recombination_rows"]}
# ---- b
P = {p: fx.pools.get(L, p) for p in ("BEN-W", "BEN-S", "BEN-T")}
qid = lambda rows: {L.index[j]["query_id"] for j in rows}            # noqa: E731
qw = qid(P["BEN-W"].rows)
imq = lambda rows: {(L.index[j].get("image_md5"), nq(L.index[j]["prompt"])) for j in rows}        # noqa: E731
iw = imq(P["BEN-W"].rows)
rep, hd, ho = L.pool("repair", valid=True), L.pool("heldout_disjoint"), L.pool("heldout")
mm = fx.pools.get(L, "MME-T")
mmq = {(mm.index[j].get("image_md5"), nq(mm.index[j]["prompt"])) for j in mm.rows}
out["b_benw_overlap"] = {"query_id": {"BEN-T": len(qw & qid(P["BEN-T"].rows)), "BEN-S": len(qw & qid(P["BEN-S"].rows)), "repair": len(qw & qid(rep)), "heldout": len(qw & qid(ho))},
                         "image_md5_and_question": {"BEN-T": len(iw & imq(P["BEN-T"].rows)), "BEN-S": len(iw & imq(P["BEN-S"].rows)), "repair": len(iw & imq(rep)), "heldout": len(iw & imq(ho)), "MME-T": len(iw & mmq)},
                         "BEN-T_vs_repair_query": len(qid(P["BEN-T"].rows) & qid(rep)), "BEN-T_vs_heldout_query": len(qid(P["BEN-T"].rows) & qid(ho)),
                         "BEN-S_vs_BEN-T_query": len(qid(P["BEN-S"].rows) & qid(P["BEN-T"].rows)), "BEN-S_vs_heldout_query": len(qid(P["BEN-S"].rows) & qid(ho)),
                         "n": {k: len(v.rows) for k, v in P.items()}}
# ---- c
mc = fx.pools.get(L, "MME-C", "decode")
mt = fx.pools.get(L, "MME-T", "decode")
img = lambda Pl: {Pl.index[j]["image_md5"] for j in Pl.rows}         # noqa: E731
out["c_mme_split"] = {"rows": [len(mc.rows), len(mt.rows)], "images": [len(img(mc)), len(img(mt))], "images_in_both_parts": len(img(mc) & img(mt)),
                      "valid_rows": [len(fx.pools.get(L, "MME-C").rows), len(fx.pools.get(L, "MME-T").rows)],
                      "questions_shared_with_cache": len({nq(mc.index[j]["prompt"]) for j in mc.rows + mt.rows} & {nq(e["prompt"]) for e in L.index}),
                      "images_shared_with_cache": len((img(mc) | img(mt)) & {e.get("image_md5") for e in L.index})}
# ---- d
allidx = L.index
fam_edit_uids = {e["edit_uid"] for e in allidx if e.get("kind") == "rel"}
probe_uids = {e.get("item_uid") for e in allidx if e.get("kind") == "probe"}
edit_mids = {(e.get("image") or "").split("/")[0] for e in allidx if e.get("kind") in ("rel", "igen") and e.get("family") == L.family}
xi = [r for r in rc if r["kind"] == "X-I"]
xq = [r for r in rc if r["kind"] == "X-Q"]
out["d_partner_items"] = {"n_xi": len(xi), "n_xq": len(xq), "source_is_an_edit": sum(r["src_uid"] in fam_edit_uids for r in xi + xq), "source_is_a_probe_item": sum(r["src_uid"] in probe_uids for r in xi + xq),
                          "xq_partner_image_entity_is_an_edit_entity": sum(r["image"].split("/")[0] in edit_mids for r in xq), "sources_from_eval_split": sum(r["src_uid"].startswith("eval:") for r in xi + xq),
                          "n_distinct_sources": len({r["src_uid"] for r in xi + xq})}
C.dump(f"cpu3_{NAME}.json", out)
print({k: v for k, v in out.items() if k != "log"}, flush=True)
# ---- e
hdv = [int(j) for j in L.rows("probe", pool="heldout_disjoint", valid=True)]
res = {}
for nm in ("rescope", "rescope_null", "rescope_nullW", "rescope_x", "cosine", "cos_null", "cos_centred", "whitecos_0.1", "retr_sum", "retr_min"):
    run, lit, cap = [], [], []
    for dr in range(20):
        M = fx.methods.build(nm, L, alpha=0.05, draw=dr)
        Ml = fx.methods.build(nm, L, alpha=0.05, draw=dr, allow_silence=True)
        run.append(100 * float(np.mean(fx.gate.decide(L, M, hdv, "tf")["fired"])))
        lit.append(100 * float(np.mean(fx.gate.decide(L, Ml, hdv, "tf")["fired"])))
        cap.append(fx.methods.n_capped(M))
    res[nm] = {"as_run_mean": round(float(np.mean(run)), 2), "literal_mean": round(float(np.mean(lit)), 2), "literal_se_over_draws": round(float(np.std(lit, ddof=1) / np.sqrt(20)), 2),
               "literal_max": round(float(np.max(lit)), 2), "capped_mean": round(float(np.mean(cap)), 1), "draw0_as_run": round(run[0], 2)}
    print(nm, res[nm], flush=True)
out["e_false_firing_20_draws"] = {"bound": 100 * 5 / 101, "n_rows": len(hdv), "per_method": res}
C.dump(f"cpu3_{NAME}.json", out)
