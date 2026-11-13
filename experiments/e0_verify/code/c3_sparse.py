"""Check 3: cached-key evaluation.  For two gates and one development log, the rows of rescope.SparseEvaluator must equal the
rows of the full rescope.Evaluator on every edit sequence and every probe row (flags and firing).  GPU, library code on both
sides (the independent recomputation of the rows is check 5).

  gq run --mem <MiB> -- python c3_sparse.py Q|L        -> out/c3_<tag>/part_*.json, out/c3_<tag>.json   (resumable)

Gates (built as docs/USAGE.md builds them, calibration draw 0, alpha = 0.05, n = 200):
  G1 = the default method (normalised coordinate, two-block);  G2 = the cosine feature, two-block (capped edits, other detector kind).
Item groups: edits (Rel / T-Gen / I-Gen of all edits), probes (every VALID probe row, all pools), probes_invalid (the probe rows
with an invalid but scorable pre-edit answer), shifted (lp = 'prompt' on every item whose teacher-forced last prompt position
differs from the decode's), hn (valid hard negatives of the final set; run last).
A row is the full dict of the Evaluator (exact, token_acc, n_tok, finite, first_mismatch, D1, tok_loc, every firing flag).
"""
from __future__ import annotations

import os
import sys
import time

import vcommon as V

os.environ.setdefault("RESCOPE_CACHE", V.VER + "/work/hcache")        # the harness' private cache: not inside the package
sys.path.insert(0, V.LIB)
import torch  # noqa: E402

torch.set_num_threads(4)
import rescope  # noqa: E402
from rescope import (HN_DIR_FINAL, Evaluator, SparseEvaluator, add_guards, calibrate_two_block, feature_gate, from_harness, load, splits)  # noqa: E402

CHUNK = 200


def main(tag):
    d = V.DEV[tag]
    name = d["log"]
    odir = f"{V.OUT}/c3_{tag}"
    os.makedirs(odir, exist_ok=True)
    mem = int(os.environ.get("GQ_MEM", d["mem"]))
    cap = (mem - 600) / 1024.0
    t0 = time.time()
    H = load(name, device="cuda:0", mem_gb=cap)
    evS = SparseEvaluator(H, cache="auto", hn_dir=HN_DIR_FINAL)
    evF = Evaluator(H, hn_dir=HN_DIR_FINAL)
    assert evS.cache is not None and evS.cache.is_final
    log = from_harness(H).to(H.dev)
    repair = evS.probe_items("repair")
    own = evS.own_keys("rel")
    gates, recs = {}, {}
    for gname, feature in (("G1_ncoord_two_block", "ncoord"), ("G2_cos_two_block", "cos")):
        S = feature_gate(log, feature, W0=H.W0)
        S, _ = add_guards(S, *own)
        benign = splits.sample(repair, 200, 0)
        S, rec = calibrate_two_block(S, [evS.eligible_keys(it, S) for it in benign], 0.05, seed=0)
        gates[gname] = S
        recs[gname] = {k: rec[k] for k in ("eta", "n1", "n2", "rank", "n_edits_capped", "bound_false_firing", "guarantee")}
    V.say(f"{name}: loaded in {time.time() - t0:.0f}s, checks {H.checks}, gates {recs}")
    groups = {"edits": (evF.edit_items(), "tf"), "probes": (evF.probe_items("all"), "tf")}
    inv = []
    for it in evF.probe_items("all", valid_only=False):
        if not it["y0_valid"]:
            try:
                if int(evF.batch(it)["answer_mask"][0].sum()) > 0:
                    inv.append(it)
            except (RuntimeError, ValueError):
                pass
    groups["probes_invalid"] = (inv, "tf")
    hn = evF.hn_items()
    sh = [it for it in groups["edits"][0] + groups["probes"][0] + hn if evF.lp_shift(it)]
    groups["shifted"] = (sh, "prompt")
    groups["hn"] = (hn, "tf")
    summary = {"log": name, "gates": recs, "n_items": {k: len(v[0]) for k, v in groups.items()}, "groups": {}}
    for gname, (items, lp) in groups.items():
        g = {"n_items": len(items), "n_rows": 0, "n_rows_equal": 0, "differing": [], "n_fired": {k: 0 for k in gates}, "lp": lp}
        for ci, a in enumerate(range(0, len(items), CHUNK)):
            path = f"{odir}/part_{gname}_{ci:03d}.json"
            if os.path.exists(path):
                part = V.jload(path)
            else:
                t1 = time.time()
                st0 = dict(evS.stats)
                chunk = items[a:a + CHUNK]
                rs = V._clean(evS.tf_rows(gates, chunk, lp=lp))
                rf = V._clean(evF.tf_rows(gates, chunk, lp=lp))
                part = {"group": gname, "chunk": ci, "n": len(chunk), "equal": {}, "differing": [], "flags": {}, "sparse_stats": {k: evS.stats[k] - st0[k] for k in st0},
                        "seconds": time.time() - t1}
                for k in gates:
                    eq = [x == y for x, y in zip(rs[k], rf[k])]
                    part["equal"][k] = [sum(eq), len(eq)]
                    part["differing"] += [{"gate": k, "sid": x["sid"], "sparse": x, "full": y} for x, y, e in zip(rs[k], rf[k], eq) if not e][:20]
                    part["flags"][k] = [[x["sid"], bool(x["fired"]), bool(x["exact"]), int(x["top_last"])] for x in rf[k]]
                V.jdump(part, path, indent=None)
                V.say(f"  {gname} chunk {ci}: {part['equal']} ({part['seconds']:.0f}s, sparse forwards {part['sparse_stats']['forwards']}, full {len(chunk) * len(gates)})")
            for k in gates:
                g["n_rows"] += part["equal"][k][1]
                g["n_rows_equal"] += part["equal"][k][0]
                g["n_fired"][k] += sum(f[1] for f in part["flags"][k])
            g["differing"] += part["differing"][:10]
        summary["groups"][gname] = g
        V.say(f"{gname}: {g['n_rows_equal']}/{g['n_rows']} rows equal, fired {g['n_fired']}")
        V.jdump(summary, f"{V.OUT}/c3_{tag}_partial.json")
    H.reset()
    summary.update(sparse_stats=dict(evS.stats), seconds=time.time() - t0, peak_gpu_gb=torch.cuda.max_memory_allocated() / 2 ** 30,
                   module_weight_is_WE_after=bool(torch.equal(H.mod.weight.data, H.WE)), rescope=os.path.dirname(rescope.__file__),
                   n_rows=sum(g["n_rows"] for g in summary["groups"].values()), n_rows_equal=sum(g["n_rows_equal"] for g in summary["groups"].values()))
    V.jdump(summary, f"{V.OUT}/c3_{tag}.json")
    V.say(f"DONE {summary['n_rows_equal']}/{summary['n_rows']} rows equal ({summary['seconds']:.0f}s)")


if __name__ == "__main__":
    main(sys.argv[1])
