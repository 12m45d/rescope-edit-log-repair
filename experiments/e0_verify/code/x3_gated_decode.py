"""Extra check X3: free generation THROUGH A GATE (the decode path of the hook: position bookkeeping across the 10 full forwards
and the KV-cache continuation), library against my own code.  GPU.

  gq run --mem <MiB> -- python x3_gated_decode.py Q|L        -> out/x3_<tag>.json

Gate: the default method, calibration draw 0 (library: built on the GPU as docs/USAGE.md does; mine: the naive calibration of
c5_behaviour.setup on the cached keys).  Items (seeded): 30 edit prompts (Rel / T-Gen / I-Gen), 30 valid HELD-OUT-DISJOINT probe
rows, 20 valid hard negatives.  Library: ev.generate(S, items).  Mine: my decoder v2 on the model at MY W_0 with my replay hook at
the prompt's last token (it acts in every forward that contains that position; a KV-cache step does not contain it).
Compared: the generated texts (10-token answer, 48-token text) and whether an edit fired during the decode.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

import vcommon as V
import vgpu as G
import c5_behaviour as C5

os.environ.setdefault("RESCOPE_CACHE", V.VER + "/work/hcache")
import torch  # noqa: E402
from rescope import HN_DIR_FINAL, SparseEvaluator, add_guards, calibrate_two_block, feature_gate, from_harness, load, splits  # noqa: E402

SEED = 20261111


def main(tag, shifted=False):
    d = V.DEV[tag]
    name, bb = d["log"], d["backbone"]
    mem = int(os.environ.get("GQ_MEM", d["mem"]))
    t0 = time.time()
    H = load(name, device="cuda:0", mem_gb=(mem - 600) / 1024.0)
    ev = SparseEvaluator(H, cache="auto", hn_dir=HN_DIR_FINAL)
    log = from_harness(H).to(H.dev)
    S = feature_gate(log, "ncoord", W0=H.W0)
    S, _ = add_guards(S, *ev.own_keys("rel"))
    benign = splits.sample(ev.probe_items("repair"), 200, 0)
    S, rec = calibrate_two_block(S, [ev.eligible_keys(it, S) for it in benign], 0.05, seed=0)
    st = C5.setup(tag)
    g = st["gates"]["M1"]
    lg = st["log"]
    values = G.rebased_values(lg["U"], lg["W"], lg["K"], lg["batch"])
    rng = np.random.default_rng([SEED, 3])
    mine_items = []
    for grp, k in (("edits", 30), ("probes", 30), ("hn", 20)):
        sub = [it for it in st["items"] if it["group"] == grp]
        mine_items += [sub[int(i)] for i in sorted(rng.choice(len(sub), k, replace=False))]
    if shifted:                                                      # every evaluated sequence whose teacher-forced last prompt position is not the decode's
        mine_items = [it for it in st["items"] if st["index"][it["sid"]].get("lp_shift")]
    lib_all = {it["sid"]: it for it in ev.edit_items() + ev.probe_items("heldout_disjoint") + ev.hn_items()}
    lib = V._clean(ev.generate(S, [lib_all[it["sid"]] for it in mine_items]))
    V.say(f"{name}: library generation done ({time.time() - t0:.0f}s)")
    Wc, _, _ = G.checkpoint_weight(bb, d["layer"])
    H.mod.weight.data.copy_(Wc.to(H.dev, H.dtype))
    rows = []
    for it, l in zip(mine_items, lib):
        bp = G.build(H.vlm, it["q"])
        with G.Replay(H.mod, "ncoord", lg["W"], lg["K"], g["b"], g["eta"], g["cap"], values) as rp:
            rp.begin(int(bp["last_prompt_idx"][0]))
            dd = G.my_decode(H.vlm, bp)
        rows.append({"sid": it["sid"], "group": it["group"], "fired_mine": rp.fired, "fired_lib": bool(l["fired"]), "long_equal": dd["long"] == l["long"], "y_equal": dd["y"] == l["y"],
                     "top_mine": rp.top, "top_lib": l["top_last"], "n_fire_calls_lib": l["n_fire_calls"], "pos_ok_lib": l.get("pos_ok"), "lp_shift": l.get("lp_shift"),
                     "mine": dd["long"], "lib": l["long"]})
    H.reset()
    res = {"log": name, "eta_lib": rec["eta"], "eta_mine": g["eta"], "n": len(rows), "fired_equal": sum(r["fired_mine"] == r["fired_lib"] for r in rows),
           "top_equal": sum(r["top_mine"] == r["top_lib"] for r in rows),
           "long_equal": sum(r["long_equal"] for r in rows), "y_equal": sum(r["y_equal"] for r in rows), "n_fired_lib": sum(r["fired_lib"] for r in rows),
           "pos_ok_lib": sum(bool(r["pos_ok_lib"]) for r in rows), "n_shifted": sum(bool(r["lp_shift"]) for r in rows),
           "by_group": {k: [sum(r["long_equal"] and r["fired_mine"] == r["fired_lib"] for r in rows if r["group"] == k), sum(r["group"] == k for r in rows),
                            sum(r["fired_lib"] for r in rows if r["group"] == k)] for k in ("edits", "probes", "hn")},
           "differing": [r for r in rows if not (r["long_equal"] and r["fired_mine"] == r["fired_lib"])][:10], "seconds": time.time() - t0,
           "module_weight_is_WE_after": bool(torch.equal(H.mod.weight.data, H.WE))}
    res["pass"] = res["fired_equal"] == res["long_equal"] == res["y_equal"] == res["n"]
    res["own_edit_fired_in_decode_of_shifted_rel"] = [[r["sid"], r["fired_lib"]] for r in rows if r["sid"].startswith("edit:rel") and r["lp_shift"]]
    V.jdump(res, f"{V.OUT}/x3{'s' if shifted else ''}_{tag}.json")
    V.say(f"{ {k: v for k, v in res.items() if k != 'differing'} }")


if __name__ == "__main__":
    main(sys.argv[1], len(sys.argv) > 2 and sys.argv[2] == "shifted")
