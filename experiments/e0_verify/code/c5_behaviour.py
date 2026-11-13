"""Check 5, GPU part: behaviour of M1 (default method) and M3 (cosine feature, two-block) for calibration draw 0 on one
development log, recomputed with my own code only (no rescope): own calibration from the cached keys (naive rule), own rebased
values, own replay hook on my own model instance, own teacher-forced scoring, items from the raw data.

  gq run --mem <MiB> -- python c5_behaviour.py Q|L        -> out/c5_<tag>/part_*.json (resumable), out/c5_<tag>_rows.json

Per item: ONE forward at W_0 (the pre-edit outcome; my hook also records the key at the last prompt position, which is compared
bit for bit with the cache); per method the decision from that key; where an edit fires, one forward with the replay hook.
A sequence on which nothing fires is the pre-edit model: its row is the pre-edit outcome.
The comparison with the stored rows of e0_sanity (draw 0) and with the reported table is made by c5_compare.py.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

import vcommon as V
import vgpu as G

import torch  # noqa: E402

CHUNK = 250
METHODS = {"M1": "ncoord", "M3": "cos"}
ALPHA, N_CAL, DRAW = "1/20", 200, 0


def setup(tag):
    """Everything that needs no model: the log, the calibration of draw 0 (naive rule on the cached keys), the items."""
    d = V.DEV[tag]
    name, bb, fam = d["log"], d["backbone"], d["family"]
    log = V.read_log(name)
    T, uids, W, K = log["T"], log["uids"], log["W"], log["K"]
    idx = V.read_index(bb)
    Ktf = V.read_keys(bb, "tf")
    prow = [e for e in idx if e["family"] == fam and e["kind"] == "probe"]
    recs = V.jsonl(f"{V.TESTBED}/{name}/probe_records_v2.jsonl")
    assert [(e["probe_idx"], e["ptype"], e["y0_text"]) for e in prow] == [(r["probe_idx"], r["ptype"], r["y0_text"]) for r in recs]
    ann = V.annotate(recs)
    repair = [e for e, a in zip(prow, ann) if a["pool"] == "repair" and e["y0_valid"]]
    cal = [repair[i] for i in V.draw(len(repair), N_CAL, DRAW)]
    Xcal = Ktf[torch.tensor([e["row_tf"] for e in cal])].double().numpy()
    rel = sorted([e for e in idx if e["family"] == fam and e["kind"] == "rel" and e["edit"] < T], key=lambda e: e["edit"])
    assert [e["edit_uid"] for e in rel] == uids
    Xb1 = Ktf[torch.tensor([e["row_tf"] for e in rel])].double().numpy()
    gedit = np.concatenate([np.arange(T), np.arange(T)])
    i1, i2 = V.halves(N_CAL, DRAW)
    gates = {}
    for mid, feature in METHODS.items():
        f = V.FEATURES[feature]
        b, eta, k = V.naive_two_block(f(Xcal, W, K), i1, i2, ALPHA)
        cap, sil = V.naive_caps(np.concatenate([f(K, W, K), f(Xb1, W, K)]), gedit, b, eta)
        gates[mid] = {"feature": feature, "b": b, "eta": eta, "rank": k, "cap": cap, "n_capped": int(np.isfinite(cap).sum()), "n_guards_silenced_literal": int(sil.sum())}
    # items from the raw data
    train = {x["uid"]: x for x in V.vlkeb("train")}
    stream = [train[r["uid"]] for r in V.stream_items(name)]
    assert [x["uid"] for x in stream] == uids
    items = []
    for m in ("rel", "tgen", "igen"):
        for i, x in enumerate(stream):
            items.append({"group": "edits", "sid": f"edit:{m}:{i}", "metric": m, "own": i, "ans": x["alt"],
                          "q": {"fmt": "vqa", "prompt": x["rephrase"] if m == "tgen" else x["src"], "image": x["image_rephrase"] if m == "igen" else x["image"]}})
    for r, a in zip(recs, ann):
        if a["disjoint"] and r["y0_valid"]:
            items.append({"group": "probes", "sid": f"probe:{r['ptype']}:{r['probe_idx']}", "ptype": r["ptype"], "strict": a["strict"], "own": -1, "ans": r["y0_text"],
                          "q": {"fmt": r["fmt"], "prompt": r["prompt"], "image": r["image"]}})
    pos = {u: i for i, u in enumerate(uids)}
    for r in V.jsonl(f"{V.CACHE}/hard_negatives/{bb}/rows.jsonl"):
        if r["edit_uid"] in pos and r["y0_valid"] and (r.get("edit_b_uid") is None or r["edit_b_uid"] in pos):
            items.append({"group": "hn", "sid": r["hn_id"], "hn_type": r["hn_type"], "own": pos[r["edit_uid"]], "own_b": pos.get(r.get("edit_b_uid"), -1), "ans": r["y0_text"],
                          "q": {"fmt": r["fmt"], "prompt": r["prompt"], "image": r["image"]}})
    by_lib_sid = {e["lib_sid"]: e for e in idx if e["family"] == fam and e.get("row_tf") is not None and (e.get("edit") is None or e["edit"] < T)}
    return {"name": name, "bb": bb, "layer": d["layer"], "mem": d["mem"], "log": log, "gates": gates, "items": items, "cal_sids": [e["lib_sid"] for e in cal],
            "index": by_lib_sid, "Ktf": Ktf}


def main(tag):
    S = setup(tag)
    odir = f"{V.OUT}/c5_{tag}"
    os.makedirs(odir, exist_ok=True)
    t0 = time.time()
    mem, cap_gb = G.cap_gpu(S["mem"])
    log, items, gates = S["log"], S["items"], S["gates"]
    W, K = log["W"], log["K"]
    values = G.rebased_values(log["U"], W, K, log["batch"])
    V.jdump({"log": S["name"], "draw": DRAW, "alpha": ALPHA, "n_cal": N_CAL, "calibration_sids": S["cal_sids"],
             "gates": {m: {"feature": g["feature"], "eta": g["eta"], "rank": g["rank"], "b": g["b"], "cap": g["cap"], "n_capped": g["n_capped"],
                           "n_guards_silenced_literal": g["n_guards_silenced_literal"]} for m, g in gates.items()},
             "n_items": {k: sum(it["group"] == k for it in items) for k in ("edits", "probes", "hn")}}, f"{odir}/gates.json")
    todo = [ci for ci in range((len(items) + CHUNK - 1) // CHUNK) if not os.path.exists(f"{odir}/part_{ci:03d}.json")]
    if todo:
        vlm = G.own_vlm(S["bb"])
        mod = vlm.edit_module(S["layer"])
        Wc, _, _ = G.checkpoint_weight(S["bb"], S["layer"])
        assert torch.equal(mod.weight.data.cpu(), Wc.to(mod.weight.dtype)), "the module does not hold the checkpoint weight"
        W0 = mod.weight.data.clone()
        V.say(f"{S['name']}: model loaded ({time.time() - t0:.0f}s), {len(items)} items, chunks to do {len(todo)}; "
              f"gates { {m: (g['eta'], g['n_capped']) for m, g in gates.items()} }")
        for ci in todo:
            t1 = time.time()
            rows, nfw = [], 0
            for it in items[ci * CHUNK:(ci + 1) * CHUNK]:
                batch = G.build(vlm, it["q"], it["ans"])
                lp = int(batch["last_prompt_idx"][0])
                with G.KeyTap(mod) as tap:
                    pred0, ans, fin0 = G.tf_forward(vlm, batch)
                nfw += 1
                key = tap.x[0][0, lp]
                e = S["index"].get(it["sid"])
                row = {"sid": it["sid"], "group": it["group"], "lp": lp, "pre": G.score(pred0, ans, fin0, pred0),
                       "key_in_cache": e is not None, "key_bitwise_cache": bool(e is not None and torch.equal(key.float().cpu(), S["Ktf"][e["row_tf"]])),
                       "lp_equals_cache": bool(e is not None and e["lp_tf"] == lp)}
                k64 = key.double().cpu().numpy()[None]
                for mid, g in gates.items():
                    r = V.FEATURES[g["feature"]](k64, W, K)
                    fired, top, cand = V.naive_decide(r, g["b"], g["eta"], g["cap"])
                    fired, top = bool(fired[0]), int(top[0])
                    if fired:
                        with G.Replay(mod, g["feature"], W, K, g["b"], g["eta"], g["cap"], values) as rp:
                            rp.begin(lp)
                            pred, _, fin = G.tf_forward(vlm, batch)
                        nfw += 1
                        assert rp.fired and rp.top == top and rp.calls == 1
                        sc = G.score(pred, ans, fin, pred0)
                    else:
                        sc = dict(row["pre"])
                    own = it["own"]
                    sc.update(fired=fired, top=top, n_cand=int(cand[0].sum()), own_cand=bool(own >= 0 and cand[0, own]), own_top=bool(fired and own >= 0 and top == own))
                    if it.get("own_b", -1) >= 0:
                        sc["b_top"] = bool(fired and top == it["own_b"])
                    row[mid] = sc
                rows.append(row)
            assert torch.equal(mod.weight.data, W0)
            V.jdump({"chunk": ci, "rows": rows, "forwards": nfw, "seconds": time.time() - t1}, f"{odir}/part_{ci:03d}.json", indent=None)
            V.say(f"  chunk {ci}: {len(rows)} items, {nfw} forwards, {time.time() - t1:.0f}s; key bitwise {sum(r['key_bitwise_cache'] for r in rows)}/{len(rows)}; "
                  f"fired { {m: sum(r[m]['fired'] for r in rows) for m in gates} }")
    rows = []
    for ci in range((len(items) + CHUNK - 1) // CHUNK):
        rows += V.jload(f"{odir}/part_{ci:03d}.json")["rows"]
    meta = {it["sid"]: {k: it[k] for k in ("group", "metric", "ptype", "strict", "hn_type", "own") if k in it} for it in items}
    V.jdump({"log": S["name"], "rows": rows, "meta": meta, "seconds": time.time() - t0,
             "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else None}, f"{V.OUT}/c5_{tag}_rows.json", indent=None)
    V.say(f"DONE {len(rows)} rows ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main(sys.argv[1])
