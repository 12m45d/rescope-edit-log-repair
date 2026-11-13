"""Check 6: mechanics at scale on TB-Q2000-ae-l2_3.  THIS IS AN UNSEEN TEST LOG: nothing about its results is printed or
stored: only pass / fail of the mechanics, counts of agreement between code paths, memory and seconds.

  gq run --mem 10500 -- python c6_scale.py        -> out/c6_scale.json      (fresh process: the peak RAM is that of this run)

  1  load the log through the library (rescope.load, lean by default for T >= 500): load-time checks, peak resident memory
  2  the default method for ONE calibration draw from the key cache (docs/USAGE.md, 'Without a model'): KeyCache -> feature_gate
     -> add_guards -> calibrate_two_block(coords) -> decide(log=);  my naive implementation on the same cached rows must give
     the same decisions
  3  the same gate built on the model's device from the Evaluator (thresholds must agree with the CPU gate to round-off);
     50 firing edit sequences through the model (SparseEvaluator.tf_rows): the hook's flags must equal the CPU decisions, the
     logits must be finite, the full Evaluator must return the same rows, and my own replay hook (own rebased values) must
     return the same outcome; 30 silent sequences: sparse rows == full rows
"""
from __future__ import annotations

import os
import resource
import sys
import time

import numpy as np

import vcommon as V
import vgpu as G

os.environ.setdefault("RESCOPE_CACHE", V.VER + "/work/hcache")
import torch  # noqa: E402
import rescope  # noqa: E402
from rescope import (HN_DIR_FINAL, Evaluator, KeyCache, SparseEvaluator, add_guards, calibrate_two_block, feature_gate, from_harness, from_testbed, load, splits)  # noqa: E402

NAME, BB, FAM, LAYER = "TB-Q2000-ae-l2_3", "qwen25vl", "Q", 20
SEED = 20261111


def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20


def main():
    t0 = time.time()
    mem = int(os.environ.get("GQ_MEM", 10500))
    cap = (mem - 600) / 1024.0
    res = {"log": NAME, "note": "unseen test log: mechanics only, no result of the log is stored", "peak_rss_gb_before_load": rss(), "mechanics": {}}
    M = res["mechanics"]
    # ------------------------------------------------------------------ 1: load
    H = load(NAME, device="cuda:0", mem_gb=cap)
    res["load"] = {"info": H.load_info, "checks": H.checks, "peak_rss_gb_after_load": rss(), "T": int(H.T), "seconds": time.time() - t0}
    M["load_checks_ok"] = all(v for k, v in H.checks.items() if k != "collapsed_working_point") and H.load_info["lean"] is True
    V.say(f"loaded: lean {H.load_info['lean']}, {res['load']['seconds']:.0f}s, peak RSS {res['load']['peak_rss_gb_after_load']:.1f} GB, load checks ok {M['load_checks_ok']}")
    # ------------------------------------------------------------------ 2: the default method from the cache, one calibration draw
    kc = KeyCache(BB)
    L = from_testbed(NAME)
    uids = L.meta["uids"]
    Gc = feature_gate(L, "ncoord")
    Gc, _ = add_guards(Gc, *kc.own_keys(uids))
    ben = splits.sample(kc.log_rows(uids, kind="probe", pool="repair", valid=True), 200, 0)
    Gc, rec = calibrate_two_block(Gc, coords=kc.coords(Gc, ben), alpha=0.05, seed=0)
    dec = kc.decide(Gc, "tf", log=L)
    rows = dec["rows"]
    M["calibration_record_ok"] = bool(rec["n1"] == 100 and rec["n2"] == 100 and rec["rank"] == 96 and abs(rec["bound_false_firing"] - 5 / 101) < 1e-12 and np.isfinite(rec["eta"]))
    M["n_cached_sequences_of_the_log"] = len(rows)
    # my naive implementation on the same rows
    idx = V.read_index(BB)
    Ktf = V.read_keys(BB, "tf")
    lg = V.read_log(NAME)
    T, W, K = lg["T"], lg["W"], lg["K"]
    assert lg["uids"] == uids
    pos = {u: i for i, u in enumerate(uids)}
    prow = [e for e in idx if e["family"] == FAM and e["kind"] == "probe"]
    ann = V.annotate(prow)
    repair = [j for j, e in enumerate(idx) if e["family"] == FAM and e["kind"] == "probe"]
    repair = [j for j, a in zip(repair, ann) if a["pool"] == "repair" and idx[j]["y0_valid"]]
    cal = [repair[i] for i in V.draw(len(repair), 200, 0)]
    M["calibration_rows_equal_mine"] = cal == ben
    f = V.FEATURES["ncoord"]
    relrows = sorted([j for j, e in enumerate(idx) if e["family"] == FAM and e["kind"] == "rel" and e["edit"] < T], key=lambda j: idx[j]["edit"])
    key = lambda js: Ktf[torch.tensor([idx[j]["row_tf"] for j in js])].double().numpy()      # noqa: E731
    i1, i2 = V.halves(200, 0)
    b, eta, k = V.naive_two_block(f(key(cal), W, K), i1, i2, "1/20")
    capv, sil = V.naive_caps(np.concatenate([f(K, W, K), f(key(relrows), W, K)]), np.concatenate([np.arange(T), np.arange(T)]), b, eta)
    fired = np.zeros(len(rows), dtype=bool)
    top = np.full(len(rows), -1)
    own_c, own_t = np.zeros(len(rows), dtype=bool), np.zeros(len(rows), dtype=bool)
    for a in range(0, len(rows), 1000):
        js = rows[a:a + 1000]
        fi, tp, cd = V.naive_decide(f(key(js), W, K), b, eta, capv)
        ow = np.array([pos.get(idx[j].get("edit_uid"), -1) for j in js])
        has = ow >= 0
        fired[a:a + 1000], top[a:a + 1000] = fi, tp
        own_c[a:a + 1000][has] = cd[has, ow[has]]
        own_t[a:a + 1000] = fi & has & (tp == ow)
    nd = int((fired != dec["fired"]).sum() + (top != dec["top_edit"]).sum() + (own_c != dec["own_cand"]).sum() + (own_t != dec["own_top"]).sum())
    M["cpu_decisions_differing_from_naive"] = [nd, 4 * len(rows)]
    M["eta_equal_naive_1e-12"] = bool(abs(eta - rec["eta"]) <= 1e-12)
    M["capped_edits_equal_naive"] = sorted(int(i) for i in np.nonzero(np.isfinite(capv))[0]) == rec["capped_edits"]
    V.say(f"cache decisions: {len(rows)} sequences decided; calibration record ok {M['calibration_record_ok']}; differing from the naive rule {nd}; "
          f"eta equal {M['eta_equal_naive_1e-12']}; capped sets equal {M['capped_edits_equal_naive']} ({time.time() - t0:.0f}s, peak RSS {rss():.1f} GB)")
    # ------------------------------------------------------------------ 3: the model on 50 firing edit sequences
    ev = SparseEvaluator(H, cache=kc, hn_dir=HN_DIR_FINAL)
    evF = Evaluator(H, hn_dir=HN_DIR_FINAL)
    log = from_harness(H).to(H.dev)
    S = feature_gate(log, "ncoord", W0=H.W0)
    S, _ = add_guards(S, *ev.own_keys("rel"))
    benign = splits.sample(ev.probe_items("repair"), 200, 0)
    S, rec2 = calibrate_two_block(S, [ev.eligible_keys(it, S) for it in benign], 0.05, seed=0)
    tg, tc = S.det_tau.cpu(), Gc.det_tau
    fin = torch.isfinite(tc)
    M["gpu_thresholds_equal_cpu_1e-12"] = bool((torch.isfinite(tg) == fin).all() and float(((tg[fin] - tc[fin]).abs() / tc[fin].abs().clamp_min(1e-300)).max()) <= 1e-12)
    M["gpu_calibration_rows_equal_cpu"] = [it["sid"] for it in benign] == [idx[j]["lib_sid"] for j in ben]
    lib_items = {it["sid"]: it for it in ev.edit_items()}
    cand_rows = [i for i, j in enumerate(rows) if idx[j]["kind"] in ("rel", "tgen", "igen") and dec["fired"][i] and not dec["near"][i]]
    sil_rows = [i for i, j in enumerate(rows) if idx[j]["kind"] in ("rel", "tgen", "igen", "probe") and not dec["fired"][i] and not dec["near"][i]
                and (idx[j]["kind"] != "probe" or idx[j]["y0_valid"])]
    rng = np.random.default_rng([SEED, 6])
    pf = [cand_rows[int(i)] for i in sorted(rng.choice(len(cand_rows), 50, replace=False))]
    ps = [sil_rows[int(i)] for i in sorted(rng.choice(len(sil_rows), 30, replace=False))]
    probe_items = {it["sid"]: it for it in ev.probe_items("all")}
    item = lambda i: (lib_items.get(idx[rows[i]]["lib_sid"]) or probe_items[idx[rows[i]]["lib_sid"]])     # noqa: E731
    M["n_firing_edit_sequences_run"] = len(pf)
    M["firing_sample_covers_stream_beyond_position_1000"] = bool(max(idx[rows[i]]["edit"] for i in pf) >= 1000)
    t1 = time.time()
    st0 = dict(ev.stats)
    rs = V._clean(ev.tf_rows(S, [item(i) for i in pf]))
    M["sparse_ran_the_model_on_every_firing_sequence"] = ev.stats["forwards"] - st0["forwards"] == len(pf)
    rf = V._clean(evF.tf_rows(S, [item(i) for i in pf]))
    M["hook_fired"] = [sum(bool(r["fired"]) for r in rs), len(rs)]
    M["hook_top_equals_cpu_decision"] = [sum(r["top_last"] == int(dec["top_edit"][i]) for r, i in zip(rs, pf)), len(rs)]
    M["hook_own_flags_equal_cpu_decision"] = [sum(bool(r["own_fired_last"]) == bool(dec["own_cand"][i]) and bool(r["own_released_last"]) == bool(dec["own_top"][i])
                                                   for r, i in zip(rs, pf)), len(rs)]
    M["logits_finite"] = [sum(bool(r["finite"]) for r in rs), len(rs)]
    M["sparse_rows_equal_full_rows_firing"] = [sum(x == y for x, y in zip(rs, rf)), len(rs)]
    st0 = dict(ev.stats)
    ss = V._clean(ev.tf_rows(S, [item(i) for i in ps]))
    M["sparse_ran_no_forward_on_silent_sequences"] = ev.stats["forwards"] - st0["forwards"] == 0
    sf = V._clean(evF.tf_rows(S, [item(i) for i in ps]))
    M["sparse_rows_equal_full_rows_silent"] = [sum(x == y for x, y in zip(ss, sf)), len(ss)]
    M["silent_rows_not_fired"] = [sum(not r["fired"] for r in sf), len(sf)]
    V.say(f"model: firing sequences run {len(pf)}, hook fired {M['hook_fired']}, top == CPU {M['hook_top_equals_cpu_decision']}, own flags == CPU {M['hook_own_flags_equal_cpu_decision']}, "
          f"finite {M['logits_finite']}, sparse == full {M['sparse_rows_equal_full_rows_firing']} (firing) {M['sparse_rows_equal_full_rows_silent']} (silent) ({time.time() - t1:.0f}s)")
    # my own replay on the same firing sequences
    train = {x["uid"]: x for x in V.vlkeb("train")}
    stream = [train[r["uid"]] for r in V.stream_items(NAME)]
    values = G.rebased_values(lg["U"], W, K, lg["batch"])
    Wc, _, _ = G.checkpoint_weight(BB, LAYER)
    W0 = Wc.to(H.dev, H.dtype)
    M["H.W0_equals_checkpoint_file"] = bool(torch.equal(H.W0, W0))
    H.mod.weight.data.copy_(W0)
    eq = 0
    for r, i in zip(rs, pf):
        e = idx[rows[i]]
        x, m = stream[e["edit"]], e["kind"]
        q = {"fmt": "vqa", "prompt": x["rephrase"] if m == "tgen" else x["src"], "image": x["image_rephrase"] if m == "igen" else x["image"]}
        batch = G.build(H.vlm, q, x["alt"])
        with G.Replay(H.mod, "ncoord", W, K, b, eta, capv, values) as rp:
            rp.begin(int(batch["last_prompt_idx"][0]))
            pred, ans, fn = G.tf_forward(H.vlm, batch)
        sc = G.score(pred, ans, fn)
        eq += int(rp.fired and rp.top == r["top_last"] and sc["exact"] == r["exact"] and abs(sc["token_acc"] - r["token_acc"]) < 1e-6 and sc["n_tok"] == r["n_tok"] and fn == r["finite"])
    H.reset()
    M["my_replay_equals_library_rows"] = [eq, len(rs)]
    M["module_weight_is_WE_after"] = bool(torch.equal(H.mod.weight.data, H.WE))
    full = lambda v: v[0] == v[1]                                                               # noqa: E731
    res["pass"] = bool(M["load_checks_ok"] and M["calibration_record_ok"] and M["calibration_rows_equal_mine"] and M["cpu_decisions_differing_from_naive"][0] == 0
                       and M["eta_equal_naive_1e-12"] and M["capped_edits_equal_naive"] and M["gpu_thresholds_equal_cpu_1e-12"] and M["gpu_calibration_rows_equal_cpu"]
                       and M["sparse_ran_the_model_on_every_firing_sequence"] and M["sparse_ran_no_forward_on_silent_sequences"]
                       and all(full(M[k]) for k in ("hook_fired", "hook_top_equals_cpu_decision", "hook_own_flags_equal_cpu_decision", "logits_finite",
                                                    "sparse_rows_equal_full_rows_firing", "sparse_rows_equal_full_rows_silent", "silent_rows_not_fired",
                                                    "my_replay_equals_library_rows"))
                       and M["H.W0_equals_checkpoint_file"] and M["module_weight_is_WE_after"])
    res.update(peak_rss_gb_end=rss(), peak_gpu_gb=torch.cuda.max_memory_allocated() / 2 ** 30, seconds=time.time() - t0, rescope=os.path.dirname(rescope.__file__))
    V.jdump(res, f"{V.OUT}/c6_scale.json")
    V.say(f"my replay == library rows {M['my_replay_equals_library_rows']}; PASS {res['pass']}; peak RSS {res['peak_rss_gb_end']:.1f} GB, peak GPU {res['peak_gpu_gb']:.2f} GB ({res['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
