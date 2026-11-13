"""Lean review, GPU part (one process per development log; through the queue).

    gq run --mem 10500|17000 --prio 5 --name rl_gpu_<x> -- python rl_gpu.py <development log>      -> out/gpu_<log>.json

Blocks (each in its own try: a failure is recorded, the others still run)
  mask      item 7: mask_edits(empty) is the method's own object and gives the same logits bit for bit; a masked edit is never released;
            frame rule on the sampled sequences; silent sequences are the pre-edit model bit for bit
  hook      item 1: two new gates (cos_centred, whitecos_0.1) and one _x gate: the edit the HOOK releases == the gate-level decision
  retr      item 2: retrieval rule: driver rows of firing sequences == the rows of a forward under a gate of MY OWN that releases that edit;
            silent rows == pre-edit rows
  rollback  item 7: fx.methods.rollback == the reviewer's own evaluation of the harness's rule from the files on disk (bitwise)
  recapture item 10: 50 recombination keys re-captured with the reviewer's own hook on the full forward, bit for bit
  prefix    item 7: the harness loader opens each prefix log; its weight == pre-edit weight + the first batch updates (own sum); with all
            ten batches == the stored deployed weight; Rel exact of the deployed prefix weight on its edits (B = 1) vs out/prefix_logs.txt
Only development logs / their prefixes; recombination keys are label-free.
"""
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP
import hashlib
import json
import os
import sys
import time
import traceback

import rl_common as C

NAME = sys.argv[1]
ONLY = sys.argv[2:] or ["mask", "hook", "retr", "rollback", "recapture", "prefix"]
import numpy as np  # noqa: E402
import torch  # noqa: E402

import fx  # noqa: E402

SEED = 77031                       # the reviewer's own seed
res = {"log": NAME, "fx": fx.code_fingerprint(), "blocks": {}}
OUTP = os.path.join(C.OUT, f"gpu_{NAME}.json")
if os.path.exists(OUTP):
    try:
        old = json.load(open(OUTP))
        if old.get("fx") == res["fx"]:
            res["blocks"] = old.get("blocks", {})
    except Exception:  # noqa: BLE001
        pass


def save():
    C.dump(f"gpu_{NAME}.json", res)


def block(name):
    def deco(fn):
        if name not in ONLY:
            return fn
        t0 = time.time()
        try:
            out = fn()
            out["seconds"] = round(time.time() - t0, 1)
            res["blocks"][name] = out
        except Exception as e:  # noqa: BLE001
            res["blocks"][name] = {"ok": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-3000:]}
        print(name, {k: v for k, v in res["blocks"][name].items() if k not in ("rows", "trace")}, flush=True)
        save()
        return fn
    return deco


L = fx.logs.open(NAME)
H = fx.logs.load_model(NAME)
C.cap_gpu()
ses = fx.behave.session(L, H)
rng = np.random.default_rng([SEED, 0 if L.family == "Q" else 1])
M = fx.methods.rescope(L, alpha=0.05, draw=0)
allrows = [int(j) for j in L.edit_rows["rel"]] + [int(j) for j in L.edit_rows["tgen"]] + [int(j) for j in L.edit_rows["igen"]] + \
          [int(j) for j in L.rows("probe", pool="heldout_disjoint", valid=True)] + [int(j) for j in L.rows("hn", valid=True)]
allrows = [j for j in allrows if L.index[j].get("lib_sid") is not None]
d0 = fx.gate.decide(L, M, allrows, "tf")
top0 = np.asarray(d0["top_edit"])
fired0 = np.asarray(d0["fired"])
near0 = np.asarray(d0["near"])


def hashes(rows_):
    return [(r["logit_sha1"], tuple(r["released"]) if r["released"] is not None else None) for r in rows_]


@block("mask")
def _mask():
    R = sorted(int(i) for i in rng.choice(L.T, 10, replace=False))
    M0 = fx.methods.mask_edits(L, M, [])
    Mm = fx.methods.mask_edits(L, M, R)
    same_obj = M0.obj is M.obj
    same_digest = M0.digest == M.digest
    # gate level, every sequence: no masked edit is ever top-1 / candidate; decisions unchanged where the released edit was not in R
    dm = fx.gate.decide(L, Mm, allrows, "tf")
    topm = np.asarray(dm["top_edit"])
    in_r = np.isin(top0, R) & fired0
    gl = {"n_rows": len(allrows), "n_released_masked_edit_after": int((np.isin(topm, R) & np.asarray(dm["fired"])).sum()),
          "n_released_R_before": int(in_r.sum()),
          "n_changed_outside_R": int(((topm != top0) | (np.asarray(dm["fired"]) != fired0))[~in_r].sum()),
          "tau_inf_for_R": bool(torch.isinf(Mm.obj.det_tau[torch.isin(Mm.obj.det_edit, torch.tensor(R))]).all()),
          "other_tau_same_bits": bool(torch.equal(Mm.obj.det_tau[~torch.isin(Mm.obj.det_edit, torch.tensor(R))], M.obj.det_tau[~torch.isin(M.obj.det_edit, torch.tensor(R))])),
          "values_same_tensor": Mm.obj.values is M.obj.values, "det_w_same_tensor": Mm.obj.det_w is M.obj.det_w}
    # behaviour on a sample: 24 rows on which an edit of R was released, 16 on which another edit was, 20 silent
    idx_r = [i for i in np.nonzero(in_r)[0]]
    idx_o = [i for i in np.nonzero(fired0 & ~in_r)[0]]
    idx_s = [i for i in np.nonzero(~fired0 & ~near0)[0]]
    pick = [int(i) for i in rng.permutation(idx_r)[:24]] + [int(i) for i in rng.permutation(idx_o)[:16]] + [int(i) for i in rng.permutation(idx_s)[:20]]
    rows = [allrows[i] for i in pick]
    a = ses.tf_hashed(M, rows)
    a0 = ses.tf_hashed(M0, rows)
    b = ses.tf_hashed(Mm, rows)
    p = ses.tf_hashed(fx.methods.preedit(L), rows)
    ha, ha0, hb, hp = hashes(a), hashes(a0), hashes(b), [r["logit_sha1"] for r in p]
    hook_vs_gate = sum(1 for i, x in zip(pick, a) if (list(x["released"]) != ([int(top0[i])] if fired0[i] else [])))
    viol = sum(1 for x, y in zip(a, b) if not (set(x["released"]) & set(R)) and x["logit_sha1"] != y["logit_sha1"])
    rel_masked = sum(1 for y in b if set(y["released"]) & set(R))
    silent_not_pre = sum(1 for x, q in zip(a, hp) if not x["released"] and x["logit_sha1"] != q)
    silent_after_mask_not_pre = sum(1 for y, q in zip(b, hp) if not y["released"] and y["logit_sha1"] != q)
    firing_equal_pre = sum(1 for x, q in zip(a, hp) if x["released"] and x["logit_sha1"] == q)
    out = {"R": R, "empty_mask_same_object": same_obj, "empty_mask_same_digest": same_digest, "gate_level": gl, "n_sequences": len(rows),
           "n_with_R_released": sum(1 for x in a if set(x["released"]) & set(R)), "empty_mask_logits_or_release_differ": sum(1 for x, y in zip(ha, ha0) if x != y),
           "hook_release_differs_from_gate_level": hook_vs_gate, "masked_edit_released": rel_masked, "frame_rule_violations": viol,
           "silent_not_bitwise_preedit": silent_not_pre, "silent_after_mask_not_bitwise_preedit": silent_after_mask_not_pre,
           "firing_rows_equal_to_preedit_hash": firing_equal_pre,
           "n_other_edit_released_after_mask": sum(1 for x, y in zip(a, b) if (set(x["released"]) & set(R)) and y["released"])}
    out["ok"] = bool(same_obj and same_digest and gl["n_released_masked_edit_after"] == 0 and gl["n_changed_outside_R"] == 0 and gl["tau_inf_for_R"]
                     and gl["other_tau_same_bits"] and out["empty_mask_logits_or_release_differ"] == 0 and hook_vs_gate == 0 and rel_masked == 0
                     and viol == 0 and silent_not_pre == 0 and silent_after_mask_not_pre == 0)
    return out


@block("hook")
def _hook():
    out = {"gates": {}}
    ok = True
    for nm in ("cos_centred", "whitecos_0.1", "white_1_shared", "cos_null", "rescope_x", "whitecos_0.1_x"):
        Mg = fx.methods.build(nm, L, alpha=0.05, draw=1)
        d = fx.gate.decide(L, Mg, allrows, "tf")
        f, t, nr = np.asarray(d["fired"]), np.asarray(d["top_edit"]), np.asarray(d["near"])
        fi, si = np.nonzero(f & ~nr)[0], np.nonzero(~f & ~nr)[0]
        r2 = np.random.default_rng([SEED, 5, len(nm)])
        pick = [int(i) for i in r2.permutation(fi)[:14]] + [int(i) for i in r2.permutation(si)[:10]]
        rows = [allrows[i] for i in pick]
        a = ses.tf_hashed(Mg, rows)
        p = ses.tf_hashed(fx.methods.preedit(L), rows)
        bad = sum(1 for i, x in zip(pick, a) if list(x["released"]) != ([int(t[i])] if f[i] else []))
        snp = sum(1 for x, q in zip(a, p) if not x["released"] and x["logit_sha1"] != q["logit_sha1"])
        # the value released: the same forward under MY gate (plain library ScopedLog releasing that edit with the cosine gate's values)
        mem = ses.tf(Mg, rows)                                         # the memo path of the driver: same flags as the hashed forward
        fl = sum(1 for x, y in zip(a, mem) if any(x.get(k) != y.get(k) for k in ("exact", "D1", "fired")))
        out["gates"][nm] = {"n": len(rows), "n_firing": int(sum(bool(x["released"]) for x in a)), "hook_release_differs_from_gate_level": bad,
                            "silent_not_bitwise_preedit": snp, "memo_path_flags_differ_from_forward": fl}
        ok &= bad == 0 and snp == 0 and fl == 0
    out["ok"] = bool(ok)
    return out


@block("retr")
def _retr():
    from rescope import ScopedLog
    out = {"rules": {}}
    ok = True
    Mc = fx.methods.cosine(L, alpha=0.05, draw=0)
    V = Mc.obj.values                                                    # the values of the registered cosine gate (fairness: the same values)
    for nm in ("retr_sum", "retr_min", "retr_conj", "lookup"):
        Mr = fx.methods.build(nm, L, alpha=0.05, draw=0)
        d = fx.gate.decide(L, Mr, allrows, "tf")
        f, t = np.asarray(d["fired"]), np.asarray(d["top_edit"])
        r2 = np.random.default_rng([SEED, 7, len(nm)])
        fi, si = np.nonzero(f)[0], np.nonzero(~f)[0]
        pick = [int(i) for i in r2.permutation(fi)[:10]] + [int(i) for i in r2.permutation(si)[:6]]
        rows = [allrows[i] for i in pick]
        drv = ses.tf(Mr, rows)
        pre = ses.tf_hashed(fx.methods.preedit(L), rows)
        bad_flag, bad_val, bad_sil = 0, 0, 0
        for i, j, x, q in zip(pick, rows, drv, pre):
            if bool(x["fired"]) != bool(f[i]):
                bad_flag += 1
            if not f[i]:
                bad_sil += int(any(x.get(k) != q.get(k) for k in ("exact", "token_acc", "D1")))
                continue
            e = int(t[i])
            tau = torch.full((L.T,), float("inf"), dtype=torch.float64)
            tau[e] = -float("inf")
            S = ScopedLog.from_log(L.log, role="last", fire="top1", response="repl", tau=tau, values=V)      # my forced gate: releases edit e, the cosine gate's value
            mine = ses.tf_hashed(fx.methods.wrap(L, S, f"rl_forced_{e}"), [j])[0]
            if mine["released"] != [e]:
                bad_val += 1
            bad_val += int(any(x.get(k) != mine.get(k) for k in ("exact", "token_acc", "D1")))
        out["rules"][nm] = {"n": len(rows), "n_firing": int(f[pick].sum()), "fired_flag_differs": bad_flag, "firing_rows_differ_from_own_forced_gate": bad_val,
                            "silent_rows_differ_from_preedit_forward": bad_sil}
        ok &= bad_flag == 0 and bad_val == 0 and bad_sil == 0
    out["values_equal_default_method"] = bool(torch.equal(V, M.obj.values))
    out["ok"] = bool(ok and out["values_equal_default_method"])
    return out


@block("rollback")
def _rollback():
    path = fx.logs.info(NAME)["path"]
    fac = torch.load(os.path.join(path, "edit_factors.pt"), map_location="cpu")
    comp = torch.load(os.path.join(path, "composite_delta.pt"), map_location="cpu")
    comp = comp["delta"] if isinstance(comp, dict) and "delta" in comp else comp
    if isinstance(comp, dict):
        comp = [v for v in comp.values() if torch.is_tensor(v)][0]
    met = json.load(open(os.path.join(path, "metrics.json")))
    batches = [list(b["edit_ids"]) for b in met["per_batch"]]
    dev = H.W0.device

    def delta(b):
        x = torch.load(os.path.join(path, "batch_deltas", f"batch_{b:03d}.pt"), map_location="cpu")
        if isinstance(x, dict):
            x = [v for v in x.values() if torch.is_tensor(v)][0]
        return x

    def mine(R):
        Rs = set(R)
        if len(Rs) == L.T:
            return H.W0.clone()
        Cd = comp.to(dev).float()
        X = torch.zeros(H.W0.shape, dtype=torch.float32, device=dev)
        part = []
        for b, ids in enumerate(batches):
            inr = [i for i in ids if i in Rs]
            if inr and len(inr) == len(ids):
                X += delta(b).to(dev).float()
            else:
                part += inr
        if part:
            idx = torch.tensor(sorted(part))
            X += (fac["U"][idx].to(dev, torch.float64).T @ fac["W"][idx].to(dev, torch.float64)).float()
        return (H.W0.float() + (Cd - X)).to(H.W0.dtype)
    sets = {"empty": [], "seven_spread": sorted(int(i) for i in rng.choice(L.T, 7, replace=False)), "batch3_plus2": batches[3] + [batches[5][0], batches[8][4]],
            "fifty": sorted(int(i) for i in rng.choice(L.T, 50, replace=False)), "all": list(range(L.T))}
    out = {"sets": {}}
    A_before = getattr(H, "A", None) is not None
    for k, R in sets.items():
        Mr = fx.methods.rollback(L, H, R)
        W = mine(R)
        out["sets"][k] = {"n": len(R), "bitwise_equal_own": bool(torch.equal(Mr.obj, W)), "max_abs_diff": float((Mr.obj.float() - W.float()).abs().max()),
                          "equals_WE": bool(torch.equal(Mr.obj, H.WE)), "equals_W0": bool(torch.equal(Mr.obj, H.W0))}
    # the rollback of one edit removes its rank-one term at its own key to float precision (sanity of the convention)
    out["handle_had_batch_deltas"] = A_before
    out["stored_composite_equals_WE_minus_W0"] = bool(torch.equal(comp.to(dev).float(), H.WE.float() - H.W0.float()))
    out["ok"] = bool(all(v["bitwise_equal_own"] for v in out["sets"].values()) and out["sets"]["empty"]["equals_WE"] and out["sets"]["all"]["equals_W0"]
                     and out["stored_composite_equals_WE_minus_W0"])
    return out


@block("recapture")
def _recapture():
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    IMG = f"{_RP.EXT_VIEW}/data/datasets/VLKEB/mmkb_images"
    bb = L.backbone
    rows = [json.loads(x) for x in open(f"{C.FIN}/cache2/recomb/{bb}/rows.jsonl") if x.strip()]
    blob = torch.load(f"{C.FIN}/cache2/recomb/{bb}/keys.pt", map_location="cpu", mmap=True)
    K, rids, lps = blob["K_last"], list(blob["rids"]), [int(x) for x in blob["last_prompt_idx"]]
    assert rids == [r["rid"] for r in rows]
    r2 = np.random.default_rng([SEED, 11, 0 if bb == "qwen25vl" else 1])
    pick = []
    for kind, n_first, n_later in (("X-I", 10, 7), ("X-Q", 10, 7), ("X-X", 10, 6)):
        a = [i for i, r in enumerate(rows) if r["kind"] == kind and r["edit_pos"] < 100]
        b = [i for i, r in enumerate(rows) if r["kind"] == kind and r["edit_pos"] >= 100]
        pick += sorted(int(a[j]) for j in r2.choice(len(a), n_first, replace=False)) + sorted(int(b[j]) for j in r2.choice(len(b), n_later, replace=False))
    v, mod = H.vlm, H.mod
    store = {}

    def pre_hook(m, args):
        store["x"] = args[0].detach()
    H.set_weight(H.W0)
    recs = []
    try:
        for i in pick:
            r = rows[i]
            p = r["image"] if os.path.isabs(r["image"]) else os.path.join(IMG, r["image"])
            with Image.open(p) as im:
                img = im.convert("RGB")
            b = v.build_batch([img], [r["prompt"]], answers=None, fmt=r.get("fmt", "vqa"))
            ids = b["inputs"]["input_ids"]
            last = ids.shape[1] - 1
            store.clear()
            h = mod.register_forward_pre_hook(pre_hook)
            try:
                with torch.no_grad():
                    v.m(**b["inputs"], use_cache=False)
            finally:
                h.remove()
            x = store["x"]
            x = x[0] if x.dim() == 3 else x
            k = x[last].float().cpu()
            recs.append({"row": i, "kind": r["kind"], "edit_pos": r["edit_pos"], "bitwise": bool(torch.equal(k, K[i])), "max_abs_diff": float((k - K[i]).abs().max()),
                         "lp_equal": last == lps[i]})
    finally:
        H.set_weight(H.WE)
    return {"n": len(recs), "n_bitwise": sum(r["bitwise"] for r in recs), "n_lp_equal": sum(r["lp_equal"] for r in recs),
            "max_abs_diff": max(r["max_abs_diff"] for r in recs), "by_kind": {k: [sum(r["bitwise"] for r in recs if r["kind"] == k), sum(r["kind"] == k for r in recs)] for k in ("X-I", "X-Q", "X-X")},
            "n_first_100_edits": sum(r["edit_pos"] < 100 for r in recs), "rows": recs,
            "ok": all(r["bitwise"] and r["lp_equal"] for r in recs) and len(recs) == 50}


@block("prefix")
def _prefix():
    path = fx.logs.info(NAME)["path"]

    def delta(b):
        x = torch.load(os.path.join(path, "batch_deltas", f"batch_{b:03d}.pt"), map_location="cpu")
        if isinstance(x, dict):
            x = [v for v in x.values() if torch.is_tensor(v)][0]
        return x
    WE_dev, W0 = H.WE.clone(), H.W0.clone()
    for a in ("C", "U", "Wfac", "Kp_log"):                             # the development handle is no longer needed: free what it holds next to the model
        if hasattr(H, a):
            setattr(H, a, None)
    torch.cuda.empty_cache()
    table = {}
    for line in open(os.path.join(C.COMMON, "out", "prefix_logs.txt")):
        t = line.split()
        if t and t[0].startswith(NAME + "-p") and t[0] not in table and len(t) > 5 and t[1].isdigit():
            table[t[0]] = t
    out = {"prefixes": {}}
    prev = H
    ok = True
    for k in (10, 20, 50, 100):
        nm = f"{NAME}-p{k}"
        Hp = fx.prefix.load_model(nm, reuse=prev, keep_batch_deltas=True, cache_dir=os.path.join(C.HERE, "hcache"))
        S = torch.zeros(W0.shape, dtype=torch.float32, device=W0.device)
        for b in range(k // 10):
            S += delta(b).to(W0.device).float()
        mineW = (W0.float() + S).to(W0.dtype)
        rec = {"T": int(Hp.T), "checks_true": all(Hp.checks.get(c) is True for c in ("probe_uids_match_records", "WE_float_minus_W0_equals_stored_composite", "C_equals_stored_composite", "factor_uids_match_stream", "rollback_empty_equals_WE")),
               "checks_not_true": {c: str(v) for c, v in Hp.checks.items() if v is not True}, "W0_equal": bool(torch.equal(Hp.W0, W0)),
               "WE_equals_own_construction": bool(torch.equal(Hp.WE, mineW)), "module_holds_WE": bool(torch.equal(Hp.mod.weight.data, Hp.WE)),
               "rollback_empty_is_WE": bool(torch.equal(Hp.rollback_weight([]), Hp.WE)), "n_batch_deltas": len(Hp.A) if Hp.A is not None else None}
        if k == 100:
            rec["WE_equals_stored_deployed_weight"] = bool(torch.equal(Hp.WE, WE_dev))
            ok &= rec["WE_equals_stored_deployed_weight"]
        else:
            Lp = fx.logs.open(nm)
            sp = fx.behave.session(Lp, Hp)
            rr = sp.tf_hashed(fx.methods.deployed(Lp), [int(j) for j in Lp.edit_rows["rel"]])
            rec["rel_exact"] = [int(sum(bool(r["exact"]) for r in rr)), len(rr)]
            tl = table.get(nm) or []
            rec["table_rel"] = tl[4] if len(tl) > 4 else None
            rec["rel_matches_table"] = bool(tl) and tl[4] == f"({rec['rel_exact'][0]}/{rec['rel_exact'][1]})"
            rec["edit_log_T"] = int(Lp.T)
            rec["hn_x_both_in_prefix"] = all(max(Lp.own(j), -1) < k for j in Lp.rows("hn", valid=True))
            ok &= rec["rel_matches_table"]
        ok &= rec["checks_true"] and rec["W0_equal"] and rec["WE_equals_own_construction"] and rec["T"] == k and rec["module_holds_WE"]
        out["prefixes"][nm] = rec
        prev = Hp
    out["ok"] = bool(ok)
    return out


try:
    res["assert_clean"] = ses.assert_clean()
except Exception as e:  # noqa: BLE001
    res["assert_clean"] = f"FAILED {type(e).__name__}: {e}"
res["stats"] = {k: (v if isinstance(v, (int, float)) else str(v)) for k, v in ses.stats.items()}
save()
print("done", {k: v.get("ok") for k, v in res["blocks"].items()}, res["assert_clean"], flush=True)
