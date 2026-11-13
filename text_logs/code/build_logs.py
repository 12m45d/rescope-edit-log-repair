"""E10 steps 7-8: assemble the per-update files into log files, audit them, write the registry.

For each source run ('dev', 'stream'), as far as its updates exist:
  factors.pt     stacked float32 tensors of all updates (public naming: left, right; k_request, k_template_mean,
                 cur_output, target, target_init, delta) + requests, case ids, scalars, context templates
For each named log whose snapshot exists (TX-G2XL-rome-dev100 / -T100 / -T500 / -T1000 / -T2000):
  hashes         sha256 of the pre-edit weight (as loaded from the checkpoint) and of the deployed weight
  reconstruction E = W_E - W_0 - sum_i outer(left_i, right_i), all in fp64: max |E|, ||E||_F / ||W_E - W_0||_F,
                 ||E||_F / ||W_0||_F
  exact replay   W_0 + the public float32 in-place writes (left[:, None] @ right[None, :], added in order) is the
                 deployed weight BIT FOR BIT (sha256)
  per update     (non-circular: residual, keys and outputs are captured separately from the returned factors)
     att_request        <left, k_request> <right, res> / <res, res>,  res = target - cur_output:  the share of the
                        residual that the applied update delivers at the request key            (expected 1)
     att_template_mean  the same at the template-averaged key                                   (NOT 1: reported)
     key_out_consistency  || k_request @ W_before + b - cur_output || / || cur_output ||   (captured key and output
                        agree with the replayed weight)
     target_attained    || k_request @ W_after + b - target || / || target ||
  key identity   distribution of the two attainments, of the division factor, of the update norms
Writes logs_edit/registry.json and out/log_audit.json.  CPU only.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tp_common as C  # noqa: E402
import tx_public as TX  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(4)
F64 = torch.float64
TENS = ("left", "right", "k_avg", "k_request", "cur_output", "target", "target_init", "delta")


def quant(x):
    x = np.asarray(x, dtype=np.float64)
    q = np.quantile(x, [0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1])
    return {"n": int(x.size), "min": float(q[0]), "q01": float(q[1]), "q05": float(q[2]), "q25": float(q[3]), "median": float(q[4]),
            "q75": float(q[5]), "q95": float(q[6]), "q99": float(q[7]), "max": float(q[8]), "mean": float(x.mean())}


def assemble(src):
    D = os.path.join(C.LOGS_EDIT, src)
    n = 0
    while os.path.exists(os.path.join(D, "updates", f"u_{n:05d}.pt")):
        n += 1
    meta = C.jload(os.path.join(D, "meta.json"))
    fpath = os.path.join(D, "factors.pt")
    if os.path.exists(fpath):
        old = torch.load(fpath, map_location="cpu")
        if old["left"].shape[0] == n:
            return old, meta, n
    ups = [torch.load(os.path.join(D, "updates", f"u_{i:05d}.pt"), map_location="cpu") for i in range(n)]
    for i, u in enumerate(ups):
        assert u["index"] == i and u["record_index"] == TX.FIRST_RECORD[src] + i and u["seed"] == meta["seed"]
        assert u["weight_name"] == C.WEIGHT_NAME
    fac = {k: torch.stack([u["tensors"][k] for u in ups]).float() for k in TENS}
    fac["k_template_mean"] = fac.pop("k_avg")
    fac.update({"case_ids": [u["case_id"] for u in ups], "record_indices": [u["record_index"] for u in ups],
                "requests": [u["request"] for u in ups], "scalars": [u["scalars"] for u in ups],
                "context_templates": meta["context_templates"], "seed": meta["seed"], "source": src, "n": n,
                "weight_name": C.WEIGHT_NAME, "note": "public naming: left (d_in), right (d_out); update = left[:, None] @ right[None, :] "
                                                      "added to the Conv1D weight (d_in, d_out)"})
    C.tsave(fac, fpath)
    return fac, meta, n


def audit(src, fac, meta, n, W0, b):
    """Replay the writes in float32 exactly as the public code performs them; per-update checks in fp64."""
    D = os.path.join(C.LOGS_EDIT, src)
    snaps = {name: T for name, (s, T) in TX.PLAN.items() if s == src and T <= n
             and os.path.exists(os.path.join(D, "snapshots", f"W_T{T}.pt"))}
    W = W0.clone()                                           # float32, the module's orientation (d_in, d_out)
    W0d, bd = W0.to(F64), b.to(F64)
    S = torch.zeros_like(W0d)                                # fp64 sum of the outer products
    per, entries = [], {}
    recorded = {}
    pp = os.path.join(D, "progress.jsonl")
    if os.path.exists(pp):
        import json
        for line in open(pp):
            r = json.loads(line)
            if r.get("w_sha256"):
                recorded[r["n_applied"]] = r["w_sha256"]
    n_hash_ok = 0
    for i in range(n):
        left, right = fac["left"][i], fac["right"][i]
        ld, rd = left.to(F64), right.to(F64)
        kreq, kavg = fac["k_request"][i].to(F64), fac["k_template_mean"][i].to(F64)
        cur_out, target = fac["cur_output"][i].to(F64), fac["target"][i].to(F64)
        res = target - cur_out
        Wd = W.to(F64)
        out_before = kreq @ Wd + bd
        W += left.unsqueeze(1) @ right.unsqueeze(0)          # the public write, float32, in place
        S += torch.outer(ld, rd)
        out_after = kreq @ W.to(F64) + bd
        gain = float((rd @ res) / (res @ res))
        d = {"index": i, "case_id": fac["case_ids"][i],
             "division_factor": float(ld @ kreq),
             "att_request": float(ld @ kreq) * gain, "att_template_mean": float(ld @ kavg) * gain,
             "key_out_consistency": float((out_before - cur_out).norm() / cur_out.norm()),
             "target_attained": float((out_after - target).norm() / target.norm()),
             "update_fro": float(ld.norm() * rd.norm()), "left_norm": float(ld.norm()), "right_norm": float(rd.norm()),
             "residual_norm": float(res.norm()), "target_norm": float(target.norm()), "cur_output_norm": float(cur_out.norm()),
             "k_request_norm": float(kreq.norm()), "k_template_mean_norm": float(kavg.norm()),
             "cos_keys": float((kreq @ kavg) / (kreq.norm() * kavg.norm())),
             "W_fro_after": float(W.to(F64).norm()), "W_minus_W0_fro_after": float((W.to(F64) - W0d).norm()),
             "v_prob_new": fac["scalars"][i].get("v_prob_new"), "v_steps": fac["scalars"][i].get("v_steps"),
             "v_nll": fac["scalars"][i].get("v_nll"), "lookup_idx_request": fac["scalars"][i]["lookup_idxs"][0],
             "n_target_tokens": fac["scalars"][i].get("n_target_tokens"),
             "subject": fac["requests"][i]["subject"], "prompt": fac["requests"][i]["prompt"],
             "target_new": fac["requests"][i]["target_new"]["str"]}
        per.append(d)
        if (i + 1) in recorded:
            assert C.sha256_tensor(W)["sha256"] == recorded[i + 1], f"{src}: replay differs from the run after {i + 1} updates"
            n_hash_ok += 1
        for name, T in snaps.items():
            if T == i + 1:
                snap = torch.load(os.path.join(D, "snapshots", f"W_T{T}.pt"), map_location="cpu")
                WE = snap["weight"]
                E = WE.to(F64) - W0d - S
                dW = WE.to(F64) - W0d
                entries[name] = {
                    "name": name, "source": src, "T": T, "first_record": TX.FIRST_RECORD[src],
                    "records": [TX.FIRST_RECORD[src], TX.FIRST_RECORD[src] + T - 1],
                    "factors": os.path.join(D, "factors.pt"),
                    "deployed_weight": os.path.join(D, "snapshots", f"W_T{T}.pt"),
                    "pre_edit_weight": os.path.join(C.LOGS_EDIT, "W0_transformer.h.17.mlp.c_proj.weight.pt"),
                    "pre_edit_sha256": C.sha256_tensor(W0)["sha256"], "deployed_sha256": C.sha256_tensor(WE)["sha256"],
                    "deployed_sha256_recorded_by_run": snap["sha256"]["sha256"],
                    "exact_float32_replay_equals_deployed": bool(torch.equal(W, WE)),
                    "reconstruction": {"max_abs": float(E.abs().max()), "fro": float(E.norm()),
                                       "rel_fro_vs_update": float(E.norm() / dW.norm()),
                                       "rel_fro_vs_W0": float(E.norm() / W0d.norm()),
                                       "update_fro": float(dW.norm()), "W0_fro": float(W0d.norm()), "WE_fro": float(WE.to(F64).norm()),
                                       "WE_max_abs": float(WE.abs().max()), "W0_max_abs": float(W0.abs().max()),
                                       "WE_finite": bool(torch.isfinite(WE).all())},
                    "module": C.MODULE, "weight_name": C.WEIGHT_NAME, "backbone": "gpt2-xl", "editor": "ROME (public code)",
                    "seed": meta["seed"], "development_log": name == TX.DEV_LOG,
                }
    return per, entries, {"n_recorded_hashes_reproduced": n_hash_ok, "n_recorded_hashes": len([k for k in recorded if k <= n])}


def main():
    w0 = torch.load(os.path.join(C.LOGS_EDIT, "W0_transformer.h.17.mlp.c_proj.weight.pt"), map_location="cpu")
    W0 = w0["weight"]
    assert C.sha256_tensor(W0)["sha256"] == w0["sha256"]["sha256"]
    # the bias of the module, straight from the checkpoint (never edited)
    from safetensors.torch import load_file
    sd = load_file(os.path.join(C.MODEL_DIR, "model.safetensors"))
    kb = [k for k in sd if k.endswith("h.17.mlp.c_proj.bias")]
    kw = [k for k in sd if k.endswith("h.17.mlp.c_proj.weight")]
    assert len(kb) == 1 and len(kw) == 1, (kb, kw)
    b = sd[kb[0]].float()
    ckpt_equal = bool(torch.equal(sd[kw[0]].float(), W0))
    registry, report = {}, {"written": C.now(), "pre_edit_weight_equals_checkpoint_tensor": ckpt_equal,
                            "pre_edit_sha256": C.sha256_tensor(W0)["sha256"], "sources": {}}
    for src in ("dev", "stream"):
        if not os.path.exists(os.path.join(C.LOGS_EDIT, src, "meta.json")):
            continue
        fac, meta, n = assemble(src)
        if n == 0:
            continue
        assert meta["w0"]["sha256"] == C.sha256_tensor(W0)["sha256"]
        per, entries, hs = audit(src, fac, meta, n, W0, b)
        for name, e in entries.items():
            T = e["T"]
            sub = per[:T]
            e["key_identity"] = {
                "att_request": quant([p["att_request"] for p in sub]),
                "att_request_max_abs_dev_from_1": float(max(abs(p["att_request"] - 1) for p in sub)),
                "att_template_mean": quant([p["att_template_mean"] for p in sub]),
                "att_template_mean_share_within_1pct": float(np.mean([abs(p["att_template_mean"] - 1) < 0.01 for p in sub])),
                "att_template_mean_share_within_10pct": float(np.mean([abs(p["att_template_mean"] - 1) < 0.10 for p in sub])),
                "att_template_mean_share_above_2": float(np.mean([p["att_template_mean"] > 2 for p in sub])),
                "division_factor": quant([p["division_factor"] for p in sub]),
                "key_out_consistency": quant([p["key_out_consistency"] for p in sub]),
                "target_attained": quant([p["target_attained"] for p in sub]),
                "cos_keys": quant([p["cos_keys"] for p in sub]),
            }
            e["context_templates"] = meta["context_templates"]
            registry[name] = e
        report["sources"][src] = {"n_updates": n, "T_planned": meta["T_planned"], "complete": n >= meta["T_planned"],
                                  "context_templates": meta["context_templates"], "seed": meta["seed"],
                                  "resumed": meta.get("resumed", []), **hs, "per_update": per}
        C.say(src, "updates", n, "logs", list(entries))
    # keys and direction do not depend on the edited weight (they are computed upstream of it): the single-edit check
    # (each record edited alone) and the sequential stream must have captured the same left vector and the same keys
    sdir = os.path.join(C.OUT, "public_results", "ROME_single", "run_000", "cases_factors")
    sfac = os.path.join(C.LOGS_EDIT, "stream", "factors.pt")
    if os.path.isdir(sdir) and os.path.exists(sfac):
        fac = torch.load(sfac, map_location="cpu")
        n_cmp = n_left = n_keys = 0
        single_expl = []
        for i in range(min(fac["left"].shape[0], C.SINGLE_N)):
            pth = os.path.join(sdir, f"case_{i}.pt")
            if not os.path.exists(pth):
                continue
            a_ = torch.load(pth, map_location="cpu")
            n_cmp += 1
            n_left += int(torch.equal(a_["tensors"]["left"], fac["left"][i]))
            n_keys += int(torch.equal(a_["tensors"]["k_request"], fac["k_request"][i]) and torch.equal(a_["tensors"]["k_avg"], fac["k_template_mean"][i]))
            if a_["scalars"]["update_fro"] > float(W0.double().norm()):
                single_expl.append({"case_id": i, "update_fro": a_["scalars"]["update_fro"], "division_factor": a_["scalars"]["div_factor"],
                                    "subject": fac["requests"][i]["subject"], "prompt": fac["requests"][i]["prompt"]})
        # determinism across processes: (a) record 0 edited alone is the first update of the stream; (b) the public
        # probability test of the pre-edit model, single-edit run ('pre') against the description run (run_W0)
        det = {}
        p0 = os.path.join(sdir, "case_0.pt")
        if os.path.exists(p0):
            a0 = torch.load(p0, map_location="cpu")
            det["record0_right_vector_bitwise_equal"] = bool(torch.equal(a0["tensors"]["right"], fac["right"][0]))
            det["record0_target_bitwise_equal"] = bool(torch.equal(a0["tensors"]["target"], fac["target"][0]))
            import json as _json
            c1 = os.path.join(os.path.dirname(sdir), "case_0.json")
            c2 = os.path.join(C.LOGS_EDIT, "stream", "write_time", "case_0.json")
            if os.path.exists(c1) and os.path.exists(c2):
                det["record0_public_scores_identical"] = _json.load(open(c1))["post"] == _json.load(open(c2))["post"]
            wdir = os.path.join(C.OUT, "public_results", "ROME_seq", "run_W0")
            if os.path.isdir(wdir):
                n_c = n_i = 0
                for i in range(C.SINGLE_N):
                    f1, f2 = os.path.join(os.path.dirname(sdir), f"case_{i}.json"), os.path.join(wdir, f"case_{i}.json")
                    if os.path.exists(f1) and os.path.exists(f2):
                        n_c += 1
                        n_i += int(_json.load(open(f1))["pre"] == _json.load(open(f2))["post"])
                det["pre_edit_public_test_records_compared"] = n_c
                det["pre_edit_public_test_records_identical"] = n_i
        report["determinism_across_processes"] = det
        C.say("determinism across processes:", det)
        report["single_vs_stream"] = {"n_records_compared": n_cmp, "n_left_vector_bitwise_equal": n_left, "n_both_keys_bitwise_equal": n_keys,
                                      "single_edits_with_exploding_update": single_expl,
                                      "note": "single edit = record edited alone from the pre-edit weight (run_single.py)"}
        C.say("single vs stream:", {k: v for k, v in report["single_vs_stream"].items() if k.startswith("n_")}, "exploding single edits:", len(single_expl))
    C.jdump(registry, TX.REGISTRY_FILE)
    C.jdump(report, os.path.join(C.OUT, "log_audit.json"), indent=None)
    for name, e in registry.items():
        r, k = e["reconstruction"], e["key_identity"]
        C.say(f"{name}: T {e['T']} exact replay {e['exact_float32_replay_equals_deployed']} recon max|E| {r['max_abs']:.3e} "
              f"rel(update) {r['rel_fro_vs_update']:.3e} rel(W0) {r['rel_fro_vs_W0']:.3e} |dW| {r['update_fro']:.1f}; "
              f"att_request max dev {k['att_request_max_abs_dev_from_1']:.2e}; att_template_mean median {k['att_template_mean']['median']:.4f} "
              f"max {k['att_template_mean']['max']:.2f}; key/out consistency max {k['key_out_consistency']['max']:.2e}; "
              f"target attained max {k['target_attained']['max']:.2e}")


if __name__ == "__main__":
    main()
