"""Audit, condition (1), CPU part: the audit's own reconstruction of every log.

For each of the five logs (factors and weights read from the builder's files; nothing of the builder's code is imported):
  a. W_0: the stored pre-edit weight against the checkpoint tensors (model.safetensors AND pytorch_model.bin, read by
     the audit) -- bitwise.
  b. Reconstruction in fp64: E = W_E - W_0 - sum_{i<T} outer(left_i, right_i): max |E|, ||E||_F / ||W_E - W_0||_F,
     ||E||_F / ||W_0||_F.  For scale: the float32 spacing at max |W_E| (one unit in the last place) and the number of
     in-place float32 additions (T).
  c. Replay in float32, in place, in order, as the public code writes (w[...] += left[:, None] @ right[None, :]):
     bitwise equality with the stored deployed weight; sha256 of both.
  d. Orientation: the stored weight has the shape of the Conv1D parameter (d_in = 6400, d_out = 1600), the outer
     product left (6400) x right (1600) has that shape without transposition; the TRANSPOSED outer product does not
     reconstruct (error reported).  y = x @ W + b is checked on the GPU (gpu_rome.py, module_output_formula_rel).
  e. Key identity from the stored files, fp64: s_i = <left_i, k_request_i>; read-out vector w_i = left_i / s_i;
     <w_i, k_request_i> - 1; sign and size of s_i; <w_i, k_template_mean_i>; and the non-circular solve check with the
     audit's replayed weight: (k_request_i @ W_before_i + b) vs stored cur_output_i, (k_request_i @ W_after_i + b) vs
     stored target_i, right_i * s_i vs target_i - cur_output_i.
  f. Library-format copies (logs_edit/library_format/<name>/edit_factors.pt): U = right * s, W = left / s, K1_solver =
     k_request; sum_i outer(U_i, W_i) = (W_E - W_0)^T; <W_i, K_i> = 1.
  g. Prefix property: T100 / T500 / T1000 are prefixes of the T2000 stream (same factor file; snapshot at T = replay of
     the first T factors); request i of the log is record first + i of the data set.
  h. Update norms: ||update_i||_F, exploding updates (> ||W_0||_F), requests whose subject is the first prompt token
     (the audit's own offsets-based index), division factors.
Output: out/recon.json
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a_common as A  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(4)
F64 = torch.float64
out = {"written": A.now(), "logs": {}}
out_path = os.path.join(A.AOUT, "recon.json")

# ---- a. W_0 against the checkpoint files
W0 = A.load_w0()
from safetensors.torch import load_file  # noqa: E402

sd = load_file(os.path.join(A.MODEL_DIR, "model.safetensors"))
kw = [k for k in sd if k.endswith("h.17.mlp.c_proj.weight")]
kb = [k for k in sd if k.endswith("h.17.mlp.c_proj.bias")]
assert len(kw) == 1 and len(kb) == 1, (kw, kb)
bias = sd[kb[0]].to(F64)
a = {"stored_W0_shape": list(W0.shape), "stored_W0_dtype": str(W0.dtype), "stored_W0_sha256": A.tensor_sha256(W0),
     "safetensors_key": kw[0], "safetensors_dtype": str(sd[kw[0]].dtype),
     "equals_model.safetensors_tensor_bitwise": bool(torch.equal(sd[kw[0]], W0))}
pb = torch.load(os.path.join(A.MODEL_DIR, "pytorch_model.bin"), map_location="cpu", weights_only=True)
kw2 = [k for k in pb if k.endswith("h.17.mlp.c_proj.weight")]
a["equals_pytorch_model.bin_tensor_bitwise"] = bool(torch.equal(pb[kw2[0]], W0))
# the two checkpoint files hold the same parameters?  (transformers 4.23.1 reads model.safetensors when it exists; the
# public environment, transformers 4.15.0, reads pytorch_model.bin)
common = sorted(set(sd) & set(pb))
neq = [k for k in common if not torch.equal(sd[k], pb[k])]
a["checkpoint_files"] = {"n_tensors_safetensors": len(sd), "n_tensors_bin": len(pb), "n_common": len(common),
                         "n_common_not_bitwise_equal": len(neq), "not_equal": neq[:10],
                         "only_in_safetensors": sorted(set(sd) - set(pb))[:10], "only_in_bin_n": len(set(pb) - set(sd)),
                         "only_in_bin_examples": sorted(set(pb) - set(sd))[:6],
                         "only_in_bin_are_all_attention_mask_buffers": all(k.endswith(".attn.bias") or k.endswith(".attn.masked_bias") for k in set(pb) - set(sd))}
del pb
out["pre_edit_weight"] = a
A.say("W0:", a)

data = A.load_counterfact()
from transformers import AutoTokenizer  # noqa: E402  (environment's transformers: tokenizer only)

tok = AutoTokenizer.from_pretrained(A.MODEL_DIR)


def subject_last_index(prompt, subject):
    text = prompt.format(subject)
    c0 = prompt.index("{}")
    c1 = c0 + len(subject)
    enc = tok(text, return_offsets_mapping=True)
    return max(i for i, (x, y) in enumerate(enc["offset_mapping"]) if x < c1 and y > c0), len(enc["input_ids"])


def quant(x):
    x = np.asarray(x, dtype=np.float64)
    q = np.quantile(x, [0, 0.05, 0.5, 0.95, 1])
    return {"min": float(q[0]), "q05": float(q[1]), "median": float(q[2]), "q95": float(q[3]), "max": float(q[4])}


FAC = {src: A.load_factors(src) for src in ("dev", "stream")}
PER = {}
for src, fac in FAC.items():
    n = fac["left"].shape[0]
    first = {"dev": 15000, "stream": 0}[src]
    assert fac["record_indices"] == list(range(first, first + n)), src
    reqs_ok = all(fac["requests"][i] == data[first + i]["requested_rewrite"] and fac["case_ids"][i] == data[first + i]["case_id"] for i in range(n))
    # replay with per-update checks (float32 in place = the public write; fp64 sum next to it)
    W = W0.clone()
    S = torch.zeros(W0.shape, dtype=F64)
    W0d = W0.to(F64)
    per = []
    snaps = {}
    for i in range(n):
        left, right = fac["left"][i], fac["right"][i]
        ld, rd = left.to(F64), right.to(F64)
        k, kavg = fac["k_request"][i].to(F64), fac["k_template_mean"][i].to(F64)
        cur, tgt = fac["cur_output"][i].to(F64), fac["target"][i].to(F64)
        out_before = k @ W.to(F64) + bias
        W += left.unsqueeze(1) @ right.unsqueeze(0)
        S += torch.outer(ld, rd)
        out_after = k @ W.to(F64) + bias
        s = float(ld @ k)
        res = tgt - cur
        req = fac["requests"][i]
        pos, ntok = subject_last_index(req["prompt"], req["subject"])
        per.append({"index": i, "s": s, "w_dot_k_minus_1": float((ld / s) @ k - 1.0), "w_dot_kavg": float((ld / s) @ kavg),
                    "update_fro": float(ld.norm() * rd.norm()), "left_norm": float(ld.norm()),
                    "out_before_vs_cur_output_rel": float((out_before - cur).norm() / cur.norm()),
                    "out_after_vs_target_rel": float((out_after - tgt).norm() / tgt.norm()),
                    "right_s_vs_residual_rel": float((rd * s - res).norm() / res.norm()),
                    "subject_last_index": pos, "n_prompt_tokens": ntok, "builder_lookup_idx": fac["scalars"][i]["lookup_idxs"][0],
                    "p_new": fac["scalars"][i].get("v_prob_new"), "target_norm": float(tgt.norm()),
                    "W_minus_W0_fro_after": float((W.to(F64) - W0d).norm())})
        if (i + 1) in (100, 500, 1000, 2000):
            snaps[i + 1] = (W.clone(), S.clone())
    PER[src] = per
    for name, (s_, T, first_) in A.LOGS.items():
        if s_ != src:
            continue
        WE = A.load_snapshot(src, T)
        Wr, Sr = snaps[T]
        WEd = WE.to(F64)
        E = WEd - W0d - Sr
        dW = WEd - W0d
        sub = per[:T]
        ulp = float(np.spacing(np.float32(WE.abs().max())))
        L = {"name": name, "source": src, "T": T, "records": [first_, first_ + T - 1], "requests_equal_dataset_records": bool(reqs_ok),
             "deployed_sha256": A.tensor_sha256(WE), "replay_sha256": A.tensor_sha256(Wr),
             "float32_replay_equals_deployed_bitwise": bool(torch.equal(Wr, WE)),
             "n_entries_differing_from_replay": int((Wr != WE).sum()),
             "reconstruction_fp64": {"max_abs": float(E.abs().max()), "fro": float(E.norm()), "rel_fro_vs_update": float(E.norm() / dW.norm()),
                                     "rel_fro_vs_W0": float(E.norm() / W0d.norm()), "update_fro": float(dW.norm()), "W0_fro": float(W0d.norm()),
                                     "WE_max_abs": float(WE.abs().max()), "float32_ulp_at_WE_max_abs": ulp,
                                     "max_abs_in_ulps_at_max": float(E.abs().max()) / ulp, "WE_finite": bool(torch.isfinite(WE).all())},
             "shapes": {"W": list(WE.shape), "left": list(fac["left"][:T].shape), "right": list(fac["right"][:T].shape),
                        "outer_left_right_matches_W_without_transpose": list(torch.outer(fac["left"][0], fac["right"][0]).shape) == list(WE.shape)},
             "key_identity": {"max_abs_w_dot_k_minus_1": float(max(abs(p["w_dot_k_minus_1"]) for p in sub)),
                              "n_negative_s": int(sum(p["s"] < 0 for p in sub)), "min_abs_s": float(min(abs(p["s"]) for p in sub)),
                              "s": quant([p["s"] for p in sub]),
                              "w_dot_k_template_mean": quant([p["w_dot_kavg"] for p in sub]),
                              "share_template_mean_within_1pct": float(np.mean([abs(p["w_dot_kavg"] - 1) < 0.01 for p in sub])),
                              "share_template_mean_within_10pct": float(np.mean([abs(p["w_dot_kavg"] - 1) < 0.10 for p in sub])),
                              "max_out_before_vs_cur_output_rel": float(max(p["out_before_vs_cur_output_rel"] for p in sub)),
                              "max_out_after_vs_target_rel": float(max(p["out_after_vs_target_rel"] for p in sub)),
                              "max_right_s_vs_residual_rel": float(max(p["right_s_vs_residual_rel"] for p in sub)),
                              "left_norm_minus_1_max": float(max(abs(p["left_norm"] - 1) for p in sub))},
             "updates": {"fro": quant([p["update_fro"] for p in sub]),
                         "exploding(>|W0|_F)": [p["index"] for p in sub if p["update_fro"] > float(W0d.norm())],
                         "subject_is_first_token": [p["index"] for p in sub if p["subject_last_index"] == 0],
                         "subject_index_audit_equals_builder": int(sum(p["subject_last_index"] == p["builder_lookup_idx"] for p in sub)),
                         "min_s_among_other_requests": float(min(p["s"] for p in sub if p["subject_last_index"] != 0)),
                         "s_of_first_token_requests": [round(p["s"], 4) for p in sub if p["subject_last_index"] == 0],
                         "p_new_median": float(np.median([p["p_new"] for p in sub])),
                         "share_p_new_below_0.5": float(np.mean([p["p_new"] < 0.5 for p in sub]))}}
        # the transposed assignment, for the record: sum_i outer(right_i, left_i) is (1600, 6400) = (W_E - W_0)^T
        Dt = fac["right"][:T].to(F64).T @ fac["left"][:T].to(F64)
        L["math_orientation"] = {"sum_outer(right,left)_shape": list(Dt.shape), "rel_fro_vs_(WE-W0)^T": float((Dt - dW.T).norm() / dW.norm())}
        # f. library-format copy
        lf = os.path.join(A.B_LOGS_EDIT, "library_format", name, "edit_factors.pt")
        if os.path.exists(lf):
            d = torch.load(lf, map_location="cpu")
            U, Wv, K = d["U"].to(F64), d["W"].to(F64), d["K1_solver"].to(F64)
            s = (fac["left"][:T].to(F64) * fac["k_request"][:T].to(F64)).sum(1)
            Dl = U.T @ Wv                                              # (1600, 6400)
            L["library_format"] = {"shapes": {"U": list(U.shape), "W": list(Wv.shape), "K1_solver": list(K.shape)}, "dtype": str(d["U"].dtype),
                                   "U_equals_right_times_s_max_rel": float(((U - fac["right"][:T].to(F64) * s[:, None]).norm(dim=1) / U.norm(dim=1)).max()),
                                   "W_equals_left_over_s_max_rel": float(((Wv - fac["left"][:T].to(F64) / s[:, None]).norm(dim=1) / Wv.norm(dim=1)).max()),
                                   "K_equals_k_request_bitwise": bool(torch.equal(d["K1_solver"].float(), fac["k_request"][:T])),
                                   "max_abs_<W_i,K_i>_minus_1": float(((Wv * K).sum(1) - 1).abs().max()),
                                   "sum_outer(U,W)_vs_(WE-W0)^T_rel_fro": float((Dl - dW.T).norm() / dW.norm()),
                                   "sum_outer(U,W)_vs_(WE-W0)^T_max_abs": float((Dl - dW.T).abs().max()),
                                   "uids_first_last": [d["uids"][0], d["uids"][-1]], "n": int(U.shape[0])}
            mj = A.jload(os.path.join(A.B_LOGS_EDIT, "library_format", name, "metrics.json"))
            L["library_format"]["metrics_deployed_sha256_equal"] = mj.get("deployed_sha256") == L["deployed_sha256"]
            L["library_format"]["per_batch_is_one_edit_per_batch"] = [b["edit_ids"] for b in mj["per_batch"]] == [[i] for i in range(T)]
        out["logs"][name] = L
        A.say(name, {k: L[k] for k in ("float32_replay_equals_deployed_bitwise", "n_entries_differing_from_replay")},
              {k: f"{v:.3g}" if isinstance(v, float) else v for k, v in L["reconstruction_fp64"].items()},
              {k: (f"{v:.3g}" if isinstance(v, float) else v) for k, v in L["key_identity"].items() if not isinstance(v, dict)})
        A.jdump(out, out_path)

# ---- g. registry of the builder against the audit's numbers
reg = A.jload(os.path.join(A.B_LOGS_EDIT, "registry.json"))
cmp_ = {}
for name, L in out["logs"].items():
    e = reg[name]
    cmp_[name] = {"deployed_sha256_equal": e["deployed_sha256"] == L["deployed_sha256"],
                  "pre_edit_sha256_equal": e["pre_edit_sha256"] == out["pre_edit_weight"]["stored_W0_sha256"],
                  "builder_rel_fro_vs_update": e["reconstruction"]["rel_fro_vs_update"], "audit_rel_fro_vs_update": L["reconstruction_fp64"]["rel_fro_vs_update"],
                  "builder_max_abs": e["reconstruction"]["max_abs"], "audit_max_abs": L["reconstruction_fp64"]["max_abs"],
                  "T_equal": e["T"] == L["T"], "records_equal": e["records"] == L["records"]}
out["registry_vs_audit"] = cmp_

# ---- h. the stream around the first exploding update; requests whose subject is the first token, whole data set
per = PER["stream"]
out["stream_updates"] = {
    "exploding": [{"index": p["index"], "update_fro": p["update_fro"], "s": p["s"], "subject_last_index": p["subject_last_index"],
                   "p_new": p["p_new"], "request": data[p["index"]]["requested_rewrite"]["prompt"].format(data[p["index"]]["requested_rewrite"]["subject"])
                   + " -> " + data[p["index"]]["requested_rewrite"]["target_new"]["str"]} for p in per if p["update_fro"] > float(W0.double().norm())],
    "first_exploding_index": next(p["index"] for p in per if p["update_fro"] > float(W0.double().norm())),
    "W_minus_W0_fro": {str(n): per[n - 1]["W_minus_W0_fro_after"] for n in (100, 150, 173, 174, 200, 500, 1000, 2000)},
    "p_new_before_173": {"n": 173, "median": float(np.median([p["p_new"] for p in per[:173]])), "share_below_0.5": float(np.mean([p["p_new"] < 0.5 for p in per[:173]]))},
    "p_new_from_173": {"n": len(per) - 173, "median": float(np.median([p["p_new"] for p in per[173:]])),
                       "share_below_0.5": float(np.mean([p["p_new"] < 0.5 for p in per[173:]])),
                       "by_log": {f"T{T}": float(np.mean([p["p_new"] < 0.5 for p in per[173:T]])) for T in (500, 1000, 2000)}},
    "target_norm_median_before_173": float(np.median([p["target_norm"] for p in per[:173]])),
    "target_norm_median_from_173": float(np.median([p["target_norm"] for p in per[173:]])),
}
first_tok = []
for i, r in enumerate(data):
    rw = r["requested_rewrite"]
    if rw["prompt"].startswith("{}"):
        pos, _ = subject_last_index(rw["prompt"], rw["subject"])
        if pos == 0:
            first_tok.append(i)
out["first_token_subjects_in_dataset"] = {"n": len(first_tok), "n_records": len(data), "in_stream_0_1999": [i for i in first_tok if i < 2000],
                                          "in_dev_15000_15099": [i for i in first_tok if 15000 <= i < 15100],
                                          "in_first_1000": [i for i in first_tok if i < 1000]}
out["finished"] = A.now()
A.jdump(out, out_path)
A.say("stream:", {k: v for k, v in out["stream_updates"].items() if k != "exploding"})
A.say("exploding:", [(e["index"], round(e["update_fro"], 1), round(e["s"], 4), e["subject_last_index"]) for e in out["stream_updates"]["exploding"]])
A.say("first-token subjects:", out["first_token_subjects_in_dataset"])
