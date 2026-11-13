"""Format check of the converted logs against the frozen library `rescope` -- DEVELOPMENT LOG ONLY
(TX-G2XL-rome-dev100; no reported log is touched by a gate here).

CPU, log quantities only:
  a. own coordinate s_i = <W_i, k_i> of the library log (key 'request'): 1 up to round-off; with key 'template_mean':
     the distribution reported in the audit.
  b. EditLog.delta() (math orientation, d_out x d_in) against (W_E - W_0)^T.
  c. rebase lemma: V_i = U_i + sum_{j < i} U_j <W_j, k_i> against the residual to the PRE-EDIT weight computed
     independently, target_i - (k_request_i @ W_0 + b).
GPU, the library's hook on the Conv1D module transformer.h.17.mlp.c_proj:
  d. thresholds +inf: logits bitwise equal to the pre-edit model (nothing fires).
  e. linear replay (role 'all', fire 'all', response 'lin', thresholds -inf) against the deployed model: the scoped log
     with every edit released everywhere is the deployed log up to the write round-off.
  f. replacement response at the request's own solver key (role 'all', fire 'top1', response 'repl', one shared
     threshold 0.999): the module output at the last subject token of the request prompt equals the logged target
     where the own edit is the released one; the number of requests on which another edit is released there is
     counted, not tuned.
Writes out/library_check.json.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tp_common as C  # noqa: E402
import tx_public as TX  # noqa: E402

NAME = TX.DEV_LOG
assert NAME == "TX-G2XL-rome-dev100"
model, tok = TX.load_model()
import torch  # noqa: E402

C.cap_gpu()
sys.path.insert(0, TX.LIB)
import rescope  # noqa: E402
from rescope import Hook, ScopedLog, rebase, rebase_error  # noqa: E402

import tp_eval as E  # noqa: E402

INF = float("inf")
out = {"log": NAME, "written": C.now(), "rescope_file": rescope.__file__}
log = TX.from_public(NAME)
log_t = TX.from_public(NAME, key="template_mean")
W0, WE = TX.weights(NAME)
mod = TX.module(model)
b = mod.bias.detach().cpu().double()
T = log.T

# a. own coordinates
s, st = log.own(), log_t.own()
out["a_own"] = {"request_max_abs_dev_from_1": float((s - 1).abs().max()),
                "template_mean": {"min": float(st.min()), "median": float(st.median()), "max": float(st.max())}}
# b. delta
D = log.delta()                                             # (1600, 6400)
dW = (WE.double() - W0.double()).T
out["b_delta"] = {"max_abs": float((D - dW).abs().max()), "rel_fro": float((D - dW).norm() / dW.norm()), "shape": list(D.shape)}
# c. rebase lemma
tgt = log.meta["target"] - (log.meta["K_request"] @ W0.double() + b)
re_ = rebase_error(log, tgt)
out["c_rebase"] = {"rebased_max": re_["rebased"]["max"], "rebased_median": re_["rebased"]["median"],
                   "not_rebased_max": re_["not_rebased"]["max"], "not_rebased_median": re_["not_rebased"]["median"],
                   "not_rebased_later_batches_median": re_["not_rebased_later_batches"]["median"]}

# prompts: the dev request prompts, their paraphrases, 100 HELD-OUT benign prompts
PT = TX.prompt_texts()
req = TX.log_prompts(NAME, "request")
par = TX.log_prompts(NAME, "paraphrase")
ben = TX.benign("heldout", NAME)[:100]
rows = req + par + ben
dev = "cuda"
logc = log.to(dev)
W0c = W0.to(dev)


@torch.no_grad()
def logits_of(ids):
    return model(input_ids=torch.tensor([ids], device=dev)).logits[0]


TX.set_weight(model, W0)
pre = [logits_of(PT["input_ids"][r["text_id"]]) for r in rows]
TX.set_weight(model, WE)
dep = [logits_of(PT["input_ids"][r["text_id"]]) for r in rows]
TX.set_weight(model, W0)

# d. thresholds +inf
S_off = ScopedLog.from_log(logc, role="all", fire="top1", response="repl", tau=INF, W0=W0c)
n_bit = 0
with Hook(mod, S_off) as hk:
    for r, p in zip(rows, pre):
        hk.begin(None)
        n_bit += int(torch.equal(logits_of(PT["input_ids"][r["text_id"]]), p))
out["d_off"] = {"n_prompts": len(rows), "n_bitwise_equal_to_pre_edit": n_bit}

# e. linear replay against the deployed model
S_lin = ScopedLog.from_log(logc, role="all", fire="all", response="lin", tau=-INF, W0=W0c)
mx, same_top, mx_pre = [], 0, []
with Hook(mod, S_lin) as hk:
    for r, d_, p in zip(rows, dep, pre):
        hk.begin(None)
        lg = logits_of(PT["input_ids"][r["text_id"]])
        mx.append(float((lg - d_).abs().max()))
        mx_pre.append(float((d_ - p).abs().max()))
        same_top += int(bool((lg.argmax(-1) == d_.argmax(-1)).all()))
out["e_linear_replay"] = {"n_prompts": len(rows), "max_abs_logit_diff_to_deployed": max(mx), "median_abs_logit_diff_to_deployed": sorted(mx)[len(mx) // 2],
                          "n_same_argmax_at_every_position": same_top,
                          "for_scale_median_max_abs_logit_diff_deployed_vs_pre_edit": sorted(mx_pre)[len(mx_pre) // 2]}
assert torch.equal(mod.weight.detach().cpu(), W0), "the hook did not restore the module weight"

# f. replacement response at the own solver key
fac = TX.factors(NAME)
S_rep = ScopedLog.from_log(logc, role="all", fire="top1", response="repl", tau=0.999, W0=W0c)
cap = {}
n_own, rel_err, r_own, others = 0, [], [], []
with Hook(mod, S_rep) as hk:
    h2 = mod.register_forward_hook(lambda m, a, o: cap.__setitem__("out", o.detach()))
    for i, r in enumerate(req):
        assert r["pos"] == i
        ids = PT["input_ids"][r["text_id"]]
        p = fac["scalars"][i]["lookup_idxs"][0]
        hk.begin(None)
        logits_of(ids)
        f = hk.last
        sel = (f.pos == p).nonzero().flatten() if f is not None else torch.zeros(0, dtype=torch.long)
        if sel.numel() == 0:
            others.append({"edit": i, "released": None})
            continue
        j = int(f.edit[sel[0]])
        r_own.append(float(f.r[sel[0], i]))
        if j == i:
            n_own += 1
            o = cap["out"][0, p].double().cpu()
            t = fac["target"][i].double()
            rel_err.append(float((o - t).norm() / t.norm()))
        else:
            others.append({"edit": i, "released": j, "r_own": float(f.r[sel[0], i]), "r_released": float(f.r[sel[0], j])})
    h2.remove()
out["f_repl_own_key"] = {"n_requests": T, "threshold": 0.999, "n_own_edit_released": n_own,
                         "max_rel_err_output_vs_target": max(rel_err) if rel_err else None,
                         "own_coordinate_min": min(r_own) if r_own else None, "own_coordinate_max": max(r_own) if r_own else None,
                         "n_other_or_none_released": len(others), "others": others}
out["summary_lines"] = [
    f"log {NAME} through rescope.EditLog / ScopedLog / Hook (library unchanged): own coordinate max |s - 1| = {out['a_own']['request_max_abs_dev_from_1']:.1e}; "
    f"delta() vs (W_E - W_0)^T rel. Frobenius {out['b_delta']['rel_fro']:.1e}",
    f"rebase lemma: rebased value vs residual to the pre-edit weight, max rel. error {out['c_rebase']['rebased_max']:.1e} "
    f"(not rebased: median {out['c_rebase']['not_rebased_median']:.2f}, max {out['c_rebase']['not_rebased_max']:.2f})",
    f"hook with thresholds +inf: {n_bit} of {len(rows)} prompts bitwise equal to the pre-edit model",
    f"linear replay (all edits released everywhere) vs deployed model: max |logit diff| {out['e_linear_replay']['max_abs_logit_diff_to_deployed']:.2e} "
    f"(deployed vs pre-edit, median of max: {out['e_linear_replay']['for_scale_median_max_abs_logit_diff_deployed_vs_pre_edit']:.1f}); "
    f"same argmax at every position on {same_top} of {len(rows)} prompts",
    f"replacement response at the request's own key (threshold 0.999): own edit released on {n_own} of {T} requests, "
    f"module output vs logged target max rel. error {out['f_repl_own_key']['max_rel_err_output_vs_target'] if rel_err else float('nan'):.1e}; "
    f"another or no edit released on {len(others)}",
]
C.jdump(out, os.path.join(C.OUT, "library_check.json"))
for line in out["summary_lines"]:
    C.say(line)
