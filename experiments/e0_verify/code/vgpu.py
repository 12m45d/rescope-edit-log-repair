"""e0_verify: my own model-side code.  No rescope import.  The only shared code is the testbed's model wrapper
(vendor_v7.vlm.VLM: checkpoint loading, tokenisation, batch building), which DEFINES the model inputs of every log.

  KeyTap          own forward hook on the edited module: the module input of one forward (optionally truncating the forward)
  tf_forward      own teacher-forced B = 1 forward: argmax at the answer-predicting positions
  Replay          own scoped replay (PROTOCOL.md section 0 + amendment of 2026-11-10 17:10): role last, top-1, REPL with rebased values
  my_decode       own decoder v2 (10 greedy ids from full no-cache forwards + greedy KV-cache continuation to 48 tokens)
  norm_answer / contains_ref / dual_match     own answer normalisation and the DUAL rule
"""
from __future__ import annotations

import os
import re
import sys
import unicodedata

import numpy as np

import vcommon as V

if V.LIB not in sys.path:
    sys.path.insert(0, V.LIB)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch  # noqa: E402

torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def cap_gpu(default_mem):
    """Cap this process at GQ_MEM - 600 MiB (README of the queue)."""
    mem = int(os.environ.get("GQ_MEM", default_mem))
    cap = (mem - 600) * 2 ** 20
    torch.cuda.set_per_process_memory_fraction(min(1.0, cap / torch.cuda.get_device_properties(0).total_memory), 0)
    return mem, cap / 2 ** 30


def own_vlm(backbone):
    from vendor_v7.vlm import VLM
    return VLM(backbone, device="cuda:0")


class _Stop(Exception):
    pass


class KeyTap:
    """Records the input of the edited module in every forward (list of (1, Tn, d) tensors, module dtype).
    stop=True truncates the forward right after the module (the input does not depend on anything after it)."""

    def __init__(self, module, stop=False):
        self.module, self.stop, self.x = module, stop, []

    def __enter__(self):
        self.x = []
        self.h = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        self.h.remove()
        return False

    def _hook(self, mod, args, out):
        self.x.append(args[0].detach())
        if self.stop:
            raise _Stop


def run_truncated(vlm, inputs, tap):
    try:
        vlm.m(**inputs, use_cache=False, logits_to_keep=1)
    except _Stop:
        pass
    return tap.x[-1]


def extend(inputs, ids):
    """Model inputs of a B = 1 prompt with token ids appended."""
    if not ids:
        return dict(inputs)
    x = dict(inputs)
    t = torch.tensor([ids], dtype=torch.long, device=inputs["input_ids"].device)
    x["input_ids"] = torch.cat([inputs["input_ids"], t], 1)
    x["attention_mask"] = torch.cat([inputs["attention_mask"], torch.ones_like(t)], 1)
    if "mm_token_type_ids" in inputs:
        x["mm_token_type_ids"] = torch.cat([inputs["mm_token_type_ids"], torch.zeros_like(t).to(inputs["mm_token_type_ids"].dtype)], 1)
    return x


def build(vlm, q, answer=None):
    """B = 1 batch of a query q = dict(fmt, prompt, image) (+ teacher-forced answer)."""
    img = q.get("image")
    return vlm.build_batch([img] if img else [None], [q["prompt"]], answers=[answer] if answer else None, fmt=q["fmt"])


@torch.no_grad()
def tf_forward(vlm, batch):
    """Teacher-forced forward of a B = 1 batch with an answer -> (pred ids (n,), answer ids (n,), finite).  The n answer tokens
    sit at the end of the sequence, so the logits that predict them are those of the n positions before the last one."""
    n = int(batch["answer_len"][0])
    assert n > 0 and int(batch["answer_start"][0]) + n == int(batch["seq_len"][0])
    lg = vlm.m(**batch["inputs"], use_cache=False, logits_to_keep=n + 1).logits[0, :n]
    return lg.argmax(-1), batch["answer_ids"][0, :n], bool(torch.isfinite(lg).all())


def score(pred, ans, finite, pre_pred=None):
    """exact / token accuracy (0 if a logit is not finite); tok_loc = share of positions with the pre-edit argmax."""
    eq = (pred == ans) if finite else torch.zeros_like(ans, dtype=torch.bool)
    out = {"exact": bool(eq.all()), "token_acc": float(eq.float().mean()), "n_tok": int(ans.numel()), "finite": finite}
    if pre_pred is not None:
        out["tok_loc"] = float((pred == pre_pred).float().mean()) if finite else 0.0
    return out


# ----------------------------------------------------------------------------- own scoped replay
def rebased_values(U, W, K, batch):
    """V_j = U_j + sum over edits l of STRICTLY EARLIER batches of U_l <W_l, k_j> (k_j = stored solver key).  numpy fp64."""
    M = K @ W.T                                                     # M[j, l] = <W_l, k_j>
    earlier = batch[None, :] < batch[:, None]                       # [j, l]: l written in an earlier batch than j
    return U + np.where(earlier, M, 0.0) @ U


class Replay:
    """Scoped replay of the log on the edited module (the model must hold W_0).  At the last prompt position lp of the sequence:
    feature r_i(k); candidates r_i - b_i > eta (or r_i >= cap_i for a capped edit); the candidate with the largest margin
    releases: output <- dtype(fp64(W_0 k) + V_j).  Decision in numpy fp64 on the CPU; nothing fires -> output untouched."""

    def __init__(self, module, feature, W, K, b, eta, cap, values):
        self.module, self.feature, self.W, self.K, self.b, self.eta, self.cap = module, feature, W, K, b, eta, cap
        self.values = torch.from_numpy(values).to(module.weight.device)
        self.lp, self.fired, self.top, self.calls = None, False, -1, 0

    def __enter__(self):
        self.h = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        self.h.remove()
        return False

    def begin(self, lp):
        self.lp, self.fired, self.top, self.calls = int(lp), False, -1, 0

    def _hook(self, mod, args, out):
        self.calls += 1
        if args[0].shape[1] <= self.lp:                              # a KV-cache step of a decode: the last prompt position is not in the input
            return None
        k = args[0][0, self.lp].detach().double().cpu().numpy()[None]
        r = V.FEATURES[self.feature](k, self.W, self.K)
        fired, top, _ = V.naive_decide(r, self.b, self.eta, self.cap)
        if not bool(fired[0]):
            return None
        self.fired, self.top = True, int(top[0])
        new = out.clone()
        new[0, self.lp] = (out[0, self.lp].double() + self.values[self.top]).to(out.dtype)
        return new


# ----------------------------------------------------------------------------- own decoder v2
def eos_ids(vlm):
    e = vlm.m.generation_config.eos_token_id
    return sorted(set(e if isinstance(e, (list, tuple)) else [e]) | {vlm.tok.eos_token_id})


@torch.no_grad()
def my_decode(vlm, batch, max_new=10, max_long=48):
    """Decoder v2 on a prompt-only B = 1 batch: the first max_new greedy ids from FULL forwards without KV cache (stop at EOS
    only); then a plain greedy KV-cache continuation up to max_long tokens in total (fresh generation config: no sampling,
    no repetition penalty).  -> dict(y = first line of the first max_new ids, long = all ids decoded, line = first line of long)."""
    from transformers import GenerationConfig
    tok, eos = vlm.tok, eos_ids(vlm)
    ids, stop = [], "max_new"
    for _ in range(max_new):
        lg = vlm.m(**extend(batch["inputs"], ids), use_cache=False, logits_to_keep=1).logits[0, -1]
        t = int(lg.argmax(-1))
        if t in eos:
            stop = "eos"
            break
        ids.append(t)
    more, lstop = [], stop
    if stop != "eos" and max_long > len(ids):
        gc = GenerationConfig(max_new_tokens=max_long - len(ids), do_sample=False, num_beams=1, repetition_penalty=1.0, eos_token_id=eos,
                              pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else eos[0])
        x = extend(batch["inputs"], ids)
        new = vlm.m.generate(**x, generation_config=gc)[0, x["input_ids"].shape[1]:].tolist()
        lstop = "max_new"
        for t in new:
            if t in eos:
                lstop = "eos"
                break
            more.append(t)
    raw = tok.decode(ids, skip_special_tokens=True)
    body = raw.lstrip()
    if "\n" in body:
        body = body.split("\n", 1)[0]
    long = tok.decode(ids + more, skip_special_tokens=True).strip()
    return {"y": body.strip(), "long": long, "line": long.split("\n")[0].strip(), "ids10": ids, "n_long": len(ids) + len(more), "stop": stop, "long_stop": lstop}


# ----------------------------------------------------------------------------- own answer normalisation and DUAL
def norm_answer(s):
    """Lower case, accents stripped, every run of non-alphanumerics -> one space, articles a / an / the removed."""
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"[^0-9a-z]+", " ", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def ref_core(ref):
    core = re.sub(r"\s*\([^()]*\)\s*$", "", ref or "")
    return core if core.strip() else (ref or "")


def dual_match(targets, line, long, variant):
    """Edits whose target the deployed answer matches: 'eq' = normalised first line equals the normalised target;
    'contains' = the normalised target (trailing parenthetical removed) occurs in the normalised 48-token answer at word boundaries."""
    if variant == "eq":
        nl = norm_answer(line)
        return [i for i, t in enumerate(targets) if norm_answer(t) and norm_answer(t) == nl]
    p = f" {norm_answer(long)} "
    return [i for i, t in enumerate(targets) if norm_answer(ref_core(t)) and f" {norm_answer(ref_core(t))} " in p]


def checkpoint_weight(backbone, layer):
    """The edited module's weight read straight from the checkpoint's safetensors files -> (tensor, key, file)."""
    from safetensors import safe_open
    d = {"qwen25vl": V.EXT_VIEW + "/data/models/Qwen2.5-VL-3B", "llava": V.EXT_VIEW + "/data/models/llava-1.5-7b"}[backbone]
    suffix = f"layers.{layer}.mlp.down_proj.weight"
    for f in sorted(os.listdir(d)):
        if f.endswith(".safetensors"):
            with safe_open(os.path.join(d, f), "pt") as sf:
                for k in sf.keys():
                    if k.endswith(suffix) and "vis" not in k:
                        return sf.get_tensor(k), k, f
    raise FileNotFoundError(suffix)


def deployed_weight(name, W0):
    """W_E from the log's files: (fp32(W_0) + sum of the stored applied batch deltas, in batch order).to(dtype); and whether
    fp32(W_E) - fp32(W_0) equals the stored composite delta."""
    met = V.jload(f"{V.TESTBED}/{name}/metrics.json")
    S = torch.zeros(W0.shape, dtype=torch.float32, device=W0.device)
    for b in range(len(met["per_batch"])):
        S += torch.load(f"{V.TESTBED}/{name}/batch_deltas/batch_{b:03d}.pt", map_location="cpu").to(W0.device)
    WE = (W0.float() + S).to(W0.dtype)
    comp = torch.load(f"{V.TESTBED}/{name}/composite_delta.pt", map_location="cpu").to(W0.device)
    return WE, bool(torch.equal(WE.float() - W0.float(), comp))
