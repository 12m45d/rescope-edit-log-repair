"""Decode and teacher-forcing utilities of the E10 description (no padding anywhere).

A batch holds prompts of ONE token length, so no attention mask and no pad position exists; the batch layout is a
deterministic function of the list of prompts (make_batches), so the pre-edit and the deployed model see exactly the
same batches and a difference between them is a difference of the weights.

    greedy(model, ids, n)             greedy continuation of n tokens (argmax, key-value cache)
    teacher_forced(model, ids, cont)  argmax at the n positions of `cont` when `cont` is fed (one full forward)
Token ids are tok(prompt)["input_ids"] (GPT-2: no BOS), as the public evaluation feeds prompts.
"""
import torch

N_DEC = 10            # length of the continuation (E10: "the pre-edit greedy continuation of 10 tokens")
BATCH = 64            # prompts per batch (one token length per batch)
MAX_TOKENS = 1536     # cap on batch_size * (prompt length + N_DEC): about 1.1 MB of GPU memory per token on top of the
                      # 6.2 GB of the model (a first run with 6144 ran out of memory at 2560 tokens under an 8.7 GiB cap)


def make_batches(lengths, bs=BATCH, max_tokens=MAX_TOKENS, extra=N_DEC):
    """Indices 0 .. n-1 grouped into batches of equal length: sorted by (length, index), chunked per length."""
    order = sorted(range(len(lengths)), key=lambda i: (lengths[i], i))
    out, cur, cur_len = [], [], None
    for i in order:
        L = lengths[i]
        cap = max(1, min(bs, max_tokens // (L + extra)))
        if cur and (L != cur_len or len(cur) >= cap):
            out.append(cur)
            cur = []
        cur_len = L
        cur.append(i)
    if cur:
        out.append(cur)
    return out


@torch.no_grad()
def greedy(model, ids, n=N_DEC):
    """ids (B, L) long on the model's device -> (tokens (B, n), finite (B,) bool: every step's logits finite)."""
    out = model(input_ids=ids, use_cache=True)
    past = out.past_key_values
    lg = out.logits[:, -1]
    finite = torch.isfinite(lg).all(-1)
    nxt = lg.argmax(-1)
    toks = [nxt]
    for _ in range(n - 1):
        out = model(input_ids=nxt[:, None], past_key_values=past, use_cache=True)
        past = out.past_key_values
        lg = out.logits[:, -1]
        finite &= torch.isfinite(lg).all(-1)
        nxt = lg.argmax(-1)
        toks.append(nxt)
    return torch.stack(toks, 1), finite


@torch.no_grad()
def teacher_forced(model, ids, cont):
    """ids (B, L), cont (B, n) -> (pred (B, n): argmax at the position that predicts cont[:, j] when
    [ids, cont[:, :j]] is the input; finite (B,) bool)."""
    L = ids.shape[1]
    full = torch.cat([ids, cont[:, :-1]], 1)
    lg = model(input_ids=full, use_cache=False).logits[:, L - 1:]
    return lg.argmax(-1), torch.isfinite(lg).all(-1).all(-1)


def degenerate_flags(tok_ids, text):
    """Flags of a continuation (list of N_DEC token ids, decoded text): empty / repetitive (fixed definitions)."""
    n_distinct = len(set(tok_ids))
    run = best = 1
    for a, b in zip(tok_ids, tok_ids[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return {"empty": text.strip() == "", "n_distinct": n_distinct, "max_run": best,
            "few_distinct": n_distinct <= 3,            # at most 3 distinct tokens among the 10
            "long_run": best >= 4,                      # one token at least 4 times in a row
            "repetitive": n_distinct <= 3 or best >= 4}
