"""Backbone adapter: GPT-2 XL (float32) with the edited module transformer.h.17.mlp.c_proj under a METHOD.

ORIENTATION.  The edited module is a transformers Conv1D: weight (d_in = 6400, d_out = 1600), y = x @ W + b.  The library
never multiplies by the module weight: its Hook reads the module input x (B, Tn, 6400) and adds values / log terms to the
module's own output (B, Tn, 1600).  Detectors and keys are 6400-vectors, values and stored residuals 1600-vectors, so the
library works on the Conv1D unchanged.  Only FULL WEIGHTS carry the orientation: every weight handed to this module is in
the MODULE's orientation (6400, 1600) = tx_public.weights(name); the library's EditLog.delta() is its transpose.

A METHOD is
    None | "preedit"         the pre-edit weight W_0
    a tensor (6400, 1600)    a full replacement weight of the edited module (the deployed weight W_E, a prefix replay, ...)
    a gate                   a rescope.ScopedLog (or any object with firing / respond): W_0 + the library's forward hook

    B = Backbone()                               # loads the model as the build did (tx_public.load_model: transformers 4.23.1)
    with B.served(method) as srv:                # every call of B.model inside is served by the method
        pred, fin = B.tf(ids, conts)             # one right-padded batch, teacher forcing: argmax at the predicting positions
        toks, fin = B.decode(ids)                # one right-padded batch, greedy decode with the key-value cache
        rows = srv.rows()                        # per-row firing flags of the LAST pass (None for weights)
Any other code that calls B.model(...) inside the context (the public evaluation functions, generate_fast) is served too: a
forward pre-hook of the model reads `attention_mask` / `past_key_values` of each call, so that PAD positions never fire and
the rows' positions continue through a key-value-cache decode.

BATCHES.  Rows are right-padded with the tokenizer's pad id and masked; GPT-2 is causal with absolute positions, so a real
position never attends to a PAD position and its position index is its index in the row.  In a decode the new token of every
row is appended after the padded prompt with an attention mask over the real tokens and explicit position ids (L_i + t).
A batch without padding is fed without mask (the very calls of the build's tp_eval).  A padded pass equals the single pass
up to the round-off of another batch layout (tests g_batch: same tokens, same firing; logits differ by ~1e-4 absolute).
"""
from __future__ import annotations

import os

import torch

import e10t
from rescope.scoped import Hook

MAX_TOKENS = 1536          # cap on rows * (longest row + N_DEC) of one batch (about 1.1 MiB of GPU memory per token: key-value cache)
MAX_ROWS = 64


class _Stop(Exception):
    pass


def batches(lengths, max_tokens: int = MAX_TOKENS, max_rows: int = MAX_ROWS, extra: int = e10t.N_DEC) -> list:
    """Indices 0 .. n-1 grouped into batches: sorted by (length, index) and cut so that rows * (longest + extra) <= max_tokens
    and rows <= max_rows.  Deterministic in the list of lengths; padding stays small because neighbours have similar lengths."""
    order = sorted(range(len(lengths)), key=lambda i: (lengths[i], i))
    out, cur = [], []
    for i in order:
        if cur and (len(cur) >= max_rows or (len(cur) + 1) * (lengths[i] + extra) > max_tokens):
            out.append(cur)
            cur = []
        cur.append(i)
    if cur:
        out.append(cur)
    return out


class RowHook(Hook):
    """rescope.Hook (its decision and response, unchanged) with per-ROW bookkeeping for padded batches and key-value-cache
    continuations.  start(B, seq_len): a new batch of B sequences (seq_len: real tokens of every row in the next module call;
    PAD positions never fire).  cont(seq_len): the next module call continues every row after the tokens seen so far.
    rows(): per-row flags since start()."""

    def __init__(self, module, gate):
        super().__init__(module, gate, set_weight=False, record=True)
        self.start(0)

    def start(self, B: int, seq_len=None):
        self.begin(None, gen=False, seq_len=seq_len)        # the library's own reset; lp = None (role 'all' needs none)
        self.B = int(B)
        self._base = torch.zeros(self.B, dtype=torch.long)  # real tokens of every row before the current module call
        self._rel = [[] for _ in range(self.B)]             # per row: (position, releasing edit | -1, number of candidate edits)
        self._cand = None                                   # (B, T) candidate edits anywhere (float accumulator)
        self.n_module_calls = 0

    def cont(self, seq_len=None):
        self.seq_len = None if seq_len is None else torch.as_tensor(seq_len).reshape(-1)

    def _hook(self, mod, args, out):
        x = args[0]
        if x.dim() != 3 or x.shape[0] != self.B:
            raise ValueError(f"module input {tuple(x.shape)} for a batch of {self.B} rows: call start(B) before a new batch")
        res = super()._hook(mod, args, out)                 # decision (+ PAD mask) and response of the library
        n = x.shape[1]
        self._base += n if self.seq_len is None else self.seq_len.cpu()
        self.n_module_calls += 1
        return res

    def _record(self, f):
        g = self.gate
        tpos = f.pos.cpu() + self._base[f.row.cpu()]        # f.pos = index inside this call (the library's pos0 is 0 here)
        ce = g._edit_any(f.cand) if f.cand is not None and hasattr(g, "_edit_any") else None
        nc = ce.sum(1).tolist() if ce is not None else [0] * tpos.numel()
        if ce is not None:
            if self._cand is None:
                self._cand = torch.zeros(self.B, ce.shape[1], dtype=torch.float32, device=ce.device)
            self._cand.index_add_(0, f.row, ce.to(torch.float32))
        for r, p, e, c in zip(f.row.tolist(), tpos.tolist(), f.edit.tolist(), nc):
            self._rel[r].append((p, e, c))

    def rows(self, own=None, max_list: int = 64) -> list:
        """Per-row flags: fired, n_fire_pos, rel [(position, releasing edit)] (fire 'top1'; None under fire 'all'), rel_edits (edits
        released at some position; fire 'all': the candidate edits), cand_edits (candidate edits anywhere), n_rel_edits,
        n_cand_edits.  An edit set with more than max_list members is given as a bit mask instead of a list ({'hex': ...}: bit e
        = edit e; nothing is truncated): read every set with edit_in / the n_* counts.  With own (one edit index or None per
        row): own_cand (candidate at some position), own_released (released at some position), other_released (another edit
        is released at some position)."""
        top1 = getattr(self.gate, "fire", "top1") == "top1"
        cand = None if self._cand is None else (self._cand > 0).cpu()
        out = []
        for r in range(self.B):
            rel = sorted(self._rel[r])
            ce = cand[r].nonzero().flatten().tolist() if cand is not None else []
            re_ = sorted({e for _, e, _ in rel}) if top1 else ce
            row = {"fired": bool(rel), "n_fire_pos": len(rel), "rel": [[p, e] for p, e, _ in rel] if top1 else None,
                   "rel_edits": pack_edits(re_, max_list), "n_rel_edits": len(re_), "n_cand_edits": len(ce),
                   "cand_edits": pack_edits(ce, max_list), "max_cand_at_a_position": max([c for _, _, c in rel], default=0)}
            if own is not None and own[r] is not None:
                o = int(own[r])
                row.update(own_cand=o in ce, own_released=o in re_, other_released=any(e != o for e in re_))
            out.append(row)
        return out


def pack_edits(edits, max_list: int = 64):
    """A set of edit indices as a sorted list, or (more than max_list members) as {'hex': bit mask}: lossless and bounded."""
    edits = sorted(int(e) for e in edits)
    if len(edits) <= max_list:
        return edits
    m = 0
    for e in edits:
        m |= 1 << e
    return {"hex": format(m, "x")}


def edit_in(packed, e: int) -> bool:
    """Is edit e in a set written by pack_edits?"""
    if isinstance(packed, dict):
        return bool((int(packed["hex"], 16) >> int(e)) & 1)
    return int(e) in packed


class Served:
    """The model under a method (see the module docstring).  Use through Backbone.served(method)."""

    def __init__(self, B, method):
        self.B, self.method = B, method
        self.hook, self._h = None, None
        self.kind = "preedit" if (method is None or (isinstance(method, str) and method == "preedit")) else ("weight" if torch.is_tensor(method) else "gate")
        if self.kind == "gate" and not (hasattr(method, "firing") and hasattr(method, "respond")):
            raise TypeError(f"not a method: {type(method).__name__}")

    def __enter__(self):
        B = self.B
        if B._served is not None:
            raise RuntimeError("the backbone already serves a method (contexts do not nest)")
        B.check_w0()
        if self.kind == "weight":
            if tuple(self.method.shape) != tuple(B.W0.shape):
                raise ValueError(f"replacement weight {tuple(self.method.shape)}: the module's orientation is {tuple(B.W0.shape)} (Conv1D: d_in, d_out)")
            B.set_weight(self.method)
        elif self.kind == "gate":
            g = self.method
            if getattr(g, "W0", None) is not None:
                raise ValueError("build text gates without W0 (the backbone holds and verifies the pre-edit weight)")
            if hasattr(g, "guard_key") and g.guard_key.shape[0]:            # a forward needs no guard keys
                g = g.replace(guard_key=g.guard_key[:0], guard_edit=g.guard_edit[:0])
            dev = getattr(g, "device", None)
            self.gate = g if dev is None or dev == B.dev else g.to(B.dev)
            self.hook = RowHook(B.mod, self.gate)
            self.hook.__enter__()
            self._h = B.model.register_forward_pre_hook(self._pre, with_kwargs=True)
        B._served = self
        return self

    def __exit__(self, *exc):
        B = self.B
        if self._h is not None:
            self._h.remove()
        if self.hook is not None:
            self.hook.__exit__(*exc)
        if self.kind == "weight":
            B.set_weight(B.W0)
        B._served = None
        B.check_w0()
        return False

    def _pre(self, module, args, kwargs):
        """Before every model call: tell the hook which positions of the call are real and whether the rows continue."""
        ids = kwargs.get("input_ids", args[0] if args else None)
        if ids is None:
            raise ValueError("served model calls must pass input_ids")
        am, past = kwargs.get("attention_mask"), kwargs.get("past_key_values")
        Bn, n = ids.shape
        seq_len = None
        if am is not None:
            cur = am[:, -n:] if past is not None else am                    # mask of the tokens this call feeds
            if cur.shape[1] != n:
                raise ValueError("attention_mask does not cover the tokens of the call")
            sl = cur.sum(1)
            if not bool((cur.bool() == (torch.arange(n, device=cur.device)[None, :] < sl[:, None])).all()):
                raise NotImplementedError("only right padding is supported (the real tokens of a row must be a prefix of the call)")
            if not bool((sl == n).all()):
                seq_len = sl
        if past is None:
            self.hook.start(Bn, seq_len)
        else:
            if Bn != self.hook.B:
                raise ValueError("a key-value-cache step must continue the rows of the last full forward")
            self.hook.cont(seq_len)
        return None

    def rows(self, own=None):
        """Per-row firing flags of the passes since the last model call without past_key_values (None: not a gate)."""
        return None if self.hook is None else self.hook.rows(own)


class Backbone:
    def __init__(self, device: str = "cuda", max_tokens: int = MAX_TOKENS, max_rows: int = MAX_ROWS):
        import sys
        import tp_common as C
        import tx_public as TX
        if "transformers" in sys.modules:                                   # e.g. after e10t.data.tokenizer(): the public stack is already set up
            import transformers
            if transformers.__version__ != "4.23.1":
                raise RuntimeError(f"transformers {transformers.__version__} is already imported: create the Backbone before importing transformers "
                                   "(the logs and the public evaluation code need the build's stack, transformers 4.23.1)")
            os.chdir(C.RUN_DIR)
            self.model, self.tok = C.load_model(device)
        else:
            self.model, self.tok = TX.load_model(device)                    # sets the public stack up (before transformers is imported)
        self.cap = C.cap_gpu()
        self.C, self.TX = C, TX
        self.mod = TX.module(self.model)
        self.dev = self.mod.weight.device
        W0 = torch.load(os.path.join(C.LOGS_EDIT, "W0_transformer.h.17.mlp.c_proj.weight.pt"), map_location="cpu")["weight"]
        self.W0 = W0.to(self.dev)
        self.bias = self.mod.bias.detach().clone()
        self.pad = int(self.tok.pad_token_id)
        self.max_tokens, self.max_rows = max_tokens, max_rows
        self._served = None
        self.n_forwards = 0                                                 # model calls made through tf / decode / keys
        self.n_sequences = 0
        self.check_w0()
        if type(self.mod).__name__ != "Conv1D" or tuple(self.mod.weight.shape) != (6400, 1600):
            raise RuntimeError("the edited module is not the Conv1D (6400, 1600) of GPT-2 XL")
        self.model.eval()

    # ------------------------------------------------------------------ weights
    def check_w0(self):
        if not torch.equal(self.mod.weight.data, self.W0):
            raise RuntimeError("the edited module does not hold the pre-edit weight W_0")

    def set_weight(self, W):
        with torch.no_grad():
            self.mod.weight.copy_(W.to(self.dev, torch.float32))

    def served(self, method) -> Served:
        return Served(self, method)

    # ------------------------------------------------------------------ batches
    def _pad(self, seqs):
        lens = [len(s) for s in seqs]
        mx = max(lens)
        ids = torch.full((len(seqs), mx), self.pad, dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, :lens[i]] = torch.as_tensor(s, dtype=torch.long)
        lens = torch.tensor(lens)
        mask = None if bool((lens == mx).all()) else (torch.arange(mx)[None, :] < lens[:, None]).long().to(self.dev)
        return ids.to(self.dev), mask, lens.to(self.dev)

    def batches(self, lengths, extra: int = e10t.N_DEC):
        return batches(lengths, self.max_tokens, self.max_rows, extra)

    @torch.no_grad()
    def tf(self, seqs, conts, feed_last: bool = e10t.FEED_LAST, return_logits: bool = False):
        """ONE batch, teacher forcing.  seqs: prompts (lists of token ids); conts: continuations, one per row, all of the same
        length n.  The model is fed [prompt, cont] (feed_last: all n tokens, else the first n - 1) and the argmax is read at
        the n positions that predict cont[j] (L_i - 1 + j).  -> (pred (B, n) long cpu, finite (B,) bool cpu[, logits (B, n, V)])."""
        n = len(conts[0])
        if any(len(c) != n for c in conts) or len(conts) != len(seqs):
            raise ValueError("one continuation per row, all of the same length")
        full = [list(s) + list(c if feed_last else c[:-1]) for s, c in zip(seqs, conts)]
        ids, mask, _ = self._pad(full)
        L = torch.tensor([len(s) for s in seqs], device=self.dev)
        kw = {} if mask is None else {"attention_mask": mask}
        lg = self.model(input_ids=ids, use_cache=False, **kw).logits
        idx = (L - 1)[:, None] + torch.arange(n, device=self.dev)[None, :]
        lg = lg[torch.arange(len(seqs), device=self.dev)[:, None], idx]             # (B, n, V)
        self.n_forwards += 1
        self.n_sequences += len(seqs)
        out = (lg.argmax(-1).cpu(), torch.isfinite(lg).all(-1).all(-1).cpu())
        return out + (lg,) if return_logits else out

    @torch.no_grad()
    def decode(self, seqs, n: int = e10t.N_DEC, feed_last: bool = e10t.FEED_LAST, return_logits: bool = False):
        """ONE batch, greedy decode of n tokens with the key-value cache (argmax; as the build's tp_eval.greedy when the batch
        needs no padding).  feed_last: the last token is fed once more (logits discarded), so that a hook sees every position
        of the decode.  -> (tokens (B, n) long cpu, finite (B,) bool cpu[, logits (B, n, V)])."""
        ids, mask, L = self._pad(seqs)
        Bn = len(seqs)
        kw = {} if mask is None else {"attention_mask": mask}
        out = self.model(input_ids=ids, use_cache=True, **kw)
        past = out.past_key_values
        lg = out.logits[torch.arange(Bn, device=self.dev), L - 1]
        fin = torch.isfinite(lg).all(-1)
        nxt = lg.argmax(-1)
        toks, lgs = [nxt], [lg] if return_logits else None
        for t in range(n - 1 + int(bool(feed_last))):
            if mask is not None:
                mask = torch.cat([mask, mask.new_ones(Bn, 1)], 1)
                kw = {"attention_mask": mask, "position_ids": (L + t)[:, None]}
            out = self.model(input_ids=nxt[:, None], past_key_values=past, use_cache=True, **kw)
            past = out.past_key_values
            if t < n - 1:
                lg = out.logits[:, -1]
                fin &= torch.isfinite(lg).all(-1)
                nxt = lg.argmax(-1)
                toks.append(nxt)
                if return_logits:
                    lgs.append(lg)
        self.n_forwards += 1
        self.n_sequences += Bn
        res = (torch.stack(toks, 1).cpu(), fin.cpu())
        return res + (torch.stack(lgs, 1),) if return_logits else res

    @torch.no_grad()
    def keys(self, seqs, output: bool = False):
        """ONE batch: the inputs of the edited module at every position of every row (one forward truncated after the module).
        -> list of (L_i, 6400) float32 tensors on the model's device [, list of the module OUTPUTS (L_i, 1600) as served]."""
        ids, mask, L = self._pad(seqs)
        cap = {}

        def hk(m, a, o):
            cap["x"], cap["o"] = a[0].detach(), o.detach()
            raise _Stop()
        h = self.mod.register_forward_hook(hk)                                  # registered after a served method's hook: sees its output
        try:
            self.model(input_ids=ids, use_cache=False, **({} if mask is None else {"attention_mask": mask}))
        except _Stop:
            pass
        finally:
            h.remove()
        self.n_forwards += 1
        xs = [cap["x"][i, :int(L[i])] for i in range(len(seqs))]
        return (xs, [cap["o"][i, :int(L[i])] for i in range(len(seqs))]) if output else xs
