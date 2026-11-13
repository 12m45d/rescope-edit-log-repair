"""Loader and registry of the E10 public text logs (ROME on GPT-2 XL, CounterFact), in the format of the project's
library `02_research/code/rescope` (which is frozen and is not modified).

    import sys
    sys.path.insert(0, "$RESCOPE_ROOT/02_research/experiments/09_testbed_scale/text_public/code")
    import tx_public as TX
    log = TX.from_public("TX-G2XL-rome-dev100")          # rescope.EditLog: U (T, 1600), W (T, 6400), K (T, 6400), batch
    W0, WE = TX.weights("TX-G2XL-rome-dev100")           # Conv1D weights (6400, 1600) float32: pre-edit, deployed
    model, tok = TX.load_model()                         # GPT-2 XL float32 at the PRE-EDIT weight (see "Stack" below)
    mod = TX.module(model)                               # transformer.h.17.mlp.c_proj  (transformers Conv1D)
    S = rescope.ScopedLog.from_log(log.to("cuda"), role="all", W0=W0.cuda())     # the library's hook works unchanged

Names (TX.NAMES): TX-G2XL-rome-dev100 (records 15000 .. 15099: the development log, the ONLY log of the family on
which a gating / repair method may be run or looked at before the registered experiment), TX-G2XL-rome-T100, -T500,
-T1000, -T2000 (records 0 .. T-1 of one stream; the shorter logs are prefixes: same factor file, sliced).

WHAT THE LOG HOLDS (public code -> library).  The public code returns per request (left_vector l, d_in = 6400;
right_vector r, d_out = 1600) and adds l[:, None] @ r[None, :] to the Conv1D weight (d_in, d_out; y = x @ W + b).
The change of the module output at module input k is therefore  <l, k> r.  In the library's notation
(W_math = W_conv^T, update = outer(U_i, W_i)):
        U_i = r * s_i          the stored residual: target_i - cur_output_i (the public compute_v divides it by s_i)
        W_i = l / s_i          the read-out vector
        s_i = <l, k_request_i> the public "Division Factor"
so that outer(U_i, W_i) = outer(r, l) (the applied update, exactly) and <W_i, k_request_i> = 1.
`batch` = arange(T): one request per update, every update sees the weight left by all earlier ones.

TWO KEYS (read this before choosing detectors).  The public code uses two different module inputs of the same edit:
  k_request        input of the module at the last subject token of the request prompt ALONE (no context template).
                   compute_v divides by <k_request, l> and subtracts the module output at this key: it is the key at
                   which the linear system is solved (the update attains the target there, up to float round-off).
  k_template_mean  mean of the inputs at the last subject token over the 21 context templates ("{}" and 20 sampled
                   prefixes).  compute_u multiplies THIS vector by the inverse second moment to get the direction l.
<W_i, k_template_mean_i> is NOT 1 (text_logs.json, "key_identity": within 10 % of 1 for about four edits in five,
2 to 3 when a short subject opens the request prompt, above 100 for the few requests whose subject is the very
first token: there the key without a prefix is almost orthogonal to the direction built from the prefixed keys, the
division factor is about 0.02 and the update explodes).  `from_public(name)` puts k_request into EditLog.K (key="request",
the default: the library's s_i = <W_i, k_i> and its rebase lemma refer to the key of the solve);
`from_public(name, key="template_mean")` puts the other one; both are always available as log.meta["K_request"] /
log.meta["K_template_mean"].  A module input does not depend on the edited weight (it is upstream of it), so both
keys are properties of the pre-edit model.

WHAT OF THE LIBRARY APPLIES.  EditLog, rebase, ScopedLog, Hook, the calibration and repair operators and the baseline
gates work on these logs unchanged (out/library_check.json: the hook on the Conv1D module, development log).  Give the
gate the weight in the MODULE's orientation, W0 = weights(name)[0] of shape (6400, 1600): Hook copies it into
module.weight and compares.  The library's harness side (rescope.load, Evaluator, SparseEvaluator, KeyCache,
stored_metrics) is built for the vision-language testbed and does not load these logs; `from_testbed` reads the
directories under logs_edit/library_format/<name> (edit_factors.pt, metrics.json) and returns the same log as
from_public(name).  Prompts are fed without BOS, as tok(prompt) gives them (the public convention).

Stack.  The logs, the pre-edit continuations and the description were produced with the public code under
transformers 4.23.1 (ext/pydeps_tf4; the project environment's transformers 5.17 breaks the public generation
function).  `load_model()` sets that stack up (tp_common.setup_public) and must be called before anything imports
transformers; importing this module alone needs torch only.
"""
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP
import json
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
TP = f"{_RP.ROOT}/02_research/experiments/09_testbed_scale/text_public"      # (as tp_common.TP: the logs live in the unpacked archive, not next to this file)
LOGS_EDIT = os.path.join(TP, "logs_edit")
OUT = os.path.join(TP, "out")
REGISTRY_FILE = os.path.join(LOGS_EDIT, "registry.json")
LIB = f"{_RP.ROOT}/02_research/code"
MODULE = "transformer.h.17.mlp.c_proj"
BACKBONE = "gpt2-xl"

# name -> (source run, T).  Static: the files may not exist yet (see available()).
PLAN = {
    "TX-G2XL-rome-dev100": ("dev", 100),
    "TX-G2XL-rome-T100": ("stream", 100),
    "TX-G2XL-rome-T500": ("stream", 500),
    "TX-G2XL-rome-T1000": ("stream", 1000),
    "TX-G2XL-rome-T2000": ("stream", 2000),
}
NAMES = list(PLAN)
DEV_LOG = "TX-G2XL-rome-dev100"                      # the only log a method may be developed / looked at on
REPORTED = [n for n in NAMES if n != DEV_LOG]
FIRST_RECORD = {"dev": 15000, "stream": 0}


def registry():
    """The registry written by build_logs.py: per name the source, T, records, hashes, reconstruction errors."""
    with open(REGISTRY_FILE) as f:
        return _RP.expand(json.load(f))


def available():
    """Names whose factors, deployed weight and audit record exist."""
    if not os.path.exists(REGISTRY_FILE):
        return []
    reg = registry()
    return [n for n in NAMES if n in reg and all(os.path.exists(reg[n][k]) for k in ("factors", "deployed_weight", "pre_edit_weight"))]


def _entry(name):
    if name not in PLAN:
        raise KeyError(f"unknown E10 log {name!r}; names: {NAMES}")
    reg = registry()
    if name not in reg:
        raise FileNotFoundError(f"{name} is not built yet (build_logs.py)")
    return reg[name]


_FACTORS = {}


def factors(name):
    """Raw per-update tensors of a log, float32, CPU, sliced to its T (public naming):
    left (T, 6400), right (T, 1600), k_request (T, 6400), k_template_mean (T, 6400), cur_output, target, target_init,
    delta (T, 1600); plus case_ids, record_indices, requests, scalars (list of dicts)."""
    e = _entry(name)
    p = e["factors"]
    if p not in _FACTORS:
        _FACTORS[p] = torch.load(p, map_location="cpu")
    f, T = _FACTORS[p], e["T"]
    out = {k: (v[:T] if torch.is_tensor(v) else v) for k, v in f.items() if torch.is_tensor(v)}
    for k in ("case_ids", "record_indices", "requests", "scalars"):
        out[k] = f[k][:T]
    out["context_templates"] = f["context_templates"]
    return out


def from_public(name, key="request"):
    """rescope.EditLog of an E10 log (CPU, fp64), in residual form (see the module docstring).
    key: 'request' (default; the key of the solve: s_i = 1) | 'template_mean' (the key behind the direction)."""
    if key not in ("request", "template_mean"):
        raise ValueError("key: 'request' | 'template_mean'")
    if LIB not in sys.path:
        sys.path.insert(0, LIB)
    from rescope.loglib import EditLog
    e, f = _entry(name), factors(name)
    left, right = f["left"].double(), f["right"].double()
    kreq, kavg = f["k_request"].double(), f["k_template_mean"].double()
    s = (left * kreq).sum(1)                              # the public division factor <cur_input, left_vector>
    U = right * s[:, None]                                # residual target - cur_output (as applied)
    W = left / s[:, None]                                 # read-out vector, <W_i, k_request_i> = 1
    K = kreq if key == "request" else kavg
    T = e["T"]
    meta = {"path": os.path.dirname(e["factors"]), "uids": [f"cf{c}" for c in f["case_ids"]], "case_ids": list(f["case_ids"]),
            "record_indices": list(f["record_indices"]), "requests": list(f["requests"]),
            "l2": None, "dtype": "float32", "layer": 17, "module": MODULE, "backbone": BACKBONE, "editor": "rome",
            "family": "TX-G2XL-rome", "source": e["source"], "key": key, "division_factor": s.clone(),
            "K_request": kreq, "K_template_mean": kavg, "target": f["target"].double(), "cur_output": f["cur_output"].double(),
            "context_templates": list(f["context_templates"]), "weight_orientation": "Conv1D (d_in, d_out): y = x @ W + b",
            "development_log": name == DEV_LOG, "registry": {k: v for k, v in e.items() if k != "per_update"}}
    return EditLog(U, W, K, torch.arange(T), name=name, meta=meta)


def weights(name, device="cpu"):
    """(W0, WE): pre-edit and deployed Conv1D weight (6400, 1600) float32 of the edited module, hashes verified."""
    import hashlib
    e = _entry(name)
    out = []
    for k, hk in (("pre_edit_weight", "pre_edit_sha256"), ("deployed_weight", "deployed_sha256")):
        w = torch.load(e[k], map_location="cpu")["weight"]
        h = hashlib.sha256(w.contiguous().numpy().tobytes()).hexdigest()
        if h != e[hk]:
            raise RuntimeError(f"{name}: {k} does not match the registry hash")
        out.append(w.to(device))
    return tuple(out)


def delta_conv(name):
    """Sum of the logged outer products in the module's orientation (6400, 1600), fp64 (W_E - W_0 up to round-off)."""
    f = factors(name)
    return f["left"].double().T @ f["right"].double()


def load_model(device="cuda"):
    """(model, tok): GPT-2 XL float32 at the pre-edit weight, loaded as the public code loads it, under the stack
    the logs were built with.  Must be called before transformers is imported by anything else."""
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    import tp_common as C
    C.setup_public()
    return C.load_model(device)


def module(model):
    """The edited module (transformers Conv1D: weight (6400, 1600), bias (1600,))."""
    m = model
    for part in MODULE.split("."):
        m = getattr(m, part)
    return m


def set_weight(model, W):
    """Write a Conv1D weight (6400, 1600) into the edited module, in place (float32)."""
    with torch.no_grad():
        module(model).weight.copy_(W.to(module(model).weight.device, torch.float32))


# ----------------------------------------------------------------------------- prompts, pools, pre-edit continuations
def prompts(set_=None, kind=None, T=None):
    """Rows of out/prompts.jsonl (see pools.py): set_ in 'repair' | 'heldout' | 'stream' | 'dev'; kind in 'request' |
    'paraphrase' | 'neighborhood'; T: for set 'stream', only the rows of the first T requests (pos < T)."""
    out = []
    with open(os.path.join(OUT, "prompts.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if set_ is not None and r["set"] != set_:
                continue
            if kind is not None and r["kind"] != kind:
                continue
            if T is not None and r["set"] == "stream" and r["pos"] >= T:
                continue
            out.append(r)
    return out


def log_prompts(name, kind=None):
    """Prompt rows of the logged records of a log: `pos` = index of the edit in the log."""
    src, T = PLAN[name]
    return prompts(src, kind, T if src == "stream" else None)


def benign(pool, name=None):
    """Benign prompts of a pool ('repair' | 'heldout').  With a log name every row gets the flags of THAT log:
    near_subject (the record's subject string is the subject of a logged request), near_mention (the prompt mentions
    the subject of a logged request), near_text (the prompt string is a request / neighbourhood prompt of a logged
    record), clean = none of them."""
    rows = prompts(pool)
    if name is None:
        return rows
    src, T = PLAN[name]
    for r in rows:
        if src == "stream":
            ns = r["subject_first_stream_pos"] is not None and r["subject_first_stream_pos"] < T
            nm = r["mentions_stream_subject_pos"] is not None and r["mentions_stream_subject_pos"] < T
            nt = any(r[k] is not None and r[k] < T for k in ("text_first_stream_neighborhood_pos", "text_first_stream_request_pos"))
        else:
            ns = bool(r["subject_in_dev"])
            nm = r["mentions_dev_subject_pos"] is not None
            nt = bool(r["text_is_dev_request"] or r["text_is_dev_neighborhood"] or r["text_is_dev_paraphrase"])
        r.update(near_subject=bool(ns), near_mention=bool(nm), near_text=bool(nt), clean=not (ns or nm or nt))
    return rows


def prompt_texts():
    """{'texts': [...], 'input_ids': [[...]]}: distinct prompt texts, indexed by text_id; ids = tok(text) (no BOS)."""
    with open(os.path.join(OUT, "prompt_texts.json")) as f:
        return json.load(f)


def pre_continuations():
    """Pre-edit greedy continuations, indexed by text_id: y0 (n, 10) token ids, pre_tf (argmax under teacher forcing
    of y0), valid (y0 reproduced under teacher forcing and finite), finite."""
    return torch.load(os.path.join(OUT, "pre_continuations.pt"), map_location="cpu")


def target_ids(tok, request):
    """Token ids of the new object as the public code tokenises it: tok(' ' + target_new)."""
    return tok(" " + request["target_new"]["str"].strip())["input_ids"]
