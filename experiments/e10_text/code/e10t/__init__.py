"""e10t: adapter of the project's repair library (02_research/code/rescope, frozen) and of the shared driver
(10_final/common/fx, frozen) to the PUBLIC TEXT LOGS of experiment E10 (ROME on GPT-2 XL, CounterFact; PROTOCOL.md
section 6, amendment "E10: public text logs").  Nothing here changes a method: detectors, thresholds, values, the hook's
decision and response are the library's; this package adds the backbone, the text prompts, the key cache for text, the
all-positions gate-level evaluation, the sparse behavioural evaluation and the public scores.

    import sys; sys.path.insert(0, "$RESCOPE_ROOT/02_research/experiments/10_final/e10_text/code")
    import e10t                                  # torch only; e10t.Backbone() loads GPT-2 XL (do that BEFORE importing transformers)

    e10t.data      TextLog (open(name)): edit log, weights, prompt rows, pools, pre-edit continuations; KeyCache for text; draws
    e10t.backbone  Backbone (GPT-2 XL float32, edited module transformer.h.17.mlp.c_proj, a Conv1D), Served (a method on the
                   model: weight or library Hook, per-row firing flags in padded batches), tf / decode batched passes
    e10t.gate      gate-level evaluation from the key cache: features, decisions per prompt and per position, decision band
    e10t.methods   the row set: deployed, preedit, rescope, rescope_shared, rescope_null, cosine, cosine_published, exact
    e10t.behave    Session: metrics (i) - (vi) of the amendment; sparse (a prompt on which nothing fires costs no forward)
    e10t.public    the public evaluation code's own functions on the served model (scores (ii), n-gram entropy (v))

CONVENTIONS (binding for every number produced through this package)
  role set       ALL positions of the prompt and of the 10-token continuation: a query presents L + N_DEC keys (L prompt tokens
                 and the N_DEC = 10 continuation tokens).  The teacher-forced pass feeds [prompt, y0] (all 10 tokens; the 10
                 predictions are read at positions L-1 .. L+8) and a greedy decode feeds its last token once more (logits
                 discarded), so that the hook sees exactly the positions of the calibrated event.
  keys           module inputs of the edited module under the PRE-EDIT model.  They do not depend on the log or on the method
                 (the module input is upstream of the edited weight), so one cache serves every log of the backbone.
  pre-edit y0    out/pre_continuations.pt of the build (greedy, 10 tokens, valid = reproduced under teacher forcing).
  decision band  BAND = 1e-5 (relative to |threshold| + |w||k|/s): inside it the cached key (float32, another batch layout)
                 and the hook's live key may decide differently; such prompts are flagged `near` and always run through the
                 hook.  The round-off of a coordinate between the cached key and the hook's live key is measured by the tests
                 c_cache / g_cache (tests/out/table.txt), for the solver coordinate and for the cosine.
"""
from __future__ import annotations
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP

import hashlib
import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):   # at most 4 BLAS threads per process
    os.environ.setdefault(_v, "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = f"{_RP.ROOT}"
RESEARCH = ROOT + "/02_research"
LIB = RESEARCH + "/code"
FINAL = RESEARCH + "/experiments/10_final"
COMMON = FINAL + "/common"
E10T = FINAL + "/e10_text"
OUT = E10T + "/out"
CACHE_ROOT = os.environ.get("E10T_CACHE_ROOT", FINAL + "/cache/text")
STORE_ROOT = os.environ.get("E10T_STORE_ROOT", E10T + "/store")
TXC = RESEARCH + "/experiments/09_testbed_scale/text_public/code"
TXOUT = RESEARCH + "/experiments/09_testbed_scale/text_public/out"
GQ = RESEARCH + "/tools/gpuq/gq"
PYTHON = f"{_RP.PYTHON}"
for _p in (LIB, COMMON, TXC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

torch.set_num_threads(min(4, max(1, torch.get_num_threads())))

VERSION = "e10t-1.0 (2026-11-11)"
BACKBONE = "gpt2-xl"
N_DEC = 10                    # length of the continuation (E10: "the pre-edit greedy continuation of 10 tokens")
FEED_LAST = True              # the last continuation token is fed too: role set = all positions of the prompt and of the decode
BAND = 1e-5                   # relative half-width of the decision band of cached keys (see the module docstring)
ALPHA, N_CAL, N_NULL = 0.05, 200, 100
N_DRAWS = 20
COS_PUBLISHED = 0.65          # NeuralDB: gamma = 0.65
DEV_LOG = "TX-G2XL-rome-dev100"


def code_fingerprint(fresh: bool = False) -> dict:
    """sha1 (12 hex digits) of this package's .py files, of the library and of the driver: which code produced a result.
    Computed ONCE at import (a result records the code that was loaded, not the files on disk when it finished); fresh=True
    reads the files again."""
    def fp(folder):
        h = hashlib.sha1()
        for f in sorted(os.listdir(folder)):
            if f.endswith(".py"):
                h.update(f.encode())
                with open(os.path.join(folder, f), "rb") as fh:
                    h.update(fh.read())
        return h.hexdigest()[:12]
    if fresh or "fp" not in _FP:
        v = {"e10t": fp(os.path.dirname(os.path.abspath(__file__))), "rescope": fp(os.path.join(LIB, "rescope")),
             "fx": fp(os.path.join(COMMON, "fx")), "tx_public": fp(TXC), "e10t_version": VERSION}
        if fresh:
            return v
        _FP["fp"] = v
    return dict(_FP["fp"])


_FP = {}
code_fingerprint()

from . import data, gate, methods, behave, public  # noqa: E402,F401
from .backbone import Backbone, Served  # noqa: E402,F401
from .data import KeyCache, TextLog, open_log  # noqa: E402,F401
