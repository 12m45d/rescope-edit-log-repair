"""Common set-up of the E10 public text logs (ROME on GPT-2 XL, CounterFact).

The public reference implementation (third_party/rome, commit recorded in out/provenance.json) is NEVER edited.
It is driven from here:
  * the drivers chdir to text_public/run, whose globals.yml carries absolute paths (the public util/globals.py
    reads ./globals.yml);
  * the model is loaded the way experiments/evaluate.py does (AutoModelForCausalLM.from_pretrained(...).cuda(),
    tok.pad_token = tok.eos_token) from the local copy of the checkpoint;
  * `Capture` wraps three public functions at run time to RECORD what they compute (factors, keys, target);
    it changes no value that the public code computes.
Every deviation is listed in out/DEVIATIONS.txt (rendered by make_deviations.py).
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

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

ROOT = f"{_RP.ROOT}/02_research"
EXT = f"{ROOT}/ext"
# nothing of the Hugging Face stack may write outside project_3 (transformers 4.x writes a cache marker on import)
os.environ["HF_HOME"] = f"{EXT}/models/.hf_home"
os.environ["TRANSFORMERS_CACHE"] = f"{EXT}/models/.hf_home/hub"
os.environ["HF_DATASETS_CACHE"] = f"{EXT}/models/.hf_home/datasets"
TP = f"{ROOT}/experiments/09_testbed_scale/text_public"
CODE = f"{TP}/code"
OUT = f"{TP}/out"
LOGS = f"{TP}/logs"
LOGS_EDIT = f"{TP}/logs_edit"
RUN_DIR = f"{TP}/run"
ROME_REPO = f"{EXT}/third_party/rome"
PYDEPS = f"{EXT}/pydeps"                    # nltk (not in the project environment)
PYDEPS_TF4 = f"{EXT}/pydeps_tf4"            # transformers 4.23.1 + tokenizers 0.13.3 + huggingface_hub 0.11.1
MODEL_DIR = f"{EXT}/models/gpt2-xl"
MODEL_NAME = "gpt2-xl"                      # the name the public code uses (statistics path, hparams file)
DATA_DIR = f"{EXT}/data/rome_data"
COUNTERFACT = f"{DATA_DIR}/counterfact.json"
NLTK_DATA = f"{EXT}/data/nltk_data"
os.environ.setdefault("NLTK_DATA", NLTK_DATA)

HPARAMS_FILE = f"{ROME_REPO}/hparams/ROME/gpt2-xl.json"
MODULE = "transformer.h.17.mlp.c_proj"       # the edited module (hparams: layers [17], rewrite_module_tmp)
WEIGHT_NAME = MODULE + ".weight"             # Conv1D weight, shape (d_in = 6400, d_out = 1600): y = x @ W + b

DEV_START, DEV_T = 15000, 100                # development log: records 15000 .. 15099
STREAM_START, STREAM_T = 0, 2000             # reported stream: records 0 .. 1999
SNAPSHOTS = (100, 500, 1000, 2000)
SINGLE_N = 1000                              # single-edit check: records 0 .. 999
POOL_SEED = 20261109
SEED = 0                                     # torch seed set before the first request of every run

import numpy as np   # noqa: E402
import torch         # noqa: E402

torch.set_num_threads(4)


def cap_gpu():
    """Cap this process to the queue reservation (GQ_MEM, MiB) minus 600 MiB of CUDA context."""
    if "GQ_MEM" in os.environ and torch.cuda.is_available():
        cap = (int(os.environ["GQ_MEM"]) - 600) * 2 ** 20
        torch.cuda.set_per_process_memory_fraction(cap / torch.cuda.get_device_properties(0).total_memory, 0)
        return cap
    return None


def setup_public():
    """Make the public repository importable exactly as if the driver were started in its root.

    Library stack.  The project environment has transformers 5.17, under which the public util/generate.py
    (generate_fast: key-value cache + attention mask of the current token only) produces garbage (smoke test of
    2026-11-11, logs/smoke_tf5.out): the context templates and every generation metric would be wrong.  The public
    files were written for transformers 4.15.0 (scripts/rome.yml; no Python 3.11 build of its tokenizers exists) and
    are shipped unchanged with MEMIT, whose environment pins transformers 4.23.1 (scripts/memit.yml).  That stack is
    installed under ext/pydeps_tf4 and put FIRST on sys.path for the E10 build only; the environment is not modified.
    The environment's `datasets` package cannot be imported next to it; rome/layer_stats.py imports load_dataset at
    import time but calls it only when the statistics file is missing (it is not: the authors' file is used), so a
    placeholder module whose load_dataset raises is registered.
    """
    os.makedirs(RUN_DIR, exist_ok=True)
    os.chdir(RUN_DIR)                        # util/globals.py: open("globals.yml")
    assert "transformers" not in sys.modules, "setup_public() must run before transformers is imported"
    for p in (PYDEPS_TF4, ROME_REPO):
        if p in sys.path:
            sys.path.remove(p)
    sys.path.insert(0, PYDEPS_TF4)
    sys.path.insert(0, ROME_REPO)
    if PYDEPS not in sys.path:
        sys.path.append(PYDEPS)              # appended: the environment's own packages take precedence
    import transformers                      # before the placeholder: its import probes find_spec("datasets")
    assert transformers.__version__ == "4.23.1", transformers.__version__
    if "datasets" not in sys.modules:
        import importlib.machinery
        import types
        stub = types.ModuleType("datasets")
        stub.__spec__ = importlib.machinery.ModuleSpec("datasets", None)

        def load_dataset(*a, **k):
            raise RuntimeError("datasets.load_dataset is not available in the E10 build: the second-moment statistics "
                               "must come from the precomputed file (see DEVIATIONS.txt)")

        stub.load_dataset = load_dataset
        stub.__e10_placeholder__ = True
        sys.modules["datasets"] = stub


def load_model(device="cuda"):
    """Model and tokenizer as experiments/evaluate.py loads them (float32), from the local checkpoint."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float32)
    model = model.cuda() if device == "cuda" else model
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    tok.pad_token = tok.eos_token
    model.config._name_or_path = MODEL_NAME   # the public code derives the statistics path from this name
    dts = {str(p.dtype) for p in model.parameters()}
    assert dts == {"torch.float32"}, dts
    assert not model.training
    return model, tok


def load_hparams():
    from rome import ROMEHyperParams
    return ROMEHyperParams.from_json(HPARAMS_FILE)


def get_weight(model):
    from util import nethook
    return nethook.get_parameter(model, WEIGHT_NAME)


def load_counterfact():
    with open(COUNTERFACT) as f:
        return json.load(f)


# ----------------------------------------------------------------------------- hashing / io
def sha256_tensor(t):
    """sha256 of the raw little-endian bytes of a tensor (CPU, contiguous), plus dtype and shape."""
    a = t.detach().cpu().contiguous().numpy()
    return {"sha256": hashlib.sha256(a.tobytes()).hexdigest(), "dtype": str(a.dtype), "shape": list(a.shape)}


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def jdump(obj, path, indent=1):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=indent)
    os.replace(tmp, path)


def jload(path):
    with open(path) as f:
        return json.load(f)


def tsave(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())


def versions():
    import transformers
    import tokenizers
    out = {"python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers.__version__,
           "tokenizers": tokenizers.__version__, "numpy": np.__version__}
    try:
        import nltk
        out["nltk"] = nltk.__version__
    except Exception:  # noqa: BLE001
        out["nltk"] = None
    try:
        import scipy
        import sklearn
        out["scipy"], out["sklearn"] = scipy.__version__, sklearn.__version__
    except Exception:  # noqa: BLE001
        pass
    if torch.cuda.is_available():
        out["gpu"] = torch.cuda.get_device_name(0)
        out["cuda"] = torch.version.cuda
    return out


# ----------------------------------------------------------------------------- capture of the public computation
class Capture:
    """Records, per request, what the public ROME code computes.  Pure observation.

    Wrapped at run time (no file of the repository is edited):
      rome.rome_main.execute_rome                  -> the returned factors (left_vector, right_vector)
      rome.repr_tools.get_reprs_at_word_tokens     -> called by compute_u: module inputs at the last subject token
                                                      under every context template (their mean is the key that
                                                      compute_u multiplies by the inverse second moment);
                                                      called by compute_v.get_module_input_output_at_word:
                                                      (cur_input, cur_output) of the request prompt alone
      and, from the frame of compute_v at that second call, its locals target, target_init, delta and the last
      loss terms (they are locals of the public function; read, never written).
    """

    def __init__(self):
        self.cur = None
        self.installed = False

    def install(self):
        if self.installed:
            return
        import rome.repr_tools as rt
        import rome.rome_main as rm
        cap = self
        orig_reprs = rt.get_reprs_at_word_tokens
        orig_exec = rm.execute_rome

        def get_reprs_at_word_tokens(*a, **k):
            out = orig_reprs(*a, **k)
            if cap.cur is not None:
                caller = sys._getframe(1).f_code.co_name
                if caller == "compute_u":
                    cap.cur["k_templates"] = out.detach().clone()                      # (n_templates, d_in)
                elif caller == "get_module_input_output_at_word":
                    cap.cur["cur_input"] = out[0][0].detach().clone()                  # (d_in,)
                    cap.cur["cur_output"] = out[1][0].detach().clone()                 # (d_out,)
                    fr = sys._getframe(2)
                    if fr.f_code.co_name == "compute_v":
                        loc = fr.f_locals
                        for name in ("target", "target_init", "delta"):
                            cap.cur[name] = loc[name].detach().clone()
                        cap.cur["v_steps"] = int(loc["it"]) + 1
                        cap.cur["v_loss"] = float(loc["loss"].detach())
                        cap.cur["v_nll"] = float(loc["nll_loss"].detach())
                        cap.cur["v_kl"] = float(loc["kl_loss"].detach())
                        cap.cur["v_wd"] = float(loc["weight_decay"].detach())
                        cap.cur["v_prob_new"] = float(torch.exp(-loc["nll_loss_each"].detach()).mean())
                        cap.cur["lookup_idxs"] = [int(x) for x in loc["lookup_idxs"]]
                        cap.cur["n_target_tokens"] = int(loc["target_ids"].numel())
            return out

        def execute_rome(model, tok, request, hparams, *a, **k):
            deltas = orig_exec(model, tok, request, hparams, *a, **k)
            if cap.cur is not None:
                assert len(deltas) == 1, list(deltas)
                (name, (left, right)), = deltas.items()
                cap.cur["weight_name"] = name
                cap.cur["left"] = left.detach().clone()                                # (d_in,)  public: left_vector
                cap.cur["right"] = right.detach().clone()                              # (d_out,) public: right_vector
            return deltas

        rt.get_reprs_at_word_tokens = get_reprs_at_word_tokens
        rm.execute_rome = execute_rome
        self.installed = True

    def begin(self):
        self.cur = {}

    def end(self):
        cur, self.cur = self.cur, None
        need = ("left", "right", "k_templates", "cur_input", "cur_output", "target", "target_init", "delta")
        missing = [n for n in need if n not in cur]
        if missing:
            raise RuntimeError(f"capture incomplete: {missing}")
        return cur


def update_record(cur):
    """Per-update tensors (float32, CPU) and scalar checks (fp64) from a Capture record.

    Orientation.  GPT-2's Conv1D computes y = x @ W + b with W of shape (d_in, d_out).  The public code adds
    left_vector[:, None] @ right_vector[None, :] (d_in, d_out) to W, so the change of the module output at input k is
    <left, k> * right.  In the paper's notation (W_math = W^T, d_out x d_in; update = outer(u, w)):
        u_i (output side)   = right_vector * <cur_input, left_vector> = target - cur_output      [residual]
        w_i (read-out)      = left_vector / <cur_input, left_vector>
    so that <w_i, cur_input> = 1 and outer(u_i, w_i) = outer(right_vector, left_vector).
    """
    f64 = torch.float64
    left, right = cur["left"].to(f64), cur["right"].to(f64)
    kt = cur["k_templates"].to(f64)
    k_avg = kt.mean(0)                                   # compute_u: .mean(0) over the context templates
    k_req = cur["cur_input"].to(f64)
    div = float(left @ k_req)                            # compute_v: torch.dot(cur_input, left_vector)
    res = cur["target"].to(f64) - cur["cur_output"].to(f64)
    w = left / div                                       # read-out vector in the paper's notation
    coord_t = (kt @ left) / div                          # normalised coordinate of every context template
    sc = {
        "div_factor": div,
        "left_norm": float(left.norm()), "right_norm": float(right.norm()),
        "update_fro": float(left.norm() * right.norm()),
        "w_dot_k_request": float(w @ k_req),             # = 1 up to round-off (the key the solver divides by)
        "w_dot_k_avg": float(w @ k_avg),                 # coordinate of the template-averaged key
        "coord_templates_min": float(coord_t.min()), "coord_templates_max": float(coord_t.max()),
        "coord_templates": [float(x) for x in coord_t],
        "resid_norm": float(res.norm()),
        "resid_minus_right_times_div_rel": float((res - right * div).norm() / res.norm()),
        "target_norm": float(cur["target"].norm()), "target_init_norm": float(cur["target_init"].norm()),
        "delta_norm": float(cur["delta"].norm()),
        "cur_output_minus_target_init_rel": float((cur["cur_output"].to(f64) - cur["target_init"].to(f64)).norm()
                                                  / cur["target_init"].to(f64).norm()),
        "k_request_norm": float(k_req.norm()), "k_avg_norm": float(k_avg.norm()),
        "cos_k_request_k_avg": float((k_req @ k_avg) / (k_req.norm() * k_avg.norm())),
    }
    for name in ("v_steps", "v_loss", "v_nll", "v_kl", "v_wd", "v_prob_new", "lookup_idxs", "n_target_tokens"):
        if name in cur:
            sc[name] = cur[name]
    ten = {
        "left": cur["left"].float().cpu(), "right": cur["right"].float().cpu(),
        "k_avg": k_avg.float().cpu(), "k_request": cur["cur_input"].float().cpu(),
        "cur_output": cur["cur_output"].float().cpu(), "target": cur["target"].float().cpu(),
        "target_init": cur["target_init"].float().cpu(), "delta": cur["delta"].float().cpu(),
    }
    return ten, sc


def public_prob_test(model, tok, record):
    """The public probability test of one record: compute_rewrite_quality_counterfact(model, tok, record, None, None)
    (rewrite / paraphrase / neighbourhood / attribute prompts; snips = vec = None is the public flag
    --skip_generation_tests).  Returns (result, fallback_used).

    The public test_batch_prediction feeds all 46 sequences of a record as ONE padded batch through model(**batch); with
    the default config (use_cache = True) the forward also returns the key / value tensors of all 48 layers, about
    0.61 MB per token.  For a record with a very long paraphrase prompt (case 1531: 179 tokens, 46 x 181 tokens, 5.1 GB
    of key / value tensors) this exceeds the memory cap of the job.  On a CUDA out-of-memory error the SAME public
    function is called again with model.config.use_cache = False: the forward then does not return the key / value
    tensors; the logits are computed by the same operations (use_cache_equivalence() checks bit-for-bit equality of the
    public function's output under both settings).  The record is flagged in its case file."""
    import gc
    from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact
    try:
        return compute_rewrite_quality_counterfact(model, tok, record, None, None), False
    except torch.cuda.OutOfMemoryError:
        pass
    gc.collect()
    torch.cuda.empty_cache()
    old = model.config.use_cache
    model.config.use_cache = False
    try:
        out = compute_rewrite_quality_counterfact(model, tok, record, None, None)
    finally:
        model.config.use_cache = old
    return out, True


def use_cache_equivalence(model, tok, records):
    """Number of records on which the public probability test returns exactly the same numbers with
    model.config.use_cache = True (default) and False."""
    from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact
    old = model.config.use_cache
    n_same = 0
    try:
        for record in records:
            model.config.use_cache = True
            a = compute_rewrite_quality_counterfact(model, tok, record, None, None)
            model.config.use_cache = False
            b = compute_rewrite_quality_counterfact(model, tok, record, None, None)
            n_same += int(json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True))
    finally:
        model.config.use_cache = old
    return {"n_records": len(records), "n_identical_output": n_same}


class Tee:
    """Duplicate the public code's prints into a log file (they are the only record of its optimisation)."""

    def __init__(self, path, echo=False):
        self.f = open(path, "a", buffering=1)
        self.echo = echo
        self.stdout = sys.stdout

    def write(self, s):
        self.f.write(s)
        if self.echo:
            self.stdout.write(s)

    def flush(self):
        self.f.flush()
        if self.echo:
            self.stdout.flush()


def say(*a):
    print(f"[{time.strftime('%H:%M:%S', time.gmtime())}]", *a, file=sys.__stdout__, flush=True)
