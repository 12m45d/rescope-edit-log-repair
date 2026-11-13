"""Common set-up of the INDEPENDENT AUDIT of the E10 public text logs (ROME on GPT-2 XL, CounterFact).

Nothing here imports the builder's code (text_public/code).  The audit reads the builder's stored FILES and drives the
public repository (ext/third_party/rome) and the checkpoint with code of its own.
All writes go under text_public/audit/ ; caches of every library are redirected there.
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
    os.environ[_v] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

ROOT = f"{_RP.ROOT}/02_research"
EXT = f"{ROOT}/ext"
TP = f"{ROOT}/experiments/09_testbed_scale/text_public"
AUD = f"{TP}/audit"
AOUT = f"{AUD}/out"
ALOGS = f"{AUD}/logs"
ARUN = f"{AUD}/run"                      # working directory of the public code (its util/globals.py reads ./globals.yml)
ATMP = f"{AUD}/tmp"
B_OUT = f"{TP}/out"                      # the builder's outputs (read only)
B_LOGS_EDIT = f"{TP}/logs_edit"          # the builder's logs (read only)
B_CODE = f"{TP}/code"
ROME_REPO = f"{EXT}/third_party/rome"
PYDEPS = f"{EXT}/pydeps"
PYDEPS_TF4 = f"{EXT}/pydeps_tf4"
MODEL_DIR = f"{EXT}/models/gpt2-xl"
DATA_DIR = f"{EXT}/data/rome_data"
COUNTERFACT = f"{EXT}/data/counterfact/counterfact.json"
STATS_FILE = f"{EXT}/data/stats/gpt2-xl/wikipedia_stats/transformer.h.17.mlp.c_proj_float32_mom2_100000.npz"
HPARAMS_FILE = f"{ROME_REPO}/hparams/ROME/gpt2-xl.json"
MODULE = "transformer.h.17.mlp.c_proj"
WEIGHT_NAME = MODULE + ".weight"
AUDIT_SEED = 20261111
LOGS = {"TX-G2XL-rome-dev100": ("dev", 100, 15000), "TX-G2XL-rome-T100": ("stream", 100, 0),
        "TX-G2XL-rome-T500": ("stream", 500, 0), "TX-G2XL-rome-T1000": ("stream", 1000, 0),
        "TX-G2XL-rome-T2000": ("stream", 2000, 0)}

# every cache a library may want goes under the audit directory (HOME is outside project_3)
_CACHE = f"{ATMP}/cache"
os.environ["HF_HOME"] = f"{_CACHE}/hf"
os.environ["TRANSFORMERS_CACHE"] = f"{_CACHE}/hf/hub"
os.environ["HF_DATASETS_CACHE"] = f"{_CACHE}/hf/datasets"
os.environ["XDG_CACHE_HOME"] = f"{_CACHE}/xdg"
os.environ["MPLCONFIGDIR"] = f"{_CACHE}/mpl"
os.environ["TORCH_HOME"] = f"{_CACHE}/torch"
os.environ["NLTK_DATA"] = f"{EXT}/data/nltk_data"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
for _d in (AOUT, ALOGS, ARUN, ATMP, _CACHE):
    os.makedirs(_d, exist_ok=True)


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())


def say(*a):
    print(f"[{time.strftime('%H:%M:%S', time.gmtime())}]", *a, file=sys.__stdout__, flush=True)


def jdump(obj, path, indent=1):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)
    os.replace(tmp, path)


def jload(path):
    with open(path) as f:
        return json.load(f)


def file_sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def tensor_sha256(t):
    import numpy as np  # noqa: F401
    a = t.detach().cpu().contiguous().numpy()
    return hashlib.sha256(a.tobytes()).hexdigest()


def load_counterfact():
    with open(COUNTERFACT) as f:
        return json.load(f)


def write_globals_yml():
    """globals.yml for the public util/globals.py: results under the audit directory; data / statistics / hparams are
    the files the builder used (read only)."""
    p = os.path.join(ARUN, "globals.yml")
    txt = ("---\n"
           f'  RESULTS_DIR: "{AOUT}/public_results"\n'
           f'  DATA_DIR: "{DATA_DIR}"\n'
           f'  STATS_DIR: "{DATA_DIR}/stats"\n'
           f'  HPARAMS_DIR: "{ROME_REPO}/hparams"\n'
           '  REMOTE_ROOT_URL: "https://rome.baulab.info"\n')
    if not os.path.exists(p) or open(p).read() != txt:
        with open(p, "w") as f:
            f.write(txt)
    return p


def setup_stack(stack="tf4", public=True):
    """stack 'tf4': transformers 4.23.1 from ext/pydeps_tf4 (the stack the logs were built with);
    stack 'env': the transformers of the project environment (5.x), for cross-stack checks.
    public=True: the public repository is importable as if the process had been started in its root."""
    import torch
    torch.set_num_threads(4)
    assert "transformers" not in sys.modules
    if stack == "tf4":
        sys.path.insert(0, PYDEPS_TF4)
    if public:
        write_globals_yml()
        os.chdir(ARUN)
        sys.path.insert(0, ROME_REPO)
    if PYDEPS not in sys.path:
        sys.path.append(PYDEPS)                      # nltk only (not in the environment)
    import transformers
    if stack == "tf4":
        assert transformers.__version__ == "4.23.1", transformers.__version__
    else:
        assert transformers.__version__.split(".")[0] == "5", transformers.__version__
    if public and "datasets" not in sys.modules:
        # rome/layer_stats.py imports datasets.load_dataset at import time; it is called only if the statistics file is
        # missing.  A placeholder that raises makes sure the audit never recomputes statistics silently.
        import importlib.machinery
        import types
        m = types.ModuleType("datasets")
        m.__spec__ = importlib.machinery.ModuleSpec("datasets", None)

        def load_dataset(*a, **k):
            raise RuntimeError("audit: datasets.load_dataset must not be called (statistics come from the stored file)")

        m.load_dataset = load_dataset
        sys.modules["datasets"] = m
    return transformers.__version__


def cap_gpu():
    import torch
    if "GQ_MEM" in os.environ and torch.cuda.is_available():
        cap = (int(os.environ["GQ_MEM"]) - 600) * 2 ** 20
        torch.cuda.set_per_process_memory_fraction(cap / torch.cuda.get_device_properties(0).total_memory, 0)
        return cap
    return None


def load_model(dtype_arg=False):
    """GPT-2 XL as the public experiments/evaluate.py loads it: AutoModelForCausalLM.from_pretrained(name).cuda(),
    tok.pad_token = tok.eos_token -- from the local checkpoint directory (no torch_dtype argument unless asked)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    kw = {"torch_dtype": torch.float32} if dtype_arg else {}
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, **kw).cuda()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    tok.pad_token = tok.eos_token
    model.config._name_or_path = "gpt2-xl"          # the public code builds the statistics path from this name
    assert not model.training
    assert {str(p.dtype) for p in model.parameters()} == {"torch.float32"}
    return model, tok


def get_module(model):
    m = model
    for part in MODULE.split("."):
        m = getattr(m, part)
    return m


def load_factors(src):
    """The builder's stacked factor file of a source run ('dev' | 'stream'), CPU."""
    import torch
    return torch.load(os.path.join(B_LOGS_EDIT, src, "factors.pt"), map_location="cpu")


def load_w0():
    import torch
    return torch.load(os.path.join(B_LOGS_EDIT, "W0_transformer.h.17.mlp.c_proj.weight.pt"), map_location="cpu")["weight"]


def load_snapshot(src, T):
    import torch
    return torch.load(os.path.join(B_LOGS_EDIT, src, "snapshots", f"W_T{T}.pt"), map_location="cpu")["weight"]


def replay(W0, left, right, n, device="cpu"):
    """W0 plus the first n logged outer products, written as the public code writes them: float32, in place, in order
    (w[...] += left[:, None] @ right[None, :])."""
    import torch
    W = W0.clone().to(device)
    with torch.no_grad():
        for i in range(n):
            W += left[i].to(device).unsqueeze(1) @ right[i].to(device).unsqueeze(0)
    return W
