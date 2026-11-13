"""Audit, condition (2): the single-edit protocol on a random sample of the first 1000 records, driven by the PUBLIC
driver itself -- experiments/evaluate.py main(), unmodified -- not by the builder's re-typed loop.

    python gpu_single.py --tag seed0 --seed 0 [--n 150]

How the public main() is made to run here (nothing of the repository is edited):
  * the baselines whose dependencies are not installed (hydra, higher, ... ) are replaced by placeholder modules, only
    if their import fails; ROME is the public rome package;
  * model and tokenizer are passed as the tuple main() accepts, loaded as main() would load them
    (AutoModelForCausalLM.from_pretrained(dir).cuda(); tok.pad_token = tok.eos_token) from the local checkpoint;
  * DS_DICT['cf'] gets a subclass of the public CounterFactDataset that loads the file with the public loader
    (size = 1000) and keeps the sampled records, in data order;
  * torch.manual_seed(seed) right before main() (the public driver sets no seed); seed 0 is the builder's seed, so the
    context templates must be the builder's; another seed gives other templates (sensitivity to the templates);
  * --skip_generation_tests as in the builder's run (no attribute snippets / tf-idf).
The audit additionally records the factors returned by the public execute_rome for each record (a wrapper of its own).
Sample: numpy default_rng(20261111).choice(1000, n, replace=False), sorted.
Output: out/public_results/ROME_audit_<tag>/run_000/case_<id>.json (public format), out/single_<tag>/factors/case_<id>.pt,
        out/single_<tag>.json
"""
import argparse
import contextlib
import importlib
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a_common as A  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--n", type=int, default=150)
ap.add_argument("--extra", default="", help="comma-separated record indices added to the sample (kept separate in the comparison)")
args = ap.parse_args()

tfv = A.setup_stack("tf4")
import numpy as np  # noqa: E402
import torch  # noqa: E402

cap = A.cap_gpu()
SAMPLE = sorted(int(x) for x in np.random.default_rng(A.AUDIT_SEED).choice(1000, args.n, replace=False))
EXTRA = sorted(int(x) for x in args.extra.split(",") if x)
RECORDS = sorted(set(SAMPLE) | set(EXTRA))

# ---- placeholders for the baselines that cannot be imported (never called: alg_name = ROME)
stubbed = {}


class _NotInstalled:
    def __init__(self, *a, **k):
        pass

    def apply_to_model(self, *a, **k):
        raise RuntimeError("audit placeholder: baseline not installed")

    @classmethod
    def from_json(cls, *a, **k):
        raise RuntimeError("audit placeholder: baseline not installed")


def _not_installed(*a, **k):
    raise RuntimeError("audit placeholder: baseline not installed")


for name, attrs in (("baselines.efk", {"EFKHyperParams": _NotInstalled, "EfkRewriteExecutor": _NotInstalled}),
                    ("baselines.ft", {"FTHyperParams": _NotInstalled, "apply_ft_to_model": _not_installed}),
                    ("baselines.kn", {"KNHyperParams": _NotInstalled, "apply_kn_to_model": _not_installed}),
                    ("baselines.mend", {"MENDHyperParams": _NotInstalled, "MendRewriteExecutor": _NotInstalled})):
    try:
        importlib.import_module(name)
    except Exception as ex:  # noqa: BLE001
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        sys.modules[name] = m
        stubbed[name] = f"{type(ex).__name__}: {str(ex)[:120]}"

import experiments.evaluate as EV  # noqa: E402   the public driver
import rome.rome_main as rm  # noqa: E402
from dsets import CounterFactDataset  # noqa: E402

assert EV.ALG_DICT["ROME"][1] is rm.apply_rome_to_model
out_dir = os.path.join(A.AOUT, f"single_{args.tag}")
fac_dir = os.path.join(out_dir, "factors")
os.makedirs(fac_dir, exist_ok=True)
pub_log = open(os.path.join(A.ALOGS, f"single_{args.tag}_public_stdout.log"), "a", buffering=1)


class AuditSample(CounterFactDataset):
    """The public loader (first 1000 records of counterfact.json), restricted to the sampled records."""

    def __init__(self, data_dir, size=None, *a, **k):
        super().__init__(data_dir, size=size, *a, **k)
        assert len(self.data) == 1000 and all(r["case_id"] == i for i, r in enumerate(self.data))
        self.data = [self.data[i] for i in RECORDS]


EV.DS_DICT["cf"] = (AuditSample, EV.DS_DICT["cf"][1])

# ---- the audit's own record of what the public execute_rome returns
_orig_exec = rm.execute_rome
_by_req = {}


def _key(req):
    return (req["prompt"], req["subject"], req["target_new"]["str"], req["target_true"]["str"])


_data = A.load_counterfact()
for i in RECORDS:
    _by_req.setdefault(_key(_data[i]["requested_rewrite"]), []).append(i)
assert all(len(v) == 1 for v in _by_req.values())


def execute_rome(model, tok, request, hparams):
    deltas = _orig_exec(model, tok, request, hparams)
    (name, (left, right)), = deltas.items()
    cid = _by_req[_key(request)][0]
    torch.save({"case_id": cid, "weight_name": name, "left": left.detach().float().cpu().clone(),
                "right": right.detach().float().cpu().clone()}, os.path.join(fac_dir, f"case_{cid}.pt"))
    return deltas


rm.execute_rome = execute_rome

model, tok = A.load_model()
W = A.get_module(model).weight
W0 = W.detach().clone()
dir_name = f"ROME_audit_{args.tag}"
run_dir = os.path.join(A.AOUT, "public_results", dir_name, "run_000")
cont = "run_000" if os.path.isdir(run_dir) else None
meta = {"written": A.now(), "tag": args.tag, "seed": args.seed, "sample_rule": f"numpy default_rng({A.AUDIT_SEED}).choice(1000, {args.n}, replace=False), sorted",
        "sample": SAMPLE, "extra_records": EXTRA, "transformers": tfv, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
        "baselines_replaced_by_placeholders": stubbed, "public_driver": EV.__file__, "continue_from_run": cont, "cap_bytes": cap,
        "pre_edit_weight_sha256": A.tensor_sha256(W0)}
A.jdump(meta, os.path.join(A.AOUT, f"single_{args.tag}.json"))
A.say("placeholders:", stubbed)

torch.manual_seed(args.seed)
with contextlib.redirect_stdout(pub_log):
    EV.main("ROME", (model, tok), "gpt2-xl.json", "cf", 1000, cont, True, False, dir_name=dir_name)   # PUBLIC main()

meta["context_templates"] = list(rm.CONTEXT_TEMPLATES_CACHE)
meta["weight_restored_bitwise"] = bool(torch.equal(W, W0))
meta["n_case_files"] = len([f for f in os.listdir(run_dir) if f.startswith("case_") and f.endswith(".json")])
meta["peak_alloc_MiB"] = torch.cuda.max_memory_allocated() / 2 ** 20
meta["peak_reserved_MiB"] = torch.cuda.max_memory_reserved() / 2 ** 20
meta["finished"] = A.now()
A.jdump(meta, os.path.join(A.AOUT, f"single_{args.tag}.json"))
A.say("done", meta["n_case_files"], "case files; weight restored:", meta["weight_restored_bitwise"])
