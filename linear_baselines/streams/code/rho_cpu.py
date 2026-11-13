"""P12 -- module-output restoration on the HELD-OUT answer keys, computed on the CPU (no model, no GPU).

  rho_r = ||(W_rep - W_0) K_r||_F / ||(W_E - W_0) K_r||_F     for every valid HELD-OUT row r,
  K_r = inputs of the edited module at the row's answer-predicting positions, W_rep = W_E + correction.
  rho = 1: the repair leaves the edited module's output at these positions where the log put it; 0: the pre-edit
  output is back; > 1: further away from it.  Only the edited module at the answer positions is looked at.

The same quantity as run_d2.module_restore (which has the model loaded); this script needs only
  cache/heldout_anskeys_<log>.pt   keys captured once by run_d2.heldout_keys (DIAGNOSTIC: read by no method),
  the checkpoint's weight of the edited module (W_0; [redacted: related project], read-only) and the testbed's composite_delta.pt
  (W_E = (W_0.float() + C).to(dtype): the harness's rollback_weight([]) identity, checked at every load),
  the corrections saved by run_streams.py in out/<log>/corr/ (seed 0, n in {10, 100, ALL}).
Check of W_0: a saved rollback of ALL edits (harness: defined as W_0 exactly) must equal the checkpoint tensor.
For adapter corrections (COMP) it also stores span_share_r = ||R^T K_r||_F / ||K_r||_F: the share of the row's
answer-key norm inside the correction's row space (0 = the correction cannot act on the row's answer positions).
Output: out/<log>/rho/<config>.json (skips configs that are done).

usage: CUDA_VISIBLE_DEVICES="" python rho_cpu.py
"""
from __future__ import annotations
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP

import glob
import json
import os

import torch
from safetensors import safe_open

P12 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT, CACHE = os.path.join(P12, "out"), os.path.join(P12, "cache")
TB = f"{_RP.ROOT}/02_research/experiments/06_testbed_v3"
MODELS = f"{_RP.EXT_VIEW}/data/models"
CKPT = {"TB-L100-ae-l2_3": ("llava-1.5-7b", "language_model.model.layers.16.mlp.down_proj.weight"),
        "TB-Q100-ae-l2_3": ("Qwen2.5-VL-3B", "model.layers.20.mlp.down_proj.weight")}
DEF = ("||(W_rep - W_0) K_r||_F / ||(W_E - W_0) K_r||_F at the answer keys of each valid HELD-OUT row "
       "(diagnostic; 1 = unrepaired, 0 = pre-edit)")
SHARE_DEF = ("adapter corrections only: ||R^T K_r||_F / ||K_r||_F, R = orthonormal basis of the correction's row space "
             "(the reported complaint keys with the protected span projected out): the share of the row's answer-key "
             "norm on which the correction acts at all")
VERSION = 2


def load_w0(log):
    d, key = os.path.join(MODELS, CKPT[log][0]), CKPT[log][1]
    fn = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"][key]
    with safe_open(os.path.join(d, fn), "pt") as sf:
        return sf.get_tensor(key)


def main():
    torch.set_num_threads(int(os.environ.get("P12_THREADS", "4")))      # at most 4 BLAS threads per process
    for log in CKPT:
        kp = os.path.join(CACHE, f"heldout_anskeys_{log}.pt")
        cdir = os.path.join(OUT, log, "corr")
        if not os.path.exists(kp) or not os.path.isdir(cdir):
            print(f"{log}: held-out answer keys not captured yet (needs one GPU start of run_d2.py) -- skipped")
            continue
        names = sorted(os.path.basename(f)[:-3] for f in glob.glob(os.path.join(cdir, "*.pt")))
        als = {os.path.basename(f)[:-6]: open(f).read().strip() for f in glob.glob(os.path.join(cdir, "*.alias"))}
        odir = os.path.join(OUT, log, "rho")
        os.makedirs(odir, exist_ok=True)
        def done(n):
            fp = os.path.join(odir, n + ".json")
            return os.path.exists(fp) and json.load(open(fp)).get("version") == VERSION
        todo = [n for n in names + sorted(als) if not done(n)]
        if not todo:
            print(f"{log}: rho of all {len(names) + len(als)} saved corrections done")
            continue
        hk = torch.load(kp)
        W0 = load_w0(log)
        C = torch.load(os.path.join(TB, log, "composite_delta.pt"))
        WE = (W0.float() + C).to(W0.dtype)
        D = WE.double() - W0.double()
        K = hk["K"].double().T.contiguous()                        # (d, N)
        DK = D @ K
        chk = {"W0_checkpoint_equals_saved_rollback_of_all_edits": None}
        T = json.load(open(os.path.join(OUT, log, "_setup.json")))["T"]
        for n in names:                                            # W_0 check on a rollback of every edit
            rec = os.path.join(OUT, log, n + ".json")
            if n.startswith("RB_") and os.path.exists(rec) and json.load(open(rec))["extra"].get("k") == T:
                chk["W0_checkpoint_equals_saved_rollback_of_all_edits"] = bool(torch.equal(torch.load(os.path.join(cdir, n + ".pt"))["W"], W0))
                chk["checked_on"] = n
                break
        print(f"{log}: W_0 {tuple(W0.shape)} {W0.dtype}; {K.shape[1]} keys of {len(hk['counts'])} rows; checks {chk}")
        if chk["W0_checkpoint_equals_saved_rollback_of_all_edits"] is False:
            raise SystemExit("the checkpoint tensor is not the harness's W_0")
        for n in todo:
            b = torch.load(os.path.join(cdir, als.get(n, n) + ".pt"), map_location="cpu")
            share, RtK = None, None
            if b["kind"] == "adapter":
                RtK = b["Rf"].T @ K                                # coordinates of the keys in the correction's row space
                XK = b["Lf"] @ RtK
            elif b["kind"] == "weight":
                XK = (b["W"].double() - WE.double()) @ K
            else:
                XK = torch.zeros_like(DK)
            R = DK + XK
            rho, j = [], 0
            if RtK is not None:
                share = []
            for A in hk["counts"]:
                rho.append(float(R[:, j:j + A].norm() / DK[:, j:j + A].norm().clamp_min(1e-30)))
                if RtK is not None:                                # Rf has orthonormal columns: ||Rf^T K_r|| / ||K_r||
                    share.append(float(RtK[:, j:j + A].norm() / K[:, j:j + A].norm().clamp_min(1e-30)))
                j += A
            tmp = os.path.join(odir, n + f".json.tmp{os.getpid()}")
            with open(tmp, "w") as f:
                json.dump({"log": log, "config": n, "alias_of": als.get(n), "source": "rho_cpu.py", "version": VERSION,
                           "module_restore": {"rho": rho, "span_share": share, "rank": None if RtK is None else int(RtK.shape[0]),
                                              "groups": hk["groups"], "stored_D1": hk["stored_D1"], "ptype": hk["ptype"], "def": DEF,
                                              "span_share_def": SHARE_DEF},
                           "checks": chk}, f)
            os.replace(tmp, os.path.join(odir, n + ".json"))
            del b, XK, R, RtK
        print(f"{log}: rho written for {len(todo)} corrections")


if __name__ == "__main__":
    main()
