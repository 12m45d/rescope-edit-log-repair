"""Build parts of the key cache for text (GPU; resumable; the model stays at the PRE-EDIT weight).

    $GQ run --mem 12000 --prio -8 --name e10t_cache -- $PY scripts/build_keycache.py --parts pools dev
    $GQ run --mem 12000 --prio -8 --name e10t_cache -- $PY scripts/build_keycache.py --log TX-G2XL-rome-T500 --reported-ok

Parts: pools (benign prompts of the REPAIR and HELD-OUT pools; independent of the log), dev (prompts of the development
log), stream_<a>_<b> (prompts of the requests a .. b-1 of the reported stream).  --log NAME builds what that log needs.
The keys are properties of the pre-edit model and of the prompt (no gate, no deployed weight is involved); the parts of
the reported stream are nevertheless only built when --reported-ok is given, i.e. when the registered experiment starts.
After building, a sample of every part is recomputed one prompt at a time and compared (verify.json in the part).
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code"))
import e10t  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--parts", nargs="*", default=[])
ap.add_argument("--log", default=None)
ap.add_argument("--reported-ok", action="store_true")
ap.add_argument("--verify", type=int, default=100)
args = ap.parse_args()
parts = list(args.parts) + (e10t.data.parts_of(args.log) if args.log else [])
parts = list(dict.fromkeys(parts))
if any(p.startswith("stream_") for p in parts) and not args.reported_ok:
    raise SystemExit("parts of the reported stream are built only with --reported-ok (when the registered experiment runs)")

B = e10t.Backbone()
import numpy as np  # noqa: E402
import torch  # noqa: E402

for p in parts:
    t0 = time.time()
    meta = e10t.data.build_part(B, p)
    print(f"part {p}: {meta['n_texts']} texts, {meta['n_positions']} positions, built in {meta['seconds']:.0f}s", flush=True)
    vp = os.path.join(e10t.CACHE_ROOT, e10t.BACKBONE, p, "verify.json")
    if args.verify and not os.path.exists(vp):
        kc = e10t.KeyCache([p])
        ix = e10t.data.jload(os.path.join(kc.dir, p, "index.json"))
        tids = ix["text_ids"]
        pick = [tids[int(i)] for i in np.sort(np.random.default_rng(0).choice(len(tids), min(args.verify, len(tids)), replace=False))]
        PT, PRE = e10t.data.prompt_texts(), e10t.data.pre_continuations()
        rel, same = [], 0
        for t in pick:
            x = B.keys([PT["input_ids"][t] + PRE["y0"][t].tolist()])[0].double().cpu()
            c = torch.from_numpy(np.asarray(kc.keys(t))).double()
            rel.append(float(((x - c).norm(dim=1) / x.norm(dim=1)).max()))
            same += int(torch.equal(x, c))
        e10t.data.jdump({"part": p, "n_sample": len(pick), "what": "cached keys (padded batches) against keys recomputed one prompt at a time",
                         "max_rel_deviation": max(rel), "median_rel_deviation": float(np.median(rel)), "n_bitwise_equal": same,
                         "band": e10t.BAND, "seconds": time.time() - t0}, vp)
        print(f"part {p}: verify max rel deviation {max(rel):.2e} on {len(pick)} texts ({same} bitwise equal)", flush=True)
print("peak MiB", torch.cuda.max_memory_allocated() / 2 ** 20, flush=True)
