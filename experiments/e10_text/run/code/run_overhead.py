"""E10 / R22 (f): serving overhead per forward pass, measured again in one process for both logs (the first measurement inside
run_r22.py had no warm-up and ran next to other jobs).  512 TEST prompts of the VIEW pool, every prompt through the model (first
token, batches of at most 32), one warm-up pass, then 7 interleaved repeats; reported: median and minimum per served object.

    $GQ run --mem 13000 --prio -2 --name e10o -- $PY run_overhead.py   ->  out/r22/overhead.json
"""
import time

import xcommon as X
import e10t

B = e10t.Backbone()
import numpy as np  # noqa: E402
import torch  # noqa: E402
import xmeth as XM  # noqa: E402
import xview as XV  # noqa: E402

VS = XV.ViewStore()
sample = XV.view_pool()["test"][:512]
ids = [VS.ids[XV.VBASE + r] for r in sample]
bl = []
for b in B.batches([len(s) for s in ids], extra=1):
    bl += [b[a:a + 32] for a in range(0, len(b), 32)]


@torch.no_grad()
def run(obj):
    with B.served(obj):
        torch.cuda.synchronize()
        t0 = time.time()
        for b in bl:
            B.decode([ids[i] for i in b], n=1, feed_last=False)
        torch.cuda.synchronize()
        return 1000.0 * (time.time() - t0) / len(ids)


out = {"what": __doc__, "logs": {}}
for lg in X.BEHAVE_LOGS:
    L = e10t.open_log(lg)
    XM.install_capture(B)
    objs = {"pre-edit weight": None, "deployed weight": L.weights()[1], "ReScope-W log-only (hook, all positions)": XM.method(L, "where_logonly").obj,
            "ReScope-W two-block (hook, all positions)": XM.method(L, "where_twoblock", 0).obj, "migration (hook, all positions)": XM.method(L, "rescope", 0).obj,
            "lookup_subject": XM.lookup_method(L).obj}
    tim = {k: [] for k in objs}
    for k, o in objs.items():
        run(o)
    for rep in range(7):
        for k, o in objs.items():
            tim[k].append(run(o))
    base = float(np.median(tim["pre-edit weight"]))
    out["logs"][lg] = {k: {"ms_median": float(np.median(v)), "ms_min": float(min(v)), "ratio_median_to_pre_edit": float(np.median(v)) / base, "all": v} for k, v in tim.items()}
    X.jdump(out, X.OUT + "/r22/overhead.json")
    X.say(lg, {k: round(v["ratio_median_to_pre_edit"], 3) for k, v in out["logs"][lg].items()})
B.check_w0()
