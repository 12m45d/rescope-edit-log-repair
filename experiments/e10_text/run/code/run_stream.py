"""E10 task 5 (part 2) and metric (vi): retention of the first 100 requests along the stream.

    $GQ run --mem 13000 --prio -2 --name e10s -- $PY run_stream.py
(a) deployed prefixes of the stream (the weight after the first P updates, replayed bit for bit): public scores and exact match of
    the first 100 requests, D1 of their neighbourhood prompts -- where the deployed model breaks down (descriptive; the split
    position 173 is fixed by the protocol, out/split.json, and is not derived from this curve);
(b) the repair (ReScope-W, log-only and two-block thresholds) and the migration (ReScope) on the same prefix logs, calibration
    draw 0, guards on (frozen default), and guards off for prefixes beyond the split.
Files: out/stream/<prefix log>/<tag>.json (+ rows_stream, public_stream), as run_behave.py.
"""
import argparse

import xcommon as X

ap = argparse.ArgumentParser()
ap.add_argument("--deployed", default="100,150,170,172,173,174,175,180,200,250,300,400,500,1000,2000")
ap.add_argument("--methods", default="100,150,173,174,200,300,500,1000,2000")
args = ap.parse_args()

import e10t  # noqa: E402

B = e10t.Backbone()
from e10t import methods as MT  # noqa: E402
import xmeth as XM  # noqa: E402
import xeval as XE  # noqa: E402

PD, PM = [int(x) for x in args.deployed.split(",")], [int(x) for x in args.methods.split(",")]
PARTS = ("edits", "neighborhood")
for P in sorted(set(PD) | set(PM)):
    L = e10t.open_log("TX-G2XL-rome-T2000", T=P)
    ses = XM.XSession(L, B)
    XE.evaluate(ses, MT.deployed(L), "deployed__dx", "stream", meta={"prefix": P}, first=100, parts=PARTS)
    if P in PM:
        for guards in (True, False) if P > X.SPLIT else (True,):
            for lab in ("where_logonly", "where_twoblock", "rescope"):
                M = XM.method(L, lab, 0, guards)
                c = M.rec.get("cal", {})
                XE.evaluate(ses, M, f"{M.label}__d0", "stream", meta={"prefix": P, "guards": guards, "eta": c.get("eta"), "n_edits_capped": c.get("n_edits_capped"),
                                                                    "own_not_candidate": M.rec["own_requests"]["n_not_candidate"]}, first=100, parts=PARTS)
    X.say(f"stream: prefix {P} done; session {dict((k, v) for k, v in ses.stats.items() if k in ('hook_mismatch', 'hook_mismatch_dec', 'sequences_run'))}")
    e10t.gate.clear_cache()
B.check_w0()
X.say("done")
