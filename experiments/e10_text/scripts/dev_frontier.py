"""Descriptive frontier on the DEVELOPMENT log only (the analogue of E1b of the protocol; nothing is selected or tuned with it).

    python scripts/dev_frontier.py        -> out/dev_frontier.json, out/dev_frontier.txt        (CPU, key cache only)

For each gate feature (normalised solver coordinate 'ncoord', raw cosine 'cos'; all positions of the prompt and of the
pre-edit continuation) ONE global threshold t is moved; read at the thresholds at which the share of HELD-OUT benign prompts
with a candidate equals 0.5, 1, 2, 5, 10 percent:
    coverage      share of paraphrase prompts on which the prompt's own edit is a candidate (own feature >= t at some position)
    request       the same for the request prompts
    neighbourhood share of neighbourhood prompts with a candidate
The thresholds of the calibrated rows sit near the 5 percent column by construction; the published cosine threshold 0.65 is
read as well.  Why this exists: in the mechanics run the published cosine gate fires on far fewer benign prompts than the
calibrated rows, at a lower coverage; the calibrated points are not at the same budget, this table is.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "code"))
import e10t  # noqa: E402
import fx  # noqa: E402
import numpy as np  # noqa: E402

L = e10t.open_log(e10t.DEV_LOG)
assert L.development
LEVELS = (0.5, 1.0, 2.0, 5.0, 10.0)
out = {"log": L.name, "code": e10t.code_fingerprint(), "levels_percent": list(LEVELS), "features": {},
       "what": "one global threshold per feature, read at matched firing on HELD-OUT benign prompts (descriptive; development log only)"}
rows = []
ho = L.benign("heldout")
for feat in ("ncoord", "cos"):
    S, _ = e10t.methods.base_gate(L, feat)
    Q = lambda rr: e10t.gate.features(L, S, [r["text_id"] for r in rr])["Q"].numpy()          # noqa: E731
    s_ho = Q(ho).max(1)
    qp, qr, qn = Q(L.rows["paraphrase"]), Q(L.rows["request"]), Q(L.rows["neighborhood"])
    own_p = qp[np.arange(len(qp)), [r["pos"] for r in L.rows["paraphrase"]]]
    own_r = qr[np.arange(len(qr)), [r["pos"] for r in L.rows["request"]]]
    s_nb = qn.max(1)
    srt = np.sort(s_ho)[::-1]

    def read(t):
        return {"threshold": float(t), "heldout_firing": float(100 * (s_ho >= t).mean()), "paraphrase_own_candidate": float(100 * (own_p >= t).mean()),
                "request_own_candidate": float(100 * (own_r >= t).mean()), "neighbourhood_firing": float(100 * (s_nb >= t).mean())}
    o = {"levels": {}}
    for lv in LEVELS:
        k = int(np.floor(lv / 100.0 * len(srt)))                        # at most k HELD-OUT prompts may have a candidate
        t = np.nextafter(srt[k], np.inf) if k < len(srt) else -np.inf   # the smallest threshold with at most k firing prompts
        o["levels"][str(lv)] = read(t)
        r = o["levels"][str(lv)]
        rows.append([feat, f"{lv:g} %", f"{r['threshold']:.4f}", f"{r['heldout_firing']:.2f}", f"{r['paraphrase_own_candidate']:.1f}", f"{r['request_own_candidate']:.1f}",
                     f"{r['neighbourhood_firing']:.2f}"])
    if feat == "cos":
        o["published_0.65"] = read(e10t.COS_PUBLISHED)
        r = o["published_0.65"]
        rows.append([feat, "t = 0.65", f"{r['threshold']:.4f}", f"{r['heldout_firing']:.2f}", f"{r['paraphrase_own_candidate']:.1f}", f"{r['request_own_candidate']:.1f}",
                     f"{r['neighbourhood_firing']:.2f}"])
    o["n"] = {"heldout": len(ho), "paraphrase": len(qp), "request": len(qr), "neighbourhood": len(qn)}
    out["features"][feat] = o
e10t.data.jdump(fx.io.clean(out), os.path.join(e10t.OUT, "dev_frontier.json"))
txt = (f"DEVELOPMENT LOG {L.name}: one global threshold per gate feature, read at matched firing on the {len(ho)} HELD-OUT benign prompts (descriptive, nothing selected)\n"
       f"code e10t {out['code']['e10t']}; all positions of the prompt and of the pre-edit 10-token continuation; percent\n\n"
       + fx.io.text_table(["feature", "HELD-OUT budget", "threshold", "HELD-OUT firing", "paraphrase: own edit candidate", "request: own", "neighbourhood firing"], rows) + "\n")
with open(os.path.join(e10t.OUT, "dev_frontier.txt"), "w") as f:
    f.write(txt)
print(txt)
