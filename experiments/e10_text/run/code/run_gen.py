"""E10 metric (v): degeneration.  n-gram entropy of the public generations (public generate_fast + n_gram_entropy, seeded as
the build) on the generation prompts of the first 100 requests: pre-edit model, deployed log, the repair (ReScope-W, both
threshold rules) and the migration (ReScope), calibration draw 0; on the collapse log also with guards off.

    $GQ run --mem 13000 --prio -2 --name e10n -- $PY run_gen.py --log TX-G2XL-rome-T100     -> out/gen/<log>/<label>.json, out/gen/<log>_summary.json
Next to the entropy: the share of the 8 most frequent words among all generated words (the audit found that the entropy barely
separates the collapsed logs; computed here from the stored texts).
"""
import argparse
import collections
import os

import xcommon as X

ap = argparse.ArgumentParser()
ap.add_argument("--log", required=True)
args = ap.parse_args()

import e10t  # noqa: E402

B = e10t.Backbone()
import xmeth as XM  # noqa: E402

L = e10t.open_log(args.log)
ses = XM.XSession(L, B)
plan = [("preedit", True), ("deployed", True)] + [(lab, g) for g in ((True, False) if L.T > X.SPLIT else (True,)) for lab in ("where_logonly", "where_twoblock", "rescope")]
out, ref = {}, None
for lab, g in plan:
    M = XM.method(L, lab, 0, g)
    p = os.path.join(X.OUT, "gen", L.name, M.label + ".json")
    e10t.public.generation(ses, M, p, records=L.records[:100], verbose=10)
    rows = e10t.data.jload(p)["rows"]
    s = e10t.public.generation_summary(rows, ref)
    if lab == "preedit":
        ref = s["per_text"]
        s = e10t.public.generation_summary(rows, ref)
    words = collections.Counter(w for r in rows if "text" in r for t in r["text"] for w in t.split())
    tot = sum(words.values())
    s["top8_word_share_pct"] = 100.0 * sum(c for _, c in words.most_common(8)) / tot if tot else None
    s["n_words"] = tot
    out[M.label] = {k: v for k, v in s.items() if k != "per_text"}
    X.jdump({"log": L.name, "draw": 0, "rows": out}, os.path.join(X.OUT, "gen", L.name + "_summary.json"))
    X.say(f"gen {L.name} {M.label}: GE {s['GE']}")
B.check_w0()
