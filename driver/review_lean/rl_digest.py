"""Review digest of the gate decisions of the EXISTING row set (both development logs, calibration draw 0, alpha = 0.05, n = 200).

    python rl_digest.py before | after
Own code (not tests/digest.py): per (log, method name) one sha1 over the bytes of every decision array of fx.gate.decide on every valid
probe / edit / hard-negative sequence at 'tf' and 'prompt', plus the bytes of the thresholds / baselines / eta.  No metric is printed.
"""
import hashlib
import json
import sys
import time

import rl_common as C
import numpy as np
import fx

STAGE1 = ("deployed", "preedit", "rescope", "rescope_shared", "rescope_null", "rescope_pac", "rescope_logonly", "cosine", "ball", "exact", "dual")
E1 = tuple(fx.methods.E1_LABELS)


def rows_of(L):
    out = {}
    out["probe"] = [int(j) for j in L.rows("probe", valid=True)]
    ed = []
    for m in sorted(L.edit_rows):
        ed.extend(int(j) for j in L.edit_rows[m])
    out["edit"] = ed
    out["hn"] = [int(j) for j in L.rows("hn", valid=True)]
    return out


def dig(L, M, R):
    h = hashlib.sha1()
    n = 0
    g = M.obj
    for name in ("det_tau", "det_b"):
        t = getattr(g, name, None)
        if t is not None and hasattr(t, "detach"):
            h.update(name.encode()); h.update(np.ascontiguousarray(t.detach().cpu().numpy()).tobytes())
    if hasattr(g, "eta"):
        h.update(repr(float(g.eta)).encode())
    try:
        for part in sorted(R):
            for which in ("tf", "prompt"):
                d = fx.gate.decide(L, M, R[part], which=which)
                for k in ("fired", "top_edit", "n_cand", "own_cand", "own_top"):
                    if k in d and d[k] is not None:
                        a = np.ascontiguousarray(np.asarray(d[k]).astype(np.int64))
                        h.update(f"{part}|{which}|{k}".encode()); h.update(a.tobytes()); n += a.size
    except Exception as e:                                   # methods that do not decide at the last prompt position
        return {"digest": h.hexdigest()[:16], "n": n, "no_decision": type(e).__name__, "m_digest": getattr(M, "digest", None)}
    return {"digest": h.hexdigest()[:16], "n": n, "m_digest": getattr(M, "digest", None)}


def run():
    res = {"fx": fx.code_fingerprint(), "logs": {}}
    for name in C.DEV:
        L = fx.logs.open(name)
        R = rows_of(L)
        out = {}
        t0 = time.time()
        for M in fx.methods.e1_rowset(L, 0.05, 0, labels=[l for l in E1]):
            out["e1:" + M.label] = dig(L, M, R)
        names = list(STAGE1) + [n_ for n_ in getattr(fx.review1, "NAMES", ()) if n_ not in STAGE1]
        for nm in names:
            try:
                M = fx.methods.build(nm, L, alpha=0.05, n=200, draw=0)
            except Exception as e:
                out["name:" + nm] = {"error": type(e).__name__}
                continue
            out["name:" + nm] = dig(L, M, R)
        res["logs"][name] = out
        print(name, len(out), "rows", round(time.time() - t0, 1), "s", flush=True)
    return res


if __name__ == "__main__":
    mode = sys.argv[1]
    r = run()
    p = C.dump(f"digest_{mode}.json", r)
    print("written", p)
    if mode == "after":
        b = json.load(open(C.OUT + "/digest_before.json"))
        diff = []; nrow = 0; ndec = 0
        for lg in b["logs"]:
            for k, v in b["logs"][lg].items():
                nrow += 1; ndec += v.get("n", 0)
                w = r["logs"][lg].get(k)
                if w is None or {x: v.get(x) for x in ("digest", "n", "m_digest", "error")} != {x: w.get(x) for x in ("digest", "n", "m_digest", "error")}:
                    diff.append((lg, k))
        C.dump("digest_compare.json", {"rows": nrow, "decisions": ndec, "differ": diff, "fx_before": b["fx"], "fx_after": r["fx"]})
        print("rows", nrow, "decisions", ndec, "differ", diff)
        sys.exit(1 if diff else 0)
