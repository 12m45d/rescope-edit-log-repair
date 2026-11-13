"""Decision digest of the EXISTING row set (stage-1 extension rule: append-only).

    python tests/digest.py before        -> tests/out/digest_before.json   (run before the first change of the extension)
    python tests/digest.py after         -> tests/out/digest_after.json + comparison with digest_before.json (exit code 1 on a difference)

For both development logs, calibration draw 0, alpha = 0.05, n = 200 and every row of fx.methods.e1_rowset that builds without the
model: Method.key, the bytes of thresholds / eta / ranking baselines / detectors / released values, and, for gates that decide at
the last prompt position, every array of fx.gate.evaluate (fired, top_edit, n_cand, own_cand, own_top) at 'tf' and 'prompt' on the
probes, edits, hard negatives and at 'decode' on the benign pool.  One sha1 per (log, row); nothing of it is a metric.
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np  # noqa: E402

import fx  # noqa: E402

OUT = os.path.join(HERE, "out")
LABELS = ("deployed", "preedit", "rescope", "rescope_shared", "rescope_null", "where_logonly", "cosine", "cosine_shared", "cosine_default", "cosine_published",
          "ball", "ball_default", "exact", "dual")          # the rows of E1_LABELS at the time of the digest that build without the model


def _b(t):
    return np.ascontiguousarray(t.detach().cpu().numpy()).tobytes()


def digest_method(L, M) -> dict:
    h = hashlib.sha1()
    g = M.obj
    out = {"key": M.key, "kind": M.kind, "variant": M.variant}
    if hasattr(g, "det_tau"):
        for name in ("det_tau", "det_b", "det_u", "det_w", "det_s", "det_edit", "values"):
            t = getattr(g, name, None)
            h.update(name.encode())
            if t is not None:
                h.update(_b(t))
        h.update(repr(float(g.eta)).encode())
        h.update(f"{g.role}|{g.fire}|{g.response}".encode())
        out["gate_sha1"] = h.hexdigest()[:16]
        out["n_edits_capped"] = (M.rec.get("cal") or {}).get("n_edits_capped")
    if fx.gate.is_scoped(g):
        ev = fx.gate.evaluate(L, M, positions=("tf", "prompt"))
        hd = hashlib.sha1()
        n = 0
        for part in sorted(ev):
            for which in sorted(ev[part]):
                d = ev[part][which]
                for k in ("fired", "top_edit", "n_cand", "own_cand", "own_top"):
                    a = np.ascontiguousarray(np.asarray(d[k]).astype(np.int64))
                    hd.update(f"{part}|{which}|{k}".encode())
                    hd.update(a.tobytes())
                    n += a.size
        out["decisions_sha1"] = hd.hexdigest()[:16]
        out["n_decisions"] = n
    return out


def run() -> dict:
    res = {"fx_before_or_after": fx.code_fingerprint(), "alpha": 0.05, "n": 200, "draw": 0, "logs": {}}
    for name in fx.logs.names("development"):
        L = fx.logs.open(name)
        rows = fx.methods.e1_rowset(L, 0.05, 0, labels=LABELS)
        res["logs"][name] = {M.label: digest_method(L, M) for M in rows}
    return res


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "after"
    r = run()
    p = os.path.join(OUT, f"digest_{mode}.json")
    with open(p, "w") as f:
        json.dump(r, f, indent=1, sort_keys=True)
    print("written", p, {k: len(v) for k, v in r["logs"].items()})
    if mode == "after":
        with open(os.path.join(OUT, "digest_before.json")) as f:
            b = json.load(f)
        diff = [(lg, lab) for lg in b["logs"] for lab in b["logs"][lg] if b["logs"][lg][lab] != r["logs"].get(lg, {}).get(lab)]
        n = sum(len(v) for v in b["logs"].values())
        res = {"n_rows": n, "n_equal": n - len(diff), "differing": diff, "fx_before": b["fx_before_or_after"], "fx_after": r["fx_before_or_after"],
               "n_decisions": sum(d.get("n_decisions", 0) for v in b["logs"].values() for d in v.values()), "ok": not diff}
        with open(os.path.join(OUT, "digest_compare.json"), "w") as f:
            json.dump(res, f, indent=1, sort_keys=True)
        print("digest unchanged:", not diff, f"({n - len(diff)}/{n} rows)", diff[:5])
        sys.exit(0 if not diff else 1)
