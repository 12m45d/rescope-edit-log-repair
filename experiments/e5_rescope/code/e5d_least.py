"""E5 (d): least repair on real streams (development logs, CPU, key cache).      python e5d_least.py <development log>
Two starts, both at the default role (last prompt position), guards kept:
  logonly   the log-only thresholds (driver method rescope_logonly: the largest pooled normalised cross-talk among the log's own
            solver keys): the start the task names.  If no benign query fires under it, the stream writes nothing and the checks hold trivially.
  neginf    all thresholds -inf (every edit is a candidate on every query: the deployed scoping at the last prompt position): the start for
            which the bounds T (1 + ... + 1/N) and T / (N + 1) are stated (per-edit running maxima); added because the first start is silent.  100 seeded streams of N = 50 benign queries drawn without
replacement from the valid REPAIR pool (numpy default_rng([20261111, 4, stream])); every query is repaired on arrival by the
library's least_repair on its teacher-forced last-prompt key (one call per query, default margins, on_conflict = 'refuse').
Per stream, 20 seeded arrival orders (order 0 = the drawn order):
  order independence   the final thresholds of the 20 orders are compared bit for bit (torch.equal)
  writes               number of threshold writes (sum of n_raised) against the bound T (1 + 1/2 + ... + 1/N)
  no regression        after every arrival, every query that was certified (no candidate edit) at an earlier step is still certified
  false firing         on heldout_disjoint (teacher-forced keys; near-scope probes separately) and on BEN-T (prompt-only keys,
                       fx.pools.firing; far = rows that the pool's flags do not relate to the family) after N repairs, against T / (N + 1)
"""
from e5common import *
from rescope import least_repair

NS, NQ, NO = 100, 50, 20

def main(name):
    for start in ("logonly", "neginf"):
        run(name, start)


def run(name, start):
    L = fx.logs.open(name)
    if start == "logonly":
        M0 = fx.methods.build("rescope_logonly", L)
    else:
        M0 = fx.methods.wrap(L, fx.methods.base_gate(L)[0], "rescope_neginf", {"cal": {"op": "none: all thresholds -inf", "n_edits_capped": 0}})
    S0 = M0.obj
    T = L.T
    rep, hd = L.pool("repair"), L.pool("heldout_disjoint")
    Krep = L.kc.keys("tf", rep)
    Rrep = fx.gate.features(L, S0, rep, "tf")                            # (n, T) the same coordinates the gate decides on
    Rhd = fx.gate.features(L, S0, hd, "tf")
    near = np.array([L.near_edit(j) is not None for j in hd])
    def ff(S, pre=""):
        f = (Rhd >= S.det_tau[None, :]).any(1).numpy()
        return {pre + "heldout_disjoint": float(f.mean() * 100), pre + "heldout_disjoint_far": float(f[~near].mean() * 100),
                pre + "heldout_disjoint_near": float(f[near].mean() * 100) if near.any() else None}
    bound_w = T * sum(1.0 / k for k in range(1, NQ + 1))
    out = {"log": name, "start_rule": start, "T": T, "n_streams": NS, "N": NQ, "n_orders": NO, "bound_writes": bound_w, "bound_false_firing": min(1.0, T / (NQ + 1.0)),
           "start": {"method": M0.label, "n_edits_capped": fx.methods.n_capped(M0), "cal": {k: v for k, v in (M0.rec.get("cal") or {}).items() if isinstance(v, (int, float, str, bool))},
                     **ff(S0), "repair_pool_firing": float((Rrep >= S0.det_tau[None, :]).any(1).double().mean() * 100)},
           "n_heldout_disjoint": len(hd), "n_heldout_near": int(near.sum()), "streams": []}
    b0 = fx.pools.firing(L, M0, "BEN-T")
    out["start"]["BEN-T"] = {"fired": b0["fired"], "far": b0["far"]["fired"], "n": b0["n"], "n_far": b0["far"]["n"], "n_related_excluded_from_far": b0["related"]["n"], "which": b0["which"]}
    t0 = time.time()
    for s in range(NS):
        idx = np.random.default_rng([20261111, 4, s]).choice(len(rep), NQ, replace=False)
        finals, writes, refused, regress, firing_q = [], [], [], 0, None
        for o in range(NO):
            order = idx if o == 0 else idx[np.random.default_rng([20261111, 5, s, o]).permutation(NQ)]
            S, w, nref, cert = S0, 0, 0, []
            for q in order:
                new, rec = least_repair(S, Krep[int(q)][None])
                if rec["refused"]:
                    nref += 1
                else:
                    S = new
                    w += rec["n_raised"]
                silent = not bool((Rrep[int(q)] >= S.det_tau).any())
                if cert:                                                # earlier certified queries stay certified
                    regress += int((Rrep[cert] >= S.det_tau[None, :]).any())
                if silent:
                    cert.append(int(q))
            finals.append(S.det_tau.clone())
            writes.append(w)
            refused.append(nref)
            if o == 0:
                Sfin, n_cert = S, len(cert)
                firing_q = int((Rrep[idx] >= S0.det_tau[None, :]).any(1).sum())
        same = all(torch.equal(finals[0], f) for f in finals[1:])
        Mf = fx.methods.wrap(L, Sfin, M0.label + "+least", {"cal": {"n_edits_capped": fx.methods.n_capped(M0)}, "stream": s})
        bt = fx.pools.firing(L, Mf, "BEN-T")
        out["streams"].append({"stream": s, "orders_identical_bitwise": bool(same), "writes": writes, "refused": refused, "n_regressions": regress,
                               "n_queries_firing_at_start": firing_q, "n_certified_after": n_cert, "n_thresholds_changed": int((finals[0] != S0.det_tau).sum()),
                               **ff(Sfin), "BEN-T": bt["fired"], "BEN-T_far": bt["far"]["fired"]})
        if s % 10 == 9:
            print(s, round(time.time() - t0), flush=True)
            jdump(out, f"{OUT}/d/{name}__{start}.partial.json")
    st = out["streams"]
    W = np.array([x["writes"] for x in st])
    out["summary"] = {"streams_with_identical_thresholds_for_all_orders": int(sum(x["orders_identical_bitwise"] for x in st)), "n_streams": NS,
                      "writes_mean": float(W.mean()), "writes_max": int(W.max()), "writes_min": int(W.min()), "writes_same_for_all_orders_in_streams": int((W.max(1) == W.min(1)).sum()),
                      "bound_writes": bound_w, "n_regressions_total": int(sum(x["n_regressions"] for x in st)), "refused_total_order0": int(sum(x["refused"][0] for x in st)),
                      "refused_same_for_all_orders": bool(all(len(set(x["refused"])) == 1 for x in st)),
                      "queries_firing_at_start_mean": float(np.mean([x["n_queries_firing_at_start"] for x in st])),
                      "certified_after_mean": float(np.mean([x["n_certified_after"] for x in st])),
                      **{k: qs([x[k] for x in st]) for k in ("heldout_disjoint", "heldout_disjoint_far", "BEN-T", "BEN-T_far")},
                      "heldout_disjoint_near": qs([x["heldout_disjoint_near"] for x in st]) if near.any() else None,
                      "bound_false_firing_pct": min(1.0, T / (NQ + 1.0)) * 100, "seconds": time.time() - t0}
    jdump(out, f"{OUT}/d/{name}__{start}.json")
    os.path.exists(f"{OUT}/d/{name}__{start}.partial.json") and os.remove(f"{OUT}/d/{name}__{start}.partial.json")
    print(json.dumps(out["summary"], indent=0), out["start"], flush=True)

if __name__ == "__main__":
    main(sys.argv[1])
