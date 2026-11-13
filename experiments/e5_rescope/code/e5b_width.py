"""E5 (b), CPU part: lineage sizes under the calibrated default method (two-block, alpha = 0.05, n = 200), 20 calibration draws.
    python e5b_width.py <development log>
Lineage size of a query = number of candidate edits at its last prompt position (fx.gate.decide(...)["n_cand"], teacher-forced key);
released = 1 if an edit is released (top-1), else 0.  Rows: valid heldout_disjoint probes (near-scope probes separately: the question
string is the question of an edit of the log), the D1 complaints of the REPAIR pool under the deployed log (they can be calibration
queries of a draw: counted), and BEN-T (prompt-only keys, fx.pools.decide; far = rows the pool's flags do not relate to the family).
The literal rule without guards (allow_silence=True) next to the method; all 20 draws are kept."""
from e5common import *

def summ(nc):
    nc = np.asarray(nc)
    nz = nc[nc > 0]
    return {"n": int(nc.size), "share_zero": float((nc == 0).mean() * 100) if nc.size else None, "median": float(np.median(nz)) if nz.size else None,
            "mean": float(nz.mean()) if nz.size else None, "max": int(nz.max()) if nz.size else None, "n_nonzero": int(nz.size),
            "share_width_at_most_1": float((nc <= 1).mean() * 100) if nc.size else None}

def main(name):
    L = fx.logs.open(name)
    hd = L.pool("heldout_disjoint")
    near = np.array([L.near_edit(j) is not None for j in hd])
    comp = [j for j in L.pool("repair") if L.probe_rec(j)["is_complaint_D1"]]
    out = {"log": name, "T": L.T, "n_heldout_disjoint": len(hd), "n_heldout_near": int(near.sum()), "n_repair_D1_complaints": len(comp), "draws": {}, "rule": __doc__}
    pooled = {k: [] for k in ("hd", "hd_far", "hd_near", "comp", "bent", "bent_far", "hd_lit", "comp_released")}
    Pt = fx.pools.get(L, "BEN-T")
    rel = fx.pools.related(L, Pt)
    for d in range(fx.draws.N_DRAWS):
        M = fx.methods.build("rescope", L, alpha=0.05, n=200, draw=d)
        Ml = fx.methods.build("rescope", L, alpha=0.05, n=200, draw=d, allow_silence=True)
        dh, dc = fx.gate.decide(L, M, hd, "tf"), fx.gate.decide(L, M, comp, "tf")
        dl = fx.gate.decide(L, Ml, hd, "tf")
        db = fx.pools.decide(L, M, Pt)
        cal = set(M.rec["draw"]["sids"]) if "draw" in M.rec and "sids" in M.rec["draw"] else set()
        nh, ncp, nb = np.asarray(dh["n_cand"]), np.asarray(dc["n_cand"]), np.asarray(db["n_cand"])
        out["draws"][str(d)] = {"n_edits_capped": fx.methods.n_capped(M), "heldout_disjoint": summ(nh), "heldout_disjoint_far": summ(nh[~near]), "heldout_disjoint_near": summ(nh[near]),
                                "heldout_disjoint_literal_rule": summ(np.asarray(dl["n_cand"])), "repair_D1_complaints": summ(ncp),
                                "repair_D1_complaints_released": float(np.asarray(dc["fired"]).mean() * 100),
                                "n_complaints_in_calibration_draw": int(sum(L.sid(j) in cal for j in comp)),
                                "BEN-T": summ(nb), "BEN-T_far": summ(nb[~rel]), "false_firing_heldout_disjoint": float(np.asarray(dh["fired"]).mean() * 100),
                                "false_firing_literal": float(np.asarray(dl["fired"]).mean() * 100)}
        for k, v in (("hd", nh), ("hd_far", nh[~near]), ("hd_near", nh[near]), ("comp", ncp), ("bent", nb), ("bent_far", nb[~rel]), ("hd_lit", np.asarray(dl["n_cand"]))):
            pooled[k].append(v)
    out["pooled_over_draws"] = {{"hd": "heldout_disjoint", "hd_far": "heldout_disjoint_far", "hd_near": "heldout_disjoint_near", "comp": "repair_D1_complaints", "bent": "BEN-T",
                                 "bent_far": "BEN-T_far", "hd_lit": "heldout_disjoint_literal_rule"}[k]: summ(np.concatenate(v)) for k, v in pooled.items() if v}
    out["n_draws"] = fx.draws.N_DRAWS
    out["BEN-T"] = {"n": len(Pt.rows), "n_related_excluded_from_far": int(rel.sum()), "which": Pt.which}
    jdump(out, f"{OUT}/b/{name}__two_block.json")
    print(name, out["pooled_over_draws"], flush=True)

if __name__ == "__main__":
    main(sys.argv[1])
