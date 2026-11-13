"""E5 (b), GPU part: lineage sizes under the deployed scoping and under the log-only thresholds of ReScope-W (role all positions).
    gq run --mem 10500 --prio 2 -- python e5b_where.py <development log>
Rows: valid heldout_disjoint probes and the D1 complaints of the REPAIR pool under the deployed log.  Per row: the number of edits that
are candidates at some position of the teacher-forced pre-edit trace (hook field n_cand_edits), through the driver's session."""
from e5common import *

def main(name):
    L = fx.logs.open(name)
    H = fx.logs.load_model(name)
    ses = fx.behave.session(L, H)
    hd = L.pool("heldout_disjoint")
    comp = [j for j in L.pool("repair") if L.probe_rec(j)["is_complaint_D1"]]
    out = {"log": name, "T": L.T, "n_heldout_disjoint": len(hd), "n_repair_D1_complaints": len(comp), "rules": {}}
    for label, M in (("where_logonly", fx.methods.build("where_logonly", L, H)), ("deployed_scoping", fx.methods.where_least(L, [], H, start="deployed"))):
        p = f"{OUT}/b/{name}__{label}.jsonl.gz"
        if not os.path.exists(p):
            rows = ses.tf(M, hd + comp)
            keep = [{"j": int(j), "sid": L.sid(j), "set": "heldout_disjoint" if i < len(hd) else "repair_D1_complaint", "near": L.near_edit(j) is not None,
                     "ptype": L.index[j].get("ptype"), "n_cand_edits": r.get("n_cand_edits"), "fired": r.get("fired"), "D1": r.get("D1"),
                     "n_fire_pos": r.get("n_fire_pos"), "fired_last": r.get("fired_last")} for i, (j, r) in enumerate(zip(hd + comp, rows))]
            fx.io.save_rows(p, keep)
        out["rules"][label] = {"rows_file": os.path.basename(p), "n_edits_capped": fx.methods.n_capped(M) if label == "where_logonly" else None}
    out["session"] = ses.assert_clean()
    out["stats"] = dict(ses.stats)
    jdump(out, f"{OUT}/b/{name}__where.json")
    print("DONE", out["session"], flush=True)

if __name__ == "__main__":
    main(sys.argv[1])
