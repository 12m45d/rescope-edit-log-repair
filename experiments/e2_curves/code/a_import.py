"""E2 part a: import the existing records of the linear repair streams (P12) into cells of this experiment.

Nothing is evaluated here: the per-row flags of every record (held-out rows: reproduced before / after the repair; edits:
exact flags of Rel / T-Gen / I-Gen at B = 1) are copied into rows in the driver's vocabulary, with the key-cache identity of each
row (query cluster, heldout_disjoint / heldout_strict membership), so that render.py computes the metrics on heldout_disjoint.
Cells: part 'a', tag <METHOD>__n<n>__s<seed>; + NONE (deployed log) and ALL (pre-edit weight).  CPU only.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2c  # noqa: E402
from e2c import fx  # noqa: E402


def rows_of(L, rec, dep_em, by_group):
    ev = rec["eval"]
    rows = []
    for h in ev["heldout_rows"]:
        j = by_group[h["group"]]
        e = L.index[j]
        if e["uid"] != h["uid"] or e["ptype"] != h["ptype"] or e["pool"] != "heldout" or not e["y0_valid"]:
            raise RuntimeError(f"held-out row {h['group']} of the record does not match the key cache")
        rows.append({"kind": "probe", "csid": e["sid"], "sid": e.get("lib_sid"), "ptype": e["ptype"], "group": e["group"], "uid": e["uid"],
                     "pool": e["pool"], "disjoint": bool(e.get("disjoint")), "strict": bool(e.get("strict")), "query_id": e["query_id"],
                     "near_edit": L.near_edit(j), "y0_valid": True, "fired": None,
                     "exact": bool(h["reproduced_after"]), "D1": not h["reproduced_after"], "D1_dep": not h["reproduced_WE"], "src": "p12_record"})
    for m in ("rel", "tgen", "igen"):
        em, acc = ev["edits"][m]["em_list"], ev["edits"][m]["acc_list"]
        if len(em) != L.T:
            raise RuntimeError("edit flags of the record do not cover the log")
        for i in range(L.T):
            rows.append({"kind": "edit", "csid": L.sid(L.edit_rows[m][i]), "metric": m, "edit": i, "exact": bool(em[i]), "token_acc": float(acc[i]),
                         "exact_dep": bool(dep_em[m][i]), "fired": None, "src": "p12_record"})
    return rows


def main():
    for name in e2c.LOGS:
        L = fx.logs.open(name)
        v = fx.draws.verify_p12(L)
        if not v["ok"]:
            raise RuntimeError(f"{name}: the complaint sets of the driver differ from the P12 records: {v['differing']}")
        d = os.path.join(e2c.P12, "out", name)
        by_group = {L.index[j]["group"]: j for j in L.probe_rows}
        none = fx.io.jload(os.path.join(d, "NONE.json"))
        dep_em = {m: none["eval"]["edits"][m]["em_list"] for m in ("rel", "tgen", "igen")}
        # the deployed flags of the P12 evaluation against the log's own decoder-v2 records (the driver's complaint definition)
        n_bad = sum(bool(h["reproduced_WE"]) == bool(L.probe_rec(by_group[h["group"]])["is_complaint_D1"]) for h in none["eval"]["heldout_rows"])
        if n_bad:
            raise RuntimeError(f"{name}: {n_bad} deployed held-out flags of P12 differ from the log's records")
        exists = {}
        for meth, tag in (("deployed", "NONE"), ("preedit", "ALL")):
            rec = fx.io.jload(os.path.join(d, tag + ".json"))
            e2c.write_cell("a", name, tag, {"method": meth, "variant": "p12_record", "n": 0, "n_label": "0", "seed": None, "file": os.path.join(d, tag + ".json"),
                                            "restores": None}, rows_of(L, rec, dep_em, by_group))
        for meth in e2c.LINEAR:
            for n in e2c.NS:
                for s in (0, 1, 2):
                    tag = f"{meth}__n{n}__s{s}"
                    p = os.path.join(d, tag + ".json")
                    if not os.path.exists(p):
                        continue
                    rec = fx.io.jload(p)
                    groups = [L.index[j]["group"] for j in fx.draws.complaints(L, n, s)]
                    if list(rec["reported_groups"]) != groups:
                        raise RuntimeError(f"{tag}: reported complaints differ from the driver's nested set")
                    ex = rec.get("extra", {})
                    meta = {"method": meth, "variant": "p12_record", "n": len(groups), "n_label": str(n), "seed": s, "file": p, "options": rec.get("options"),
                            "restores": e2c.RESTORES[meth], "n_reported_resolved": sum(bool(c["D1_resolved"]) for c in rec["eval"]["complaints"]),
                            "fro_rel_to_log": rec["eval"].get("fro_rel_to_log"), "rank": rec.get("rank", {}).get("correction"),
                            "rollback_k": ex.get("k"), "rollback_R": ex.get("R"), "reedit_installed": ex.get("installed"),
                            "repair_wall_sec": rec.get("repair_wall_sec"), "record_time": rec.get("time")}
                    e2c.write_cell("a", name, tag, meta, rows_of(L, rec, dep_em, by_group))
                    exists.setdefault(meth, {}).setdefault(str(n), []).append(s)
        fx.io.jdump({"log": name, "cells_that_exist": exists, "verify_p12": {k: v[k] for k in ("n_records", "n_equal", "seeds", "ns", "ok")}},
                    os.path.join(e2c.OUT, "parts", f"a_exists_{name}.json"), indent=1)
        print(name, {m: {n: "".join(map(str, s)) for n, s in d2.items()} for m, d2 in exists.items()}, flush=True)


if __name__ == "__main__":
    main()
