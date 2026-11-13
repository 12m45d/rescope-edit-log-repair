"""E10 verification (CPU): numbers of out/paper_numbers.json and out/table.txt reproduced from the per-row files with the
verifier's own aggregation.

    python v6_cells.py  ->  out/cells.json
(1) A seeded, stratified sample of 30 entries of paper_numbers.json (8 behaviour metrics, 4 public scores, 4 gate level, 7 relational
    workload, 5 equal evidence, 1 screening, 1 generation).  Each is recomputed from the raw per-row material of the experiment:
      behaviour     out/rows/<log>/<tag>.jsonl.gz (pooled over the calibration draws)
      public        out/public/<log>/<tag>/run_000/case_*.json (per-record success, mean over records, mean over draws)
      gate level    out/gate/<log>_flags.npz (per-row firing flags of the 20 draws); capped edits: per-draw records of out/gate/<log>.json
      relational    per-tuple values of out/r22/<log>.sqlite (table val), loaded into the VERIFIER's tables and scored by the verifier's
                    SQL (v3_r22.sql); the tuple table (relations, halves, intended values) is built by the verifier
      equal         out/rows_equal/<log>/<repair>__n<n>_s<seed>.jsonl.gz (pooled over seeds)
      screening     the verifier's out/screening_T2000.json (from the public factors)
      generation    out/gen/<log>/<label>.json (per-record n-gram entropy)
    and looked up in table.txt (the number formatted with one or two decimals occurs in the table text).
(2) EVERY data row of the main table (section 1 of table.txt: T100; T500 guards on / off; before / from position 173) is parsed and
    its behavioural cells (request exact, paraphrase exact, D1 HELD-OUT, kept, changed first tokens, D1 neighbourhood, kept; ES, PS, NS)
    are recomputed from the per-row files / case files; the intervals of the exact-match and D1 cells are recomputed with the
    verifier's crossed bootstrap (other random numbers: agreement within Monte-Carlo error is recorded).
"""
import os
import re
import sqlite3

import numpy as np

import vcommon as V

import e10t  # noqa: E402

P = V.jload(V.XOUT + "/paper_numbers.json")
TABLE = open(V.XOUT + "/table.txt").read()
_rows = {}


def rows_of(sub, log, tag):
    k = (sub, log, tag)
    if k not in _rows:
        _rows[k] = V.load_rows(V.XOUT + f"/{sub}/{log}/{tag}.jsonl.gz")
    return _rows[k]


def draws_of(sub, log, label):
    d = V.XOUT + f"/{sub}/{log}"
    tags = [f"{label}__d{i}" for i in (0, 1, 2) if os.path.exists(f"{d}/{label}__d{i}.jsonl.gz")]
    return tags or [f"{label}__dx"]


def select(rows, metric):
    """rows and field of a behavioural metric name (the summary's naming)."""
    p = metric.split(".")
    if p[0] == "edit":
        seg = p[3] if len(p) > 3 else None
        sub = [r for r in rows if r["kind"] == p[1] and r["set"] == "stream" and "new_exact" in r and r.get("part") is None]
        fld, unit = {"exact": "new_exact"}[p[2]], "pos"
    else:
        seg = p[1] if p[1] in ("before", "from") else None
        nm = p[-1]
        sub = [r for r in rows if (r["set"] == "heldout" if p[0] == "heldout" else r["kind"] == "neighborhood") and r["valid"] and r.get("part") is None]
        fld, unit = {"D1_tf": "D1_tf", "kept_tf": "tok_loc", "first_changed_tf": "first_changed_tf", "fired": "fired"}[nm], "case_id"
    if seg == "before":
        sub = [r for r in sub if r["pos"] < V.SPLIT]
    elif seg == "from":
        sub = [r for r in sub if r["pos"] >= V.SPLIT]
    return sub, fld, unit


def behave_value(sub, log, tags, metric, ci=False):
    Y, units = [], None
    for t in tags:
        rs, fld, unit = select(rows_of(sub, log, t), metric)
        rs = sorted(rs, key=lambda r: r["prompt_id"])
        Y.append([float(r[fld]) for r in rs])
        u = [r[unit] for r in rs]
        if units is not None and u != units:
            raise RuntimeError("rows of the draws differ")
        units = u
    v = 100.0 * float(np.mean([x for y in Y for x in y]))
    if not ci:
        return v
    b = V.crossed_bootstrap(Y, units)
    return v, b["ci_lo"], b["ci_hi"]


def public_value(sub, log, tags, seg, k):
    vals = []
    for t in tags:
        d = V.XOUT + f"/{sub}/{log}/{t}/run_000"
        per = []
        for f in os.listdir(d):
            cid = int(f[5:-5])
            if (seg == "before" and cid >= V.SPLIT) or (seg == "from" and cid < V.SPLIT):
                continue
            c = V.jload(os.path.join(d, f))["post"]
            if k == "ES":
                per.append(np.mean([x["target_true"] > x["target_new"] for x in c["rewrite_prompts_probs"]]))
            elif k == "PS":
                per.append(np.mean([x["target_true"] > x["target_new"] for x in c["paraphrase_prompts_probs"]]))
            else:
                per.append(np.mean([x["target_true"] < x["target_new"] for x in c["neighborhood_prompts_probs"]]))
        vals.append(100.0 * float(np.mean(per)))
    return float(np.mean(vals))


_gate = {}


def gate_value(log, lab, part, stat):
    if log not in _gate:
        _gate[log] = (np.load(V.XOUT + f"/gate/{log}_flags.npz"), V.jload(V.XOUT + f"/gate/{log}.json"))
    fl, g = _gate[log]
    base, mode = (lab[:-5], "goff") if lab.endswith("@goff") else (lab, "gon")
    if part == "cal":
        cal = g["rows"][base][mode]["cal"]
        return float(np.mean([cal[d][stat] for d in cal]))
    n = g["meta"]["n"][part]
    want = {"heldout": "fired", "paraphrase": "own_released", "neighborhood": "fired", "request": "own_released"}[part]
    if stat != want:
        raise KeyError(f"no per-row flags for {part}.{stat}")
    ds = sorted(int(k.split("|")[2]) for k in fl.files if k.startswith(f"{base}|{mode}|") and k.endswith("|" + part))
    return float(np.mean([100.0 * np.unpackbits(fl[f"{base}|{mode}|{d}|{part}"])[:n].mean() for d in ds]))


# ----------------------------------------------------------------------------- relational workload: verifier's tables, experiment's per-tuple values
SQL = {}
with open(os.path.join(V.CODE, "v3_r22.sql")) as f:
    for blk in re.split(r"^-- name: ", f.read(), flags=re.M)[1:]:
        nm, body = blk.split("\n", 1)
        SQL[nm.strip()] = body
_db = {}


def r22_db(log):
    if log in _db:
        return _db[log]
    import tp_common as C
    cf = C.load_counterfact()
    tok = e10t.data.tokenizer()
    L = e10t.open_log(log)
    pool = V.jload(V.OUT + "/r22/" + V.T100 + ".json")["pool"]
    if not (pool["same_records_as_experiment"] and pool["same_report_half"] and pool["same_test_half"]):
        raise RuntimeError("the verifier's VIEW pool differs from the experiment's")
    xp = V.jload(V.XOUT + "/r22/view_pool.json")                           # equal to the verifier's own pool (checked in v3_r22.py)
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.executescript(SQL["schema"])
    rows = []
    for i, r in enumerate(L.records):
        rr = cf[r]["requested_rewrite"]
        rows.append((r, rr["subject"], rr["relation_id"], rr["prompt"].format(rr["subject"]), 1, None, i, int(tok(" " + rr["target_new"]["str"].strip())["input_ids"][0])))
    for half in ("report", "test"):
        for r in xp[half]:
            rr = cf[r]["requested_rewrite"]
            rows.append((r, rr["subject"], rr["relation_id"], rr["prompt"].format(rr["subject"]), 0, half, None, None))
    db.executemany("INSERT INTO tuple VALUES (?,?,?,?,?,?,?,?)", rows)
    x = sqlite3.connect(f"file:{V.XOUT}/r22/{log}.sqlite?mode=ro", uri=True)
    db.executemany("INSERT INTO state (name, rid, v) VALUES ('R0', ?, ?)", x.execute("SELECT rid, value FROM val WHERE model = 'preedit'").fetchall())
    db.execute(SQL["fill_inview"])
    db.execute(SQL["intended_state"])
    db.commit()
    _db[log] = (db, x, set())
    return _db[log]


def r22_value(log, label, m, metric):
    db, x, have = r22_db(log)
    xr = V.jload(V.XOUT + f"/r22/{log}.json")
    models = [r["model"] for r in xr["rows"] if r["label"] == label and r.get("m") == m and r.get("scores") is not None]
    vals = []
    for model in models:
        if model not in have:
            db.executemany("INSERT INTO state (name, rid, v) VALUES (?, ?, ?)", [(model, a, b) for a, b in x.execute("SELECT rid, value FROM val WHERE model = ?", (model,)).fetchall()])
            have.add(model)
        q = lambda name, **kw: [dict(r) for r in db.execute(SQL[name], dict(kw, model=model)).fetchall()]      # noqa: E731
        if metric == "reference_preservation":
            r = q("reference_preservation")[0]
            vals.append(100.0 * r["n_ok"] / r["n"])
        elif metric in ("retention", "retention_before", "retention_from"):
            r = q("update_retention_range", lo=V.SPLIT if metric == "retention_from" else 0, hi=V.SPLIT if metric == "retention_before" else 10 ** 9)[0]
            vals.append(100.0 * r["n_ok"] / r["n"])
        elif metric == "queries_exact_pct":
            sv, ag = q("selection_views", half="test"), q("aggregates", half="test")
            vals.append(100.0 * (sum(v["n_wrong"] == 0 for v in sv) + sum(a["half_l1"] == 0 for a in ag)) / (len(sv) + len(ag)))
        elif metric == "aggregate_tv_mean":
            vals.append(float(np.mean([a["tv"] for a in q("aggregates", half="test")])))
        else:
            raise KeyError(metric)
    return float(np.mean(vals)), len(models)


def in_table(v):
    if isinstance(v, str):
        return v in TABLE
    cands = {f"{v:.1f}", f"{v:.2f}", f"{v:.3f}"} | ({str(int(v))} if float(v).is_integer() else set())
    return any(re.search(r"(?<![\d.])" + re.escape(c) + r"(?![\d])", TABLE) for c in cands)


# ----------------------------------------------------------------------------- (1) the sample of 30 cells
def kind_of(e):
    f, k = e.get("file", ""), e.get("key", "")
    if "summary/behave" in f:
        return "public" if ".public." in k else "behaviour"
    if "summary/gate" in f:
        return "gate"
    if "summary/r22" in f:
        return "relational"
    if "summary/equal" in f:
        return "equal"
    if "screening" in f:
        return "screening"
    if "/gen/" in f:
        return "generation"
    return "other"


groups = {}
for name in sorted(P):
    groups.setdefault(kind_of(P[name]), []).append(name)
# rounding of the paper numbers: the largest number of decimals printed for a kind of number (kind, statistic) gives the tolerance
DEC = {}
for name, e in P.items():
    if isinstance(e.get("value"), (int, float)) and not isinstance(e["value"], bool):
        kk = (kind_of(e), e["key"].split(".")[-2] if kind_of(e) != "screening" else e["key"].split(".")[-1])
        txt = repr(float(e["value"]))
        DEC[kk] = max(DEC.get(kk, 0), len(txt.split(".")[1]) if "." in txt and "e" not in txt else 0)
rng = np.random.default_rng([20261111, 30])
QUOTA = (("behaviour", 8), ("public", 4), ("gate", 4), ("relational", 7), ("equal", 5), ("screening", 1), ("generation", 1))
sample = [groups[g][int(i)] for g, n in QUOTA for i in sorted(rng.choice(len(groups[g]), size=n, replace=False))]
cells = []
for name in sample:
    e = P[name]
    f, key, val = e["file"], e["key"], e["value"]
    log = re.search(r"TX-G2XL-rome-\w+?(?=[._]|$)", f.split("/")[-1])
    log = log.group(0) if log else None
    kind = kind_of(e)
    rec = {"name": name, "kind": kind, "paper_value": val, "file": f, "key": key}
    try:
        if kind == "behaviour":
            i = key.index(".metrics.")
            label, metric = key[:i], key[i + 9:-6]
            mine = behave_value("rows", log, draws_of("rows", log, label), metric)
            rec.update(source=f"out/rows/{log}/{label}__d*.jsonl.gz", n_instances=len(draws_of("rows", log, label)))
        elif kind == "public":
            i = key.index(".public.")
            label, (seg, k, _) = key[:i], key[i + 8:].split(".")
            mine = public_value("public", log, draws_of("rows", log, label), seg, k)
            rec.update(source=f"out/public/{log}/{label}__d*/run_000/case_*.json")
        elif kind == "gate":
            body = key[len("rows."):-len(".mean")]
            lab, part, stat = body.rsplit(".", 2)
            mine = gate_value(log, lab, part, stat)
            rec.update(source=f"out/gate/{log}_flags.npz" if part != "cal" else f"out/gate/{log}.json (per-draw calibration records)")
        elif kind == "relational":
            i = int(re.match(r"rows\[(\d+)\]", key).group(1))
            srow = V.jload(V.XOUT + "/" + f.split("out/", 1)[1])["rows"][i]
            metric = key.split(".")[1]
            mine, ninst = r22_value(log, srow["label"], srow["m"], metric)
            rec.update(source=f"out/r22/{log}.sqlite (table val) -> verifier's tables and SQL", row=[srow["label"], srow["m"]], n_instances=ninst)
        elif kind == "equal":
            i = key.index(".metrics.")
            (rep, nn), metric = key[:i].split("|"), key[i + 9:-6]
            tags = [f"{rep}__{nn}_s{s}" for s in (0, 1, 2) if os.path.exists(V.XOUT + f"/rows_equal/{log}/{rep}__{nn}_s{s}.jsonl.gz")]
            Y = []
            for t in tags:
                rs, fld, _ = select(rows_of("rows_equal", log, t), metric)
                Y += [float(r[fld]) for r in rs]
            mine = 100.0 * float(np.mean(Y))
            rec.update(source=f"out/rows_equal/{log}/{rep}__{nn}_s*.jsonl.gz", n_instances=len(tags))
        elif kind == "screening":
            parts = key.split(".")
            s = V.jload(V.OUT + "/screening_T2000.json")["logs"][parts[1]]
            mine = s["n_flagged"] if parts[2] == "n_flagged" else s["against"][parts[3]][parts[4]]
            rec.update(source="verifier's out/screening_T2000.json (public factors)")
        elif kind == "generation":
            label = key.split(".")[1]
            g = V.jload(V.XOUT + f"/gen/{log}/{label}.json")["rows"]
            mine = 100.0 * float(np.mean([r["ngram_entropy"] for r in g if "failed" not in r]))
            rec.update(source=f"out/gen/{log}/{label}.json")
        tol = 0.5 * 10.0 ** (-DEC[(kind, key.split(".")[-2] if kind != "screening" else key.split(".")[-1])]) * 1.001
        rec["paper_decimals_of_this_kind_of_number"] = DEC[(kind, key.split(".")[-2] if kind != "screening" else key.split(".")[-1])]
        rec.update(recomputed=mine, abs_diff=abs(float(mine) - float(val)), reproduced=bool(abs(float(mine) - float(val)) <= tol), in_table_txt=in_table(float(val)))
    except Exception as ex:                                              # recorded, not hidden
        rec.update(error=f"{type(ex).__name__}: {ex}", reproduced=False)
    cells.append(rec)
    print(("ok  " if rec.get("reproduced") else "DIFF"), name, val, rec.get("recomputed"), rec.get("in_table_txt"), rec.get("error", ""))

# ----------------------------------------------------------------------------- (2) the main table, parsed
LAB = {"deployed log": "deployed", "pre-edit model": "preedit", "ReScope-W log-only (repair)": "where_logonly", "ReScope-W two-block (repair)": "where_twoblock",
       "ReScope (migration)": "rescope", "cosine, same calibration": "cosine", "whitecos_0.1 (best R3 gate)": "whitecos_0.1", "cosine 0.65, all positions": "cosine_published",
       "exact match": "exact", "lookup_subject [other class]": "lookup_subject"}
lines = TABLE.split("\n")
a = next(i for i, x in enumerate(lines) if x.startswith("1. MAIN TABLE"))
b = next(i for i, x in enumerate(lines) if x.startswith("2. GATE LEVEL"))
blocks, cur = [], None
for ln in lines[a:b]:
    s = ln.strip()
    if s.startswith("TX-G2XL-rome-T"):
        log = V.T100 if s.startswith(V.T100 + " ") else V.T500
        seg = ".before" if "BEFORE position" in s else (".from" if "FROM position" in s else "")
        cur = {"log": log, "seg": seg, "title": s[:90], "width": None, "rows": []}
        blocks.append(cur)
    elif cur is not None and s.startswith("----"):
        cur["width"] = len(ln[3:].split("  ")[0])
    elif cur is not None and cur["width"] and s and not s.startswith("row "):
        cur["rows"].append(ln)
main = {"n_rows": 0, "n_cells": 0, "n_cells_differ": 0, "differences": [], "ci_max_abs_deviation": 0.0, "n_ci_cells": 0, "n_ci_outside_1_point": 0, "blocks": []}
NUM = re.compile(r"-?\d+\.\d+|-?\d+")
for bl in blocks:
    log, seg = bl["log"], bl["seg"]
    nb = 0
    for ln in bl["rows"]:
        lab = ln[3:3 + bl["width"]].strip()
        goff = lab.endswith("/ guards off")
        base = LAB[lab.replace(" / guards off", "").strip()]
        label = base + ("@goff" if goff else "")
        x = [float(v) for v in NUM.findall(ln[3 + bl["width"]:])]
        # draws | req v lo hi | par v lo hi | ES v lo hi | PS v lo hi | NS v lo hi | D1 held-out v lo hi | kept | 1st chg | D1 nb v lo hi | kept
        tv = {"draws": x[0], "req": x[1:4], "par": x[4:7], "ES": x[7:10], "PS": x[10:13], "NS": x[13:16], "d1h": x[16:19], "kept_h": x[19], "chg_h": x[20], "d1n": x[21:24], "kept_n": x[24]}
        tags = draws_of("rows", log, label)
        ns = "neighborhood" + seg
        want = {"req": f"edit.request.exact{seg}", "par": f"edit.paraphrase.exact{seg}", "d1h": "heldout.D1_tf", "d1n": f"{ns}.D1_tf"}
        diffs = []
        for k, metric in want.items():
            v, lo, hi = behave_value("rows", log, tags, metric, ci=True)
            main["n_cells"] += 1
            if abs(v - tv[k][0]) > 0.0501:
                diffs.append((k, tv[k][0], v))
            for mine_, tab_ in ((lo, tv[k][1]), (hi, tv[k][2])):
                dev = abs(mine_ - tab_)
                main["n_ci_cells"] += 1
                main["ci_max_abs_deviation"] = max(main["ci_max_abs_deviation"], dev)
                main["n_ci_outside_1_point"] += int(dev > 1.0)
        for k, metric in (("kept_h", "heldout.kept_tf"), ("chg_h", "heldout.first_changed_tf"), ("kept_n", f"{ns}.kept_tf")):
            v = behave_value("rows", log, tags, metric)
            main["n_cells"] += 1
            if abs(v - tv[k]) > 0.0501:
                diffs.append((k, tv[k], v))
        for k in ("ES", "PS", "NS"):
            v = public_value("public", log, tags, seg[1:] if seg else "all", k)
            main["n_cells"] += 1
            if abs(v - tv[k][0]) > 0.06:
                diffs.append((k, tv[k][0], v))
        if len(tags) != int(tv["draws"]):
            diffs.append(("draws", tv["draws"], len(tags)))
        main["n_rows"] += 1
        main["n_cells_differ"] += len(diffs)
        nb += 1
        if diffs:
            main["differences"].append({"block": bl["title"], "row": lab, "cells": diffs})
    main["blocks"].append({"title": bl["title"], "n_rows": nb})
print("main table:", {k: v for k, v in main.items() if k != "blocks"})
# ----------------------------------------------------------------------------- (3) the relational rows quoted in the report, both logs, all rows
r22 = {}
for log in (V.T100, V.T500):
    xr = V.jload(V.XOUT + f"/r22/{log}.json")
    xs = V.jload(V.XOUT + f"/summary/r22_{log}.json")
    o = {"rows": {}, "n_rows": 0, "n_values_differ_from_summary": 0, "differences": []}
    for srow in xs["rows"]:
        if srow.get("reference_preservation") is None:
            continue
        key = f"{srow['label']}|m={srow['m']}"
        e = {}
        for metric in ("reference_preservation", "retention", "queries_exact_pct", "aggregate_tv_mean") + (("retention_before",) if log == V.T500 else ()):
            mine, ninst = r22_value(log, srow["label"], srow["m"], metric)
            theirs = srow[metric]["value"]
            e[metric] = [mine, theirs]
            if abs(mine - theirs) > 1e-6:
                o["n_values_differ_from_summary"] += 1
                o["differences"].append([key, metric, mine, theirs])
        if log == V.T500:
            e["retention_from_173_verifier_sql"] = r22_value(log, srow["label"], srow["m"], "retention_from")[0]
        e["cost"] = {k: srow[k]["value"] for k in ("diagnosis_passes", "repair_passes", "statements_changed")}
        e["complaints_refused"], e["n_tuple_complaints"], e["edits_lost"] = srow.get("complaints_refused"), srow.get("n_tuple_complaints"), srow.get("edits_lost")
        o["rows"][key] = e
        o["n_rows"] += 1
    o["max_queries_exact_pct"] = max(v["queries_exact_pct"][0] for v in o["rows"].values())
    # the charged passes, recounted: 2 forward passes per tuple of the complained views (experiment's complaint sets)
    cs = xr["complaints"]["sets"]
    o["diagnosis_passes_equal_two_per_view_tuple"] = all(v["diagnosis_passes"] == 2 * v["n_tuples_in_complained_views"] for v in cs.values())
    o["diagnosis_passes_mean_by_m"] = {m: float(np.mean([v["diagnosis_passes"] for v in cs.values() if v["m"] == m])) for m in (1, 3, 10)}
    o["tuple_complaints_mean_by_m"] = {m: float(np.mean([v["n_tuple_complaints"] for v in cs.values() if v["m"] == m])) for m in (1, 3, 10)}
    r22[log] = o
    print("r22", log, o["n_rows"], "rows; values differing from the summary:", o["n_values_differ_from_summary"], "max queries exact", o["max_queries_exact_pct"], o["diagnosis_passes_mean_by_m"], o["tuple_complaints_mean_by_m"])

out = {"relational_rows": r22, "sample_rule": "numpy default_rng([20261111, 30]); quota " + ", ".join(f"{g} {n}" for g, n in QUOTA), "n_cells": len(cells),
       "n_reproduced": sum(bool(c.get("reproduced")) for c in cells), "n_found_in_table_txt": sum(bool(c.get("in_table_txt")) for c in cells), "cells": cells, "main_table": main,
       "n_paper_numbers": len(P), "paper_numbers_by_kind": {g: len(v) for g, v in groups.items()}}
V.jdump(out, V.OUT + "/cells.json")
print(out["n_reproduced"], "of", out["n_cells"], "reproduced;", out["n_found_in_table_txt"], "found in table.txt")
