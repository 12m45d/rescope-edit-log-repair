#!/usr/bin/env python3
"""Collect the facts of the experimental setup (Sec. 5.1) into ONE JSON file, from files that the testbed and the
final-experiment caches stored. No model, no GPU, no import of project code: json, re and (for the pool split) numpy.

    python3 scripts/collect_setup.py            writes scripts/data/setup_facts.json
    python3 scripts/collect_setup.py --print    also prints one line per log

scripts/gen_numbers.py calls build() before it resolves the manifest, so `make numbers` refreshes the file. The JSON is
kept in the paper directory: where the research tree is not readable, the last collected copy is used.

Inputs (all read-only; paths relative to ROOT = project_3; the list of logs is manifest["logs"]):
  02_research/experiments/06_testbed_v3/<log>/metrics.json           T, backbone, layer, batch, editor rule, Rel / T-Gen /
                                                                      I-Gen (exact, token), stored benchmark T-Loc / I-Loc
  .../<log>/probe_records_v2.jsonl                                    one row per probe query: valid y0, D1 and D2 flags
  .../<log>/complaints_v2.json                                        the testbed's own D1 / D2 summary (cross-check),
                                                                      definitions of D1 / D2 (decoder lengths)
  02_research/experiments/09_testbed_scale/{registry,registry_text,registry_text2}.json, out_part2/plan.json
                                                                      dataset, editor, planned T of logs that are not built
  02_research/experiments/10_final/cache/cache_table.json             probe pools, hard negatives, benign pool (counts)
  02_research/experiments/10_final/cache/<backbone>/index.jsonl       pool / disjoint flag of every probe row (cross-check)
  02_research/experiments/10_final/PROTOCOL.md                        the frozen protocol: constants, by anchored patterns
  02_research/experiments/07_repair_v3/harness/sampler.py             SPLIT_SEED (cross-check of the seed in PROTOCOL.md)

What is DERIVED here (everything else is copied):
  * pools. The harness splits the probe ITEMS once per family (sampler.split_pools: sorted uids, numpy default_rng(seed)
    permutation, first half = REPAIR). HELD-OUT-DISJOINT = held-out rows whose query (ptype, prompt, image) does not
    occur in the REPAIR pool (rescope/splits.py: annotate). Both rules are re-implemented below and compared with the
    flags stored in the key-cache index wherever the index covers the family; a disagreement withholds the numbers.
  * complaint rate of a deployed log on a pool = D1 complaints / rows with a valid pre-edit answer (percent).
  * hard negatives: rows per edit = rows / edits of a group (must be an integer); per log, from rows.jsonl of the
    cache: the rows of the log's own edits (edit position < T; cross rows: both edits), how many are valid, how many
    have a source item that is related to another edit of the log (not a clean negative) and, for HN-I, how many ask a
    question string that an edit of the log also asks.
  * edit streams: the uids of a log's edits (per_edit_metrics.jsonl) are compared with the family stream of its backbone
    (key-cache index, kind rel: stream position -> edit uid). "is_stream_prefix" is what the text means by "longer logs
    continue shorter ones" and "the same edits" (edit ITEMS and their order; the bitwise identity of the factors is in
    the testbed reports). "n_edits_that_are_probe_items" must be 0.
  * paper names of the logs (manifest["logs"]["names"]): <backbone letter>-<editor letter><data letter><T>, with the
    ridge appended where two listed logs would otherwise share a name. Composed from stored fields, never typed.
  * boundary rule of PROTOCOL.md, amendment of 2026-11-10 17:40 (checked, not applied silently): a log listed for the
    paper (manifest["logs"]["items"]) must not [redacted: describes a related submission under double-blind review]. Violations are written to "boundary_violations" and gen_numbers.py refuses to continue.
A log is used only when it is COMPLETE: metrics.json, per_edit_metrics.jsonl, probe_records_v2.jsonl and
complaints_v2.json exist (the builder writes them in this order, the complaint file last), every one of them parses,
and the probe rows agree with the summary of complaints_v2.json (a file that is being written does neither). An
incomplete or missing log has "complete": false and no numbers, so its macros stay placeholders until `make numbers`
is run again; T, backbone and editor of a log that is not built are taken from the build plan (T_source = plan).
"""
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP
import hashlib
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
OUT_REL = "01_paper/v4_rescope/scripts/data/setup_facts.json"
TB = "02_research/experiments/06_testbed_v3"
SCALE = "02_research/experiments/09_testbed_scale"
FINAL = "02_research/experiments/10_final"
SAMPLER = "02_research/experiments/07_repair_v3/harness/sampler.py"
LOG_FILES = ("metrics.json", "per_edit_metrics.jsonl", "probe_records_v2.jsonl", "complaints_v2.json")
POOLS = ("all", "repair", "heldout", "heldout_disjoint")
# probe uid prefix -> benchmark (checked against the registries for every registered log)
UID_DATASET = {"eval": "VLKEB", "mmke-entity-eval": "MMKE-Bench", "cf": "CounterFact"}
# [not released: two constants of the boundary check (an internal tag and a log name); they describe a related submission under double-blind review]


def find_root():
    """ROOT = the directory that holds 02_research (PROJECT3_ROOT overrides; works from a scratch copy of the paper)."""
    env = os.environ.get("PROJECT3_ROOT")
    if env:
        return env
    return _RP.ROOT          # the working-tree view of this repository (rescope_paths.py)


def jload(path):
    with open(path) as f:
        return json.load(f)


def jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class Inputs:
    """Remembers every file that was read (mtime, size), so that a stale collection can be detected."""

    def __init__(self, root):
        self.root, self.seen = root, {}

    def path(self, rel):
        return os.path.join(self.root, rel)

    def exists(self, rel):
        return os.path.exists(self.path(rel))

    def note(self, rel):
        p = self.path(rel)
        if os.path.exists(p):
            st = os.stat(p)
            self.seen[rel] = {"mtime": int(st.st_mtime), "size": st.st_size}
        else:
            self.seen[rel] = None
        return p

    def json(self, rel):
        return jload(self.note(rel))

    def jsonl(self, rel):
        return jsonl(self.note(rel))


# ------------------------------------------------------------------------------------------------ pools
def harness_split(uids, seed):
    """REPAIR-pool uids, as 07_repair_v3/harness/sampler.py: split_pools(frac = 0.5)."""
    import numpy as np
    u = sorted(set(uids))
    perm = np.random.default_rng(int(seed)).permutation(len(u))
    return {u[int(i)] for i in perm[:len(u) // 2]}


def annotate(rows, repair_uids):
    """pool and disjoint flag of every probe row, as rescope/splits.py: annotate (query = ptype, prompt, image)."""
    repq = set()
    for r in rows:
        if r["uid"] in repair_uids:
            repq.add((r["ptype"], r["prompt"], r["image"]))
    out = []
    for r in rows:
        pool = "repair" if r["uid"] in repair_uids else "heldout"
        out.append({"pool": pool, "disjoint": pool == "heldout" and (r["ptype"], r["prompt"], r["image"]) not in repq})
    return out


def count(rows):
    v = [r for r in rows if r["y0_valid"]]
    d1 = sum(bool(r["is_complaint_D1"]) for r in v)
    return {"n_rows": len(rows), "n_valid": len(v), "n_D1": d1, "D1_rate": (100.0 * d1 / len(v)) if v else None,
            "n_pre_correct": sum(bool(r["pre_correct_long"]) for r in rows), "n_D2": sum(bool(r["is_complaint_D2"]) for r in rows)}


def index_flags(inp, backbone):
    """From the key-cache index of a backbone: {(family, group): (pool, disjoint, uid)} of the probe rows and
    {family: [edit uid at stream position 0, 1, ...]}; ({}, {}) if the index cannot be read."""
    rel = f"{FINAL}/cache/{backbone}/index.jsonl"
    if not inp.exists(rel):
        return {}, {}
    out, pos = {}, {}
    with open(inp.note(rel)) as f:
        for line in f:
            if '"kind": "probe"' in line:
                r = json.loads(line)
                out[(r["family"], r["group"])] = (r["pool"], bool(r["disjoint"]), r["uid"])
            elif '"kind": "rel"' in line:
                r = json.loads(line)
                pos.setdefault(r["family"], {})[int(r["edit"])] = r["edit_uid"]
    streams = {}
    for fam, d in pos.items():
        if sorted(d) == list(range(len(d))):
            streams[fam] = [d[i] for i in range(len(d))]
    return out, streams


# ------------------------------------------------------------------------------------------------ hard negatives
def hn_rows(inp, backbone):
    """Rows of the final hard-negative set of a backbone (cache/hard_negatives/<backbone>/rows.jsonl), reduced to the
    fields that the counts need; None if the file cannot be read."""
    rel = f"{FINAL}/cache/hard_negatives/{backbone}/rows.jsonl"
    if not inp.exists(rel):
        return None
    out = []
    try:
        for r in inp.jsonl(rel):
            out.append({"type": r["hn_type"], "edit": int(r["edit_pos"]), "edit_b": r.get("edit_b"), "valid": bool(r["y0_valid"]),
                        "new": bool(r["new"]), "src_rel": r.get("src_related_edit_first_pos"),
                        "q_edit": r.get("q_in_edit_first_pos"), "in_pool": r.get("src_in_final_pool")})
    except (ValueError, KeyError, TypeError):
        return None
    return out


def hn_for_log(rows, T):
    """Counts of the hard negatives of the first T edits of the stream (the rows on which a log with T edits is tested)."""
    own = [r for r in rows if r["edit"] < T and (r["edit_b"] is None or int(r["edit_b"]) < T)]
    out = {"rows": len(own), "valid": sum(r["valid"] for r in own),
           "source_related_to_an_edit_of_the_log": sum(1 for r in own if r["src_rel"] is not None and r["src_rel"] < T),
           "by_type": {}}
    for typ in sorted({r["type"] for r in rows}):
        sub = [r for r in own if r["type"] == typ]
        out["by_type"][typ] = {"rows": len(sub), "valid": sum(r["valid"] for r in sub),
                               "question_of_an_edit_of_the_log": sum(1 for r in sub if r["q_edit"] is not None and r["q_edit"] < T)}
    return out


# ------------------------------------------------------------------------------------------------ one log
def registries(inp):
    reg = {}
    for fn in ("registry.json", "registry_text.json", "registry_text2.json"):
        rel = f"{SCALE}/{fn}"
        if inp.exists(rel):
            for k, v in inp.json(rel).items():
                reg[k] = dict(v, _registry=fn)
    rel = f"{SCALE}/out_part2/plan.json"
    if inp.exists(rel):
        plan = inp.json(rel)
        for k, v in (plan.get("registry_entries") or {}).items():
            reg.setdefault(k, dict(v, _registry="out_part2/plan.json (planned)"))
        for fam in (plan.get("families") or {}).values():
            for t, name in (fam.get("names") or {}).items():
                reg.setdefault(name, {})["_planned_T"] = int(t)
    return reg


def editor_family(rule, reg):
    if rule:
        m = re.match(r"([a-z]+)\(", rule)
        if m:
            return m.group(1)
    if reg:
        return reg.get("editor") or ("alphaedit" if reg.get("gamma") is not None else None)
    return None


def log_record(inp, name, reg, seed, idx_cache, warn, hn_families=None):
    d = f"{TB}/{name}"
    have = {fn: inp.exists(f"{d}/{fn}") for fn in LOG_FILES}
    r = reg.get(name) or {}
    rec = {"name": name, "exists": inp.exists(d), "complete": all(have.values()),
           "missing_files": [fn for fn, ok in have.items() if not ok],
           "T": r.get("_planned_T"), "T_source": "plan" if r.get("_planned_T") else None,
           "backbone": r.get("backbone"), "editor": editor_family(None, r), "dataset": r.get("dataset"),
           "statistics": r.get("source"), "registry": r.get("_registry")}
    if not rec["complete"]:
        for fn in LOG_FILES:                                   # note the absent files: a later run sees them appear
            inp.note(f"{d}/{fn}")
        return rec
    try:
        m = inp.json(f"{d}/metrics.json")
        c = inp.json(f"{d}/complaints_v2.json")
        rows = inp.jsonl(f"{d}/probe_records_v2.jsonl")
        pe = inp.jsonl(f"{d}/per_edit_metrics.jsonl")
        edit_uids = [x["uid"] for x in sorted(pe, key=lambda x: x["edit_id"])]
        int(m["n_edits"]), rows[0]["uid"], rows[0]["is_complaint_D1"]
    except (ValueError, KeyError, IndexError, TypeError) as err:    # a file that is being written
        rec.update(complete=False, incomplete_reason=f"a stored file does not parse ({type(err).__name__}): being written?")
        warn(f"{name}: a stored file does not parse ({type(err).__name__}); the log is treated as not built")
        return rec
    cfg = m.get("config") or {}
    rule = (m.get("build") or {}).get("rule")
    prefixes = sorted({x["uid"].rsplit(":", 1)[0] for x in rows})
    ds_uid = UID_DATASET.get(prefixes[0]) if len(prefixes) == 1 else None
    ds_reg = cfg.get("dataset") or r.get("dataset")
    if ds_reg and ds_uid and ds_reg != ds_uid:
        warn(f"{name}: dataset of the registry ({ds_reg}) and of the probe uids ({ds_uid}) differ")
    rec.update({
        "T": int(m["n_edits"]), "T_source": "metrics.json:n_edits", "backbone": cfg.get("backbone"), "layer": cfg.get("layer"),
        "layer_name": m.get("layer_name"), "dtype": m.get("dtype"), "batch": cfg.get("batch"), "rule": rule,
        "editor": editor_family(rule, r), "ridge": cfg.get("l2"), "gamma": cfg.get("gamma"),
        "lam_hat": cfg.get("lam_hat", r.get("lam_hat")), "statistics": cfg.get("source"),
        "dataset": ds_reg or ds_uid, "dataset_source": "registry / metrics.json" if ds_reg else "probe uid prefix",
        "modality": cfg.get("modality") or "image+text",
        "edits": {k: {"exact": m.get(f"{K}_exact"), "token": m.get(f"{K}_token"), "n": ((m.get("post") or {}).get(k) or {}).get("n"),
                      "nonfinite": ((m.get("post") or {}).get(k) or {}).get("nonfinite")}
                  for k, K in (("rel", "Rel"), ("tgen", "TGen"), ("igen", "IGen"))},
        "pre_edits": {k: ((m.get("pre") or {}).get(k) or {}).get("exact") for k in ("rel", "tgen", "igen")},
        "locality": {"tloc": m.get("TLoc_probes"), "iloc": m.get("ILoc_probes"), "nonfinite": m.get("loc_nonfinite"),
                     "what": "stored benchmark locality (vendor pass, reference answers, all probe items)"},
        "update_norm_over_W0": (m.get("build") or {}).get("composite_over_W0"),
        "definitions": c.get("definitions"),
    })
    # ---- pools: recomputed split (numpy) and the flags of the key-cache index, where the index has the family
    probes = {"all": count(rows)}
    for pt in sorted({x["ptype"] for x in rows}):
        probes["ptype_" + pt] = count([x for x in rows if x["ptype"] == pt])
    ann, how = None, []
    try:
        ann = annotate(rows, harness_split([x["uid"] for x in rows], seed))
        how.append("recomputed (harness rule, numpy)")
    except ImportError:
        warn(f"{name}: numpy is not available, the pool split is taken from the key-cache index only")
    bb = rec["backbone"]
    if bb not in idx_cache:
        idx_cache[bb] = index_flags(inp, bb) + (hn_rows(inp, bb),)
    flags, streams, hnr = idx_cache[bb]
    fams = sorted({k[0] for k in flags})
    fam = next((f for f in fams if all((f, x["group"]) in flags and flags[(f, x["group"])][2] == x["uid"] for x in rows)), None)
    # ---- edit stream: the cached family stream of which the log is a prefix (None: the key cache has no such stream)
    sfam = next((f for f, u in streams.items() if u[:len(edit_uids)] == edit_uids), None)
    rec["stream"] = {"cache_family": sfam, "n_edits_listed": len(edit_uids), "edit_uids": edit_uids,
                     "n_edits_that_are_probe_items": len(set(edit_uids) & {x["uid"] for x in rows})}
    if len(edit_uids) != rec["T"]:
        warn(f"{name}: per_edit_metrics.jsonl lists {len(edit_uids)} edits, metrics.json says T = {rec['T']}")
    if rec["stream"]["n_edits_that_are_probe_items"]:
        warn(f"{name}: {rec['stream']['n_edits_that_are_probe_items']} edit items are probe items")
    if hnr is not None and sfam is not None and sfam in ((hn_families or {}).get(bb) or ()):    # the stream that has hard negatives
        rec["hard_negatives"] = hn_for_log(hnr, rec["T"])
    if fam is not None:
        ann_idx = [{"pool": flags[(fam, x["group"])][0], "disjoint": flags[(fam, x["group"])][1]} for x in rows]
        how.append(f"key-cache index, family {fam}")
        if ann is not None and ann != ann_idx:
            warn(f"{name}: the recomputed pool split DIFFERS from the key-cache index: pool numbers withheld")
            rec["pool_check"] = "MISMATCH between the recomputed split and the key-cache index"
            ann = None
        elif ann is None:
            ann = ann_idx
        else:
            rec["pool_check"] = "recomputed split == key-cache index"
    elif ann is not None:
        rec["pool_check"] = "recomputed only (the key-cache index does not cover this family)"
    if ann is not None:
        probes["repair"] = count([x for x, a in zip(rows, ann) if a["pool"] == "repair"])
        probes["heldout"] = count([x for x, a in zip(rows, ann) if a["pool"] == "heldout"])
        probes["heldout_disjoint"] = count([x for x, a in zip(rows, ann) if a["disjoint"]])
        probes["n_items"] = len({x["uid"] for x in rows})
        probes["n_heldout_rows_with_a_repair_query"] = probes["heldout"]["n_rows"] - probes["heldout_disjoint"]["n_rows"]
    rec["probes"], rec["pool_source"] = probes, how
    # ---- cross-check with the testbed's own summary
    s = (c.get("summary") or {}).get("all") or {}
    ok = s.get("D1_n") == probes["all"]["n_D1"] and s.get("n_valid_y0") == probes["all"]["n_valid"] and s.get("D2_n") == probes["all"]["n_D2"]
    rec["summary_check"] = "probe rows == complaints_v2.json summary" if ok else "MISMATCH with complaints_v2.json summary"
    if not ok:                                                 # inconsistent stored files: no number of this log is used
        warn(f"{name}: counts of probe_records_v2.jsonl differ from complaints_v2.json ({probes['all']['n_D1']} against "
             f"{s.get('D1_n')} D1 complaints): the log is treated as not built")
        return {k: rec[k] for k in ("name", "exists", "T", "T_source", "backbone", "editor", "dataset", "statistics", "registry",
                                    "ridge", "summary_check")} | {"complete": False, "missing_files": [],
                                                                  "incomplete_reason": "probe rows and complaint summary disagree"}
    return rec


# ------------------------------------------------------------------------------------------------ caches
def cache_record(inp, warn):
    rel = f"{FINAL}/cache/cache_table.json"
    if not inp.exists(rel):
        warn("cache_table.json not found: pools and hard negatives are missing")
        return {}, {}, {}
    ct = inp.json(rel)
    pools, hn, bbs = {}, {}, {}
    for bb, v in ct["backbones"].items():
        meta = v.get("meta") or {}
        bbs[bb] = {"layer": meta.get("layer"), "d_in": meta.get("d_in"), "module_dtype": meta.get("module_dtype"),
                   "families": meta.get("families"),
                   "families_with_hard_negatives": sorted({k.split("|")[0] for k in (v.get("by_family_kind") or {}) if "|hn_" in k})}
        pools[bb] = {}
        for fam, pp in (v.get("probe_pools") or {}).items():
            q = {k: dict(x) for k, x in pp.items()}
            q["n_heldout_rows_with_a_repair_query"] = pp["heldout"]["rows"] - pp["heldout_disjoint"]["rows"]
            q["n_heldout_valid_with_a_repair_query"] = pp["heldout"]["valid"] - pp["heldout_disjoint"]["valid"]
            pools[bb][fam] = q
        h = {"types": {}, "source_pool_items": (v.get("hn_source_pool") or {}).get("n_items"),
             "later_edits_per_prefix": v.get("hn_later100_per_prefix")}
        for key, g in (v.get("hard_negatives") or {}).items():
            typ, grp, _ = key.split("|")
            t = h["types"].setdefault(typ, {"rows": 0, "valid": 0, "groups": {}})
            gg = t["groups"].setdefault(grp, {"rows": 0, "valid": 0, "edits": g["n_edits"]})
            if gg["edits"] != g["n_edits"]:
                warn(f"hard negatives {bb} {key}: groups with different numbers of edits")
            for a, b in (("rows", "n"), ("valid", "valid")):
                gg[a] += g[b]
                t[a] += g[b]
        for typ, t in h["types"].items():
            for grp, gg in t["groups"].items():
                per = gg["rows"] / gg["edits"]
                gg["rows_per_edit"] = int(per) if per == int(per) else None
                if gg["rows_per_edit"] is None:
                    warn(f"hard negatives {bb} {typ} {grp}: {gg['rows']} rows for {gg['edits']} edits is not an integer per edit")
        h["rows_drawn_earlier"] = sum(g["n"] for key, g in (v.get("hard_negatives") or {}).items() if key.endswith("|existing"))
        h["rows"] = sum(t["rows"] for t in h["types"].values())
        h["valid"] = sum(t["valid"] for t in h["types"].values())
        hn[bb] = h
        b = v.get("benign") or {}
        pools[bb]["benign"] = {"rows": sum(x.get("n", 0) for x in b.values()),
                               "by_ptype": {k: {"rows": x.get("n"), "answer_subset": x.get("subset"), "answer_subset_valid": x.get("subset_valid")}
                                            for k, x in b.items()}}
    return pools, hn, bbs


# ------------------------------------------------------------------------------------------------ protocol
def floats(s):
    return [float(x) for x in s.split(",")]


def ints(s):
    return [int(x) for x in s.split(",") if x.strip().isdigit()]


#   name -> (pattern on the text with white space collapsed, {field: (group, converter)})
PROTOCOL_PATTERNS = {
    "default_calibration": (r"Default alpha = (0\.\d+), n = (\d+)\.", {"alpha": (1, float), "n_cal": (2, int)}),
    # added 2026-11-12 (FIXES_FOR_FINAL_PASS F26): the fraction that the log-only rule of ReScope-W adds (macro \pcLogOnlyPct)
    "logonly_rule": (r"Thresholds: log-only rule\s*\((\d+) % above the largest coordinate on the other logged sequences\)", {"margin_pct": (1, int)}),
    "default_calibration_two_block": (r"Default n = (\d+), alpha = (0\.\d+)\.", {"alpha": (2, float), "n_cal": (1, int)}),
    "e1": (r"\*\*E1 Main comparison.{0,120}?alpha in \{([^}]+)\}, n = (\d+), (\d+) random calibration draws",
           {"alpha_grid": (1, floats), "alpha_grid_text": (1, str), "n_cal": (2, int), "draws": (3, int)}),
    "e2": (r"\*\*E2 Evidence curves.{0,200}?n in \{([^}]+)\}", {"evidence_grid": (1, ints), "evidence_grid_text": (1, str)}),
    "e2_seeds": (r"(\d+) seeds for n <= (\d+), (\d+) seed above", {"seeds_small": (1, int), "n_small_max": (2, int), "seeds_large": (3, int)}),
    "e2_switch": (r"while n < (\d+), and switches to the two-block rule at alpha = (0\.\d+) from n = (\d+) on",
                  {"n_switch": (3, int), "alpha": (2, float)}),
    "e3": (r"\*\*E3 Scale.{0,60}?T in \{([^}]+)\} \(Qwen\), \{([^}]+)\} \(LLaVA\)",
           {"T_qwen": (1, ints), "T_qwen_text": (1, str), "T_llava": (2, ints), "T_llava_text": (2, str)}),
    "e5c": (r"empirical false-firing rate versus alpha for n in \{([^}]+)\}, (\d+) draws",
            {"n_grid": (1, ints), "n_grid_text": (1, str), "draws": (2, int)}),
    "bootstrap": (r"percentile bootstrap, (\d+) resamples", {"resamples": (1, int)}),
    "spread": (r"mean and (\d+)th-(\d+)th percentile over draws", {"lo": (1, int), "hi": (2, int)}),
    "split": (r"harness split by item, seed (\d+)", {"seed": (1, int)}),
    "duplicates": (r"contains (\d+) rows that duplicate REPAIR queries", {"n_heldout_rows_with_a_repair_query": (1, int)}),
    "exact_gate": (r"EXACT \(cosine >= (0\.\d+)\)", {"cosine": (1, float)}),
    "ridges": (r"the same stream at ridge l2 = ([0-9, ]+?) \|", {"ridges": (1, floats), "ridges_text": (1, str)}),
    "public_default": (r"lam_hat = (\d+) fixed in advance", {"lam_hat": (1, int)}),
    "boundary": (r"enter the paper only for T <= (\d+)", {"T_max_boundary": (1, int)}),
    "d2": (r"whose (\d+)-token answer loses the reference", {"long_answer_tokens": (1, int)}),
    "contamination": (r"contamination of the calibration sample by in-scope queries \(([0-9, ]+) percent",
                      {"percent": (1, ints), "percent_text": (1, str)}),
    "dual_scale": (r"first (\d+) edits, (\d+) sampled later edits", {"first_edits": (1, int), "later_edits": (2, int)}),
    "freeze": (r"whatever logs exist on (\d{4}-\d{2}-\d{2}) \(results freeze\)", {"date": (1, str)}),
    # ---- amendments of 2026-11-11 (review round 1: R6, R8-R10, R12, R13; reduction R19; text logs R14, R21, R22; E9).
    # Added with the setup of the reduced programme: the constants that Sec. 5 states BEFORE the results (endpoints
    # H1 and H3 with their decision rules, draws, later edits, prefixes, the relational workload).
    "h1": (r"H1 \(repair at equal evidence; E2 a, b\): held-out D1 after n = (\d+) reported complaints.{0,260}?"
           r"if D1 is at most (\d+) percent with Rel exact within (\d+) points of the deployed log",
           {"n_complaints": (1, int), "d1_max_percent": (2, int), "rel_points": (3, int)}),
    "h3": (r"averaged over the three kinds, is at most (\d+) percent.{0,420}?by at least (\d+) points of H3 on both backbones",
           {"fit_firing_max_percent": (1, int), "dominates_points": (2, int)}),
    "later_edits": (r"stream position >= (\d+) for Qwen, >= (\d+) for LLaVA", {"qwen": (1, int), "llava": (2, int)}),
    "pac": (r"calibration-conditional form, delta = (0\.\d+)", {"delta": (1, float)}),
    "crossed_bootstrap": (r"crossed bootstrap over calibration draws and units \((\d+) resamples\)", {"resamples": (1, int)}),
    "prefixes": (r"Prefix logs of the two development streams after (\d+), (\d+) and (\d+) edits",
                 {"a": (1, int), "b": (2, int), "c": (3, int)}),
    "synthetic_scale": (r"E8 adds synthetic logs with T up to (\d+)", {"T_max": (1, int)}),
    "r19_e1": (r"\(a\) E1: behaviour at alpha = (0\.\d+) for calibration draws (\d+) to (\d+)",
               {"alpha": (1, float), "draw_first": (2, int), "draw_last": (3, int)}),
    "e9": (r"revocation sets R of m in \{([^}]+)\} edits", {"m_grid": (1, ints), "m_grid_text": (1, str)}),
    "text_equal_evidence": (r"Equal evidence, n in \{([^}]+)\} reported D1 complaints", {"n_grid": (1, ints), "n_grid_text": (1, str)}),
    "text_split": (r"The split position is (\d+), from the audit", {"position": (1, int)}),
    "text_decode": (r"the pre-edit greedy continuation of (\d+) tokens is not reproduced", {"tokens": (1, int)}),
    "r22_view": (r"\*\*R22 Relational maintenance workload.{0,900}?a VIEW pool of (\d+) records", {"records": (1, int)}),
    "r22_min": (r"For every relation r with at least (\d+) tuples: the selection view", {"tuples": (1, int)}),
    "r22_m": (r"m in \{([^}]+)\} complained queries, (\d+) seeds", {"m_grid": (1, ints), "m_grid_text": (1, str), "seeds": (2, int)}),
    "r22_diag": (r"under the deployed model \((\w+) forward passes per tuple\)", {"passes_word": (1, str)}),
}


def protocol_record(inp, warn):
    rel = f"{FINAL}/PROTOCOL.md"
    if not inp.exists(rel):
        warn("PROTOCOL.md not found: protocol constants are missing")
        return {}
    raw = open(inp.note(rel)).read()
    # collapse white space, keep a map from collapsed offsets to line numbers
    flat, line_of, line = [], [], 1
    prev_space = False
    for ch in raw:
        if ch.isspace():
            if not prev_space:
                flat.append(" ")
                line_of.append(line)
            prev_space = True
        else:
            flat.append(ch)
            line_of.append(line)
            prev_space = False
        if ch == "\n":
            line += 1
    flat = "".join(flat)
    out = {"_file": rel}
    for name, (pat, fields) in PROTOCOL_PATTERNS.items():
        m = re.search(pat, flat)
        if not m:
            warn(f"PROTOCOL.md: pattern of '{name}' not found (the protocol text changed?): {pat}")
            continue
        rec = {"_line": line_of[m.start()], "_match": m.group(0)[-160:]}
        for f, (g, conv) in fields.items():
            rec[f] = conv(m.group(g).strip()) if conv is not str else " ".join(m.group(g).split())
        out[name] = rec
    if "r19_e1" in out:                                     # number of behavioural calibration draws (R19 a)
        out["r19_e1"]["n_draws"] = out["r19_e1"]["draw_last"] - out["r19_e1"]["draw_first"] + 1
    a, b = out.get("default_calibration"), out.get("default_calibration_two_block")
    if a and b and (a["alpha"], a["n_cal"]) != (b["alpha"], b["n_cal"]):
        warn("PROTOCOL.md: the default (alpha, n) of Section 0 and of the two-block amendment differ")
        out["default_calibration_conflict"] = True
    srel = SAMPLER
    if inp.exists(srel) and "split" in out:
        m = re.search(r"(?m)^SPLIT_SEED\s*=\s*(\d+)", open(inp.note(srel)).read())
        out["split"]["harness_seed"] = int(m.group(1)) if m else None
        if m and int(m.group(1)) != out["split"]["seed"]:
            warn("PROTOCOL.md: the split seed differs from harness/sampler.py")
    return out


# ------------------------------------------------------------------------------------------------ paper names
def paper_names(logs, items, spec):
    """Name of every listed log as the paper prints it. Rule (manifest["logs"]["names"]): backbone letter, "-", editor
    letter, data letter (empty for the main benchmark), T; where two listed logs would share a name, all of them get
    their ridge appended. "testbed" as rule keeps the names of the testbed."""
    rule = spec.get("names", "testbed")
    out = {}
    if rule == "testbed":
        return {it["log"]: it["log"] for it in items}
    for it in items:
        r = logs[it["log"]]
        try:
            out[it["log"]] = rule["backbone"][r["backbone"]] + "-" + rule["editor"][r["editor"]] + rule["dataset"][r["dataset"]] + str(int(r["T"]))
        except (KeyError, TypeError, ValueError):
            out[it["log"]] = None                              # a field is missing: the macro stays a placeholder
    for nm in {v for v in out.values() if v and list(out.values()).count(v) > 1}:
        for k in [k for k, v in out.items() if v == nm]:
            ridge = logs[k].get("ridge")
            out[k] = None if ridge is None else nm + rule["ridge"].format(ridge)
    return out


# ------------------------------------------------------------------------------------------------ boundary rule
def boundary(logs, items, protocol):
    # [not released: the logic of the boundary check (rules (i) and (iii) of the amendment of 2026-11-10 17:40 and the legacy log);
    #  it describes a related submission under double-blind review.  The check needs metadata that the released logs do not carry
    #  (README, section "What can be reproduced"); in the authors' working tree it reported no violation (the stored facts,
    #  scripts/data/setup_facts.json: "boundary_violations": []).  Here it returns no violation and the registered limit.]
    tmax = (protocol.get("boundary") or {}).get("T_max_boundary")
    return [], tmax


# ------------------------------------------------------------------------------------------------ build
def build(root=None, manifest=None, out_path=None, verbose=False):
    """Collect and write the JSON. Returns (document, warnings). Raises OSError if the testbed is not readable."""
    root = root or find_root()
    if manifest is None:
        manifest = jload(os.path.join(HERE, "numbers_manifest.json"))
    spec = manifest["logs"]
    if not os.path.isdir(os.path.join(root, TB)):
        raise OSError(f"testbed not readable: {os.path.join(root, TB)}")
    warnings = []
    warn = warnings.append
    inp = Inputs(root)
    protocol = protocol_record(inp, warn)
    seed = (protocol.get("split") or {}).get("seed")
    if seed is None:
        warn("no split seed: pool-level numbers are missing")
    reg = registries(inp)
    pools, hn, bbs = cache_record(inp, warn)
    hn_families = {bb: b["families_with_hard_negatives"] for bb, b in bbs.items()}
    idx_cache, logs = {}, {}
    items = list(spec["items"])
    for it in [dict(x, in_paper=True) for x in items] + [dict(x, in_paper=False, in_table=False) for x in spec.get("not_in_paper", [])]:
        logs[it["log"]] = log_record(inp, it["log"], reg, seed, idx_cache, warn, hn_families)
        logs[it["log"]].update({"in_paper": it["in_paper"], "in_table": bool(it.get("in_table", True)), "role": it.get("role"),
                                "group": it.get("group"), "tag": it.get("tag"), "where": it.get("where"),
                                "why_not_in_paper": it.get("why")})
    bad, tmax = boundary(logs, items, protocol)
    names = paper_names(logs, items, spec)
    for k, r in logs.items():
        r["paper_name"] = names.get(k)
    if len({v for v in names.values() if v}) != len([v for v in names.values() if v]):
        warn("paper names of the listed logs are not unique")
    # ---- facts shared by the listed logs (asserted to be common; None if they are not)
    listed = [logs[it["log"]] for it in items]
    built = [r for r in listed if r["complete"]]

    def common(key, rows):
        vals = {json.dumps(r.get(key)) for r in rows}
        return json.loads(vals.pop()) if len(vals) == 1 else None
    com = {"batch": common("batch", built), "n_listed": len(listed), "n_built": len(built),
           "not_built": sorted(r["name"] for r in listed if not r["complete"]), "T_max": {}, "layer": {}, "definitions": common("definitions", [r for r in built if r.get("modality") == "image+text"])}
    for bb in sorted({r["backbone"] for r in listed if r.get("backbone")}):
        ts = [r["T"] for r in listed if r.get("backbone") == bb and r.get("T")]
        com["T_max"][bb] = max(ts) if ts else None
        com["layer"][bb] = common("layer", [r for r in built if r["backbone"] == bb])
    # longest log of each editor family among the ROWS of the table (T of a log that is not built: from the build plan)
    com["T_max_table"] = {}
    for ed in sorted({r["editor"] for r in listed if r.get("editor") and r["in_table"]}):
        ts = [r["T"] for r in listed if r["in_table"] and r.get("editor") == ed and r.get("T")]
        com["T_max_table"][ed] = max(ts) if ts else None
    # the ridge of the development logs and the other ridges of the working points (AlphaEdit-style rows)
    dev = sorted({r["ridge"] for r in built if r.get("role") == "dev" and r.get("ridge") is not None})
    wp = sorted({r["ridge"] for r in built if r.get("role") == "wp" and r.get("ridge") is not None}, reverse=True)
    unseen_a = sorted({r["ridge"] for r in built if r["in_table"] and r.get("role") is None and r.get("editor") == "alphaedit"})
    com["ridge"] = {"development": dev[0] if len(dev) == 1 else None, "working_points": wp,
                    "unseen_alphaedit_rows": unseen_a[0] if len(unseen_a) == 1 else None}
    if com["ridge"]["development"] is None or com["ridge"]["unseen_alphaedit_rows"] != com["ridge"]["development"]:
        warn("the development logs and the unseen AlphaEdit-style rows do not share one ridge: check the text of Sec. 5.1")
    # benchmark probes of the main benchmark and facts that the two backbones must share (None if they do not)
    main = [r for r in built if r.get("dataset") == spec.get("default_dataset") and "n_items" in r.get("probes", {})]
    # one edit stream per backbone and benchmark: every listed built log must be a prefix (same items, same order) of the
    # longest one. This is what the text means by "longer logs continue shorter ones" and "on the same edits".
    com["streams"], ok_all = {}, True
    for key in sorted({(r["backbone"], r["dataset"]) for r in built}):
        sub = sorted((r for r in built if (r["backbone"], r["dataset"]) == key), key=lambda r: -r["T"])
        ref = sub[0]["stream"]["edit_uids"]
        notp = sorted(r["name"] for r in sub if ref[:r["T"]] != r["stream"]["edit_uids"])
        com["streams"]["/".join(key)] = {"longest_log": sub[0]["name"], "n_logs": len(sub), "not_a_prefix_of_it": notp,
                                         "cache_family": sub[0]["stream"]["cache_family"]}
        ok_all = ok_all and not notp
        for nm in notp:
            warn(f"{nm}: its edits are not a prefix of the edit stream of {sub[0]['name']} (other items or another order)")
    com["one_edit_stream_per_backbone_and_benchmark"] = ok_all
    # a log whose files are complete but which its builder has not appended to a registry yet: its own load-time checks
    # may still be running (the numbers are those of the stored files either way)
    com["built_but_not_registered"] = sorted(r["name"] for r in built if r.get("registry") is None or "planned" in str(r.get("registry")))
    reg_names = set(reg)
    com["built_but_not_registered"] = [n for n in com["built_but_not_registered"] if n in reg_names]
    for nm in com["built_but_not_registered"]:
        warn(f"{nm}: its stored files are complete, but its builder has not registered it yet (load-time checks pending?)")
    com["no_edit_is_a_probe_item"] = all(r["stream"]["n_edits_that_are_probe_items"] == 0 for r in built)
    for r in logs.values():                                    # the uid lists were only needed for the comparison
        if "stream" in r:
            r["stream"]["edit_uids_sha1"] = hashlib.sha1("|".join(r["stream"].pop("edit_uids")).encode()).hexdigest()[:12]
    hn_dirty = [r["hard_negatives"]["source_related_to_an_edit_of_the_log"] for r in built if "hard_negatives" in r]
    com["hn_rows_related_to_an_edit_max"] = max(hn_dirty) if hn_dirty else None
    com["probe_items"] = common("_n_items", [dict(_n_items=r["probes"]["n_items"]) for r in main])
    com["probe_rows"] = common("_n_rows", [dict(_n_rows=r["probes"]["all"]["n_rows"]) for r in main])
    com["d_in"] = common("d_in", [bbs[bb] for bb in com["T_max"] if bb in bbs])
    shapes = []
    for bb in com["T_max"]:
        h = hn.get(bb)
        if h:
            shape = {"source_pool_items": h["source_pool_items"]}
            for typ, t in h["types"].items():
                for grp, gg in t["groups"].items():
                    shape.setdefault(grp, {})[typ] = gg["rows_per_edit"]
                    shape[grp]["edits"] = gg["edits"]
            shapes.append({"hn": shape})
    com["hn"] = common("hn", shapes) if shapes else None
    if shapes and com["hn"] is None:
        warn("hard negatives: the construction (rows per edit, source pool) differs between the backbones")
    d = com["definitions"] or {}
    m1 = re.search(r"\((\d+)-token v2 greedy\)", d.get("D1", ""))
    m2 = re.search(r"pre (\d+)-token v2 answer", d.get("D2", ""))
    dec = {"short_answer_tokens": int(m1.group(1)) if m1 else None, "long_answer_tokens": int(m2.group(1)) if m2 else None,
           "_source": "definitions of complaints_v2.json (common to the listed image+text logs)"}
    p2 = (protocol.get("d2") or {}).get("long_answer_tokens")
    if p2 and dec["long_answer_tokens"] and p2 != dec["long_answer_tokens"]:
        warn("the long-answer length of complaints_v2.json differs from PROTOCOL.md")
    # duplicate-query rule: the protocol's count against the caches
    pd = (protocol.get("duplicates") or {}).get("n_heldout_rows_with_a_repair_query")
    for bb, fams in pools.items():
        for fam, q in fams.items():
            if fam in ("Q", "L") and pd is not None and q.get("n_heldout_rows_with_a_repair_query") != pd:
                warn(f"pools {bb}/{fam}: {q.get('n_heldout_rows_with_a_repair_query')} held-out rows repeat a REPAIR query, PROTOCOL.md says {pd}")
    doc = {
        "_about": "Facts of the experimental setup, collected by scripts/collect_setup.py from stored files; do not edit. "
                  "See the docstring of that script for the inputs and for what is derived.",
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"), "root": root,
        "logs": logs, "common": com, "decoder": dec, "pools": pools, "hard_negatives": hn, "backbones": bbs,
        "protocol": protocol, "boundary_violations": bad, "warnings": warnings, "inputs": inp.seen,
    }
    out_path = out_path or os.path.join(root, OUT_REL)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + f".tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
        f.write("\n")
    os.replace(tmp, out_path)
    if verbose:
        for it in items + spec.get("not_in_paper", []):
            r = logs[it["log"]]
            if r["complete"]:
                hd = (r["probes"].get("heldout_disjoint") or {})
                print(f"  {r['name']:20s} T={r['T']:<5d} {r['editor']:9s} Rel {r['edits']['rel']['exact']!s:>5} "
                      f"D1(held-out) {hd.get('D1_rate') and round(hd['D1_rate'], 1)!s:>5}  {'table' if r['in_table'] else 'paper, no row' if r['in_paper'] else 'not in the paper'}")
            else:
                print(f"  {r['name']:20s} not complete (missing: {', '.join(r['missing_files'])})  {'table' if r['in_table'] else 'paper, no row' if r['in_paper'] else 'not in the paper'}")
    return doc, warnings


def stale(root=None, path=None):
    """Reasons why the stored collection is out of date (inputs changed, appeared or vanished); [] if it is current,
    None if the inputs cannot be read here."""
    root = root or find_root()
    path = path or os.path.join(root, OUT_REL)
    if not os.path.exists(path):
        return ["the collection does not exist"]
    if not os.path.isdir(os.path.join(root, TB)):
        return None
    doc = jload(path)
    out = []
    for rel, was in (doc.get("inputs") or {}).items():
        p = os.path.join(root, rel)
        now = None
        if os.path.exists(p):
            st = os.stat(p)
            now = {"mtime": int(st.st_mtime), "size": st.st_size}
        if now != was:
            out.append(f"{rel}: " + ("appeared" if was is None else "vanished" if now is None else "changed"))
    return out


if __name__ == "__main__":
    doc, warnings = build(verbose="--print" in sys.argv[1:])
    print(f"wrote {OUT_REL}: {doc['common']['n_built']} of {doc['common']['n_listed']} listed logs complete; "
          f"{len(doc['protocol']) - 1} protocol records; {len(warnings)} warnings; {len(doc['boundary_violations'])} boundary violations")
    for w in warnings:
        print("  warning:", w)
    for b in doc["boundary_violations"]:
        print("  BOUNDARY:", b)
