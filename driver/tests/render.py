"""Render tests/out/table.txt and tests/out/summary.json from the JSON files written by t_cpu.py and t_gpu.py (no number typed by hand).

  python render.py
"""
from __future__ import annotations

import glob
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import fx  # noqa: E402

OUT = os.path.join(HERE, "out")
SHORT = {"TB-Q100-ae-l2_3": "Q", "TB-L100-ae-l2_3": "L"}


def load(pattern):
    return {os.path.basename(p)[:-5]: fx.io.jload(p) for p in sorted(glob.glob(os.path.join(OUT, pattern))) if not p.endswith("_smoke.json")}


def verdict(r):
    return "PASS" if r.get("ok") else "FAIL"


COST = {"scoped": "forward_gate_role_last", "deployed": "forward_deployed_weight", "weight": "forward_deployed_weight", "gate": "forward_gate_role_all",
        "correction": "forward_gate_role_all", "dense": "forward_gate_role_all"}


def project(e, d) -> dict:
    """Time of the E1 / E3 row sets on the log of mechanics record e (per-item costs and sequence counts measured there) for one
    calibration draw with a cold memo: `upper` = one forward per sequence for every row; `scaled` = the forwards per sequence that the
    end-to-end test d measured for that row on the development log (record d).  The z-score gate also needs one truncated forward per
    sequence once per log (cost: the development log's seconds per capture relative to its seconds per full forward).  DUAL = two
    decodes per item on 300 edits (Rel, T-Gen, I-Gen) and the held-out probes.  preedit costs nothing; the P12 rows are left out."""
    import fx
    c, ns, m, n_tf = e["seconds_per_item"], e["n_sequences"], d["methods"], d["n_tf"]
    n_all = sum(ns.values())
    n_dual = 2 * (3 * min(300, e["T"]) + ns["probe_heldout_disjoint"])
    dual_s = n_dual * c["decode_deployed_weight"]
    z = m.get("zscore") or {}
    fwd_dev = m["deployed"]["seconds_tf"] / n_tf                         # seconds per full forward on the development log in test d
    # the z-score row of test d spent its teacher-forced time on `captures` truncated forwards (the cached largest z-score per sequence) and
    # on `forwards` full ones: seconds per capture there, relative to a full forward there, times the full forward measured on this log
    cap = max(0.0, (z["seconds_tf"] - z["forwards"] * fwd_dev) / z["captures"]) / fwd_dev * c["forward_deployed_weight"] if z.get("captures") else 0.0
    out = {"log": e["log"], "T": e["T"], "scaled_from": d["log"], "n_all": n_all, "n_sequences": ns, "seconds_per_item": c, "n_dual_decodes": n_dual,
           "dual_minutes": dual_s / 60, "pass_minutes": n_all * c["forward_gate_role_last"] / 60, "seconds_per_capture": cap, "rows": {}}
    for lab, r in m.items():
        if lab.startswith("p12_") or r["kind"] in ("preedit", "dual"):
            continue
        per = c[COST[r["kind"]]]
        f = (r["forwards"] or r["lib"]) / n_tf                          # forwards per sequence on the development log (library path: every sequence)
        extra = n_all * cap if r.get("captures") else 0.0
        out["rows"][lab] = {"path": r["kind"], "forwards_per_sequence_dev": f, "upper_s": n_all * per + extra, "scaled_s": f * n_all * per + extra}
    for name, labels in (("E1", [x for x in fx.methods.E1_LABELS if not x.startswith("p12_")]), ("E3", list(fx.methods.E3_LABELS))):
        rows = [out["rows"][x] for x in labels if x in out["rows"]]
        dual = dual_s if "dual" in labels else 0.0
        memo = [out["rows"][x] for x in labels if x in out["rows"] and out["rows"][x]["path"] == "scoped"]
        out[name] = {"labels": labels, "n_rows": len(labels), "upper_hours": (sum(x["upper_s"] for x in rows) + dual) / 3600,
                     "scaled_hours": (sum(x["scaled_s"] for x in rows) + dual) / 3600, "passes_dev": sum(x["forwards_per_sequence_dev"] for x in rows),
                     "n_memo_rows": len(memo), "passes_memo_rows_dev": sum(x["forwards_per_sequence_dev"] for x in memo), "dual_included": bool(dual)}
    return out


def main():
    cpu, gpu = load("cpu_*.json"), load("gpu_*.json")
    code = fx.code_fingerprint()
    rows, summ = [], {"code": code, "rendered": time.strftime("%Y-%m-%d %H:%M:%S"), "tests": {}}

    def add(test, scope, r, detail):
        stale = "" if r.get("code") == code else " (older code)"
        rows.append([test, scope, verdict(r) + stale, detail])
        summ["tests"][f"{test} {scope}".strip()] = {"ok": bool(r.get("ok")), "detail": detail, "code_current": r.get("code") == code}

    r = cpu.get("cpu_registry")
    if r:
        vp = r.get("values_prefix") or {}
        add("registry", f"{r['n_registered']} logs", r, f"{r['n_existing']} of {r['n_registered']} registered logs exist and open (probe rows == key cache, edits -> cache rows); missing: "
            f"{r['missing'] or 'none'}; released values vs rescope.rebase: max relative deviation {vp.get('max_rel_deviation_from_library_rebase', float('nan')):.1e}, "
            f"bit for bit equal on {vp.get('n_bitwise_equal')}/{vp.get('n_prefix_pairs')} common prefixes of same-family logs; edits of the long Qwen stream that ask a probe "
            f"question: {(r.get('protocol_677') or {}).get('n_edits_asking_a_probe_question_TB-Q2000-ae-l2_3')} (protocol: 677); pre-edit fields of the probe records identical across "
            f"the logs of a family: {sum(v['identical'] for v in (r.get('pre_edit_probe_records') or {}).values())}/{len(r.get('pre_edit_probe_records') or {})} families"
            + (f"; roles and flags as registered (never reported: the {len(r['roles']['never_reported'])} internal logs; cut by R19: {len(r['roles']['cut_by_R19'])} logs): "
               f"{r['roles']['ok']}; a log registered under another directory opens and gives the same factors, thresholds and values bit for bit: {r['register']['ok']}"
               if r.get("register") and r.get("roles") else ""))
    r = cpu.get("cpu_draws")
    if r:
        d = "; ".join(f"{SHORT[n]}: n = 200 draw == library draw {v['default_equals_library']}/20, nested {v['nested']}, P12 records with identical complaint sets "
                      f"{v['p12']['n_equal']}/{v['p12']['n_records']} (seeds {v['p12']['seeds']})" for n, v in r["logs"].items())
        add("draws", "dev logs", r, d)
    r = cpu.get("cpu_gate")
    if r:
        d = "; ".join(f"{SHORT[n]}: {v['n_differing']} of {v['n_compared']} values differ from KeyCache.decide (9 gates x 4 positions), fx {v['seconds_fx']:.0f}s vs library "
                      f"{v['seconds_library']:.0f}s" for n, v in r["logs"].items())
        add("gate == library", "dev logs", r, d)
    r = cpu.get("cpu_regress_gate")
    if r:
        d = "; ".join(f"{SHORT[n]}: thresholds bit for bit {v['n_tau_bitwise']}/{v['n_gates']} gates, {v['n_differing']} of {v['n_decisions']} decisions differ (20 draws, capped + "
                      f"uncapped, tf / prompt / decode), table means max diff {max(x['max_abs_diff'] for x in v['table_means'].values()) if v['table_means'] else float('nan'):.1e}"
                      for n, v in r["logs"].items() if v.get("available"))
        add("(a) sanity M1-M6, gate level", "dev logs", r, d)
    r = cpu.get("cpu_frontier")
    if r:
        add("frontier, matched coverage (E1b, E1c)", "dev logs", r, "; ".join(
            f"{SHORT[n]}: fixed gates == points of the sweep {all(v[k] for k in v if k.startswith('readout_'))}, grid points differing from the read-out "
            f"{v['ncoord']['grid_points_differing_from_readout'] + v['cos']['grid_points_differing_from_readout']}, matched-coverage thresholds consistent and largest "
            f"{v['ncoord']['matched_coverage_consistent'] and v['cos']['matched_coverage_consistent'] and v['ncoord']['threshold_is_largest'] and v['cos']['threshold_is_largest']}, "
            f"{v['ncoord']['seconds']:.1f}s per feature (501 thresholds, {v['ncoord']['n_curves']} curves)" for n, v in r["logs"].items()))
    for nm, lab in (("cpu_stats", "statistics"), ("cpu_io_memo", "io + memo (6 concurrent writers)"), ("cpu_methods", "constructors + e1_rowset + wrap + recalibrate")):
        r = cpu.get(nm)
        if r:
            if nm == "cpu_io_memo":
                m = r["memo"]
                d = (f"6 processes writing at the same time for {m['seconds_all_writing_together']:.2f}s ({m['n_duplicate_writes']} duplicate writes of the same key) + a reader "
                     f"({m['concurrent_reader']['refreshes']} refreshes): {m['n_keys_read']}/{m['n_expected_keys']} keys read back, conflicts {m['conflicts_between_writers']}, conflict detection {m['conflict_detected']}, other library "
                     f"version ignored {m['other_library_version_ignored']} (admitted when named in FX_MEMO_ACCEPT: {m.get('accept_list_admits_named_version')})")
            elif nm == "cpu_stats":
                d = f"bootstrap CI == direct resampling {r['bootstrap']['ok']}, paired {r['paired']['ok']}, McNemar == scipy {r['mcnemar']['n_equal_scipy']}/{r['mcnemar']['n_cases']}"
            else:
                d = "; ".join(f"{SHORT[n]}: built {v['built']}/{v['n_variants']} variants, keys distinct {v['keys_distinct']}, only calibrated rows change with (alpha, draw) "
                              f"{v['e1_only_calibrated_rows_change']}, ReScope-N == independent assembly bit for bit {v['rescope_null']['thresholds_bitwise_equal_independent_assembly']} "
                              f"(null sample: 100 distinct queries, none a calibration query, in {v['rescope_null']['by_query']['n_draws']} draws: {v['rescope_null']['by_query']['ok']}; "
                              f"calibration rows of draw 0 that the row-wise rule shared with it: {v['rescope_null']['by_query']['old_rule_shared_at_draw_0']}), "
                              f"C0 / S / C1 disjoint {v['partition']['disjoint']}" for n, v in r["logs"].items())
            add(lab, "", r, d)
    r = cpu.get("cpu_scale")
    if r and r.get("available"):
        pm = r["per_method"]
        add("gate level at scale", r["log"], r, f"T = {r['T']}, {sum(r['n_rows'].values())} sequences x 2 positions, first draw / 19 further draws: "
            + ", ".join(f"{k} {v['first_draw_s']:.0f}s / {v['next_19_draws_s']:.0f}s" for k, v in pm.items()) + f"; peak RAM {r['peak_rss_gb']:.1f} GB; no metric computed")
    for name, s in SHORT.items():
        r = gpu.get(f"gpu_{name}_a_regress")
        if r and r.get("available"):
            bm = r["by_method"]
            d = (f"{r['n_rows']} rows x 12 fields (deployed, pre-edit, M5, M6; M1-M4 draws 0-2): {r['n_rows_differing']} rows differ; explained by the GPU-built gate "
                 f"{r['n_explained']}, unexplained {r['n_unexplained']}; per method " + ", ".join(f"{k} {v['n_differing']}" for k, v in bm.items())
                 + f"; {r['forwards']} fresh forwards (private empty memo), hook mismatches {r['hook_mismatch']}"
                 + (f"; the same through the shared memo: {r['shared_memo']['n_rows_differing']} rows differ ({r['shared_memo']['forwards']} forwards)" if r.get("shared_memo") else ""))
            if r.get("row_sets"):
                d += f"; fx row sets == the library Evaluator's item sets: {sum(v['same_set'] for v in r['row_sets'].values())}/{len(r['row_sets'])} (edits, hard negatives, 5 probe pools, D2 rows)"
            nb = r.get("numbers") or {}
            if nb.get("available"):
                d += f"; the sanity agent's rendered numbers (results.json) vs fx.metrics: {nb['n_differing']} of {nb['n_compared']} differ (max {nb['max_abs_diff']:.1e})"
            add("(a) sanity rows, behaviour", s, r, d)
        r = gpu.get(f"gpu_{name}_b_memo")
        if r:
            sp = r["second_pass"]
            d = (f"memo == fresh forward on {r['n_equal']}/{r['n_checked']} (sequence, firing edit) pairs, deployed {r['deployed']['n_equal']}/{r['deployed']['n']}, decodes "
                 f"{r['decode']['n_equal']}/{r['decode']['n']} (+ deployed {r['decode_deployed']['n_equal']}/12), DUAL == library DUAL {r.get('dual', {}).get('n_equal')}/{r.get('dual', {}).get('n')}, hard-negative decodes == library {r.get('hn_decode_all', {}).get('n_equal_library')}/{r.get('hn_decode_all', {}).get('n')}; second pass of {sp['n_rows']} rows: {sp['forwards']} forwards, "
                 f"{sp['decodes']} decodes, rows equal {sp['rows_equal']}, {sp['seconds']:.0f}s")
            fo = r.get("forced")
            if fo:
                d += (f"; forced misprediction (falsified decision): counted {fo['tf']['counted']}/{fo['tf']['n']} forwards + {fo['decode']['counted']}/{fo['decode']['n']} decodes, "
                      f"rows == true rows {fo['tf']['rows_equal_true_rows']}/{fo['tf']['n']} + {fo['decode']['rows_equal_true_rows']}/{fo['decode']['n']}, stored under the falsified "
                      f"signature {fo['tf']['stored_under_falsified_signature'] + fo['decode']['stored_under_falsified_signature']}")
            add("(b) memo", s, r, d)
        r = gpu.get(f"gpu_{name}_c_hook")
        if r:
            d = "; ".join(f"{m} {lp}: {v[lp]['n']} sequences, fired in the hook {v[lp]['n_fired_hook']}, fx.gate (CPU) differs on {v[lp]['cpu_vs_hook']}, fx.behave decision on "
                          f"{v[lp]['device_vs_hook']}, memo rows on {v[lp]['rows_vs_hook']} (near {v[lp]['n_near']})" for m, v in r["methods"].items() for lp in ("tf", "prompt")
                          if isinstance(v.get(lp), dict) and v[lp].get("n"))
            if r.get("dense"):
                d += "; dense fast paths vs full pass: " + ", ".join(f"{k} {v['rows_vs_hook']} of {v['n']} rows differ ({v['sources']['forwards']} forwards instead of {v['n']})"
                                                                     for k, v in r["dense"].items())
            add("(c) fx.gate == hook", s, r, d)
        r = gpu.get(f"gpu_{name}_d_e1")
        if r:
            d = (f"{len(r['methods'])} methods end to end ({sum(r['n_sequences'][k] for k in ('edits', 'probes', 'hn'))} sequences each + decode subsample), {r['n_records']} records, "
                 f"{r['seconds']:.0f}s; unavailable {list(r['unavailable']) or 'none'}")
            add("(d) e1_rowset", s, r, d)
        r = gpu.get(f"gpu_{name}_f_where")
        if r:
            w10, w50 = (r[f"where_least_n{n}"] for n in r.get("ns", (10, 50)))
            d = (f"where_least n = {' / '.join(str(n) for n in r.get('ns', (10, 50)))}: refused {w10['n_refused']} / {w50['n_refused']}, complaints still firing {w10['complaints_firing_after']} / "
                 f"{w50['complaints_firing_after']}, order independent {w10['order_independent']} / {w50['order_independent']}, build {w10['seconds_build']:.0f} / "
                 f"{w50['seconds_build']:.0f}s; where_logonly: {r['where_logonly']['capped_edits']} capped edits; evidence curve n = 0 / 10 / 39 / 40: "
                 + " / ".join(r[f"evidence_n{n}"]["rule"] for n in (0, 10, 39, 40)))
            add("(f) where_least, evidence", s, r, d)
        r = gpu.get(f"gpu_{name}_g_ablations")
        if r:
            v = r["variants"]
            add("(g) ablation switches", s, r, f"{len(v)} variants build and run on {r['n_sequences']} sequences; rows differing from the plain Evaluator with the same gate: "
                f"{sum(x['rows_differing'] for x in v.values())}; via the memo path {sum(bool(x['memo_path']) for x in v.values())}, dense {sum(x['kind'] == 'dense' for x in v.values())}, "
                f"library path {sum(x['kind'] == 'gate' or (x['kind'] == 'scoped' and not x['memo_path']) for x in v.values())}; calibration on captured keys (roles last+answer / all): "
                + ", ".join(f"{k} {x['seconds_build']:.0f}s" for k, x in v.items() if "role=" in k))
    for k, r in gpu.items():
        if k.endswith("_e_mechanics"):
            t = r["seconds"]
            pi = r.get("seconds_per_item") or {}
            d = (f"T = {r['T']} ({r['role']}): open {t['open']:.1f}s, build rescope {t['build_rescope']:.1f}s, gate-level pass over {sum(r['n_gate_rows'].values())} sequences "
                 f"{t['gate_level_pass']:.1f}s (next draw {t['second_draw_build_and_gate_pass']:.1f}s), model load {t['load_model']:.0f}s (lean {r['lean']}), 50 behavioural sequences "
                 f"(the last 50 edits' own prompts, empty private memo) {t['behaviour_50_sequences']:.1f}s = {r['forwards']} forwards, second pass {t['behaviour_50_sequences_second_pass']:.2f}s "
                 f"from the memo with equal rows {r['second_pass_equal']}"
                 + (f"; per forward {pi['forward_gate_role_last']:.2f}s, per decode {pi['decode_deployed_weight']:.2f}s" if pi else "")
                 + f"; peak RAM {r['peak_rss_gb']:.1f} GB, peak GPU {r['peak_gpu_gb']:.2f} GB; no metric computed")
            add("(e) mechanics", r["log"], r, d)
    L = [f"TESTS OF THE fx DRIVER   (rendered {summ['rendered']} by tests/render.py from tests/out/*.json; code {code})", "",
         fx.io.text_table(["test", "scope", "result", "detail"], rows, align=["l", "l", "l", "l"]), ""]
    # timings per method (test d: cold private memo store)
    for name, s in SHORT.items():
        r = gpu.get(f"gpu_{name}_d_e1")
        if not r:
            continue
        tr = []
        for lab, m in r["methods"].items():
            path = "memo" if m.get("memo_path") else ("weights" if m["kind"] in ("deployed", "preedit", "weight") else ("dense" if m["kind"] == "dense" else
                                                                                                                         ("generation" if m["kind"] == "dual" else "library")))
            nd = m.get("next_draw")
            tr.append([lab, path, f"{m['seconds_build']:.1f}", "-" if "seconds_tf" not in m else f"{m['seconds_tf']:.0f}",
                       str(m["lib"] if path == "library" else m["forwards"]) + (f" + {m['captures']} trunc." if m.get("captures") else ""), f"{m['pre']}",
                       "-" if nd is None else f"{m['seconds_tf_next_draw']:.0f} ({nd['forwards']} forwards)",
                       "-" if "seconds_decode" not in m else f"{m['seconds_decode']:.0f} ({m['decodes_run']} decodes for {m['n_decode_items']} items)",
                       "-" if "seconds_gate_level" not in m else f"{m['seconds_gate_level']:.1f}"])
        L += [f"TIMINGS, test (d) on {name}: the E1 row set for ONE calibration draw with an empty memo store; {r['n_tf']} teacher-forced sequences per method "
              f"({r['n_sequences']['edits']} edit sequences, {r['n_sequences']['probes']} HELD-OUT-DISJOINT probes, {r['n_sequences']['hn']} hard negatives = every 4th); "
              f"total {r['seconds']:.0f}s on a shared GPU",
              fx.io.text_table(["method", "path", "build s", "teacher-forced s", "forwards run", "rows from pre-edit cache", "next draw s", "decode metrics s (subsample)",
                                "gate level s"], tr), ""]
        summ.setdefault("timings", {})[name] = r["methods"]
    # projection for the long logs: per-item costs measured there (test e) x the sequences of the log x, per row of the row set, either one
    # forward per sequence (a certain upper bound) or the forwards per sequence that test (d) measured on the development log of the backbone
    for k, r in sorted(gpu.items()):
        if not (k.endswith("_e_mechanics") and r.get("seconds_per_item")):
            continue
        dev = next((n for n in SHORT if n[3] == r["log"][3] and gpu.get(f"gpu_{n}_d_e1")), None)
        if dev is None:
            continue
        d = gpu[f"gpu_{dev}_d_e1"]
        pj = project(r, d)
        summ.setdefault("projection", {})[r["log"]] = pj
        e1, e3, c = pj["E1"], pj["E3"], pj["seconds_per_item"]
        L += [f"PROJECTION for {r['log']} (T = {r['T']}), one calibration draw, cold memo.  Measured on this log (test e; outcomes discarded): {pj['n_all']} teacher-forced "
              f"sequences per row ({pj['n_sequences']['edit']} edit sequences, {pj['n_sequences']['probe_heldout_disjoint']} HELD-OUT-DISJOINT probes, {pj['n_sequences']['hn_valid']} hard "
              f"negatives); per forward {c['forward_deployed_weight']:.3f}s (deployed weight), {c['forward_gate_role_last']:.3f}s (gate at the last prompt position), "
              f"{c['forward_gate_role_all']:.3f}s (gate at every position); per decode {c['decode_deployed_weight']:.2f}s; one pass over all sequences = {pj['pass_minutes']:.0f} min.",
              f"  E1 row set without the P12 rows ({e1['n_rows']} rows; DUAL on 300 edits + the held-out probes = {pj['n_dual_decodes']} decodes = {pj['dual_minutes']:.0f} min, included): "
              f"at most {e1['upper_hours']:.1f} h (every row a forward for every sequence); {e1['scaled_hours']:.1f} h with the forwards per sequence that test (d) measured on "
              f"{dev} ({e1['passes_dev']:.1f} passes in all: the {e1['n_memo_rows']} memo rows release the same values and shared {e1['passes_memo_rows_dev']:.2f} of a pass there).",
              f"  E3 row set ({', '.join(pj['E3']['labels'])}): at most {e3['upper_hours']:.1f} h; {e3['scaled_hours']:.1f} h scaled from {dev} ({e3['passes_dev']:.1f} passes).",
              f"  The scaled figure assumes the development log's forwards per sequence; on this log {100 * pj['n_sequences']['edit'] / pj['n_all']:.0f} % of the sequences are edit "
              f"sequences ({100 * d['n_sequences']['edits'] / d['n_tf']:.0f} % in test d), which fire more often than probes: expect a cost between the two figures.  A further "
              "calibration draw costs only the calibrated rows' newly firing (sequence, edit) pairs; rows with fixed thresholds are evaluated once.", ""]
    summ["all_pass"] = all(v["ok"] for v in summ["tests"].values())
    txt = "\n".join(L)
    with open(os.path.join(OUT, "table.txt"), "w") as f:
        f.write(txt + "\n")
    fx.io.jdump(summ, os.path.join(OUT, "summary.json"), indent=1)
    print(txt)


if __name__ == "__main__":
    main()
