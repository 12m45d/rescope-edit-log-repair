"""E8: render everything that exists.  Reads out/latency/, out/cpu/, the linear cost records (read-only) and the NeuralDB record;
writes out/results.json (fx.io schema), out/table.txt, out/paper_numbers.json + paper_numbers_doc.json, out/scale_series.json.
    python render.py"""
import os as _go, sys as _gs  # ARTIFACT: a failed run leaves the stored output files untouched (rescope_paths.guard_outputs)
_gd = _go.environ.get("RESCOPE_HOME") or _go.path.dirname(_go.path.abspath(__file__))
while not _go.path.exists(_go.path.join(_gd, "rescope_paths.py")) and _go.path.dirname(_gd) != _gd: _gd = _go.path.dirname(_gd)
_gs.path.insert(0, _gd); import rescope_paths as _RPG; _RPG.guard_outputs(__file__, __name__, needs=("linear_baselines/streams/out/TB-Q100-ae-l2_3", "linear_baselines/streams/out/TB-L100-ae-l2_3"))
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP
import glob
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from synth import *          # noqa: F401,F403,E402

R = fx.io.Results("E8", OUT)
PN, DOC, TXT, MISSING = {}, {}, [], []
tt = fx.io.text_table
EXP = FINAL + "/.."
P12 = EXP + "/08_rescope/p12_linear_streams/out"
SMOKE = EXP + "/07_repair_v3/harness/out/smoke"
NDB = f"{_RP.ROOT}/03_analysis/scope_lit/D_unlearning_editing_scale"
BB = ("qwen25vl", "llava")
SYN_T = (1000, 10000, 100000)


def pn(key, value, doc):
    PN[key] = value
    DOC[key] = doc


def f(x, nd=2):
    return "-" if x is None else (x if isinstance(x, str) else f"{x:.{nd}f}")


def mpp(d, nd=2, scale=1.0):
    return "-" if d is None else f"{d['median'] * scale:.{nd}f} [{d['p10'] * scale:.{nd}f}, {d['p90'] * scale:.{nd}f}]"


def load(p):
    return jload(p) if os.path.exists(p) else None


PRIMARY = {}                                                            # (backbone, gate set) -> directory of the primary latency pass


def lat_file(bb, name):
    return f"{PRIMARY.get((bb, name), OUT + '/latency')}/{bb}__{name}.json"


def min_ratio(r, mode):
    """Ratio of the sums over sequences of the per-sequence minimum over repetitions, gated / plain (the library's estimator), minus 1, percent."""
    T = np.array(r["raw_seconds"])
    best = T.min(0)
    return float(best[r["mode_names"].index(mode)].sum() / best[0].sum() - 1) * 100


def by_type(r, mode):
    """Median paired overhead (percent) on the 20 image sequences (10 edit sequences + 10 image probes) and on the 10 text probes."""
    T = np.array(r["raw_seconds"])
    rat = (T[:, r["mode_names"].index(mode)] / T[:, 0] - 1) * 100
    return float(np.median(rat[:, :20])), float(np.median(rat[:, 20:]))


def latency():
    rows, rows2, sel = [], [], []
    passes = [(1, f"{OUT}/latency")] + [(k, f"{OUT}/latency_pass{k}") for k in (2, 3) if os.path.isdir(f"{OUT}/latency_pass{k}")]
    for bb in BB:
        for name in REAL[bb] + [f"synthetic-T{T}" for T in (1000, 10000)]:
            util = {}
            for k, d in passes:
                x = load(f"{d}/{bb}__{name}.json")
                if x is not None:
                    util[k] = x["load_utilization_pct"]["median"]
            if util:
                best = min(util, key=lambda k: (util[k], k))
                PRIMARY[(bb, name)] = dict(passes)[best]
                if len(util) > 1:
                    sel.append(f"{name}: pass {best} (median card utilisation " + ", ".join(f"pass {k}: {v:.0f} %" for k, v in sorted(util.items())) + ")")
        for k, d in passes:
            for name in REAL[bb] + [f"synthetic-T{T}" for T in (1000, 10000)]:
                prim = PRIMARY.get((bb, name)) == d
                r = load(f"{d}/{bb}__{name}.json")
                if r is None:
                    if k == 1:
                        MISSING.append(f"latency {bb} {name}")
                    continue
                T = r["T"]
                for mode, m in r["modes"].items():
                    if mode == "plain_W0":
                        continue
                    mr = min_ratio(r, mode)
                    kw = dict(pool=f"30 sequences x 9 repetitions, pass {k}", n_units=m["overhead_end_to_end_pct"]["n"], T=T, backbone=bb, kind=r["kind"], latency_pass=k, primary_pass=prim)
                    lab = name if r["kind"] == "real" else f"{bb}:{name}"
                    for metric, dd in (("ms_per_forward", m["ms_per_forward"]), ("overhead_end_to_end_pct", m["overhead_end_to_end_pct"]), ("gate_only_ms", m["gate_only_ms"])):
                        R.add(lab, mode, metric + "@median", dd["median"], ci_lo=dd["p10"], ci_hi=dd["p90"], **kw)
                    R.add(lab, mode, "overhead_end_to_end_pct_min_based", mr, **kw)
                    R.add(lab, mode, "gate_only_ms_mean_of_min", m["gate_only_ms_mean_of_min"], **kw)
                    rows.append([bb, k if not prim else f"{k}*", name, T, mode, mpp(r["modes"]["plain_W0"]["ms_per_forward"], 1), mpp(m["ms_per_forward"], 1), mpp(m["overhead_end_to_end_pct"], 1), "%.1f / %.1f" % by_type(r, mode), f(mr, 2),
                                 mpp(m["gate_only_ms"], 3), f(m["gate_only_ms_mean_of_min"], 3), f(m["gate_only_pct_of_plain_forward"], 2), m["n_sequences_firing"],
                                 f"{r['load_utilization_pct']['median']:.0f} [{r['load_utilization_pct']['p10']:.0f}, {r['load_utilization_pct']['p90']:.0f}]",
                                 f"{r['load_other_processes_median']:.0f} / {r['load_other_processes_max']}", r.get("n_edits_capped")])
                if r["kind"] == "real" and prim:
                    m = r["modes"]["last_fp64"]
                    pre = name
                    ld = f"pass {k}; card utilisation median {r['load_utilization_pct']['median']:.0f} %, up to {r['load_other_processes_max']} other process(es) on the card"
                    pn(f"{pre}.gate_ms_per_forward", m["gate_only_ms"]["median"], f"E8, {name}: the gate alone (decision + response, fp64, role last) on captured module inputs, median ms per forward pass over 30 sequences x 50 runs (GPU; {ld})")
                    pn(f"{pre}.gate_ms_per_forward_p10", m["gate_only_ms"]["p10"], "same: 10th percentile")
                    pn(f"{pre}.gate_ms_per_forward_p90", m["gate_only_ms"]["p90"], "same: 90th percentile")
                    pn(f"{pre}.gate_ms_per_forward_min_based", m["gate_only_ms_mean_of_min"], "same: mean over the 30 sequences of the per-sequence minimum (nearly free of contention by other processes on the card)")
                    pn(f"{pre}.gate_pct_of_forward_min_based", m["gate_only_pct_of_plain_forward"], "gate alone (per-sequence minima) as a percentage of the plain forward (per-sequence minima)")
                    pn(f"{pre}.overhead_end_to_end_pct", m["overhead_end_to_end_pct"]["median"], f"E8, {name}: forward with the gate (fp64, role last) against the plain pre-edit forward of the same repetition and sequence, median of the paired ratios minus 1, percent (30 sequences x 9 repetitions; shared card: {ld})")
                    pn(f"{pre}.overhead_end_to_end_pct_p10", m["overhead_end_to_end_pct"]["p10"], "same: 10th percentile")
                    pn(f"{pre}.overhead_end_to_end_pct_p90", m["overhead_end_to_end_pct"]["p90"], "same: 90th percentile")
                    pn(f"{pre}.overhead_end_to_end_pct_min_based", min_ratio(r, "last_fp64"), "same forwards: ratio of the sums of the per-sequence minima over repetitions, gated / plain, minus 1, percent (the library's estimator; robust to other jobs on the card)")
                    pn(f"{pre}.overhead_end_to_end_pct_image_sequences", by_type(r, "last_fp64")[0], "same forwards, the 20 image sequences only: median paired overhead, percent")
                    pn(f"{pre}.overhead_end_to_end_pct_text_sequences", by_type(r, "last_fp64")[1], "same forwards, the 10 text probes only: median paired overhead, percent")
                    pn(f"{pre}.overhead_end_to_end_pct_module_dtype", r["modes"]["last_module"]["overhead_end_to_end_pct"]["median"], "median paired overhead with the gate in the module's dtype")
                    pn(f"{pre}.overhead_end_to_end_pct_all_positions", r["modes"]["all_fp64"]["overhead_end_to_end_pct"]["median"], "median paired overhead with role all positions (fire all, linear response; the calibrated thresholds), fp64")
                    pn(f"{pre}.overhead_end_to_end_pct_all_positions_min_based", min_ratio(r, "all_fp64"), "same, min-based estimator")
                    pn(f"{pre}.gate_ms_per_forward_all_positions", r["modes"]["all_fp64"]["gate_only_ms"]["median"], "gate alone, role all positions, fp64, median ms")
                    pn(f"{pre}.plain_forward_ms", r["modes"]["plain_W0"]["ms_per_forward"]["median"], "plain pre-edit forward, median ms (B = 1, 30 sequences)")
                    pn(f"{pre}.card_utilisation_pct_median", r["load_utilization_pct"]["median"], "utilisation of the card during the measurement (samples before / after every repetition), median percent")
                    pn(f"{pre}.other_processes_on_card_max", r["load_other_processes_max"], "largest number of other compute processes on the card during the measurement")
    TXT.append("E8 (1)  GATE LATENCY PER FORWARD PASS ON THE GPU (B = 1; 30 sequences: 10 edit sequences, 10 image probes, 10 text probes; 1 warm-up + 9 repetitions, modes rotating within\n"
               "  every repetition and sequence; shared card through the queue, no card reserved).  median [10th, 90th percentile] over sequences x repetitions (the spread of plain / gated ms is\n"
               "  mostly the sequence length: text probes about 40 tokens, image sequences about 1000).  overhead % = paired ratio to the plain forward of the same (repetition, sequence) minus 1;\n"
               "  img / text % = its median on the 20 image sequences / on the 10 text probes; min % = ratio of the sums of the per-sequence minima over repetitions minus 1 (robust to other jobs); gate alone = decision + response on the captured module\n"
               "  input (50 runs per sequence for role last, 10 for all positions; min = mean of per-sequence minima; % of fwd = min-based gate alone / min-based plain forward).\n"
               "  Modes: last_fp64 = role last, top-1, REPL, fp64 (as evaluated); last_module = module dtype; all_fp64 = all positions, fire all, linear response, the same calibrated thresholds;\n"
               "  all_nothr_fp64 = all thresholds -inf (every edit released at every position); where_logonly = ReScope-W.  Real logs: detectors, thresholds (two-block, alpha 0.05, n 200,\n"
               "  draw 0) and values of the log; synthetic logs: synth.py.  util %: card utilisation; others: other compute processes on the card (median / max).  pass: * = primary pass\n"
               "  (per gate set the pass with the lower median card utilisation; rule fixed before a second pass was run); gate sets with two passes: " + ("; ".join(sel) if sel else "none") + "\n"
               + tt(["backbone", "pass", "gate set", "T", "mode", "plain ms", "gated ms", "overhead %", "img / text %", "min %", "gate alone ms", "min", "% of fwd", "seq firing", "util %", "others", "capped"], rows))
    for bb in BB:
        for T in SYN_T:
            r = load(f"{OUT}/latency/{bb}__synthetic-T{T}__gate_only_gpu.json")
            if r is None:
                MISSING.append(f"gate-only GPU synthetic {bb} T={T}")
                continue
            R.add(f"{bb}:synthetic-T{T}", "last_fp64", "gate_only_ms_one_key@median", r["gate_only_ms"]["median"], ci_lo=r["gate_only_ms"]["p10"], ci_hi=r["gate_only_ms"]["p90"],
                  pool="30 keys x 50 runs", n_units=r["gate_only_ms"]["n"], T=T, backbone=bb, kind="synthetic")
            R.add(f"{bb}:synthetic-T{T}", "last_fp64", "gate_only_ms_one_key_mean_of_min", r["gate_only_ms_mean_of_min"], pool="30 keys x 50 runs", n_units=30, T=T, backbone=bb, kind="synthetic")
            rows2.append([bb, T, mpp(r["gate_only_ms"], 3), f(r["gate_only_ms_mean_of_min"], 3), r["n_keys_firing"], f(r["peak_gpu_bytes"] / 2 ** 20, 0),
                          " / ".join(f"{x.get('utilization_pct', float('nan')):.0f}" for x in r["load_samples"]), " / ".join(str(x.get("n_other_processes")) for x in r["load_samples"]),
                          r["calibration"]["n_edits_capped"]])
    if rows2:
        TXT.append("E8 (1b) GATE ALONE ON THE GPU FOR SYNTHETIC LOGS (no model in the process; fp64; one eligible key per sequence; 30 keys x 50 runs; median [p10, p90] ms over all runs;\n"
                   "  min = mean over keys of the per-key minimum, the figure that is nearly free of contention on the shared card and the one used in the scale table).\n"
                   "  All six rows come from ONE job (14.5 GB reservation).  Earlier partial runs of the same code are kept and not used: out/latency/superseded_10runs/ (10 runs per key:\n"
                   "  minima still inflated by the load) and out/latency/superseded_small_reservation/ (T <= 10000 only, before the 14.5 GB reservation was granted)\n"
                   + tt(["backbone", "T", "gate alone ms", "min", "keys firing", "peak MiB", "util % before / after", "others before / after", "capped"], rows2))


def cpu_and_storage():
    series = {}
    rows, rows_s = [], []
    for bb in BB:
        mi = load(f"{OUT}/latency/{bb}__model.json")
        if mi and not mi.get("model_parameters"):
            mi = None
        items = [(n, f"{OUT}/cpu/{n}.json") for n in REAL[bb]] + [(f"synthetic-T{T}", f"{OUT}/cpu/{bb}__synthetic-T{T}.json") for T in SYN_T]
        for name, p in items:
            r = load(p)
            if r is None:
                MISSING.append(f"cpu {bb} {name}")
                continue
            st = r["storage"]
            if mi and "model_bytes" not in st:
                st.update(model_bytes=mi["model_bytes"], share_of_model_bytes_fp64_pct=st["numbers_serving"] * 8 / mi["model_bytes"] * 100,
                          share_of_model_bytes_module_dtype_pct=st["numbers_serving"] * 2 / mi["model_bytes"] * 100)
            if mi:
                st["share_of_model_parameters_pct"] = st["numbers_serving"] / mi["model_parameters"] * 100
                st["T_at_model_parameters"] = mi["model_parameters"] / st["numbers_per_edit"]
            T = r["T"]
            lat = load(lat_file(bb, name))
            go = load(f"{OUT}/latency/{bb}__{name}__gate_only_gpu.json")
            if r["kind"] == "synthetic":                                # one source for the whole synthetic series: the gate-only job (all T in one job, no model)
                lat_s, lat = lat, None
            else:
                go = None
            gate_gpu = lat["modes"]["last_fp64"]["gate_only_ms_mean_of_min"] if lat else (go["gate_only_ms_mean_of_min"] if go else None)      # min-based: comparable across loads
            gate_gpu_med = lat["modes"]["last_fp64"]["gate_only_ms"]["median"] if lat else (go["gate_only_ms"]["median"] if go else None)
            gate_gpu_src = "captured sequences, per-sequence minima" if lat else ("one key per forward, per-key minima" if go else "-")
            if r["kind"] == "synthetic":
                lat = lat_s
            cal = r["calibration"]
            lr = r["least_repair"]
            e = {"backbone": bb, "log": name, "kind": r["kind"], "T": T, "storage_mb_fp64": st["mb_serving_fp64"], "storage_mb_module_dtype": st["mb_serving_module_dtype"],
                 "storage_numbers": st["numbers_serving"], "module_mb": st["mb_module"], "storage_share_of_module_numbers_pct": st["share_of_module_numbers_pct"],
                 "storage_share_of_model_parameters_pct": st.get("share_of_model_parameters_pct"), "storage_share_of_model_bytes_fp64_pct": st.get("share_of_model_bytes_fp64_pct"),
                 "gate_gpu_ms": gate_gpu, "gate_gpu_ms_median_under_load": gate_gpu_med, "gate_gpu_source": gate_gpu_src, "gate_cpu_fp64_ms": r["gate_cpu_ms"]["fp64"]["median"], "gate_cpu_fp32_ms": r["gate_cpu_ms"]["fp32"]["median"],
                 "overhead_end_to_end_pct": lat["modes"]["last_fp64"]["overhead_end_to_end_pct"]["median"] if lat else None,
                 "calibration_s_library": cal.get("library_with_guards_s"), "calibration_s_slim": cal["slim_s"],
                 "least_repair_ms_library": lr.get("library_with_guards_ms", {}).get("median") if "library_with_guards_ms" in lr else None, "least_repair_ms_slim": lr["slim_ms"]["median"],
                 "revocation_ms": r["revocation"]["threshold_only_ms"], "revocation_ms_driver_mask_edits": r["revocation"].get("ms"),
                 "null_repair_s": {A: v["s"] for A, v in r.get("null_repair", {}).items()}, "minimum_cut_s": (r.get("minimum_cut") or {}).get("s"),
                 "n_edits_capped": cal.get("n_edits_capped", cal["slim_stages"]["n_edits_capped"])}
            series.setdefault(bb, []).append(e)
            lab = name if r["kind"] == "real" else f"{bb}:{name}"
            for k, v in e.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool) and k != "T":
                    R.add(lab, "rescope", k, v, alpha=0.05, n=200, draw=0, pool="cost", T=T, backbone=bb, kind=r["kind"])
            rows.append([bb, name, T, f(e["storage_mb_fp64"], 1), f(e["storage_mb_module_dtype"], 1), f(e["storage_share_of_module_numbers_pct"], 1), f(e["storage_share_of_model_parameters_pct"], 3),
                         f(e["storage_share_of_model_bytes_fp64_pct"], 3), f(e["gate_gpu_ms"], 3), mpp(r["gate_cpu_ms"]["fp64"], 2), f(e["gate_cpu_fp32_ms"], 3),
                         f(e["calibration_s_library"], 3), f(e["calibration_s_slim"], 3), f(e["least_repair_ms_library"], 2), f(e["least_repair_ms_slim"], 2), f(e["revocation_ms"], 3),
                         f(e["revocation_ms_driver_mask_edits"], 1), " / ".join(f(e["null_repair_s"].get(A), 2) for A in ("1", "10", "100")) if e["null_repair_s"] else "-", f(e["minimum_cut_s"], 3)])
            if r["kind"] == "real":
                pn(f"{name}.storage_mb", e["storage_mb_fp64"], f"E8, {name}: megabytes of the scoped log as evaluated (fp64): T (d_in + d_out + 3) numbers = detectors, released values, scales, thresholds, ranking baselines; guard keys (2 T d_in, used by calibration and repair only) not included")
                pn(f"{name}.storage_mb_module_dtype", e["storage_mb_module_dtype"], "same numbers stored in the module's dtype (2 bytes each)")
                pn(f"{name}.storage_numbers", e["storage_numbers"], "numbers of the scoped log: T (d_in + d_out + 3)")
                pn(f"{name}.module_mb", e["module_mb"], "megabytes of the edited module (d_out x d_in, module dtype)")
                pn(f"{name}.storage_share_of_module_numbers_pct", e["storage_share_of_module_numbers_pct"], "numbers of the scoped log as a percentage of the numbers of the module")
                pn(f"{name}.calibration_s", e["calibration_s_library"], f"E8, {name}: seconds of one two-block calibration (n = 200) from stored keys on the CPU (4 threads): coordinates of the 200 queries + the library rule with guards; median of 5")
                pn(f"{name}.repair_ms_per_complaint", e["least_repair_ms_library"], f"E8, {name}: milliseconds of the least repair of one complaint from its stored key on the CPU (library least_repair with the guard check); median of 30 complaints")
                pn(f"{name}.revocation_ms", e["revocation_ms"], "milliseconds of one revocation (threshold of one edit set to +inf; a new threshold vector)")
                pn(f"{name}.gate_cpu_ms_per_forward", e["gate_cpu_fp64_ms"], "the gate alone on the CPU (fp64, one eligible position), median ms")
                for A in ("1", "10", "100"):
                    pn(f"{name}.null_repair_s.A{A}", e["null_repair_s"].get(A), f"seconds to remove {A} keys from all detectors (library null_repair, ridge 0), CPU, median of 3")
                pn(f"{name}.minimum_cut_s", e["minimum_cut_s"], "seconds of one minimum cut (library scope_cut: all valid REPAIR rows, one text-rephrase witness per edit, lambda 1), CPU, median of 3")
            else:
                for pre in ([f"synthetic.{bb}.T{T}"] + ([f"synthetic.T{T}"] if bb == "qwen25vl" else [])):
                    note = f"E8 synthetic log, backbone dimensions of {bb}, T = {T}" + (" (unqualified key = Qwen2.5-VL-3B)" if pre.count(".") == 1 else "")
                    pn(f"{pre}.storage_share_of_model_pct", e["storage_share_of_model_parameters_pct"], f"{note}: numbers of the scoped log, T (d_in + d_out + 3), as a percentage of the number of parameters of the whole model (the convention of the published figure: parameter counts)")
                    pn(f"{pre}.storage_share_of_model_bytes_fp64_pct", e["storage_share_of_model_bytes_fp64_pct"], f"{note}: bytes of the scoped log in fp64 (as evaluated) as a percentage of the bytes of the model in its dtype")
                    pn(f"{pre}.storage_mb", e["storage_mb_fp64"], f"{note}: megabytes in fp64")
                    pn(f"{pre}.storage_mb_per_edit", e["storage_mb_fp64"] / T, f"{note}: megabytes per edit in fp64")
                    pn(f"{pre}.gate_ms_per_forward", e["gate_gpu_ms"], f"{note}: the gate alone on the GPU (fp64, one eligible position per forward), ms, min-based ({gate_gpu_src}; shared card)")
                    pn(f"{pre}.gate_ms_per_forward_median_under_load", e["gate_gpu_ms_median_under_load"], f"{note}: same, median over all runs on the shared card (load-dependent)")
                    pn(f"{pre}.gate_cpu_ms_per_forward", e["gate_cpu_fp64_ms"], f"{note}: the gate alone on the CPU (fp64, 4 threads), median ms")
                    pn(f"{pre}.calibration_s", e["calibration_s_slim"], f"{note}: seconds of one two-block calibration (n = 200) from stored keys, CPU")
                    pn(f"{pre}.repair_ms_per_complaint", e["least_repair_ms_slim"], f"{note}: milliseconds of the least repair of one complaint from its stored key, CPU (library least_repair + the own-coordinate cap)")
                    pn(f"{pre}.revocation_ms", e["revocation_ms"], f"{note}: milliseconds of one revocation (threshold of one edit to +inf; a new threshold vector)")
        # storage crossover
        any_r = next((load(p) for _, p in items if os.path.exists(p)), None)
        if any_r:
            st = any_r["storage"]
            t_mod = st["T_at_module_size_numbers"]
            for pre in ([f"storage.{bb}"] + (["storage"] if bb == "qwen25vl" else [])):
                pn(f"{pre}.T_at_module_size", int(t_mod), f"E8: number of edits at which the scoped log holds as many numbers as the edited module, d_out d_in / (d_in + d_out + 3), rounded down; {bb}" + (" (unqualified key = Qwen2.5-VL-3B)" if pre == "storage" else ""))
                pn(f"{pre}.T_at_module_size_fp64_store", int(st["T_at_module_size_bytes_fp64_store"]), "same in bytes with the store in fp64 and the module in its dtype (a quarter)")
                if mi:
                    pn(f"{pre}.T_at_model_size", int(mi["model_parameters"] / st["numbers_per_edit"]), "number of edits at which the scoped log holds as many numbers as the whole model has parameters")
                pn(f"{pre}.numbers_per_edit", st["numbers_per_edit"], "numbers stored per edit: d_in + d_out + 3")
            rows_s.append([bb, st["d_in"], st["d_out"], st["numbers_per_edit"], f(st["mb_per_edit_fp64"], 4), f(st["mb_per_edit_module_dtype"], 4), f(st["mb_module"], 1), int(t_mod),
                           int(st["T_at_module_size_bytes_fp64_store"]), mi["model_parameters"] if mi else "-", int(mi["model_parameters"] / st["numbers_per_edit"]) if mi else "-"])
    TXT.append("E8 (2, 3)  STORAGE AND REPAIR COST ON THE CPU FROM STORED KEYS (4 threads), real logs and synthetic logs (synth.py), default method (two-block, alpha 0.05, n 200, draw 0)\n"
               "  store = T (d_in + d_out + 3) numbers (detectors, values, scales, thresholds, baselines); guard keys (2 T d_in, calibration / repair only) not included; % module = numbers of the\n"
               "  store / numbers of the module; % model = numbers / parameters of the whole model; (fp64) = bytes of the fp64 store / bytes of the model.  gate GPU = gate alone (fp64, one eligible\n"
               "  position); cal = one calibration, n = 200: lib = library rule with guards on stored keys, slim = the same thresholds without materialised guard keys (the form used for the\n"
               "  synthetic logs); least = least repair of one complaint; revoke = one threshold to +inf (mask = fx.methods.mask_edits); null = removal of 1 / 10 / 100 keys from all detectors; cut = one minimum cut\n"
               + tt(["backbone", "log", "T", "MB fp64", "MB mod", "% module", "% model", "% model (fp64)", "gate GPU ms", "gate CPU ms [p10, p90]", "CPU fp32", "cal lib s", "cal slim s",
                     "least lib ms", "least slim ms", "revoke ms", "mask ms", "null s 1/10/100", "cut s"], rows)
               + "\n  storage per edit and the T at which the store reaches the size of the module / of the model:\n"
               + tt(["backbone", "d_in", "d_out", "numbers / edit", "MB / edit fp64", "MB / edit module dtype", "module MB", "T = module (numbers)", "T = module (fp64 store)", "model parameters",
                     "T = model (numbers)"], rows_s))
    jdump({"doc": "E8: cost as a function of T, real logs and synthetic logs, per backbone (list of entries; fields as in table (2, 3))", "series": series}, f"{OUT}/scale_series.json")
    return series


def scale_table(series):
    """The one table of the task: 10^3, 10^4, 10^5 edits + the published figures of NeuralDB in the last row."""
    rec = None
    for x in jload(f"{NDB}/records.json") if isinstance(jload(f"{NDB}/records.json"), list) else jload(f"{NDB}/records.json").get("records", []):
        if isinstance(x, dict) and x.get("id") == "fei2026neuraldb":
            rec = x
    rows = []
    for bb in BB:
        for e in series.get(bb, []):
            if e["kind"] != "synthetic":
                continue
            rows.append([f"synthetic, {bb}", e["T"], f(e["storage_mb_fp64"] / e["T"], 4), f(e["storage_mb_fp64"], 0), f(e["storage_share_of_model_parameters_pct"], 2),
                         f(e["storage_share_of_model_bytes_fp64_pct"], 1), f(e["gate_gpu_ms"], 3), f(e["gate_cpu_fp64_ms"], 2), f(e["calibration_s_slim"], 2), f(e["least_repair_ms_slim"], 2),
                         f(e["revocation_ms"], 3)])
    quote, place = None, None
    if rec:
        for q in rec["quotes"]:
            if "20% of the original model" in q["text"]:
                quote, place = q["text"], q["where"]
    page = f"{NDB}/_raw/2507.18028/html/page.txt"
    found = {}
    if os.path.exists(page):
        for i, ln in enumerate(open(page), 1):
            if "the additional memory usage reached 20% of the original model" in ln:
                found["storage_1e5"] = i
            if "increases only by $1.5\\%$" in ln:
                found["time_1e4"] = i
            if "the memory of 10,000 facts is about 150 million parameters" in ln:
                found["storage_1e4"] = i
    elif rec and rec.get("local_copy_lines"):                 # ARTIFACT: the text copy of the cited paper is not released; the released input keeps its line numbers
        found = dict(rec["local_copy_lines"])
    src = "Fei et al., NeuralDB, arXiv 2507.18028v1 (ICLR 2026)"
    rows.append([f"published: NeuralDB, Llama-3 8B, 10^5 facts ({src}, Appendix A 'Limitation'; local copy line {found.get('storage_1e5')})", 100000, "(d1 + d2 = 18432 numbers)", "-",
                 "20 (of the model, as published)", "-", "not published at 10^5 (evaluation time +1.5 % at 10^4 facts, Sec. 4)", "-", "-", "not published", "not published (no deletion experiment)"])
    TXT.append("E8 (5)  SCALE TABLE: synthetic scoped logs at 10^3, 10^4, 10^5 edits and, in the last row, the figures that the gated key-value editor NeuralDB publishes for 10^5 facts\n"
               "  (contextual, not normalised to our setting, not re-run).  storage per edit and in total; % model = numbers / model parameters ((fp64): bytes of the fp64 store / bytes of the model);\n"
               "  lookup = the gate alone per forward pass (one eligible position); repair = one least repair; revocation = one threshold write\n"
               + tt(["system", "T", "MB / edit", "MB total", "% model", "% model (fp64)", "lookup GPU ms", "lookup CPU ms", "calibration s", "repair ms", "revocation ms"], rows)
               + (f"\n  quotation ({place}): \"{quote}\"" if quote else "\n  quotation: record fei2026neuraldb not found")
               + f"\n  source record: 03_analysis/scope_lit/D_unlearning_editing_scale/records.json (id fei2026neuraldb); text: _raw/2507.18028/html/page.txt lines {found}")
    pn("published.neuraldb.storage_at_1e5_text", "20\\% of the model", f"published figure, not measured here: {src}, Appendix A (Limitation): '{quote}' (local copy _raw/2507.18028/html/page.txt line {found.get('storage_1e5')})")
    pn("published.neuraldb.storage_at_1e5_pct", 20, "same, as a number (percent of the original model, Llama-3 8B, 100,000 facts)")
    pn("published.neuraldb.storage_at_1e4_pct", 2.2, f"published: about 150 million parameters, 'approximately 2.2%' of the model at 10,000 facts (Sec. 4; local copy line {found.get('time_1e4')})")
    pn("published.neuraldb.evaluation_time_increase_at_1e4_pct", 1.5, f"published: evaluation time for 10,000 facts on CounterFact 'increases only by 1.5%' (Sec. 4; local copy line {found.get('time_1e4')}); no figure at 100,000 facts")
    pn("published.neuraldb.source", src + ", Appendix A (Limitation) and Sec. 4", "where the published figures are printed")


def linear():
    rows = []
    for name in (DEV["qwen25vl"], DEV["llava"]):
        agg = {}
        for p in sorted(glob.glob(f"{P12}/{name}/*__n*__s*.json")):
            r = jload(p)
            if not isinstance(r.get("n_reported"), int) or r["n_reported"] < 1 or "cost" not in r or r.get("method") in (None, "NONE"):
                continue
            c = r["cost"]
            agg.setdefault((r["method"], r["n"]), []).append((r["n_reported"], c.get("n_forward", 0), c.get("n_backward", 0), c.get("sec_total", 0.0), r.get("repair_wall_sec")))
        for (meth, n), v in sorted(agg.items(), key=lambda kv: (kv[0][0], kv[0][1] if isinstance(kv[0][1], int) else 10 ** 6)):
            if n not in (1, 10, 100):
                continue
            nr = np.array([x[0] for x in v], dtype=float)
            fw, bw, sec = (np.array([x[i] for x in v], dtype=float) / nr for i in (1, 2, 3))
            rows.append([name, meth, n, len(v), f(fw.mean(), 1), f(bw.mean(), 1), f(sec.mean() * 1e3, 0), f"{P12.split('experiments/')[1]}/{name}/{meth}__n{n}__s*.json"])
            for k, val in (("forward_passes_per_complaint", fw.mean()), ("backward_passes_per_complaint", bw.mean()), ("ms_per_complaint", sec.mean() * 1e3)):
                R.add(name, meth, k, float(val), n=n if isinstance(n, int) else None, pool="P12 records", n_units=len(v))
                pn(f"linear.{name}.{meth}.n{n}.{k}", float(val), f"E8 (4), from the existing P12 record of {name}: {meth} with n = {n} reported complaints, {k} (mean over {len(v)} seeds; repair only, setup excluded)")
    TXT.append("E8 (4a) COST PER COMPLAINT OF THE LINEAR REPAIRS, from the existing P12 records (no new run): passes and seconds of the repair divided by the number of reported complaints,\n"
               "  mean over seeds; fwd = forward passes of any kind (full, prefix, suffix), bwd = backward passes; wall-clock of the records (shared cards; setup excluded)\n"
               + tt(["log", "method", "n complaints", "seeds", "fwd / complaint", "bwd / complaint", "ms / complaint", "record"], rows))
    t = load(f"{SMOKE}/timing.json")
    res = load(f"{SMOKE}/results.json")
    if t:
        rows = []
        cfg = (res or {}).get("configs", {})
        for c, v in t["configs"].items():
            ps = [x["passes"] for x in v.get("rows", [])] if "rows" in v else []
            fwd = np.mean([p["full"] + p["prefix"] + p["suffix"] + p["fwd_bwd"] + p["suffix_grad"] for p in ps]) if ps else None
            bwd = np.mean([p["fwd_bwd"] + p["suffix_grad"] for p in ps]) if ps else None
            rows.append([c, v["label"], f(fwd, 1), f(bwd, 1), f(v["sec_total_median"] * 1e3, 1), f"{v['sec_total_p10'] * 1e3:.1f} / {v['sec_total_p90'] * 1e3:.1f}"])
            key = v["label"].split("[")[0] + "." + c
            pn(f"linear.smoke.{t['log']}.{key}.ms_per_complaint", v["sec_total_median"] * 1e3, f"E8 (4), clean timing of the harness smoke on {t['log']} (one repair per complaint, 10 complaints, one card with {t['processes_on_this_gpu_at_end']} process): {v['label']}, median ms")
            R.add(t["log"], v["label"], "ms_per_complaint_clean@median", v["sec_total_median"] * 1e3, ci_lo=v["sec_total_p10"] * 1e3, ci_hi=v["sec_total_p90"] * 1e3, n=1, pool="harness smoke, clean timing", n_units=10)
        TXT.append(f"E8 (4b) CLEAN WALL-CLOCK OF ONE LINEAR REPAIR OF ONE COMPLAINT (07_repair_v3/harness/out/smoke/timing.json; {t['log']}, 10 complaints, {t['gpu']}, {t['processes_on_this_gpu_at_end']} "
                   f"process on the card, {t['torch_threads']} CPU threads)\n" + tt(["config", "method", "fwd / complaint", "bwd / complaint", "median ms", "p10 / p90 ms"], rows))
    rows = []
    for bb in BB:
        r = load(f"{OUT}/latency/{bb}__dual.json")
        if r is None:
            MISSING.append(f"answer-match baseline timing {bb}")
            continue
        for k in ("preedit_one_generation", "deployed_one_generation", "dual_two_generations"):
            rows.append([r["log"], k, r["n_queries"], r[k]["n_generations_per_query"], f(r[k]["ms_per_query"], 1), f"{r['load_before'].get('utilization_pct')} / {r['load_after'].get('utilization_pct')}",
                         f"{r['load_before'].get('n_other_processes')} / {r['load_after'].get('n_other_processes')}"])
            R.add(r["log"], k, "ms_per_query", r[k]["ms_per_query"], pool="50 heldout_disjoint queries", n_units=r["n_queries"])
            pn(f"{r['log']}.answer_match.{k}.ms_per_query", r[k]["ms_per_query"], f"E8 (4), {r['log']}: {k}, wall-clock ms per query over 50 held-out queries (decoder-v2 generation; shared card)")
    if rows:
        TXT.append("E8 (4c) ANSWER-MATCH BASELINE (two generations per query), 50 heldout_disjoint queries, wall-clock per query next to one generation of the pre-edit / deployed model\n"
                   + tt(["log", "what", "queries", "generations / query", "ms / query", "util % before / after", "others before / after"], rows))


def main():
    series = {}
    for fn in (latency, cpu_and_storage, linear):
        try:
            out = fn()
            if fn is cpu_and_storage:
                series = out
        except Exception as ex:
            import traceback
            MISSING.append(f"{fn.__name__}: render error {ex!r}")
            traceback.print_exc()
    try:
        scale_table(series)
    except Exception as ex:
        import traceback
        MISSING.append(f"scale_table: render error {ex!r}")
        traceback.print_exc()
    head = ("E8  COST.  Rendered by code/render.py on " + time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()) + f"; driver fx {fx.code_fingerprint()}\n"
            "    Not finished / missing: " + ("; ".join(MISSING) if MISSING else "nothing"))
    if any("render error" in m for m in MISSING):                       # ARTIFACT: a part that failed must not silently drop its paper numbers
        raise SystemExit("e8 render: stopped, nothing written: " + "; ".join(m for m in MISSING if "render error" in m)
                         + "\n  Inputs: out/ (results_rows.tar.gz unpacked into the clone) and experiments/e8_cost/inputs/published_figures.json "
                           "(mounted by rescope_paths.py at 03_analysis/scope_lit/D_unlearning_editing_scale/records.json).")
    R.save(missing=MISSING)
    R.table(head + "\n\n" + "\n\n".join(TXT))
    fx.io.jdump(dict(sorted(PN.items())), f"{OUT}/paper_numbers.json", indent=1)
    fx.io.jdump(dict(sorted(DOC.items())), f"{OUT}/paper_numbers_doc.json", indent=1)
    print(head)
    print(len(R.records), "records;", len(PN), "paper numbers")


if __name__ == "__main__":
    main()
