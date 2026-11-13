"""Render USAGE.txt from USAGE.template.txt and the JSON written by the tests and the runs (no number is typed by hand).

    python scripts/make_usage.py        (after: tests/render.py, scripts/render_table.py, scripts/project_cost.py)
"""
import hashlib
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "code"))
import e10t  # noqa: E402
import fx  # noqa: E402
from e10t import backbone as BB  # noqa: E402

J = e10t.data.jload
T = fx.io.text_table
tests = {n: J(os.path.join(ROOT, "tests", "out", f)) for n, f in
         (("cache", "gpu_cache.json"), ("batch", "gpu_batch.json"), ("target", "gpu_target.json"), ("cguard", "cpu_guard.json"), ("gguard", "gpu_guard.json"))}
summ = J(os.path.join(ROOT, "tests", "out", "summary.json"))
beh = J(os.path.join(e10t.OUT, "dev", "parts", "behave.json"))
pub = J(os.path.join(e10t.OUT, "dev", "parts", "public.json"))
gen = J(os.path.join(e10t.OUT, "dev", "parts", "gen.json"))
gate = J(os.path.join(e10t.OUT, "dev", "parts", "gate.json"))
cost = J(os.path.join(e10t.OUT, "cost_projection.json"))
kc_meta = {p: J(os.path.join(e10t.CACHE_ROOT, e10t.BACKBONE, p, "meta.json")) for p in ("pools", "dev")}
now = e10t.code_fingerprint()

# ----------------------------------------------------------------------------- memory figure
import glob  # noqa: E402

peaks = [J(p).get("peak_alloc_MiB", 0) for p in glob.glob(os.path.join(ROOT, "tests", "out", "gpu_*.json"))] + [m["peak_alloc_MiB"] for m in cost["mem"].values()]
peak = max(peaks)                                                        # peak ALLOCATED by torch (reserved memory grows to whatever cap the job has: allocator cache)
SLACK = 3500                                                             # MiB between allocated and what the allocator needs (reserved - allocated reached 3.0 GiB under an 11.4 GiB cap)
MEM = int(math.ceil((peak + 600 + SLACK) / 500.0) * 500)

# ----------------------------------------------------------------------------- tests
rows = []
for name, t in summ["tests"].items():
    rows.append([name, t["status"], f"{t['seconds']:.0f}" if "seconds" in t else "-", t.get("what", "")])
tests_txt = (f"  {summ['n_pass_current_code']} of {summ['n_tests']} tests pass with the code of the fingerprint below; failed: {summ['n_fail']}; not run: {summ['n_not_run']}"
             f"   (evidence line of every test: tests/out/table.txt)\n" + "\n".join("  " + ln for ln in T(["test", "status", "s", "what it establishes"], rows, align=["l", "l", "r", "l"]).splitlines()))

# ----------------------------------------------------------------------------- fingerprint
files = []
for f in sorted(os.listdir(os.path.join(ROOT, "code", "e10t"))):
    if f.endswith(".py"):
        b = open(os.path.join(ROOT, "code", "e10t", f), "rb").read()
        files.append([f"code/e10t/{f}", hashlib.sha1(b).hexdigest()[:12], str(len(b.splitlines()))])
fxf = {f: hashlib.sha1(open(os.path.join(e10t.COMMON, "fx", f), "rb").read()).hexdigest()[:12] for f in ("draws.py", "io.py", "stats.py")}
fx_parts = sorted({x["meta"]["code"]["fx"] for x in (beh, pub, gen, gate)} | {summ["code"]["fx"], cost["code"]["fx"]})
fp_txt = (f"  e10t {now['e10t']} ({now['e10t_version']}); library rescope {now['rescope']}; driver fx {now['fx']}; builder's loader tx_public {now['tx_public']}\n"
          f"  driver files this package calls (fx.draws.ranking, fx.io, fx.stats): " + ", ".join(f"{k} {v}" for k, v in fxf.items())
          + f"; whole-driver fingerprints recorded by the results: {', '.join(fx_parts)}\n"
          + ("" if fx_parts == [now["fx"]] else "  (the driver package changed while this work ran: other files of fx than the three above; e10t imports fx as a whole)\n")
          + f"  (sha1 over the sorted .py files of each package, 12 hex digits: e10t.code_fingerprint(); test results and part files carry the value they ran with:\n"
          f"   tests {summ['code']['e10t']}, development-log parts " + ", ".join(sorted({x['meta']['code']['e10t'] for x in (beh, pub, gen, gate)})) + f", cost measurements {cost['code']['e10t']})\n"
          + "\n".join("  " + ln for ln in T(["file", "sha1", "lines"], files).splitlines()))

# ----------------------------------------------------------------------------- cost projection for a T = 2000 log
allp = e10t.data.all_prompts()
TT = 2000
n = {k: sum(1 for r in allp if r["set"] == "stream" and r["pos"] < TT and r["kind"] == k) for k in ("request", "paraphrase", "neighborhood")}
n["heldout"] = sum(1 for r in allp if r["set"] == "heldout")
uniq = {}
for r in allp:
    if (r["set"] == "stream" and r["pos"] < TT) or r["set"] in ("heldout", "repair"):
        uniq.setdefault(r["set"] if r["set"] != "stream" else "stream", {})[r["text_id"]] = r["n_tok"] + e10t.N_DEC
npos = {k: sum(v.values()) for k, v in uniq.items()}
ntext = {k: len(v) for k, v in uniq.items()}
P = cost["passes"]
w_rp, w_b = P["weight"]["request_paraphrase"], P["weight"]["benign"]
g = P[f"gate_T{TT}"]
g_rp, g_b, g_sil = g["rescope"]["request_paraphrase"], g["rescope"]["benign"], g["silent_hook"]["benign"]
g100 = P["gate_T100"]
both = lambda x: x["tf"] + x["dec"]                                      # noqa: E731  seconds per prompt (teacher-forced + decode)
n_rp, n_bl = n["request"] + n["paraphrase"], n["neighborhood"] + n["heldout"]
dev_m = beh["methods"]
gate_rows = [k for k in dev_m if dev_m[k]["kind"] == "gate"]
frac = {k: (dev_m[k]["heldout"]["n_fired"] + dev_m[k]["neighborhood"]["n_fired"]) / (dev_m[k]["heldout"]["n"] + dev_m[k]["neighborhood"]["n"]) for k in gate_rows}
t_weight = n_rp * both(w_rp) + n_bl * both(w_b)
t_gate_A = {k: n_rp * both(g_rp) + frac[k] * n_bl * both(g_b) for k in gate_rows}
t_gate_B = n_rp * both(g_rp) + n_bl * both(g_b)
pubT = P[f"public_T{TT}"]
t_pub_w, t_pub_g = TT * pubT["prob_test_seconds_per_record_weight"], TT * pubT["prob_test_seconds_per_record_gate"]
t_gen_w, t_gen_g = 100 * pubT["generation_seconds_per_record_weight"], 100 * pubT["generation_seconds_per_record_gate"]
gl = cost["gate_level"][f"T{TT}"]
bd = cost["build"][f"T{TT}"]
n_feat_pos = npos["stream"] + npos["heldout"]                            # positions scored per detector set (the evaluated parts; calibration prompts are few)
feat = {d: n_feat_pos * TT * gl[f"ns_per_position_detector_{d}"] * 1e-9 for d in ("cpu", "cuda")}
n_sets = 2 + e10t.N_DRAWS                                                # 'ncoord' base, 'cos' base (shared by cosine / cosine_published / exact), one null-repaired set per draw
t_dec_q = gl["decide_100_request_prompts_seconds"] / 100.0             # per-position refinement of ONE firing query (measured on 100 request prompts)
cal_rows = [k for k in gate_rows if k in e10t.methods.DEPENDS_ON_DRAW]
n_gate_evals = len(cal_rows) * e10t.N_DRAWS + (len(gate_rows) - len(cal_rows))
# construction: measured for the six gate rows of the eight; a ReScope-W row calibrates like the default method (same detectors)
t_build = (e10t.N_DRAWS * (bd["rescope_next_draw"] * (2 if "where_twoblock" in gate_rows else 1) + bd["rescope_null_next_draw"] + bd["rescope_shared"] + bd["cosine"])
           + bd["rescope"] + bd["cosine_published"] + bd["exact"])
fired_A = sum((e10t.N_DRAWS if k in cal_rows else 1) * (n_rp + frac[k] * n_bl) for k in gate_rows)     # firing queries over all evaluations (development-log shares)
t_gl = {d: n_sets * feat[d] + t_dec_q * fired_A + t_build for d in ("cpu", "cuda")}
t_gl_B = {d: n_sets * feat[d] + t_dec_q * n_gate_evals * (n_rp + n_bl) + t_build for d in ("cpu", "cuda")}
rate_cache = kc_meta["pools"]["n_positions"] / kc_meta["pools"]["seconds"]
t_cache = npos["stream"] / rate_cache
# retention: deployed + default method + the two ReScope-W rows at T = 100, 500, 1000, 2000 on the first 100 requests (300 edit prompts, 1000
# neighbourhood prompts, 100 records per row and T)
ret_gates = [k for k in ("rescope", "where_logonly", "where_twoblock") if k in gate_rows]
t_ret = 4 * ((300 * both(w_rp) + 1000 * both(w_b)) + sum(300 * both(g_rp) + frac[k] * 1000 * both(g_b) for k in ret_gates)
             + 100 * (pubT["prob_test_seconds_per_record_weight"] + len(ret_gates) * pubT["prob_test_seconds_per_record_gate"]))
t_ret_B = 4 * ((300 * both(w_rp) + 1000 * both(w_b)) + len(ret_gates) * (300 * both(g_rp) + 1000 * both(g_b))
               + 100 * (pubT["prob_test_seconds_per_record_weight"] + len(ret_gates) * pubT["prob_test_seconds_per_record_gate"]))
A = 2 * t_weight + sum(t_gate_A.values()) + 2 * t_pub_w + len(gate_rows) * t_pub_g + 2 * t_gen_w + t_gen_g + t_ret
Bu = 2 * t_weight + len(gate_rows) * t_gate_B + 2 * t_pub_w + len(gate_rows) * t_pub_g + 2 * t_gen_w + t_gen_g + t_ret_B
dA = sum(t_gate_A[k] for k in cal_rows) + len(cal_rows) * t_pub_g
dB = len(cal_rows) * (t_gate_B + t_pub_g)
m = lambda s: f"{s / 60.0:.1f}"                                          # noqa: E731
ms = lambda x: f"{1000 * x:.1f}"                                         # noqa: E731
lines = []
w = lines.append
w("  MEASURED (seconds per sequence; development-log prompts in padded batches; synthetic logs = the development log's factors tiled with 2 % noise)")
r = [["weight (deployed / pre-edit)", "-", ms(w_b["tf"]), ms(w_b["dec"]), ms(w_rp["tf"]), ms(w_rp["dec"])]]
for Tn in (100, 500, 2000):
    gg = P[f"gate_T{Tn}"]
    r.append([f"gate, T = {Tn}: hook scores, nothing fires", str(Tn), ms(gg["silent_hook"]["benign"]["tf"]), ms(gg["silent_hook"]["benign"]["dec"]), "-", "-"])
    r.append([f"gate, T = {Tn}: default method", str(Tn), ms(gg["rescope"]["benign"]["tf"]), ms(gg["rescope"]["benign"]["dec"]),
              ms(gg["rescope"]["request_paraphrase"]["tf"]), ms(gg["rescope"]["request_paraphrase"]["dec"])])
    r.append([f"gate, T = {Tn}: linear log, every position fires", str(Tn), "-", "-", ms(gg["where_all_fire"]["request_paraphrase"]["tf"]), ms(gg["where_all_fire"]["request_paraphrase"]["dec"])])
for ln in T(["served model", "T", "benign tf ms", "benign dec ms", "req/par tf ms", "req/par dec ms"], r).splitlines():
    w("  " + ln)
r = []
for Tn in (100, 500, 2000):
    pp, b2, g2, mm = P[f"public_T{Tn}"], cost["build"][f"T{Tn}"], cost["gate_level"][f"T{Tn}"], cost["mem"][f"T{Tn}"]
    r.append([str(Tn), f"{pp['prob_test_seconds_per_record_weight']:.3f}", f"{pp['prob_test_seconds_per_record_gate']:.3f}", f"{pp['generation_seconds_per_record_weight']:.2f}",
              f"{pp['generation_seconds_per_record_gate']:.2f}", f"{b2['rescope']:.1f}", f"{b2['rescope_next_draw']:.1f}", f"{b2['rescope_null_next_draw']:.1f}", f"{b2['cosine']:.1f}",
              f"{g2['ns_per_position_detector_cpu']:.0f}", f"{g2['ns_per_position_detector_cuda']:.1f}", f"{mm['peak_alloc_MiB']:.0f}", f"{mm['peak_reserved_MiB']:.0f}"])
for ln in T(["T", "public test s/rec weight", "gate", "generation s/rec weight", "gate", "build rescope s", "next draw", "null next draw", "cosine",
             "features ns/(pos x det) cpu", "cuda", "peak alloc MiB", "reserved"], r).splitlines():
    w("  " + ln)
w(f"  key cache: {kc_meta['pools']['n_positions']} positions ({kc_meta['pools']['n_texts']} texts) in {kc_meta['pools']['seconds']:.0f} s; "
  f"{4 * 6400 / 2 ** 20:.4f} MiB per position on disk (float32).")
w(f"  GPU memory: peak allocated by torch over all tests and measurements (T up to 2000) {peak:.0f} MiB -> --mem {MEM} (+ 600 MiB context + {SLACK} MiB allocator slack,")
w("  rounded up; the batch shapes vary, so the allocator's reserved memory grows to whatever cap the job has).")
w("")
w(f"  PROJECTION for one log with T = {TT} (prompt counts from the build's prompt table: {n['request']} request, {n['paraphrase']} paraphrase, {n['neighborhood']} neighbourhood,")
w(f"  {n['heldout']} HELD-OUT benign prompts; {ntext['stream']} distinct stream texts with {npos['stream']} cached positions = {npos['stream'] * 6400 * 4 / 2 ** 30:.1f} GiB of key cache).")
w("  Every prompt costs a teacher-forced and a decode pass.  A weight row runs every prompt.  A gate row runs the prompts on which an edit is released;")
w("  how many that is on a reported log is NOT known (no method was run there), so two scenarios are given: A = the request and paraphrase prompts")
w("  and the share of benign / neighbourhood prompts that fired on the development log; B = every prompt runs through the hook (upper bound).")
r = [["key cache of the stream's prompts (once per stream)", m(t_cache), m(t_cache)],
     [f"(iv) gate level, 20 draws ({n_gate_evals} gate evaluations), features on the GPU (--mem 4000)", m(t_gl["cuda"]), m(t_gl_B["cuda"])],
     [f"(iv) gate level, 20 draws ({n_gate_evals} gate evaluations), features on the CPU, 4 threads", m(t_gl["cpu"]), m(t_gl_B["cpu"])],
     ["(i) + (iii) deployed", m(t_weight), m(t_weight)], ["(i) + (iii) pre-edit", m(t_weight), m(t_weight)]]
for k in gate_rows:
    r.append([f"(i) + (iii) {k}   (A: {100 * frac[k]:.1f} % of the benign / neighbourhood prompts run)", m(t_gate_A[k]), m(t_gate_B)])
r += [["(ii) public scores, 2 weight rows", m(2 * t_pub_w), m(2 * t_pub_w)], [f"(ii) public scores, {len(gate_rows)} gate rows (no shortcut: every record runs)", m(len(gate_rows) * t_pub_g), m(len(gate_rows) * t_pub_g)],
      ["(v) generation, first 100 requests: deployed, pre-edit, default method", m(2 * t_gen_w + t_gen_g), m(2 * t_gen_w + t_gen_g)],
      [f"(vi) retention, first 100 requests at 4 values of T: deployed + {', '.join(ret_gates)}", m(t_ret), m(t_ret_B)],
      [f"TOTAL GPU, behaviour of the {len(dev_m)} rows at one calibration draw (without gate level and key cache)", m(A), m(Bu)],
      [f"each further calibration draw of the {len(cal_rows)} calibrated rows ((i) - (iii))", m(dA), m(dB)]]
for ln in T(["part", "A minutes", "B minutes"], r).splitlines():
    w("  " + ln)
w(f"  The {len(dev_m)} rows: {', '.join(dev_m)}.")
w(f"  In hours: behaviour A {A / 3600:.1f} h, B {Bu / 3600:.1f} h of one RTX 5090 at the measured (shared-GPU) rates; gate level A {t_gl['cuda'] / 3600:.1f} h, B {t_gl_B['cuda'] / 3600:.1f} h.")
w("  The T = 500 log costs about a quarter of the parts that scale with the number of requests (the HELD-OUT prompts and the generation do not).")
w(f"  Gate level: the features of {n_sets} detector sets ('ncoord', 'cos', one null-repaired set per draw) over all cached positions take {n_sets * feat['cuda'] / 60:.0f} min on the GPU")
w(f"  ({n_sets * feat['cpu'] / 60:.0f} min on the CPU); the rest is the per-position refinement of the firing queries ({1000 * t_dec_q:.1f} ms each at T = {TT}, single-threaded Python).")
w(f"  RAM: one (texts x T) fp64 matrix per cached feature, {(ntext['stream'] + ntext['heldout']) * TT * 8 / 2 ** 30:.1f} GiB each at T = {TT} (e10t.gate keeps at most 4).")
w("  The pre-edit row's outcomes are shared by all logs (store namespace of W_0); outcomes of a gate are keyed by its signature, so another draw or another T")
w("  is computed anew.")
cost_txt = "\n".join(lines)

import textwrap  # noqa: E402

und = "\n".join(textwrap.fill(f"{k}: {v}", 150, initial_indent="  ", subsequent_indent="      ") for k, v in e10t.methods.UNDEFINED.items())
gg = tests["gguard"]["detail"]
nr = beh["records"]["rescope_null"]["null_repair"]
pre = dev_m["preedit"]
c = tests["cache"]["detail"]
bt = tests["batch"]["detail"]
rep = {"RENDERED": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), "E10T": e10t.E10T, "PYTHON": e10t.PYTHON, "GQ": e10t.GQ, "MEM": str(MEM),
       "MAX_TOKENS": str(BB.MAX_TOKENS), "ROWSET": ", ".join(e10t.methods.ROWSET), "NULL_KEYS": str(nr["n_keys"]), "NULL_RANK": str(nr["rank_benign"]),
       "F16": f"{c['f16_max_dr_over_scale']:.1e}", "LIVE": f"{max(c['tf_max_dr_over_scale'], c['dec_max_dr_over_scale']):.1e}", "BAND": f"{e10t.BAND:.0e}",
       "UNDEFINED": und, "PRE_D1": f"{max(pre['heldout']['D1_tf']['value'], pre['heldout']['D1_dec']['value'], pre['neighborhood']['D1_tf']['value'], pre['neighborhood']['D1_dec']['value']):.1f}",
       "PRE_N": str(pre["heldout"]["n"] + pre["neighborhood"]["n"]),
       "BATCH_LOGIT": f"{max(max(v['max_abs_logit_diff_tf'], v['max_abs_logit_diff_dec']) for v in bt.values()):.0e}", "BATCH_ROWS": str(sum(v["n"] for v in bt.values())),
       "TARGET_ERR": f"{tests['target']['detail']['rebased']['max_rel_err']:.1e}", "TESTS": tests_txt, "FINGERPRINT": fp_txt, "COST": cost_txt,
       "GUARD_N": str(tests["cguard"]["detail"]["rescope"]["n_lowered"]),
       "GUARD_REL": f"{max(v.get('max_relative_lowering', 0.0) for v in tests['cguard']['detail'].values()):.1e}",
       "GUARD_CASES": str(gg["n_cases"]),
       "GUARD_FAIL": str(gg["n_cases"] - min(gg["library_thresholds_only"]["tf_own_candidate"], gg["library_thresholds_only"]["dec_own_candidate"]))}
txt = open(os.path.join(ROOT, "USAGE.template.txt")).read()
for k, v in rep.items():
    txt = txt.replace("{{" + k + "}}", v)
if "{{" in txt:
    raise SystemExit("unfilled placeholder: " + txt[txt.index("{{"):txt.index("{{") + 40])
LAB = {"setup": "set-up (shared by the examples below)", "i": "metric (i)", "ii": "metric (ii)", "iii": "metric (iii)", "iv": "metric (iv)", "v": "metric (v)",
       "vi": "metric (vi)"}
out_lines = []
for ln in txt.splitlines():
    if ln.startswith("#BEGIN"):
        out_lines.append(f"# ---------------- {LAB[ln.split()[1]]}")
    elif not ln.startswith("#END"):
        out_lines.append(ln)
txt = "\n".join(out_lines) + "\n"
with open(os.path.join(ROOT, "USAGE.txt"), "w") as f:
    f.write(txt)
e10t.data.jdump(fx.io.clean({"T": TT, "prompts": n, "positions": npos, "texts": ntext, "fraction_benign_like_run_dev": frac, "seconds": {
    "key_cache": t_cache, "gate_level_A": t_gl, "gate_level_B": t_gl_B, "weight_row": t_weight, "gate_row_A": t_gate_A, "gate_row_B": t_gate_B, "public_weight_row": t_pub_w, "public_gate_row": t_pub_g,
    "generation_weight_row": t_gen_w, "generation_gate_row": t_gen_g, "retention_A": t_ret, "retention_B": t_ret_B, "total_A": A, "total_B": Bu, "further_draw_A": dA,
    "further_draw_B": dB}, "rows": list(dev_m), "note": "A: the share of benign / neighbourhood prompts that ran on the DEVELOPMENT log; B: every prompt runs (upper bound)",
    "mem_MiB": MEM, "peak_alloc_MiB": peak, "code": now}), os.path.join(e10t.OUT, "cost_projection_T2000.json"))
print(cost_txt)
print("written", os.path.join(ROOT, "USAGE.txt"))
