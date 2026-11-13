"""Render out/dev_table.txt (or out/<tag>_table.txt) from the JSON written by scripts/run_rowset.py.  No number is typed here.

    python scripts/render_table.py [--tag dev] [--out <file>]
"""
import argparse
import hashlib
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "code"))
import e10t  # noqa: E402
import fx  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="dev")
ap.add_argument("--out", default=None)
args = ap.parse_args()
D = os.path.join(e10t.OUT, args.tag, "parts")
dst = args.out or os.path.join(e10t.OUT, f"{args.tag}_table.txt")


def load(name):
    p = os.path.join(D, name)
    return e10t.data.jload(p) if os.path.exists(p) else None


def f(v, nd=1):
    if v is None:
        return "-"
    if isinstance(v, dict):
        v = v.get("value")
        if v is None:
            return "-"
    if isinstance(v, str):
        return v
    return f"{v:.{nd}f}"


def fci(s, nd=1):
    if s is None or s.get("value") is None:
        return "-"
    if s.get("ci_lo") is None:
        return f"{s['value']:.{nd}f}"
    return f"{s['value']:.{nd}f} [{s['ci_lo']:.{nd}f}, {s['ci_hi']:.{nd}f}]"


def fd(s, nd=1):
    """mean [p05, p95] over draws."""
    if s is None or s.get("mean") is None:
        return "-"
    if s["n_draws"] == 1:
        return f"{s['mean']:.{nd}f}"
    return f"{s['mean']:.{nd}f} [{s['p05']:.{nd}f}, {s['p95']:.{nd}f}]"


T = fx.io.text_table
gate, beh, pub, gen, ret = load("gate.json"), load("behave.json"), load("public.json"), load("gen.json"), load("retention.json")
meta = next((x["meta"] for x in (beh, gate, pub, gen, ret) if x is not None), None)
if meta is None:
    raise SystemExit(f"no part file under {D}")
order = [x for x in e10t.methods.ROWSET_R14]
des = meta["describe"]
is_dev = bool(des["development"])
lines = []
w = lines.append
w(f"E10 ROW SET ON {meta['log']} (ROME, public code, GPT-2 XL, CounterFact; T = {des['T']})" + ("  --  MECHANICS RUN ON THE DEVELOPMENT LOG, NOT A RESULT" if is_dev else ""))
w(f"rendered {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())} by scripts/render_table.py from {os.path.relpath(D, e10t.E10T)}/*.json")
codes = {k: (x["meta"]["code"]["e10t"] if x is not None else "-") for k, x in (("gate", gate), ("behave", beh), ("public", pub), ("gen", gen), ("retention", ret))}
now = e10t.code_fingerprint()
w(f"code now: e10t {now['e10t']}, rescope {now['rescope']}, fx {now['fx']}, tx_public {now['tx_public']}; e10t code of each part: " + ", ".join(f"{k} {v}" for k, v in codes.items())
  + ("" if all(v in (now["e10t"], "-") for v in codes.values()) else "   (!! a part was produced by another code version)"))
w(f"role set: all positions of the prompt and of the {e10t.N_DEC}-token continuation; calibration: alpha = {meta['alpha']}, n = {e10t.N_CAL} benign prompts of the REPAIR pool, "
  f"behaviour at draw {meta['draw']}; all rates in percent")
w(f"prompts: {des['n_request']} request, {des['n_paraphrase']} paraphrase, {des['n_neighborhood']} neighbourhood ({des['n_neighborhood_clean']} clean); "
  f"HELD-OUT benign {des['n_heldout']} ({des['n_heldout_clean']} clean), REPAIR benign {des['n_repair']}")
w("")

if beh is not None:
    w(f"1. THE ROWS (calibration draw {meta['draw']})")
    rows = []
    for lab in order:
        r = beh["records"].get(lab)
        if r is None:
            continue
        c, o, g = r.get("cal", {}), r.get("own_requests") or {}, r.get("gate") or {}
        th = r.get("thresholds") or {}
        nr = r.get("null_repair") or {}
        rows.append([lab, g.get("feature") or "-", f"{g.get('fire', '-')}/{g.get('response', '-')}" if g else "-", r.get("calibration", "-"),
                     f(c.get("eta"), 4), f(th.get("min"), 3), f(th.get("median"), 3), f(th.get("max"), 3), f(c.get("bound_false_firing"), 4),
                     "-" if c.get("n_edits_capped") is None else str(c["n_edits_capped"]), "-" if not g else str(g.get("n_guards")),
                     "-" if not o else str(o["n_not_candidate"]), "-" if not o else str(o["n_not_released"]), "-" if not o else str(o["n_other_released"]),
                     f(nr.get("detector_norm_inflation", {}).get("median"), 2) if nr else "-", r.get("fingerprint", "-")])
    w(T(["method", "feature", "fire/resp", "calibration", "eta", "thr min", "median", "max", "bound", "capped", "guards", "own !cand", "own !rel", "other rel", "|w'|/|w|", "key"], rows))
    w("   The last two rows are the ReScope-W rows that amendment R14 registers for E10 (thresholds as the only repair variable: all positions, every candidate")
    w("   releases its original log term <W_i, k> U_i).  capped: edits whose threshold the guards lowered (guarantee void for them); own !cand / !rel: requests")
    w("   whose own edit is no candidate / is not released anywhere in the own request prompt (key cache); other rel: requests on which another edit is released")
    w("   (fire 'all': is a candidate) at some position; |w'|/|w|: median norm inflation of the detectors by the null repair.")
    w("")
    w("2. BEHAVIOUR")
    w("   (i) new object on request / paraphrase prompts: greedy decode, exact and token-level match [95 % bootstrap interval over edits]")
    rows = []
    for lab in order:
        m = beh["methods"].get(lab)
        if m is None:
            continue
        e = m["edits"]
        rq, pa = e["request"], e["paraphrase"]
        rows.append([lab, str(rq["n"]), fci(rq["exact"]), f(rq["token"]), f(rq["true_exact"]), f(rq["own_released"]), str(pa["n"]), fci(pa["exact"]), f(pa["token"]),
                     f(pa["true_exact"]), f(pa["fired"]), fci(pa["own_released"]), f(pa["other_released"]), f(pa["exact_given_own_released"]),
                     str(rq["n_nonfinite"] + pa["n_nonfinite"])])
    w(T(["method", "req n", "req exact", "tok", "true", "own rel", "par n", "par exact", "tok", "true", "fired", "own released", "other", "exact|rel", "non-fin"], rows))
    mt = sum(beh["methods"][lab]["edits"][k]["n_multi_token_objects"] for lab in beh["methods"] for k in ("request", "paraphrase"))
    w(f"   true: the decode starts with the TRUE object.  own released: the prompt's own edit is released at some position of the decode (coverage).  Objects with")
    w(f"   more than one token in this log: {mt} (rows x methods): exact = token-level and the decode test is a first-token test here.")
    w("")
if pub is not None:
    w("   (ii) public scores (public eval_utils_counterfact + summarize.py on the served model; probabilities of the new vs the true object; not comparable with (i))")
    rows = []
    for lab in order:
        s = pub["methods"].get(lab)
        if s is None:
            continue
        rf = s.get("rows_fired_pct") or {}
        rows.append([lab, str(s["n"]), f(s["ES"]), f(s["PS"]), f(s["NS"]), f(s["EM"]), f(s["PM"]), f(s["NM"]), f(s["S"]), f(rf.get("rewrite")), f(rf.get("paraphrase")),
                     f(rf.get("neighborhood")), f(rf.get("attribute")), str(s["n_records_with_nonfinite_prob"])])
    w(T(["method", "n rec", "ES", "PS", "NS", "EM", "PM", "NM", "S", "fire rew", "fire par", "fire nbr", "fire attr", "non-fin"], rows))
    w("   fire ...: share of the public batch's rows (prompt + object) of each prompt group on which the gate released an edit.")
    w("")
if beh is not None:
    w("   (iii) D1: the pre-edit greedy 10-token continuation is not reproduced (tf = teacher-forced, dec = greedy decode) [95 % bootstrap interval over records]")
    for part, title in (("heldout", "HELD-OUT benign prompts"), ("neighborhood", "neighbourhood prompts of the logged requests")):
        rows = []
        for lab in order:
            m = beh["methods"].get(lab)
            if m is None:
                continue
            p = m[part]
            cl, nr = p.get("clean", {}), p.get("near_scope", {})
            row = [lab, str(p["n"]), fci(p["D1_tf"]), f(p.get("D1_dec")), str(p.get("n_D1_tf_differs_from_dec", "-")), f(p["tok_loc"]), fci(p["fired"]) if p.get("fired") else "-",
                   f(p.get("D1_tf_given_fired")), str(cl.get("n", "-")), f(cl.get("D1_tf")), f(cl.get("fired")), str(nr.get("n", "-")), f(nr.get("D1_tf")), f(nr.get("fired")),
                   str(p["n_from_cache"]), str(p["n_nonfinite"]), "-" if p.get("n_fired_last_only") is None else str(p["n_fired_last_only"])]
            if part == "neighborhood":
                row += [f(p.get("own_released")), f(p.get("other_released"))]
            rows.append(row)
        hd = ["method", "n", "D1 tf", "D1 dec", "tf!=dec", "tok-loc", "fired", "D1|fired", "clean n", "D1", "fired", "near n", "D1", "fired", "no fwd", "non-fin", "last-only"]
        w(f"   {title}")
        w(T(hd + (["own rel", "other rel"] if part == "neighborhood" else []), rows))
    w("   tok-loc: share of the pre-edit tokens kept (tf).  near: prompts that share a subject string with / mention a subject of / equal a prompt of a logged request.")
    w("   no fwd: rows on which nothing fires: the cached pre-edit continuation, no forward pass.  last-only: rows whose only release is at the tenth continuation")
    w("   token (fired, but the ten measured tokens cannot change; top-1 gates).  own rel / other rel: the neighbourhood prompt's own edit / another edit is released.")
    w("")
if gen is not None:
    w("   (v) degeneration: n-gram entropy of the public generations (GE x100; generate_fast, top-5 sampling, 100 tokens; generation prompts of the first 100 requests)")
    rows = []
    for lab in ("preedit", "deployed", "rescope"):
        g = gen["methods"].get(lab)
        if g is None:
            continue
        rows.append([lab, str(g["n_records"]), str(g["n_texts"]), f(g["GE"]), f(g["GE_se"]), f(g.get("pct_below_ref_q01")), f(g["per_text_q05"]), f(g["per_text_min"]),
                     f(g.get("prompts_fired_pct")), f(g.get("mean_fire_positions_per_text"), 2), str(g["n_failed"])])
    w(T(["method", "records", "texts", "GE", "se", "%<q01", "text q05", "text min", "prompts fired", "fire pos/text", "failed"], rows))
    w("   %<q01: share of texts whose entropy is below the 1st percentile of the pre-edit model's texts on the same prompts.")
    w("")
if gate is not None:
    w(f"3. GATE LEVEL over {gate['n_draws']} calibration draws (iv): mean [5th, 95th percentile over draws]; from the key cache, no model")
    s = gate["summary"]
    rows = []
    for lab in order:
        g = s.get(lab)
        if g is None:
            continue
        rows.append([lab, fd(g.get("heldout.fired"), 2), fd(g.get("heldout.fired_clean"), 2), fd(g.get("heldout.fired_near_scope")), fd(g.get("heldout.fired_last_only"), 2),
                     fd(g.get("paraphrase.own_released")),
                     fd(g.get("paraphrase.fired")), fd(g.get("neighborhood.fired"), 2), fd(g.get("neighborhood.own_released"), 2), fd(g.get("request.own_released")),
                     fd(g.get("request.other_released")), fd(g.get("cal.eta"), 3), fd(g.get("cal.n_edits_capped"), 1)])
    w(T(["method", "false firing HO", "HO clean", "HO near", "HO last-only", "coverage paraphrase", "par fired", "firing NB", "NB own rel", "req own rel", "req other", "eta", "capped"], rows))
    n = gate["n"]
    w(f"   false firing HO: HELD-OUT benign prompts on which an edit is released at some position (n = {n['heldout']}); coverage: paraphrase prompts whose own edit is")
    w(f"   released (n = {n['paraphrase']}); firing NB: neighbourhood prompts on which an edit is released (n = {n['neighborhood']}).  Bound of the calibrated rows: alpha.")
    w("   HO last-only: HELD-OUT prompts whose only release is at the tenth continuation token (counted as firing; it cannot change the ten measured tokens).")
    w("   where_* rows (fire 'all'): 'own rel' = the own edit is a candidate; every candidate edit releases its log term.")
    ib = {lab: sum(int(v) for part in ("heldout", "paraphrase", "neighborhood", "request") for v in [gate['per_draw'][lab][d][part]['near_band'] for d in gate['per_draw'][lab]])
          for lab in gate["per_draw"]}
    w(f"   prompts inside the decision band (decided by the hook, not by the cache), summed over draws and row sets: " + ", ".join(f"{k} {v}" for k, v in ib.items()))
    w("")
if ret is not None:
    w(f"4. RETENTION (vi): the first {ret['first']} requests as T grows" + (" (prefixes of the development log; the registered read-out uses the first 100 requests of the stream)" if is_dev else ""))
    rows = []
    for Tn in sorted(ret["by_T"], key=int):
        for lab, o in ret["by_T"][Tn].items():
            e, nb = o["edits"], o["neighborhood"]
            pb = o.get("public") or {}
            rows.append([str(Tn), lab, f(e["request"]["exact"]), f(e["paraphrase"]["exact"]), f(e["paraphrase"]["own_released"]), f(pb.get("ES")), f(pb.get("PS")),
                         f(pb.get("NS")), f(nb["D1_tf"]), f(nb.get("fired")),
                         "-" if not o.get("cal") else f(o["cal"]["eta"], 3), "-" if not o.get("cal") else str(o["cal"]["n_edits_capped"])])
    w(T(["T", "method", "req exact", "par exact", "par own rel", "ES", "PS", "NS", "NB D1 tf", "NB fired", "eta", "capped"], rows))
    w("")
if beh is not None:
    w("5. COST of the behavioural part (this run; the GPU is shared, timings are load-dependent)")
    rows = []
    for lab in order:
        m = beh["methods"].get(lab)
        if m is None:
            continue
        c = m["cost"]
        rows.append([lab, str(c["n_prompts"]), str(c["n_passes"]), str(c["sequences_run"]), str(c["silent"]), str(c["from_store"]), str(c["batches"]), f(c["seconds_model"]),
                     f(1000.0 * c["seconds_model"] / c["sequences_run"], 1) if c["sequences_run"] else "-", str(c["near"]), str(c["hook_mismatch"]),
                     str(c.get("hook_mismatch_dec", "-"))])
    w(T(["method", "prompts", "passes", "run", "silent", "store", "batches", "model s", "ms/pass", "in band", "hook!=cache tf", "dec"], rows))
    w("   passes = prompts x 2 (teacher-forced + decode).  run: passes through the model; silent: nothing fires, cached pre-edit continuation; store: computed earlier.")
    if pub is not None:
        w("   public scores: " + ", ".join(f"{lab} {pub['methods'][lab]['seconds']:.0f}s/{pub['methods'][lab]['n_new']} rec" for lab in order if lab in pub["methods"]))
    if gen is not None:
        w("   generation: " + ", ".join(f"{lab} {g['seconds']:.0f}s" for lab, g in gen["methods"].items()))
    if gate is not None:
        w(f"   gate level: {gate['seconds']:.0f}s for {gate['n_draws']} draws (CPU, 4 threads)")
    w("")
# cross-check of the two weight rows against the build's own description of the same log (other code: tp_eval, unpadded batches)
bj = os.path.join(e10t.TXOUT, "text_logs.json")
if beh is not None and os.path.exists(bj) and meta["log"] in e10t.data.jload(bj)["logs"]:
    bl = e10t.data.jload(bj)["logs"][meta["log"]]
    w("6. CROSS-CHECK: the deployed and the pre-edit row of this table against the build's description of the same log (09_testbed_scale/text_public/out/text_logs.json)")
    rows = []
    for lab, bk in (("deployed", "deployed"), ("preedit", "pre_edit")):
        m, b = beh["methods"].get(lab), bl["models"].get(bk)
        if m is None or b is None:
            continue
        mine = [m["edits"]["request"]["exact"]["value"], m["edits"]["paraphrase"]["exact"]["value"], m["heldout"]["D1_tf"]["value"], m["heldout"]["D1_dec"]["value"],
                m["heldout"]["tok_loc"]["value"], m["neighborhood"]["D1_tf"]["value"], m["neighborhood"]["tok_loc"]["value"]]
        ref = [b["edits"]["request"]["dec_exact"], b["edits"]["paraphrase"]["dec_exact"], b["heldout"]["all"]["D1_tf"], b["heldout"]["all"]["D1_dec"],
               b["heldout"]["all"]["tok_loc"], b["neighborhood"]["all"]["D1_tf"], b["neighborhood"]["all"]["tok_loc"]]
        if pub is not None and lab in pub["methods"]:
            mine += [pub["methods"][lab][k] for k in ("ES", "PS", "NS")]
            ref += [bl["public"]["all"][bk][k] for k in ("ES", "PS", "NS")]
        else:
            mine += [None] * 3
            ref += [None] * 3
        g, bg = (gen or {}).get("methods", {}).get(lab), (b.get("generation") or {}).get("first100")
        mine.append(g["GE"] if g else None)
        ref.append(bg["GE"] if bg else None)
        rows.append([lab, "this table"] + [f(v, 2) for v in mine])
        rows.append(["", "build"] + [f(v, 2) for v in ref])
        rows.append(["", "max |diff|", f(max(abs(a - c) for a, c in zip(mine, ref) if a is not None and c is not None), 4)] + [""] * (len(mine) - 1))
    w(T(["row", "source", "req exact", "par exact", "HO D1 tf", "HO D1 dec", "HO tok-loc", "NB D1 tf", "NB tok-loc", "ES", "PS", "NS", "GE"], rows))
    w("")
w("7. ROWS OF THE DRIVER'S ROW SET THAT ARE NOT IN THIS TABLE")
for k, v in meta["undefined_rows"].items():
    w(f"   {k}: {v}")
txt = "\n".join(lines) + "\n"
tmp = dst + f".tmp{os.getpid()}"
with open(tmp, "w") as fh:
    fh.write(txt)
os.replace(tmp, dst)
print(txt)
print("written", dst, hashlib.sha1(txt.encode()).hexdigest()[:12])
