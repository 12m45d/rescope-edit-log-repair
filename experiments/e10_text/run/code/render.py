"""E10: script-rendered tables.  python render.py  ->  out/table.txt, out/results.json, out/paper_numbers.json, out/DEVIATIONS.txt

Reads only JSON written by the experiment code (out/summary/*.json by summarise.py, out/gate, out/r22, out/equal, out/stream,
out/cert, out/gen, out/screening.json).  No number is typed here.
"""
import os as _go, sys as _gs  # ARTIFACT: a failed run leaves the stored output files untouched (rescope_paths.guard_outputs)
_gd = _go.environ.get("RESCOPE_HOME") or _go.path.dirname(_go.path.abspath(__file__))
while not _go.path.exists(_go.path.join(_gd, "rescope_paths.py")) and _go.path.dirname(_gd) != _gd: _gd = _go.path.dirname(_gd)
_gs.path.insert(0, _gd); import rescope_paths as _RPG; _RPG.guard_outputs(__file__, __name__)
import os
import time

import numpy as np

import xcommon as X

SUM = os.path.join(X.OUT, "summary")
S = X.SPLIT
PN = {}                 # paper numbers: name -> {"value": v, "file": f, "key": k}
LINES = []
NICE = {"deployed": "deployed log", "preedit": "pre-edit model", "where_logonly": "ReScope-W log-only (repair)", "where_twoblock": "ReScope-W two-block (repair)",
        "rescope": "ReScope (migration)", "cosine": "cosine, same calibration", "cosine_published": "cosine 0.65, all positions", "exact": "exact match",
        "lookup_subject": "lookup_subject [other class]"}
R3 = ("cos_centred", "cos_null", "white_0.01", "white_0.1", "white_1", "whitecos_0.01", "whitecos_0.1", "whitecos_1")


def P(*a):
    LINES.append(" ".join(str(x) for x in a))


def jl(p):
    return X.jload(p) if os.path.exists(p) else None


def f1(v, nd=1):
    if v is None:
        return "-"
    if isinstance(v, str):
        return v
    return f"{v:.{nd}f}"


def ci(st, nd=1):
    if not st or st.get("value") is None:
        return "-"
    if st.get("ci_lo") is None:
        return f1(st["value"], nd)
    return f"{st['value']:.{nd}f} [{st['ci_lo']:.{nd}f}, {st['ci_hi']:.{nd}f}]"


def dr(s, nd=1):
    """mean [p05, p95] over draws"""
    if not s or s.get("mean") is None:
        return "-"
    if s["n_draws"] == 1:
        return f1(s["mean"], nd)
    return f"{s['mean']:.{nd}f} [{s['p05']:.{nd}f}, {s['p95']:.{nd}f}]"


def table(headers, rows, indent=3):
    rows = [[("-" if c is None else str(c)) for c in r] for r in rows]
    w = [max(len(str(h)), *(len(r[i]) for r in rows)) if rows else len(str(h)) for i, h in enumerate(headers)]
    pad = " " * indent
    P(pad + "  ".join(str(h).ljust(w[i]) if i == 0 else str(h).rjust(w[i]) for i, h in enumerate(headers)))
    P(pad + "  ".join("-" * x for x in w))
    for r in rows:
        P(pad + "  ".join(r[i].ljust(w[i]) if i == 0 else r[i].rjust(w[i]) for i in range(len(headers))))


def pn(name, value, file, key):
    PN[name] = {"value": value, "file": os.path.relpath(file, X.RUN), "key": key}


def short(log):
    return log.replace("TX-G2XL-rome-", "")


def nice(lab, best):
    base, off = lab.split("@")[0], lab.endswith("@goff")
    n = NICE.get(base, base + (" (best R3 gate)" if base == best else ""))
    return n + (" / guards off" if off else "")


# ============================================================================= header
best = (jl(X.OUT + "/best_gate_text.json") or {}).get("selected")
aud = jl(X.AUDIT + "/verdicts.json")
P("E10: THE FROZEN METHODS ON THE PUBLIC ROME LOGS OF GPT-2 XL (CounterFact)  --  PROTOCOL.md section 6: E10, R14, R19 (f), R21, R22")
P(f"rendered {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())} by code/render.py from JSON under out/ (every number is written by code)")
P("alpha = 0.05, n = 200 benign calibration prompts of the REPAIR pool; role set = all positions of the prompt and of the 10-token continuation; all rates in percent")
P("intervals: crossed bootstrap over calibration draws (or complaint seeds) and records / edits, 2000 resamples, 95 % percentile (xcommon.xboot; tests: out/tests_boot.json);")
P("           gate level: mean [5th, 95th percentile over the 20 calibration draws], and the crossed-bootstrap interval in the columns marked xb")
P("")
P("0. LOGS (independent audit: 09_testbed_scale/text_public/audit/verdicts.json)")
if aud:
    rows = []
    for lg in (X.DEV,) + X.LOGS:
        a = aud["per_log"][lg]
        d = a["deployed"]
        use = {X.DEV: "development log: selection of the R3 gate, tests", "TX-G2XL-rome-T100": "behaviour, R22, equal evidence, gate level, screening",
               "TX-G2XL-rome-T500": "behaviour (guards on / off; split at 173), R22, equal evidence, gate level, screening"}.get(lg, "gate level, screening, stream description")
        rows.append([lg, a["T"], a["verdict"], a["class"], "/".join(f1(x) for x in d["ES_PS_NS"]), f1(d["D1_heldout_pct"]), f1(d["tokens_kept_pct"]), use])
        pn("Audit" + short(lg).replace("100", "Hundred").replace("dev", "Dev") + "Verdict", a["verdict"], X.AUDIT + "/verdicts.json", f"per_log.{lg}.verdict")
    table(["log", "T", "audit verdict", "class", "deployed ES/PS/NS", "D1", "kept", "used here for"], rows)
    P(f"   all checks of the audit pass: {aud['all_checks_pass']}.  Split position of the collapse logs: {S} (out/split.json; fixed by R21 b).")
P(f"   R3 gate carried into the behavioural rows (selected on the development log before any behavioural run; out/best_gate_text.json): {best}")
bg = jl(X.OUT + "/best_gate_text.json")
if bg:
    P("   coverage of paraphrases at gate level on the development log (mean over 20 draws): " + ", ".join(f"{k} {bg['candidates'][k]['coverage_paraphrase']:.2f}" for k in bg["ranking"]))
    pn("BestGateText", best, X.OUT + "/best_gate_text.json", "selected")
P("")

# ============================================================================= 1. main table
GATE = {lg: jl(os.path.join(SUM, f"gate_{lg}.json")) for lg in (X.DEV,) + X.LOGS}
BEH = {lg: jl(os.path.join(SUM, f"behave_{lg}.json")) for lg in X.BEHAVE_LOGS}
ORDER = ["deployed", "preedit", "where_logonly", "where_twoblock", "rescope", "cosine", best, "cosine_published", "exact", "lookup_subject"]


def m(o, name):
    return (o or {}).get("metrics", {}).get(name)


def pub(o, k, seg="all"):
    e = (o or {}).get("public", {}).get(f"{seg}.{k}")
    return None if e is None else e


def gl(g, lab, name):
    return ((g or {}).get("rows", {}).get(lab) or {}).get(name)


def main_table(lg, labs, seg=""):
    b, g = BEH[lg], GATE[lg]
    sp = "" if not seg else "." + seg
    rows = []
    for lab in labs:
        o = b.get(lab)
        if o is None:
            continue
        ps = seg or "all"
        ff = gl(g, lab, "heldout.fired")
        cov = gl(g, lab, f"paraphrase{sp}.own_released")
        cap = gl(g, lab, "cal.n_edits_capped")
        rows.append([nice(lab, best), o["n_draws"], ci(m(o, f"edit.request.exact{sp}")), ci(m(o, f"edit.paraphrase.exact{sp}")),
                     ci(pub(o, "ES", ps)), ci(pub(o, "PS", ps)), ci(pub(o, "NS", ps)),
                     ci(m(o, "heldout.D1_tf")), f1((m(o, "heldout.kept_tf") or {}).get("value")), f1((m(o, "heldout.first_changed_tf") or {}).get("value")),
                     ci(m(o, f"neighborhood{sp}.D1_tf")), f1((m(o, f"neighborhood{sp}.kept_tf") or {}).get("value")),
                     dr(ff, 2) if ff else "-", dr(cov) if cov else "-", f1(cap["mean"]) if cap else "-"])
    table(["row", "draws", "request exact", "paraphrase exact", "ES", "PS", "NS", "D1 HELD-OUT (tf)", "kept", "1st chg", "D1 neighbourhood (tf)", "kept", "false firing", "coverage", "capped"], rows)


P("1. MAIN TABLE (behaviour: calibration draws 0-2 for calibrated rows; false firing / coverage / capped: gate level over 20 draws)")
P("   request / paraphrase exact: greedy decode starts with the new object (metric i).  ES PS NS: public scores (public probability test + public summariser; metric ii;")
P("   never comparable with the exact-match columns).  D1: the pre-edit 10-token continuation is not reproduced (teacher-forced); kept: share of pre-edit tokens kept;")
P("   1st chg: share of changed first tokens.  false firing: HELD-OUT benign prompts on which an edit is released; coverage: paraphrase prompts whose own edit is released;")
P("   capped: edits whose threshold the guards lowered (mean over draws; the calibration bound is void for them).")
for lg in X.BEHAVE_LOGS:
    if not BEH.get(lg):
        continue
    collapse = int(short(lg)[1:]) > S
    P("")
    P(f"   {lg}" + ("  (collapse log; rows with guards = the frozen default)" if collapse else "  (the clean reported log)"))
    main_table(lg, ORDER)
    if collapse:
        off = [x + "@goff" for x in ("where_logonly", "where_twoblock", "rescope", "cosine", best)]
        P(f"   {lg}, GUARDS OFF (R21 b: an update whose own request is not separated from benign prompts is silenced on its own request)")
        main_table(lg, off)
        for seg, txt in (("before", f"requests logged BEFORE position {S} (0..{S - 1}); edit and neighbourhood columns restricted to them"),
                         ("from", f"requests logged FROM position {S} on; edit and neighbourhood columns restricted to them (targets computed on collapsed weights)")):
            P(f"   {lg}, {txt}")
            main_table(lg, ORDER + off, seg)

# paper numbers of the main table
for lg in X.BEHAVE_LOGS:
    b = BEH.get(lg)
    if not b:
        continue
    f = os.path.join(SUM, f"behave_{lg}.json")
    tag = {"TX-G2XL-rome-T100": "TOne", "TX-G2XL-rome-T500": "TFive"}[lg]
    names = {"deployed": "Deployed", "preedit": "Preedit", "where_logonly": "RepairLogonly", "where_twoblock": "RepairTwoblock", "rescope": "Migration", "cosine": "Cosine",
             best: "BestGate", "cosine_published": "CosinePublished", "exact": "Exact", "lookup_subject": "Lookup"}
    for lab, nm in names.items():
        for sfx, snm in (("", ""), ("@goff", "GuardsOff")):
            o = b.get(lab + sfx)
            if o is None:
                continue
            for met, mn in (("edit.request.exact", "ReqExact"), ("edit.paraphrase.exact", "ParExact"), ("heldout.D1_tf", "DOneHeldout"), ("heldout.kept_tf", "KeptHeldout"),
                            ("heldout.first_changed_tf", "FirstChangedHeldout"), ("neighborhood.D1_tf", "DOneNeighbourhood"), ("edit.request.exact.before", "ReqExactBefore"),
                            ("edit.paraphrase.exact.before", "ParExactBefore"), ("edit.request.exact.from", "ReqExactFrom"), ("edit.paraphrase.exact.from", "ParExactFrom")):
                if m(o, met):
                    pn(f"Text{tag}{nm}{snm}{mn}", round(m(o, met)["value"], 2), f, f"{lab + sfx}.metrics.{met}.value")
            for k in ("ES", "PS", "NS"):
                for seg, sn in (("all", ""), ("before", "Before"), ("from", "From")):
                    if pub(o, k, seg):
                        pn(f"Text{tag}{nm}{snm}{k}{sn}", round(pub(o, k, seg)["value"], 2), f, f"{lab + sfx}.public.{seg}.{k}.value")
    g = GATE.get(lg)
    if g:
        fg = os.path.join(SUM, f"gate_{lg}.json")
        for lab, nm in names.items():
            for sfx, snm in (("", ""), ("@goff", "GuardsOff")):
                for met, mn in (("heldout.fired", "FalseFiring"), ("paraphrase.own_released", "Coverage"), ("neighborhood.fired", "FiringNeighbourhood"), ("cal.n_edits_capped", "Capped")):
                    e = gl(g, lab + sfx, met)
                    if e and e.get("mean") is not None:
                        pn(f"Text{tag}{nm}{snm}{mn}", round(e["mean"], 2), fg, f"rows.{lab + sfx}.{met}.mean")

# ============================================================================= 2. gate level
P("")
P("2. GATE LEVEL (metric iv) over the 20 calibration draws; from the key cache, no model.  mean [5th, 95th percentile over draws]; xb: crossed bootstrap over draws and records")
P("   FF: false firing on HELD-OUT benign prompts (bound of a calibrated row: alpha = 5, void for capped edits); clean / near: prompts without / with a subject or text shared")
P("   with a logged request; cov: paraphrase prompts whose own edit is released; NB: neighbourhood prompts on which an edit is released; req own / other: request prompts")
P("   whose own / another edit is released; capped: edits whose threshold the guards lowered; width: candidate edits per firing HELD-OUT prompt.")
for lg in (X.DEV,) + X.LOGS:
    g = GATE.get(lg)
    if not g:
        continue
    collapse = g["meta"]["T"] > S
    nn = g["meta"]["n"]
    P("")
    P(f"   {lg}: {nn['heldout']} HELD-OUT, {nn['paraphrase']} paraphrase, {nn['neighborhood']} neighbourhood, {nn['request']} request prompts" + ("  (development log: not a result)" if lg == X.DEV else ""))
    labs = ["rescope", "where_twoblock", "cosine"] + list(R3) + ["where_logonly", "cosine_published", "exact", "lookup_subject"]
    for sfx, title in (("", "guards on (frozen default)"), ("@goff", "guards off")):
        if sfx and not collapse:
            continue
        rows = []
        for lab in labs:
            e = g["rows"].get(lab + sfx)
            if e is None:
                continue
            r = [nice(lab, best).replace(" (repair)", "").replace(" (migration)", ""), dr(e.get("heldout.fired"), 2), ci(e.get("xboot.heldout.fired"), 2), dr(e.get("heldout.fired_clean"), 2),
                 dr(e.get("heldout.fired_near_scope")), dr(e.get("paraphrase.own_released")), ci(e.get("xboot.paraphrase.own_released")),
                 dr(e.get("neighborhood.fired"), 2), dr(e.get("request.own_released")), dr(e.get("request.other_released")),
                 dr(e.get("cal.n_edits_capped")), f1((e.get("cal.n_edits_capped") or {}).get("max"), 0), dr(e.get("heldout.lineage_width"), 2), dr(e.get("cal.eta"), 3)]
            if collapse:
                r += [dr(e.get("paraphrase.before.own_released")), dr(e.get("paraphrase.from.own_released")), dr(e.get("request.before.own_released")), dr(e.get("request.from.own_released"))]
            rows.append(r)
        hd = ["row", "FF HELD-OUT", "FF xb", "FF clean", "FF near", "cov paraphrase", "cov xb", "NB firing", "req own", "req other", "capped", "max", "width", "eta"]
        if collapse:
            hd += [f"cov <{S}", f"cov >={S}", f"req own <{S}", f"req own >={S}"]
        P(f"   {short(lg)}, {title}")
        table(hd, rows)
    cu = {lab: g["rows"][lab].get("capped_edits_union") for lab in ("rescope", "cosine", "where_logonly") if lab in g["rows"] and g["rows"][lab].get("n_capped_edits_union")}
    if cu:
        P("   capped edits (union over draws, guards on): " + "; ".join(f"{k}: {v[:30]}{' ...' if len(v) > 30 else ''}" for k, v in cu.items()))

# ============================================================================= 3. behaviour detail
P("")
P("3. BEHAVIOUR IN DETAIL (calibration draws 0-2; intervals: crossed bootstrap over draws and edits / records)")
for lg in X.BEHAVE_LOGS:
    b = BEH.get(lg)
    if not b:
        continue
    collapse = int(short(lg)[1:]) > S
    labs = [x for x in ORDER + [y + "@goff" for y in ORDER] if x in b]
    P("")
    P(f"   {lg}  (i) new object on request / paraphrase prompts (greedy decode); R7 read-outs against the deployed log: kept = exact under both, gained = exact only under the row,")
    P("   lost = exact only under the deployed log (percent of all prompts of the kind); own rel: the prompt's own edit is released in the decode")
    for seg in [""] + ([".before", ".from"] if collapse else []):
        if seg:
            P(f"   requests {'before' if seg == '.before' else 'from'} position {S}")
        rows = []
        for lab in labs:
            o = b[lab]
            g1 = lambda n: f1((m(o, n) or {}).get("value"))      # noqa: E731
            rows.append([nice(lab, best), ci(m(o, f"edit.request.exact{seg}")), g1(f"edit.request.token{seg}"), g1(f"edit.request.own_released{seg}"), g1(f"edit.request.kept{seg}"),
                         g1(f"edit.request.gained{seg}"), g1(f"edit.request.lost{seg}"), ci(m(o, f"edit.paraphrase.exact{seg}")), g1(f"edit.paraphrase.token{seg}"),
                         g1(f"edit.paraphrase.own_released{seg}"), g1(f"edit.paraphrase.other_released{seg}"), g1(f"edit.paraphrase.kept{seg}"), g1(f"edit.paraphrase.gained{seg}"),
                         g1(f"edit.paraphrase.lost{seg}"), g1(f"edit.paraphrase.true_exact{seg}")])
        table(["row", "request exact", "token", "own rel", "kept", "gained", "lost", "paraphrase exact", "token", "own rel", "other rel", "kept", "gained", "lost", "true obj"], rows)
    P(f"   {lg}  (ii) public scores (public eval code on the served model): success ES / PS / NS, probability differences EM / PM / NM, harmonic score S")
    for seg in ["all"] + (["before", "from"] if collapse else []):
        if seg != "all":
            P(f"   records {'before' if seg == 'before' else 'from'} position {S}")
        rows = []
        for lab in labs:
            o = b[lab]
            if not o.get("public"):
                continue
            agree = all((pub(o, k, seg) or {}).get("agrees_with_public_summariser", True) for k in ("ES", "PS", "NS"))
            rows.append([nice(lab, best), (pub(o, "ES", seg) or {}).get("n_records"), ci(pub(o, "ES", seg)), ci(pub(o, "PS", seg)), ci(pub(o, "NS", seg))] +
                        [f1((pub(o, k, seg) or {}).get("value")) for k in ("EM", "PM", "NM", "S")] + ["yes" if agree else "NO"])
        table(["row", "records", "ES", "PS", "NS", "EM", "PM", "NM", "S", "interval arithmetic = public"], rows)
    for part, title in (("heldout", "HELD-OUT benign prompts"), ("neighborhood", "neighbourhood prompts of the logged requests")):
        P(f"   {lg}  (iii) D1 on {title}: tf = teacher-forced, dec = greedy decode; kept = share of the pre-edit tokens kept; 1st = share of changed first tokens;")
        P("   clean / near: prompts without / with a subject or text shared with a logged request (reported separately)")
        subs = [""] + ([".before", ".from"] if collapse and part == "neighborhood" else [])
        for seg in subs:
            if seg:
                P(f"   neighbourhood prompts of the requests {'before' if seg == '.before' else 'from'} position {S}")
            rows = []
            for lab in labs:
                o = b[lab]
                g1 = lambda n: f1((m(o, n) or {}).get("value"))      # noqa: E731
                pre = f"{part}{seg}"
                r = [nice(lab, best), g1(pre + ".n"), ci(m(o, pre + ".D1_tf")), g1(pre + ".D1_dec"), ci(m(o, pre + ".kept_tf")), g1(pre + ".kept_dec"), ci(m(o, pre + ".first_changed_tf")),
                     ci(m(o, pre + ".fired"))]
                if not seg:
                    r += [g1(f"{part}.clean.n"), g1(f"{part}.clean.D1_tf"), g1(f"{part}.clean.kept_tf"), g1(f"{part}.clean.fired"),
                          g1(f"{part}.near_scope.n"), g1(f"{part}.near_scope.D1_tf"), g1(f"{part}.near_scope.kept_tf"), g1(f"{part}.near_scope.fired")]
                rows.append(r)
            hd = ["row", "n", "D1 tf", "D1 dec", "kept tf", "kept dec", "1st changed", "fired"]
            if not seg:
                hd += ["clean n", "D1", "kept", "fired", "near n", "D1", "kept", "fired"]
            table(hd, rows)
    P(f"   {lg}  calibration records and cost of the behavioural runs (summed over draws)")
    rows = []
    for lab in labs:
        o = b[lab]
        c = o["cal"]
        l2 = lambda v: "/".join("-" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x)) for x in v)      # noqa: E731
        rows.append([nice(lab, best), o["n_draws"], l2(c["eta"]), l2(c["n_edits_capped"]), l2(c["own_not_candidate"]), l2(c["own_not_released"]),
                     int(o["cost"]["sequences_run"]), int(o["cost"]["silent"]), int(o["cost"]["near"]), int(o["cost"]["hook_mismatch"]), int(o["cost"]["hook_mismatch_dec"]), f1(o["cost"]["seconds"], 0)])
    table(["row", "draws", "eta per draw", "capped", "own req not candidate", "not released", "passes run", "silent (no pass)", "in band", "hook!=cache tf", "dec", "seconds"], rows)

X.jdump({"lines": len(LINES)}, os.path.join(SUM, "_render_part1.json"))
exec(open(os.path.join(X.CODE, "render2.py")).read()) if os.path.exists(os.path.join(X.CODE, "render2.py")) else None

with open(os.path.join(X.OUT, "table.txt.tmp"), "w") as fh:
    fh.write("\n".join(LINES) + "\n")
os.replace(os.path.join(X.OUT, "table.txt.tmp"), os.path.join(X.OUT, "table.txt"))
X.jdump(dict(sorted(PN.items())), os.path.join(X.OUT, "paper_numbers.json"))                     # name -> {value, file, key}
X.jdump({k: v["value"] for k, v in sorted(PN.items())}, os.path.join(X.OUT, "paper_numbers_flat.json"))       # name -> value
recs = []
rd = os.path.join(X.OUT, "records")
for f in sorted(os.listdir(rd)) if os.path.isdir(rd) else []:
    recs += X.jload(os.path.join(rd, f))
X.jdump(recs, os.path.join(X.OUT, "results.json"), indent=None)
print(f"table.txt: {len(LINES)} lines; results.json: {len(recs)} records; paper_numbers.json: {len(PN)} numbers")
