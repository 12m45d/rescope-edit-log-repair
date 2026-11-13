#!/usr/bin/env python3
"""Task 1: the macros whose manifest entry reads one of the five fix-up directories (from MY COPIES of the manifest
and of vlm_macros.py), with file, key, stored value and the place where each is printed (body / report)."""
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP
import json, os, re, sys
V = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = f"{_RP.ROOT}"
PAPER = ROOT + "/01_paper/v4_rescope"
F = "02_research/experiments/10_final"
DIRS = ("e1_main/out/fixup", "e2_curves/out/fixup", "e3_qwen/out/fixup", "e3_llava/out/fixup", "x_release/out")
BODY = ["main.tex", "sec_intro.tex", "sec_problem.tex", "sec_theory.tex", "sec_method.tex", "sec_experiments.tex", "sec_related.tex",
        "sec_conclusion.tex", "tab_logs.tex", "tab_vlm_main.tex", "tab_vlm_scale.tex", "tab_vlm_maint.tex", "tab_e10_equal.tex",
        "tab_e10_text.tex", "tab_e10_maint.tex", "tab_positioning.tex", "alg_calibrate.tex", "theory_helpers.tex", "alg_helpers.tex"]
REPORT = ["techreport.tex", "app_linear.tex", "app_rescope.tex", "tab_methods.tex", "fig_validity.tex", "alg_complaints.tex",
          "tab_P1b_confront.tex"] + sorted(f for f in os.listdir(PAPER) if f.startswith("tab_vlm_tr_"))
# facts of vlm_facts.json that gen_vlm.py derives from a fix-up file although their 'source' string is generic
# (read in gen_vlm.py: every one of them calls fxp(...) = e3_*/out/fixup/paper_numbers_fixup.json)
FXP_FACT = re.compile(r"^(e3_[QL][am]_s_cap_(min|max)|e3_[QL][am]_h2_(min|max)|e3_[QL][am]_h3p[12]_(min|max)|e3_[QL]_retr_h3_(min|max)|"
                      r"e3_retr_h3_abs_(min|max)|e3_[QL]_draws_above_(min|max)|e3_draws_above_(min|max)|e3_[QL]_near_rel_(min|max)|e3_near_rel_(min|max)|"
                      r"e3l_[QL][am]_h2|e3l_[QL][am]_cap|h3_cap_.*_max|h3_unreachable_max)$")


def text_of(fn):
    return "\n".join(re.split(r"(?<!\\)%", l)[0] for l in open(os.path.join(PAPER, fn)))


def main():
    m = json.load(open(V + "/copies/numbers_manifest.json"))
    facts = json.load(open(PAPER + "/scripts/data/vlm_facts.json"))["facts"]
    tb = {fn: text_of(fn) for fn in BODY}
    tr = {fn: text_of(fn) for fn in REPORT}
    ent = []
    for e in m["macros"]:
        f = e.get("file")
        if f and any(d in f for d in DIRS):
            ent.append({"name": e["name"], "file": f, "key": e["key"][0], "kind": "pn", "fmt": e.get("fmt"), "value": json.load(open(os.path.join(ROOT, f)))[e["key"][0]]})
        elif e.get("source") == "vlm":
            k = e["key"][1]
            fa = facts[k]
            if any(d in str(fa.get("source")) for d in DIRS) or FXP_FACT.match(k):
                ent.append({"name": e["name"], "file": "scripts/data/vlm_facts.json <- " + str(fa["source"]), "key": k, "kind": "fact", "fmt": e.get("fmt"),
                            "scale": e.get("scale"), "value": fa["value"], "rule": fa.get("rule")})
    for fam in m["families"]:
        if fam["prefix"] != "lg":
            continue
        for it in m["logs"]["items"]:
            for fld, spec in fam["fields"].items():
                if spec.get("source") == "vlm":
                    k = spec["key"][1].replace("{log}", it["log"])
                    fa = facts.get(k)
                    if fa and any(d in str(fa.get("source")) for d in DIRS):
                        ent.append({"name": "lg" + it["tag"] + fld, "file": "scripts/data/vlm_facts.json <- " + fa["source"], "key": k, "kind": "lg", "fmt": spec.get("fmt"),
                                    "value": fa["value"], "rule": fa.get("rule"), "in_table": it.get("in_table", True)})
    for e in ent:
        pat = re.compile(r"\\" + e["name"] + r"(?![A-Za-z])")
        e["body_files"] = [fn for fn, t in tb.items() if pat.search(t)]
        e["report_files"] = [fn for fn, t in tr.items() if pat.search(t)]
        e["printed"] = "body" if e["body_files"] else ("report" if e["report_files"] else "unused")
    json.dump(ent, open(V + "/out/macro_list.json", "w"), indent=1)
    import collections
    print(len(ent), collections.Counter((e["kind"], e["printed"]) for e in ent))
    for e in ent:
        print(f"{e['name']:22s} {e['printed']:6s} {e['key'][:52]:52s} {str(e['value'])[:22]:22s} {','.join(e['body_files'] + e['report_files'])}")


main()
