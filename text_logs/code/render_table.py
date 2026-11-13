"""Render out/text_logs_table.txt from out/text_logs.json (+ single_edit_check.json, provenance.json).  No number is
typed here: every cell is read from the JSON files written by the experiment scripts."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tp_common as C  # noqa: E402

ORDER = ["TX-G2XL-rome-dev100", "TX-G2XL-rome-T100", "TX-G2XL-rome-T500", "TX-G2XL-rome-T1000", "TX-G2XL-rome-T2000"]
SHORT = {"TX-G2XL-rome-dev100": "dev100", "TX-G2XL-rome-T100": "T100", "TX-G2XL-rome-T500": "T500",
         "TX-G2XL-rome-T1000": "T1000", "TX-G2XL-rome-T2000": "T2000"}
LAB = {"deployed": "deployed", "pre_edit": "pre-edit"}


def f(x, w=7, p=1):
    if x is None:
        return " " * (w - 1) + "-"
    return f"{x:{w}.{p}f}"


def e(x, w=10):
    return " " * (w - 1) + "-" if x is None else f"{x:{w}.2e}"


def table(header, rows, widths):
    out = ["".join(h.rjust(w) if i else h.ljust(w) for i, (h, w) in enumerate(zip(header, widths)))]
    out.append("-" * sum(widths))
    for r in rows:
        out.append("".join(str(c).rjust(w) if i else str(c).ljust(w) for i, (c, w) in enumerate(zip(r, widths))))
    return out


def render():
    R = C.jload(os.path.join(C.OUT, "text_logs.json"))
    logs = [R["logs"][n] for n in ORDER if n in R["logs"]]
    L = []
    L.append("E10 PUBLIC TEXT LOGS: ROME (public reference implementation) on GPT-2 XL, CounterFact requests applied sequentially")
    L.append(f"rendered {C.now()} by render_table.py from out/text_logs.json (aggregated {R['written']})")
    pv = os.path.join(C.OUT, "provenance.json")
    if os.path.exists(pv):
        P = C.jload(pv)
        L.append(f"public code: {P['rome_repo']['url']} commit {P['rome_repo']['commit']}; hyper-parameters hparams/ROME/gpt2-xl.json "
                 f"(sha256 {P['hparams']['sha256'][:12]}), unchanged")
        L.append(f"model: {P['model']['repo']} @ {P['model']['revision'][:12]} float32; statistics: {P['stats']['source']}")
        L.append(f"data: counterfact.json, {P['counterfact']['n_records']} records (sha256 {P['counterfact']['sha256'][:12]}); "
                 f"stack: torch {P['versions']['torch']}, transformers {P['versions']['transformers']}")
    if R["pending"]:
        L.append(f"NOT YET BUILT / DESCRIBED: {', '.join(R['pending'])}")
    L.append("Only the deployed and the pre-edit model are evaluated here.  dev100 is the development log (records 15000 .. 15099);")
    L.append("T100 .. T2000 are prefixes of one stream (records 0 .. T-1).  All rates in percent.")
    L.append("")

    # ---------------------------------------------------------------- 1. logs
    L.append("1. THE LOGS (audit condition 1: deployed weight = pre-edit weight + logged rank-one terms)")
    rows = []
    for g in logs:
        r = g["reconstruction"]
        rows.append([SHORT[g["name"]], f"{g['records'][0]}..{g['records'][1]}", g["T"], g["deployed_sha256"][:10],
                     "yes" if g["exact_float32_replay_equals_deployed"] else "NO", e(r["max_abs"]), e(r["rel_fro_vs_update"]),
                     e(r["rel_fro_vs_W0"]), f(r["W0_fro"], 9, 1), f(r["update_fro"], 10, 1), f(r["WE_max_abs"], 9, 1),
                     "yes" if r["WE_finite"] else "NO"])
    L += table(["log", "records", "T", "W_E sha256", "replay=", "max|E|", "|E|/|dW|", "|E|/|W0|", "|W0|_F", "|W_E-W0|_F", "max|W_E|", "finite"],
               rows, [8, 14, 6, 12, 8, 10, 10, 10, 9, 11, 9, 7])
    if logs:
        L.append(f"pre-edit weight sha256 {logs[0]['pre_edit_sha256']} (transformer.h.17.mlp.c_proj.weight, Conv1D (6400, 1600), float32)")
    L.append("E = W_E - W_0 - sum_i outer(left_i, right_i) in fp64; 'replay=': W_0 plus the public float32 in-place writes of the logged")
    L.append("factors, in order, equals the deployed weight bit for bit.")
    L.append("")
    L.append("   Key identity per update (non-circular: residual = target - module output, captured separately from the factors).")
    L.append("   att(k) = share of the residual that the applied update delivers at key k; 1 means <read-out vector, k> = 1.")
    rows = []
    for g in logs:
        k = g["key_identity"]
        a, b = k["att_request"], k["att_template_mean"]
        rows.append([SHORT[g["name"]], e(k["att_request_max_abs_dev_from_1"]), f(b["min"], 8, 3), f(b["q05"], 8, 3), f(b["median"], 8, 3),
                     f(b["q95"], 8, 3), f(b["max"], 9, 2), f(100 * k["att_template_mean_share_within_1pct"], 8, 1),
                     f(100 * k["att_template_mean_share_within_10pct"], 8, 1), f(100 * k["att_template_mean_share_above_2"], 8, 1),
                     e(k["key_out_consistency"]["max"]), e(k["target_attained"]["max"])])
    L += table(["log", "req:max|1-att|", "tmpl:min", "q05", "median", "q95", "max", "%in1%", "%in10%", "%>2", "key/out", "target"],
               rows, [8, 15, 9, 8, 8, 8, 9, 8, 8, 8, 10, 10])
    L.append("   req  = key of the request prompt alone (the key the public compute_v divides by): the identity holds to float32 round-off.")
    L.append("   tmpl = mean key over the 21 context templates (the vector the public compute_u multiplies by the inverse second moment):")
    L.append("          the identity does NOT hold for it.  key/out: max relative gap between the captured module output and")
    L.append("          captured key @ replayed weight + bias; target: max relative gap between the target and the output after the write.")
    L.append("")

    # ---------------------------------------------------------------- 2. main table
    L.append("2. DEPLOYED AND PRE-EDIT MODEL ON EVERY LOG")
    L.append("   (i) new object, greedy decode (dec) and teacher forcing (tf): exact / token-level match, request and paraphrase prompts")
    rows = []
    for g in logs:
        for lab in ("deployed", "pre_edit"):
            m = g["models"][lab]["edits"]
            a, b = m["request"], m["paraphrase"]
            rows.append([SHORT[g["name"]], LAB[lab], a["n"], f(a["dec_exact"]), f(a["dec_tok"]), f(a["tf_exact"]), f(a["tf_tok"]),
                         b["n"], f(b["dec_exact"]), f(b["dec_tok"]), f(b["tf_exact"]), f(b["tf_tok"]), f(a["true_dec_exact"], 9)])
    L += table(["log", "model", "n req", "dec ex", "dec tok", "tf ex", "tf tok", "n para", "dec ex", "dec tok", "tf ex", "tf tok", "req:true"],
               rows, [8, 10, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 10])
    L.append("   req:true = share of request prompts whose greedy decode starts with the TRUE (old) object.")
    if os.path.exists(pv):
        oc = C.jload(pv)["counterfact"].get("object_token_counts")
        if oc:
            L.append(f"   Object lengths in GPT-2 tokens over all {P['counterfact']['n_records']} records: new object {oc['target_new']}, true object "
                     f"{oc['target_true']}.")
            if set(oc["target_new"]) == {"1"}:
                L.append("   Every new object is ONE token: here exact = token-level and decode = teacher forcing (the first decoded token is the")
                L.append("   teacher-forced argmax), so (i) is a first-token test; the multi-token comparisons of this family are those of (iii).")
    L.append("")
    L.append("   (ii) public scores (public experiments/summarize.py; probabilities of the new vs the true object).  Not comparable with (i).")
    rows = []
    for g in logs:
        if not g.get("public"):
            continue
        for lab in ("deployed", "pre_edit"):
            s = g["public"]["all"][lab]
            rows.append([SHORT[g["name"]], LAB[lab], g["public"]["all"]["n"], f(s["ES"]), f(s["PS"]), f(s["NS"]), f(s["EM"]), f(s["PM"]), f(s["NM"]),
                         f(s["S"]), g["public"]["n_records_with_nonfinite_prob"]["post" if lab == "deployed" else "pre"]])
        w = g["public_write_time"]
        rows.append([SHORT[g["name"]], "at write", w["n"], f(w["ES"]), f(w["PS"]), f(w["NS"]), f(w["EM"]), f(w["PM"]), f(w["NM"]), f(w["S"]), ""])
    L += table(["log", "model", "n rec", "ES", "PS", "NS", "EM", "PM", "NM", "S", "non-finite"], rows, [8, 10, 7, 8, 8, 8, 8, 8, 8, 8, 11])
    L.append("   'at write' = each record scored right after its own update, with the weight of that moment (not a deployed model).")
    L.append("")
    L.append("   (iii) D1: the pre-edit greedy 10-token continuation is not reproduced (valid rows; tf = teacher-forced, dec = decode)")
    rows = []
    for g in logs:
        for lab in ("deployed", "pre_edit"):
            h, nb = g["models"][lab]["heldout"], g["models"][lab]["neighborhood"]
            rows.append([SHORT[g["name"]], LAB[lab], h["all"]["n_valid"], f(h["all"]["D1_tf"]), f(h["all"]["D1_dec"]), f(h["all"]["tok_loc"]),
                         h["clean"]["n_valid"], f(h["clean"]["D1_tf"]), h["near"]["n_valid"], f(h["near"]["D1_tf"]),
                         nb["all"]["n_valid"], f(nb["all"]["D1_tf"]), f(nb["all"]["D1_dec"]), f(nb["all"]["tok_loc"]),
                         nb["clean"]["n_valid"], f(nb["clean"]["D1_tf"]), nb["near"]["n_valid"], f(nb["near"]["D1_tf"])])
    L += table(["log", "model", "HO n", "D1 tf", "D1 dec", "tok-loc", "clean n", "D1 tf", "near n", "D1 tf",
                "NB n", "D1 tf", "D1 dec", "tok-loc", "clean n", "D1 tf", "near n", "D1 tf"],
               rows, [8, 10, 6, 7, 8, 8, 9, 7, 8, 7, 7, 7, 8, 8, 9, 7, 8, 7])
    L.append("   HO = HELD-OUT benign prompts (request + paraphrase prompts of 1000 records in no log); NB = neighbourhood prompts of the")
    L.append("   logged records.  near (HO): shares a subject string with a logged request, mentions a logged subject, or equals a logged")
    L.append("   prompt; near (NB): equals a logged request or mentions another logged subject.  tok-loc = share of y0 tokens kept (tf).")
    L.append("   The pre-edit row is the pre-edit model re-run in other batches: its D1 is the numerical noise floor, not an effect.")
    nd = sum(g["models"]["deployed"][k]["all"]["n_D1_tf_differs_from_D1_dec"] for g in logs for k in ("heldout", "neighborhood"))   # no default: a missing count must not read as 0
    nt = sum(g["models"]["deployed"][k]["all"]["n_valid"] for g in logs for k in ("heldout", "neighborhood"))
    L.append(f"   D1 tf and D1 dec are the same event up to round-off (the decode leaves y0 at the first teacher-forced token that changes):")
    L.append(f"   they differ on {nd} of the {nt} deployed rows of this table.  The decode variant carries no further information here; tok-loc does.")
    L.append("")
    L.append("   (iv) degeneration: public n-gram entropy of generations (GE, x100) and 10-token greedy continuations")
    rows = []
    for g in logs:
        for lab in ("deployed", "pre_edit"):
            m = g["models"][lab]
            ge = m["generation"]
            a, b = ge.get("first100"), ge.get("last100")
            h, nb, rq = m["heldout"]["all"], m["neighborhood"]["all"], m["request_continuations"]
            rows.append([SHORT[g["name"]], LAB[lab], f(a["GE"]) if a else "-", f(a.get("share_texts_below_pre_edit_q01")) if a else "-",
                         f(b["GE"]) if b else "-", f(b.get("share_texts_below_pre_edit_q01")) if b else "-",
                         f(h["repetitive"]), f(h["empty"]), f(h["mean_n_distinct_tokens"], 7, 2), f(nb["repetitive"]), f(rq["repetitive"]),
                         h["n_nonfinite"] + nb["n_nonfinite"] + rq["n_nonfinite"]])
    L += table(["log", "model", "GE f100", "%<q01", "GE l100", "%<q01", "HO rep", "HO empty", "HO #tok", "NB rep", "REQ rep", "non-finite"],
               rows, [8, 10, 9, 7, 9, 7, 8, 9, 8, 8, 8, 11])
    L.append("   GE f100 / l100: generation prompts of the first / last 100 requests of the log (10 prompts per record, 100 tokens, top-5")
    L.append("   sampling of the public generate_fast, seeded per call of 2 records).  %<q01: share of generated texts whose n-gram entropy is below")
    L.append("   the 1st percentile of the pre-edit model's texts on the same prompts.  rep: at most 3 distinct tokens among the 10 greedy")
    L.append("   tokens, or one token 4 times in a row.  #tok: mean number of distinct tokens among the 10.")
    for g in logs:
        hd = g["models"]["deployed"].get("heldout_tokens_that_are_objects_of_exploding_updates")
        hp = g["models"]["pre_edit"].get("heldout_tokens_that_are_objects_of_exploding_updates")
        if hd:
            L.append(f"   {SHORT[g['name']]}: {hd['share_of_tokens']:.1f} % of the greedy tokens on HELD-OUT prompts are new objects of the {g['updates']['n_exploding']} exploding "
                     f"updates ({hd['n_object_tokens']} distinct tokens; pre-edit model {hp['share_of_tokens']:.1f} %); {hd['share_of_prompts_with_one']:.1f} % of the "
                     f"continuations contain one (pre-edit {hp['share_of_prompts_with_one']:.1f} %).")
    sj0 = os.path.join(C.OUT, "single_edit_check.json")
    if os.path.exists(sj0):
        pr = C.jload(sj0)["paper"]["rows"]
        ours = sorted({round(g["models"]["pre_edit"]["generation"]["first100"]["GE"], 1) for g in logs
                       if g["models"]["pre_edit"]["generation"].get("first100") and g["models"]["pre_edit"]["generation"]["first100"]["GE"] is not None})
        L.append(f"   For reference, the published GE (same public functions; arXiv 2202.05262v5, same table): unedited GPT-2 XL {pr['GPT-2 XL']['GE'][0]:.1f} "
                 f"({pr['GPT-2 XL']['GE'][1]:.1f}), after single ROME edits {pr['ROME']['GE'][0]:.1f} ({pr['ROME']['GE'][1]:.1f}); "
                 f"ours for the pre-edit model: {', '.join(f'{x:.1f}' for x in ours)}.")
    L.append("")

    # ---------------------------------------------------------------- 3. retention
    L.append("3. RETENTION: the first 100 requests of the stream as T grows ((v); deployed model unless stated)")
    rows = []
    st = [g for g in logs if g["source"] == "stream"]
    for g in st:
        m = g["models"]["deployed"]["edits_first100"]
        a, b = m["request"], m["paraphrase"]
        s = g["public"]["first100"]["deployed"] if g.get("public") else None
        nb = g["models"]["deployed"]["neighborhood_first100"]
        rows.append([SHORT[g["name"]], f(a["dec_exact"]), f(a["tf_exact"]), f(a["tf_tok"]), f(b["dec_exact"]), f(b["tf_exact"]), f(b["tf_tok"])]
                    + ([f(s["ES"]), f(s["PS"]), f(s["NS"]), f(s["EM"]), f(s["PM"]), f(s["NM"])] if s else ["-"] * 6)
                    + [f(nb["D1_tf"]), f(nb["D1_dec"])])
    if st:
        g = st[0]
        m = g["models"]["pre_edit"]["edits_first100"]
        a, b = m["request"], m["paraphrase"]
        s = g["public"]["first100"]["pre_edit"] if g.get("public") else None
        nb = g["models"]["pre_edit"]["neighborhood_first100"]
        rows.append(["pre-edit", f(a["dec_exact"]), f(a["tf_exact"]), f(a["tf_tok"]), f(b["dec_exact"]), f(b["tf_exact"]), f(b["tf_tok"])]
                    + ([f(s["ES"]), f(s["PS"]), f(s["NS"]), f(s["EM"]), f(s["PM"]), f(s["NM"])] if s else ["-"] * 6)
                    + [f(nb["D1_tf"]), f(nb["D1_dec"])])
    L += table(["log", "req dec", "req tf", "tf tok", "par dec", "par tf", "tf tok", "ES", "PS", "NS", "EM", "PM", "NM", "NB D1tf", "NB D1dec"],
               rows, [9, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9])
    L.append("   req / par: exact match of the new object on the 100 request / 200 paraphrase prompts; NB: D1 on their neighbourhood prompts.")
    L.append("")

    # ---------------------------------------------------------------- 4. updates
    L.append("4. UPDATES ALONG THE STREAM (Frobenius norms; ||W_0||_F = %s)" % (f(logs[0]["reconstruction"]["W0_fro"], 0, 2).strip() if logs else "-"))
    rows = []
    for g in logs:
        u = g["updates"]
        rows.append([SHORT[g["name"]], u["n"], f(u["update_fro"]["min"], 8, 2), f(u["update_fro"]["median"], 8, 2), f(u["update_fro"]["q95"], 8, 2),
                     f(u["update_fro"]["max"], 10, 1), u["n_large"], u["n_exploding"],
                     "-" if u["first_exploding_index"] is None else u["first_exploding_index"], u["n_subject_at_position_0"],
                     f(u["editor_p_new_median"], 8, 3), f(100 * u["editor_p_new_share_below_0.5"], 8, 1)])
    L += table(["log", "T", "min", "median", "q95", "max", ">5xmed", "exploding", "first at", "subj@0", "p_new med", "%p_new<.5"],
               rows, [8, 6, 8, 8, 8, 10, 8, 10, 9, 8, 10, 10])
    L.append("   exploding: ||update||_F > ||W_0||_F.  subj@0: requests whose subject ends at the first token of the request prompt.")
    L.append("   p_new: probability of the new object at the end of the editor's own optimisation (its printed 'avg prob').")
    ft = os.path.join(C.OUT, "first_token_subjects.json")
    if os.path.exists(ft):
        F = C.jload(ft)
        sc = F["stream_check"]
        dfs = list(sc["division_factor_of_these_requests"].values())
        pr = F["probability_that_T_consecutive_requests_contain_none"]
        L.append(f"   Requests whose subject is the first token of the request prompt (public lookup index 0): {F['n_first_token_subject']} of {F['n_records']} records "
                 f"({100 * F['rate']:.2f} %); in the stream: {F['in_stream_0_1999']}; in dev100: {F['in_dev_15000_15099'] or 'none'}.")
        L.append(f"   Their division factors: {min(dfs):.3f} .. {max(dfs):.3f}; the smallest division factor of all other requests of the stream: "
                 f"{sc['smallest_division_factor_of_all_other_requests']:.3f}.  Probability that T consecutive requests contain none, at that rate: "
                 + ", ".join(f"T = {T}: {100 * pr[str(T)]:.1f} %" for T in (100, 500, 1000, 2000)) + ".")
    seg_rows = []
    for g in logs:
        for sname, lab in (("before_first_exploding", "before"), ("from_first_exploding", "from")):
            w = (g.get("write_time_segments") or {}).get(sname)
            if w:
                seg_rows.append([SHORT[g["name"]], f"{lab} {g['updates']['first_exploding_index']}", f"{w['range'][0]}..{w['range'][1] - 1}", w["n"],
                                 f(w["ES"]), f(w["PS"]), f(w["NS"]), f(w["editor_p_new_median"], 9, 3), f(w["editor_p_new_share_below_0.5"], 9, 1),
                                 f(w["target_norm_median"], 10, 1)])
    if seg_rows:
        L.append("   The editor before and from the first exploding update (public scores of each record right after its own update):")
        L += ["   " + x for x in table(["log", "segment", "updates", "n", "ES", "PS", "NS", "p_new med", "%p_new<.5", "|target| med"],
                                       seg_rows, [8, 12, 12, 6, 8, 8, 8, 10, 11, 14])]
        L.append("   |target|: norm of the value the editor asks the module to output at the subject token (it is optimised through the")
        L.append("   current, already edited model).")
    longest = max(logs, key=lambda g: g["T"] if g["source"] == "stream" else -1) if logs else None
    for g in ([x for x in logs if x["source"] == "dev"] + ([longest] if longest and longest["source"] == "stream" else [])):
        ex = g["updates"]["exploding"]
        L.append(f"   exploding updates of {SHORT[g['name']]}: {len(ex)}" + (" (all of them are listed in out/text_logs.json; here: the 12 largest "
                                                                           "and the first)" if len(ex) > 40 else ""))
        if len(ex) > 40:
            keep = {x["index"] for x in sorted(ex, key=lambda x: -x["update_fro"])[:12]} | {ex[0]["index"]}
            ex = [x for x in ex if x["index"] in keep]
        rows = [[x["index"], x["case_id"], f(x["update_fro"], 10, 1), f(x["division_factor"], 8, 4), f(x["att_template_mean"], 9, 1),
                 x["lookup_idx_request"], f(x["v_prob_new"], 7, 3), f(x["W_minus_W0_fro_after"], 10, 1),
                 "  " + x["prompt"].format(x["subject"]) + " -> " + x["target_new"]] for x in ex]
        if rows:
            L += ["   " + s for s in table(["index", "case", "|update|", "div", "att tmpl", "subj tok", "p_new", "|W-W0|", "  request"],
                                           rows, [7, 7, 10, 8, 9, 9, 7, 10, 60])]
    if longest and longest["source"] == "stream":
        ws = longest["weight_norm_along_stream"]
        marks = [x for x in ws if x["n_applied"] in (1, 50, 100, 150, 173, 174, 200, 250, 300, 400, 500, 750, 1000, 1250, 1500, 1750, 2000)]
        L.append("   |W - W_0|_F along the stream: " + "; ".join(f"{x['n_applied']}: {x['W_minus_W0_fro']:.0f}" for x in marks))
    L.append("")

    # ---------------------------------------------------------------- 5. pools
    P = R["pools"]
    L.append("5. POOLS (records in no log, split by record with seed %d)" % P["seed"])
    for s in ("repair", "heldout"):
        q = P["pools"][s]
        L.append(f"   {s.upper():8s} {q['n_records']} records, {q['n_benign_prompts']} benign prompts ({q['n_distinct_texts']} distinct); records sharing a "
                 f"subject string with a request of T100/T500/T1000/T2000: {q['records_subject_in_T100']}/{q['records_subject_in_T500']}/"
                 f"{q['records_subject_in_T1000']}/{q['records_subject_in_T2000']}, of dev100: {q['records_subject_in_dev']}; prompts mentioning a "
                 f"logged subject: {q['prompts_mention_subject_of_T100']}/{q['prompts_mention_subject_of_T500']}/{q['prompts_mention_subject_of_T1000']}/"
                 f"{q['prompts_mention_subject_of_T2000']}, dev100: {q['prompts_mention_subject_of_dev']}")
    pc = R["pre_continuations"]
    L.append(f"   pre-edit continuations stored for {pc['n_texts']} distinct prompts: {pc['n_valid']} valid, {pc['n_invalid']} not reproduced under "
             f"teacher forcing, {pc['n_nonfinite']} non-finite, {pc['n_empty']} empty, {pc['n_repetitive']} repetitive, {pc['n_has_eos']} with <|endoftext|>")
    nn = P["neighborhood_prompts_mentioning_another_logged_subject"]
    ne = P["neighborhood_prompts_equal_to_a_logged_request"]
    L.append("   neighbourhood prompts that mention another logged subject (T100/T500/T1000/T2000, dev100): "
             f"{nn['T100']}/{nn['T500']}/{nn['T1000']}/{nn['T2000']}, {nn['dev']}; that equal a logged request: "
             f"{ne['T100']}/{ne['T500']}/{ne['T1000']}/{ne['T2000']}, {ne['dev']}")
    L.append("")

    # ---------------------------------------------------------------- 6. single-edit check
    sp = os.path.join(C.OUT, "single_edit_check.txt")
    if os.path.exists(sp):
        L.append("6. SINGLE-EDIT CHECK (audit condition 2; out/single_edit_check.txt)")
        L += ["   " + s for s in open(sp).read().rstrip("\n").split("\n")]
        L.append("")
    cl = os.path.join(C.OUT, "library_check.json")
    if os.path.exists(cl):
        K = C.jload(cl)
        L.append("7. LIBRARY FORMAT CHECK (development log only; out/library_check.json)")
        for line in K.get("summary_lines", []):
            L.append("   " + line)
        L.append("")
    dc = os.path.join(C.OUT, "d1_curve_dev.txt")
    if os.path.exists(dc):
        L.append("8. D1 AS A FUNCTION OF THE NUMBER OF UPDATES (development log only; out/d1_curve_dev.txt)")
        L += ["   " + x for x in open(dc).read().rstrip("\n").split("\n")]
        L.append("")
    cp = os.path.join(C.OUT, "collapse_point.txt")
    if os.path.exists(cp):
        L.append("9. THE DEPLOYED MODEL JUST BEFORE AND JUST AFTER THE FIRST EXPLODING UPDATE (reported stream; out/collapse_point.txt)")
        L += ["   " + x for x in open(cp).read().rstrip("\n").split("\n")]
        L.append("")
    # ---------------------------------------------------------------- audit conditions
    L.append("10. AUDIT CONDITIONS OF E10 (PROTOCOL.md, section 6) AS MEASURED BY THE BUILD (the independent audit decides)")
    for g in logs:
        r = g["reconstruction"]
        L.append(f"   (1) {SHORT[g['name']]:7s} W_E = W_0 + logged rank-one terms: relative Frobenius error {r['rel_fro_vs_update']:.1e} of the update "
                 f"({r['rel_fro_vs_W0']:.1e} of W_0), max entry {r['max_abs']:.1e}; bitwise float32 replay: {g['exact_float32_replay_equals_deployed']}")
    sj = os.path.join(C.OUT, "single_edit_check.json")
    if os.path.exists(sj):
        S = C.jload(sj)
        c = S["comparison"]["post"]
        L.append(f"   (2) single edits, records 0 .. {S['n_records'] - 1} ({'complete' if S['complete'] else 'NOT complete'}): ES {c['ES']['ours']:.1f} vs "
                 f"{c['ES']['published']:.1f}, PS {c['PS']['ours']:.1f} vs {c['PS']['published']:.1f}, NS {c['NS']['ours']:.1f} vs {c['NS']['published']:.1f} "
                 f"(published: arXiv 2202.05262v5, Table 'Quantitative Editing Results'); within 3 points: {S['condition_2_pass']}")
    for g in logs:
        m, u = g["models"]["deployed"]["edits"]["request"], g["updates"]
        L.append(f"   (3) {SHORT[g['name']]:7s} multi-token: {m['pct_multi_token_objects']:.0f} % of the new objects have more than one token (max "
                 f"{m['max_object_tokens']}), exact = all tokens; signs: {u['n_negative_division_factor']} updates with a negative division factor, "
                 f"smallest |division factor| {u['min_abs_division_factor']:.4f}")
    pc = R["pre_continuations"]
    L.append(f"   (3) degenerate pre-edit continuations among {pc['n_texts']} prompts: {pc['n_empty']} empty, {pc['n_repetitive']} repetitive, "
             f"{pc['n_invalid']} not reproduced under teacher forcing (excluded from D1)")
    ap = os.path.join(C.OUT, "log_audit.json")
    if os.path.exists(ap):
        A = C.jload(ap)
        sv, dt = A.get("single_vs_stream"), A.get("determinism_across_processes") or {}
        if sv:
            L.append(f"   (-) single edit vs stream, records 0 .. {sv['n_records_compared'] - 1}: left vector bitwise equal on {sv['n_left_vector_bitwise_equal']}, both keys on "
                     f"{sv['n_both_keys_bitwise_equal']} of {sv['n_records_compared']} (keys and direction do not depend on the edited weight); "
                     f"single edits with an exploding update: {[x['case_id'] for x in sv['single_edits_with_exploding_update']]}")
        if dt:
            L.append(f"   (-) determinism across processes: record 0 edited alone = first update of the stream (right vector bitwise: "
                     f"{dt.get('record0_right_vector_bitwise_equal')}, public scores identical: {dt.get('record0_public_scores_identical')}); pre-edit public test "
                     f"identical in two runs on {dt.get('pre_edit_public_test_records_identical')} of {dt.get('pre_edit_public_test_records_compared')} records")
        for src, v in A["sources"].items():
            rs = v.get("resumed") or []
            L.append(f"   (-) run '{src}': {v['n_updates']} updates, resumed {len(rs)} time(s)"
                     + (f" (after {', '.join(str(x['n_have']) for x in rs)} updates)" if rs else "")
                     + f"; weight hashes recorded during the run and reproduced by the replay: {v['n_recorded_hashes_reproduced']} of {v['n_recorded_hashes']}")
    sc = os.path.join(C.OUT, "selfcheck.json")
    if os.path.exists(sc):
        K = C.jload(sc)["tags"]
        L.append("   (-) self-check of the batched description against one-prompt-at-a-time recomputation without key-value cache (equal rows / rows): "
                 + "; ".join(f"{t}: decode {v['dec_equal']}/{v['n_rows']}, teacher forcing {v['tf_equal']}/{v['n_rows']}, object {v['edit_tf_equal']}/{v['n_edit_rows']}"
                             for t, v in K.items()))
    scc = os.path.join(C.OUT, "selfcheck_cpu.json")
    if os.path.exists(scc):
        K = C.jload(scc)["tags"]
        L.append("   (-) the same on the CPU (float32, another device than the stored results): "
                 + "; ".join(f"{t}: decode {v['dec_equal']}/{v['n_rows']}, teacher forcing {v['tf_equal']}/{v['n_rows']}" for t, v in K.items()))
    L.append("")
    open(os.path.join(C.OUT, "text_logs_table.txt"), "w").write("\n".join(L) + "\n")
    return L


if __name__ == "__main__":
    print("\n".join(render()))
