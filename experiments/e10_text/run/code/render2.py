# E10 render, part 2 (executed inside render.py: same namespace).  Sections 4-11: R21 expectations, R22, equal evidence,
# screening, stream, certificate, generation, anomalies, deviations.

# ============================================================================= 4. R21 (b): expectations written before the run, and what happened
g5, b5 = GATE.get("TX-G2XL-rome-T500"), BEH.get("TX-G2XL-rome-T500")
if g5:
    P("")
    P("4. COLLAPSE LOG TX-G2XL-rome-T500: THE EXPECTATIONS OF R21 (b), WRITTEN BEFORE THE RUN, NEXT TO WHAT HAPPENED")
    e_on, e_off = g5["rows"].get("rescope"), g5["rows"].get("rescope@goff")
    P("   (1) expected: with guards (frozen default) the exploding updates are capped and keep firing on most benign prompts; the calibrated rows MISS their bound.")
    if e_on:
        P(f"       measured (gate level, 20 draws): ReScope / ReScope-W two-block false firing on HELD-OUT {dr(e_on.get('heldout.fired'), 2)} % against the bound 5 %; capped edits per draw "
          f"{dr(e_on.get('cal.n_edits_capped'))} (max {f1((e_on.get('cal.n_edits_capped') or {}).get('max'), 0)} of {g5['meta']['T']}); eta {dr(e_on.get('cal.eta'), 3)}.")
        for lab in ("cosine", best):
            e = g5["rows"].get(lab)
            if e:
                P(f"       {nice(lab, best)}: false firing {dr(e.get('heldout.fired'), 2)} %, capped {dr(e.get('cal.n_edits_capped'))}, coverage {dr(e.get('paraphrase.own_released'))} %  (cosine-family rows are NOT driven to collapse)")
    P("   (2) expected: with guards off an update whose own request is not separated from benign prompts is silenced on its own request and counted as lost.")
    if e_off:
        P(f"       measured (gate level, 20 draws): ReScope guards off: false firing {dr(e_off.get('heldout.fired'), 2)} %; own request released {dr(e_off.get('request.own_released'))} % "
          f"(before {S}: {dr(e_off.get('request.before.own_released'))}; from {S}: {dr(e_off.get('request.from.own_released'))}); coverage of paraphrases {dr(e_off.get('paraphrase.own_released'))} %.")
    de = jl(X.OUT + "/diag_eta.json")
    if de:
        P("       which detectors set the common scale eta of the two-block rule (out/diag_eta.json; arithmetic on the calibration matrices of the 20 draws, nothing served):")
        rows = []
        for lg_, o_ in de["logs"].items():
            rows.append([lg_, o_["T"], o_["n_exploding"], dr(o_["eta"], 3), f1(o_["eta"]["max"], 3), dr(o_.get("eta_without_exploding"), 3), f1((o_.get("eta_without_exploding") or {}).get("max"), 3),
                         dr(o_.get("share_attained_by_exploding_pct")), o_["n_draws_eta_above_0.5"], o_["n_draws_eta_at_least_1"]])
        table(["log", "T", "exploding updates", "eta", "max", "eta without their detectors", "max", "% of block-2 scores >= eta attained by them", "draws eta > 0.5", "draws eta >= 1"], rows, indent=7)
        P("       (the rule has ONE scale eta for all detectors, in units of the raw excess over the per-detector baseline; an own coordinate is 1, so with eta >= 1 every edit is capped (guards on)")
        P("       or silenced on its own request (guards off), not only the exploding ones)")
    P(f"   (3) expected: the requests from position {S} on have targets computed on collapsed weights; no method recovers them.  Measured: tables of section 1 / 3, columns 'from {S}'.")
    if b5:
        rows = []
        for lab in [x for x in ORDER + [y + "@goff" for y in ORDER] if x in b5]:
            o = b5[lab]
            rows.append([nice(lab, best), ci(m(o, "edit.request.exact.before")), ci(m(o, "edit.paraphrase.exact.before")), ci(pub(o, "ES", "before")), ci(pub(o, "PS", "before")),
                         ci(m(o, "edit.request.exact.from")), ci(m(o, "edit.paraphrase.exact.from")), ci(pub(o, "ES", "from")), ci(pub(o, "PS", "from")), ci(m(o, "heldout.D1_tf")),
                         f1((m(o, "heldout.kept_tf") or {}).get("value"))])
        table(["row", f"req exact <{S}", f"par exact <{S}", f"ES <{S}", f"PS <{S}", f"req exact >={S}", f"par exact >={S}", f"ES >={S}", f"PS >={S}", "D1 HELD-OUT", "kept"], rows)

# ============================================================================= 5. R22
R22 = {lg: jl(os.path.join(SUM, f"r22_{lg}.json")) for lg in X.BEHAVE_LOGS}
if any(R22.values()):
    P("")
    P("5. RELATIONAL MAINTENANCE WORKLOAD (R22): Facts(subject, relation, value) over the log's requests (edited tuples) and a VIEW pool of 5000 unedited records")
    P("   value of a tuple under a served model = first token of the greedy continuation of the record's prompt.  States: pre-edit R0, deployed, INTENDED R* (new object for edited tuples,")
    P("   pre-edit value for all others).  Every state and every repaired model is a table of out/r22/<log>.sqlite; every score below is computed by a statement of code/r22.sql.")
    P("   ref: reference preservation = unedited TEST tuples whose value equals the pre-edit value; ret: intended-update retention = edited tuples whose value is the new object;")
    P("   Q exact: test-half queries (selection view V_r and aggregate GROUP BY value, every relation with >= 20 tuples) whose result equals the result on R* exactly; TV: mean total-variation")
    P("   distance of the aggregates to R*; wrong: mean share of wrong tuples per view.  Restoring the pre-edit model everywhere scores ref 100 and ret 0.")
    P("   Evidence contract: a complaint is a report-half QUERY whose deployed result differs from R*; the complainant supplies the query only.  Diagnosis (charged): two forward passes per tuple")
    P("   of the complained view (pre-edit, deployed); changed tuples that are not own requests of the log (a prompt position carries a logged solver key) become the tuple complaints that")
    P("   EVERY repair receives.  diag: forward passes of the diagnosis; rep: further forward passes of the repair; stmts: statements / thresholds changed; lost: edits lost (deleted updates;")
    P("   own requests no longer released).  Rows with draws / seeds: mean over the 3 calibration draws / complaint seeds, interval = crossed bootstrap over them and tuples.")
    for lg in X.BEHAVE_LOGS:
        r = R22.get(lg)
        if not r:
            continue
        collapse = r["meta"]["split"] is not None
        c = r["meta"]["counts"]
        P("")
        P(f"   {lg}: {c['n_edited']} edited tuples, VIEW pool {c['n_report']} report + {c['n_test']} test tuples ({r['meta']['view_pool']['n_excluded_shared_subject']} records excluded: subject of a stream request); "
          f"queries per half: {c['n_queries_report']} / {c['n_queries_test']} relations x (view, aggregate); relations with < 20 tuples dropped: {len(r['meta']['dropped_relations'])};")
        P(f"   VIEW tuples whose prompt carries a logged solver key (treated as own requests): {c['n_view_tuples_carrying_a_solver_key']}; edited tuples not recognised from the log: {c['n_edited_tuples_not_recognised']}")
        if r.get("complaints"):
            cs = r["complaints"]
            P(f"   complaint candidates (report-half queries whose deployed result differs from R*): {cs['n_complaint_candidates']} of {cs['n_report_queries']}")
        order = ["deployed", "restore-base", "blocklist", "rb_suffix", "rb_lineage", "rb_screen", f"oracle: delete update {S}", "comp_min", "ReScope-W log-only", "ReScope-W log-only (guards off)",
                 "ReScope-W least repair (sacrificing)", "ReScope-W least repair (protecting)", "ReScope-W two-block", "ReScope-W two-block (guards off)", "migration (ReScope)",
                 "migration (ReScope) (guards off)", "lookup_subject"]
        rows = []
        for lab in order:
            for g in sorted([x for x in r["rows"] if x["label"] == lab], key=lambda x: -1 if x["m"] is None else x["m"]):
                if g["n_infeasible"] == g["n_instances"]:
                    rows.append([lab, g["evidence"], g["m"], g["n_instances"], "infeasible"] + ["-"] * (10 if collapse else 8) + [f1(g["diagnosis_passes"]["value"], 0), "-", f1(g["repair_seconds"]["value"], 2), "-", "-", "-"])
                    continue
                el = g.get("edits_lost") or {}
                row = [lab + (" [OTHER CLASS]" if lab == "lookup_subject" else ""), g["evidence"], "-" if g["m"] is None else g["m"], g["n_instances"], ci(g.get("reference_preservation")), ci(g.get("retention"))]
                if collapse:
                    row += [ci(g.get("retention_before")), ci(g.get("retention_from"))]
                row += [f1(g["queries_exact_pct"]["value"]), f1(g["views_exact_pct"]["value"]), f1(g["aggregates_exact_pct"]["value"]), f1(g["aggregate_tv_mean"]["value"], 3),
                        f1(g["view_wrong_share_mean_pct"]["value"]), f1(g["diagnosis_passes"]["value"], 0), f1(g["repair_passes"]["value"], 0), f1(g["repair_seconds"]["value"], 2),
                        f1(g["statements_changed"]["value"], 1), f1(el.get("n"), 1) + (f" ({f1(el.get('before'), 1)} / {f1(el.get('from'), 1)})" if collapse and "before" in el else ""),
                        "-" if g.get("complaints_refused") is None else f"{f1(g['complaints_refused'], 1)} / {f1(g.get('n_tuple_complaints'), 1)}"]
                rows.append(row)
        hd = ["repair", "evidence", "m", "inst", "ref", "ret"] + ([f"ret <{S}", f"ret >={S}"] if collapse else []) + ["Q exact", "views", "aggr", "TV", "wrong", "diag", "rep", "repair s", "stmts",
                                                                                                                   "lost" + (f" (<{S} / >={S})" if collapse else ""), "refused / tuples"]
        table(hd, rows)
        so = r.get("serving_overhead")
        if so:
            P("   serving overhead per forward pass, FIRST measurement inside the run (no warm-up, next to other jobs; superseded by the next line) (" + so["_note"] + "): " + "; ".join(f"{k} {v['ms_per_forward_min']:.2f} ms (x{v['ratio_to_pre_edit']:.2f})" for k, v in so.items() if k != "_note"))
        ov = (jl(X.OUT + "/r22/overhead.json") or {}).get("logs", {}).get(lg)
        if ov:
            P("   serving overhead, measured again (out/r22/overhead.json: warm-up, 7 interleaved repeats, median): " + "; ".join(f"{k} {v['ms_median']:.2f} ms (x{v['ratio_median_to_pre_edit']:.2f})" for k, v in ov.items()))
            for k, v in ov.items():
                pn("Rel" + {"TX-G2XL-rome-T100": "TOne", "TX-G2XL-rome-T500": "TFive"}[lg] + "Overhead" + "".join(c for c in k.title() if c.isalpha()), round(v["ratio_median_to_pre_edit"], 3), X.OUT + "/r22/overhead.json", f"logs.{lg}.{k}.ratio_median_to_pre_edit")
        hm = sum((g.get("hook_mismatch") or 0) for g in r["rows"])
        P(f"   prompts on which the hook decided otherwise than the cached pre-edit trace (outside the decision band), all gate rows: {hm}")
        tag = {"TX-G2XL-rome-T100": "TOne", "TX-G2XL-rome-T500": "TFive"}[lg]
        f = os.path.join(SUM, f"r22_{lg}.json")
        nmap = {"deployed": "Deployed", "restore-base": "RestoreBase", "blocklist": "Blocklist", "rb_suffix": "RbSuffix", "rb_lineage": "RbLineage", "rb_screen": "RbScreen", "comp_min": "CompMin",
                "ReScope-W log-only": "RepairLogonly", "ReScope-W least repair (sacrificing)": "LeastSacrificing", "ReScope-W least repair (protecting)": "LeastProtecting",
                "ReScope-W two-block": "RepairTwoblock", "migration (ReScope)": "Migration", "lookup_subject": "Lookup", f"oracle: delete update {S}": "OracleDeleteSplit",
                "ReScope-W two-block (guards off)": "RepairTwoblockGuardsOff", "migration (ReScope) (guards off)": "MigrationGuardsOff", "ReScope-W log-only (guards off)": "RepairLogonlyGuardsOff"}
        for i, g in enumerate(r["rows"]):
            if g["label"] in nmap and g.get("reference_preservation"):
                ms = "" if g["m"] is None else {1: "MOne", 3: "MThree", 10: "MTen"}[g["m"]]
                pn(f"Rel{tag}{nmap[g['label']]}{ms}Ref", round(g["reference_preservation"]["value"], 2), f, f"rows[{i}].reference_preservation.value")
                pn(f"Rel{tag}{nmap[g['label']]}{ms}Ret", round(g["retention"]["value"], 2), f, f"rows[{i}].retention.value")
                pn(f"Rel{tag}{nmap[g['label']]}{ms}QExact", round(g["queries_exact_pct"]["value"], 2), f, f"rows[{i}].queries_exact_pct.value")
                pn(f"Rel{tag}{nmap[g['label']]}{ms}TV", round(g["aggregate_tv_mean"]["value"], 4), f, f"rows[{i}].aggregate_tv_mean.value")
                if "retention_before" in g:
                    pn(f"Rel{tag}{nmap[g['label']]}{ms}RetBefore", round(g["retention_before"]["value"], 2), f, f"rows[{i}].retention_before.value")

# ============================================================================= 6. equal evidence
EQ = {lg: jl(os.path.join(SUM, f"equal_{lg}.json")) for lg in X.BEHAVE_LOGS}
if any(EQ.values()):
    P("")
    P("6. EQUAL EVIDENCE (R14 / R21 d): n reported D1 complaints of the REPAIR pool (benign prompts whose pre-edit continuation the deployed log does not reproduce), with their keys")
    P("   n = 1, 10: mean over 3 complaint seeds; n = 100: one seed; intervals: crossed bootstrap over seeds and records / edits.  D1, kept, 1st: HELD-OUT benign prompts and neighbourhood prompts")
    P("   (teacher-forced); resolved: reported complaints whose continuation is reproduced after the repair; lost: edits lost (deleted updates / own requests silenced); stmts: statements or")
    P("   thresholds changed; passes: further forward passes of the repair (prefix replays); refused: complaints the protecting form refuses.  blocklist: public scores = deployed log.")
    REPS = ["blocklist", "rb_suffix", "rb_lineage", "comp_min", "least_sacrificing", "least_protecting"]
    RN = {"blocklist": "blocklist", "rb_suffix": "rb_suffix", "rb_lineage": "rb_lineage", "comp_min": "comp_min", "least_sacrificing": "ReScope-W least repair (sacrificing)",
          "least_protecting": "ReScope-W least repair (protecting)"}
    for lg in X.BEHAVE_LOGS:
        e, b = EQ.get(lg), BEH.get(lg)
        if not e:
            continue
        collapse = int(short(lg)[1:]) > S
        ci_ = jl(os.path.join(X.OUT, "equal", lg, "_complaints.json")) or {}
        P("")
        P(f"   {lg}: {ci_.get('n_D1_complaints_under_deployed')} D1 complaints among {ci_.get('n_repair_prompts')} REPAIR-pool prompts ({ci_.get('n_complaints_near_scope')} of them near-scope)")
        rows = []
        if b:
            for lab in ("deployed", "preedit", "where_logonly") + (("where_logonly@goff",) if collapse else ()):
                o = b.get(lab)
                if o:
                    rows.append([nice(lab, best) + " (no complaint)", 0, o["n_draws"], ci(m(o, "heldout.D1_tf")), f1((m(o, "heldout.kept_tf") or {}).get("value")), f1((m(o, "heldout.first_changed_tf") or {}).get("value")),
                                 ci(m(o, "neighborhood.D1_tf")), ci(m(o, "edit.request.exact" + (".before" if collapse else ""))), ci(m(o, "edit.paraphrase.exact" + (".before" if collapse else ""))),
                                 f1((pub(o, "ES", "before" if collapse else "all") or {}).get("value")), f1((pub(o, "PS", "before" if collapse else "all") or {}).get("value")),
                                 f1((pub(o, "NS", "all") or {}).get("value")), "-", "-", "-", "-", "-", "-"])
        for n in (1, 10, 100):
            for rep in REPS:
                o = e.get(f"{rep}|n{n}")
                inf = [f for f in os.listdir(os.path.join(X.OUT, "equal", lg)) if f.startswith(f"{rep}__n{n}_") and f.endswith(".infeasible.json")]
                if o is None:
                    if inf:
                        rows.append([RN[rep], n, len(inf), "infeasible"] + ["-"] * 14)
                    continue
                mt = [x for x in o["meta"] if x]
                mean = lambda k: float(np.mean([x[k] for x in mt])) if mt and all(k in x for x in mt) else None      # noqa: E731
                el = [x["edits_lost"] for x in mt if "edits_lost" in x]
                lost = f1(float(np.mean([x["n"] for x in el])), 1) if el else "-"
                if collapse and el and all("before" in x for x in el):
                    lost += f" ({f1(float(np.mean([x['before'] for x in el])), 1)} / {f1(float(np.mean([x['from'] for x in el])), 1)})"
                seg = ".before" if collapse else ""
                rows.append([RN[rep] + (f" [{len(inf)} infeasible]" if inf else ""), n, o["n_draws"], ci(m(o, "heldout.D1_tf")), f1((m(o, "heldout.kept_tf") or {}).get("value")),
                             f1((m(o, "heldout.first_changed_tf") or {}).get("value")), ci(m(o, "neighborhood.D1_tf")), ci(m(o, "edit.request.exact" + seg)), ci(m(o, "edit.paraphrase.exact" + seg)),
                             f1((pub(o, "ES", "before" if collapse else "all") or {}).get("value")), f1((pub(o, "PS", "before" if collapse else "all") or {}).get("value")), f1((pub(o, "NS", "all") or {}).get("value")),
                             f1(100.0 - m(o, "complaints.D1_tf")["value"]) if m(o, "complaints.D1_tf") else "-", lost, f1(mean("statements_changed"), 1), f1(mean("repair_passes"), 0),
                             f1(mean("repair_seconds"), 2), f1(mean("complaints_refused"), 1)])
                tag = {"TX-G2XL-rome-T100": "TOne", "TX-G2XL-rome-T500": "TFive"}[lg]
                nm = {"blocklist": "Blocklist", "rb_suffix": "RbSuffix", "rb_lineage": "RbLineage", "comp_min": "CompMin", "least_sacrificing": "LeastSacrificing", "least_protecting": "LeastProtecting"}[rep]
                ns = {1: "NOne", 10: "NTen", 100: "NHundred"}[n]
                fq = os.path.join(SUM, f"equal_{lg}.json")
                for met, mn in (("heldout.D1_tf", "DOneHeldout"), ("heldout.kept_tf", "KeptHeldout"), ("neighborhood.D1_tf", "DOneNeighbourhood"), ("edit.request.exact" + seg, "ReqExact"),
                                ("edit.paraphrase.exact" + seg, "ParExact")):
                    if m(o, met):
                        pn(f"Equal{tag}{nm}{ns}{mn}", round(m(o, met)["value"], 2), fq, f"{rep}|n{n}.metrics.{met}.value")
        sgt = f" <{S}" if collapse else ""
        table(["repair", "n", "seeds", "D1 HELD-OUT", "kept", "1st", "D1 neighbourhood", "request exact" + sgt, "paraphrase exact" + sgt, "ES" + sgt, "PS" + sgt, "NS", "resolved", "lost" + (f" (<{S} / >={S})" if collapse else ""),
               "stmts", "passes", "repair s", "refused"], rows)
        if collapse:
            P(f"   edit columns: requests logged before position {S}; requests from {S} on (targets computed on collapsed weights):")
            rows = []
            for n in (1, 10, 100):
                for rep in REPS:
                    o = e.get(f"{rep}|n{n}")
                    if o:
                        rows.append([RN[rep], n, ci(m(o, "edit.request.exact.from")), ci(m(o, "edit.paraphrase.exact.from")), f1((pub(o, "ES", "from") or {}).get("value")), f1((pub(o, "PS", "from") or {}).get("value"))])
            table(["repair", "n", f"request exact >={S}", f"paraphrase exact >={S}", f"ES >={S}", f"PS >={S}"], rows)

# ============================================================================= 7. screening
sc = jl(X.OUT + "/screening.json")
if sc:
    P("")
    P("7. SCREENING FROM THE LOG ALONE (R21 c; no query, no model): self-separation margin of edit i = 1 - max over the other edits j of <W_i, k_j> / <W_i, k_i>; flagged: margin <= 0")
    P(f"   reference sets: updates of exploding norm (audit's list, update norm > |W_0|_F: {sc['audit_exploding_updates']}); requests whose subject is the first prompt token (public lookup index 0)")
    rows = []
    for lg, o in sc["logs"].items():
        a, c = o["against"]["exploding_norm"], o["against"]["first_token_subject"]
        rows.append([lg, o["T"], o["n_flagged"], str(o["flagged"]), a["n_reference"], f1(a["precision"]), f1(a["recall"]), str(a["false_negatives"]), c["n_reference"], f1(c["precision"]), f1(c["recall"]),
                     f"{o['margin_quantiles']['0']:.2f} / {o['margin_quantiles']['0.5']:.3f}"])
        tg = short(lg).replace("T100", "TOne").replace("T500", "TFive").replace("TOne0", "TThousand").replace("T2000", "TTwoThousand")
        f = X.OUT + "/screening.json"
        pn(f"Screen{tg}Flagged", o["n_flagged"], f, f"logs.{lg}.n_flagged")
        for k, kn in (("exploding_norm", "Exploding"), ("first_token_subject", "FirstToken")):
            for q, qn in (("precision", "Precision"), ("recall", "Recall")):
                if o["against"][k][q] is not None:
                    pn(f"Screen{tg}{kn}{qn}", round(o["against"][k][q], 1), f, f"logs.{lg}.against.{k}.{q}")
    table(["log", "T", "flagged", "flagged edits", "exploding", "precision", "recall", "exploding not flagged", "first-token", "precision", "recall", "margin min / median"], rows)

# ============================================================================= 8. the stream
ds = jl(X.OUT + "/stream/description.json")
if ds:
    P("")
    P("8. WHAT THE LOG SHOWS ABOUT THE STREAM (task 5; descriptive, from the log's factors; no method)")
    P(f"   |W_0|_F = {ds['W0_fro']:.2f}.  Per prefix of the stream: norms of the logged updates, norm of W - W_0, exploding updates, own coordinates in the library's normalisation (1 by construction),")
    P("   norm of the normalised read-out vectors (detectors), first-token-subject requests, the editor's own final probability of the new object.")
    rows = [[k, f1(v["update_fro_median"], 2), f1(v["update_fro_max"], 1), f1(v["W_minus_W0_fro"], 1), v["n_exploding"], f1(100 * v["share_of_squared_norm_in_exploding"], 2),
             f"{v['own_coordinate_min_max'][0]:.6f} .. {v['own_coordinate_min_max'][1]:.6f}", f1(v["readout_norm_median"], 3), f1(v["readout_norm_max"], 1), v["n_first_token_subject"],
             f1(v["editor_p_new_median"], 3)] for k, v in ds["summary"].items()]
    table(["updates", "|update| median", "max", "|W - W_0|", "exploding", "% of squared norm", "own coordinate", "|read-out| median", "max", "first-token subj", "editor p(new) median"], rows)
    P("   updates of exploding norm (audit's list):")
    table(["update", "|update|", "division factor", "|read-out|", "subject token", "self margin", "|W - W_0| after", "request"],
          [[x["index"], f1(x["update_fro"], 1), f1(x["div_factor"], 4), f1(x["readout_norm"], 2), x["solver_pos"], f1(x["self_margin"], 2), f1(x["W_minus_W0_fro_after"], 1), x["request"]] for x in ds["exploding_updates"]])
sd = os.path.join(X.OUT, "stream")
pref = sorted([d for d in os.listdir(sd) if os.path.isdir(os.path.join(sd, d))], key=lambda d: int(d.split("@")[1]) if "@" in d else int(d.split("-T")[-1])) if os.path.isdir(sd) else []
if pref:
    import summarise as SMR
    P("   retention of the FIRST 100 REQUESTS along the stream (metric vi): prefix logs of the stream (deployed weight replayed bit for bit); calibration draw 0; request / paraphrase exact:")
    P("   greedy decode; ES PS NS: public scores of the first 100 records; NB D1: D1 on their neighbourhood prompts; own !cand: requests of the prefix log whose own edit is no candidate")
    rows, curve = [], {}
    for d in pref:
        Tp = int(d.split("@")[1]) if "@" in d else int(d.split("-T")[-1])
        inst = SMR.instances(d, "stream")
        for key in sorted(inst, key=lambda k: (["deployed", "where_logonly", "where_twoblock", "rescope"].index(k.split("@")[0]), k)):
            it = inst[key][sorted(inst[key])[0]]
            rr = X.load_rows(it["rows"])
            e = lambda kind, fld: 100.0 * float(np.mean([r[fld] for r in rr if r["kind"] == kind]))      # noqa: E731
            pb = (it["public"] or {}).get("all") or {}
            mt = it["meta"].get("meta") or {}
            row = {"prefix": Tp, "row": key, "req": e("request", "new_exact"), "par": e("paraphrase", "new_exact"), "ES": pb.get("ES"), "PS": pb.get("PS"), "NS": pb.get("NS"),
                   "nb_D1": e("neighborhood", "D1_tf"), "nb_kept": e("neighborhood", "tok_loc"), "eta": mt.get("eta"), "capped": mt.get("n_edits_capped"), "own_not_candidate": mt.get("own_not_candidate")}
            curve.setdefault(key, []).append(row)
            rows.append([Tp, nice(key, best), f1(row["req"]), f1(row["par"]), f1(row["ES"]), f1(row["PS"]), f1(row["NS"]), f1(row["nb_D1"]), f1(row["nb_kept"]),
                         "-" if row["eta"] is None else f1(row["eta"], 3), "-" if row["capped"] is None else row["capped"], "-" if row["own_not_candidate"] is None else row["own_not_candidate"]])
    table(["prefix", "row", "request exact", "paraphrase exact", "ES", "PS", "NS", "NB D1", "NB kept", "eta", "capped", "own !cand"], rows)
    X.jdump(curve, os.path.join(SUM, "stream_retention.json"))
    dc = curve.get("deployed", [])
    last_ok = max([r["prefix"] for r in dc if r["ES"] is not None and r["ES"] >= 90.0], default=None)
    first_bad = min([r["prefix"] for r in dc if r["ES"] is not None and r["ES"] < 90.0], default=None)
    P(f"   deployed log: the public efficacy score of the first 100 requests is at least 90 up to the prefix of {last_ok} updates and below 90 from the prefix of {first_bad} updates on"
      f" (prefix P = the first P updates, i.e. update index {None if first_bad is None else first_bad - 1} is the first one in the broken prefix); the split position {S} is fixed by the protocol.")

# ============================================================================= 9. certificate
if os.path.isdir(os.path.join(X.OUT, "cert")):
    P("")
    P("9. CERTIFICATE CHECK (task 6): HELD-OUT benign prompts on whose pre-edit trace a calibrated row releases nothing; served logits of the 10-token decode against the pre-edit model's, bit for bit")
    rows = []
    for lg in X.BEHAVE_LOGS:
        c = jl(os.path.join(X.OUT, "cert", lg + ".json"))
        if not c:
            continue
        t = c["total"]
        P(f"   {lg}: {t['n_checked_certified']} certified (prompt, row, draw) cases checked, {t['n_equal_certified']} bit-identical, {t['n_violations_certified']} violations; inside the decision band (not certified by the cache; "
          f"decided by the hook): {t['n_checked_in_band']} checked, {t['n_equal_in_band']} bit-identical.  Pre-edit decode in these batches equals the stored continuation on {c['pre_edit_decode_in_these_batches_equals_stored_continuation']} of {c['n_heldout_prompts']} prompts.")
        tg = {"TX-G2XL-rome-T100": "TOne", "TX-G2XL-rome-T500": "TFive"}[lg]
        pn(f"Cert{tg}Checked", t["n_checked_certified"], os.path.join(X.OUT, "cert", lg + ".json"), "total.n_checked_certified")
        pn(f"Cert{tg}Violations", t["n_violations_certified"], os.path.join(X.OUT, "cert", lg + ".json"), "total.n_violations_certified")
        for tag, r in c["rows"].items():
            rows.append([short(lg), nice(tag.split("__")[0], best), tag.split("__d")[1], r["n_certified"], r["n_equal"], r["n_violations_certified"], r["n_in_band"], r["n_in_band_equal"],
                         "; ".join(sorted({v["cause"] for v in r["violations"]}))[:150] or "-"])
    table(["log", "row", "draw", "certified", "bit-identical", "violations", "in band", "bit-identical", "causes of violations"], rows)

# ============================================================================= 10. generation
if os.path.isdir(os.path.join(X.OUT, "gen")):
    P("")
    P("10. DEGENERATION (metric v): n-gram entropy of the public generations (GE x100; public generate_fast, top-5 sampling, 100 tokens; generation prompts of the first 100 requests; draw 0)")
    P("    %<q01: texts whose entropy is below the 1st percentile of the pre-edit model's texts; top-8 words: share of the 8 most frequent words among all generated words")
    rows = []
    for lg in X.BEHAVE_LOGS:
        g = jl(os.path.join(X.OUT, "gen", lg + "_summary.json"))
        if not g:
            continue
        for lab, s in g["rows"].items():
            rows.append([short(lg), nice(lab, best), s["n_records"], s["n_texts"], f1(s["GE"]), f1(s.get("GE_se")), f1(s.get("pct_below_ref_q01")), f1(s["per_text_q05"]), f1(s.get("top8_word_share_pct")),
                         f1(s.get("prompts_fired_pct")), f1(s.get("mean_fire_positions_per_text"), 2), s["n_failed"]])
            tg = {"TX-G2XL-rome-T100": "TOne", "TX-G2XL-rome-T500": "TFive"}[lg]
            nm = {"deployed": "Deployed", "preedit": "Preedit", "where_logonly": "RepairLogonly", "where_twoblock": "RepairTwoblock", "rescope": "Migration"}.get(lab.split("@")[0])
            if nm and s["GE"] is not None:
                pn(f"Gen{tg}{nm}{'GuardsOff' if lab.endswith('@goff') else ''}GE", round(s["GE"], 1), os.path.join(X.OUT, "gen", lg + "_summary.json"), f"rows.{lab}.GE")
    table(["log", "row", "records", "texts", "GE", "se", "%<q01", "text q05", "top-8 words %", "prompts fired %", "fire pos / text", "failed"], rows)

# ============================================================================= 11. checks, anomalies, deviations
P("")
P("11. CHECKS AND ANOMALIES (counts from the run files)")
tb, tg_ = jl(X.OUT + "/tests_boot.json"), jl(X.OUT + "/tests_gate.json")
P(f"   crossed bootstrap on synthetic data (out/tests_boot.json): all_ok = {(tb or {}).get('all_ok')};  fast gate-level evaluator against the adapter on the development log (out/tests_gate.json): "
  f"all_ok = {(tg_ or {}).get('all_ok')} ({((tg_ or {}).get('a') or {}).get('n_compared')} + {((tg_ or {}).get('b') or {}).get('n_compared')} numbers compared, "
  f"{((tg_ or {}).get('a') or {}).get('n_different')} + {((tg_ or {}).get('b') or {}).get('n_different')} different)")
for lg in X.BEHAVE_LOGS:
    for sub in ("behave", "equal"):
        s = jl(os.path.join(SUM, f"{sub}_{lg}.json"))
        if s:
            hm = sum(o["cost"]["hook_mismatch"] for o in s.values())
            hd_ = sum(o["cost"]["hook_mismatch_dec"] for o in s.values())
            nb = sum(o["cost"]["near"] for o in s.values())
            P(f"   {lg} {sub}: hook != cached decision, teacher-forced passes: {int(hm)} (must be 0); decodes that left the cached continuation before a release: {int(hd_)}; passes inside the decision band: {int(nb)}")
            bad = [k for k, o in s.items() for kk, e in o.get("public", {}).items() if e.get("agrees_with_public_summariser") is False]
            if bad:
                P(f"      rows whose interval arithmetic does not reproduce the public summariser: {sorted(set(bad))}")
for lg in (X.DEV,) + X.LOGS:
    g = GATE.get(lg)
    if g:
        P(f"   {lg} lookup_subject: own request released at the solver position on {g['meta'].get('lookup_own_request_released_at_solver_position')} of {g['meta']['T']} requests")
DEV_NOTES = [
    ("driver fx", "The shared driver 10_final/common/fx was being extended by other agents during the run; at 13:17 UTC its __init__ imported a module that did not exist yet and six E10 jobs "
     "stopped at import (no result lost: nothing had been computed).  E10 imports a FROZEN minimal copy (code/fx_frozen/fx: draws.py, io.py, stats.py byte-identical to the versions the adapter "
     "was tested with, sha1 79bb3a2819dc / 26e6a1349917 / 6a3cf7ff2965, and a reduced __init__).  The adapter's code fingerprint field 'fx' names the live directory, not this copy."),
    ("R3 gates on text", "The R3 gates are built by the feature layer 10_final/feat2/white (role 'all', rebased values, the adapter's guards and calibration).  Statistics: keys at all cached positions of "
     "500 REPAIR-pool prompts that are in none of the 20 calibration draws (10k+ keys > d_in: the whitener's primal form).  cos_null: null repair on the keys (all positions) of the 100 prompts "
     "that the adapter's rescope_null would use in the same draw."),
    ("patch of the per-position refinement", "e10t.gate._scores_sub re-types the score formulas for lin / cos / ball detectors; for the query-normalised cosine gates (cos_centred, whitecos) the "
     "experiment routes the refinement through the gate's own scores at run time (code/xmeth.py); no file of the adapter is changed; tested (out/tests_gate.json, part b)."),
    ("gate level", "Gate-level numbers come from code/xgate.py (per-position decisions on the GPU, one pass per feature for all draws), tested equal to the adapter's e10t.gate on the development "
     "log; thresholds are built on the CPU by the adapter's calibration.  Guards-off rows are also evaluated at gate level on T1000 / T2000."),
    ("rows not run", "rescope_shared, rescope_null (R21: not run).  Rows of the driver's row set that are undefined on these logs: e10t.methods.UNDEFINED (adapter's USAGE.txt): cosine_default "
     "(needs a marked write position), ball / router (decide at the last prompt position, where a subject-token key is never presented), zscore, dual (cut by R19), the P12 linear repairs "
     "(their text counterparts comp_min, rb_suffix, rb_lineage, blocklist are run)."),
    ("guards off / log-only", "R21 (b) names the calibrated rows for the guards-off run; ReScope-W with log-only thresholds is subject to the same guard rule and is run both ways as well."),
    ("least repair, two forms", "sacrificing = library least_repair (on_conflict='force') from the log-only thresholds WITHOUT guards (allow_silence); protecting = library least_repair "
     "(on_conflict='refuse', complaint by complaint) from the log-only thresholds WITH guards.  Library margins (1e-6 relative): a complaint may stay inside the adapter's decision band; "
     "the table reports the share of complaints resolved as served."),
    ("rb_suffix", "Every prefix 1..T is replayed (float32, bit for bit) and tested on all complaints, stopping at the first unresolved batch; the longest resolving prefix is taken; forward "
     "passes of this scan are the repair's cost."),
    ("comp_min", "Minimum Frobenius-norm correction C with k (W_E + C) = k W_0 at every distinct complaint key and k_i C = 0 at every solver key (fp64, eigen-decomposition of the Gram matrix, "
     "relative cut-off 1e-10); served as float32(W_E + C); infeasible if distinct complaint keys + solver keys exceed 6400."),
    ("blocklist", "Exact-match list on the prompt text; the public test prompts and the HELD-OUT prompts are never listed, so its rows equal the deployed log except for listed texts."),
    ("lookup_subject", "String containment of the edit's subject in the prompt (decoded token ids, no clean-up); release at the token holding the last character of the occurrence; the latest "
     "edit wins a position; in a teacher-forced pass only the prompt is searched; in the public probability test the whole row (prompt + object)."),
    ("R22", "VIEW pool: continuation of the build's seeded permutation (REPAIR / HELD-OUT reproduced), records sharing a subject with any of the 2000 stream requests excluded, first 5000; report half = "
     "first 2500.  Complaint candidates: report-half views and aggregates whose deployed result differs from R*; m of them in a seeded order.  Keys of the complaints come from the diagnosis' "
     "pre-edit pass (no further pass charged).  Calibrated rows: draws 0-2.  Serving overhead measured on shared GPUs (noisy)."),
    ("public score intervals", "ES / PS / NS are the public summariser's numbers (it rounds to two decimals; calibrated rows: mean of the per-draw numbers); their intervals re-apply the summariser's per-record arithmetic to the public case files (agreement within the rounding is checked for every row)."),
    ("serving overhead", "out/r22/overhead.json: one process, warm-up pass, 7 interleaved repeats, median; shared GPUs.  The keyed store's string search is cached per prompt after the warm-up "
     "pass, so its timing covers the model call and the release only."),
    ("calibration-scale diagnostic", "out/diag_eta.json (section 4) is a description added during the run to explain the outcome of expectation R21 (b): it recomputes the common scale eta of the "
     "two-block rule on the same calibration matrices with and without the detectors of the exploding updates.  Nothing is served with the recomputed scale and no row uses it."),
    ("certificate", "Checked on every HELD-OUT prompt that a calibrated row leaves silent on the cached trace, in the batch layout of the behavioural runs; prompts inside the decision band are "
     "listed apart (the cache does not certify them)."),
]
P("")
P("12. DEVIATIONS AND IMPLEMENTATION CHOICES (also out/DEVIATIONS.txt)")
dl = []
for i, (k, v) in enumerate(DEV_NOTES, 1):
    dl.append(f"{i:2d}. {k}: {v}")
for x in dl:
    P("   " + x)
with open(os.path.join(X.OUT, "DEVIATIONS.txt"), "w") as fh:
    fh.write("E10 (public ROME logs of GPT-2 XL): deviations and implementation choices; rendered by code/render.py (render2.py)\n" + "\n".join(dl) + "\n")
P("")
P("13. ROWS OF THE DRIVER'S ROW SET THAT ARE NOT DEFINED / NOT BUILT ON THESE LOGS (e10t.methods.UNDEFINED, the adapter's list)")
try:
    import e10t as _e
    for k, v in _e.methods.UNDEFINED.items():
        P(f"   {k}: {v}")
    P("   rescope_shared, rescope_null: defined; not run (R21: the shared-scale and null-repair variants of the default method are not run).")
except Exception as ex:                                                 # the table must render even if the adapter cannot be imported
    P(f"   (adapter not importable: {type(ex).__name__})")
P("")
P("FILES: out/results.json (flat records), out/paper_numbers.json, out/summary/*.json, out/gate/*.json (+ _flags.npz), out/behave, out/rows (per-row files), out/public (public case files),")
P("       out/r22 (*.json, *.sqlite, flags), out/equal (+ rows_equal, public_equal), out/screening.json, out/stream, out/cert, out/gen, out/split.json, out/best_gate_text.json; code/ ; logs/")
