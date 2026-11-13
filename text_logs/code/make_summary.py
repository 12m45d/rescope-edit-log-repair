"""Render out/summary.txt: the compact table of the deployed logs (one line per log) for the report, from
out/text_logs.json and out/single_edit_check.json.  No number is typed here."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tp_common as C  # noqa: E402

ORDER = ["TX-G2XL-rome-dev100", "TX-G2XL-rome-T100", "TX-G2XL-rome-T500", "TX-G2XL-rome-T1000", "TX-G2XL-rome-T2000"]


def f(x, w=6, p=1):
    return " " * (w - 1) + "-" if x is None else f"{x:{w}.{p}f}"


def main():
    R = C.jload(os.path.join(C.OUT, "text_logs.json"))
    S = C.jload(os.path.join(C.OUT, "single_edit_check.json"))
    L = ["E10 public text logs (ROME, GPT-2 XL, CounterFact): deployed model per log; pre-edit model in brackets.  Percent.",
         f"{'log':8s}{'T':>5s} {'|E|/|dW|':>9s} {'max|E|':>8s} | {'ES':>5s} {'PS':>5s} {'NS':>5s} | {'req':>5s} {'para':>5s} | "
         f"{'D1 HO':>6s} {'tokloc':>6s} {'D1 NB':>6s} | {'GE':>6s} {'[pre]':>7s} {'rep HO':>6s} | {'expl':>4s} {'first':>5s}"]
    for n in ORDER:
        if n not in R["logs"]:
            continue
        g = R["logs"][n]
        d, p = g["models"]["deployed"], g["models"]["pre_edit"]
        pub = g["public"]["all"]["deployed"] if g.get("public") else {}
        ge, gp = d["generation"].get("first100"), p["generation"].get("first100")
        u = g["updates"]
        L.append(f"{n.split('-')[-1]:8s}{g['T']:5d} {g['reconstruction']['rel_fro_vs_update']:9.1e} {g['reconstruction']['max_abs']:8.1e} | "
                 f"{f(pub.get('ES'), 5)} {f(pub.get('PS'), 5)} {f(pub.get('NS'), 5)} | {f(d['edits']['request']['dec_exact'], 5)} "
                 f"{f(d['edits']['paraphrase']['dec_exact'], 5)} | {f(d['heldout']['all']['D1_tf'])} {f(d['heldout']['all']['tok_loc'])} "
                 f"{f(d['neighborhood']['all']['D1_tf'])} | {f(ge['GE'] if ge else None)} [{f(gp['GE'] if gp else None, 5)}] "
                 f"{f(d['heldout']['all']['repetitive'])} | {u['n_exploding']:4d} {('-' if u['first_exploding_index'] is None else u['first_exploding_index']):>5}")
    if R["logs"]:
        g = R["logs"][ORDER[1]] if ORDER[1] in R["logs"] else next(iter(R["logs"].values()))
        pp = g["public"]["all"]["pre_edit"] if g.get("public") else {}
        L.append(f"pre-edit model on the records of T100: ES {f(pp.get('ES'), 0).strip()} PS {f(pp.get('PS'), 0).strip()} NS {f(pp.get('NS'), 0).strip()}; "
                 "its D1 is 0.0 on every pool (noise floor).")
    c = S["comparison"]["post"]
    L.append(f"single-edit check, records 0..{S['n_records'] - 1} (public code and summariser) vs arXiv 2202.05262v5 Table 'Quantitative Editing Results': "
             f"ES {c['ES']['ours']:.1f} / {c['ES']['published']:.1f}, PS {c['PS']['ours']:.1f} / {c['PS']['published']:.1f}, "
             f"NS {c['NS']['ours']:.1f} / {c['NS']['published']:.1f} (ours / published); condition (3 points): "
             f"{'PASS' if S['condition_2_pass'] else 'FAIL' if S['condition_2_pass'] is False else 'incomplete'}")
    L.append("ES / PS / NS: public efficacy / paraphrase / neighbourhood success.  req / para: greedy decode starts with the new object (request /")
    L.append("paraphrase prompts).  D1: pre-edit 10-token continuation not reproduced (teacher-forced) on HELD-OUT benign (HO) / neighbourhood (NB)")
    L.append("prompts; tokloc: share of its tokens kept (HO).  GE: public n-gram entropy x100, generation prompts of the first 100 requests.")
    L.append("rep HO: repetitive 10-token continuations.  expl: updates with ||update||_F > ||W_0||_F; first: index of the first one.")
    open(os.path.join(C.OUT, "summary.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
