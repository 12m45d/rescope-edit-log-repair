"""E10 step 5: score the single-edit check with the PUBLIC summariser and compare with the published table.

  ours        experiments/summarize.py main() of the public repository on out/public_results/ROME_single/run_000
              (case files written by run_single.py in the public format).  Nothing is re-implemented: the success
              and magnitude scores are the public ones (rewrite / paraphrase / neighborhood _success and _diff).
  published   parsed by this script from the LaTeX source of arXiv:2202.05262 (tables/cf-results.tex, Table
              "Quantitative Editing Results"), rows "ROME" and "GPT-2 XL" of the GPT-2 XL block.  The table gives
              95 % confidence intervals in parentheses and was computed on 7,500 records (text/experiments.tex).
  condition   audit condition 2 of E10 (PROTOCOL.md): |ours - published| <= 3 points for each of efficacy,
              paraphrase and neighbourhood success, on the first 1000 records.
Writes out/single_edit_check.json and out/single_edit_check.txt.
"""
import glob
import io
import json
import os
import re
import sys
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tp_common as C  # noqa: E402

C.setup_public()
import numpy as np  # noqa: E402
from experiments.summarize import main as public_summarize  # noqa: E402

PAPER_SRC = os.path.join(C.EXT, "data", "paper_rome", "src")
RUN = "run_000"
DIR_NAME = "ROME_single"
TOL = 3.0
COLS = ["S", "ES", "EM", "PS", "PM", "NS", "NM", "GE", "RS"]
OURS = {"ES": "rewrite_success", "EM": "rewrite_diff", "PS": "paraphrase_success", "PM": "paraphrase_diff",
        "NS": "neighborhood_success", "NM": "neighborhood_diff", "S": "score"}


def parse_paper():
    """{'ROME': {col: (value, ci95)}, 'GPT-2 XL': {...}} from the GPT-2 XL block of tables/cf-results.tex."""
    path = os.path.join(PAPER_SRC, "tables", "cf-results.tex")
    src = open(path).read()
    block = src.split("GPT-2 XL &", 1)[1].split("\\midrule\\midrule", 1)[0]
    block = "GPT-2 XL &" + block
    out = {}
    for line in block.split("\\\\"):
        line = line.replace("\\midrule", "").strip()
        if not line or "&" not in line:
            continue
        cells = [c.strip() for c in line.split("&")]
        name = cells[0]
        vals = {}
        for col, cell in zip(COLS, cells[1:]):
            cell = re.sub(r"\\(?:goodmetric|badmetric)\{([^}]*)\}", r"\1", cell)
            m = re.match(r"^(-?\d+(?:\.\d+)?)(?:\s*\((\d+(?:\.\d+)?)\))?$", cell.strip())
            assert m, (name, col, cell)
            vals[col] = (float(m.group(1)), float(m.group(2)) if m.group(2) else None)
        out[name] = vals
    n_text = open(os.path.join(PAPER_SRC, "text", "experiments.tex")).read()
    m = re.search(r"over ([\d,]+) and ([\d,]+)-record test sets", n_text)
    return out, {"file": path, "n_records_gpt2xl": int(m.group(1).replace(",", "")) if m else None,
                 "sha256": C.sha256_file(path)}


def per_record(run_dir, n=None):
    """Per-record public scores recomputed from the case files with the public definitions (for intervals only)."""
    files = sorted(glob.glob(os.path.join(run_dir, "case_*.json")), key=lambda p: int(os.path.basename(p)[5:-5]))
    rows = []
    for p in files:
        d = json.load(open(p))
        if n is not None and d["case_id"] >= n:
            break
        r = {"case_id": d["case_id"]}
        for pre in ("pre", "post"):
            if pre not in d:
                continue
            for key, short in (("rewrite_prompts_probs", "rewrite"), ("paraphrase_prompts_probs", "paraphrase")):
                x = d[pre][key]
                r[f"{pre}_{short}_success"] = float(np.mean([a["target_true"] > a["target_new"] for a in x]))
                r[f"{pre}_{short}_diff"] = float(np.mean([np.exp(-a["target_new"]) - np.exp(-a["target_true"]) for a in x]))
            x = d[pre]["neighborhood_prompts_probs"]
            r[f"{pre}_neighborhood_success"] = float(np.mean([a["target_true"] < a["target_new"] for a in x]))
            r[f"{pre}_neighborhood_diff"] = float(np.mean([np.exp(-a["target_true"]) - np.exp(-a["target_new"]) for a in x]))
        rows.append(r)
    return rows


def main():
    from util.globals import RESULTS_DIR
    run_dir = os.path.join(str(RESULTS_DIR), DIR_NAME, RUN)
    n_files = len(glob.glob(os.path.join(run_dir, "case_*.json")))
    paper, paper_meta = parse_paper()
    out = {"written": C.now(), "run_dir": run_dir, "n_case_files": n_files, "n_planned": C.SINGLE_N,
           "complete": n_files >= C.SINGLE_N, "tolerance_points": TOL, "paper": {"arxiv": "2202.05262v5", **paper_meta,
                                                                                 "rows": paper},
           "run_meta": C.jload(os.path.join(run_dir, "run_meta.json")) if os.path.exists(os.path.join(run_dir, "run_meta.json")) else None}
    buf = io.StringIO()
    with redirect_stdout(buf):                                   # the public function pretty-prints its summary
        summ = public_summarize(DIR_NAME, [RUN], first_n_cases=C.SINGLE_N)
    assert len(summ) == 1, len(summ)
    s = summ[0]
    out["public_summary"] = {k: (list(map(float, v)) if isinstance(v, tuple) else v) for k, v in s.items()}
    n = int(s["num_cases"])
    rows = per_record(run_dir, C.SINGLE_N)
    assert len(rows) == n
    rng = np.random.default_rng(0)
    comp = {}
    for which, paper_row in (("post", "ROME"), ("pre", "GPT-2 XL")):
        comp[which] = {}
        for col in ("ES", "PS", "NS", "EM", "PM", "NM", "S"):
            key = f"{which}_{OURS[col]}"
            ours = float(s[key][0])
            pv, pci = paper[paper_row][col]
            d = {"ours": ours, "published": pv, "published_ci95": pci, "diff": ours - pv}
            if col != "S":
                x = np.array([r[key] for r in rows]) * 100
                assert abs(x.mean() - ours) < 0.006, (key, x.mean(), ours)      # the public summary is rounded to 0.01
                boot = x[rng.integers(0, n, (2000, n))].mean(1)
                d.update({"ours_se": float(x.std(ddof=1) / np.sqrt(n)), "ours_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]})
                if pci is not None:
                    se = np.sqrt(d["ours_se"] ** 2 + (pci / 1.96) ** 2)
                    d["z_diff"] = float(d["diff"] / se) if se > 0 else None
            if col in ("ES", "PS", "NS"):
                d["within_tolerance"] = bool(abs(d["diff"]) <= TOL)
            comp[which][col] = d
    out["n_records"] = n
    out["comparison"] = comp
    out["condition_2_pass"] = bool(all(comp["post"][c]["within_tolerance"] for c in ("ES", "PS", "NS"))) if out["complete"] else None
    out["note"] = ("published: 7,500 records; ours: the first %d records.  'pre' = the unedited model on the same records "
                   "(depends on the model, the data and the public evaluation code only)." % n)
    C.jdump(out, os.path.join(C.OUT, "single_edit_check.json"))

    L = []
    L.append("E10 single-edit check (audit condition 2): ROME on GPT-2 XL, CounterFact, each record edited alone")
    L.append(f"records 0 .. {n - 1} ({n} case files of {C.SINGLE_N}; complete: {out['complete']}); scores by the public experiments/summarize.py")
    L.append(f"published: arXiv 2202.05262v5, tables/cf-results.tex (Table 'Quantitative Editing Results', GPT-2 XL block, "
             f"{paper_meta['n_records_gpt2xl']} records; 95 % CI in parentheses)")
    L.append("")
    hdr = f"{'':26s}{'score':>7s}{'ours':>9s}{'(se)':>8s}{'published':>11s}{'(ci95)':>8s}{'diff':>8s}{'<= 3 ?':>8s}"
    names = {"ES": "efficacy success", "PS": "paraphrase success", "NS": "neighbourhood success",
             "EM": "efficacy magnitude", "PM": "paraphrase magnitude", "NM": "neighbourhood magnitude", "S": "harmonic mean"}
    for which, title in (("post", "ROME, edited model (row 'ROME')"), ("pre", "unedited model (row 'GPT-2 XL')")):
        L.append(title)
        L.append(hdr)
        for col in ("ES", "PS", "NS", "EM", "PM", "NM", "S"):
            d = comp[which][col]
            se = f"({d['ours_se']:.2f})" if "ours_se" in d else ""
            ci = f"({d['published_ci95']:.1f})" if d["published_ci95"] is not None else ""
            ok = ("yes" if d["within_tolerance"] else "NO") if "within_tolerance" in d else ""
            L.append(f"{names[col]:26s}{col:>7s}{d['ours']:9.2f}{se:>8s}{d['published']:11.1f}{ci:>8s}{d['diff']:+8.2f}{ok:>8s}")
        L.append("")
    L.append(f"condition 2 (|diff| <= {TOL:g} points for ES, PS, NS of the edited model): "
             + ("PASS" if out["condition_2_pass"] else ("FAIL" if out["condition_2_pass"] is False else "not complete")))
    open(os.path.join(C.OUT, "single_edit_check.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
