#!/usr/bin/env python3
"""Figures of the technical report, Appendix C (they left the body of the paper: page decisions P2, P5).

    python3 scripts/figs/fig_report.py      writes figs/frontier_tr.pdf (+ .json) and figs/validity_tr.pdf (+ .json)

frontier_tr  (Fig. fig:frontier; experiment E1b, e1_main/out/frontier.json): coverage of rephrasings (text and image)
             against false firing on Held-out (left) and against mean near-scope firing on the test hard negatives
             (right) as ONE common threshold moves, per detector; rows: the two development logs; markers: the
             calibrated operating points at the default level (mean of 20 draws).
validity_tr  (Fig. fig:validity; experiment E5 c, e5_rescope/out/paper_numbers.json): realised release rate of the
             two-block rule (filled) and of the calibration-conditional rank (hollow) over 500 calibration draws, mean
             with the 5th to 95th percentile, per calibration -> test pair; one panel per development log. The two pairs
             calibrated on the out-of-benchmark workload and tested on benchmark or benign queries (90 % and more) are
             off the scale and are given in the caption.
DESIGN (scripts/figs/style.py; dataviz method): colour by method family, marker shape by method; one axis per panel,
small multiples on shared axes; hairline grid; direct labels where a legend is not needed.
"""
import os as _go, sys as _gs  # ARTIFACT: a failed run leaves the stored output files untouched (rescope_paths.guard_outputs)
_gd = _go.environ.get("RESCOPE_HOME") or _go.path.dirname(_go.path.abspath(__file__))
while not _go.path.exists(_go.path.join(_gd, "rescope_paths.py")) and _go.path.dirname(_gd) != _gd: _gd = _go.path.dirname(_gd)
_gs.path.insert(0, _gd); import rescope_paths as _RPG; _RPG.guard_outputs(__file__, __name__)
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import style                                                                 # noqa: E402

PAPER = style.PAPER
ROOT = _RP.ROOT
F = "02_research/experiments/10_final"
LOGS = (("Q", "TB-Q100-ae-l2_3"), ("L", "TB-L100-ae-l2_3"))
FEATS = (("ncoord", "rescope"), ("cos", "cosine"), ("white_0.1", "white"))
PAIRS = (("repair_to_repair_rest", "benchmark, same pool"), ("ben_s_to_ben_s_rest", "benign, same pool"),
         ("mme_c_to_mme_c_rest", "other workload, same pool"), ("repair_to_heldout", "benchmark to Held-out"),
         ("repair_to_ben_t", "benchmark to benign"), ("ben_s_to_heldout", "benign to Held-out"),
         ("repair_to_mme_t", "benchmark to other workload"), ("ben_s_to_mme_t", "benign to other workload"))


def names():
    return json.load(open(os.path.join(PAPER, "scripts", "data", "setup_facts.json")))["logs"]


def frontier():
    src = f"{F}/e1_main/out/frontier.json"
    fr = json.load(open(os.path.join(ROOT, src)))["logs"]
    doc = {"title": "Coverage against far and near-scope firing under one common threshold (Fig. fig:frontier)",
           "source": {"files": [src]}, "panels": []}
    for b, lg in LOGS:
        p = {"log": lg, "title": names()[lg]["paper_name"], "series": []}
        for feat, meth in FEATS:
            c = fr[lg]["features"][feat]["curves"]
            cp = fr[lg]["features"][feat]["calibrated_points"]["two_block@0.05"]
            p["series"].append({"feature": feat, "method": meth,
                                "far": c["probe.heldout_disjoint.fired"], "near": c["hn.test.mean.fired"], "cov": c["edit.rephrase.own_top"],
                                "point": {"far": cp["probe.heldout_disjoint.fired"], "near": cp["hn.test.mean.fired"], "cov": cp["edit.rephrase.own_top"]}})
        r = fr[lg]["retrieval"]["retr_sum"]
        cp = r["calibrated_points"]["two_block@0.05"]
        p["series"].append({"feature": "retr_sum", "method": "retr", "far": r["curves"]["probe.heldout_disjoint.fired"],
                            "near": None, "cov": r["curves"]["edit.rephrase.own_top"],
                            "point": {"far": cp["probe.heldout_disjoint.fired"], "near": cp["hn.test.mean.fired"], "cov": cp["edit.rephrase.own_top"]}})
        doc["panels"].append(p)
    style.apply()
    labels = [style.series_label(m) for _, m in FEATS] + [style.series_label("retr")]
    ncol, rows = style.legend_columns(labels, width="column", ncol_max=4)
    fig, axes = style.figure(ncols=2, nrows=2, width="column", panel_height=1.0, legend_rows=rows, titles=True, sharey=True)
    for row, p in enumerate(doc["panels"]):
        for col, key in enumerate(("far", "near")):
            ax = axes[2 * row + col]
            for s in p["series"]:
                if s[key] is not None:
                    style.plot_trace(ax, s["method"], s[key], s["cov"])
                style.plot_series(ax, s["method"], [s["point"][key]], [s["point"]["cov"]])
            ax.set_xlim(-1, 31 if key == "far" else 101)
            ax.set_ylim(-3, 103)
            ax.set_title(p["title"], loc="left")
            ax.set_xlabel("False firing on Held-out (%)" if key == "far" else "Near-scope firing (%)")
        axes[2 * row].set_ylabel("Rephrasings covered (%)")
    style.legend(fig, axes, ncol=ncol)
    out = os.path.join(PAPER, "figs", "frontier_tr.pdf")
    style.save(fig, out, doc)
    json.dump(doc, open(out[:-4] + ".json", "w"))
    return fig, out


def validity():
    src = f"{F}/e5_rescope/out/paper_numbers.json"
    pn = json.load(open(os.path.join(ROOT, src)))
    doc = {"title": "Realised release rate over 500 calibration draws, per calibration -> test pair (Fig. fig:validity)",
           "source": {"files": [src]}, "level_pct": 5.0, "panels": []}
    for b, lg in LOGS:
        p = {"log": lg, "title": names()[lg]["paper_name"], "pairs": []}
        for pair, label in PAIRS:
            rec = {"pair": pair, "label": label}
            for rule in ("two_block", "pac"):
                k = f"validity.{lg}.{rule}.{pair}."
                rec[rule] = {"mean": pn[k + "ff_mean"], "p05": pn[k + "ff_p05"], "p95": pn[k + "ff_p95"]}
            p["pairs"].append(rec)
        doc["panels"].append(p)
    style.apply()
    fig, axes = style.figure(ncols=2, nrows=1, width="column", panel_height=1.35, legend_rows=1, titles=True, sharey=True)
    fig.subplots_adjust(left=0.43, wspace=0.10)
    n = len(PAIRS)
    for ax, p in zip(axes, doc["panels"]):
        for i, rec in enumerate(p["pairs"]):
            y = n - 1 - i
            for rule, variant, dy in (("two_block", "default", 0.17), ("pac", "published", -0.17)):
                r = rec[rule]
                ln = style.plot_series(ax, "rescope", [r["mean"]], [y + dy], variant=variant, xerr=[(r["p05"], r["p95"])])
                ln.set_label("two-block rule" if rule == "two_block" else "calibration-conditional rank")
        ax.axvline(doc["level_pct"], color=style.colour("linear"), linewidth=style.HAIR_PT + 0.3, zorder=1.5)
        ax.set_xlim(-1, 41)
        ax.set_ylim(-0.6, n - 0.4)
        ax.set_title(p["title"], loc="left")
        ax.set_xlabel("Release rate (%)")
    axes[0].set_yticks(range(n))
    axes[0].set_yticklabels([lab for _, lab in reversed(PAIRS)])
    style.legend(fig, axes, ncol=2, order=["two-block rule", "calibration-conditional rank"])
    out = os.path.join(PAPER, "figs", "validity_tr.pdf")
    style.save(fig, out, doc)
    json.dump(doc, open(out[:-4] + ".json", "w"), indent=1)
    return fig, out


if __name__ == "__main__":
    for fn in (frontier, validity):
        fig, out = fn()
        print("wrote", os.path.relpath(out, PAPER))
        if "--png" in sys.argv:
            fig.savefig(out[:-4] + "_preview.png", dpi=200)
