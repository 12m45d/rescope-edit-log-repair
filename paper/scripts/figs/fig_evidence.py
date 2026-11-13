#!/usr/bin/env python3
"""Panels (a), (b) of Figure fig:evidence (Sec. 5.2, RQ1): repair at equal evidence on the two development logs.

    python3 scripts/figs/fig_evidence.py            writes figs/evidence_ab.pdf and, next to it, figs/evidence_ab.json
                                                    (the plotted values with their sources)

(a) restoration-retention plane (decision E3 of _ext_review/TRIAGE_RESULTS.txt): x = requests still answered with the
    target (Rel exact, percent), y = held-out D1 (percent of valid Held-out queries whose pre-edit answer is not
    reproduced); ONE POINT PER REPAIR AND NUMBER OF COMPLAINTS n, joined in the order of n; the deployed log and the
    pre-edit model as reference points.
(b) evidence curve: the same y against n (0 = the log alone, 1, 2, 5, 10, 20, 50, 100, all).
Rows: the two development logs (small multiples on one shared y axis; never a second axis).

Data (ROOT = project_3; nothing is typed): 02_research/experiments/10_final/e2_curves/out/results.json
  part a  linear and rollback repairs of n reported complaints (records of the pilot P12 re-scored on Held-out):
          COMP_B (compensation, answer positions), COMP_B_ALLPOS (all positions; exists for n <= 10), RB_UNION, RB_TOT,
          REEDIT_B (re-running the editor: NOT a repair of a frozen log, drawn as a reference outside the setting)
  part b  ReScope-W, least repair of the same complaints, from the log-only thresholds and from no thresholds
  base    the deployed log and the pre-edit model
  n = 0 of ReScope-W (log-only thresholds, no complaint): e1_main/out/paper_numbers.json (Eone<Q|L>RescopeWDOne, ...Rel)
  means over the complaint streams (3 for n <= 20, 1 above), as the experiment stores them (metric '@mean').

DESIGN (scripts/figs/style.py; dataviz method): form = scatter with trajectories (a) and line chart (b); colour by
method family (ours / linear repairs / neutral ink for reference points), methods differ by marker shape, so the
figure reads in grey scale; one legend above the panels, two direct labels; hairline grid; one y axis.
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
RES = f"{F}/e2_curves/out/results.json"
E1 = f"{F}/e1_main/out/paper_numbers.json"
LOGS = (("Q", "TB-Q100-ae-l2_3"), ("L", "TB-L100-ae-l2_3"))
NS = ["0", "1", "2", "5", "10", "20", "50", "100", "ALL"]
SERIES = (  # key, style method, variant, part, method of the records, variant of the records
    ("comp_b", "comp_b", "default", "a", "COMP_B", "p12_record"),
    ("comp_b_allpos", "comp_b", "allpos", "a", "COMP_B_ALLPOS", "p12_record"),
    ("reedit_b", "reedit_b", "default", "a", "REEDIT_B", "p12_record"),
    ("rb_union", "rb_union", "default", "a", "RB_UNION", "p12_record"),
    ("rb_tot", "rb_tot", "default", "a", "RB_TOT", "p12_record"),
    ("rescope_where_start", "rescope_where_start", "default", "b", "where_least", "deployed"),
    ("rescope_where", "rescope_where", "default", "b", "where_least", "logonly"),
)
CURVES_B = ("comp_b", "comp_b_allpos", "rb_union", "rescope_where_start", "rescope_where")


def collect():
    recs = json.load(open(os.path.join(ROOT, RES)))
    e1 = json.load(open(os.path.join(ROOT, E1)))
    names = json.load(open(os.path.join(PAPER, "scripts", "data", "setup_facts.json")))["logs"]
    idx = {}
    for r in recs:
        idx.setdefault((r["log"], r["part"], r["method"], r["variant"], str(r["n_label"]), r["pool"], r["metric"]), []).append(r)

    def one(*key):
        rs = idx.get(key, [])
        if not rs:
            return None
        assert len({round(r["value"], 9) for r in rs}) == 1, key
        return rs[0]["value"]
    doc = {"title": "Repair at equal evidence on the two development logs (Fig. fig:evidence, panels a and b)",
           "source": {"files": [RES, E1], "x_a": "Rel exact (percent of the T requests answered with the target)",
                      "y": "held-out D1 (percent of valid Held-out queries)", "x_b": "number of reported complaints n"},
           "panels": []}
    for b, lg in LOGS:
        p = {"log": lg, "title": names[lg]["paper_name"], "series": [], "reference": {}}
        for key, meth, variant, part, rm, rv in SERIES:
            pts = []
            for n in NS:
                d1 = one(lg, part, rm, rv, n, "heldout_disjoint", "probe.D1@mean")
                rel = one(lg, part, rm, rv, n, "edits", "edit.rel.exact@mean")
                if d1 is not None and rel is not None:
                    pts.append({"n": n, "d1": d1, "rel": rel})
            if key == "rescope_where":
                pts.insert(0, {"n": "0", "d1": e1[f"Eone{b}RescopeWDOne"], "rel": e1[f"Eone{b}RescopeWRel"]})
            p["series"].append({"key": key, "method": meth, "variant": variant, "points": pts})
        for ref in ("deployed", "preedit"):
            p["reference"][ref] = {"d1": one(lg, "base", ref, "", "0", "heldout_disjoint", "probe.D1"),
                                   "rel": one(lg, "base", ref, "", "0", "edits", "edit.rel.exact")}
        doc["panels"].append(p)
    return doc


def draw(doc):
    style.apply()
    labels = [style.series_label(m, v) for _, m, v, *_ in SERIES]
    ncol, rows = style.legend_columns(labels, width="column", ncol_max=4)
    ph = 0.375                                                    # height of one panel, inches
    fig, axes = style.figure(ncols=2, nrows=2, width="column", panel_height=ph, legend_rows=rows, titles=True, sharey=True)
    # rows share the x axis of their column: only the titles stand between them (the module leaves room for x labels)
    w, gap = fig.get_figwidth(), (style.TITLE_PT + 3.0) / 72.0
    head = (rows * style.LEGEND_ROW_PT + 3.0 + style.LEGEND_TOP_PT + style.TITLE_PT) / 72.0
    h = head + 2 * ph + gap + style.BOTTOM_PT / 72.0
    fig.set_size_inches(w, h)
    fig.subplots_adjust(bottom=style.BOTTOM_PT / (h * 72.0), top=1.0 - head / h, hspace=gap / ph)
    for row, p in enumerate(doc["panels"]):
        axa, axb = axes[2 * row], axes[2 * row + 1]
        for s in p["series"]:
            xs = [q["rel"] for q in s["points"]]
            ys = [q["d1"] for q in s["points"]]
            style.plot_trace(axa, s["method"], xs, ys)
            style.plot_series(axa, s["method"], xs, ys, variant=s["variant"])
            if s["key"] in CURVES_B:
                style.plot_series(axb, s["method"], [NS.index(q["n"]) for q in s["points"]], ys, curve=True, variant=s["variant"])
        dep, pre = p["reference"]["deployed"], p["reference"]["preedit"]
        style.plot_series(axa, "linear", [dep["rel"]], [dep["d1"]]).set_label("_nolegend_")
        style.plot_series(axa, "preedit", [pre["rel"]], [pre["d1"]]).set_label("_nolegend_")
        axb.axhline(dep["d1"], color=style.colour("linear"), linewidth=style.HAIR_PT + 0.3, zorder=1.5)
        if row == 0:
            style.direct_label(axb, len(NS) - 1, dep["d1"], "deployed log", dx=0, dy=1.5, ha="right")
            allp = [s for s in p["series"] if s["key"] == "rb_union"][0]["points"][-1]
            style.direct_label(axa, allp["rel"], allp["d1"], "all complaints", dx=7, dy=-1, va="center")
        if row == 1:
            style.direct_label(axa, dep["rel"], dep["d1"], "deployed", dx=-4, dy=-14, ha="right", va="center")
            style.direct_label(axa, pre["rel"], pre["d1"], "pre-edit", dx=10, dy=0, va="center")
        axa.set_ylabel("$D_1$ (%)")
        axa.set_title(("(a) " if row == 0 else "") + p["title"], loc="left")
        axb.set_title(("(b) " if row == 0 else "") + p["title"], loc="left")
        axa.set_xlim(-4, 104)
        axb.set_xlim(-0.4, len(NS) - 0.6)
        axb.set_xticks(range(len(NS)))
        axb.set_xticklabels([n if n != "ALL" else "all" for n in NS])
        for ax in (axa, axb):
            ax.set_ylim(-4, 72)
            ax.set_yticks([0, 30, 60])
        if row == 1:
            axa.set_xlabel("Requests answered (Rel, %)")
            axb.set_xlabel("Reported complaints $n$")
        else:
            axa.set_xticklabels([])
            axb.set_xticklabels([])
    style.legend(fig, axes, ncol=ncol)
    return fig


def main():
    doc = collect()
    out = os.path.join(PAPER, "figs", "evidence_ab.pdf")
    fig = draw(doc)
    path = style.save(fig, out, doc)
    with open(out[:-4] + ".json", "w") as f:
        json.dump(doc, f, indent=1)
    print("wrote", os.path.relpath(path, PAPER), "and", os.path.relpath(out[:-4] + ".json", PAPER))
    if "--png" in sys.argv:
        fig.savefig(out[:-4] + "_preview.png", dpi=300)


if __name__ == "__main__":
    main()
