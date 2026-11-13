#!/usr/bin/env python3
"""Two-panel frontier figure: what a repair keeps of the edits against what it leaves of the complaints.

    python3 scripts/figs/fig_frontier.py <data.json> --out <figure.pdf> [--print-table]
    python3 scripts/figs/fig_frontier.py --make-example      writes example/frontier_synthetic.json and draws
                                                             example/frontier_synthetic.pdf (made-up data, watermarked;
                                                             an artificial, neutral layout: see make_example)

Planned use: Fig. fig:frontier of Sec. 5.2 (experiment E1), one panel per development log:
    x  rate of complaints (or of false firing) on Held-out, percent      y  mean of T-Gen and I-Gen (exact match), percent
    curves: \\sys and the calibrated gates over the calibration levels; points: the deployed log, the editor's other
    working points, the linear repairs with all complaints, the gates at their published thresholds, Dual, \\sysW.
The figure is included at its final size: \\includegraphics[width=\\columnwidth]{figs/frontier}.

INPUT: a JSON document of the schema "rescope.fig.frontier/1" (scripts/figs/frontier.schema.json documents every field;
validate() below enforces it without the jsonschema package):
    {"schema": "rescope.fig.frontier/1", "synthetic": false, "title": "...",
     "source": {"experiment": "E1", "files": ["02_research/experiments/10_final/out/E1_main.json"], ...},
     "x": {"metric": "d1", "label": "Complaints on Held-out (%)"}, "y": {"metric": "gen", "label": "..."},
     "panels": [{"log": "<testbed name>", "title": "<paper name>", "series": [
         {"method": "rescope", "curve": true, "label_point": 2, "points": [{"x": .., "y": .., "x_lo": .., "x_hi": ..,
                                                                             "y_lo": .., "y_hi": .., "level": 0.05}, ...]},
         {"method": "rescope", "variant": "shared", "trace": [{"x": .., "y": ..}, ...], "points": [...]},
         {"method": "cosine", "variant": "published", "points": [{"x": .., "y": ..}]}, ...]}, {second panel}]}
A series has marked "points" (one per calibration level; joined if "curve" is true) and / or a "trace", a curve
without markers (one global threshold swept over a fine grid: PROTOCOL.md, amendment of 2026-11-11, E1b). Variants:
default (the method's default calibration, filled marker), shared (one shared scale, half-filled marker), published
(published threshold, hollow marker).
The experiment code writes this file from its result JSON; no value is typed. The script draws what the file holds
and nothing else: no smoothing, no sorting of curve points, no selection of series.

DESIGN (scripts/figs/style.py; dataviz method)
  form     two measures against each other for a handful of methods: a scatter with curves, small multiples for the two
           logs on ONE shared y axis (never a second axis)
  colour   by method family, three hues (ours, gates, linear repairs) plus neutral ink for reference points; methods
           differ by marker shape, variants of a method by its fill (full / half / hollow)
  labels   a legend for the whole figure and at most four direct labels per panel (the series with "label_point")
  twin     --print-table prints the plotted values (the numbers of the figure are also those of Table tab:main)
Synthetic data ("synthetic": true) is drawn with a watermark in every panel and only into scripts/figs/example/.
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import style                                                                      # noqa: E402

SCHEMA = "rescope.fig.frontier/1"
EXAMPLE_JSON = os.path.join(style.EXAMPLE_DIR, "frontier_synthetic.json")
EXAMPLE_PDF = os.path.join(style.EXAMPLE_DIR, "frontier_synthetic.pdf")
MAX_DIRECT_LABELS = 4


def validate(doc):
    """Errors of a document against the schema rescope.fig.frontier/1 (same constraints as frontier.schema.json)."""
    err = []
    num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
    if not isinstance(doc, dict):
        return ["the document is not an object"]
    if doc.get("schema") != SCHEMA:
        err.append(f"schema must be {SCHEMA!r}, found {doc.get('schema')!r}")
    extra = set(doc) - {"schema", "synthetic", "title", "note", "source", "x", "y", "panels"}
    if extra:
        err.append(f"unknown top-level fields: {sorted(extra)}")
    if not isinstance(doc.get("synthetic"), bool):
        err.append("'synthetic' must be true or false")
    if not isinstance(doc.get("title"), str):
        err.append("'title' must be a string")
    if doc.get("synthetic") is False:
        src = doc.get("source")
        if not (isinstance(src, dict) and isinstance(src.get("experiment"), str) and isinstance(src.get("files"), list) and src["files"]):
            err.append("real data needs 'source' with 'experiment' and a non-empty list 'files'")
    for ax in ("x", "y"):
        a = doc.get(ax)
        if not (isinstance(a, dict) and isinstance(a.get("metric"), str) and isinstance(a.get("label"), str)):
            err.append(f"'{ax}' needs 'metric' and 'label'")
        elif set(a) - {"metric", "label", "min", "max"} or not all(num(a[k]) for k in ("min", "max") if k in a):
            err.append(f"'{ax}': only metric, label, min, max (numbers) are allowed")
    panels = doc.get("panels")
    if not (isinstance(panels, list) and len(panels) == 2):
        return err + ["'panels' must be a list of exactly two panels"]
    for i, p in enumerate(panels):
        where = f"panels[{i}]"
        if not (isinstance(p, dict) and isinstance(p.get("log"), str) and isinstance(p.get("title"), str)
                and isinstance(p.get("series"), list) and p["series"]) or set(p) - {"log", "title", "series"}:
            err.append(f"{where}: needs exactly 'log', 'title' and a non-empty list 'series'")
            continue
        seen, n_labels = set(), 0
        for j, s in enumerate(p["series"]):
            w = f"{where}.series[{j}]"
            if not isinstance(s, dict) or set(s) - {"method", "variant", "curve", "label_point", "points", "trace"}:
                err.append(f"{w}: only method, variant, curve, label_point, points, trace are allowed")
                continue
            if s.get("method") not in style.METHODS:
                err.append(f"{w}: unknown method {s.get('method')!r} (ids of style.METHODS: {', '.join(style.METHODS)})")
            if s.get("variant", "default") not in style.VARIANTS:
                err.append(f"{w}: variant must be one of {sorted(style.VARIANTS)}")
            elif s.get("variant", "default") != "default" and s.get("method") in style.METHODS and style.METHODS[s["method"]][3] != "full":
                err.append(f"{w}: the method {s['method']!r} has no variants")
            if "curve" in s and not isinstance(s["curve"], bool):
                err.append(f"{w}: 'curve' must be true or false")
            key = (s.get("method"), s.get("variant", "default"))
            if key in seen:
                err.append(f"{w}: the series {key} occurs twice in the panel")
            seen.add(key)
            pts, tr = s.get("points", []), s.get("trace", [])
            if not isinstance(pts, list) or not isinstance(tr, list) or not (pts or tr):
                err.append(f"{w}: give a non-empty list 'points', a non-empty list 'trace', or both")
                continue
            for k, pt in enumerate(tr):
                if not (isinstance(pt, dict) and set(pt) <= {"x", "y", "threshold"} and num(pt.get("x")) and num(pt.get("y"))):
                    err.append(f"{w}.trace[{k}]: a trace point is {{x, y}} (finite numbers), optionally with 'threshold'")
                    break
            for k, pt in enumerate(pts):
                if not isinstance(pt, dict) or set(pt) - {"x", "y", "x_lo", "x_hi", "y_lo", "y_hi", "level", "setting"}:
                    err.append(f"{w}.points[{k}]: only x, y, x_lo, x_hi, y_lo, y_hi, level, setting are allowed")
                    continue
                if not (num(pt.get("x")) and num(pt.get("y"))):
                    err.append(f"{w}.points[{k}]: x and y must be finite numbers")
                    continue
                for c in ("x", "y"):
                    lo, hi = pt.get(c + "_lo"), pt.get(c + "_hi")
                    if (lo is None) != (hi is None):
                        err.append(f"{w}.points[{k}]: give both {c}_lo and {c}_hi or neither")
                    elif lo is not None and not (num(lo) and num(hi) and lo <= pt[c] <= hi):
                        err.append(f"{w}.points[{k}]: need {c}_lo <= {c} <= {c}_hi")
            if "label_point" in s:
                n_labels += 1
                if not (isinstance(s["label_point"], int) and not isinstance(s["label_point"], bool) and 0 <= s["label_point"] < len(pts)):
                    err.append(f"{w}: label_point must index a point")
        if n_labels > MAX_DIRECT_LABELS:
            err.append(f"{where}: {n_labels} direct labels; at most {MAX_DIRECT_LABELS} (the legend names the rest)")
    return err


def draw(doc):
    style.apply()
    drawn = sorted({(s["method"], s.get("variant", "default")) for p in doc["panels"] for s in p["series"] if s.get("points")})
    ncol, rows = style.legend_columns([style.series_label(m, v) for m, v in drawn], width="column")
    fig, axes = style.figure(ncols=2, width="column", panel_height=1.05, legend_rows=rows, sharey=True)
    for ax, panel in zip(axes, doc["panels"]):
        for s in panel["series"]:
            if s.get("trace"):
                style.plot_trace(ax, s["method"], [p["x"] for p in s["trace"]], [p["y"] for p in s["trace"]])
            pts = s.get("points", [])
            if not pts:
                continue
            xs, ys = [p["x"] for p in pts], [p["y"] for p in pts]
            xerr = [(p["x_lo"], p["x_hi"]) if "x_lo" in p else None for p in pts]
            yerr = [(p["y_lo"], p["y_hi"]) if "y_lo" in p else None for p in pts]
            style.plot_series(ax, s["method"], xs, ys, curve=bool(s.get("curve")) and len(pts) > 1,
                              variant=s.get("variant", "default"), xerr=xerr, yerr=yerr)
            if "label_point" in s:
                k = s["label_point"]
                style.direct_label(ax, xs[k], ys[k], style.METHODS[s["method"]][1])
        ax.set_title(panel["title"])
        ax.set_xlabel(doc["x"]["label"])
        if doc["synthetic"]:
            style.watermark(ax)
    # both panels share both scales, so that the two logs can be compared by position
    style.set_limits(axes, doc["x"].get("min"), doc["x"].get("max"), doc["y"].get("min"), doc["y"].get("max"))
    axes[0].set_ylabel(doc["y"]["label"])
    style.legend(fig, axes, ncol=ncol)
    return fig


def table(doc):
    """The plotted values as text lines (the table view of the figure)."""
    out = [f"# {doc['title']}" + ("   [SYNTHETIC EXAMPLE DATA]" if doc["synthetic"] else ""),
           f"# x = {doc['x']['label']}; y = {doc['y']['label']}", "panel\tmethod\tvariant\tlevel\tx\tx_lo\tx_hi\ty\ty_lo\ty_hi"]
    f = lambda v: "" if v is None else f"{v:g}"
    for p in doc["panels"]:
        for s in p["series"]:
            for pt in s.get("points", []):
                out.append("\t".join([p["title"], s["method"], s.get("variant", "default"), f(pt.get("level")), f(pt["x"]), f(pt.get("x_lo")),
                                      f(pt.get("x_hi")), f(pt["y"]), f(pt.get("y_lo")), f(pt.get("y_hi"))]))
            if s.get("trace"):
                out.append(f"# {p['title']}\t{s['method']}\t{s.get('variant', 'default')}\ttrace of {len(s['trace'])} points (not listed)")
    return out


def make_example():
    """Made-up data that exercises every element of the figure (traces, marked points with intervals on and beside a
    trace, joined points, single points, the three variants, direct labels, two panels). The layout is deliberately
    artificial and NEUTRAL: parallel straight lines in the alphabetical order of the method ids, single points on a
    regular grid in alphabetical order, offsets that alternate in sign. The numbers are not results, not estimates and
    not expectations of any experiment, and the position of a method in the picture means nothing."""
    levels = [0.01, 0.02, 0.05, 0.10, 0.20]
    lines = sorted(["rescope", "cosine", "ball", "router", "zscore"])
    traced = sorted(["rescope", "cosine"])
    singles = sorted(["rescope_where", "comp_b", "rb_union", "linear", "dual"])
    panels = []
    for name, k in (("SYNTHETIC LOG A", 0.0), ("SYNTHETIC LOG B", 4.0)):
        series = []
        for j, m in enumerate(lines):                                    # y = 8 + 14 j + 0.6 x: parallel lines

            def pts(dy):
                out = []
                for lv in levels:
                    x = 100 * lv
                    y = 8 + 14 * j + 0.6 * x + k + dy
                    out.append({"level": lv, "x": round(x, 2), "x_lo": round(0.8 * x, 2), "x_hi": round(1.25 * x, 2),
                                "y": round(y, 2), "y_lo": round(y - 2, 2), "y_hi": round(y + 2, 2)})
                return out
            if m in traced:                                              # a trace, points on it, points beside it
                series.append({"method": m, "variant": "shared", "points": pts(0.0),
                               "trace": [{"x": 0.25 * i, "y": round(8 + 14 * j + 0.15 * i + k, 3)} for i in range(0, 101)]})
                series.append({"method": m, "points": pts(3.5 if traced.index(m) % 2 == 0 else -3.5)})
            else:
                series.append({"method": m, "curve": True, "points": pts(0.0)})
        series[-1]["label_point"] = len(levels) - 1
        for j, m in enumerate(singles):                                  # a regular grid: x = 32 + 9 j, y alternating
            s_ = {"method": m, "points": [{"x": 32.0 + 9 * j, "y": (15.0 if j % 2 == 0 else 45.0) + k}]}
            if m in ("linear", "dual"):
                s_["label_point"] = 0
            series.append(s_)
        series.append({"method": "cosine", "variant": "published", "points": [{"x": 32.0, "y": 70.0 + k}]})
        series.append({"method": "editor", "points": [{"x": 50.0, "y": 70.0 + k}, {"x": 68.0, "y": 70.0 + k}]})
        panels.append({"log": name, "title": name, "series": series})
    return {"schema": SCHEMA, "synthetic": True,
            "title": "Example of the frontier figure with made-up data",
            "note": "SYNTHETIC EXAMPLE DATA written by scripts/figs/fig_frontier.py --make-example from closed formulas "
                    "(parallel lines and a regular grid, methods in alphabetical order, offsets of alternating sign). Not a "
                    "result of any experiment; the position of a method means nothing. Never include a figure drawn from "
                    "this file in the paper.",
            "x": {"metric": "ff", "label": "False firing on Held-out (%)", "min": 0},
            "y": {"metric": "cov", "label": "Rephrasings covered (%)", "min": 0},
            "panels": panels}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("data", nargs="?", help="JSON file of the schema " + SCHEMA)
    ap.add_argument("--out", help="PDF file to write")
    ap.add_argument("--print-table", action="store_true", help="print the plotted values")
    ap.add_argument("--make-example", action="store_true", help="write and draw the synthetic example")
    a = ap.parse_args()
    if a.make_example:
        os.makedirs(style.EXAMPLE_DIR, exist_ok=True)
        with open(EXAMPLE_JSON, "w") as f:
            json.dump(make_example(), f, indent=1)
            f.write("\n")
        a.data, a.out = EXAMPLE_JSON, a.out or EXAMPLE_PDF
    if not a.data or not a.out:
        ap.error("give a data file and --out, or --make-example")
    with open(a.data) as f:
        doc = json.load(f)
    errors = validate(doc)
    if errors:
        sys.exit(f"{a.data} does not follow the schema {SCHEMA}:\n  " + "\n  ".join(errors))
    fig = draw(doc)
    path = style.save(fig, a.out, doc)
    print(f"wrote {path} ({fig.get_figwidth():.3f} x {fig.get_figheight():.3f} in)" + ("  [SYNTHETIC EXAMPLE DATA: not for the paper]" if doc["synthetic"] else ""))
    if a.print_table:
        print("\n".join(table(doc)))


if __name__ == "__main__":
    main()
