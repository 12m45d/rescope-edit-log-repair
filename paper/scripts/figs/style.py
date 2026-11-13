#!/usr/bin/env python3
"""Shared style of the figures of the paper: sizes that match the PVLDB layout, the fonts of the paper, one fixed
colour per METHOD FAMILY, one fixed marker per METHOD. Every figure script imports this module and nothing else for
its looks, so that a method is drawn the same way in every figure.

    import style
    style.apply()                                   # rcParams: fonts, sizes, hairline axes and grid, embedded fonts
    ncol, rows = style.legend_columns([style.series_label(m, v) for m, v in series])   # fits the legend to the width
    fig, axes = style.figure(ncols=2, width="column", panel_height=1.05, legend_rows=rows)
    style.plot_series(ax, "rescope", xs, ys, curve=True)       # colour, marker and label come from the registry
    style.plot_trace(ax, "rescope", xs_fine, ys_fine)          # a swept threshold: a curve without markers
    style.legend(fig, axes, ncol=ncol)              # one legend for the figure, above the panels
    style.save(fig, "figs/frontier.pdf", data)      # PDF with embedded fonts; refuses synthetic data in the paper

    python3 scripts/figs/style.py                   # self-check: palette validation, fonts, layout constants

LAYOUT (measured from the class with pdflatex, 2026-11-10: acmart sigconf + pvldb.sty, official fonts)
  text width 506.295pt = 7.006in; column separation 24pt; column width 241.148pt = 3.337in; text height 626pt.
  Body text 9/11pt; \\footnotesize 7/8pt (tables, and the figure text here); \\scriptsize 6/7pt (small annotations).
  A figure is drawn at its FINAL size (width="column" or "text") and included with width=\\columnwidth / \\textwidth, so
  a 7pt label is 7pt on the page. Never scale a figure in LaTeX.

FONTS
  The paper is set in Linux Libertine (text) and Linux Biolinum (sans). Figure text uses Linux Biolinum O, registered
  from the project-local TeX tree (ACMFONTS, as in the Makefile); without that tree the fallback is DejaVu Sans and
  apply() says so. Fonts are embedded as Type 42 (never Type 3).

COLOUR (the dataviz method: colour by the job it does; palette values from its documented reference palette; the
palette is VALIDATED by computation, not by eye)
  The job is identity of the method FAMILY, a categorical job. Frontier and scatter figures put any two marks side by
  side, so the all-pairs test applies, and the documented palette carries at most THREE hues in such forms. Hence:
    ours            blue    #2a78d6   \\sys, \\sysW, the complaint operators
    post-hoc gates  orange  #eb6834   Cosine, Ball, Router, Z-Score, Exact
    linear repairs  aqua    #199e70   Comp-B, Re-Edit, RB-Union, RB-Tot
    context         ink     #0b0b0b / #52514e / #898781   the deployed log, the editor's working points, Dual: reference
                    points, drawn in neutral ink with their own markers and direct labels (not a fourth hue)
  Methods of one family share the hue and differ by MARKER (hue x shape); a variant of a method (published threshold
  instead of the calibrated one) is the same marker, hollow. Identity is therefore never carried by colour alone: the
  figure survives a grey-scale print. Text, ticks and legend labels wear ink colours, never the series colour.
  Validation (skill script validate_palette.py, 2026-11-10; surface = white paper):
    python3 validate_palette.py "#2a78d6,#eb6834,#199e70" --mode light --surface "#ffffff" --pairs all
      lightness band PASS; chroma floor PASS; CVD separation PASS (worst pair aqua-orange, Delta E 8.4, protan; target 8);
      normal-vision floor PASS (worst pair aqua-blue, 21.6; floor 15); contrast vs surface PASS (all three >= 3:1)
    The documented light step of aqua (#1baf7a) has 2.82:1 on white; the palette's darker aqua step is used instead
    (snap-to-passing). check_palette() below recomputes the same quantities (OKLab distance x 100 under the Machado
    et al. 2009 simulation, WCAG contrast), so a change of FAMILY_COLOUR fails the self-check instead of passing by eye.
  Not used: a fourth hue, a rainbow, status colours, dual axes, dashed grid lines.

MARKS  lines 1.0pt, round joins; markers 3.9pt with a 0.5pt ring in the surface colour, so that overlapping marks stay
  legible; intervals as hairlines in the series colour; grid and axes: solid hairlines one step off the surface.
  (Print sizes: the screen specification of the method, 2px lines and markers of at least 8px, corresponds to about
  1pt and 4pt in a figure of one column, 3.34in wide, in which the text is 7pt.)
  A marker is a SHAPE and a FILL. The shape names the method; the fill names how its threshold was set: full = the
  default calibration (two-block rule), half = one shared scale (variant="shared"), hollow = the published threshold
  (variant="published"; also the editor's other working points, hollow stars). Shapes are unique across all methods:
  the three hues have similar lightness, so in a grey-scale print only shape and fill tell them apart (Dual, a thin
  diamond in dark ink, and Ball, a wide orange diamond, are the closest pair: give Dual a direct label).
  A TRACE is a curve without markers (a threshold swept over a fine grid); marked points lie on or beside it.
"""
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(os.path.dirname(HERE))
EXAMPLE_DIR = os.path.join(HERE, "example")
ACMFONTS = os.environ.get("ACMFONTS", os.path.join(_RP.EXT, "texmf-acm"))    # a TeX tree with the Libertine fonts of acmart (optional)

# ------------------------------------------------------------------------------------------------ layout
TEX_PT = 72.27                                  # TeX points per inch
TEXT_WIDTH_PT, COLUMN_SEP_PT, TEXT_HEIGHT_PT = 506.295, 24.0, 626.0
COLUMN_WIDTH_PT = (TEXT_WIDTH_PT - COLUMN_SEP_PT) / 2
COLUMN_IN, TEXT_IN = COLUMN_WIDTH_PT / TEX_PT, TEXT_WIDTH_PT / TEX_PT
FONT_PT, SMALL_PT = 7.0, 6.0                    # \footnotesize and \scriptsize of the paper
LINE_PT, MARKER_PT, RING_PT, HAIR_PT = 1.0, 3.9, 0.5, 0.4

# ------------------------------------------------------------------------------------------------ colour
SURFACE = "#ffffff"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
FAMILY_COLOUR = {"ours": "#2a78d6", "gate": "#eb6834", "linear_repair": "#199e70"}       # the three hues, in slot order
CONTEXT_COLOUR = {"deployed": INK, "editor": MUTED, "answer_match": INK2}                 # neutral reference points
FAMILY_LABEL = {"ours": "ours", "gate": "post-hoc gates", "linear_repair": "linear repairs", "deployed": "deployed log",
                "editor": "editor, other ridges", "answer_match": "answer match"}

# method id (the ids of the result JSON files) -> family, label as printed, marker shape, fill ("full" | "none" = hollow
# | "line" = a marker made of strokes). One shape per method, in every figure; variants change the fill (method_style).
METHODS = {
    "rescope":       ("ours", "ReScope", "o", "full"),
    "rescope_where": ("ours", "ReScope-W", "2", "line"),
    "rescope_where_start": ("ours", "ReScope-W from none", "1", "line"),
    "cosine":        ("gate", "Cosine", "s", "full"),
    "ball":          ("gate", "Ball", "D", "full"),
    "router":        ("gate", "Router", "^", "full"),
    "zscore":        ("gate", "Z-Score", "v", "full"),
    "exact":         ("gate", "Exact", "X", "full"),
    "comp_b":        ("linear_repair", "Comp-B", "P", "full"),
    "reedit_b":      ("linear_repair", "Re-Edit", "p", "full"),
    "rb_union":      ("linear_repair", "RB-Union", "<", "full"),
    "rb_tot":        ("linear_repair", "RB-Tot", ">", "full"),
    "linear":        ("deployed", "Linear", "*", "full"),
    "editor":        ("editor", "Linear, other ridge", "*", "none"),
    "dual":          ("answer_match", "Dual", "d", "full"),
    "preedit":       ("editor", "Pre-edit model", "+", "line"),
    "white":         ("gate", "Whitened coordinate", "h", "full"),
    "retr":          ("answer_match", "Retrieval (requests)", "8", "full"),
}
MARKER_SCALE = {"*": 1.55, "P": 1.2, "X": 1.2, "d": 1.2, "p": 1.15, "2": 1.7, "1": 1.7, "+": 1.5}            # optical size of the thin shapes
VARIANTS = {"default": "", "shared": " (shared scale)", "published": " (published)", "allpos": " (all positions)"}       # variant -> suffix of the label


def colour(method):
    fam = METHODS[method][0]
    return FAMILY_COLOUR.get(fam) or CONTEXT_COLOUR[fam]


def method_style(method, variant="default"):
    """Keyword arguments for Axes.plot of a method. variant: "default" (the method's marker, filled: its default
    calibration), "shared" (the same marker, half filled: one shared scale) or "published" (the same marker, hollow:
    the gate at its published threshold)."""
    if method not in METHODS:
        raise KeyError(f"unknown method {method!r}: add it to style.METHODS (family, label, marker, fill)")
    if variant not in VARIANTS:
        raise ValueError(f"variant: one of {sorted(VARIANTS)}")
    fam, label, marker, fill = METHODS[method]
    c = colour(method)
    if variant != "default":
        if fill != "full":
            raise ValueError(f"{method}: variants exist for methods with a full marker only")
        fill = {"published": "none", "shared": "bottom", "allpos": "none"}[variant]
    st = dict(color=c, marker=marker, markersize=MARKER_PT * MARKER_SCALE.get(marker, 1.0), linewidth=LINE_PT,
              solid_capstyle="round", solid_joinstyle="round", label=label + VARIANTS[variant], zorder=3 if fam == "ours" else 2.5)
    if fill == "full":                                  # filled mark with a ring in the surface colour
        st.update(markerfacecolor=c, markeredgecolor=SURFACE, markeredgewidth=RING_PT)
    elif fill == "none":                                # hollow mark
        st.update(markerfacecolor=SURFACE, markeredgecolor=c, markeredgewidth=0.8)
    elif fill == "line":                                # a marker made of strokes
        st.update(markeredgecolor=c, markeredgewidth=1.0)
    else:                                               # half-filled mark
        st.update(fillstyle=fill, markerfacecolor=c, markerfacecoloralt=SURFACE, markeredgecolor=c, markeredgewidth=0.6)
    return st


def plot_trace(ax, method, xs, ys):
    """A traced curve of a method (a threshold swept over a fine grid): a line in the family colour, no markers, no
    legend entry of its own (the marked points of the method name it)."""
    return ax.plot(xs, ys, color=colour(method), linewidth=LINE_PT * 0.8, alpha=0.8, solid_capstyle="round", solid_joinstyle="round",
                   zorder=2.0, label="_nolegend_")[0]


# ------------------------------------------------------------------------------------------------ palette checks
_MACHADO = {"protan": ((0.152286, 1.052583, -0.204868), (0.114503, 0.786281, 0.099216), (-0.003882, -0.048116, 1.051998)),
            "deutan": ((0.367322, 0.860646, -0.227968), (0.280085, 0.672501, 0.047413), (-0.011820, 0.042940, 0.968881))}


def _lin(h):
    h = h.lstrip("#")
    return tuple((c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4) for c in (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)))


def _oklab(rgb):
    r, g, b = rgb
    l, m, s = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b, 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b,
               0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    l, m, s = l ** (1 / 3), m ** (1 / 3), s ** (1 / 3)
    return (0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s, 1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s)


def _sim(rgb, kind):
    return tuple(max(0.0, min(1.0, sum(a * b for a, b in zip(row, rgb)))) for row in _MACHADO[kind])


def delta_e(h1, h2, kind=None):
    """Euclidean distance in OKLab x 100; kind = "protan" | "deutan" simulates the deficiency (Machado et al. 2009)."""
    a, b = (_oklab(_sim(_lin(h), kind) if kind else _lin(h)) for h in (h1, h2))
    return 100 * sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def contrast(h1, h2):
    lum = sorted((0.2126 * r + 0.7152 * g + 0.0722 * b for r, g, b in (_lin(h1), _lin(h2))), reverse=True)
    return (lum[0] + 0.05) / (lum[1] + 0.05)


def check_palette(colours=None, surface=SURFACE):
    """The computable checks of the dataviz method on the family hues, ALL pairs (scatter-type figures): lightness
    band 0.43-0.77 and chroma >= 0.10 (OKLCH), CVD separation >= 8 (protan, deutan), normal-vision separation >= 15,
    contrast against the surface >= 3. Returns (ok, lines)."""
    cols = list((colours or FAMILY_COLOUR).values())
    pairs = [(a, b) for i, a in enumerate(cols) for b in cols[i + 1:]]
    lab = [_oklab(_lin(c)) for c in cols]
    light = [round(x[0], 3) for x in lab]
    chroma = [round((x[1] ** 2 + x[2] ** 2) ** 0.5, 3) for x in lab]
    cvd = min(delta_e(a, b, k) for a, b in pairs for k in _MACHADO)
    normal = min(delta_e(a, b) for a, b in pairs)
    con = min(contrast(c, surface) for c in cols)
    checks = [("lightness band 0.43-0.77", all(0.43 <= x <= 0.77 for x in light), str(light)),
              ("chroma floor 0.10", all(x >= 0.10 for x in chroma), str(chroma)),
              ("CVD separation (all pairs) >= 8", cvd >= 8.0, f"{cvd:.1f}"),
              ("normal-vision separation (all pairs) >= 15", normal >= 15.0, f"{normal:.1f}"),
              (f"contrast against {surface} >= 3", con >= 3.0, f"{con:.2f}")]
    return all(ok for _, ok, _ in checks), [f"[{'PASS' if ok else 'FAIL'}] {name}: {val}" for name, ok, val in checks]


# ------------------------------------------------------------------------------------------------ matplotlib
_FONT = {"family": None}


def _register_fonts():
    """Linux Biolinum O from the project-local TeX tree; returns the family name that matplotlib should use."""
    if _FONT["family"]:
        return _FONT["family"]
    import glob
    from matplotlib import font_manager as fm
    d = os.path.join(ACMFONTS, "texmf/fonts/opentype/public/libertine")
    for f in sorted(glob.glob(os.path.join(d, "LinBiolinum_R*.otf"))):
        fm.fontManager.addfont(f)
    names = {f.name for f in fm.fontManager.ttflist}
    _FONT["family"] = "Linux Biolinum O" if "Linux Biolinum O" in names else "DejaVu Sans"
    return _FONT["family"]


def apply(verbose=False):
    """Set the rcParams of the paper's figures (call once, before any figure is created)."""
    import matplotlib
    matplotlib.use("pdf", force=False) if matplotlib.get_backend().lower() not in ("pdf", "agg") else None
    fam = _register_fonts()
    if fam != "Linux Biolinum O":
        print("style: Linux Biolinum O not found (ACMFONTS); falling back to DejaVu Sans -- NOT the paper's typography", file=sys.stderr)
    elif verbose:
        print("style: figure text in Linux Biolinum O")
    matplotlib.rcParams.update({
        "font.family": [fam, "DejaVu Sans"], "font.size": FONT_PT, "axes.labelsize": FONT_PT, "axes.titlesize": FONT_PT,
        "xtick.labelsize": FONT_PT, "ytick.labelsize": FONT_PT, "legend.fontsize": FONT_PT, "figure.titlesize": FONT_PT,
        "mathtext.fontset": "custom", "mathtext.rm": fam, "mathtext.it": fam + ":italic", "mathtext.bf": fam + ":bold",
        "text.color": INK, "axes.labelcolor": INK2, "axes.titlecolor": INK, "xtick.color": AXIS, "ytick.color": AXIS,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.edgecolor": AXIS, "axes.linewidth": HAIR_PT, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": HAIR_PT, "grid.linestyle": "-",
        "xtick.major.width": HAIR_PT, "ytick.major.width": HAIR_PT, "xtick.major.size": 2.0, "ytick.major.size": 2.0,
        "xtick.major.pad": 1.5, "ytick.major.pad": 1.5, "axes.labelpad": 2.0, "axes.titlepad": 3.0,
        "lines.linewidth": LINE_PT, "lines.markersize": MARKER_PT, "lines.solid_capstyle": "round",
        "legend.frameon": False, "legend.handlelength": 1.4, "legend.handletextpad": 0.4, "legend.columnspacing": 1.0,
        "legend.borderaxespad": 0.0, "legend.labelspacing": 0.25,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": SURFACE, "savefig.pad_inches": 0.0, "figure.dpi": 150,
    })
    return fam


LEGEND_ROW_PT, TITLE_PT, BOTTOM_PT, LEFT_PT = 8.6, 11.0, 23.5, 24.0       # room for a legend row, panel titles, x and y labels
LEGEND_TOP_PT = 2.0                                                       # between the top edge and the legend


def figure(ncols=1, nrows=1, width="column", panel_height=1.05, legend_rows=0, titles=True, sharey=True, sharex=False):
    """A figure at its final size: width "column" (3.337in) or "text" (7.006in); panel_height is the height of ONE
    panel in inches, to which the room for x labels, panel titles and legend_rows rows of a legend above the panels is
    added. Returns (fig, list of axes)."""
    import matplotlib.pyplot as plt
    w = {"column": COLUMN_IN, "text": TEXT_IN}[width]
    head = (legend_rows * LEGEND_ROW_PT + (3.0 + LEGEND_TOP_PT if legend_rows else 0.0) + (TITLE_PT if titles else 3.0)) / 72.0
    gap = (BOTTOM_PT + (TITLE_PT if titles else 3.0)) / 72.0                       # between two rows of panels
    height = head + nrows * panel_height + (nrows - 1) * gap + BOTTOM_PT / 72.0
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, height), sharey=sharey, sharex=sharex, squeeze=False)
    fig.subplots_adjust(left=LEFT_PT / (w * 72.0), right=1.0 - 5.0 / (w * 72.0), bottom=BOTTOM_PT / (height * 72.0),
                        top=1.0 - head / height, wspace=0.08 if sharey else 0.30, hspace=gap / panel_height)
    return fig, [ax for row in axes for ax in row]


def set_limits(axes, xmin=None, xmax=None, ymin=None, ymax=None, pad=0.10):
    """Common limits for panels that share an axis: the range of everything drawn (points and intervals), padded;
    explicit bounds win. Call after all series are drawn."""
    xs = [v for ax in axes for ln in ax.get_lines() for v in ln.get_xdata()]
    ys = [v for ax in axes for ln in ax.get_lines() for v in ln.get_ydata()]

    def lim(vals, lo, hi):
        a, b = min(vals), max(vals)
        d = (b - a) or 1.0
        return (a - pad * d if lo is None else lo), (b + pad * d if hi is None else hi)
    for ax in axes:
        ax.set_xlim(*lim(xs, xmin, xmax))
        ax.set_ylim(*lim(ys, ymin, ymax))


def plot_series(ax, method, xs, ys, curve=False, variant="default", xerr=None, yerr=None):
    """One series of a method: a curve through its points (calibration levels) or separate points. xerr / yerr:
    lists of (low, high) intervals in data units, drawn as hairlines."""
    st = method_style(method, variant)
    if xerr or yerr:
        for k, (x, y) in enumerate(zip(xs, ys)):
            if xerr and xerr[k] is not None:
                ax.plot(list(xerr[k]), [y, y], color=st["color"], linewidth=HAIR_PT + 0.2, alpha=0.75, zorder=st["zorder"] - 0.3, solid_capstyle="butt")
            if yerr and yerr[k] is not None:
                ax.plot([x, x], list(yerr[k]), color=st["color"], linewidth=HAIR_PT + 0.2, alpha=0.75, zorder=st["zorder"] - 0.3, solid_capstyle="butt")
    if not curve:
        st["linestyle"] = "none"
    return ax.plot(xs, ys, **st)[0]


def direct_label(ax, x, y, text, dx=3.0, dy=3.0, ha="left", va="bottom"):
    """A sparing direct label next to a mark (ink colour, never the series colour). Use for at most four marks."""
    return ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=SMALL_PT, color=INK2, zorder=5)


LEGEND_PT = SMALL_PT + 0.5
LEGEND_KW = dict(fontsize=LEGEND_PT, handlelength=1.0, handletextpad=0.4, columnspacing=0.9, borderpad=0.0, labelspacing=0.25)


def series_label(method, variant="default"):
    return METHODS[method][1] + VARIANTS[variant]


def legend_columns(labels, width="column", ncol_max=4):
    """(number of columns, number of rows) of a legend with these labels that fits the figure width: the largest
    number of columns, at most ncol_max, whose columns (handle + label, as matplotlib fills them) fit side by side.
    Call it BEFORE style.figure, which needs the number of rows."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath
    fp = FontProperties(family=_register_fonts(), size=LEGEND_PT)
    text = [TextPath((0, 0), l, prop=fp).get_extents().width for l in labels]             # points
    avail = {"column": COLUMN_IN, "text": TEXT_IN}[width] * 72.0 - 4.0
    n = len(labels)
    for ncol in range(min(ncol_max, n), 0, -1):
        q, r = divmod(n, ncol)                                   # matplotlib: the first r columns hold q + 1 entries
        cols, i = [], 0
        for c in range(ncol):
            k = q + (1 if c < r else 0)
            cols.append(max(text[i:i + k]) + (LEGEND_KW["handlelength"] + LEGEND_KW["handletextpad"]) * LEGEND_PT)
            i += k
        if sum(cols) + (ncol - 1) * LEGEND_KW["columnspacing"] * LEGEND_PT <= avail or ncol == 1:
            return ncol, -(-n // ncol)


def legend(fig, axes, ncol=4, order=None):
    """One legend for the whole figure, above the panels: every drawn series once, methods in the order of
    style.METHODS, variants in the order of style.VARIANTS. ncol: from legend_columns()."""
    handles = {}
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l, h)
    labels = [m[1] for m in METHODS.values()]
    suffixes = [v for v in VARIANTS.values() if v]

    def key(l):                                      # methods in the order of METHODS, variants in the order of VARIANTS
        base, var = l, 0
        for k, suf in enumerate(suffixes, 1):
            if l.endswith(suf):
                base, var = l[:-len(suf)], k
        return (labels.index(base) if base in labels else 99, var)
    names = order or sorted(handles, key=key)
    return fig.legend([handles[n] for n in names], names, loc="upper center", ncol=ncol,
                      bbox_to_anchor=(0.5, 1.0 - LEGEND_TOP_PT / (fig.get_figheight() * 72.0)), **LEGEND_KW)


def watermark(ax, text="SYNTHETIC EXAMPLE DATA"):
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", rotation=24, fontsize=FONT_PT + 1,
            color=MUTED, alpha=0.45, zorder=1, fontweight="bold", clip_on=True)


def save(fig, path, data=None):
    """Write the figure as PDF (fonts embedded, no creation date: the file is reproducible). data: the JSON document that
    was plotted; if it is marked "synthetic", every panel must carry the watermark and the file may only be written
    below scripts/figs/example/ -- synthetic data never enters the directory from which the paper includes figures."""
    path = os.path.abspath(path)
    synthetic = bool(data and data.get("synthetic"))
    if synthetic:
        if os.path.commonpath([path, EXAMPLE_DIR]) != EXAMPLE_DIR:
            raise SystemExit(f"refused: synthetic example data may only be drawn into {EXAMPLE_DIR}, not {path}")
        if "synthetic" not in os.path.basename(path).lower():
            raise SystemExit("refused: the file name of a figure of synthetic data must contain 'synthetic'")
    elif os.path.commonpath([path, EXAMPLE_DIR]) == EXAMPLE_DIR:
        raise SystemExit("refused: scripts/figs/example/ holds synthetic examples only")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    meta = {"CreationDate": None, "Creator": "scripts/figs (style.py)",
            "Title": ("SYNTHETIC EXAMPLE, not a result: " if synthetic else "") + str((data or {}).get("title", os.path.basename(path))),
            "Keywords": "SYNTHETIC EXAMPLE DATA" if synthetic else "rescope"}
    fig.savefig(path, format="pdf", metadata=meta)
    return path


def _self_check():
    ok, lines = check_palette()
    print("palette (family hues, all pairs, surface %s):" % SURFACE)
    for line in lines:
        print("  " + line)
    fam = apply()
    print(f"font: {fam}; column width {COLUMN_IN:.3f} in, text width {TEXT_IN:.3f} in; text {FONT_PT:g} pt, small {SMALL_PT:g} pt")
    marks = [(m[2], m[3]) for m in METHODS.values()]
    dup = sorted({m for m in marks if marks.count(m) > 1})
    shapes = [m[2] for m in METHODS.values() if m[3] != "none"]                # hollow marks are variants of a full one
    dup += sorted({m for m in shapes if shapes.count(m) > 1})
    print("markers: one shape per method" if not dup else f"markers used twice: {dup}")
    return ok and not dup and fam == "Linux Biolinum O"


if __name__ == "__main__":
    sys.exit(0 if _self_check() else 1)
