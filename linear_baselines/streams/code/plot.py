"""P12 -- out/results.json -> figures/*.png  (static figures; the numbers are in results.md = the table view).

  fig_heldout_d1.png     HELD-OUT D1 rate versus n per method, one panel per log (full HELD-OUT pool = the protocol)
  fig_heldout_d1_disjoint.png   the same on the held-out rows whose query does not occur in the REPAIR pool
  fig_rel_exact.png      Rel exact (all 100 logged edits) versus n per method, one panel per log
  fig_edits_broken.png   logged-edit checks broken (Rel + T-Gen + I-Gen) versus n per method, one panel per log
  fig_tradeoff.png       HELD-OUT D1 rate against Rel exact, every configuration, one panel per log

Encoding (dataviz skill; palette validated with scripts/validate_palette.js --pairs all): hue = method FAMILY
(three categorical slots: compensation / re-edit / rollback), marker = variant inside the family, so identity
never rests on colour alone; a legend plus selective end labels; one y axis per panel; solid hairline grid.
Lines = mean over the seeds available (3 for n <= 20, 1 for n >= 50); thin vertical bars = min..max over seeds.
A sequential COMP variant is drawn only if it differs from the joint solve of the same complaints (they are the
same operator up to round-off: results.json seq_vs_batch_identity); otherwise the joint line is labelled for both.

usage: python plot.py
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

P12 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(P12, "figures")
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
C_COMP, C_REEDIT, C_RB = "#2a78d6", "#eb6834", "#1baf7a"          # categorical slots 1-3
NAMES = {"TB-Q100-ae-l2_3": "Qwen2.5-VL-3B (TB-Q100-ae, l2 = 3)", "TB-L100-ae-l2_3": "LLaVA-1.5-7B (TB-L100-ae, l2 = 3)"}
# method -> (colour, marker, filled, label)
STYLE = {"COMP_B": (C_COMP, "o", True, "COMP-B"), "COMP_S": (C_COMP, "v", True, "COMP-S"),
         "COMP_BP": (C_COMP, "o", False, "COMP-B+"), "COMP_SP": (C_COMP, "v", False, "COMP-S+"),
         "COMP_B_ALLPOS": (C_COMP, "^", True, "COMP-B-ALLPOS"),
         "REEDIT_B": (C_REEDIT, "s", True, "REEDIT-B"), "REEDIT_S": (C_REEDIT, "s", False, "REEDIT-S"),
         "RB_TOT": (C_RB, "D", True, "RB-TOT"), "RB_UNION": (C_RB, "D", False, "RB-UNION")}
ORDER = ["COMP_B", "COMP_S", "COMP_BP", "COMP_SP", "COMP_B_ALLPOS", "REEDIT_B", "REEDIT_S", "RB_TOT", "RB_UNION"]
NS = ["1", "2", "5", "10", "20", "50", "100", "ALL"]


def series_plan(L):
    """Methods to draw and their labels; a sequential COMP variant identical to the joint solve is merged."""
    ident = {}
    for i in L["seq_vs_batch_identity"]:
        m = i["config"].split("__")[0]
        same = i["alias"] or (i["same_heldout_D1"] and i["same_reported_resolved"] and i["same_edits"])
        ident[m] = ident.get(m, True) and bool(same)
    plan = []
    for m in ORDER:
        if m not in L["curves"]:
            continue
        if m in ("COMP_S", "COMP_SP") and ident.get(m, False):
            continue
        col, mk, filled, lab = STYLE[m]
        if m == "COMP_B" and ident.get("COMP_S", False):
            lab = "COMP-B = COMP-S"
        if m == "COMP_BP" and ident.get("COMP_SP", False):
            lab = "COMP-B+ = COMP-S+"
        plan.append((m, col, mk, filled, lab))
    return plan


def style_axes(ax):
    ax.set_facecolor(SURF)
    ax.grid(color=GRID, lw=1, ls="-")
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)


def spread_labels(items, min_gap):
    """items: [(y, text, colour)] -> y positions pushed apart (top-down) so end labels do not collide."""
    items = sorted(items, key=lambda t: -t[0])
    out, last = [], None
    for y, text, col in items:
        yy = y if last is None else min(y, last - min_gap)
        out.append((y, yy, text, col))
        last = yy
    return out


def seeds_note(R):
    parts = []
    for lg in NAMES:
        if lg not in R["logs"]:
            continue
        cs = [c for cn in R["logs"][lg]["curves"].values() for c in cn.values()]
        ns = sorted({len(c["seeds"]) for c in cs})
        parts.append(f"{NAMES[lg].split(' (')[0]}: {ns[0]}" + (f"-{ns[-1]}" if len(ns) > 1 else "") + " seed(s) per point")
    return ("Lines: mean over the seeds run (" + "; ".join(parts) + "; protocol: 3 seeds for n <= 20, seed 0 for n >= 50); "
            "vertical bars: min..max over seeds.")


def curve_fig(R, key, ylabel, fname, title, ref_none=None, ref_all=None, ymax=None, note=""):
    logs = [lg for lg in NAMES if lg in R["logs"]]
    any_fp = [False]
    fig, axs = plt.subplots(1, len(logs), figsize=(7.8 * len(logs), 5.6), squeeze=False, facecolor=SURF)
    for ax, lg in zip(axs[0], logs):
        L = R["logs"][lg]
        style_axes(ax)
        n_all = L["n_all"]
        xs_all = {n: (n_all if n == "ALL" else int(n)) for n in NS}
        ends, top = [], 0.0
        for m, col, mk, filled, lab in series_plan(L):
            cn = L["curves"][m]
            pts = [(xs_all[n], cn[n][key]) for n in NS if n in cn and cn[n][key] is not None]
            if not pts:
                continue
            x = [p[0] for p in pts]
            y = [p[1]["mean"] for p in pts]
            ax.plot(x, y, color=col, lw=2, solid_capstyle="round", solid_joinstyle="round", zorder=3)
            for xi, p in pts:
                if p["n_seeds"] > 1:
                    ax.plot([xi, xi], [p["min"], p["max"]], color=col, lw=1, alpha=0.55, zorder=2)
            ax.plot(x, y, ls="none", marker=mk, ms=7.5, mfc=col if filled else SURF, mec=col if not filled else SURF,
                    mew=1.6, zorder=4, label=lab)
            for n in NS:                                   # first-pass REEDIT records (re-run missing): dagger
                if n in cn and cn[n].get("first_pass") and cn[n][key] is not None:
                    ax.annotate("\u2020", (xs_all[n], cn[n][key]["mean"]), xytext=(0, 7), textcoords="offset points", fontsize=9,
                                color=INK, ha="center", va="bottom", zorder=5)
                    any_fp[0] = True
            ends.append((y[-1], lab, col, x[-1]))
            top = max(top, max(p[1]["max"] for p in pts))
        refs = []
        if ref_none is not None:
            v = ref_none(L)
            ax.plot([0.85, n_all], [v, v], color=INK2, lw=1.4, zorder=1)
            refs.append((v, f"unrepaired log ({v:.1f})", INK2, n_all))
            top = max(top, v)
        if ref_all is not None:
            v = ref_all(L)
            ax.plot([0.85, n_all], [v, v], color=MUTED, lw=1.0, zorder=1)
            refs.append((v, f"W_0, all edits rolled back ({v:.1f})", MUTED, n_all))
        ax.set_xscale("log")
        ax.set_xticks([xs_all[n] for n in NS])
        ax.set_xticklabels([n if n != "ALL" else f"ALL\n({n_all})" for n in NS])
        ax.minorticks_off()
        hi = ymax if ymax is not None else top * 1.08 + 1e-9
        ax.set_ylim(0, hi)
        ax.set_xlim(0.85, n_all * 4.6)
        gap = hi * 0.047
        # end labels: series that reach n = ALL (and the reference lines) are labelled in the right margin with a
        # leader line; a series that stops earlier is labelled next to its last point
        full = [e for e in ends if e[3] == n_all] + refs
        for y0, yy, text, col in spread_labels([(e[0], e[1], e[2]) for e in full], gap):
            ax.annotate(text, (n_all, y0), xytext=(n_all * 1.3, yy), textcoords="data", fontsize=8.5, color=INK, va="center",
                        arrowprops=dict(arrowstyle="-", color=col, lw=0.8, shrinkA=0, shrinkB=3))
        for y0, text, col, xl in [e for e in ends if e[3] != n_all]:
            ax.annotate(text, (xl, y0), xytext=(7, 9), textcoords="offset points", fontsize=8.5, color=INK, va="bottom",
                        arrowprops=dict(arrowstyle="-", color=col, lw=0.8, shrinkA=0, shrinkB=3))
        ax.set_xlabel("n reported complaints repaired (REPAIR pool; log scale)", color=INK2, fontsize=9.5)
        ax.set_ylabel(ylabel, color=INK2, fontsize=9.5)
        ax.set_title(NAMES[lg], color=INK, fontsize=11, loc="left")
        ax.legend(fontsize=8.5, frameon=False, loc="upper left", bbox_to_anchor=(0.0, -0.25), ncol=4, handletextpad=0.4,
                  columnspacing=1.2, labelcolor=INK)
    fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha="left")
    if any_fp[0]:
        note += ("  \u2020 first-pass REEDIT record (harness default call: complaints without an oracle z* not installed); "
                 "the re-run did not get GPU time.")
    if note:
        fig.text(0.01, 0.005, note, color=MUTED, fontsize=8, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fn = os.path.join(FIG, fname)
    fig.savefig(fn, dpi=160, facecolor=SURF)
    plt.close(fig)
    print(fn)


def tradeoff_fig(R, fname):
    logs = [lg for lg in NAMES if lg in R["logs"]]
    fig, axs = plt.subplots(1, len(logs), figsize=(7.4 * len(logs), 5.4), squeeze=False, facecolor=SURF)
    tol = R["protocol"]["rel_tol"]
    for ax, lg in zip(axs[0], logs):
        L = R["logs"][lg]
        style_axes(ax)
        none = L["rows"]["NONE"]
        rel0, d0 = none["edits"]["rel"]["exact"], none["heldout"]["D1_rate"]
        ax.axvspan(rel0 - tol, 101, color=INK2, alpha=0.06, lw=0, zorder=0)
        ax.annotate(f"Rel exact within {tol:g} points\nof the unrepaired log", (rel0 - tol, 2), xytext=(3, 0),
                    textcoords="offset points", fontsize=8, color=INK2, va="bottom")
        for m, col, mk, filled, lab in series_plan(L):
            rows = [r for r in L["rows"].values() if r["method"] == m]
            ax.plot([r["edits"]["rel"]["exact"] for r in rows], [r["heldout"]["D1_rate"] for r in rows], ls="none", marker=mk,
                    ms=6.5, mfc=col if filled else SURF, mec=col if not filled else SURF, mew=1.4, zorder=3, label=lab,
                    alpha=0.9)
        ax.plot([rel0], [d0], ls="none", marker="*", ms=15, mfc=INK, mec=SURF, mew=1.5, zorder=5, label="unrepaired log")
        if "ALL" in L["rows"]:
            a = L["rows"]["ALL"]
            ax.plot([a["edits"]["rel"]["exact"]], [a["heldout"]["D1_rate"]], ls="none", marker="X", ms=10, mfc=MUTED,
                    mec=SURF, mew=1.5, zorder=5, label="W_0 (all edits rolled back)")
        if L["operating_points"]:
            o = L["operating_points"][0]
            ax.annotate(f"best within tolerance:\n{STYLE[o['method']][3]}, n = {o['n']} ({o['heldout_D1_rate']:.1f}%)",
                        (o["rel_exact"], o["heldout_D1_rate"]), xytext=(-150, -34), textcoords="offset points", fontsize=8.5,
                        color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
        ax.set_xlim(-3, 101)
        ax.set_ylim(0, max(r["heldout"]["D1_rate"] for r in L["rows"].values()) * 1.08)
        ax.set_xlabel("Rel exact after the repair (% of the 100 logged edits)", color=INK2, fontsize=9.5)
        ax.set_ylabel("HELD-OUT D1 rate after the repair (%)", color=INK2, fontsize=9.5)
        ax.set_title(NAMES[lg], color=INK, fontsize=11, loc="left")
        ax.legend(fontsize=8.5, frameon=False, loc="upper left", bbox_to_anchor=(0.0, -0.16), ncol=4, handletextpad=0.4,
                  columnspacing=1.2, labelcolor=INK)
    fig.suptitle("What a linear repair buys and costs: every configuration (method x n x seed)", color=INK, fontsize=12,
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fn = os.path.join(FIG, fname)
    fig.savefig(fn, dpi=160, facecolor=SURF)
    plt.close(fig)
    print(fn)


def main():
    R = json.load(open(os.path.join(P12, "out", "results.json")))
    os.makedirs(FIG, exist_ok=True)
    note = seeds_note(R)
    curve_fig(R, "heldout_D1_rate", "HELD-OUT D1 rate (% of valid held-out rows)", "fig_heldout_d1.png",
              "Unreported complaints after a linear repair of n reported ones -- full HELD-OUT pool (protocol)", note=note,
              ref_none=lambda L: L["rows"]["NONE"]["heldout"]["D1_rate"], ref_all=lambda L: L["rows"]["ALL"]["heldout"]["D1_rate"])
    curve_fig(R, "heldout_disjoint_D1_rate", "HELD-OUT D1 rate (% of the disjoint valid held-out rows)", "fig_heldout_d1_disjoint.png",
              "Unreported complaints after a linear repair of n reported ones -- held-out rows whose query does not occur in the REPAIR pool",
              note=note, ref_none=lambda L: L["rows"]["NONE"]["heldout_disjoint"]["D1_rate"],
              ref_all=lambda L: L["rows"]["ALL"]["heldout_disjoint"]["D1_rate"])
    curve_fig(R, "rel_exact", "Rel exact (% of the 100 logged edits)", "fig_rel_exact.png",
              "Logged edits still installed after a linear repair of n reported complaints", note=note, ymax=100,
              ref_none=lambda L: L["rows"]["NONE"]["edits"]["rel"]["exact"], ref_all=lambda L: L["rows"]["ALL"]["edits"]["rel"]["exact"])
    curve_fig(R, "edits_broken_total", "edit checks broken (of 300: Rel + T-Gen + I-Gen)", "fig_edits_broken.png",
              "Logged-edit checks broken by a linear repair of n reported complaints", note=note)
    tradeoff_fig(R, "fig_tradeoff.png")


if __name__ == "__main__":
    main()
