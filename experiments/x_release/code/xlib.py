"""Fix-up of E3 (amendment of 2026-11-12 06:16 UTC): shared library of e3_qwen/code/fixup_*.py, e3_llava/code/fixup_*.py and
x_release/code/*.py.  ADDITIONS ONLY: every number is recomputed from the stored per-row files (out/rows), the behavioural cell records
(out/behave) and the gate-level cells (out/gate: all 20 calibration draws) of the two E3 experiments, of E1 / E2 (development logs) and
of the prefix logs.  No model is run, no existing file is written; nothing is selected by looking at results (the reference gates are
read from e1_main/out/best_cosine_gate.json through the driver's reader fx.review1.selected).

The two E3 experiments were written by different agents; class Backbone hides the differences of their layouts:
  Qwen   out/gate/<log>/_context.json (hd: near flags over the heldout_disjoint rows), cells: per_edit[k] (own edit released), hd_flags;
         out/factor/<log>.json (self_separation_margin)
  LLaVA  out/gate/<log>/_meta.json (probe.masks over ALL valid probes), cells: edit[k].own_top, probe_fired; out/gate/<log>/_factor.json
  both   out/rows/<log>__<label>__d<draw|x>__<part>.jsonl.gz (driver rows), out/behave/<log>/<label>__d<draw|x>__<part>.json

Statistics: fx.stats (PROTOCOL section 4, R10): crossed_bootstrap_diff over calibration draws x units (2000 resamples, seed 0, as
e3_qwen/code/e3_analyze.py), mcnemar_exact, holm, over_draws.
"""
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP
import sys

sys.dont_write_bytecode = True                      # nothing is written next to existing code (no __pycache__ entry)
import os  # noqa: E402

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "4")
FINAL = f"{_RP.ROOT}/02_research/experiments/10_final"
sys.path.insert(0, FINAL + "/common")
sys.path.insert(0, f"{_RP.ROOT}/02_research/code")
import re  # noqa: E402
import time  # noqa: E402

import fx  # noqa: E402
import numpy as np  # noqa: E402

ST, MX = fx.stats, fx.metrics
XR = FINAL + "/x_release"
START_FILE = XR + "/out/work_start_epoch.txt"
N_BOOT = ST.N_BOOT
RES = 1.0 / N_BOOT                                  # resolution of a two-sided bootstrap value
HN = ("HN-I", "HN-Q", "HN-X")
SUB = ("all", "first100", "later")
SUBW = {"all": "All", "first100": "First", "later": "Later"}
DW = {0: "Zero", 1: "One", 2: "Two"}
MCODE = {"deployed": "Deployed", "where_logonly": "Where", "rescope": "Rescope", "rescope_shared": "RescopeShared", "rescope_x": "RescopeX",
         "white_1_shared": "WhiteOneShared", "white_0.1_shared": "WhiteTenthShared", "white_1_x": "WhiteOneX", "retr_sum": "RetrSum",
         "where_least_n10": "WhereLeastTen", "where_cal": "WhereCal"}
MTITLE = {"deployed": "deployed log", "where_logonly": "ReScope-W, log-only thresholds (where_logonly)", "rescope": "ReScope (rescope)",
          "rescope_shared": "ReScope, shared scale (rescope_shared)", "rescope_x": "ReScope, log-derived negatives (rescope_x)",
          "white_1_shared": "cosine family, selected for H3 (white_1_shared)", "white_0.1_shared": "cosine family, selected for H2 (white_0.1_shared)",
          "white_1_x": "white_1_x", "retr_sum": "retrieval reference (retr_sum)",
          "where_least_n10": "ReScope-W, least repair of 10 complaints from the log-only thresholds (E2 b; H1 row)",
          "where_cal": "ReScope-W, two-block thresholds (E2 c')"}
REPORT_METHODS = ("where_logonly", "rescope", "rescope_shared", "rescope_x", "white_1_shared", "white_0.1_shared", "retr_sum")
FAM_TITLE = {"ae-l2_3": "FIRST EDITOR FAMILY (AlphaEdit-style, [redacted: describes a related submission under double-blind review], ridge l2 = 3; names ending in -ae-l2_3)",
             "memitw": "SECOND EDITOR FAMILY (MEMIT-style, public configuration; names ending in -memitw)"}
EARLIER = (FINAL + "/e1_main/out/fixup/paper_numbers_fixup.json", FINAL + "/e2_curves/out/fixup/paper_numbers_fixup.json")


def jload(p):
    return fx.io.jload(p)


def jdump(o, p, indent=1):
    fx.io.jdump(o, p, indent=indent)


def unflags(s):
    return np.array([c == "1" for c in s], dtype=bool)


def pct(v):
    v = [float(x) for x in v]
    return 100.0 * float(np.mean(v)) if v else None


def mean(v):
    v = [float(x) for x in v if x is not None]
    return float(np.mean(v)) if v else None


def camel(s):
    """The name rule of e3_llava/code/e3_analyze.py (digits become words)."""
    s = re.sub(r"[^A-Za-z0-9]+", " ", str(s))
    words = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}
    return "".join(w[:1].upper() + w[1:] for w in "".join(words.get(ch, ch) if ch.isdigit() else ch for ch in s).split())


def family_of(log):
    return "memitw" if log.endswith("-memitw") else "ae-l2_3"


# ----------------------------------------------------------------------------- the two layouts
class Backbone:
    ROMAN = {100: "C", 200: "CC", 500: "D", 1000: "M", 2000: "MM"}
    TW = {100: "Hundred", 200: "TwoHundred", 500: "FiveHundred", 1000: "Thousand"}

    def __init__(self, key):
        self.key = key
        if key == "Q":
            self.exp, self.title, self.bbname, self.later_from = "e3_qwen", "Qwen2.5-VL-3B", "qwen25vl", 500
            self.logs = ("TB-Q100-ae-l2_3", "TB-Q200-ae-l2_3", "TB-Q500-ae-l2_3", "TB-Q100-memitw", "TB-Q500-memitw", "TB-Q1000-memitw", "TB-Q2000-memitw")
        else:
            self.exp, self.title, self.bbname, self.later_from = "e3_llava", "LLaVA-1.5-7B", "llava", 100
            self.logs = ("TB-L100-ae-l2_3", "TB-L200-ae-l2_3", "TB-L500-ae-l2_3", "TB-L100-memitw", "TB-L500-memitw", "TB-L1000-memitw")
        self.dev = self.logs[0]
        self.unseen = self.logs[1:]
        self.root = f"{FINAL}/{self.exp}"
        self.out = self.root + "/out"
        self.fix = self.out + "/fixup"
        self.fam = {f: tuple(n for n in self.logs if family_of(n) == f) for f in ("ae-l2_3", "memitw")}
        self._ctx, self._rows, self._cells = {}, {}, {}
        h3, h2 = fx.review1.selected(self.bbname, "H3"), fx.review1.selected(self.bbname, "H2")
        self.sel = {"cos_H3": h3["cosine"], "cos_H3_x": h3["cosine_x"], "cos_H2": h2["cosine"], "retr_H3": h3["retr"], "retr_H2": h2["retr"]}

    def rel(self, p):
        return os.path.relpath(p, FINAL)

    def ctx(self, log):
        if log not in self._ctx:
            if self.key == "Q":
                p = f"{self.out}/gate/{log}/_context.json"
                c = jload(p)
                o = {"T": c["T"], "later": list(c["later_edits"]), "hd_near": unflags(c["hd"]["near"]), "n_hd": len(c["hd"]["csid"]), "mask_hd": None}
            else:
                p = f"{self.out}/gate/{log}/_meta.json"
                c = jload(p)
                hd = unflags(c["probe"]["masks"]["hd"])
                o = {"T": c["T"], "later": list(c["later_edits"]), "hd_near": unflags(c["probe"]["masks"]["hd_near"])[hd], "n_hd": int(hd.sum()), "mask_hd": hd}
            o.update(describe=c["describe"], selection=c["selection"], file=self.rel(p))
            self._ctx[log] = o
        return self._ctx[log]

    def T(self, log):
        return self.ctx(log)["T"]

    def code(self, log):
        """Log code of the experiment's own paper numbers (after 'Ethree')."""
        T = self.T(log)
        return self.key + ("Ae" if family_of(log) == "ae-l2_3" else "Me") + (self.ROMAN[T] if self.key == "Q" else self.TW[T])

    def bdraws(self, log):
        return (0,) if self.T(log) >= 1000 else (0, 1, 2)

    def row_path(self, log, lab, d, part="tf"):
        return f"{self.out}/rows/{log}__{lab}__d{'x' if d is None else d}__{part}.jsonl.gz"

    def cellrec(self, log, lab, d, part="tf"):
        c = f"{self.out}/behave/{log}/{lab}__d{'x' if d is None else d}__{part}.json"
        return jload(c) if os.path.exists(c) else None

    def rows(self, log, lab, d, part="tf"):
        k = (log, lab, d, part)
        if k not in self._rows:
            p = self.row_path(log, lab, d, part)
            c = f"{self.out}/behave/{log}/{lab}__d{'x' if d is None else d}__{part}.json"
            self._rows[k] = fx.io.load_rows(p) if (os.path.exists(p) and os.path.exists(c)) else None
        return self._rows[k]

    def by_draw(self, log, lab, part="tf"):
        """{draw: rows} of a calibrated row over the behavioural draws; a row without calibration: {None: rows}.  None if a file is missing."""
        fixed = lab in ("deployed", "where_logonly")
        out = {}
        for d in ([None] if fixed else self.bdraws(log)):
            r = self.rows(log, lab, d, part)
            if r is None:
                return None
            out[d] = r
        return out

    def forget(self):
        self._rows, self._cells = {}, {}

    def cells(self, log, lab):
        """The 20 gate-level cells of the registered read-out, in the order of the calibration draws; None: no file / not defined."""
        k = (log, lab)
        if k not in self._cells:
            p = f"{self.out}/gate/{log}/{lab}__reg.json"
            cs = None
            if os.path.exists(p):
                b = jload(p)
                cs = [b["cells"][i] for i in sorted(b["cells"], key=int)]
                if "not_defined" in cs[0]:
                    cs = None
            self._cells[k] = cs
        return self._cells[k]

    def cell_path(self, log, lab):
        return self.rel(f"{self.out}/gate/{log}/{lab}__reg.json")

    def cov(self, c, k):
        """Per-edit flags 'own edit released on the rephrase' of one gate-level cell (k = tgen | igen | rel)."""
        return unflags(c["per_edit"][k]) if self.key == "Q" else unflags(c["edit"][k]["own_top"])

    def hd_fired(self, log, c):
        """Firing flags of one gate-level cell on the valid heldout_disjoint rows."""
        return unflags(c["hd_flags"]) if self.key == "Q" else unflags(c["probe_fired"])[self.ctx(log)["mask_hd"]]

    def margin(self, log):
        p = f"{self.out}/factor/{log}.json" if self.key == "Q" else f"{self.out}/gate/{log}/_factor.json"
        m = jload(p)["self_separation_margin"]
        return {"min": m["min"], "n_nonpositive": m["n_nonpositive"], "n_positive": m["n_positive"], "file": self.rel(p)}


class Numbers:
    """Paper numbers of one output directory: name -> value, name -> documentation line, name -> item."""

    def __init__(self):
        self.num, self.doc, self.item, self.cur = {}, {}, {}, 0

    def put(self, name, value, text):
        if value is None:
            return
        if not name.isalpha():
            raise RuntimeError("a paper number name holds letters only: " + name)
        if name in self.num:
            raise RuntimeError("paper number defined twice: " + name)
        if isinstance(value, (bool, np.bool_)):
            value = int(value)
        self.num[name], self.doc[name], self.item[name] = (float(value) if isinstance(value, (float, np.floating)) else int(value)), text, self.cur


class Checks:
    def __init__(self):
        self.list = []

    def add(self, item, what, ok, detail=None, skipped=False):
        # ARTIFACT: a check that is skipped (the comparison of modification times, in a copy) is recorded as skipped, not as passed
        self.list.append({"item": item, "check": what, "ok": None if skipped else bool(ok), "detail": detail, **({"skipped": True} if skipped else {})})
        print(f"  [{'skipped' if skipped else 'ok' if ok else 'FAILED'}] item {item}: {what}" + ("" if detail is None else f"  ({str(detail)[:260]})"), flush=True)


# ----------------------------------------------------------------------------- items 1 / 2: paired differences H2 and H3
def sub_where(sub, later):
    if sub == "all":
        return lambda r: True
    if sub == "first100":
        return lambda r: r["edit"] < 100
    return lambda r: r["edit"] in later


def gate_cov_diff(B, log, a, b, key="per_edit", idx=None):
    """Gate-level coverage form over the 20 calibration draws (R25 (3)): paired crossed bootstrap of (T-Gen + I-Gen) / 2 coverage,
    units = edits; key='h3': under the near-scope constraint on the odd edits; a draw on which the raise is unreachable counts 0."""
    def mat(lab):
        cs = B.cells(log, lab)
        if cs is None:
            return None
        out = []
        for c in cs:
            if key == "per_edit":
                v = 0.5 * (B.cov(c, "tgen").astype(float) + B.cov(c, "igen").astype(float))
                out.append(v if idx is None else v[idx])
            else:
                h = c.get("h3")
                if h is None or "not_defined" in h:
                    return None
                if not h["reachable"]:
                    out.append(None)
                    continue
                out.append(0.5 * (unflags(h["per_edit"]["tgen"]).astype(float) + unflags(h["per_edit"]["igen"]).astype(float)))
        n = next((len(x) for x in out if x is not None), None)
        if n is None:
            return None
        return np.array([x if x is not None else np.zeros(n) for x in out])
    A, Bm = mat(a), mat(b)
    if A is None or Bm is None or A.shape != Bm.shape:
        return None
    d = ST.crossed_bootstrap_diff(A, Bm, scale=100.0)
    return {k: d[k] for k in ("value", "ci_lo", "ci_hi", "p_two_sided", "mean_a", "mean_b")} | {"n_draws": int(A.shape[0]), "n_units": int(A.shape[1])}


def h3_rows(B, log, lab, draws):
    out, info = {}, {}
    for d in draws:
        r = B.cellrec(log, lab, d, "h3")
        if r is None or r.get("not_defined"):
            return None, None
        info[d] = r
        if r["reachable"]:
            out[d] = fx.io.load_rows(B.row_path(log, lab, d, "h3"))
        else:                                                       # rule of E1 / e3_qwen: an unreachable draw contributes exact = 0 on every odd edit
            out[d] = [{"kind": "edit", "metric": m, "edit": e, "exact": False} for m in ("tgen", "igen") for e in r["odd_edits"]]
    return out, info


def endpoints_log(B, log):
    """H2 and H3 of one log with the method of e3_qwen/code/e3_analyze.py (endpoints): the same statistic, units, draws and seed."""
    draws = B.bdraws(log)
    c = B.ctx(log)
    later = set(c["later"])
    s = B.sel
    out = {"H2": {}, "H3": {}, "n_behavioural_draws": len(draws), "behavioural_draws": list(draws), "T": c["T"], "n_later_edits": len(later),
           "later_edits_from_stream_position": B.later_from}
    ex = lambda r: float(bool(r["exact"]))                                             # noqa: E731
    A = B.by_draw(log, "rescope")
    for role, lab in (("best_cosine_H2", s["cos_H2"]), ("retr", s["retr_H2"])):
        Bb = B.by_draw(log, lab)
        if A is None or Bb is None:
            out["H2"][role] = {"method": lab, "missing": True}
            continue
        e = {"method": lab, "role": role, "n_draws": len(draws)}
        for sub in SUB:
            w0 = sub_where(sub, later)
            w = lambda r, w0=w0: r["kind"] == "edit" and r["metric"] in ("tgen", "igen") and w0(r)        # noqa: E731
            if not any(w(r) for r in A[draws[0]]):
                continue
            ga, gb = MX.grid(A, ex, unit="edit", where=w), MX.grid(Bb, ex, unit="edit", where=w)
            d = ST.crossed_bootstrap_diff(ga["num"], gb["num"], ga["den"], gb["den"], scale=100.0)
            bd = [float(100 * np.mean([ex(r) for r in A[k] if w(r)]) - 100 * np.mean([ex(r) for r in Bb[k] if w(r)])) for k in draws]
            e[sub] = {"diff": d["value"], "ci_lo": d["ci_lo"], "ci_hi": d["ci_hi"], "p": d["p_two_sided"], "mean_rescope": d["mean_a"], "mean_other": d["mean_b"],
                      "n_units": len(ga["units"]), "by_draw": bd}
            idx = None if sub == "all" else (np.arange(min(100, c["T"])) if sub == "first100" else np.array(sorted(later), dtype=np.int64))
            e[sub]["gate_coverage_form_20_draws"] = gate_cov_diff(B, log, "rescope", lab, idx=idx)
        out["H2"][role] = e
    w = lambda r: r["kind"] == "edit" and r["metric"] in ("tgen", "igen")           # noqa: E731
    for pair, a, b in (("pair1", "rescope", s["cos_H3"]), ("pair2_x", "rescope_x", s["cos_H3_x"]), ("retr", "rescope", s["retr_H3"])):
        e = {"a": a, "b": b, "n_draws": len(draws)}
        e["gate_coverage_form_20_draws"] = gate_cov_diff(B, log, a, b, key="h3")
        for nm, lab in (("a", a), ("b", b)):
            cs = B.cells(log, lab)
            if cs is not None and cs[0].get("h3") is not None and "not_defined" not in cs[0]["h3"]:
                e["gate_" + nm] = {"n_draws": len(cs), "n_unreachable": sum(not x["h3"]["reachable"] for x in cs),
                                   "n_edits_at_cap_by_draw": [x["h3"].get("n_edits_at_cap") for x in cs]}
        ra, ia = h3_rows(B, log, a, draws)
        rb, ib = h3_rows(B, log, b, draws)
        if ra is None or rb is None:
            e["missing_behaviour"] = True
        else:
            ga, gb = MX.grid(ra, ex, unit="edit", where=w), MX.grid(rb, ex, unit="edit", where=w)
            d = ST.crossed_bootstrap_diff(ga["num"], gb["num"], ga["den"], gb["den"], scale=100.0)
            bd = [float(100 * np.mean([ex(r) for r in ra[k] if w(r)]) - 100 * np.mean([ex(r) for r in rb[k] if w(r)])) for k in draws]
            e.update(diff=d["value"], ci_lo=d["ci_lo"], ci_hi=d["ci_hi"], p=d["p_two_sided"], mean_a=d["mean_a"], mean_b=d["mean_b"], n_units=len(ga["units"]), by_draw=bd,
                     n_unreachable_a=sum(not ia[k]["reachable"] for k in draws), n_unreachable_b=sum(not ib[k]["reachable"] for k in draws),
                     n_edits_at_cap_a_by_draw=[ia[k].get("n_edits_at_cap") for k in draws], n_edits_at_cap_b_by_draw=[ib[k].get("n_edits_at_cap") for k in draws])
        out["H3"][pair] = e
    return out


H2W = {"best_cosine_H2": "Cos", "retr": "Retr"}
H3W = {"pair1": "PairOne", "pair2_x": "PairTwo", "retr": "Retr"}


def endpoint_numbers(B, E, PN, note):
    """Paper numbers of the H2 / H3 differences of every log.  note(log, kind, key, sub) -> text appended to the documentation line."""
    for log, e in E.items():
        base = "EthreeFx" + B.code(log)
        src = f"{B.exp}/out/rows/{log}__<method>__d<draw>__tf.jsonl.gz"
        PN.put(base + "BehDraws", e["n_behavioural_draws"], f"{log}: number of behavioural calibration draws behind the H2 / H3 differences (R23: draw 0 only from 1000 edits on)")
        for role, w in H2W.items():
            x = e["H2"][role]
            if x.get("missing"):
                continue
            for sub in ("all", "later"):
                if sub not in x:
                    continue
                y, g = x[sub], x[sub]["gate_coverage_form_20_draws"]
                what = (f"{log}: H2, rephrase exact (T-Gen + I-Gen) / 2 of rescope minus {x['method']}, "
                        + ("all edits" if sub == "all" else f"later edits only (stream position >= {B.later_from}, R8)"))
                tail = f"; paired crossed bootstrap over {e['n_behavioural_draws']} calibration draw(s) x {y['n_units']} edits, 2000 resamples ({src}){note(log, 'H2', role, sub)}"
                n = base + "Htwo" + w + SUBW[sub]
                PN.put(n + "Diff", y["diff"], what + ": difference in points" + tail)
                PN.put(n + "Lo", y["ci_lo"], what + ": lower end of the 95 percent interval" + tail)
                PN.put(n + "Hi", y["ci_hi"], what + ": upper end of the 95 percent interval" + tail)
                PN.put(n + "P", y["p"], what + f": two-sided bootstrap value (resolution {RES}: a value at the resolution is an upper bound)" + tail)
                PN.put(n + "MeanRescope", y["mean_rescope"], what + ": level of rescope (percent)" + tail)
                PN.put(n + "MeanOther", y["mean_other"], what + f": level of {x['method']} (percent)" + tail)
                for i, v in enumerate(y["by_draw"]):
                    PN.put(n + "Draw" + DW[e["behavioural_draws"][i]], v, what + f": difference at calibration draw {e['behavioural_draws'][i]}" + tail)
                if g is not None:
                    gt = f"; gate-level coverage form (own edit released on text and image rephrases), paired crossed bootstrap over {g['n_draws']} calibration draws x {g['n_units']} edits ({B.exp}/out/gate/{log}/<method>__reg.json)"
                    PN.put(n + "GateDiff", g["value"], what + ": difference of coverage in points" + gt)
                    PN.put(n + "GateLo", g["ci_lo"], what + ": lower end of the 95 percent interval" + gt)
                    PN.put(n + "GateHi", g["ci_hi"], what + ": upper end of the 95 percent interval" + gt)
                    PN.put(n + "GateP", g["p_two_sided"], what + ": two-sided bootstrap value" + gt)
        for pair, w in H3W.items():
            x = e["H3"][pair]
            what = f"{log}: H3 (R6 with R25: raised common scale, odd edits, test rows of the near-scope negatives), rephrase exact of {x['a']} minus {x['b']}"
            n = base + "Hthree" + w
            if not x.get("missing_behaviour"):
                tail = (f"; paired crossed bootstrap over {x['n_draws']} calibration draw(s) x {x['n_units']} odd edits, 2000 resamples "
                        f"({B.exp}/out/rows/{log}__<method>__d<draw>__h3.jsonl.gz){note(log, 'H3', pair, None)}")
                PN.put(n + "Diff", x["diff"], what + ": difference in points" + tail)
                PN.put(n + "Lo", x["ci_lo"], what + ": lower end of the 95 percent interval" + tail)
                PN.put(n + "Hi", x["ci_hi"], what + ": upper end of the 95 percent interval" + tail)
                PN.put(n + "P", x["p"], what + f": two-sided bootstrap value (resolution {RES}: a value at the resolution is an upper bound)" + tail)
                PN.put(n + "MeanA", x["mean_a"], what + f": level of {x['a']} (percent)" + tail)
                PN.put(n + "MeanB", x["mean_b"], what + f": level of {x['b']} (percent)" + tail)
                PN.put(n + "Unreachable", x["n_unreachable_a"] + x["n_unreachable_b"], what + ": behavioural draws (both methods) on which the raise of the scale does not reach the limit (they count 0)" + tail)
                PN.put(n + "AtCapA", mean(x["n_edits_at_cap_a_by_draw"]), what + f": edits of {x['a']} at their guard cap at the raised scale, mean over the behavioural draws (R25 (4))" + tail)
                PN.put(n + "AtCapB", mean(x["n_edits_at_cap_b_by_draw"]), what + f": edits of {x['b']} at their guard cap at the raised scale, mean over the behavioural draws (R25 (4))" + tail)
                for i, v in enumerate(x["by_draw"]):
                    PN.put(n + "Draw" + DW[e["behavioural_draws"][i]], v, what + f": difference at calibration draw {e['behavioural_draws'][i]}" + tail)
            g = x["gate_coverage_form_20_draws"]
            if g is not None:
                gt = f"; gate-level coverage form under the near-scope constraint, paired crossed bootstrap over {g['n_draws']} calibration draws x {g['n_units']} odd edits ({B.exp}/out/gate/{log}/<method>__reg.json, h3.per_edit)"
                PN.put(n + "GateDiff", g["value"], what + ": difference of coverage in points" + gt)
                PN.put(n + "GateLo", g["ci_lo"], what + ": lower end of the 95 percent interval" + gt)
                PN.put(n + "GateHi", g["ci_hi"], what + ": upper end of the 95 percent interval" + gt)
                PN.put(n + "GateP", g["p_two_sided"], what + ": two-sided bootstrap value" + gt)


def compare_verifier(B, E):
    """Every H2 / H3 value that the verification lists in claims_extra.json against the value computed here.
    -> rows {log, quantity, field, mine, verifier, abs_diff, limit, above_limit, reason}."""
    V = jload(FINAL + "/e3_qwen/verify/out/claims_extra.json")
    rows = []

    def add(log, quantity, field, mine, ver, limit):
        if mine is None or ver is None:
            rows.append({"log": log, "quantity": quantity, "field": field, "mine": mine, "verifier": ver, "abs_diff": None, "limit": limit, "above_limit": None,
                         "reason": "value missing on one side"})
            return
        d = abs(float(mine) - float(ver))
        above = d > limit - 1e-9                                    # the verification lists one decimal: a difference AT the limit is listed too
        reason = None
        if above:
            reason = ("interval end of another bootstrap (the verification drew its own resamples; this file uses the driver's crossed bootstrap with seed 0)"
                      if field in ("ci_lo", "ci_hi") else "value differs beyond the rounding of the verification's list (one decimal): see FIXUP.txt")
        rows.append({"log": log, "quantity": quantity, "field": field, "mine": float(mine), "verifier": float(ver), "abs_diff": d, "limit": limit, "above_limit": bool(above), "reason": reason})

    if B.key == "L":
        for key, sub in (("LLaVA H2 (rescope minus white_0.1_shared, rephrase exact, all edits; my bootstrap)", "all"), ("LLaVA H2, later edits", "later")):
            for short, v in V[key].items():
                log = "TB-" + short
                y = E[log]["H2"]["best_cosine_H2"].get(sub) or {}
                for f, ver, lim in (("diff", v[0], 0.1), ("ci_lo", v[1], 0.5), ("ci_hi", v[2], 0.5)):
                    add(log, f"H2 {sub} edits, rescope minus {B.sel['cos_H2']}", f, y.get(f), ver, lim)
        for short, v in V["LLaVA H3 pair 1 / pair 2 / retr, behaviour"].items():
            log = "TB-" + short
            for pair, ver in zip(("pair1", "pair2_x", "retr"), v):
                add(log, f"H3 {pair}", "diff", E[log]["H3"][pair].get("diff"), ver, 0.1)
    else:
        for key, pair in (("Qwen H3 pair 2 (rescope_x minus white_1_x), behaviour", "pair2_x"), ("Qwen H3 against retr_sum (rescope minus retr_sum), behaviour", "retr")):
            for short, ver in V[key].items():
                log = "TB-" + short
                add(log, f"H3 {pair}", "diff", E[log]["H3"][pair].get("diff"), ver, 0.1)
    return rows


# ----------------------------------------------------------------------------- item 3: exact McNemar
def d1_flags(rows):
    """{csid: D1} on the valid heldout_disjoint rows (teacher-forced probe rows of the pool)."""
    return {r["csid"]: bool(r["D1"]) for r in rows if r["kind"] == "probe" and r.get("disjoint") and r.get("lp", "tf") == "tf" and not r.get("d2_row")}


def rel_flags(rows):
    return {r["csid"]: bool(r["exact"]) for r in rows if r["kind"] == "edit" and r["metric"] == "rel"}


MC_STATE = {"n_driver": 0, "n_integer_only": 0, "max_abs_diff_to_driver": 0.0}


def mcnemar_exact(a, b):
    """Exact two-sided McNemar test (two-sided exact binomial test on the discordant pairs): p = min(1, 2 P(Bin(b + c, 1/2) <= min(b, c))),
    in integer arithmetic.  fx.stats.mcnemar_exact computes the same value in floating point and overflows from 1024 discordant pairs on
    (2.0 ** n); below that it is called as well and the two values are compared (MC_STATE; a check of the fix-up).  log10_p is given
    because p can be below the smallest float."""
    from fractions import Fraction
    import math
    pairs = [(bool(x), bool(y)) for x, y in zip(a, b)]
    n10 = sum(x and not y for x, y in pairs)
    n01 = sum(y and not x for x, y in pairs)
    n, k = n10 + n01, min(n10, n01)
    if n == 0:
        p, lg = 1.0, 0.0
    else:
        num = 2 * sum(math.comb(n, i) for i in range(k + 1))
        if num >= 2 ** n:
            p, lg = 1.0, 0.0
        else:
            p, lg = float(Fraction(num, 2 ** n)), math.log10(num) - n * math.log10(2.0)
    out = {"n10": n10, "n01": n01, "n_discordant": n, "n": len(pairs), "p": p, "log10_p": lg, "p_from": "integer arithmetic"}
    if n < 1000:
        d = ST.mcnemar_exact(a, b)
        if (d["n10"], d["n01"], d["n"]) != (n10, n01, len(pairs)):
            raise RuntimeError("discordant counts differ from the driver's")
        MC_STATE["n_driver"] += 1
        MC_STATE["max_abs_diff_to_driver"] = max(MC_STATE["max_abs_diff_to_driver"], abs(d["p"] - p))
        out["p_from"] = "integer arithmetic; equals fx.stats.mcnemar_exact"
    else:
        MC_STATE["n_integer_only"] += 1
        out["p_from"] = "integer arithmetic (fx.stats.mcnemar_exact overflows from 1024 discordant pairs on)"
    return out


MC = (("D1", "where_logonly", "deployed"), ("D1", "rescope", "deployed"), ("Rel", "where_logonly", "deployed"), ("Rel", "rescope", "deployed"),
      ("D1", "rescope", "cos_H2"), ("D1", "rescope", "retr_H2"))
MCW = {("where_logonly", "deployed"): "WhereVsDeployed", ("rescope", "deployed"): "RescopeVsDeployed", ("rescope", "cos_H2"): "RescopeVsCos", ("rescope", "retr_H2"): "RescopeVsRetr"}


def mcnemar_log(B, log, PN, missing):
    out = {}
    base = "EthreeFx" + B.code(log)
    for outcome, a, brole in MC:
        b = B.sel.get(brole, brole)
        fl = d1_flags if outcome == "D1" else rel_flags
        ra, rb = B.by_draw(log, a), B.by_draw(log, b)
        key = f"{outcome}: {a} against {b}"
        if ra is None or rb is None:
            missing.append({"log": log, "test": key, "reason": f"no per-row file of {a if ra is None else b}"})
            continue
        draws = [d for d in ra if d is not None] or [d for d in rb if d is not None] or [None]
        tests = []
        for d in draws:
            fa, fb = fl(ra.get(d, ra.get(None))), fl(rb.get(d, rb.get(None)))
            if set(fa) != set(fb):
                raise RuntimeError(f"{log} {key}: the two rows do not hold the same sequences")
            ks = list(fa)
            m = mcnemar_exact([fa[k] for k in ks], [fb[k] for k in ks])
            tests.append({"draw": d, "b": m["n10"], "c": m["n01"], "n_discordant": m["n_discordant"], "n_pairs": m["n"], "p": m["p"], "log10_p": m["log10_p"],
                          "p_from": m["p_from"],
                          "rate_a": pct(fa.values()), "rate_b": pct(fb.values())})
        out[key] = {"outcome": outcome, "a": a, "b": b, "paired_on": "the same calibration draw index" if (None not in ra and None not in rb) else
                    ("one run of each row (no calibration)" if draws == [None] else "the one run of the row without calibration against each draw of the calibrated row"),
                    "tests": tests, "pooled_description_only": {"b": sum(t["b"] for t in tests), "c": sum(t["c"] for t in tests)}}
        ow = "Done" if outcome == "D1" else "Rel"
        for t in tests:
            n = base + "Mc" + ow + MCW[(a, brole)] + ("" if t["draw"] is None else "Draw" + DW[t["draw"]])
            what = (f"{log}: exact McNemar test, {'D1 on heldout_disjoint' if outcome == 'D1' else 'Rel exact'}, {a} against {b}"
                    + ("" if t["draw"] is None else f", calibration draw {t['draw']}") + f" ({t['n_pairs']} pairs; {B.exp}/out/rows/{log}__{a}__*__tf.jsonl.gz and __{b}__)")
            PN.put(n + "B", t["b"], what + f": pairs with the outcome under {a} only")
            PN.put(n + "C", t["c"], what + f": pairs with the outcome under {b} only")
            PN.put(n + "P", t["p"], what + ": two-sided exact binomial value on the discordant pairs")
            PN.put(n + "LogP", t["log10_p"], what + ": decimal logarithm of the two-sided exact binomial value (for values too small to print)")
    return out


# ----------------------------------------------------------------------------- item 4: reporting numbers next to far locality
def reporting_log(B, log, PN):
    c = B.ctx(log)
    T, later = c["T"], set(c["later"])
    base = "EthreeFx" + B.code(log)
    near = c["hd_near"]
    out = {"T": T, "n_heldout_disjoint": c["n_hd"], "n_near_scope_probes": int(near.sum()), "methods": {}}
    PN.put(base + "NearN", int(near.sum()), f"{log}: valid heldout_disjoint probes that ask the question of an edit of the log (near-scope probes; {c['file']})")
    PN.put(base + "HdN", c["n_hd"], f"{log}: valid heldout_disjoint rows ({c['file']})")
    bound = None
    cs0 = B.cells(log, "rescope")
    if cs0:
        bound = 100.0 * cs0[0]["cal"]["bound_false_firing"]
        PN.put(base + "Bound", bound, f"{log}: bound of the two-block rule on the false-firing rate, percent (alpha = 0.05, n = 200: 5 / 101; {B.cell_path(log, 'rescope')}, cal.bound_false_firing)")
    out["bound_two_block_percent"] = bound
    dep = B.rows(log, "deployed", None)
    dep_e = [r for r in dep if r["kind"] == "edit"] if dep else None
    for lab in REPORT_METHODS:
        m = {}
        code = base + MCODE[lab]
        by = B.by_draw(log, lab)
        # (a) near-scope probes: behaviour
        if by is not None:
            per, n = [], None
            for d, rows in by.items():
                v = [bool(r["fired"]) for r in rows if r["kind"] == "probe" and r.get("disjoint") and r.get("near_edit") is not None]
                per.append({"draw": d, "n_released": int(sum(v)), "n": len(v), "percent": pct(v)})
                n = len(v)
            m["near_released_behaviour"] = {"percent": mean([p["percent"] for p in per]), "n_rows": n, "n_draws": len(per), "by_draw": per}
            PN.put(code + "NearRelBeh", m["near_released_behaviour"]["percent"],
                   f"{log}, {lab}: near-scope held-out probes on which an edit is released, percent, behaviour, mean over {len(per)} run(s) ({n} rows; {B.exp}/out/rows/{log}__{lab}__*__tf.jsonl.gz, fired, near_edit)")
        cs = B.cells(log, lab)
        if cs is not None:
            F = np.array([B.hd_fired(log, x) for x in cs])
            if F.shape[1] != near.size:
                raise RuntimeError(f"{log} {lab}: firing flags and near-scope mask differ in length")
            nr = [100.0 * float(f[near].mean()) for f in F] if near.any() else []
            if nr:
                s = ST.over_draws(nr)
                m["near_released_gate"] = {"percent": s["mean"], "p05": s["p05"], "p95": s["p95"], "n_rows": int(near.sum()), "n_draws": s["n_draws"], "by_draw": nr}
                PN.put(code + "NearRelGate", s["mean"], f"{log}, {lab}: near-scope held-out probes on which an edit is released, percent, gate level, mean over {s['n_draws']} calibration draws ({int(near.sum())} rows; {B.cell_path(log, lab)})")
            # (b) calibration draws above the bound
            rate = [100.0 * float(f.mean()) for f in F]
            s = ST.over_draws(rate)
            own = [(x.get("cal") or {}).get("bound_false_firing") for x in cs]
            m["false_firing_20_draws"] = {"n_draws": len(rate), "n_draws_above_bound": int(sum(r > bound + 1e-12 for r in rate)), "bound_percent": bound, "mean": s["mean"], "p05": s["p05"],
                                          "p95": s["p95"], "min": s["min"], "max": s["max"], "by_draw": rate,
                                          "bound_stated_in_own_cells": sorted({None if o is None else round(100.0 * o, 6) for o in own}, key=str)}
            src = f"gate level, {len(rate)} calibration draws; {B.cell_path(log, lab)}"
            PN.put(code + "DrawsAbove", m["false_firing_20_draws"]["n_draws_above_bound"], f"{log}, {lab}: calibration draws (of {len(rate)}) whose false-firing rate on heldout_disjoint exceeds the bound of the two-block rule ({bound:.4f} percent) ({src})")
            PN.put(code + "FireMean", s["mean"], f"{log}, {lab}: false-firing rate on heldout_disjoint, percent, mean over the draws ({src})")
            PN.put(code + "FirePFive", s["p05"], f"{log}, {lab}: false-firing rate on heldout_disjoint, percent, 5th percentile over the draws ({src})")
            PN.put(code + "FirePNinetyFive", s["p95"], f"{log}, {lab}: false-firing rate on heldout_disjoint, percent, 95th percentile over the draws ({src})")
            PN.put(code + "FireMax", s["max"], f"{log}, {lab}: false-firing rate on heldout_disjoint, percent, largest draw ({src})")
            # (c) capped edits
            cap = [int(x["n_capped"]) for x in cs]
            m["capped_edits_20_draws"] = {"mean": float(np.mean(cap)), "max": int(max(cap)), "min": int(min(cap)), "by_draw": cap, "n_draws": len(cap)}
            PN.put(code + "CappedMean", float(np.mean(cap)), f"{log}, {lab}: edits at their guard cap (for them the bound is void), mean over {len(cap)} calibration draws ({B.cell_path(log, lab)}, n_capped)")
            PN.put(code + "CappedMax", int(max(cap)), f"{log}, {lab}: edits at their guard cap, largest count over {len(cap)} calibration draws ({B.cell_path(log, lab)}, n_capped)")
        # (e) Rel exact kept / gained / lost against the deployed log
        if lab in ("where_logonly", "rescope") and by is not None and dep_e is not None:
            kg = {}
            for sub in SUB:
                w0 = sub_where(sub, later)
                per = []
                for d, rows in by.items():
                    k = [r for r in MX.kept_gained_rows([r for r in rows if r["kind"] == "edit"], dep_e) if r["metric"] == "rel" and w0(r)]
                    if not k:
                        break
                    per.append({"draw": d, "n_edits": len(k), **{f: int(sum(r[f] for r in k)) for f in ("kept", "gained", "lost")},
                                "n_exact_deployed": int(sum(r["exact_deployed"] for r in k)), "n_exact_method": int(sum(r["exact"] for r in k))})
                if not per:
                    continue
                e = {"n_edits": per[0]["n_edits"], "n_draws": len(per), "by_draw": per}
                for f in ("kept", "gained", "lost"):
                    e[f + "_percent"] = mean([100.0 * p[f] / p["n_edits"] for p in per])
                    e[f + "_count_mean"] = mean([p[f] for p in per])
                    sw = {"all": "all edits", "first100": "the first 100 edits of the stream", "later": f"the later edits (stream position >= {B.later_from})"}[sub]
                    fw = {"kept": "answered with the target under both", "gained": "answered with the target only under the method", "lost": "answered with the target only under the deployed log"}[f]
                    PN.put(code + "Rel" + f.capitalize() + SUBW[sub], e[f + "_percent"],
                           f"{log}, {lab} against the deployed log: Rel exact {f} ({fw}), percent of {sw} ({per[0]['n_edits']} edits), mean over {len(per)} run(s) ({B.exp}/out/rows/{log}__{lab}__*__tf.jsonl.gz and __deployed__dx__tf)")
                kg[sub] = e
            m["rel_kept_gained_lost"] = kg
        out["methods"][lab] = m
    # (d) self-separation margin
    mg = B.margin(log)
    out["self_separation_margin"] = mg
    PN.put(base + "MarginSmallest", mg["min"], f"{log}: smallest self-separation margin of the log ({mg['file']}, self_separation_margin.min)")
    PN.put(base + "MarginNonPositive", mg["n_nonpositive"], f"{log}: edits with a non-positive self-separation margin ({mg['file']}, self_separation_margin.n_nonpositive)")
    return out


def compare_reporting(B, REP):
    """Item 4 against the verification's own values (claims_extra.json)."""
    V = jload(FINAL + "/e3_qwen/verify/out/claims_extra.json")
    rows = []
    k = ("Qwen" if B.key == "Q" else "LLaVA") + ": rescope, near-scope held-out probes released (behaviour %) / gate level"
    for short, v in V[k].items():
        m = REP["TB-" + short]["methods"]["rescope"]
        for nm, mine, ver in (("near released, behaviour", m["near_released_behaviour"]["percent"], v[0]), ("near released, gate level", m["near_released_gate"]["percent"], v[1])):
            rows.append({"log": "TB-" + short, "quantity": "rescope: " + nm, "mine": mine, "verifier": ver, "abs_diff": abs(mine - ver), "limit": 0.05, "above_limit": abs(mine - ver) > 0.05 + 1e-9})
    if B.key == "Q":
        for short, ver in V["Qwen: draws above 4.95 % of 20, rescope"].items():
            mine = REP["TB-" + short]["methods"]["rescope"]["false_firing_20_draws"]["n_draws_above_bound"]
            rows.append({"log": "TB-" + short, "quantity": "rescope: draws above the bound", "mine": mine, "verifier": ver, "abs_diff": abs(mine - ver), "limit": 0, "above_limit": mine != ver})
    else:
        for short, v in V["LLaVA: self-separation margin min / edits with a non-positive margin"].items():
            mg = REP["TB-" + short]["self_separation_margin"]
            rows.append({"log": "TB-" + short, "quantity": "smallest self-separation margin", "mine": mg["min"], "verifier": v[0], "abs_diff": abs(mg["min"] - v[0]), "limit": 0.005,
                         "above_limit": abs(mg["min"] - v[0]) > 0.005 + 1e-9})
            rows.append({"log": "TB-" + short, "quantity": "edits with a non-positive margin", "mine": mg["n_nonpositive"], "verifier": v[1], "abs_diff": abs(mg["n_nonpositive"] - v[1]), "limit": 0,
                         "above_limit": mg["n_nonpositive"] != v[1]})
    return rows


# ----------------------------------------------------------------------------- checks shared by the three output directories
def equal_existing(B, E, MCN, REP):
    """Numbers computed here that also exist in the experiment's paper_numbers.json: equality to the last stored digit
    (Qwen stores two decimals, four for the margin; LLaVA stores the full float: equal up to 1e-9)."""
    old = jload(B.out + "/paper_numbers.json")
    rows = []

    def cmp(log, what, mine, name, nd):
        if name not in old or mine is None:
            rows.append({"log": log, "what": what, "name": name, "mine": mine, "stored": old.get(name), "equal": False, "note": "name or value missing"})
            return
        m = round(float(mine), nd) if nd is not None else float(mine)
        rows.append({"log": log, "what": what, "name": name, "mine": m, "stored": old[name], "equal": abs(m - float(old[name])) <= 1e-9})

    for log in B.logs:
        pre = "Ethree" + B.code(log)
        e, r = E[log], REP[log]["methods"]
        d1dep = MCN[log]["D1: rescope against deployed"]["tests"]
        if B.key == "Q":
            cmp(log, "H3 pair 1: level of rescope", e["H3"]["pair1"].get("mean_a"), pre + "RescopeHthreeExact", 2)
            cmp(log, "H3 pair 1: level of the selected gate", e["H3"]["pair1"].get("mean_b"), pre + MCODE[B.sel["cos_H3"]] + "HthreeExact", 2)
            cmp(log, "D1 of the deployed log on heldout_disjoint", d1dep[0]["rate_b"], pre + "DeployedHdDone", 2)
            cmp(log, "D1 of rescope on heldout_disjoint (mean of the draws)", mean([t["rate_a"] for t in d1dep]), pre + "RescopeHdDone", 2)
            cmp(log, "near-scope probes released, rescope, behaviour", r["rescope"]["near_released_behaviour"]["percent"], pre + "RescopeHdNearFired", 2)
            cmp(log, "near-scope probes released, rescope, gate level", r["rescope"]["near_released_gate"]["percent"], pre + "RescopeGateHdNearFired", 2)
            cmp(log, "false firing of rescope, mean of 20 draws", r["rescope"]["false_firing_20_draws"]["mean"], pre + "RescopeGateHdFired", 2)
            cmp(log, "capped edits of rescope, mean of 20 draws", r["rescope"]["capped_edits_20_draws"]["mean"], pre + "RescopeGateCapped", 2)
            cmp(log, "smallest self-separation margin", REP[log]["self_separation_margin"]["min"], pre + "MarginMin", 4)
        else:
            cmp(log, "H2: level of rescope, all edits", e["H2"]["best_cosine_H2"]["all"]["mean_rescope"], pre + "RescopeReph", None)
            cmp(log, "H2: level of the selected gate, all edits", e["H2"]["best_cosine_H2"]["all"]["mean_other"], pre + camel(B.sel["cos_H2"]) + "Reph", None)
            cmp(log, "H2: level of retr_sum, all edits", e["H2"]["retr"]["all"]["mean_other"], pre + camel(B.sel["retr_H2"]) + "Reph", None)
            cmp(log, "D1 of the deployed log on heldout_disjoint", d1dep[0]["rate_b"], pre + "DeployedDOne", None)
            cmp(log, "D1 of rescope on heldout_disjoint (mean of the draws)", mean([t["rate_a"] for t in d1dep]), pre + "RescopeDOne", None)
            cmp(log, "near-scope probes released, rescope, gate level", r["rescope"]["near_released_gate"]["percent"], pre + "RescopeFireNear", None)
            cmp(log, "false firing of rescope, mean of 20 draws", r["rescope"]["false_firing_20_draws"]["mean"], pre + "RescopeFire", None)
            cmp(log, "capped edits of rescope, mean of 20 draws", r["rescope"]["capped_edits_20_draws"]["mean"], pre + "RescopeCapped", None)
            cmp(log, "smallest self-separation margin", REP[log]["self_separation_margin"]["min"], pre + "MarginMin", None)
    return rows


def collisions(names, files):
    """Names that exist in any of the given paper-number files."""
    hit = {}
    for f in files:
        for k in set(names) & set(jload(f)):
            hit.setdefault(k, []).append(os.path.relpath(f, FINAL))
    return hit


def work_start():
    with open(START_FILE) as f:
        return float(f.read().strip())


MINE = ("e3_qwen/out/fixup/", "e3_llava/out/fixup/", "x_release/", "e3_qwen/code/fixup_", "e3_llava/code/fixup_")


def modified_since_start(dirs=("e3_qwen", "e3_llava", "e1_main", "e2_curves", "e2_prefix", "common")):
    """Files of the experiments read here whose modification time is not before the start of this work, the new files of the fix-up excluded.
    ARTIFACT: modification times say something in the authors' working tree only.  In a copy (a clone, an unpacked archive) every
    file carries the time of the copy, so the comparison is skipped there with a printed note and the check is recorded as skipped;
    it does not fail.  The outcome in the working tree is in the released out/checks.json.  RESCOPE_MTIME_CHECK=1 runs it anyway."""
    t0, bad, n = work_start(), [], 0
    if os.environ.get("RESCOPE_MTIME_CHECK") != "1":
        print("[fix-up] comparison of modification times with the start of the work: SKIPPED (meaningful in the authors' working tree only; "
              "a copy carries the times of the copy; RESCOPE_MTIME_CHECK=1 runs it)", flush=True)
        return {"start_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t0)), "n_files_checked": 0, "modified": [], "directories": list(dirs),
                "skipped": "SKIPPED in a copy: modification times are meaningful in the authors' working tree only"}
    for d in dirs:
        for root, _, files in os.walk(os.path.join(FINAL, d)):
            for f in files:
                p = os.path.join(root, f)
                r = os.path.relpath(p, FINAL)
                if any(r.startswith(m) for m in MINE):
                    continue
                n += 1
                if os.path.getmtime(p) >= t0:
                    bad.append({"file": r, "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(os.path.getmtime(p)))})
    return {"start_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t0)), "n_files_checked": n, "modified": bad, "directories": list(dirs)}


# ----------------------------------------------------------------------------- one backbone: items 1 - 4
def run_backbone(key):
    B = Backbone(key)
    os.makedirs(B.fix, exist_ok=True)
    PN, CK = Numbers(), Checks()
    t0 = time.time()
    E, MCN, REP, missing = {}, {}, {}, []
    PN_MC, PN_REP = Numbers(), Numbers()
    stored = jload(B.out + "/endpoints.json") if key == "Q" else None
    for log in B.logs:
        print(f"{log}  [{time.time() - t0:.0f}s]", flush=True)
        E[log] = endpoints_log(B, log)
        MCN[log] = mcnemar_log(B, log, PN_MC, missing)
        REP[log] = reporting_log(B, log, PN_REP)
        B.forget()
    # ---- items 1 / 2: endpoints
    PN.cur = 1 if key == "L" else 2
    ep_file = "endpoints.json" if key == "L" else "endpoints_added.json"
    status = {}
    if key == "Q":
        # what the experiment's endpoints.json already holds, and whether the recomputation equals it
        worst, n_cmp = 0.0, 0
        for log in B.logs:
            for role in H2W:
                for sub in SUB:
                    a, b = E[log]["H2"][role].get(sub), (stored[log]["H2"].get(role) or {}).get(sub)
                    if a is None:
                        continue                                    # the log has no such edits (later edits: none up to 500 edits)
                    status[(log, "H2", role, sub)] = b is not None
                    if b is None:
                        continue
                    for f in ("diff", "ci_lo", "ci_hi", "p", "mean_rescope", "mean_other"):
                        worst, n_cmp = max(worst, abs(a[f] - b[f])), n_cmp + 1
                    for f in ("value", "ci_lo", "ci_hi", "p_two_sided"):
                        worst, n_cmp = max(worst, abs(a["gate_coverage_form_20_draws"][f] - b["gate_coverage_form_20_draws"][f])), n_cmp + 1
            for pair in H3W:
                a, b = E[log]["H3"][pair], stored[log]["H3"].get(pair) or {}
                status[(log, "H3", pair, None)] = "diff" in b
                if "diff" in a and "diff" in b:
                    for f in ("diff", "ci_lo", "ci_hi", "p", "mean_a", "mean_b"):
                        worst, n_cmp = max(worst, abs(a[f] - b[f])), n_cmp + 1
                    for f in ("value", "ci_lo", "ci_hi", "p_two_sided"):
                        worst, n_cmp = max(worst, abs(a["gate_coverage_form_20_draws"][f] - b["gate_coverage_form_20_draws"][f])), n_cmp + 1
        CK.add(2, "every H2 / H3 quantity recomputed here that out/endpoints.json holds equals it (value, interval ends, bootstrap value, levels, gate-level form)", worst <= 1e-9,
               f"{n_cmp} numbers compared; largest absolute difference {worst:.3g}")
        n_in = sum(bool(v) for v in status.values())
        CK.add(2, "which of the quantities computed here out/endpoints.json already holds (reported, not a failure)", True,
               f"{n_in} of {len(status)} (log, endpoint, reference, subset) entries are already in out/endpoints.json (H3 pair 2, the retrieval reference and the later edits included); "
               "out/paper_numbers.json holds no paired difference at all: the addition is the paper numbers, the documentation and the tables per editor family")
    note = (lambda log, kind, k, sub: "") if key == "L" else (
        lambda log, kind, k, sub: "; the experiment's out/endpoints.json holds the same value (recomputed here, equal)" if status.get((log, kind, k, sub)) else "; not in out/endpoints.json")
    endpoint_numbers(B, E, PN, note)
    cmpv = compare_verifier(B, E)
    above = [r for r in cmpv if r["above_limit"] or r["above_limit"] is None]
    CK.add(PN.cur, "comparison with the verification's own values (e3_qwen/verify/out/claims_extra.json): no VALUE differs by 0.1 points or more "
           "(interval ends at or above 0.5 points are listed with the reason: the verification drew its own resamples)", not [r for r in above if r["field"] == "diff"],
           f"{len(cmpv)} numbers compared; values at or above 0.1: {sum(r['field'] == 'diff' for r in above)}; interval ends at or above 0.5: "
           f"{[(r['log'], r['quantity'], r['field'], round(r['mine'], 2), r['verifier']) for r in above if r['field'] != 'diff']}; largest difference of a value {max([r['abs_diff'] for r in cmpv if r['field'] == 'diff' and r['abs_diff'] is not None], default=0):.3f}, "
           f"of an interval end {max([r['abs_diff'] for r in cmpv if r['field'] != 'diff' and r['abs_diff'] is not None], default=0):.3f}")
    about = {"what": "paired differences of the registered read-outs H2 and H3 on every log of E3 (" + B.title + "), method of e3_qwen/code/e3_analyze.py (endpoints)",
             "statistic": "fx.stats.crossed_bootstrap_diff over calibration draws x edits, 2000 resamples, seed 0, 95 percent percentile interval; p = two-sided bootstrap value "
                          f"(share of resamples on the other side of 0, doubled; resolution {RES}: a value at the resolution is an upper bound)",
             "H2": "rephrase exact (T-Gen + I-Gen) / 2 of rescope minus the reference, on all edits, on the first 100 edits and on later edits only (R8: stream position >= "
                   f"{B.later_from}); gate_coverage_form_20_draws: the same difference of the coverage of text and image rephrases over the 20 calibration draws (R25 (3))",
             "H3": "R6 with R25: raised common scale, odd edits, test rows of the near-scope negatives; pair1 = rescope against the gate selected for H3, pair2_x = rescope_x against its "
                   "_x form, retr = rescope against the retrieval reference in the form of pair 1; a draw on which the raise is unreachable counts 0 for that method",
             "selection": B.sel, "selection_file": "e1_main/out/best_cosine_gate.json (fx.review1.selected)", "development_log": B.dev,
             "status": "replication read-outs on unseen logs (not the confirmatory family of R6); the Holm correction over the unseen logs of both backbones is in x_release/out/holm.json",
             "written": time.strftime("%Y-%m-%d %H:%M:%S"), "code": fx.code_fingerprint()}
    if key == "Q":
        about["already_in_out_endpoints_json"] = [{"log": k[0], "endpoint": k[1], "reference": k[2], "subset": k[3], "in_endpoints_json": bool(v)} for k, v in status.items()]
    jdump({"about": about, "logs": E, "comparison_with_verification": cmpv}, os.path.join(B.fix, ep_file))
    # ---- item 3: McNemar
    PN.cur = 3
    for k, v in PN_MC.num.items():
        PN.put(k, v, PN_MC.doc[k])
    n_tests = sum(len(c["tests"]) for lg in MCN.values() for c in lg.values())
    CK.add(3, "no McNemar test missing (every per-row file exists and the paired rows hold the same sequences)", not missing, f"{n_tests} tests; missing: {missing}")
    CK.add(3, "the exact value computed in integer arithmetic equals the driver's fx.stats.mcnemar_exact wherever the driver can compute it (fewer than 1000 discordant pairs)",
           MC_STATE["max_abs_diff_to_driver"] <= 1e-12, f"{MC_STATE['n_driver']} tests compared, largest absolute difference {MC_STATE['max_abs_diff_to_driver']:.3g}; "
           f"{MC_STATE['n_integer_only']} test(s) with 1000 or more discordant pairs computed in integer arithmetic only (the driver's function overflows there)")
    jdump({"about": {"what": "exact McNemar tests (fx.stats.mcnemar_exact: two-sided exact binomial test on the discordant pairs), per log and per calibration draw, from the per-row flags",
                     "b": "pairs with the outcome under the first row only", "c": "pairs with the outcome under the second row only",
                     "outcomes": {"D1": "the pre-edit answer is not reproduced, valid heldout_disjoint rows", "Rel": "the request of an edit is answered with its target (Rel exact), all edits"},
                     "comparisons": "(a) D1 and (b) Rel exact: where_logonly against the deployed log, rescope against the deployed log; (c) D1: rescope against the gate selected for H2 "
                                    f"({B.sel['cos_H2']}) and against {B.sel['retr_H2']}",
                     "draws": "a calibrated row is tested at each behavioural draw (0-2; draw 0 only from 1000 edits on); two calibrated rows are paired at the same draw index",
                     "status": "descriptive per-log tests (PROTOCOL section 4); no Holm family is formed over them; pooled counts over the draws are a description, not a test",
                     "n_tests": n_tests, "missing": missing, "computation": dict(MC_STATE), "written": time.strftime("%Y-%m-%d %H:%M:%S")},
           "logs": MCN}, os.path.join(B.fix, "mcnemar.json"))
    # ---- item 4: reporting
    PN.cur = 4
    for k, v in PN_REP.num.items():
        PN.put(k, v, PN_REP.doc[k])
    cmpr = compare_reporting(B, REP)
    bad = [r for r in cmpr if r["above_limit"]]
    CK.add(4, "item 4 against the verification's own values (near-scope release of rescope at behaviour and gate level"
           + ("; draws above the bound" if key == "Q" else "; smallest margin and edits with a non-positive margin") + ")", not bad,
           f"{len(cmpr)} numbers compared; differing beyond the rounding of the verification's list: {[(r['log'], r['quantity'], r['mine'], r['verifier']) for r in bad]}")
    nn = [(lg, REP[lg]["n_near_scope_probes"], m["near_released_behaviour"]["n_rows"]) for lg in REP for m in REP[lg]["methods"].values() if "near_released_behaviour" in m]
    CK.add(4, "the near-scope probes of the gate-level cells are the rows with near_edit set in the per-row files (same number per log and method)", all(a == b for _, a, b in nn),
           f"{len(nn)} (log, method) pairs")
    jdump({"about": {"what": "reporting numbers printed next to far locality (amendment of 2026-11-12 06:16 UTC, decisions 4 and 5), per log",
                     "a": "near_released_*: valid heldout_disjoint probes that ask the question of an edit of the log, percent on which an edit is released; behaviour = per-row files "
                          "(mean over the behavioural draws), gate level = mean over the 20 calibration draws",
                     "b": "false_firing_20_draws: rate on heldout_disjoint per calibration draw against the bound of the two-block rule (5 / 101); where_logonly has no gate-level read-out",
                     "c": "capped_edits_20_draws: edits at their guard cap (for them the bound is void)", "d": "self_separation_margin: smallest margin and number of edits with a non-positive margin",
                     "e": "rel_kept_gained_lost: Rel exact against the deployed log; kept = target under both, gained = only under the method, lost = only under the deployed log; percent of "
                          f"all edits, of the first 100 edits and of the later edits (stream position >= {B.later_from})",
                     "methods": list(REPORT_METHODS), "written": time.strftime("%Y-%m-%d %H:%M:%S")},
           "logs": REP, "comparison_with_verification": cmpr}, os.path.join(B.fix, "reporting.json"))
    # ---- checks of the directory
    eq = equal_existing(B, E, MCN, REP)
    CK.add(0, f"numbers that also exist in out/paper_numbers.json are equal to the last stored digit ({'two decimals, four for the margin' if key == 'Q' else 'full float, tolerance 1e-9'})",
           all(r["equal"] for r in eq), f"{len(eq)} numbers compared on {len(B.logs)} logs; differing: {[(r['log'], r['name'], r['mine'], r['stored']) for r in eq if not r['equal']]}")
    coll = collisions(PN.num, (B.out + "/paper_numbers.json",) + EARLIER)
    CK.add(0, "no name of out/fixup/paper_numbers_fixup.json exists in out/paper_numbers.json or in the earlier fix-up files (e1_main, e2_curves)", not coll, f"collisions: {len(coll)}")
    CK.add(0, "every paper number has a documentation line", set(PN.num) == set(PN.doc) and all(PN.doc.values()), f"{len(PN.num)} names")
    mod = modified_since_start()
    CK.add(0, "no existing file of the experiments was modified (modification times against the start of this work; the new files of the fix-up excluded)", not mod["modified"],
           (mod["skipped"] + "; " if mod.get("skipped") else "") + f"start {mod['start_utc']} UTC; {mod['n_files_checked']} files checked in {', '.join(mod['directories'])}; modified: {mod['modified'][:5]}", skipped=bool(mod.get("skipped")))
    jdump(dict(sorted(PN.num.items())), os.path.join(B.fix, "paper_numbers_fixup.json"))
    jdump(dict(sorted(PN.doc.items())), os.path.join(B.fix, "paper_numbers_fixup_doc.json"))
    by_item = {str(i): sum(v == i for v in PN.item.values()) for i in sorted(set(PN.item.values()))}
    jdump({"written": time.strftime("%Y-%m-%d %H:%M:%S"), "code": fx.code_fingerprint(), "backbone": B.title, "endpoints_file": ep_file, "n_paper_numbers": len(PN.num),
           "n_paper_numbers_by_item": by_item, "n_collisions": len(coll), "collisions": coll, "equal_to_existing_paper_numbers": eq, "unmodified": mod, "n_mcnemar_tests": n_tests,
           "n_checks": len(CK.list), "n_checks_failed": sum(c["ok"] is False for c in CK.list), "n_checks_skipped": sum(bool(c.get("skipped")) for c in CK.list), "checks": CK.list}, os.path.join(B.fix, "checks.json"))
    print(f"{B.title}: paper numbers {len(PN.num)} {by_item}; collisions: {len(coll)}; checks failed: {sum(c['ok'] is False for c in CK.list)}, skipped: {sum(bool(c.get('skipped')) for c in CK.list)} of {len(CK.list)}  [{time.time() - t0:.0f}s]")

