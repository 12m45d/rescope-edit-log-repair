"""fx.io: ONE result format for all experiments.

  out/results.json        a JSON LIST of flat records with the fields FIELDS:
                          experiment, log, method, variant, alpha, n, draw, pool, metric, value, ci_lo, ci_hi, n_units
                          (alpha / n / draw are null where a method has no calibration; draw = null for a summary over draws,
                          whose metric name ends in '@mean' / '@p05' / '@p95'; further keys of a record are allowed and kept)
  out/results_meta.json   what produced the records (code fingerprints, time, arguments)
  out/rows/<tag>.jsonl.gz per-row files (one JSON object per row), so that every number can be recomputed
  out/paper_numbers.json  flat dictionary name -> value for the paper's LaTeX macros (names: letters only)
  out/table.txt           a plain-text table rendered by code from the records

    R = fx.io.Results("E1", out_dir)
    R.add(log=..., method=..., metric="tgen_exact", value=62.0, ci_lo=.., ci_hi=.., n_units=100, alpha=0.05, n=200, draw=0, pool="edits")
    R.add_draws(log=..., method=..., metric="fire", values_by_draw={0: 4.3, 1: 5.1, ...}, alpha=0.05, n=200, pool="heldout_disjoint")
    R.rows("rescope_draw0_probes", rows); R.save(); R.paper({"EoneQwenTGen": 62.0}); R.table(text)
Everything is written atomically (temporary file + rename).  Non-finite numbers become the strings 'inf' / '-inf' / 'nan'.
"""
from __future__ import annotations

import gzip
import json
import math
import os
import re
import time

import fx
from . import stats

FIELDS = ("experiment", "log", "method", "variant", "alpha", "n", "draw", "pool", "metric", "value", "ci_lo", "ci_hi", "n_units")


def clean(x):
    """JSON-safe copy: numpy / torch scalars and arrays -> Python, non-finite floats -> 'inf' / '-inf' / 'nan'."""
    import numpy as np
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        x = float(x)
    if hasattr(x, "numel") and hasattr(x, "tolist"):                    # torch tensor
        return clean(x.tolist())
    if isinstance(x, float) and not math.isfinite(x):
        return "nan" if math.isnan(x) else ("inf" if x > 0 else "-inf")
    if hasattr(x, "numerator") and hasattr(x, "denominator") and not isinstance(x, (int, bool)):      # Fraction
        return f"{x.numerator}/{x.denominator}"
    if callable(x):                                                     # e.g. the setup hook of a registry entry
        return f"<callable {getattr(x, '__qualname__', type(x).__name__)}>"
    return x


def jdump(obj, path: str, indent=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(clean(obj), f, indent=indent, allow_nan=False)
    os.replace(tmp, path)


def jload(path: str):
    with open(path) as f:
        return json.load(f)


def record(experiment, log, method, metric, value, variant="", alpha=None, n=None, draw=None, pool="", ci_lo=None, ci_hi=None, n_units=None, **extra) -> dict:
    """One flat record of the result schema (FIELDS first, then any further keys)."""
    for nm, v in (("value", value), ("ci_lo", ci_lo), ("ci_hi", ci_hi)):
        if v is not None and not isinstance(v, (int, float)) and not hasattr(v, "dtype"):
            raise TypeError(f"{nm} must be a number or None, got {type(v).__name__}")
    r = {"experiment": str(experiment), "log": str(log), "method": str(method), "variant": "" if variant is None else str(variant),
         "alpha": None if alpha is None else float(alpha), "n": None if n is None else int(n), "draw": None if draw is None else int(draw),
         "pool": "" if pool is None else str(pool), "metric": str(metric), "value": None if value is None else float(value),
         "ci_lo": None if ci_lo is None else float(ci_lo), "ci_hi": None if ci_hi is None else float(ci_hi),
         "n_units": None if n_units is None else int(n_units)}
    r.update(extra)
    return r


def check(records) -> int:
    """Raise if a record lacks a field of the schema -> number of records."""
    for i, r in enumerate(records):
        miss = [f for f in FIELDS if f not in r]
        if miss:
            raise ValueError(f"record {i} lacks {miss}")
    return len(records)


def save_rows(path: str, rows):
    """Per-row file: gzip-compressed JSON lines."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + f".tmp{os.getpid()}"
    with gzip.open(tmp, "wt") as f:
        for r in rows:
            f.write(json.dumps(clean(r), allow_nan=False) + "\n")
    os.replace(tmp, path)


def load_rows(path: str) -> list:
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


class Results:
    """The records of one experiment and the files next to them (see the module docstring)."""

    def __init__(self, experiment: str, out_dir: str, load: bool = False):
        self.experiment, self.out = str(experiment), out_dir
        self.records, self.meta = [], {"experiment": self.experiment, "code": fx.code_fingerprint(), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
        os.makedirs(out_dir, exist_ok=True)
        if load and os.path.exists(self.path("results.json")):
            self.records = jload(self.path("results.json"))

    def path(self, name: str) -> str:
        return os.path.join(self.out, name)

    def add(self, log, method, metric, value, **kw) -> dict:
        r = record(self.experiment, log, method, metric, value, **kw)
        self.records.append(r)
        return r

    def add_stat(self, log, method, metric, stat: dict, **kw) -> dict:
        """A record from a fx.stats result ({value, ci_lo, ci_hi, n_units, ...})."""
        return self.add(log, method, metric, stat.get("value"), ci_lo=stat.get("ci_lo"), ci_hi=stat.get("ci_hi"), n_units=stat.get("n_units"), **kw)

    def add_nested(self, log, method, nested: dict, prefix: str = "", **kw) -> int:
        """Records from a nested dict of fx.metrics ({group: {metric: number | stat dict}}): metric name = 'group.metric'."""
        n = 0
        for k, v in nested.items():
            name = f"{prefix}{k}"
            if isinstance(v, dict) and "value" in v and ("ci_lo" in v or "n_units" in v):
                self.add_stat(log, method, name, v, **kw)
                n += 1
            elif isinstance(v, dict):
                n += self.add_nested(log, method, v, prefix=name + ".", **kw)
            elif v is None or isinstance(v, (int, float)) and not isinstance(v, bool):
                self.add(log, method, name, v, **kw)
                n += 1
        return n

    def add_draws(self, log, method, metric, values_by_draw: dict, **kw) -> dict:
        """One record per calibration draw + the summary over draws (metric@mean with ci_lo / ci_hi = the 5th / 95th percentile over
        draws, and metric@p05 / metric@p95; n_units = number of draws)."""
        for d, v in sorted(values_by_draw.items()):
            self.add(log, method, metric, v, draw=d, **kw)
        s = stats.over_draws(list(values_by_draw.values()))
        self.add(log, method, metric + "@mean", s["mean"], ci_lo=s["p05"], ci_hi=s["p95"], n_units=s["n_draws"], **kw)
        self.add(log, method, metric + "@p05", s["p05"], n_units=s["n_draws"], **kw)
        self.add(log, method, metric + "@p95", s["p95"], n_units=s["n_draws"], **kw)
        return s

    def rows(self, tag: str, rows) -> str:
        """Write per-row flags next to the results: out/rows/<tag>.jsonl.gz."""
        if not re.fullmatch(r"[A-Za-z0-9_.+=-]+", tag):
            raise ValueError("row-file tag: letters, digits, _ . + = - only")
        p = self.path(os.path.join("rows", tag + ".jsonl.gz"))
        save_rows(p, rows)
        return p

    def save(self, **meta) -> str:
        check(self.records)
        self.meta.update(meta, written=time.strftime("%Y-%m-%d %H:%M:%S"), n_records=len(self.records), fields=list(FIELDS))
        jdump(self.records, self.path("results.json"))
        jdump(self.meta, self.path("results_meta.json"), indent=1)
        return self.path("results.json")

    def paper(self, numbers: dict, strict_names: bool = True) -> str:
        """Merge numbers into out/paper_numbers.json (name -> value).  Names become LaTeX macro names: letters only."""
        for k, v in numbers.items():
            if strict_names and not re.fullmatch(r"[A-Za-z]+", k):
                raise ValueError(f"paper number {k!r}: a LaTeX macro name holds letters only")
            if v is not None and not isinstance(v, (int, float, str)) and not hasattr(v, "dtype"):
                raise TypeError(f"paper number {k!r}: a number or a string")
        p = self.path("paper_numbers.json")
        old = jload(p) if os.path.exists(p) else {}
        old.update(clean(numbers))
        jdump(dict(sorted(old.items())), p, indent=1)
        return p

    def table(self, text: str, name: str = "table.txt") -> str:
        tmp = self.path(name) + f".tmp{os.getpid()}"
        with open(tmp, "w") as f:
            f.write(text if text.endswith("\n") else text + "\n")
        os.replace(tmp, self.path(name))
        return self.path(name)


# ----------------------------------------------------------------------------- script-rendered plain-text tables
def fmt(v, nd: int = 1, lo=None, hi=None) -> str:
    """'12.3' or '12.3 [10.1, 14.2]'; '-' for None."""
    if v is None or isinstance(v, str):
        return "-" if v is None else v
    s = f"{v:.{nd}f}"
    return s if lo is None or hi is None else f"{s} [{lo:.{nd}f}, {hi:.{nd}f}]"


def text_table(headers, rows, align=None) -> str:
    """Aligned plain-text table (first column left, the others right by default)."""
    rows = [[str(c) for c in r] for r in rows]
    cols = list(zip(*([list(headers)] + rows))) if rows else [[h] for h in headers]
    w = [max(len(str(x)) for x in c) for c in cols]
    align = align or ["l"] + ["r"] * (len(headers) - 1)

    def line(r):
        return "  ".join(str(x).ljust(w[i]) if align[i] == "l" else str(x).rjust(w[i]) for i, x in enumerate(r)).rstrip()
    return "\n".join([line(headers), "  ".join("-" * x for x in w)] + [line(r) for r in rows])


def pivot(records, row=("method", "variant"), col=("metric",), where=None, nd: int = 1, ci: bool = True, col_order=None, row_order=None) -> str:
    """Plain-text pivot of flat records: one line per distinct `row` key, one column per distinct `col` key; a cell shows
    value [ci_lo, ci_hi].  where: predicate on a record.  Several records in one cell are an error (filter first)."""
    cells, rkeys, ckeys = {}, [], []
    for r in records:
        if where is not None and not where(r):
            continue
        rk = tuple("" if r.get(k) is None else r.get(k) for k in row)
        ck = tuple("" if r.get(k) is None else r.get(k) for k in col)
        if (rk, ck) in cells:
            raise ValueError(f"two records in cell {rk} x {ck}: add a filter (where=) or a key")
        cells[(rk, ck)] = fmt(r["value"], nd, r.get("ci_lo") if ci else None, r.get("ci_hi") if ci else None)
        if rk not in rkeys:
            rkeys.append(rk)
        if ck not in ckeys:
            ckeys.append(ck)
    if col_order:
        ckeys = [c for c in col_order if c in ckeys] + [c for c in ckeys if c not in col_order]
    if row_order:
        rkeys = [c for c in row_order if c in rkeys] + [c for c in rkeys if c not in row_order]
    lab = lambda k: " / ".join(str(x) for x in k if x != "")            # noqa: E731
    return text_table([" / ".join(row)] + [lab(c) for c in ckeys], [[lab(rk)] + [cells.get((rk, ck), "-") for ck in ckeys] for rk in rkeys])
