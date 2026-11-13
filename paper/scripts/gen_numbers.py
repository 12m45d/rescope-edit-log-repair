#!/usr/bin/env python3
"""Numbers pipeline of the paper: scripts/numbers_manifest.json -> numbers.tex (and the generated table tab_logs.tex).

Rule of the project: no number and no experiment-specific string is typed into a .tex file. Every such value is a
macro; the manifest says which JSON file and which key it comes from. One manifest ENTRY gives one LaTeX MACRO:

    {"name": "exT", "source": "ex", "key": "T", "fmt": "{:d}", "fallback": "todo"}
      ->  \\newcommand{\\exT}{100\\xspace}            if the JSON file exists and the key resolves
      ->  \\newcommand{\\exT}{\\todo{exT}}            otherwise: a red placeholder, so the paper compiles at every stage
                                                    and an unresolved value is visible in the PDF

Entry fields
  name      macro name, letters only (TeX)
  source    tag of manifest["sources"] (path relative to ROOT = project_3), or
  file      a path relative to ROOT, given directly
  key       path into the JSON document. A LIST of keys / list indices is taken literally. A STRING is split at "."; a
            component may itself contain dots, and the longest literal match wins at every level, so both nested
            documents ("edit.question") and flat ones whose keys contain dots ("rollback.TB-Q100-ae-l2_3.c0.1.median_lf"
            as ONE key of a flat paper_numbers.json) are read as written
  fmt       Python format applied to the value (default "{}");  scale: factor applied first (e.g. 100 for a share)
  kind      "number" (default) | "text" (TeX-escaped string)
  map       {raw value: display text}: the value is looked up first (labels for identifiers)
  lookup    {"source" | "file", "key"}: a second look-up with the value just read; "{value}" in the key is replaced by it
            (e.g. the name that the paper prints for a log whose testbed name an experiment file holds). If the second
            look-up finds nothing, the raw value is printed and `--check` says so
  null      TeX printed when the key resolves to JSON null, i.e. the quantity does not apply (e.g. "--"); without it a
            null is a placeholder
  fallback  placeholder printed while the value is missing: "todo" (default: \\todo{<macro name>}), "tbd" (\\tbd, for
            table cells) or any other string, which is taken as TeX
  value     a constant instead of source / key: for display LABELS only (names of models), never for a measurement
Compact form for tables: manifest["families"] holds templates; a family {prefix, items | items_from, fields} expands
to one entry per (item, field) with the name <prefix><item tag><field>, "{var}" in file / key replaced by the item's
variables. `--list` prints every expanded entry, so the one-entry-one-macro table can be audited.

Derived sources: manifest["derived"] names modules of this directory with a function build(root, manifest, out) that
collect stored files into a JSON inside the paper directory (collect_setup.py -> scripts/data/setup_facts.json).
They are refreshed before the manifest is resolved; where the research tree is not readable the stored copy is used.

Table tab:logs: manifest["logs"] lists the logs that the paper uses ("items", in the order of the table; "in_table":
false = used, but without a row) and those that it does not use ("not_in_paper", each with its reason). tab_logs.tex
is generated from it (function tab_logs); its cells are the macros of the family "lg". numbers.tex and tab_logs.tex
belong together: copy both into a scratch build. Since 2026-11-11 the table ends with the PUBLIC TEXT LOGS of
manifest["text_logs"] (cells: family "tx", read from the description of the deployed logs that their build stored).
These rows are written only if the independent audit named there exists and declares every listed log "safe";
otherwise they are left out and the run says so (function text_log_rows).
The collector checks the boundary rule of the protocol on "items";
if it is violated, nothing is written.

    python3 scripts/gen_numbers.py              refresh derived sources, write numbers.tex and tab_logs.tex (make numbers)
    python3 scripts/gen_numbers.py --check      write nothing; list every placeholder that is still unfilled, with the
                                                file / key it waits for and the .tex lines that use it; stale derived
                                                sources; macros that no .tex file uses; \\tbd / \\todo typed in .tex files
    python3 scripts/gen_numbers.py --strict     as the default, but exit 1 if a placeholder is left (make strict);
                                                with --check: exit 1, nothing written
    python3 scripts/gen_numbers.py --list       print the expanded entries (macro, file, key, format, fallback)
    options: --no-collect (do not refresh derived sources), --out-dir DIR (write the .tex files there: scratch builds)
"""
import os as _go, sys as _gs  # ARTIFACT: a failed run leaves the stored output files untouched (rescope_paths.guard_outputs)
_gd = _go.environ.get("RESCOPE_HOME") or _go.path.dirname(_go.path.abspath(__file__))
while not _go.path.exists(_go.path.join(_gd, "rescope_paths.py")) and _go.path.dirname(_gd) != _gd: _gd = _go.path.dirname(_gd)
_gs.path.insert(0, _gd); import rescope_paths as _RPG; _RPG.guard_outputs(__file__, __name__)
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP
import importlib
import json
import os
import re
import sys
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal, localcontext

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
MANIFEST = os.path.join(HERE, "numbers_manifest.json")
sys.path.insert(0, HERE)

TEX_ESCAPES = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
               "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
MISSING = object()


def find_root():
    """ROOT = the directory that holds 02_research and 01_paper (PROJECT3_ROOT overrides)."""
    env = os.environ.get("PROJECT3_ROOT")
    if env:
        return env
    return _RP.ROOT          # the working-tree view of this repository (rescope_paths.py)


def tex_escape(s):
    return "".join(TEX_ESCAPES.get(ch, ch) for ch in str(s))


def fmt_number(fmt, value):
    """fmt.format(value), except that a float under a fixed-point format "{:.Nf}" is rounded on its DECIMAL value,
    half away from zero (4.55 -> 4.6, 4.45 -> 4.5), as a reader rounds by hand.
    Why (final check of 2026-11-11, CHECK_LOG.tex, D3): Python rounds the binary double, so the share 91 / 2000 =
    4.55 %, whose double lies just below 4.55, printed as 4.5, although the count rounds to 4.6
    (checks/check_tab_logs_rows.py recomputes the cells of Table tab:logs from the per-edit flags). The decimal value
    is the shortest decimal that reads back as the same double (repr), snapped to N + 9 decimals first, so that noise in
    the last bits of a computed value (4.549999999999999) does not hide a tie."""
    m = re.fullmatch(r"\{:\.(\d+)f\}", fmt)
    if m and isinstance(value, float) and value == value and abs(value) != float("inf"):
        n = int(m.group(1))
        with localcontext() as ctx:
            ctx.prec = 400                                     # any finite double, to N + 9 decimals
            d = Decimal(repr(value)).quantize(Decimal(1).scaleb(-(n + 9)), rounding=ROUND_HALF_EVEN)
            return format(d.quantize(Decimal(1).scaleb(-n), rounding=ROUND_HALF_UP), "f")
    return fmt.format(value)


def key_parts(key):
    if isinstance(key, (list, tuple)):
        return [str(k) for k in key]
    return str(key).split(".")


def _walk(obj, parts, literal):
    if not parts:
        return obj
    if isinstance(obj, list):
        try:
            return _walk(obj[int(parts[0])], parts[1:], literal)
        except (ValueError, IndexError):
            return MISSING
    if isinstance(obj, dict):
        # literal: one component = one key. Otherwise the longest run of components that, joined by ".", is a key
        for n in ([1] if literal else range(len(parts), 0, -1)):
            k = ".".join(parts[:n])
            if k in obj:
                r = _walk(obj[k], parts[n:], literal)
                if r is not MISSING:
                    return r
    return MISSING


def resolve(obj, key):
    """Value at the key path, or MISSING. A list is an exact path; a string is split at "." (docstring: key)."""
    return _walk(obj, key_parts(key), isinstance(key, (list, tuple)))


def subst(x, variables):
    """Replace {var} in a string or in every string of a key list."""
    if isinstance(x, list):
        return [subst(v, variables) for v in x]
    if isinstance(x, str):
        return re.sub(r"\{([A-Za-z_]+)\}", lambda m: str(variables.get(m.group(1), m.group(0))), x)
    return x


def expand(man):
    """Manifest -> list of entries, one per macro (explicit entries first, then the families, in order)."""
    out = [dict(m) for m in man.get("macros", [])]
    for fam in man.get("families", []):
        if "items_from" in fam:
            items = [(it["tag"], it) for it in man[fam["items_from"]]["items"]]
        else:
            items = list(fam["items"].items())
        for tag, variables in items:
            for field, spec in fam["fields"].items():
                e = dict(spec)
                e["name"] = fam["prefix"] + tag + field
                for k in ("file", "key", "source"):
                    if k in e:
                        e[k] = subst(e[k], variables)
                e["_family"] = fam["prefix"]
                out.append(e)
    return out


class Resolver:
    def __init__(self, man, root):
        self.man, self.root, self.cache = man, root, {}

    def rel(self, e):
        return e["file"] if "file" in e else self.man["sources"][e["source"]]

    def doc(self, rel):
        if rel not in self.cache:
            p = os.path.join(self.root, rel)
            try:
                with open(p) as f:
                    self.cache[rel] = json.load(f)
            except OSError:
                self.cache[rel] = MISSING
            except ValueError as err:                     # a file that is being written, or not JSON
                self.cache[rel] = ("unreadable", str(err))
        return self.cache[rel]

    def entry(self, e):
        """-> dict(name, body, status in {'ok', 'na', 'placeholder'}, where, reason)"""
        name = e["name"]
        if "value" in e:                                  # constant display label
            body = tex_escape(e["value"]) if e.get("kind", "text") == "text" else str(e["value"])
            return dict(name=name, body=body + "\\xspace", status="ok", where="constant label (manifest)", reason=None)
        rel, key = self.rel(e), e["key"]
        where = f"{rel} :: {'.'.join(key_parts(key))}"
        doc = self.doc(rel)
        reason = None
        if doc is MISSING:
            value, reason = MISSING, "file missing"
        elif isinstance(doc, tuple):
            value, reason = MISSING, "file unreadable: " + doc[1][:60]
        else:
            value = resolve(doc, key)
            if value is MISSING:
                reason = "key missing"
        if value is None:
            if "null" in e:
                return dict(name=name, body=e["null"] + "\\xspace", status="na", where=where, reason="not applicable (null)")
            value, reason = MISSING, "value is null"
        note = None
        if value is not MISSING and "lookup" in e:
            lk = e["lookup"]
            rel2 = lk["file"] if "file" in lk else self.man["sources"][lk["source"]]
            doc2 = self.doc(rel2)
            v2 = resolve(doc2, subst(lk["key"], {"value": value})) if isinstance(doc2, (dict, list)) else MISSING
            if v2 is MISSING or v2 is None:
                note = f"lookup of {value!r} in {rel2} found nothing: raw value printed"
            else:
                where += f"  ->  {rel2} :: {'.'.join(key_parts(subst(lk['key'], {'value': value})))}"
                value = v2
        if value is not MISSING:
            try:
                if "map" in e:
                    value = e["map"][str(value)]
                if e.get("kind", "number") == "text":
                    body = tex_escape(value)
                else:
                    if "scale" in e:
                        value = value * e["scale"]
                    fmt = e.get("fmt", "{}")
                    if isinstance(value, float) and value.is_integer() and re.search(r"d\}", fmt):
                        value = int(value)
                    body = fmt_number(fmt, value)
                    if body.startswith("-"):              # a minus sign, not a hyphen, in text and in math
                        body = "\\ensuremath{-}" + body[1:]
                return dict(name=name, body=body + "\\xspace", status="ok", where=where, reason=None, note=note)
            except (KeyError, ValueError, TypeError) as err:
                reason = f"cannot format {value!r}: {type(err).__name__} {err}"
        fb = e.get("fallback", "todo")
        body = f"\\todo{{{name}}}" if fb == "todo" else "\\tbd" if fb == "tbd" else fb
        return dict(name=name, body=body, status="placeholder", where=where, reason=reason, fallback=fb)


def defined_elsewhere():
    """Macro names defined by the hand-written preamble files (a generated macro must not clash with them)."""
    names = {}
    for fn in ("notation.tex", "preamble.tex", "theory_numbers.tex", "theory_helpers.tex"):
        p = os.path.join(PAPER, fn)
        if os.path.exists(p):
            for m in re.finditer(r"\\(?:re)?newcommand\*?\{?\\([A-Za-z]+)\}?|\\providecommand\*?\{?\\([A-Za-z]+)\}?|"
                                 r"\\DeclareMathOperator\*?\{\\([A-Za-z]+)\}|\\NewDocumentCommand\{?\\([A-Za-z]+)\}?",
                                 open(p, errors="replace").read()):
                names[next(g for g in m.groups() if g)] = fn
    return names


def strip_comments(src):
    return re.sub(r"(?m)(?<!\\)%.*$", "", src)


def tex_sources():
    """(file name, text without comments) of the hand-written .tex files of the paper directory."""
    skip = {"numbers.tex", "theory_numbers.tex", "NOTES_FOR_AUTHORS.tex", "CHECK_LOG.tex"}
    for fn in sorted(os.listdir(PAPER)):
        if fn.endswith(".tex") and fn not in skip:
            yield fn, strip_comments(open(os.path.join(PAPER, fn), errors="replace").read())


def uses(names):
    """{macro name: [file:line, ...]} over the .tex files (comments excluded)."""
    out = {n: [] for n in names}
    pat = re.compile(r"\\([A-Za-z]+)")
    for fn, src in tex_sources():
        for i, line in enumerate(src.split("\n"), 1):
            for m in pat.finditer(line):
                if m.group(1) in out:
                    loc = f"{fn}:{i}"
                    if loc not in out[m.group(1)]:
                        out[m.group(1)].append(loc)
    return out


def refresh_derived(man, root, check):
    """Run (or, in check mode, only inspect) the collectors of manifest['derived']. Returns report lines."""
    lines = []
    for d in man.get("derived", []):
        mod = importlib.import_module(d["module"])
        out = os.path.join(root, d["out"])
        if check:
            st = mod.stale(root, out)
            if st is None:
                lines.append(f"derived {d['out']}: inputs not readable here, stored copy used")
            elif st:
                lines.append(f"derived {d['out']}: STALE ({len(st)} input(s) changed; run `make numbers`): " + "; ".join(st[:6])
                             + (" ..." if len(st) > 6 else ""))
            else:
                lines.append(f"derived {d['out']}: up to date")
            if os.path.exists(out):
                doc = json.load(open(out))
                for w in doc.get("warnings", []):
                    lines.append(f"  warning of the collector: {w}")
                for b in doc.get("boundary_violations", []):
                    lines.append(f"  BOUNDARY VIOLATION: {b}")
            continue
        try:
            doc, warnings = mod.build(root, man, out)
            lines.append(f"derived {d['out']}: refreshed")
            for w in warnings:
                lines.append(f"  warning of the collector: {w}")
            if doc.get("boundary_violations"):
                for b in doc["boundary_violations"]:
                    lines.append(f"  BOUNDARY VIOLATION: {b}")
                raise SystemExit("\n".join(lines) + "\nboundary rule of PROTOCOL.md violated by manifest['logs']: nothing written")
        except OSError as err:
            lines.append(f"derived {d['out']}: NOT refreshed ({err}); " + ("stored copy used" if os.path.exists(out) else "no stored copy"))
    return lines


# ---------------------------------------------------------------------------------------------- generated table
problems_note = []          # messages of the last tab_logs() call that are not errors (text logs left out, and why)


def text_log_rows(man, res):
    """Rows of the public text logs (manifest["text_logs"]) for Table tab:logs, or ([], reason) if they must be left
    out. Condition (task of 2026-11-11): the independent audit of these logs exists and declares every listed log
    "safe" (per_log.<log>.verdict) with all of its checks passed. Columns that do not apply (I-Gen, I-Loc) print "--".
    The superscript of a row comes from the audit's class of the log (labels.class_mark), not from the manifest."""
    spec = man.get("text_logs")
    if not spec:
        return [], None
    audit = res.doc(spec["audit"])
    if not isinstance(audit, dict):
        return [], f"text logs left out of Table tab:logs: audit file {spec['audit']} missing or unreadable"
    if audit.get("all_checks_pass") is not True:
        return [], "text logs left out of Table tab:logs: the audit does not report all checks as passed"
    rows = []
    for it in spec["items"]:
        v = audit.get("per_log", {}).get(it["log"], {})
        if v.get("verdict") != "safe":
            return [], f"text logs left out of Table tab:logs: the audit does not declare {it['log']} safe"
        mark = spec.get("class_mark", {}).get(v.get("class"), "")
        t = it["tag"]
        cells = [f"{{\\tx{t}Name}}{mark}", f"{{\\tx{t}T}}", f"{{\\tx{t}Rel}}", f"{{\\tx{t}TGen}}", "--",
                 f"{{\\tx{t}TLoc}}", "--", f"{{\\tx{t}Cmp}}"]
        rows.append("    " + " & ".join(cells) + f" \\\\  % {it['log']} (audit: {v.get('class')})")
    return rows, None


def tab_logs(man, res):
    """LaTeX source of Table tab:logs. One row per item of manifest['logs']['items'] (those without "in_table": false),
    in that order, with a small space between groups; every cell is a macro of numbers.tex (family 'lg'), so nothing in
    the table is typed. The group that the manifest declares for a log (backbone, editor, benchmark) is CHECKED against
    what the testbed stored for it. Options of manifest['logs']: caption, font (default \\footnotesize), group_headings
    (true: one heading row per group, composed from 'labels'; default false: the names of the logs carry the group),
    labels.role_mark (superscripts of development logs and working points)."""
    spec = man["logs"]
    setup = res.doc(man["sources"]["setup"])
    logs = setup.get("logs", {}) if isinstance(setup, dict) else {}
    lab = spec["labels"]
    groups = {g["id"]: g for g in spec["groups"]}
    problems = []
    rows, last = [], None
    del problems_note[:]
    for it in spec["items"]:
        if not it.get("in_table", True):
            continue
        g = groups[it["group"]]
        r = logs.get(it["log"], {})
        if r.get("complete"):                                 # the declared group must be what the testbed stored
            for k in ("backbone", "editor", "dataset"):
                if r.get(k) != g[k]:
                    problems.append(f"{it['log']}: group {g['id']} declares {k} = {g[k]!r}, the testbed stored {r.get(k)!r}")
        if it["group"] != last:
            if last:
                rows.append("    \\addlinespace[2pt]")
            if spec.get("group_headings", False):                 # a heading row per group; off: the names carry the group
                head = f"{lab['backbone_macro'][g['backbone']]}, {lab['editor'][g['editor']]}"
                if g["dataset"] != spec["default_dataset"]:
                    head += f", {lab['dataset'][g['dataset']]}"
                rows.append(f"    \\multicolumn{{8}}{{@{{}}l}}{{\\emph{{{head}}}}} \\\\")
            last = it["group"]
        t = it["tag"]
        mark = lab["role_mark"].get(it.get("role") or "unseen", "")
        # braces: the macros end with \xspace, which must not see the alignment tab
        cells = [f"{{\\lg{t}Name}}{mark}"] + [f"{{\\lg{t}{f}}}" for f in ("T", "Rel", "TGen", "IGen", "TLoc", "ILoc", "Cmp")]
        rows.append("    " + " & ".join(cells) + f" \\\\  % {it['log']}")
    trows, tnote = text_log_rows(man, res)
    if trows:
        rows.append("    \\addlinespace[2pt]")
        rows.extend(trows)
    if tnote:
        problems_note.append(tnote)
    cap, font = spec["caption"], spec.get("font", "\\footnotesize")
    if not trows and spec.get("caption_without_text_logs"):
        cap = spec["caption_without_text_logs"]
    src = "\n".join([
        "% tab_logs.tex -- GENERATED by scripts/gen_numbers.py (function tab_logs) from scripts/numbers_manifest.json, key",
        "% \"logs\" (list of logs, groups, labels, caption). Do not edit by hand: change the manifest (rows, caption) or the",
        "% function (layout) and run `make numbers`. Every cell is a macro of numbers.tex (\\lg<tag><field>), filled from",
        "% scripts/data/setup_facts.json, which scripts/collect_setup.py collects from the stored files of the testbed",
        "% (06_testbed_v3/<log>/{metrics.json, probe_records_v2.jsonl, complaints_v2.json}). A log that is not built yet",
        "% keeps red tbd cells; `python3 scripts/gen_numbers.py --check` lists them. Input by sec_experiments.tex.",
        "% Last block: the public text logs (manifest key \"text_logs\"; macros \\tx<tag><field> from the stored description",
        "% of the deployed logs, 09_testbed_scale/text_public/out/text_logs.json); present only while their independent",
        "% audit declares every listed log safe.",
        "\\begin{table}[t]",
        f"  \\caption{{{cap}}}",
        "  \\label{tab:logs}",
        f"  {font}\\setlength{{\\tabcolsep}}{{3.4pt}}",
        "  \\begin{tabular}{@{}l@{\\hspace{5pt}}rrrrrrr@{}}",
        "    \\toprule",
        "    Log & $T$ & \\Rel & \\TGen & \\IGen & \\TLoc & \\ILoc & \\Done \\\\",
        "    \\midrule",
        *rows,
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
        ""])
    return src, problems


def write_atomic(path, text):
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def main(argv):
    flags = {a for a in argv if a.startswith("--")}
    unknown = flags - {"--strict", "--check", "--list", "--no-collect", "--out-dir"}
    if unknown:
        sys.exit(f"unknown option(s): {', '.join(sorted(unknown))}\n{__doc__}")
    strict, check, listing = "--strict" in flags, "--check" in flags, "--list" in flags
    out_dir = PAPER
    if "--out-dir" in argv:
        if argv.index("--out-dir") + 1 >= len(argv):
            sys.exit("--out-dir needs a directory")
        out_dir = argv[argv.index("--out-dir") + 1]
    root = find_root()
    man = json.load(open(MANIFEST))
    if not ("--no-collect" in flags or listing or check):               # ARTIFACT: collection mode needs the released logs; stop before anything is rewritten
        absent = _RP.missing_archives()
        if absent:
            sys.exit("gen_numbers.py without --no-collect re-collects the facts of Table 1 from the released logs and pools; these archives are not unpacked under "
                     + _RP.LOGS + ":\n" + "".join(f"  {a}   (absent: {w})\n" for a, w in absent)
                     + "Unpack them (README, 'Data archives':  tar -xf <archive> -C data/logs), or run\n  python paper/scripts/gen_numbers.py --no-collect\n"
                     "which reads the stored copy paper/scripts/data/setup_facts.json and needs no archive.  Nothing was written.")
    report = [] if "--no-collect" in flags or listing else refresh_derived(man, root, check)
    entries = expand(man)
    clash = defined_elsewhere()
    seen = set()
    for e in entries:
        n = e["name"]
        assert re.fullmatch(r"[A-Za-z]+", n), f"macro name must be alphabetic: {n}"
        assert n not in seen, f"duplicate macro: {n}"
        assert n not in clash, f"macro \\{n} is already defined in {clash[n]}"
        assert ("value" in e) != ("key" in e), f"entry {n}: give either a constant 'value' or 'source'/'file' + 'key'"
        seen.add(n)
    res = Resolver(man, root)
    done = [res.entry(e) for e in entries]
    if listing:
        for e, d in zip(entries, done):
            print(f"\\{e['name']:24s} <- {d['where']}  fmt={e.get('fmt', '{}')!r} kind={e.get('kind', 'number')} "
                  f"fallback={e.get('fallback', 'todo')!r}  [{d['status']}]")
        return 0
    ok = [d for d in done if d["status"] == "ok"]
    na = [d for d in done if d["status"] == "na"]
    ph = [d for d in done if d["status"] == "placeholder"]
    table, problems = tab_logs(man, res)
    if problems:
        sys.exit("tab_logs: the groups declared in the manifest contradict the stored files:\n  " + "\n  ".join(problems))
    if not check:
        lines = ["% AUTO-GENERATED by scripts/gen_numbers.py from scripts/numbers_manifest.json -- do not edit by hand.",
                 "% One macro per manifest entry: <value>  % <JSON file relative to project_3> :: <key>.",
                 "% A macro whose source does not exist yet is a red placeholder (\\todo{<macro name>} or \\tbd); its line",
                 "% is marked PLACEHOLDER. `python3 scripts/gen_numbers.py --check` lists what each of them waits for.", ""]
        fam = None
        for e, d in zip(entries, done):
            if e.get("_family") != fam:
                fam = e.get("_family")
                lines.append(f"% ---- family {fam}" if fam else "% ---- explicit entries")
            if d["status"] == "placeholder":
                lines.append(f"\\newcommand{{\\{d['name']}}}{{{d['body']}}}  % PLACEHOLDER \\todo{{{d['name']}}} <- {d['where']} ({d['reason']})")
            else:
                lines.append(f"\\newcommand{{\\{d['name']}}}{{{d['body']}}}  % {d['where']}" + (" (not applicable)" if d["status"] == "na" else ""))
        write_atomic(os.path.join(out_dir, "numbers.tex"), "\n".join(lines) + "\n")
        write_atomic(os.path.join(out_dir, "tab_logs.tex"), table)
        print(f"wrote {os.path.join(out_dir, 'numbers.tex')}: {len(ok)} values, {len(na)} not applicable, {len(ph)} placeholders")
        print(f"wrote {os.path.join(out_dir, 'tab_logs.tex')}")
    else:
        print(f"manifest: {len(done)} macros: {len(ok)} values, {len(na)} not applicable, {len(ph)} placeholders (nothing written)")
        cur = os.path.join(PAPER, "numbers.tex")
        if os.path.exists(cur):
            now = {m.group(1): m.group(2) for m in re.finditer(r"(?m)^\\newcommand\{\\([A-Za-z]+)\}\{(.*?)\}  % ", open(cur).read())}
            diff = sorted(d["name"] for d in done if now.get(d["name"]) != d["body"])
            gone = sorted(set(now) - {d["name"] for d in done})
            if diff or gone:
                print(f"numbers.tex is OUT OF DATE: {len(diff)} macro(s) would change, {len(gone)} would disappear (run `make numbers`)"
                      + (": " + ", ".join("\\" + n for n in (diff + gone)[:8]) + (" ..." if len(diff + gone) > 8 else "")))
            else:
                print("numbers.tex is up to date with the manifest and its sources")
            tl = os.path.join(PAPER, "tab_logs.tex")
            if not os.path.exists(tl) or open(tl).read() != table:
                print("tab_logs.tex is OUT OF DATE (run `make numbers`)")
    for line in report:
        print(line)
    for line in problems_note:
        print("NOTE: " + line)
    used = uses([d["name"] for d in done])
    for d in done:
        if d.get("note"):
            print(f"note: \\{d['name']}: {d['note']}")
    if ph:
        print("placeholders still unfilled, by the file they wait for:")
        by = {}
        for d in ph:
            rel = d["where"].split(" :: ")[0]
            by.setdefault((rel, d["reason"].split(":")[0]), []).append(d)
        for (rel, why), ds in sorted(by.items()):
            print(f"  [{why}] {rel}  ({len(ds)} macro(s))")
            for d in ds:
                u = used[d["name"]]
                print(f"      \\{d['name']:22s} <- {d['where'].split(' :: ')[1]:44s} "
                      + ("used in " + ", ".join(u[:4]) + (" ..." if len(u) > 4 else "") if u else "not used in any .tex file"))
    if check:
        unused = sorted(d["name"] for d in done if not used[d["name"]])
        if unused:
            print(f"macros that no .tex file uses ({len(unused)}): " + ", ".join("\\" + n for n in unused))
        typed = []
        for fn, src in tex_sources():
            n_tbd = len(re.findall(r"\\tbd(?![A-Za-z])", src)) - len(re.findall(r"\\newcommand\{\\tbd\}", src))
            n_todo = len(re.findall(r"\\todo\{", src)) - len(re.findall(r"\\NewDocumentCommand\{\\todo\}", src))
            if n_tbd > 0 or n_todo > 0:
                typed.append(f"{fn}: {n_tbd} \\tbd, {n_todo} \\todo{{...}}")
        if typed:
            print("placeholders typed in .tex files (each names its experiment and key in a comment or caption):")
            for t in typed:
                print("  " + t)
    if ph and not check:                                                # ARTIFACT: a placeholder means a source file is missing: fail, the guard puts every file back
        print(f"gen_numbers.py: {len(ph)} placeholder(s): a source file of these macros is missing (listed above with the file each waits for). "
              "Exit code 1; rescope_paths.guard_outputs restores numbers.tex and the other generated files.", file=sys.stderr)
        return 1
    return 1 if strict and ph else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
