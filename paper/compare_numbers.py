#!/usr/bin/env python
"""Compare the regenerated numbers file with the one the paper was built from, macro by macro.

    python paper/scripts/gen_numbers.py         regenerates paper/numbers.tex, paper/tab_logs.tex, the table bodies paper/tab_*.tex and the
                                                derived facts paper/scripts/data/*.json from experiments/, text_logs/ and the released logs
    python paper/compare_numbers.py             compares paper/numbers.tex and the table bodies with paper/expected/ (the files of the paper)
    python paper/compare_numbers.py A.tex B.tex compares two numbers files

Exit code 1 if a macro differs or if the regenerated file holds a placeholder.  A macro whose source file is missing is a placeholder
in the regenerated file; it is listed with the file it waits for (in the released repository every source is present: 0 placeholders).
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RX = re.compile(r"(?m)^\\newcommand\{\\([A-Za-z]+)\}\{(.*?)\}  % (.*)$")


def macros(path):
    with open(path) as f:
        return {m.group(1): (m.group(2), m.group(3)) for m in RX.finditer(f.read())}


def main(argv):
    new, old = (argv + [os.path.join(HERE, "numbers.tex"), os.path.join(HERE, "expected", "numbers.tex")])[:2] if len(argv) < 2 else argv[:2]
    for f in (new, old):                                               # ARTIFACT: a clear message instead of a traceback
        if not os.path.exists(f):
            print(f"{f} not found: generate it first (python paper/scripts/gen_numbers.py --no-collect; README, QUICK START, step 4)")
            return 2
    a, b = macros(new), macros(old)
    ph = sorted(n for n, (v, w) in a.items() if w.startswith("PLACEHOLDER"))
    diff = sorted(n for n in set(a) & set(b) if a[n][0] != b[n][0] and n not in ph)
    only_new, only_old = sorted(set(a) - set(b)), sorted(set(b) - set(a))
    same = len(set(a) & set(b)) - len(diff) - len([n for n in ph if n in b])
    print(f"{new}: {len(a)} macros; {old}: {len(b)} macros")
    print(f"identical: {same}; different: {len(diff)}; only in the regenerated file: {len(only_new)}; only in the expected file: {len(only_old)}; "
          f"placeholders in the regenerated file (source not in the repository): {len(ph)}")
    for n in diff:
        print(f"  DIFFERENT \\{n}: regenerated {a[n][0]!r} | expected {b[n][0]!r}   <- {a[n][1]}")
    for n in ph:
        print(f"  placeholder \\{n}: expected {b.get(n, ('?',))[0]!r}   <- {a[n][1][:160]}")
    for n in only_new:
        print(f"  only regenerated \\{n}")
    for n in only_old:
        print(f"  only expected \\{n}")
    tabs_bad = []
    if len(argv) < 2:
        exp = os.path.join(HERE, "expected")
        tabs = sorted(f for f in os.listdir(exp) if f.startswith("tab_") and f.endswith(".tex"))
        for t in tabs:
            p = os.path.join(HERE, t)
            if not os.path.exists(p) or open(p).read() != open(os.path.join(exp, t)).read():
                tabs_bad.append(t)
        print(f"generated table bodies identical to the expected files: {len(tabs) - len(tabs_bad)} of {len(tabs)}" + (f"; DIFFERENT or missing: {tabs_bad}" if tabs_bad else ""))
    return 1 if diff or only_new or only_old or tabs_bad or ph else 0   # ARTIFACT: a placeholder is a lost number


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
