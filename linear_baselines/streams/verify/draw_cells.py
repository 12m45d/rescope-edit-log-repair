"""P12 verifier -- the two further (method, n) cells per log, drawn at random BEFORE any GPU number of mine exists.

Universe = the cells of the report's two summary tables (methods x n in {1, 10, 100, ALL}; ALLPOS only n in {1, 10}),
minus the reported best operating point (COMP_B, ALL), which is checked anyway.  Seeded numpy draw, written once to
out/draw.json (never redrawn: the file is kept if it exists).
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = ("TB-Q100-ae-l2_3", "TB-L100-ae-l2_3")
METHODS = ("COMP_B", "COMP_BP", "COMP_B_ALLPOS", "REEDIT_B", "RB_TOT", "RB_UNION")
NS = (1, 10, 100, "ALL")
SEED = [2026, 11, 10, 12]            # date + experiment number, fixed before the draw

cells = [(m, n) for m in METHODS for n in NS if not (m == "COMP_B_ALLPOS" and n in (100, "ALL")) and (m, n) != ("COMP_B", "ALL")]
p = os.path.join(HERE, "out", "draw.json")
if os.path.exists(p):
    print("draw exists:", json.load(open(p)))
else:
    out = {"seed": SEED, "universe": [list(c) for c in cells], "n_universe": len(cells), "draw": {}}
    for li, lg in enumerate(LOGS):
        rng = np.random.default_rng(SEED + [li])
        idx = rng.choice(len(cells), 2, replace=False)
        out["draw"][lg] = [list(cells[int(i)]) for i in idx]
    json.dump(out, open(p, "w"), indent=1)
    print(json.dumps(out["draw"]), len(cells))
