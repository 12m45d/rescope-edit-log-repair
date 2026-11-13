"""Verification of E3 (anomaly: cosine_published has the flags of cosine_default): the published all-positions cosine gate (0.65) and its
last-position form through the plain Evaluator on a seeded sample of one log; where does the all-positions gate fire?
    gq run --mem 10500 -- python v_pub.py TB-Q500-ae-l2_3      -> out/pub_<log>.json"""
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP
import gzip, json, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np, torch
torch.set_num_threads(4)
sys.path.insert(0, f"{_RP.ROOT}/02_research/code")
from rescope import Evaluator, cosine_gate, from_harness, load
ROOT = f"{_RP.ROOT}/02_research/experiments/10_final/e3_qwen/verify"
name = sys.argv[1]
H = load(name, device="cuda:0", mem_gb=(int(os.environ["GQ_MEM"]) - 600) / 1024.0)
ev = Evaluator(H, hard_negatives=False)
log = from_harness(H)
hd = ev.probe_items("heldout_disjoint")
pi = np.sort(np.random.default_rng(20261115).choice(len(hd), 150, replace=False))
es = sorted(int(i) for i in np.random.default_rng(20261116).choice(H.T, 50, replace=False))
items = [hd[int(i)] for i in pi] + ev.edit_items(("rel", "tgen", "igen"), es)
G = {"all": cosine_gate(log, 0.65, role="all"), "last": cosine_gate(log, 0.65, role="last")}
G = {k: g.replace(guard_key=g.guard_key[:0], guard_edit=g.guard_edit[:0]) for k, g in G.items()}
rows = ev.tf_rows(G, items)
E = f"{_RP.ROOT}/02_research/experiments/10_final/" + ("e3_qwen" if name.startswith("TB-Q") else "e3_llava") + "/out/rows/"
ex = {}
for lab, k in (("cosine_published", "all"), ("cosine_default", "last")):
    with gzip.open(f"{E}{name}__{lab}__dx__tf.jsonl.gz", "rt") as f:
        ex[k] = {r["sid"]: r for r in map(json.loads, f)}
a, b = rows["all"], rows["last"]
out = {"log": name, "n": len(items), "n_probes": 150, "n_edit_sequences": 150,
       "all_positions": {"n_fired": sum(r["fired"] for r in a), "n_fired_before_last_prompt": sum(bool(r["fired_before"]) for r in a), "n_fired_after_last_prompt": sum(bool(r["fired_later"]) for r in a),
                         "n_fired_last": sum(bool(r["fired_last"]) for r in a), "max_fire_positions": max(r["n_fire_pos"] for r in a)},
       "last_position": {"n_fired": sum(r["fired"] for r in b)},
       "n_outcome_differs_all_vs_last_mine": sum(r["exact"] != q["exact"] for r, q in zip(a, b)), "n_fired_differs_all_vs_last_mine": sum(r["fired"] != q["fired"] for r, q in zip(a, b)),
       "n_token_acc_differs_all_vs_last_mine": sum(abs(r["token_acc"] - q["token_acc"]) > 1e-9 for r, q in zip(a, b)),
       "vs_experiment": {k: {"n_exact_differs": sum(r["exact"] != ex[k][r["sid"]]["exact"] for r in rows[k]), "n_fired_differs": sum(bool(r["fired"]) != bool(ex[k][r["sid"]]["fired"]) for r in rows[k])} for k in rows}}
json.dump(out, open(f"{ROOT}/out/pub_{name}.json", "w"), indent=1)
print(json.dumps(out))
