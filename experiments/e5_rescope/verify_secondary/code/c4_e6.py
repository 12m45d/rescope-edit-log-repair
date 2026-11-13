"""Check 4 (E6, calibration draw 0): rows response = LIN, values = logged, role = all positions: false firing and D1 on heldout_disjoint, Rel / T-Gen / I-Gen exact.
Mine: forwards through the library's plain Evaluator (g_dev.py e6) + own gate arithmetic on cached keys; against the experiment's row files."""
from vgate import *
import gzip, fx
from rescope import splits, split_blocks
def rd(p):
    with gzip.open(p, "rt") as f: return [json.loads(l) for l in f]
res = {}
for name, sh in ((QDEV, "Q"), (LDEV, "L")):
    if not os.path.exists(f"{OUT}/g_e6_{name}.json"): continue
    rows = rd(f"{OUT}/g_e6_{name}_rows.jsonl.gz")
    L = fx.logs.open(name)
    # own gate arithmetic for the role-last rows: draw 0 = seeded sample of 200 valid REPAIR rows; split = seeded permutation; thresholds = b_j + eta with guard caps
    rep, hd = L.pool("repair"), L.pool("heldout_disjoint")
    cal = splits.sample(rep, 200, 0); i1, i2 = split_blocks(200, (1, 1), 0)
    Rc = ncoord(L.log, keys_of(L.kc, cal, "tf")); Rh = ncoord(L.log, keys_of(L.kc, hd, "tf"))
    b = Rc[i1].max(0); eta = float(np.sort((Rc[i2] - b[None, :]).max(1))[95])
    cap, _ = guard_caps(L)
    lit, grd, ncap = fire(Rh, b, eta, cap)
    own_ff_last = 100 * float(grd.mean())
    hd_sid = [L.index[j]["lib_sid"] for j in hd]
    out = {}
    for key in ("resp_lin", "values_logged", "role_all"):
        ex = rd(f"{FINAL}/e6_ablations/out/rows/{sh}__{key}__d0.jsonl.gz")
        E = {r["sid"]: r for r in ex if r["kind"] in ("edit", "probe")}
        pr = [r for r in rows if r["kind"] == "probe"]; ed = [r for r in rows if r["kind"] == "edit"]
        mine = {"FF": 100 * float(np.mean([r[key]["fired"] for r in pr])), "D1": 100 * float(np.mean([r[key]["D1"] for r in pr])),
                **{m: 100 * float(np.mean([r[key]["exact"] for r in ed if r["metric"] == m])) for m in ("rel", "tgen", "igen")}, "n_probes": len(pr), "n_edit_sequences": len(ed)}
        epr = [r for r in ex if r["kind"] == "probe"]; eed = [r for r in ex if r["kind"] == "edit"]
        exp = {"FF": 100 * float(np.mean([r["fired"] for r in epr])), "D1": 100 * float(np.mean([r["D1"] for r in epr])),
               **{m: 100 * float(np.mean([r["exact"] for r in eed if r["metric"] == m])) for m in ("rel", "tgen", "igen")}, "n_probes": len(epr), "n_edit_sequences": len(eed)}
        if key == "role_all":
            own = 100 * float(np.mean([r["own_gate_fire_role_all"] for r in pr]))
            own_vs_hook = sum(r["own_gate_fire_role_all"] != r[key]["fired"] for r in pr)
        else:
            own = own_ff_last
            f = dict(zip(hd_sid, grd.tolist()))
            own_vs_hook = sum(f[r["sid"]] != r[key]["fired"] for r in pr)
        rowdiff = {"fired": sum(r[key]["fired"] != E[r["sid"]]["fired"] for r in rows), "exact": sum(r[key]["exact"] != E[r["sid"]]["exact"] for r in rows),
                   "D1": sum(r[key]["D1"] != E[r["sid"]]["D1"] for r in pr)}
        nonfire_d1 = sum(bool(r[key]["D1"]) for r in pr if not r[key]["fired"])
        out[key] = {"mine": mine, "experiment_rows": exp, "own_gate_FF": own, "own_gate_vs_hook_rows_differing": own_vs_hook, "rows_differing_from_experiment": rowdiff,
                    "D1_on_rows_without_release": nonfire_d1, "max_abs_diff": max(abs(mine[k] - exp[k]) for k in ("FF", "D1", "rel", "tgen", "igen"))}
    res[name] = dict(out, n_capped_last=ncap, eta_own=eta, job=jload(f"{OUT}/g_e6_{name}.json"))
    res[name]["job"].pop("cal_sids")
    for k, v in out.items(): print(name, k, {a: (round(b, 2) if isinstance(b, float) else b) for a, b in v["mine"].items()}, "| exp", {a: (round(b, 2) if isinstance(b, float) else b) for a, b in v["experiment_rows"].items()},
                                   "| ownFF %.2f diff-hook %d | rowdiff %s | D1 w/o release %d" % (v["own_gate_FF"], v["own_gate_vs_hook_rows_differing"], v["rows_differing_from_experiment"], v["D1_on_rows_without_release"]))
jdump(res, OUT + "/c4_e6.json")
