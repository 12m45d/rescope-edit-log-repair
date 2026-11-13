"""Check 7: twenty numbers of each paper_numbers.json (E5, E8, E6, E9, E2 on prefix logs) recomputed from the per-row / per-draw / raw files with my own aggregation."""
from vcommon import *
import gzip
def rd(p):
    with gzip.open(p, "rt") as f: return [json.loads(l) for l in f]
def cmp(name, mine, stored, tol):
    ok = (mine == stored) if isinstance(stored, str) or isinstance(mine, str) else abs(float(mine) - float(stored)) <= tol
    return {"name": name, "mine": mine, "stored": stored, "ok": bool(ok)}
res = {}
# ------------------------------------------------------------------ E9: out/hash/*.jsonl.gz
P = jload(f"{FINAL}/e9_maintenance/out/paper_numbers.json"); out = []
W = {1: "One", 5: "Five", 10: "Ten", 20: "Twenty", 50: "Fifty"}
for name, sh in ((QDEV, "Q"), (LDEV, "L")):
    S = jload(f"{FINAL}/e9_maintenance/out/setup_{name}.json")["sets"]
    H = lambda t: {r["sid"]: r for r in rd(f"{FINAL}/e9_maintenance/out/hash/{sh}__{t}.jsonl.gz")}
    base, dep = H("base_scoped"), H("base_deployed")
    def kept(kind, m, metric, what="kept"):
        v = []
        for s in range(3):
            R = set(S[str(m)][s]); rows = H(f"{kind}_m{m}_s{s}")
            v.append(100 * np.mean([r["exact"] for r in rows.values() if r["kind"] == "edit" and r["metric"] == metric and ((r["edit"] not in R) if what == "kept" else (r["edit"] in R))]))
        return float(np.mean(v))
    def d1(kind, m): return float(np.mean([100 * np.mean([r["D1"] for r in H(f"{kind}_m{m}_s{s}").values() if r["kind"] == "probe"]) for s in range(3)]))
    def pflags(kind, m):
        ref = base if kind == "scoped" else dep
        return float(np.mean([sum(r["D1"] != ref[k]["D1"] for k, r in H(f"{kind}_m{m}_s{s}").items() if r["kind"] == "probe") for s in range(3)]))
    nfr, nviol = 0, 0
    for m in (1, 5, 10, 20, 50):
        for s in range(3):
            R = set(S[str(m)][s]); rows = H(f"scoped_m{m}_s{s}")
            for k, r in rows.items():
                if not (set(base[k]["released"]) & R):
                    nfr += 1; nviol += base[k]["logit_sha1"] != r["logit_sha1"]
    for key, v, tol in ((f"Enine{sh}ScopedMTenKeptTGen", kept("scoped", 10, "tgen"), .006), (f"Enine{sh}ScopedMFiveKeptIGen", kept("scoped", 5, "igen"), .006),
                        (f"Enine{sh}ScopedMTwentyRevokedRel", kept("scoped", 20, "rel", "rev"), .006), (f"Enine{sh}LinearMTenKeptRel", kept("linear", 10, "rel"), .006),
                        (f"Enine{sh}LinearMFiftyKeptTGen", kept("linear", 50, "tgen"), .006), (f"Enine{sh}LinearMOneDOne", d1("linear", 1), .006), (f"Enine{sh}ScopedMFiftyDOne", d1("scoped", 50), .006),
                        (f"Enine{sh}LinearMTwentyProbeFlags", pflags("linear", 20), .006), (f"Enine{sh}FrameViolations", nviol, 0), (f"Enine{sh}FrameSequences", nfr, 0)):
        out.append(cmp(key, v, P[key], tol))
res["E9"] = out
# ------------------------------------------------------------------ E6: out/rows/*.jsonl.gz
P = jload(f"{FINAL}/e6_ablations/out/paper_numbers.json"); out = []
for sh in ("Q", "L"):
    for key, nm in (("resp_lin", "RespLin"), ("values_logged", "ValuesLogged"), ("role_all", "RoleAll"), ("cal_shared", "CalShared"), ("guards_off", "GuardsOff")):
        rows = [rd(f"{FINAL}/e6_ablations/out/rows/{sh}__{key}__d{d}.jsonl.gz") for d in range(3)]
        d1 = float(np.mean([100 * np.mean([r["D1"] for r in rs if r["kind"] == "probe"]) for rs in rows]))
        tg = float(np.mean([100 * np.mean([r["exact"] for r in rs if r["kind"] == "edit" and r["metric"] == "tgen"]) for rs in rows]))
        out.append(cmp(f"Esix{sh}{nm}DOne", d1, P[f"Esix{sh}{nm}DOne"], .006)); out.append(cmp(f"Esix{sh}{nm}TGen", tg, P[f"Esix{sh}{nm}TGen"], .006))
res["E6"] = out
# ------------------------------------------------------------------ E2 on prefix logs: out/rows/*.jsonl.gz
P = jload(f"{FINAL}/e2_prefix/out/paper_numbers.json"); out = []
for par, sh in ((QDEV, "Q"), (LDEV, "L")):
    for k, kn, lin, ln in ((10, "Ten", "REEDIT_B", "ReeditB"), (50, "Fifty", "RB_UNION", "RbUnion")):
        pn = f"{par}-p{k}"
        R = lambda c: rd(f"{FINAL}/e2_prefix/out/rows/{pn}__{c}.jsonl.gz")
        D1 = lambda rs: 100 * float(np.mean([r["D1"] for r in rs if r["kind"] == "probe" and r["pool"] != "repair"]))
        REL = lambda rs: 100 * float(np.mean([r["exact"] for r in rs if r["kind"] == "edit" and r["metric"] == "rel"]))
        dep = R("deployed")
        out.append(cmp(f"Pfx{sh}{kn}DeployedDOne", D1(dep), P[f"Pfx{sh}{kn}DeployedDOne"], 1e-9)); out.append(cmp(f"Pfx{sh}{kn}DeployedRel", REL(dep), P[f"Pfx{sh}{kn}DeployedRel"], 1e-9))
        wl = [R(f"WHERE_LEAST__n10__s{s}") for s in range(3)]
        out.append(cmp(f"Pfx{sh}{kn}WLeastNTenDOne", float(np.mean([D1(x) for x in wl])), P[f"Pfx{sh}{kn}WLeastNTenDOne"], 1e-9))
        out.append(cmp(f"Pfx{sh}{kn}WLeastNTenRel", float(np.mean([REL(x) for x in wl])), P[f"Pfx{sh}{kn}WLeastNTenRel"], 1e-9))
        ll = [R(f"{lin}__n10__s{s}") for s in range(3)]
        out.append(cmp(f"Pfx{sh}{kn}{ln}NTenDOne", float(np.mean([D1(x) for x in ll])), P[f"Pfx{sh}{kn}{ln}NTenDOne"], 1e-9))
res["E2P"] = out
# ------------------------------------------------------------------ E5: per-edit lists (a), per-draw lists (c), per-complaint ranks (f)
P = jload(f"{FINAL}/e5_rescope/out/paper_numbers.json"); out = []
for lg in ("TB-Q500-memitw", "TB-L200-ae-l2_3"):
    A = jload(f"{FINAL}/e5_rescope/out/a/{lg}.json")
    er, el, mg = np.array(A["rebase"]["err_rebased_per_edit"]), np.array(A["rebase"]["err_logged_per_edit"]), np.array(A["self_separation_margin"]["per_edit"])
    for key, v in (("err_rebased_max", er.max()), ("err_rebased_median", np.median(er)), ("err_logged_max", el.max()), ("margin_min", mg.min())):
        out.append(cmp(f"logs.{lg}.rebase.{key}", float(v), P[f"logs.{lg}.rebase.{key}"], 1e-12))
for lg in (QDEV, LDEV):
    C = jload(f"{FINAL}/e5_rescope/out/c/{lg}.json")
    by = [x for x in C["cross"]["repair_to_heldout"]["by_n"] if x["n"] == 200][0]
    c = [x for x in by["cells"] if x["rule"] == "two_block" and x["alpha"] == 0.05][0]["tests"]["heldout_disjoint"]["all"]
    g, l = np.array(c["grd"]["per_draw"]), np.array(c["lit"]["per_draw"])
    pre = f"validity.{lg}.two_block.repair_to_heldout."
    for key, v, tol in (("ff_mean", g.mean(), 1e-4), ("ff_p95", np.quantile(g, .95), 1e-4), ("share_of_draws_above_alpha", 100 * (l > 5).mean(), 1e-9), ("ff_se_mean", l.std(ddof=1) / np.sqrt(len(l)), 1e-4)):
        out.append(cmp(pre + key, float(v), P[pre + key], tol))
    for o in (0, 1):
        F = jload(f"{FINAL}/e5_rescope/out/f/{lg}__s{o}.json")
        rk = [p["rank_after"] for p in F["per_complaint"]]
        v = max([i + 1 for i, r in enumerate(rk) if r <= F["d_in"] - F["T"]])
        out.append(cmp(f"capacity.{lg}.largest_n_rank_at_most_d_minus_T.order{o}", v, P[f"capacity.{lg}.largest_n_rank_at_most_d_minus_T.order{o}"], 0))
res["E5"] = out
# ------------------------------------------------------------------ E8: raw seconds of the latency passes; storage by count
P = jload(f"{FINAL}/e8_cost/out/paper_numbers.json"); out = []
for bb, lg, T, d, m in (("qwen25vl", "TB-Q100-ae-l2_3", 100, 11008, 2048), ("llava", "TB-L100-ae-l2_3", 100, 11008, 4096), ("qwen25vl", "TB-Q2000-memitw", 2000, 11008, 2048), ("llava", "TB-L1000-memitw", 1000, 11008, 4096)):
    J = jload(f"{FINAL}/e8_cost/out/latency/{bb}__{lg}.json")
    X = np.array(J["raw_seconds"]); i = J["mode_names"].index("last_fp64"); p0, p1 = X[:, 0], X[:, i]
    ratio = (p1 / p0 - 1) * 100
    for key, v, tol in (("plain_forward_ms", np.median(p0) * 1e3, 1e-9), ("overhead_end_to_end_pct", np.median(ratio), 1e-9), ("overhead_end_to_end_pct_p90", np.quantile(ratio, .9), 1e-9),
                        ("overhead_end_to_end_pct_min_based", (p1.min(0).sum() / p0.min(0).sum() - 1) * 100, 1e-9), ("storage_numbers", T * (d + m + 3), 0)):
        out.append(cmp(f"{lg}.{key}", float(v), P[f"{lg}.{key}"], tol))
res["E8"] = out
summ = {e: {"n": len(v), "n_ok": sum(x["ok"] for x in v), "not_ok": [x for x in v if not x["ok"]]} for e, v in res.items()}
jdump({"summary": summ, "numbers": res}, OUT + "/c7_numbers.json")
print(json.dumps(summ, indent=1))
