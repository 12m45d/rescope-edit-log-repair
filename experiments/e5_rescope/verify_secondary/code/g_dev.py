"""GPU job of the verification on ONE development log (library's plain Evaluator, own gate arithmetic; no driver session, no memo).
    gq run --mem 10500|17000 --prio 3 -- python g_dev.py <log> <stage> [...]
stages: cap (check 0: all-position keys of 30 complaints), e9 (check 5), e6 (check 4), lat (check 3), pre (check 6: prefixes of 10 and 50 edits)
Each stage writes out/g_<stage>_<log>.json (+ rows); a finished stage is skipped."""
from vcommon import *
import gzip
import rescope
from rescope import from_testbed, feature_gate, add_guards, calibrate_two_block, splits, KeyCache, Evaluator, EditLog, ScopedLog, Hook

INF = float("inf")
name, stages = sys.argv[1], sys.argv[2:]
bb = "qwen25vl" if name.startswith("TB-Q") else "llava"
t00 = time.time()
def say(*a): print(f"[{time.time() - t00:6.0f}s]", *a, flush=True)
def sha(t, n=20): return hashlib.sha1(np.ascontiguousarray(t.detach().float().cpu().numpy()).tobytes()).hexdigest()[:n]
def save_rows(p, rows):
    with gzip.open(p + ".tmp", "wt") as f:
        for r in rows: f.write(json.dumps(r, default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n")
    os.replace(p + ".tmp", p)

need_deltas = bool({"pre"} & set(stages))
H = rescope.load(name, device="cuda:0", mem_gb=(int(os.environ.get("GQ_MEM", 10500)) - 600) / 1024.0, keep_batch_deltas=need_deltas)
dev = torch.device(H.dev)
ev = Evaluator(H, hard_negatives=False)
kc = KeyCache(bb)
log = from_testbed(name)
uids = log.meta["uids"]
T = log.T
say("loaded", name, "T", T, H.checks)
ITEMS = {it["sid"]: it for it in ev.probe_items("all") + ev.edit_items()}

def rebased(lg):                                                    # own batch-by-batch rebase: V_i = U_i + sum_{earlier batches} U_j <W_j, k_i>
    V = lg.U.clone()
    for b in torch.unique(lg.batch, sorted=True).tolist():
        idx = (lg.batch == b).nonzero().flatten(); ear = (lg.batch < b).nonzero().flatten()
        if ear.numel(): V[idx] = lg.U[idx] + (lg.K[idx] @ lg.W[ear].T) @ lg.U[ear]
    return V

def cal_rows():                                                     # calibration draw 0: seeded sample of 200 valid REPAIR rows (library rule)
    return splits.sample(kc.log_rows(uids, kind="probe", pool="repair", valid=True), 200, 0)

def slim(G): return G.replace(guard_key=G.guard_key[:0], guard_edit=G.guard_edit[:0]).to(dev)

def default_gate(**kw):                                             # library rule on the CPU from the key cache; calibration draw 0
    S = feature_gate(log, "ncoord", **kw)
    S, _ = add_guards(S, *kc.own_keys(uids))
    return calibrate_two_block(S, coords=kc.coords(S, cal_rows()), alpha=0.05, seed=0)

Wd, sd = log.W.to(dev), (log.W * log.K).sum(1).to(dev)
def allpos(it, nlog=None):                                          # (positions, T) normalised coordinates at every position of the teacher-forced sequence; last prompt index
    x, lp = ev.capture(it, True)
    W, s = (Wd, sd) if nlog is None else (Wd[:nlog], sd[:nlog])
    return (x.double() @ W.T) / s[None, :], lp, x

def run(methods, items, extra=None, verbose=200):                   # one forward per (item, method) through the plain Evaluator, with logits
    out = []
    with torch.no_grad(), ev.hz.HE.vision_cache(H):
        for n, it in enumerate(items):
            ev.trim_vcache()
            b = ev.batch(it)
            row = {"sid": it["sid"], "kind": it["kind"], "metric": it.get("metric"), "edit": it.get("edit"), "ptype": it.get("ptype")}
            L0 = {}
            for k, m in methods.items():
                if m == "skip" or (isinstance(m, tuple) and it["kind"] not in m[1]): continue
                obj = m[0] if isinstance(m, tuple) else m
                with ev.applied(obj) as hk:
                    r, lg = ev.tf(it, b, logits=True)
                    rel = None
                    if hk is not None:
                        f = hk.last
                        rel = [] if f is None else (sorted({int(e) for e in f.edit.tolist() if int(e) >= 0}) if obj.fire == "top1" else [int(e) for e in obj._edit_any(f.cand).any(0).nonzero().flatten().tolist()])
                row[k] = {"exact": r["exact"], "D1": r.get("D1"), "fired": r.get("fired"), "own_fired_last": r.get("own_fired_last"), "released": rel, "sha": sha(lg), "finite": r["finite"]}
                L0[k] = lg
            if extra: extra(row, L0)
            out.append(row)
            if verbose and (n + 1) % verbose == 0: say("  run", n + 1, "/", len(items))
    return out

# ============================================================================================ cap
if "cap" in stages and not os.path.exists(f"{OUT}/g_cap_{name}.json"):
    E = jload(f"{FINAL}/e5_rescope/out/f/{name}__s0.json")
    per = E["per_complaint"][:30]
    Ks, rec = [], []
    with torch.no_grad(), ev.hz.HE.vision_cache(H):
        for p in per:
            ev.trim_vcache()
            it = ITEMS[p["sid"].split(":", 1)[1]]
            assert it["kind"] == "probe" and it["pool"] == "repair" and it["D1_WE"], it
            x, lp = ev.capture(it, True)
            h = hashlib.sha1(x.double().cpu().numpy().tobytes()).hexdigest()[:16]
            rec.append({"sid": p["sid"], "n_keys": int(x.shape[0]), "experiment_n_keys": p["n_keys"], "keys_sha1": h, "experiment_keys_sha1": p["keys_sha1"], "lp": lp})
            Ks.append(x.cpu())
    torch.save({"K": Ks, "sids": [p["sid"] for p in per], "dtype": str(Ks[0].dtype)}, f"{OUT}/g_cap_{name}_keys.pt")
    jdump({"log": name, "seed": 0, "n": len(per), "per": rec, "n_sha_equal": sum(r["keys_sha1"] == r["experiment_keys_sha1"] for r in rec),
           "n_nkeys_equal": sum(r["n_keys"] == r["experiment_n_keys"] for r in rec), "capture": "rescope.Evaluator.capture (module input at every position of prompt + pre-edit answer)"}, f"{OUT}/g_cap_{name}.json")
    say("cap done", sum(r["keys_sha1"] == r["experiment_keys_sha1"] for r in rec), "of", len(rec), "hashes equal")

# ============================================================================================ e9
if "e9" in stages and not os.path.exists(f"{OUT}/g_e9_{name}.json"):
    G, crec = default_gate(values=rebased(log))
    R = sorted(int(i) for i in np.random.default_rng([20261111, 10, 0]).choice(T, 10, replace=False))
    tau = G.det_tau.clone(); tau[torch.tensor(R)] = INF
    keep = ~torch.isin(G.guard_edit, torch.tensor(R))
    G2 = G.replace(det_tau=tau, guard_key=G.guard_key[keep], guard_edit=G.guard_edit[keep])
    Gd, G2d = slim(G), slim(G2)
    idx = torch.tensor(R, device=dev)
    X = (log.U.to(dev)[idx].T @ Wd[idx]).float()                    # rank-one terms of the revoked edits, fp64 -> fp32
    WR = (H.W0.float() + ((H.WE.float() - H.W0.float()) - X)).to(H.W0.dtype)
    if getattr(H, "A", None) is None:
        H.A = [None] * len(H.batches)                               # no whole batch is in R (checked next): the harness then needs no batch delta
    whole = [b for b, ids in enumerate(H.batches) if all(i in set(R) for i in ids)]
    Wh = H.rollback_weight(R) if not whole else None
    items = ev.edit_items() + ev.probe_items("heldout_disjoint")
    def extra(row, L0):
        row["same_logits_before_after"] = bool(torch.equal(L0["scoped"], L0["masked"]))
        row["same_logits_rollback_deployed"] = bool(torch.equal(L0["rollback"], L0["deployed"]))
    rows = run({"scoped": Gd, "masked": G2d, "rollback": WR, "deployed": None}, items, extra)
    save_rows(f"{OUT}/g_e9_{name}_rows.jsonl.gz", rows)
    jdump({"log": name, "R": R, "whole_batches_in_R": whole, "eta": crec["eta"], "n_edits_capped": crec["n_edits_capped"], "bound": crec["bound_false_firing"],
           "rollback_weight_equals_harness": None if Wh is None else bool(torch.equal(WR, Wh)), "rollback_weight_sha": sha(WR, 40),
           "rollback_max_abs_diff_to_harness": None if Wh is None else float((WR.float() - Wh.float()).abs().max()),
           "rollback_equals_WE": bool(torch.equal(WR, H.WE)), "rollback_equals_W0": bool(torch.equal(WR, H.W0)), "n_rows": len(rows)}, f"{OUT}/g_e9_{name}.json")
    say("e9 done")
    del Gd, G2d, WR, X; torch.cuda.empty_cache()

# ============================================================================================ e6
if "e6" in stages and not os.path.exists(f"{OUT}/g_e6_{name}.json"):
    Glin, c1 = default_gate(response="lin")
    Glog, c2 = default_gate(values="stored")
    cal = cal_rows()
    items_hd = ev.probe_items("heldout_disjoint")
    Q, Qhd = [], []
    with torch.no_grad(), ev.hz.HE.vision_cache(H):
        for j in cal:
            ev.trim_vcache(); Q.append(allpos(ITEMS[kc.index[j]["lib_sid"]])[0].max(0).values.cpu())
        for it in items_hd:
            ev.trim_vcache(); Qhd.append(allpos(it)[0].max(0).values.cpu())
    Q, Qhd = torch.stack(Q), torch.stack(Qhd)
    say("captured all-position maxima", tuple(Q.shape), tuple(Qhd.shape))
    Sall = feature_gate(log, "ncoord", role="all")
    Sall, _ = add_guards(Sall, *kc.own_keys(uids))
    Gall, c3 = calibrate_two_block(Sall, coords=Q, alpha=0.05, seed=0)
    own_fire_all = (Qhd >= Gall.det_tau[None, :]).any(1)             # own gate arithmetic: an edit is a candidate somewhere on the sequence
    items = ev.edit_items() + items_hd
    rows = run({"resp_lin": slim(Glin), "values_logged": slim(Glog), "role_all": slim(Gall)}, items)
    for r, f in zip(rows[len(rows) - len(items_hd):], own_fire_all.tolist()): r["own_gate_fire_role_all"] = bool(f)
    save_rows(f"{OUT}/g_e6_{name}_rows.jsonl.gz", rows)
    torch.save({"Q": Q, "Qhd": Qhd, "cal_rows": cal, "tau_all": Gall.det_tau, "tau_last": Glin.det_tau, "b_all": Gall.det_b, "b_last": Glin.det_b}, f"{OUT}/g_e6_{name}_gate.pt")
    jdump({"log": name, "draw": 0, "cal_sids": [kc.index[j]["sid"] for j in cal], "eta_last": c1["eta"], "n_capped_last": c1["n_edits_capped"], "eta_all": c3["eta"], "n_capped_all": c3["n_edits_capped"],
           "capped_all": c3.get("capped_edits"), "thresholds_lin_equal_logged": bool(torch.equal(Glin.det_tau, Glog.det_tau)), "n_rows": len(rows)}, f"{OUT}/g_e6_{name}.json")
    say("e6 done")
    torch.cuda.empty_cache()

# ============================================================================================ lat
if "lat" in stages and not os.path.exists(f"{OUT}/g_lat_{name}.json"):
    G, crec = default_gate(values=rebased(log))
    g = slim(G).replace(W0=H.W0)
    rng = np.random.default_rng(20261110)                           # the experiment's 30 sequences
    pr = ev.probe_items("all")
    pick = lambda xs, n: [xs[int(i)] for i in sorted(rng.choice(len(xs), n, replace=False))]
    items = pick(ev.edit_items(), 10) + pick([p for p in pr if p["ptype"] == "image"], 10) + pick([p for p in pr if p["ptype"] == "text"], 10)
    batches = [ev.batch(it) for it in items]
    REPS = 9
    Tm = np.zeros((REPS + 1, 2, len(batches)))
    def util():
        import subprocess
        try: return float(subprocess.run(["nvidia-smi", "-i", os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0], "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout.strip())
        except Exception: return None
    ut = [util()]
    with torch.no_grad(), H.at(H.W0):
        for rep in range(REPS + 1):
            for bi, b in enumerate(batches):
                for o in range(2):
                    mi = (o + rep + bi) % 2
                    hk = None
                    if mi:
                        hk = Hook(H.mod, g, set_weight=False, record=False); hk.__enter__(); hk.begin(b["last_prompt_idx"])
                    try:
                        torch.cuda.synchronize(); t0 = time.perf_counter()
                        H.vlm.forward(b, logits_to_keep="answer")
                        torch.cuda.synchronize(); Tm[rep, mi, bi] = time.perf_counter() - t0
                    finally:
                        if hk is not None: hk.__exit__(None, None, None)
            ut.append(util())
    NR = 50
    Gt = np.zeros((len(batches), NR)); nfire = 0
    with torch.no_grad():
        for bi, (it, b) in enumerate(zip(items, batches)):
            x, lp = ev.capture(it, batch=b)
            x, lpt = x[None], torch.tensor([lp], device=x.device)
            out0 = torch.nn.functional.linear(x, H.W0)
            for r in range(NR):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                f = g.firing(x, lpt, 0, {})
                if f is not None: g.respond(x, out0, f)
                torch.cuda.synchronize(); Gt[bi, r] = time.perf_counter() - t0
            nfire += f is not None
    P = Tm[1:]
    ratio = P[:, 1] / P[:, 0]
    q3 = lambda v, s=1.0: {"median": float(np.median(v) * s), "p10": float(np.quantile(v, .1) * s), "p90": float(np.quantile(v, .9) * s)}
    best = None
    for v in vars(H.vlm).values():
        if isinstance(v, torch.nn.Module):
            n = sum(p.numel() for p in v.parameters())
            if best is None or n > best: best = n
    jdump({"log": name, "mode": "last_fp64", "plain_ms": q3(P[:, 0], 1e3), "gated_ms": q3(P[:, 1], 1e3), "overhead_pct": q3((ratio - 1) * 100),
           "overhead_min_based_pct": float((P[:, 1].min(0).sum() / P[:, 0].min(0).sum() - 1) * 100), "gate_alone_ms": q3(Gt, 1e3), "gate_alone_ms_mean_of_min": float(Gt.min(1).mean() * 1e3),
           "gate_alone_pct_of_plain_forward": float(Gt.min(1).mean() / P[:, 0].min(0).mean() * 100), "n_sequences_firing": int(nfire), "utilisation_pct": ut,
           "model_parameters_largest_module": best, "module_shape": list(H.W0.shape), "module_dtype": str(H.W0.dtype), "reps": REPS, "gate_runs_per_sequence": NR}, f"{OUT}/g_lat_{name}.json")
    say("lat done")

# ============================================================================================ pre
if "pre" in stages:
    PRE = f"{FINAL}/e2_prefix/out"
    LINEAR = {10: "REEDIT_B__n10__s0", 50: "RB_UNION__n10__s0"}
    items_hd = ev.probe_items("heldout_disjoint")
    for k in (10, 50):
        pn = f"{name}-p{k}"
        if os.path.exists(f"{OUT}/g_pre_{pn}.json"): continue
        S = torch.zeros(H.W0.shape, dtype=torch.float32, device=dev)
        for b in range(k // 10): S += H.A[b].to(dev)                # stored applied batch updates, in batch order, fp32
        Wp = (H.W0.float() + S).to(H.W0.dtype)
        comp = torch.load(f"{FINAL}/common/out/prefix_logs/{pn}/composite_delta.pt", map_location="cpu").to(dev)
        eq_comp = bool(torch.equal(Wp.float() - H.W0.float(), comp.float()))
        del comp, S
        setup = jload(f"{PRE}/cells/{pn}/setup.json")
        csids = setup["streams"]["0"][:10]
        citems = [ITEMS[s.split(":", 1)[1]] for s in csids]
        assert all(it["kind"] == "probe" and it["pool"] == "repair" for it in citems)
        blob = torch.load(f"{PRE}/corr/{pn}/{LINEAR[k]}.pt", map_location="cpu")
        assert blob["kind"] == "weight"
        Wl = blob["W"].to(dev)
        # ---- thresholds-only repair rebuilt: log-only thresholds, then the least repair of the 10 complaints (all positions)
        m = torch.full((k,), -INF, dtype=torch.float64, device=dev); g1 = torch.zeros(k, dtype=torch.float64, device=dev); npos = 0
        with torch.no_grad(), ev.hz.HE.vision_cache(H):
            for i in range(k):
                ev.trim_vcache()
                r, lp, _ = allpos(ITEMS[f"edit:rel:{i}"], k)
                g1[i] = r[lp, i]
                r = r.clone(); r[lp, i] = -INF                      # its own solver position is not a negative
                m = torch.maximum(m, r.max(0).values); npos += r.shape[0]
        gst = ((log.K[:k].to(dev) @ Wd[:k].T) / sd[:k]).diagonal()  # stored solver key: coordinate 1
        gmin = torch.minimum(g1, gst)
        cap = gmin - 1e-9 * gmin.abs()
        lit = m + 0.05 * m.abs()
        tau0 = torch.minimum(lit, cap)
        n_capped0 = int((cap < lit).sum())
        tau, refused, raised = tau0.clone(), [], 0
        with torch.no_grad(), ev.hz.HE.vision_cache(H):
            for it in citems:
                ev.trim_vcache()
                r, lp, _ = allpos(it, k)
                mc = r.max(0).values
                new = torch.maximum(tau, torch.nextafter(mc + 1e-6 * mc.abs() + 1e-12, torch.full_like(mc, INF)))
                if bool((gmin < new).any()): refused.append(it["sid"]); continue          # a guard key would be silenced: refuse
                raised += int((new > tau).sum()); tau = new
        plog = EditLog(log.U[:k], log.W[:k], log.K[:k], log.batch[:k], name=pn, meta={"uids": uids[:k]})
        Gw = ScopedLog.from_log(plog, role="all", fire="all", response="lin", tau=tau.cpu()).to(dev)
        erel = ev.edit_items(("rel", "tgen", "igen"), range(k))
        rows = run({"deployed_prefix": Wp, "linear": Wl, "where_least": Gw}, erel + items_hd + citems)
        save_rows(f"{OUT}/g_pre_{pn}_rows.jsonl.gz", rows)
        st = jload(f"{PRE}/cells/{pn}/eval__WHERE_LEAST__n10__s0.json")
        th = np.array(st["theta"], dtype=np.float64)
        jdump({"log": pn, "k": k, "prefix_weight_equals_stored_composite": eq_comp, "linear": LINEAR[k], "complaint_sids": csids, "theta": tau.cpu().tolist(), "theta_logonly": tau0.cpu().tolist(),
               "n_capped_logonly": n_capped0, "n_refused": len(refused), "n_thresholds_raised": raised, "n_positions_negatives": npos,
               "experiment_theta": st["theta"], "theta_max_abs_diff": float(np.abs(tau.cpu().numpy() - th).max()), "theta_max_rel_diff": float((np.abs(tau.cpu().numpy() - th) / np.abs(th)).max()),
               "n_rows": len(rows)}, f"{OUT}/g_pre_{pn}.json")
        say("pre done", pn, "theta max abs diff", float(np.abs(tau.cpu().numpy() - th).max()))
        del Wp, Wl, Gw; torch.cuda.empty_cache()
say("ALL DONE", stages)
