"""CPU tests on the REAL recombination store.  A gate is run on the two development logs only; every other log is touched through its
edit uids alone (which rows it may use), and the scale test uses synthetic detectors.

    python tests/t_real.py [qwen25vl] [llava]            -> tests/out/t_real.json        (default: every backbone whose store exists)

r1  the store: rows and keys belong together (md5 of keys.pt, sha256 of rows.jsonl), 4 rows of every kind per edit, finite keys, the
    capture path was verified bit for bit, the re-capture check passed, the selection check passed
r2  logs: every registered VLKEB log of the backbone is served with the row counts of the select summary; logs of another stream /
    order / benchmark are refused
r3  the baseline on the development log: default feature = direct formula <W_i, k> / <W_i, k_i>; the driver's base gate (with guards),
    the feature name and a callable give the same bits; deterministic
r4  WITHOUT recombinations the wiring is the driver's default method bit for bit (fx.methods.rescope; cosine gate: fx.methods.cosine),
    draws 0 - 2, alpha 0.05 and 0.1: identical thresholds, baselines, eta, capped edits
r5  WITH recombinations on the development log: det_b = max(block-1 baseline, x) bit for bit; uncapped thresholds = the plain-loop
    reference; every guard fires; block 2 has exactly n2 - rank queries with a candidate when no edit is capped; fx.gate decides as the
    library's KeyCache.decide; the record is JSON-serialisable
r6  <gate>_x for the raw cosine gate: x = the largest cosine of the edit's stored key to its own recombination keys
r7  scale (label-free): own_baseline for T = the whole stream with synthetic detectors on the real keys: seconds, peak memory
r8  the score functions of the sibling package `white` (un-thresholded gate and score-matrix builder) give the same baseline; skipped
    with the reason if that package is not usable
"""
from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tcommon as tc  # noqa: E402
from tcommon import check, np, torch  # noqa: E402

import rescope  # noqa: E402
import recomb  # noqa: E402
from recomb import wiring  # noqa: E402

fx = wiring.use_fx()
NEG = float("-inf")
DATA = recomb.data_root()


def stream_uids(name):
    with open(os.path.join(fx.TESTBED, name, "edit_batch.jsonl")) as f:
        return [json.loads(line)["uid"] for line in f if line.strip()]


def r1_store(bb):
    st = recomb.Store(bb, verify=True)                                  # md5 of keys.pt against capture_meta.json
    summ = json.load(open(os.path.join(st.dir, "select_summary.json")))
    ck = json.load(open(os.path.join(st.dir, "check_select.json")))
    check(st.n == 12 * st.T_stream == summ["n_rows"] and st.T_stream == summ["T"] and st.d == 11008, "sizes")
    check(st.rows_sha256 == summ["rows_file"]["sha256"] == st.meta["rows_sha256"] == ck["rows_sha256"], "rows.jsonl is not the checked selection")
    check(ck["ok"] and all(c["ok"] for c in ck["checks"].values()), "the selection check did not pass")
    check(st.meta["complete"] and st.meta["checks_ok"] and st.meta["verify"]["ok"] and st.meta["recheck"]["ok"], "capture meta")
    v = st.meta["verify"]["summary"]
    check(all(v[k]["prompt_key_bitwise"] == [50, 50] and v[k]["tf_key_bitwise"] == [50, 50] for k in ("hard_negatives", "probes")), "bitwise verification")
    bad = 0
    for a in range(0, st.n, 4000):
        bad += int((~torch.isfinite(st.K[a:a + 4000])).sum())
    check(bad == 0 and int(st.lp.min()) > 0, "non-finite key")
    check(st.layer == {"qwen25vl": 20, "llava": 16}[bb] and st.family == {"qwen25vl": "Q", "llava": "L"}[bb])
    kc = fx.logs.keycache(bb)                                           # the owner's image / question is the cached edit's
    rel = {int(e["edit"]): e for e in kc.index if e["kind"] == "rel" and e["family"] == st.family}
    for r in st.rows[::97]:
        e = rel[r["edit_pos"]]
        check(e["edit_uid"] == r["edit_uid"] and (r["image"] == e["image"] if r["kind"] != "X-Q" else r["prompt"] == e["prompt"]), r["rid"])
    return dict(st.summary(), recheck=st.meta["recheck"]["n_bitwise"], verify_rows=st.meta["verify"]["n_rows"])


def r2_logs(bb):
    st = recomb.store(bb)
    summ = json.load(open(os.path.join(st.dir, "select_summary.json")))["rows_usable_by_log_length_T"]
    served, refused = {}, {}
    for name in fx.logs.names():
        e = fx.logs.REGISTRY[name]
        if e["backbone"] != bb or not fx.logs.exists(name):
            continue
        uids = stream_uids(name)
        try:
            R = recomb.log_rows((bb, uids))
        except LookupError as ex:
            refused[name] = str(ex)[:60]
            check(e["family"] != st.family or uids != st.stream[:len(uids)], f"{name}: a prefix of the stream was refused")
            continue
        check(e["family"] == st.family and uids == st.stream[:len(uids)], f"{name}: served although it is not a prefix of the stream")
        c = R.counts()
        check(str(len(uids)) in summ and {k: summ[str(len(uids))][k] for k in recomb.KINDS} == c, f"{name}: {c}")
        check(c == {k: 4 * len(uids) for k in recomb.KINDS} and bool((np.maximum(R.edit, R.partner) < len(uids)).all()), f"{name}: {c}")
        served[name] = len(R)
    check(len(served) >= 2, "no log served")
    for name in ("TB-Q500-ae", "TB-Q100-mmke-l2_3", "TB-Q100-onehop-l2_3"):   # other order / other benchmark / other split: never served
        p = os.path.join(fx.TESTBED, name, "edit_batch.jsonl")
        if bb == "qwen25vl" and os.path.exists(p):
            try:
                recomb.log_rows((bb, stream_uids(name)))
                check(False, f"{name} was served")
            except LookupError:
                refused.setdefault(name, "refused")
    for T in (10, 20, 50):                                              # the prefix logs of the development stream
        c = recomb.log_rows((bb, st.stream[:T])).counts()
        check({k: summ[str(T)][k] for k in recomb.KINDS} == c and c["X-I"] == c["X-Q"] == 4 * T and c["X-X"] >= 2 * T, f"prefix {T}: {c}")
    return {"served": served, "refused": sorted(refused)}


def r3_baseline(bb):
    L = fx.logs.open(tc.DEV[bb])
    R = recomb.log_rows(L)
    check(len(R) == 1200 and R.counts() == {k: 400 for k in recomb.KINDS})
    K = R.keys().double()
    own = np.array([float((L.log.W[int(e)] * K[i]).sum() / (L.log.W[int(e)] * L.log.K[int(e)]).sum()) for i, e in enumerate(R.edit)])
    x_dir = np.full(L.T, NEG)
    np.maximum.at(x_dir, R.edit, own)
    o = recomb.own_scores(L)
    dev = float(np.abs(o.value - own).max())
    check(dev < 1e-11 and float(np.abs(o.x.numpy() - x_dir).max()) < 1e-11, f"default feature deviates from the direct formula by {dev:.2e}")
    base, _ = fx.methods.base_gate(L, "ncoord")
    for name, sc in (("driver base gate (with guards)", base), ("feature name", "ncoord"), ("callable", lambda k: base.scores(k)),
                     ("EditLog", None)):
        v = recomb.own_scores(L if name != "EditLog" else L.log, sc)
        check(torch.equal(v.x, o.x) and np.array_equal(v.value, o.value), f"{name}: other bits")
    check(torch.equal(recomb.own_scores(tc.DEV[bb]).x, o.x), "log name")
    x2, rec = recomb.own_baseline(L, base)
    check(torch.equal(x2, o.x) and rec["n_rows"] == 1200 and rec["n_detectors_without_rows"] == 0 and rec["keys_md5"] == recomb.store(bb).meta["keys_md5"])
    json.dumps(rec)
    per = {k: o.per_edit(k) for k in recomb.KINDS}
    check(np.array_equal(np.maximum.reduce(list(per.values())), o.x.numpy()))
    # the same bits in a fresh process under other BLAS thread settings (MKL's products depend on the thread count; own_scores uses one)
    code = ("import sys; sys.path[:0] = [%r, %r]; import rescope, recomb; "
            "print(recomb.own_baseline(rescope.from_testbed(%r))[1]['x_sha1'])" % (tc.LIB, tc.FEAT2, tc.DEV[bb]))
    shas = {}
    for th in ("1", "3", "8"):
        env = dict(os.environ, OMP_NUM_THREADS=th, MKL_NUM_THREADS=th, OPENBLAS_NUM_THREADS=th, PYTHONDONTWRITEBYTECODE="1")
        p = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
        check(p.returncode == 0, p.stderr[-400:])
        shas[th] = p.stdout.strip().splitlines()[-1]
    check(set(shas.values()) == {rec["x_sha1"]}, f"x differs between thread settings: {shas} vs {rec['x_sha1']}")
    return {"n_rows": len(R), "max_abs_dev_from_direct_formula": dev, "x_min_median_max": rec["x_min_median_max"], "x_sha1": rec["x_sha1"],
            "x_sha1_in_fresh_processes_with_1_3_8_threads": shas}


def same(A, B):
    return torch.equal(A.det_tau, B.det_tau) and torch.equal(A.det_b, B.det_b) and A.eta == B.eta and A.det_u is None and B.det_u is None


def r4_without(bb):
    L = fx.logs.open(tc.DEV[bb])
    n = 0
    for draw in (0, 1, 2):
        for alpha in (0.05, 0.1):
            A = fx.methods.rescope(L, alpha=alpha, n=200, draw=draw)
            B = wiring.method(L, alpha, 200, draw, x=False)
            check(same(A.obj, B.obj), f"rescope: draw {draw}, alpha {alpha}")
            check(A.rec["cal"]["capped_edits"] == B.rec["cal"]["capped_edits"] and A.rec["cal"]["eta"] == B.rec["cal"]["eta"]
                  and A.rec["cal"]["block2"] == B.rec["cal"]["block2"] and A.rec["cal"]["block1"] == B.rec["cal"]["block1"], "records")
            check(A.rec["own_prompts"] == B.rec["own_prompts"] and A.rec["thresholds"] == B.rec["thresholds"], "driver summaries")
            C = fx.methods.cosine(L, "two_block", alpha=alpha, n=200, draw=draw)
            D = wiring.method(L, alpha, 200, draw, feature="cos", x=False)
            check(same(C.obj, D.obj), f"cosine: draw {draw}, alpha {alpha}")
            n += 2
    return {"n_methods_bitwise_identical_to_the_driver": n}


def r5_with(bb):
    L = fx.logs.open(tc.DEV[bb])
    out = {}
    for draw in (0, 1):
        base, _ = fx.methods.base_gate(L, "ncoord")
        dr = fx.draws.calibration(L, 200, draw)
        Q = fx.methods.calib_coords(L, base, dr)
        x = recomb.own_baseline(L, base)[0]
        M = wiring.method(L, 0.05, 200, draw)
        S, c = M.obj, M.rec["cal"]
        i1, i2 = np.array(c["block1"]), np.array(c["block2"])
        check(np.array_equal(i1, rescope.split_blocks(200, (1, 1), draw)[0]) and c["n"] == 200 and c["n1"] == 100 and c["seed"] == draw, "blocks")
        b1 = Q.numpy()[i1].max(0)
        check(np.array_equal(S.det_b.numpy(), np.maximum(b1, x.numpy())), "det_b != max(block-1 baseline, x)")
        b, eta, tau = tc.naive_rule(Q.numpy(), x.numpy(), 1, 20, i1, i2)
        capped = np.zeros(L.T, dtype=bool)
        capped[c["capped_edits"]] = True
        t = S.det_tau.numpy()
        check(eta == c["eta"] and np.array_equal(t[~capped], tau[~capped]) and bool((t[capped] < tau[capped]).all()), "thresholds != plain-loop reference")
        check(bool(S.guard_ok().all()) and M.rec["own_prompts"]["n_not_candidate"] == 0, "a guard does not fire")
        cand2 = (Q[torch.from_numpy(i2)] >= torch.from_numpy(tau)[None, :]).any(1).numpy()
        check(int(cand2.sum()) == c["n2"] - c["rank"], f"{int(cand2.sum())} block-2 queries above the literal thresholds")
        check(c["recomb"]["n_baselines_raised_by_x"] == int((x.numpy() > b1).sum()) and c["recomb"]["eta_with_x"] == eta)
        rows = L.pool("heldout_disjoint") + L.rows("hn", valid=True)[:300] + L.rows("tgen")
        a, lib = fx.gate.decide(L, M, rows, "tf"), fx.gate.library_decide(L, M, rows, "tf")
        check(all(np.array_equal(a[k], lib[k]) for k in ("fired", "top_edit", "own_cand", "own_top")), "fx.gate != KeyCache.decide")
        json.dumps(fx.io.clean(M.rec))
        d = recomb.decide_rows(S, L)
        if not capped.any() and eta >= 0:                              # r_i <= x_i <= b_i and eta >= 0: an uncapped edit cannot fire on its own rows
            check(not d["own_cand"].any(), "an edit is a candidate on its own recombination row although eta >= 0")
        out[f"draw{draw}"] = {"eta": eta, "n_capped": int(capped.sum()), "n_baselines_raised": c["recomb"]["n_baselines_raised_by_x"],
                              "own_candidates_on_own_rows": int(d["own_cand"].sum()), "key": M.key}
    return out


def r6_cosine_x(bb):
    L = fx.logs.open(tc.DEV[bb])
    base, _ = fx.methods.base_gate(L, "cos")
    R = recomb.log_rows(L)
    K = R.keys().double()
    kn = L.log.K / L.log.K.norm(dim=1, keepdim=True)
    cos = np.array([float((kn[int(e)] * K[i]).sum() / K[i].norm()) for i, e in enumerate(R.edit)])
    x_dir = np.full(L.T, NEG)
    np.maximum.at(x_dir, R.edit, cos)
    x, rec = recomb.own_baseline(L, base)
    check(float(np.abs(x.numpy() - x_dir).max()) < 1e-12 and torch.equal(x, recomb.own_baseline(L, "cos")[0]), "cosine baseline")
    M = wiring.method(L, 0.05, 200, 0, feature="cos")
    dr = fx.draws.calibration(L, 200, 0)
    b1 = fx.methods.calib_coords(L, base, dr).numpy()[np.array(M.rec["cal"]["block1"])].max(0)
    check(np.array_equal(M.obj.det_b.numpy(), np.maximum(b1, x.numpy())) and bool(M.obj.guard_ok().all()) and M.name == "cos_x")
    return {"x_min_median_max": rec["x_min_median_max"], "n_baselines_raised": M.rec["cal"]["recomb"]["n_baselines_raised_by_x"],
            "n_capped": M.rec["cal"]["n_edits_capped"]}


def r8_other_packages(bb):
    """The score functions of the sibling feature package `white` (cosine family of R3) as input of own_baseline: its un-thresholded gate
    and its score-matrix builder give the same baseline.  The package is developed by another agent: if it cannot be imported or has no
    statistics yet, the test is recorded as skipped (ok), with the reason."""
    L = fx.logs.open(tc.DEV[bb])
    try:
        import white
        st = white.wstats.load(bb)
        gates = {"cos_centred": (white.scores.feature_gate(L.log, "cos_centred", stats=st), lambda K: white.scores.scores(L.log, K, "cos_centred", stats=st)),
                 "whitecos_0.1": (white.scores.feature_gate(L.log, "whitecos", stats=st, kappa=0.1),
                                  lambda K: white.scores.scores(L.log, K, "whitecos", stats=st, kappa=0.1)),
                 "white_0.1": (white.scores.feature_gate(L.log, "white", stats=st, kappa=0.1), lambda K: white.scores.scores(L.log, K, "white", stats=st, kappa=0.1))}
    except Exception as e:                                              # not our code: skipped, with the reason
        return {"skipped": f"package white not usable here: {type(e).__name__}: {str(e)[:200]}"}
    out = {}
    R = recomb.log_rows(L)
    for name, (S, fn) in gates.items():
        xg, rec = recomb.own_baseline(L, S)
        xc, _ = recomb.own_baseline(L, fn)
        dev = float((xg - xc).abs().max())
        check(dev < 1e-9, f"{name}: gate and matrix builder give baselines that differ by {dev:.2e}")
        full = S.scores(R.keys())
        ref = torch.full((L.T,), NEG, dtype=torch.float64).scatter_reduce(0, torch.from_numpy(R.edit), full[torch.arange(len(R)), torch.from_numpy(R.edit)], reduce="amax")
        check(float((xg - ref).abs().max()) < 1e-9 and bool(torch.isfinite(xg).all()), f"{name}: baseline != maximum of the own scores")
        Q = S.query_max([k[None] for k in L.kc.keys("tf", fx.draws.calibration(L, 200, 0).rows)])
        A, _ = rescope.calibrate_two_block(S, coords=Q, alpha=0.05, seed=0)
        B, _ = recomb.calibrate_two_block_x(S, coords=Q, x=None, alpha=0.05, seed=0)
        C, rc = recomb.calibrate_two_block_x(S, coords=Q, x=xg, alpha=0.05, seed=0)
        check(same(A, B) and type(C) is type(S), f"{name}: without recombinations != library, or the gate class was lost")
        out[name] = {"x_min_median_max": rec["x_min_median_max"], "max_dev_gate_vs_builder": dev, "n_baselines_raised": rc["recomb"]["n_baselines_raised_by_x"],
                     "gate_class": type(S).__name__}
    return out


def r7_scale(bb):
    st = recomb.store(bb)
    T, d = st.T_stream, st.d
    g = torch.Generator().manual_seed(0)
    Kk = torch.randn(T, d, generator=g, dtype=torch.float64)
    W = Kk / (Kk * Kk).sum(1, keepdim=True) + 1e-4 * torch.randn(T, d, generator=g, dtype=torch.float64)
    log = rescope.EditLog(torch.zeros(T, 4, dtype=torch.float64), W, Kk, torch.arange(T) // 10, name="synthetic-scale",
                          meta={"uids": list(st.stream), "backbone": bb})
    S = rescope.feature_gate(log, "ncoord")
    t0 = time.time()
    x, rec = recomb.own_baseline(log, S)
    dt = time.time() - t0
    check(rec["n_rows"] == 12 * T and rec["n_detectors_without_rows"] == 0 and bool(torch.isfinite(x).all()))
    Q = torch.randn(200, T, generator=g, dtype=torch.float64)
    t1 = time.time()
    S2, c = recomb.calibrate_two_block_x(S, coords=Q, x=x, alpha=0.05, seed=0)
    return {"T": T, "n_rows": rec["n_rows"], "own_baseline_seconds": round(dt, 1), "calibrate_seconds": round(time.time() - t1, 1),
            "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 2), "threads": torch.get_num_threads()}


if __name__ == "__main__":
    want = [a for a in sys.argv[1:] if a in recomb.BACKBONES] or [b for b in recomb.BACKBONES if recomb.available(b)]
    R = tc.Results("t_real")
    for bb in want:
        for name, fn in (("r1_store", r1_store), ("r2_logs_served", r2_logs), ("r3_baseline_dev", r3_baseline), ("r4_without_equals_driver", r4_without),
                         ("r5_with_dev", r5_with), ("r6_cosine_x_dev", r6_cosine_x), ("r7_scale_synthetic_detectors", r7_scale),
                         ("r8_score_functions_of_package_white", r8_other_packages)):
            R.run(f"{name}[{bb}]", fn, bb)
    missing = [b for b in recomb.BACKBONES if b not in want]
    if missing:
        print(f"  (no store yet for {missing}: not tested)")
    sys.exit(0 if R.save() else 1)
