"""Tests of white.wstats: statistics of the BEN-W keys and the Woodbury form of M = (C_W + rho I)^-1."""
from __future__ import annotations

import os
import shutil

import numpy as np

import naive
import tcommon as T
from tcommon import C, test
from white import benparts as BP
from white import wstats as WSt

torch = C.use_library()

TOL_SMALL = 1e-10            # small random problems against numpy.linalg.inv: cond(C + rho I) <= 1e4, so round-off is far below this
TOL_REAL = 1e-8              # full-size identity M (C_W + rho I) x = x: cond(M) = 6e5 at kappa = 0.01 (meta.json), round-off about 1e-10


def _rand(n, d, seed):
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(d, generator=g, dtype=torch.float64)
    return 1.5 * base[None, :] + torch.randn(n, d, generator=g, dtype=torch.float64)       # a common direction + noise, like real keys


def _rel(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.abs(a - b).max() / np.abs(b).max())


@test("stats", "Woodbury (Gram form, N < d) and the primal form (N > d) against an explicit inverse numpy.linalg.inv(C + rho I) on small random problems")
def s1_woodbury_small():
    res, worst = {}, 0.0
    for nm, (N, d) in {"gram N=30 d=50": (30, 50), "gram N=d=40": (40, 40), "primal N=80 d=50": (80, 50), "gram N=1 d=12": (1, 12)}.items():
        K = _rand(N, d, seed=N + d)
        st = WSt.KeyStats.from_keys(K)
        X, Y = _rand(7, d, 1), _rand(5, d, 2)
        e = {}
        for kappa in (0.01, 0.1, 1.0, 37.0):
            wh = st.whitener(kappa=kappa)
            Cn = K.numpy().T @ K.numpy() / N
            rho = kappa * np.trace(Cn) / d
            Mn = np.linalg.inv(Cn + rho * np.eye(d))                      # the explicit inverse
            e[str(kappa)] = {"form": wh.form, "rho": abs(wh.rho - rho) / rho, "dense": _rel(wh.dense().numpy(), Mn),
                             "apply": _rel(wh.apply(X).numpy(), X.numpy() @ Mn), "quad": _rel(wh.quad(X).numpy(), np.einsum("nd,de,ne->n", X.numpy(), Mn, X.numpy())),
                             "bilinear": _rel(wh.bilinear(X, Y).numpy(), X.numpy() @ Mn @ Y.numpy().T),
                             "naive_inverse": _rel(naive.explicit_inverse(K.numpy(), rho), Mn),
                             "cond": float(np.linalg.cond(Cn + rho * np.eye(d)))}
            worst = max(worst, *(e[str(kappa)][k] for k in ("rho", "dense", "apply", "quad", "bilinear")))
        want = "gram" if N <= d else "primal"
        e["form_ok"] = all(v["form"] == want for v in e.values())
        res[nm] = e
    ok = worst <= TOL_SMALL and all(v["form_ok"] for v in res.values())
    return {"pass": ok, "problems": res, "tolerance": TOL_SMALL, "worst_relative_deviation": worst,
            "note": f"4 problems x 4 ridges, rho / dense M / x M / x M x' / x M y': worst relative deviation {worst:.1e} (tolerance {TOL_SMALL:.0e})"}


@test("stats", "the same second moment from other key sets gives the same statistics: keys repeated (Gram form and primal form), keys permuted")
def s2_same_moment_other_keys():
    d, N = 48, 20
    K = _rand(N, d, 3)
    st = WSt.KeyStats.from_keys(K)
    st2 = WSt.KeyStats.from_keys(torch.cat([K, K]))                      # N = 40 <= d: Gram form with a singular Gram matrix
    st3 = WSt.KeyStats.from_keys(torch.cat([K, K, K]))                   # N = 60 > d: primal form
    stp = WSt.KeyStats.from_keys(K[torch.randperm(N, generator=torch.Generator().manual_seed(0))])
    dev = {}
    for nm, s in (("repeated twice (gram)", st2), ("repeated three times (primal)", st3), ("permuted", stp)):
        dev[nm] = {"mu": _rel(s.mu.numpy(), st.mu.numpy()), "trace": abs(s.trace - st.trace) / st.trace,
                   "M": max(_rel(s.whitener(kappa=k).dense().numpy(), st.whitener(kappa=k).dense().numpy()) for k in (0.01, 1.0)),
                   "form": s.whitener(kappa=1.0).form}
    worst = max(max(v["mu"], v["trace"], v["M"]) for v in dev.values())
    ok = worst <= TOL_SMALL and dev["repeated twice (gram)"]["form"] == "gram" and dev["repeated three times (primal)"]["form"] == "primal"
    return {"pass": ok, "deviations": dev, "note": f"mu, tr(C), M of repeated / permuted reference keys: worst relative deviation {worst:.1e}; Gram form == primal form"}


@test("stats", "the stored statistics against a fresh numpy computation from the key cache: rows, keys (bitwise), mu, Gram matrix, tr(C_W), rho, L^-1")
def s3_stored_vs_numpy():
    from rescope import KeyCache
    out, ok = {}, True
    for bb in C.BACKBONES:
        st = WSt.load(bb)
        kc = KeyCache(bb, verify=True)
        P = BP.load(bb)
        sids = P.sids("BEN-W")
        rows = [kc.by_sid(s) for s in sids]
        Kc = torch.stack([kc.K("decode")[int(kc.index[j]["row_decode"])] for j in rows])        # the cache's decode key of every BEN-W row
        Kn = Kc.numpy().astype(np.float64)
        N, d = Kn.shape
        Gn = Kn @ Kn.T
        tr = float(np.trace(Gn)) / N
        e = {"n": N, "d": d, "rows_are_BEN-W": st.sids == sids and all(kc.index[j]["kind"] == "benign" and P.part(kc.index[j]["sid"]) == "BEN-W" for j in rows),
             "keys_bitwise_equal": bool(torch.equal(Kc, st.keys)), "keys_dtype": str(st.keys.dtype),
             "mu": _rel(st.mu.numpy(), Kn.mean(0)), "gram": _rel(st.gram.numpy(), Gn), "trace": abs(st.trace - tr) / tr, "kappa": {}}
        # information (no criterion): the statistics use the PROMPT-ONLY key; how far is the teacher-forced key of the same row (answer subset)?
        tf = [(i, int(kc.index[j]["row_tf"])) for i, j in enumerate(rows) if kc.index[j].get("row_tf") is not None]
        if tf:
            Kt = kc.K("tf")[torch.tensor([r for _, r in tf])].double()
            Kp = Kc[torch.tensor([i for i, _ in tf])].double()
            rel = ((Kt - Kp).norm(dim=1) / Kp.norm(dim=1)).numpy()
            shift = [bool(kc.index[rows[i]].get("lp_shift")) for i, _ in tf]
            e["teacher_forced_vs_prompt_only_key"] = {"n_rows_with_teacher_forced_key": len(tf), "n_bitwise_equal": int((rel == 0).sum()),
                                                      "n_other_token (lp_shift)": int(sum(shift)),
                                                      "relative_distance_median_max_same_token": [float(np.median(rel[~np.array(shift)])), float(rel[~np.array(shift)].max())],
                                                      "relative_distance_max_other_token": float(rel[np.array(shift)].max()) if any(shift) else None}
        for kappa in C.KAPPAS:
            t = WSt.kappa_tag(kappa)
            rho = kappa * tr / d
            linv = st._linv[t].numpy()
            A = Gn + N * rho * np.eye(N)
            e["kappa"][t] = {"rho": abs(st.rho(kappa) - rho) / rho, "rho_meta": abs(st.meta["rho"][t] - rho) / rho,
                             "linv_A_linvT_minus_I": float(np.abs(linv @ A @ linv.T - np.eye(N)).max()),
                             "linv_lower_triangular": bool(np.abs(np.triu(linv, 1)).max() == 0.0)}
        worst = max([e["mu"], e["gram"], e["trace"]] + [v[k] for v in e["kappa"].values() for k in ("rho", "rho_meta")])
        lin = max(v["linv_A_linvT_minus_I"] for v in e["kappa"].values())
        good = (e["rows_are_BEN-W"] and e["keys_bitwise_equal"] and N == 600 and d == 11008 and worst <= 1e-12 and lin <= 1e-9
                and all(v["linv_lower_triangular"] for v in e["kappa"].values()) and N == P.n_rows["BEN-W"])
        ok &= good
        out[bb] = dict(e, worst_relative_deviation=worst, worst_linv_residual=lin)
    return {"pass": ok, "per_backbone": out,
            "note": "; ".join(f"{bb}: N = {v['n']}, keys bitwise {v['keys_bitwise_equal']}, mu / Gram / tr / rho rel. dev. {v['worst_relative_deviation']:.1e}, "
                              f"|L^-1 A L^-T - I| {v['worst_linv_residual']:.1e}" for bb, v in out.items())}


@test("stats", "determinism: fresh processes with 1, 2 and 4 BLAS threads reproduce stats.pt and meta.json byte for byte")
def s4_determinism_files():
    res, ok = {}, True
    for nt in (1, 2, 4):
        rc, log = T.child("white.wstats", ["check"], env={"OMP_NUM_THREADS": nt, "MKL_NUM_THREADS": nt, "OPENBLAS_NUM_THREADS": nt})
        lines = [ln for ln in log.splitlines() if "reproduced" in ln or "DIFFERS" in ln]
        good = rc == 0 and len(lines) == len(C.BACKBONES) and all("DIFFERS" not in ln and f"threads {nt}" in ln for ln in lines)
        ok &= good
        res[f"{nt} threads"] = {"exit": rc, "lines": lines}
    return {"pass": ok, "runs": res, "note": f"stats.pt + meta.json of {len(C.BACKBONES)} backbones rebuilt under 1 / 2 / 4 threads: byte-identical: {ok}"}


@test("stats", "the reader refuses a changed stats.pt (md5) and statistics whose keys are not the BEN-W keys of the key cache")
def s5_loader_detects_tampering():
    bb = "llava"
    d = T.tmpdir("whiten")
    shutil.copytree(WSt.out_dir(bb), os.path.join(d, bb))
    p = os.path.join(d, bb, "stats.pt")
    ok_copy = WSt.load(bb, True, d).n == 600
    WSt._LOADED.pop((bb, d), None)
    with open(p, "r+b") as f:
        f.seek(os.path.getsize(p) // 2)
        b = f.read(1)
        f.seek(-1, 1)
        f.write(bytes([b[0] ^ 0x01]))                                    # one flipped bit in the middle of the file
    err1 = None
    try:
        WSt.load(bb, True, d)
    except RuntimeError as e:
        err1 = str(e)[:100]
    WSt._LOADED.pop((bb, d), None)
    T.rmtmp("whiten")
    orig, err2 = WSt.benw_keys, None
    try:                                                                  # the key cache now holds other keys for the BEN-W rows
        def other(backbone, verify_cache=False):
            sids, kr, keys, kc, P = orig(backbone, verify_cache)
            keys = keys.clone()
            keys[17, 5] += 1e-3
            return sids, kr, keys, kc, P
        WSt.benw_keys = other
        WSt._LOADED.pop((bb, None), None)
        try:
            WSt.load(bb)
        except RuntimeError as e:
            err2 = str(e)[:100]
    finally:
        WSt.benw_keys = orig
        WSt._LOADED.pop((bb, None), None)
    return {"pass": ok_copy and err1 is not None and err2 is not None, "flipped_bit": err1, "other_keys": err2,
            "note": f"copy loads: {ok_copy}; flipped bit refused: {err1 is not None}; other cache keys refused: {err2 is not None}"}


@test("stats", "full size (d = 11008): M (C_W + rho I) x = x for BEN-W keys, solver keys of the development logs and random vectors, per kappa")
def s6_whitener_identity_full_size():
    out, worst = {}, 0.0
    for bb in C.BACKBONES:
        st = WSt.load(bb)
        Kd = st.keys.double()
        g = torch.Generator().manual_seed(T.SEED)
        log = T.context(C.DEV_LOGS[bb]).log
        X = torch.cat([Kd[:25], log.K[:25], torch.randn(25, st.d, generator=g, dtype=torch.float64)])
        e = {}
        for kappa in C.KAPPAS:
            wh = st.whitener(kappa=kappa)
            Y = wh.apply(X)                                               # rows x M
            Z = ((Y @ Kd.T) @ Kd) / st.n + wh.rho * Y                     # rows x M (C_W + rho I), C_W applied through the keys (definition)
            rel = ((Z - X).norm(dim=1) / X.norm(dim=1))
            q = wh.quad(X)
            q2 = (Y * X).sum(1)                                           # x M x' through x M
            sym = wh.bilinear(X, X)
            e[WSt.kappa_tag(kappa)] = {"residual_ben_w_keys": float(rel[:25].max()), "residual_solver_keys": float(rel[25:50].max()),
                                       "residual_random": float(rel[50:].max()), "quad_vs_apply": float(((q - q2).abs() / q).max()),
                                       "bilinear_symmetry": float((sym - sym.T).abs().max() / sym.abs().max()), "quad_positive": bool((q > 0).all()),
                                       "cancellation_ben_w_keys": float(((Kd[:25] ** 2).sum(1) / (wh.rho * q[:25])).median()),
                                       "cancellation_solver_keys": float(((log.K[:25] ** 2).sum(1) / (wh.rho * q[25:50])).median())}
            worst = max(worst, float(rel.max()), e[WSt.kappa_tag(kappa)]["quad_vs_apply"])
        out[bb] = e
    return {"pass": worst <= TOL_REAL and all(v["quad_positive"] for e in out.values() for v in e.values()), "per_backbone": out, "tolerance": TOL_REAL,
            "worst_relative_residual": worst,
            "note": f"|x M (C_W + rho I) - x| / |x| and x M x' two ways, 75 vectors x 3 ridges x 2 backbones: worst {worst:.1e} (tolerance {TOL_REAL:.0e}); "
                    f"median cancellation |x|^2 / (rho x M x') at kappa 0.01: BEN-W keys {out['qwen25vl']['0.01']['cancellation_ben_w_keys']:.0f}, "
                    f"solver keys {out['qwen25vl']['0.01']['cancellation_solver_keys']:.0f} (Qwen)"}


@test("stats", "the whitening factor from the stored L^-1 equals a fresh factorisation bit for bit, under 1 and 4 threads")
def s7_factor_bitwise():
    st = WSt.load("qwen25vl")
    res = {}
    n0 = torch.get_num_threads()
    try:
        for nt in (1, 4):
            torch.set_num_threads(nt)
            fresh = WSt.KeyStats.from_keys(st.keys)                       # no stored factors: Gram matrix and Cholesky are recomputed
            stored = WSt.KeyStats(st.keys, st.mu, st.gram, st.trace, linv=st._linv)
            res[nt] = {"gram": bool(torch.equal(fresh.gram, st.gram)), "mu": bool(torch.equal(fresh.mu, st.mu)), "trace": fresh.trace == st.trace,
                       "F": {WSt.kappa_tag(k): bool(torch.equal(fresh.whitener(kappa=k).F, stored.whitener(kappa=k).F)) for k in C.KAPPAS},
                       "threads_restored": torch.get_num_threads() == nt}
    finally:
        torch.set_num_threads(n0)
    ok = all(v["gram"] and v["mu"] and v["trace"] and all(v["F"].values()) and v["threads_restored"] for v in res.values())
    return {"pass": ok, "threads": res, "note": f"Gram matrix, mu, tr(C_W) and the factor R = L^-1 K (3 ridges) recomputed under 1 and 4 threads: bitwise equal to the stored ones: {ok}"}
