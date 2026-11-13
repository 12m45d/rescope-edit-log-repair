"""Lean review, CPU part 1: items 1 (new gates against the amendments, own re-implementation), 3 (_x construction), 4 (read-out H3),
5 (rescope_null with n_A = 0, rescope_pac).

    python rl_cpu1.py <development log>            -> out/cpu1_<log>.json
Own implementation = numpy float64 from the raw files: the log's factors (W, stored solver keys K), the key cache's teacher-forced keys,
the BEN-W keys taken from cache2/benign_parts by this file, the recombination store (rows.jsonl, keys.pt).  Nothing of fx.review1, of the
feature packages' score code or of the library's calibration is called for the quantity under test; the driver supplies only the
calibration draw's rows (checked here against rescope.splits.sample) and the rows of the key cache.
"""
import json
import math
import os
import sys
import time

import rl_common as C

NAME = sys.argv[1]
import numpy as np  # noqa: E402
import torch  # noqa: E402

import fx  # noqa: E402

SEED = 77033
TIE = 1e-9
ALPHA = float(os.environ.get("RL_ALPHA", "0.05"))
FAST = bool(os.environ.get("RL_FAST"))
TAG = "" if not FAST else f"_alpha{ALPHA:g}"
t00 = time.time()
L = fx.logs.open(NAME)
T = L.T
res = {"log": NAME, "fx": fx.code_fingerprint(), "items": {}}
rng = np.random.default_rng([SEED, 0 if L.family == "Q" else 1])


def say(*a):
    print(f"[{time.time() - t00:6.1f}s]", *a, flush=True)


def keys(which, rows):
    return L.kc.keys(which, [int(j) for j in rows]).double().numpy()


# ------------------------------------------------------------------ raw quantities
KE = L.log.K.double().numpy()                        # stored solver keys (T, d)
W = L.log.W.double().numpy()                         # read-out vectors (T, d)
s_own = (W * KE).sum(1)                              # s_i = <W_i, k_i>
d_in = KE.shape[1]
own_rel = keys("tf", L.edit_rows["rel"])             # B = 1 keys of the own Rel prompts (teacher-forced)
parts = json.load(open(f"{C.FIN}/cache2/benign_parts/{L.backbone}.json"))["parts"]
benw_rows = [j for j in L.benign_rows if parts[L.index[j]["sid"]] == "BEN-W"]
KW = keys("decode", benw_rows)                       # BEN-W keys (prompt-only key)
N = KW.shape[0]
mu = KW.mean(0)
trC = float((KW * KW).sum() / N)
say("BEN-W keys", KW.shape, "trace", trC)

# test sequences: 100 edit sequences, 100 heldout_disjoint probes, 100 hard negatives
ed = [int(j) for k in ("rel", "tgen", "igen") for j in L.edit_rows[k]]
hd = [int(j) for j in L.rows("probe", pool="heldout_disjoint", valid=True)]
hn = [int(j) for j in L.rows("hn", valid=True)]
SEQ = sorted(int(x) for x in rng.choice(ed, 100, replace=False)) + sorted(int(x) for x in rng.choice(hd, 100, replace=False)) + sorted(int(x) for x in rng.choice(hn, 100, replace=False))
KQ = keys("tf", SEQ)

# recombination rows of the log (own reading of the store)
rc_rows = [json.loads(x) for x in open(f"{C.FIN}/cache2/recomb/{L.backbone}/rows.jsonl") if x.strip()]
rc_blob = torch.load(f"{C.FIN}/cache2/recomb/{L.backbone}/keys.pt", map_location="cpu", mmap=True)
assert list(rc_blob["rids"]) == [r["rid"] for r in rc_rows]
pos_of_uid = {u: i for i, u in enumerate(L.uids)}
use = [i for i, r in enumerate(rc_rows) if r["edit_uid"] in pos_of_uid and (r["kind"] != "X-X" or r["edit_b_uid"] in pos_of_uid)]
rc_owner = np.array([pos_of_uid[rc_rows[i]["edit_uid"]] for i in use])
KR = rc_blob["K_last"][use].double().numpy()
say("recombination rows usable", len(use), "per edit", np.bincount(rc_owner, minlength=T).min(), np.bincount(rc_owner, minlength=T).max())

# ------------------------------------------------------------------ draws (driver's rows; checked against the library's sampler)
from rescope import splits  # noqa: E402

pool = [int(j) for j in L.pool("repair", valid=True)]
draws, nullA = {}, {}
chk_draw = {}
for dr in (0, 1):
    D = fx.draws.calibration(L, 200, dr)
    rows = [int(j) for j in D.rows]
    lib = sorted(pool[i] for i in splits.sample(list(range(len(pool))), 200, dr))
    A = [int(j) for j in fx.draws.null_sample(L, 100, dr)]
    qid = lambda rr: [L.index[j]["query_id"] for j in rr]          # noqa: E731
    chk_draw[dr] = {"draw_equals_library_sample": rows == lib, "all_valid_repair": set(rows) <= set(pool), "n": len(rows),
                    "null_n": len(A), "null_in_repair": set(A) <= set(pool), "null_distinct_queries": len(set(qid(A))), "null_rows_disjoint": not set(A) & set(rows),
                    "null_queries_shared_with_calibration": len(set(qid(A)) & set(qid(rows)))}
    draws[dr], nullA[dr] = rows, A
res["items"]["draws"] = chk_draw
say("draws", chk_draw)


def blocks(n, seed):
    perm = np.random.default_rng(seed).permutation(n)
    return np.sort(perm[:n // 2]), np.sort(perm[n // 2:])


def rank_k(n2, alpha=ALPHA):
    from fractions import Fraction
    a = Fraction(alpha).limit_denominator(10 ** 6)
    return math.ceil((1 - a) * (n2 + 1))


# ------------------------------------------------------------------ own features: name -> function keys (n, d) -> (n, T)
def unit(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def f_ncoord(K):
    return K @ W.T / s_own


def f_cos(K):
    return unit(K) @ unit(KE).T


def f_ball(K):
    return -np.stack([np.linalg.norm(K - KE[i][None, :], axis=1) for i in range(T)], 1)


def f_centred(K):
    return unit(K - mu) @ unit(KE - mu).T


_CH = {}


def chol(kappa):
    """Direct d x d factorisation of C_W + rho I (no Woodbury): the reviewer's 'mathematically equivalent direct solve'."""
    if kappa not in _CH:
        rho = kappa * trC / d_in
        A = torch.from_numpy(KW).T @ torch.from_numpy(KW) / N
        A.diagonal().add_(rho)
        _CH[kappa] = torch.linalg.cholesky(A)
        del A
    return _CH[kappa]


def Msolve(kappa, X):
    """rows x -> x M, M = (C_W + rho I)^-1."""
    return torch.cholesky_solve(torch.from_numpy(np.ascontiguousarray(X.T)), chol(kappa)).T.numpy()


_ME = {}


def f_white(kappa):
    def f(K):
        if kappa not in _ME:
            _ME[kappa] = Msolve(kappa, KE)
        ME = _ME[kappa]
        return (K @ ME.T) / (KE * ME).sum(1)
    return f


def f_whitecos(kappa):
    def f(K):
        if kappa not in _ME:
            _ME[kappa] = Msolve(kappa, KE)
        ME = _ME[kappa]
        MK = Msolve(kappa, K)
        return (K @ ME.T) / np.sqrt((KE * ME).sum(1))[None, :] / np.sqrt((K * MK).sum(1))[:, None]
    return f


def null_det(w, own_key, B):
    """Minimum-change w' with <w', b> = 0 for the rows b of B and <w', own_key> = <w, own_key> (ridge 0): w' = w - A'(A A')^-1 (A w - c)."""
    A = np.vstack([B, own_key[None, :]])
    r = A @ w
    r[-1] = 0.0
    lam = np.linalg.solve(A @ A.T, r)
    return w - A.T @ lam


def f_cos_null(B):
    Wn = np.stack([null_det(KE[i] / np.linalg.norm(KE[i]), KE[i], B) for i in range(T)])

    def f(K):
        return unit(K) @ Wn.T
    return f


def f_ncoord_null(B):
    G = B @ B.T
    Wn = np.empty_like(W)
    Gi = np.linalg.inv(G) if B.shape[0] <= 150 else None
    for i in range(T):
        if Gi is not None:
            Wn[i] = null_det(W[i], KE[i], B)
        else:                                               # 600 keys: block elimination of the same linear system (one factorisation of B B')
            Wn[i] = null_det_fast(W[i], KE[i], B, G)

    def f(K):
        return K @ Wn.T / s_own
    return f


_LU = {}


def null_det_fast(w, k, B, G):
    import scipy.linalg as sl
    if "c" not in _LU:
        _LU["c"] = sl.cho_factor(G)
    cf = _LU["c"]
    Bk, Bw = B @ k, B @ w
    y = sl.cho_solve(cf, np.stack([Bw, Bk], 1))             # G^-1 [Bw, Bk]
    # unknowns (lam, t): G lam + Bk t = Bw ; Bk' lam + |k|^2 t = 0
    sch = k @ k - Bk @ y[:, 1]
    t = -(Bk @ y[:, 0]) / sch
    lam = y[:, 0] - y[:, 1] * t
    return w - B.T @ lam - k * t


FEATS = {"rescope": f_ncoord, "cosine": f_cos, "ball": f_ball, "cos_centred": f_centred}
for kp, tag in ((0.01, "0.01"), (0.1, "0.1"), (1.0, "1")):
    FEATS[f"white_{tag}"] = f_white(kp)
    FEATS[f"whitecos_{tag}"] = f_whitecos(kp)
KAPPA_OF = {f"{f}_{tag}": kp for f in ("white", "whitecos") for kp, tag in ((0.01, "0.01"), (0.1, "0.1"), (1.0, "1"))}


def feature_fn(base, draw):
    if base == "cos_null":
        return f_cos_null(keys("tf", nullA[draw]))
    if base == "rescope_null":
        return f_ncoord_null(keys("tf", nullA[draw]))
    if base == "rescope_nullW":
        return f_ncoord_null(KW)
    return FEATS[base]


def own_rule(f, draw, rule, x=None, alpha=ALPHA):
    """Own calibration -> dict(lit, th, b (ranking baseline or None), eta, capped)."""
    Q = f(keys("tf", draws[draw]))
    g = np.minimum(np.diag(f(KE)), np.diag(f(own_rel)))        # the smallest coordinate of every edit on its own guard keys
    if rule == "two_block":
        i1, i2 = blocks(Q.shape[0], draw)
        b = Q[i1].max(0)
        if x is not None:
            b = np.maximum(b, x)
        s2 = np.sort((Q[i2] - b[None, :]).max(1))
        k = rank_k(len(i2), alpha)
        eta = s2[k - 1] if k <= len(i2) else np.inf
        strict = lambda R: (R - b[None, :]) > eta              # noqa: E731
        lit = b + eta
    else:
        b = None
        s = np.sort(Q.max(1))
        k = rank_k(Q.shape[0], alpha)
        eta = s[k - 1] if k <= Q.shape[0] else np.inf
        strict = lambda R: R > eta                             # noqa: E731
        lit = np.full(T, eta)
    own_pass = strict(g[None, :] * np.ones((1, T)))[0] if rule != "two_block" else ((g - b) > eta)
    capped = ~own_pass
    cap = g - TIE * np.abs(g)
    return {"b": b, "eta": float(eta), "lit": lit, "capped": capped, "cap": cap, "strict": strict, "rank": k, "g": g}


def own_decide(f, cal, K, delta=0.0, R=None):
    R = f(K) if R is None else R
    if delta == 0.0:
        cand = cal["strict"](R)
    else:
        base = cal["b"] if cal["b"] is not None else 0.0
        cand = (R - base) > (cal["eta"] + delta)
    lit_sil = (cal["g"] - (cal["b"] if cal["b"] is not None else 0.0)) <= cal["eta"] + delta            # the literal rule at this scale silences the own guard
    cand = np.where(lit_sil[None, :], R >= cal["cap"][None, :], cand)
    key = R - cal["b"][None, :] if cal["b"] is not None else R
    top = np.where(cand, key, -np.inf).argmax(1)
    fired = cand.any(1)
    return fired, np.where(fired, top, -1), cand, R


def own_x(f):
    R = f(KR)
    x = np.full(T, -np.inf)
    v = R[np.arange(len(use)), rc_owner]
    np.maximum.at(x, rc_owner, v)
    return x


# ------------------------------------------------------------------ item 1 + 3: every scoped gate of the review round, draws 0 and 1
BASES = ["cosine", "cos_centred", "cos_null", "white_0.01", "white_0.1", "white_1", "whitecos_0.01", "whitecos_0.1", "whitecos_1"]
GATES = [(b, "two_block", False) for b in BASES] + [(b, "shared", False) for b in BASES] + [("rescope", "two_block", False), ("rescope_null", "two_block", False),
         ("rescope_nullW", "two_block", False)] + [(b, "two_block", True) for b in ["rescope", "rescope_null", "rescope_nullW", "cosine", "ball"] + BASES[1:]]


if FAST:
    keepb = ("cosine", "cos_centred", "cos_null", "white_0.1", "whitecos_0.1", "rescope", "rescope_nullW")
    GATES = [g_ for g_ in GATES if g_[0] in keepb]


def drv_name(base, rule, x):
    return base + ("_shared" if rule == "shared" else "") + ("_x" if x else "")


own_ix = np.array([L.own(j) for j in SEQ])
Mcos = {dr: fx.methods.cosine(L, alpha=ALPHA, draw=dr) for dr in (0, 1)}
item1, item3, fair = {}, {}, {}
n_dec = n_diff = 0
xcache = {}
for base, rule, isx in GATES:
    nm = drv_name(base, rule, isx)
    for dr in (0, 1):
        t0 = time.time()
        M = fx.methods.build(nm, L, alpha=ALPHA, n=200, draw=dr)
        f = feature_fn(base, dr)
        x = None
        if isx:
            xk = (base, dr if base in ("cos_null", "rescope_null") else None)
            if xk not in xcache:
                xcache[xk] = own_x(f)
            x = xcache[xk]
        cal = own_rule(f, dr, rule, x)
        fired, top, cand, R = own_decide(f, cal, KQ)
        d = fx.gate.decide(L, M, SEQ, "tf")
        dfire, dtop = np.asarray(d["fired"]), np.asarray(d["top_edit"])
        diff = (fired != dfire) | (top != dtop)
        g = M.obj
        eta_d = float(M.rec["cal"]["eta"])
        capd = sorted(int(e) for e in M.rec["cal"].get("capped_edits", []))
        th_d = g.det_tau.numpy()[:T]
        th_mine = np.where(cal["capped"], np.minimum(cal["lit"], cal["cap"]), cal["lit"])
        fin = np.isfinite(th_d) & np.isfinite(th_mine)
        rec = {"n": len(SEQ), "n_decisions_differ": int(diff.sum()), "n_firing_mine": int(fired.sum()), "n_firing_driver": int(dfire.sum()),
               "eta_mine": cal["eta"], "eta_driver": eta_d, "eta_abs_diff": abs(cal["eta"] - eta_d), "rank": int(M.rec["cal"]["rank"]), "rank_mine": cal["rank"],
               "capped_equal": capd == sorted(int(i) for i in np.nonzero(cal["capped"])[0]), "n_capped": len(capd),
               "max_threshold_abs_diff": float(np.abs(th_d[fin] - th_mine[fin]).max()) if fin.any() else None,
               "own_cov_mine": int(((top == own_ix) & (own_ix >= 0)).sum()), "seconds": round(time.time() - t0, 2)}
        if diff.any():                                           # how close to a threshold are the differing rows?
            marg = np.abs(R - th_mine[None, :]).min(1)
            rec["differing_rows_min_distance_to_a_threshold"] = float(marg[diff].max())
        if isx:
            xd = fx.review1.x_baseline(L, fx.review1.unthresholded(L, base if base != "rescope_null" else "rescope_null", draw=dr))[0].numpy()[:T] if base not in ("rescope", "cosine", "ball") \
                else fx.review1.x_baseline(L, fx.methods.base_gate(L, {"rescope": "ncoord", "cosine": "cos", "ball": "negdist"}[base])[0])[0].numpy()[:T]
            b0 = fx.methods.build(drv_name(base, rule, False), L, alpha=ALPHA, n=200, draw=dr)
            Mn = fx.methods.build(nm, L, alpha=ALPHA, n=200, draw=dr, baseline_x=torch.full((T,), -float("inf"), dtype=torch.float64))
            Q = f(keys("tf", draws[dr]))
            i1, i2 = blocks(200, dr)
            b1 = Q[i1].max(0)
            item3.setdefault(nm, {})[dr] = {
                "x_max_abs_diff_driver_vs_own": float(np.abs(xd - x)[np.isfinite(x)].max()), "x_inf_pattern_equal": bool((np.isfinite(xd) == np.isfinite(x)).all()),
                "n_baselines_raised_mine": int((x > b1).sum()), "n_baselines_raised_driver": M.rec["cal"].get("baseline_x", {}).get("n_baselines_raised"),
                "baseline_is_max_block1_x": float(np.abs(g.det_b.numpy()[:T] - np.maximum(b1, x)).max()),
                "block2_equal_base_gate": M.rec["cal"]["block2"] == b0.rec["cal"]["block2"] and M.rec["cal"]["block2"] == [int(i) for i in i2],
                "rank_equal_base_gate": M.rec["cal"]["rank"] == b0.rec["cal"]["rank"], "guards_same_tensor_as_base": bool(torch.equal(g.guard_key, b0.obj.guard_key) and torch.equal(g.guard_edit, b0.obj.guard_edit)),
                "without_recombinations_thresholds_bitwise_base": bool(torch.equal(Mn.obj.det_tau, b0.obj.det_tau) and torch.equal(Mn.obj.det_b, b0.obj.det_b)),
                "without_recombinations_digest_equal": Mn.digest == b0.digest, "detectors_same_bits_as_base": bool(torch.equal(g.det_w, b0.obj.det_w))}
        # fairness against the registered cosine gate of the same draw
        if base not in ("rescope", "rescope_null", "rescope_nullW", "ball"):
            mc = Mcos[dr]
            fair.setdefault(nm, {})[dr] = {"values_bitwise": bool(torch.equal(g.values, mc.obj.values)), "role": g.role, "fire": g.fire, "response": g.response,
                                           "guard_keys_bitwise": bool(torch.equal(g.guard_key, mc.obj.guard_key) and torch.equal(g.guard_edit, mc.obj.guard_edit)),
                                           "same_draw_rows": M.rec["draw"]["sids"] == mc.rec["draw"]["sids"],
                                           "same_blocks": rule != "two_block" or M.rec["cal"]["block1"] == mc.rec["cal"]["block1"],
                                           "det_key_is_solver_key": bool(torch.equal(g.det_key[:T], L.log.K)), "alpha": M.alpha, "n": M.n}
        item1.setdefault(nm, {})[dr] = rec
        n_dec += len(SEQ)
        n_diff += int(diff.sum())
    say(nm, {dr: (item1[nm][dr]["n_decisions_differ"], item1[nm][dr]["n_firing_mine"], f"{item1[nm][dr]['eta_abs_diff']:.1e}", item1[nm][dr]["capped_equal"]) for dr in (0, 1)})
res["items"]["1_gates"] = {"n_gates": len(GATES), "n_decisions": n_dec, "n_differ": n_diff, "per_gate": item1,
                           "max_eta_abs_diff": max(v[dr]["eta_abs_diff"] for v in item1.values() for dr in v),
                           "all_capped_sets_equal": all(v[dr]["capped_equal"] for v in item1.values() for dr in v),
                           "all_ranks_equal": all(v[dr]["rank"] == v[dr]["rank_mine"] for v in item1.values() for dr in v)}
bad_fair = [(k, dr) for k, v in fair.items() for dr, r in v.items() if not (r["values_bitwise"] and r["role"] == "last" and r["fire"] == "top1" and r["response"] == "repl"
                                                                             and r["guard_keys_bitwise"] and r["same_draw_rows"] and r["same_blocks"] and r["det_key_is_solver_key"])]
res["items"]["1_fairness"] = {"n_checked": sum(len(v) for v in fair.values()), "violations": bad_fair, "per_gate": fair}
bad3 = [(k, dr) for k, v in item3.items() for dr, r in v.items() if not (r["x_max_abs_diff_driver_vs_own"] < 1e-9 and r["x_inf_pattern_equal"] and r["baseline_is_max_block1_x"] < 1e-9
                                                                         and r["block2_equal_base_gate"] and r["rank_equal_base_gate"] and r["guards_same_tensor_as_base"]
                                                                         and r["without_recombinations_thresholds_bitwise_base"] and r["detectors_same_bits_as_base"])]
res["items"]["3_x"] = {"n_checked": sum(len(v) for v in item3.values()), "violations": bad3, "per_gate": item3}
C.dump(f"cpu1_{NAME}{TAG}.json", res)
say("item 1:", n_dec, "decisions,", n_diff, "differ; fairness violations", bad_fair, "; item 3 violations", bad3)
if FAST:
    sys.exit(0)

# ------------------------------------------------------------------ item 4: read-out H3, own computation
pos = [int(L.index[L.edit_rows["rel"][i]].get("edit_pos", i)) for i in range(T)]
hn_kind = {j: L.index[j]["hn_type"] for j in hn}
hn_own = {j: L.own(j) for j in hn}
fit = {k: [j for j in hn if hn_kind[j] == k and pos[hn_own[j]] % 2 == 0] for k in ("HN-I", "HN-Q", "HN-X")}
test = {k: [j for j in hn if hn_kind[j] == k and pos[hn_own[j]] % 2 == 1] for k in ("HN-I", "HN-Q", "HN-X")}
odd = [i for i in range(T) if pos[i] % 2 == 1]
# the own edit of a hard negative, checked from the raw index: HN-I / HN-X: the edit whose image the row shows; HN-Q: the edit whose question it asks
img_of_edit = {L.index[L.edit_rows["rel"][i]]["image_md5"]: i for i in range(T)}
q_of_edit = {}
for i in range(T):
    q_of_edit.setdefault(L.index[L.edit_rows["rel"][i]]["prompt"].strip().lower(), []).append(i)
own_chk = {"HN-I": 0, "HN-Q": 0, "HN-X": 0}
for j in hn:
    e = L.index[j]
    if hn_kind[j] in ("HN-I", "HN-X"):
        own_chk[hn_kind[j]] += int(img_of_edit.get(e["image_md5"]) == hn_own[j])
    else:
        own_chk["HN-Q"] += int(hn_own[j] in q_of_edit.get(e["prompt"].strip().lower(), []))
h3 = {"own_edit_matches_raw_index": {k: [own_chk[k], sum(1 for j in hn if hn_kind[j] == k)] for k in own_chk}, "n_fit": {k: len(v) for k, v in fit.items()},
      "n_test": {k: len(v) for k, v in test.items()}, "gates": {}}
for nm, base, rule in (("rescope", "rescope", "two_block"), ("whitecos_0.1", "whitecos_0.1", "two_block"), ("cosine_shared", "cosine", "shared"), ("cos_centred", "cos_centred", "two_block"),
                       ("cosine", "cosine", "two_block"), ("rescope_x", "rescope", "two_block"), ("cosine_x", "cosine", "two_block"), ("whitecos_0.1_x", "whitecos_0.1", "two_block"),
                       ("cos_null_x", "cos_null", "two_block"), ("white_1_shared", "white_1", "shared")):
    for dr in (0, 1):
        M = fx.methods.build(nm, L, alpha=ALPHA, n=200, draw=dr)
        f = feature_fn(base, dr)
        cal = own_rule(f, dr, rule, own_x(f) if nm.endswith("_x") else None)
        h3.setdefault("n_capped", {}).setdefault(nm, {})[dr] = int(cal["capped"].sum())
        Rf = {k: f(keys("tf", v)) for k, v in fit.items() if len(v)}

        def fit_rate(delta):
            return float(np.mean([100.0 * own_decide(f, cal, None, delta, R=Rf[k])[0].mean() for k in Rf]))
        # critical shifts: a fit row stops firing just above max_j (r_j - b_j - eta) unless a capped edit holds it
        if fit_rate(0.0) <= 10.0:
            dstar = 0.0
        else:
            crit = []
            for k in Rf:
                R = Rf[k]
                base_b = cal["b"][None, :] if cal["b"] is not None else 0.0
                crit.append(((R - base_b) - cal["eta"]).max(1))
            crit = np.unique(np.concatenate(crit))
            crit = crit[crit > 0]
            dstar = None
            for c in crit:                                       # smallest critical value just above which the mean fit rate is <= 10
                if fit_rate(float(np.nextafter(c, np.inf)) + 1e-12) <= 10.0:
                    dstar = float(c)
                    break
        h = fx.sweeps.h3(L, M)
        dd = h["delta"]
        rec = {"reachable_driver": h["reachable"], "delta_driver": dd, "delta_mine": dstar}
        if h["reachable"] and dstar is not None:
            rec["delta_abs_diff"] = abs(dd - dstar)
            rec["fit_mean_at_driver_delta_mine"] = fit_rate(dd)
            rec["fit_mean_at_driver_delta_driver"] = h["fit"]["at_raise"]["mean"]
            rec["fit_mean_just_below_mine"] = fit_rate(dd - 1e-9) if dd > 0 else None
            rec["fit_mean_just_above_mine"] = fit_rate(dd + 1e-9)
            cov = {}
            for k in ("tgen", "igen"):
                fr, tp, _, _ = own_decide(f, cal, keys("tf", [L.edit_rows[k][i] for i in odd]), dd)
                cov[k] = 100.0 * float((tp == np.array(odd)).mean())
            rec["coverage_mine"] = {"tgen": cov["tgen"], "igen": cov["igen"], "mean": 0.5 * (cov["tgen"] + cov["igen"])}
            rec["coverage_driver"] = {k: h["rephrase"][k] for k in ("tgen", "igen", "mean")}
            tm, td = {}, {}
            for k in test:
                if not test[k]:
                    continue
                fr, tp, _, _ = own_decide(f, cal, keys("tf", test[k]), dd)
                tm[k] = 100.0 * float(fr.mean())
                td[k] = h["test"][k]["fired"]
            rec["test_fired_mine"], rec["test_fired_driver"] = tm, td
            fr, _, _, _ = own_decide(f, cal, keys("tf", hd), dd)
            rec["heldout_disjoint_fired_mine"], rec["heldout_disjoint_fired_driver"] = 100.0 * float(fr.mean()), h["heldout_disjoint"]["fired"]
            rec["n_heldout_disjoint"] = [len(hd), h["heldout_disjoint"]["n"]]
            rec["odd_edits_equal"] = h["odd_edits"] == odd
            rec["test_rows_equal"] = sorted(h["test_rows"]["rows"]) == sorted(j for k in test for j in test[k])
            rec["agree"] = bool(rec["delta_abs_diff"] < 1e-9 and abs(rec["fit_mean_just_above_mine"] - rec["fit_mean_at_driver_delta_driver"]) < 1e-9
                                and all(abs(rec["coverage_mine"][k] - rec["coverage_driver"][k]) < 1e-9 for k in ("tgen", "igen", "mean"))
                                and all(abs(tm[k] - td[k]) < 1e-9 for k in tm) and abs(rec["heldout_disjoint_fired_mine"] - rec["heldout_disjoint_fired_driver"]) < 1e-9
                                and rec["odd_edits_equal"] and rec["test_rows_equal"] and (dd == 0 or rec["fit_mean_just_below_mine"] > 10.0) and rec["fit_mean_just_above_mine"] <= 10.0)
        else:
            rec["agree"] = (not h["reachable"]) and dstar is None
        h3["gates"].setdefault(nm, {})[dr] = rec
    say("H3", nm, {dr: (h3["gates"][nm][dr]["agree"], h3["gates"][nm][dr].get("delta_abs_diff")) for dr in (0, 1)})
h3["all_agree"] = all(r["agree"] for v in h3["gates"].values() for r in v.values())
res["items"]["4_h3"] = h3

# ------------------------------------------------------------------ item 5: rescope_null with n_A = 0, rescope_pac
M0 = fx.methods.rescope(L, alpha=ALPHA, draw=0)
Mn0 = fx.methods.rescope_null(L, alpha=ALPHA, draw=0, n_A=0)
same = all(torch.equal(getattr(M0.obj, a), getattr(Mn0.obj, a)) for a in ("det_w", "det_s", "det_tau", "det_b", "values", "guard_key", "guard_edit", "det_edit", "det_key"))
i5 = {"rescope_null_nA0": {"tensors_bitwise_equal": bool(same), "digest_equal": M0.digest == Mn0.digest, "eta_equal": M0.obj.eta == Mn0.obj.eta,
                            "rank_baselines_equal": bool(torch.equal(M0.obj.det_b, Mn0.obj.det_b))}}
# rescope_null: the detectors against the own closed form, draw 0 (0 on the null keys, own coordinate kept)
Mn = fx.methods.rescope_null(L, alpha=ALPHA, draw=0)
Bn = keys("tf", nullA[0])
Rn = Mn.obj.scores(torch.from_numpy(Bn)).numpy()[:, :T]
i5["rescope_null"] = {"max_abs_coordinate_on_null_keys": float(np.abs(Rn).max()), "own_coordinate_kept_max_dev": float(np.abs(np.diag(Mn.obj.scores(L.log.K).numpy()[:, :T]) - 1).max()),
                      "null_sids_equal_cos_null": fx.methods.build("cos_null", L, alpha=ALPHA, draw=0).rec["null_sample"]["sids"] == Mn.rec["null_repair"]["sids"],
                      "null_sids_are_driver_sample": Mn.rec["null_repair"]["sids"] == [L.sid(j) for j in nullA[0]]}
# rescope_pac: the rank by exact computation (fractions) and by scipy's Beta law
from fractions import Fraction  # noqa: E402
from math import comb  # noqa: E402


def pac_exact(n2, alpha, delta):
    a, dl = Fraction(alpha).limit_denominator(10 ** 6), Fraction(delta).limit_denominator(10 ** 6)
    for k in range(1, n2 + 1):
        # P[Beta(n2 + 1 - k, k) <= a] = P[Bin(n2, a) >= n2 + 1 - k]
        p = sum(Fraction(comb(n2, i)) * a ** i * (1 - a) ** (n2 - i) for i in range(n2 + 1 - k, n2 + 1))
        if p >= 1 - dl:
            return k
    return n2 + 1


from scipy import stats as sst  # noqa: E402

grid = [(100, 0.05, 0.1), (100, 0.01, 0.1), (100, 0.02, 0.1), (100, 0.1, 0.1), (100, 0.2, 0.1), (50, 0.05, 0.1), (200, 0.05, 0.1), (20, 0.05, 0.1), (100, 0.05, 0.05), (150, 0.1, 0.2), (10, 0.2, 0.1), (30, 0.1, 0.1)]
pac = []
for n2, a, dl in grid:
    k_d = fx.methods.pac_rank(n2, a, dl)
    k_m = pac_exact(n2, a, dl)
    k_s = next((k for k in range(1, n2 + 1) if sst.beta.cdf(a, n2 + 1 - k, k) >= 1 - dl - 1e-12), n2 + 1)
    pac.append({"n2": n2, "alpha": a, "delta": dl, "driver": k_d, "exact": k_m, "scipy": k_s})
Mp = fx.methods.rescope_pac(L, alpha=ALPHA, draw=0)
Qp = f_ncoord(keys("tf", draws[0]))
i1, i2 = blocks(200, 0)
bp = Qp[i1].max(0)
s2 = np.sort((Qp[i2] - bp).max(1))
kp_ = pac_exact(100, 0.05, 0.1)
i5["rescope_pac"] = {"grid": pac, "all_equal": all(r["driver"] == r["exact"] == r["scipy"] for r in pac), "rank_n100": kp_, "rank_in_record": Mp.rec["cal"]["rank"],
                     "eta_mine": float(s2[kp_ - 1]), "eta_driver": float(Mp.rec["cal"]["eta"]), "eta_default": float(M0.rec["cal"]["eta"]),
                     "detectors_equal_default": bool(torch.equal(Mp.obj.det_w, M0.obj.det_w)), "baselines_equal_default": bool(torch.equal(Mp.obj.det_b, M0.obj.det_b))}
res["items"]["5_null_pac"] = i5
say("item 5", {k: {kk: vv for kk, vv in v.items() if kk != "grid"} for k, v in i5.items()})
res["seconds"] = round(time.time() - t00, 1)
C.dump(f"cpu1_{NAME}.json", res)
say("written")
