"""white.wstats: statistics of a set of reference keys and the whitening operator M = (C + rho I)^-1 (PROTOCOL.md section 6, amendment
"Review round 1", item R3: "mu = mean BEN-W key", "C_W = uncentred second moment of the BEN-W keys, rho = kappa tr(C_W) / d_in").

DEFINITIONS (reference keys = rows of K, an (N, d) matrix; for the paper: the BEN-W keys, N = 600, d = d_in = 11008)
    mu     = (1/N) sum_n k_n                                   the mean key
    C      = (1/N) sum_n k_n k_n' = K'K / N                    the UNCENTRED second moment (d x d; never formed)
    G      = K K'                                              the N x N Gram matrix;  tr(C) = tr(G) / N = mean squared key norm
    rho    = kappa tr(C) / d                                   kappa in {0.01, 0.1, 1}: rho = kappa times the mean eigenvalue of C
    M      = (C + rho I)^-1

WOODBURY ON THE GRAM MATRIX (N <= d; the form used for the paper: d = 11008 exceeds N = 600, so no d x d inverse is formed)
    M = (1/rho) [ I - K' (G + N rho I)^-1 K ]
  With the Cholesky factor  L L' = G + N rho I  and  R = L^-1 K  (N x d):      M = (I - R'R) / rho,   hence for rows x, y
    x M      = (x - (x R') R) / rho                 Whitener.apply
    x M x'   = (|x|^2 - |x R'|^2) / rho             Whitener.quad        (a difference of two positive numbers: see PRECISION)
    x M y'   = (x y' - (x R')(y R')') / rho         Whitener.bilinear
  PRIMAL FORM (N > d, e.g. keys at all positions of many prompts): L L' = C + rho I (d x d), F = L^-1, M = F'F:
    x M = (x F') F,  x M x' = |x F'|^2,  x M y' = (x F')(y F')'.   Chosen automatically; both forms are tested against an explicit inverse.

PRECISION.  Everything is float64.  The keys are float32 images of the module dtype (exact); the mean is accumulated sequentially in
  float64.  cond(G + N rho I) = (lambda_1 + N rho) / (lambda_N + N rho) (BEN-W, kappa = 0.01: about 1e5; stored in meta.json), so the
  factors carry about 1e-11 relative error at worst.  x M x' loses a further factor |x|^2 / (rho x M x') by cancellation: 1 / (share of
  |x|^2 outside the span of the reference keys) for a fresh key (a few units), about 2e3 for a reference key itself at kappa = 0.01.
  Measured against an explicit d x d inverse in tests/t_scores.py (naive path).  Float32 is NOT sufficient for x M x'.
DETERMINISM.  Every factorisation and every product that defines a stored or derived quantity runs inside `single_thread()` (one BLAS /
  LAPACK thread): MKL's results depend on the number of threads in their last bits, with one thread they are the same bits in every
  process (tests/t_stats.py re-runs the build under other thread settings and compares the files byte for byte).
COST.  BEN-W statistics: 0.6 s; a whitener: 0.3 s and N x d float64 (53 MB at N = 600, d = 11008); x M: 2 N d flops per row.

FILES  10_final/cache2/whiten/<backbone>/   (written by `python -m white.wstats build`; no time stamp inside)
  stats.pt     torch file: sids (N BEN-W rows, pool order), key_rows (rows of the cache's keys_decode.pt), keys (N, d) float32 = the
               prompt-only key at the last prompt position of every BEN-W row (the cache's decode key), mu (d,) float64, gram (N, N)
               float64, trace = tr(C_W), and per kappa: rho and linv = L^-1 (N, N) float64 with L L' = gram + N rho I
  meta.json    provenance (parts file, pool, key cache: md5), shapes, tr(C_W), |mu|, spectrum summary of the Gram matrix, per kappa rho /
               effective degrees of freedom / condition numbers, sha1 of every tensor, md5 of stats.pt
  README.txt   this description with the numbers

READER
    st = white.wstats.load("qwen25vl")          # KeyStats: st.mu, st.keys, st.gram, st.trace, st.rho(kappa), st.whitener(kappa)
    wh = st.whitener(0.1)                        # Whitener: wh.apply(X), wh.quad(X), wh.bilinear(X, Y), wh.rho
    st2 = white.wstats.KeyStats.from_keys(keys)  # the same statistics of any other key matrix (N <= d or N > d)
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os

from . import benparts
from . import common as C

torch = C.use_library()

KAPPAS = C.KAPPAS
PART = "BEN-W"
WHICH = "decode"                                       # the key of a BEN-W row: the prompt-only key at the last prompt position


@contextlib.contextmanager
def single_thread():
    """One BLAS / LAPACK thread inside the block (restored afterwards): the same bits in every process, whatever its thread setting."""
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(n)


def kappa_tag(kappa: float) -> str:
    """'0.01', '0.1', '1' (the name of a ridge in files and labels)."""
    return f"{float(kappa):g}"


def sha1_tensor(t) -> str:
    t = t.detach().cpu().contiguous()
    return hashlib.sha1(str(t.dtype).encode() + str(tuple(t.shape)).encode() + t.numpy().tobytes()).hexdigest()[:16]


def seq_mean(K):
    """(d,) float64 mean of the rows of K, accumulated row by row in float64 (one fixed order of additions)."""
    acc = torch.zeros(K.shape[1], dtype=torch.float64)
    for n in range(K.shape[0]):
        acc += K[n].double()
    return acc / K.shape[0]


class Whitener:
    """M = (C + rho I)^-1 for C = K'K / N, applied without a d x d inverse where N <= d (module docstring).

      form 'gram'    F = R = L^-1 K (N, d), L L' = K K' + N rho I:    M = (I - F'F) / rho      (c0, c1) = (1/rho, -1/rho)
      form 'primal'  F = L^-1 (d, d),       L L' = K'K / N + rho I:   M = F'F                  (c0, c1) = (0, 1)
    so that for rows x, y, with p_x = x F':   x M = c0 x + c1 p_x F;   x M y' = c0 x y' + c1 p_x p_y';   x M x' = c0 |x|^2 + c1 |p_x|^2.
    rho, kappa (None if rho was given directly), n, d, F, c0, c1, rec (how it was built)."""

    def __init__(self, form, F, rho, n, d, kappa=None, rec=None):
        if form not in ("gram", "primal"):
            raise ValueError(form)
        self.form, self.F, self.rho, self.n, self.d, self.kappa = form, F, float(rho), int(n), int(d), kappa
        self.c0, self.c1 = (1.0 / self.rho, -1.0 / self.rho) if form == "gram" else (0.0, 1.0)
        self.rec = dict(rec or {})

    def _rows(self, X):
        X = torch.as_tensor(X)
        return X.reshape(-1, self.d).to(self.F.device, torch.float64)

    def apply(self, X):
        """Rows x -> x M (= (M x')'; M is symmetric).  (n, d) float64."""
        X = self._rows(X)
        return self.c0 * X + self.c1 * ((X @ self.F.T) @ self.F)

    def quad(self, X):
        """(n,) float64: x M x' for every row (positive; raises if round-off made it non-positive)."""
        X = self._rows(X)
        P = X @ self.F.T
        q = self.c0 * (X * X).sum(1) + self.c1 * (P * P).sum(1)
        if X.shape[0] and not bool((q > 0).all()):
            raise FloatingPointError("x M x' is not positive for some row (a zero key, or cancellation beyond float64)")
        return q

    def bilinear(self, X, Y):
        """(n, m) float64: x M y' for every pair of rows."""
        X, Y = self._rows(X), self._rows(Y)
        return self.c0 * (X @ Y.T) + self.c1 * ((X @ self.F.T) @ (Y @ self.F.T).T)

    def dense(self):
        """M as a (d, d) matrix (tests on small problems only)."""
        return self.apply(torch.eye(self.d, dtype=torch.float64, device=self.F.device))


def factor(K, rho: float, gram=None, linv=None, kappa=None) -> Whitener:
    """The whitener of the key matrix K (N, d) at ridge rho (module docstring).  gram: K K' if already known; linv: a stored L^-1
    (gram form), used instead of a new factorisation.  Runs single-threaded (deterministic bits)."""
    N, d = K.shape
    if not rho > 0:
        raise ValueError("rho must be > 0")
    with single_thread():
        Kd = K.double()
        if N <= d:
            if linv is None:
                G = Kd @ Kd.T if gram is None else gram
                A = G + N * rho * torch.eye(N, dtype=torch.float64)
                L = torch.linalg.cholesky(A)
                linv = torch.linalg.solve_triangular(L, torch.eye(N, dtype=torch.float64), upper=False)
            return Whitener("gram", linv @ Kd, rho, N, d, kappa, {"form": "gram", "N": N, "d": d, "rho": rho})
        Cm = (Kd.T @ Kd) / N + rho * torch.eye(d, dtype=torch.float64)
        L = torch.linalg.cholesky(Cm)
        Li = torch.linalg.solve_triangular(L, torch.eye(d, dtype=torch.float64), upper=False)
        return Whitener("primal", Li, rho, N, d, kappa, {"form": "primal", "N": N, "d": d, "rho": rho})


class KeyStats:
    """Statistics of a set of reference keys.

      keys (N, d) float32 (as cached) or float64;  n, d;  mu (d,) float64;  gram (N, N) float64 (None if N > d);  trace = tr(C)
      rho(kappa) = kappa trace / d;   whitener(kappa=None, rho=None) -> Whitener (cached per ridge)
      name, sids (reference rows, if they are rows of a pool), meta (dict)
    """

    def __init__(self, keys, mu, gram, trace, name="", sids=None, linv=None, meta=None):
        self.keys, self.mu, self.gram, self.trace = keys, mu, gram, float(trace)
        self.n, self.d = int(keys.shape[0]), int(keys.shape[1])
        self.name, self.sids, self.meta = name, sids, dict(meta or {})
        self._linv = dict(linv or {})                                    # kappa tag -> stored L^-1
        self._wh = {}

    @staticmethod
    def from_keys(keys, name: str = "", sids=None) -> "KeyStats":
        """Statistics of any key matrix (N, d) (float32 / float64 / module dtype)."""
        keys = torch.as_tensor(keys).detach().cpu()
        if keys.dtype not in (torch.float32, torch.float64):
            keys = keys.float()                                          # half / bfloat16 values are exact in float32
        if keys.dim() != 2 or keys.shape[0] < 1 or not bool(torch.isfinite(keys).all()):
            raise ValueError("keys: a finite (N, d) matrix with N >= 1")
        N, d = keys.shape
        with single_thread():
            Kd = keys.double()
            mu = seq_mean(keys)
            gram = Kd @ Kd.T if N <= d else None
            trace = float(gram.diagonal().sum()) / N if gram is not None else float((Kd * Kd).sum()) / N
        return KeyStats(keys, mu, gram, trace, name, sids)

    def rho(self, kappa: float) -> float:
        """rho = kappa tr(C) / d."""
        if not kappa > 0:
            raise ValueError("kappa must be > 0")
        return float(kappa) * self.trace / self.d

    def whitener(self, kappa=None, rho=None) -> Whitener:
        """M = (C + rho I)^-1 with rho = kappa tr(C) / d (or an explicit rho)."""
        if (kappa is None) == (rho is None):
            raise ValueError("give exactly one of kappa and rho")
        key = ("kappa", kappa_tag(kappa)) if kappa is not None else ("rho", repr(float(rho)))
        if key not in self._wh:
            r = self.rho(kappa) if kappa is not None else float(rho)
            linv = self._linv.get(kappa_tag(kappa)) if kappa is not None else None
            wh = factor(self.keys, r, gram=self.gram, linv=linv, kappa=None if kappa is None else float(kappa))
            wh.rec.update(stats=self.name, linv="stored" if linv is not None else "computed")
            self._wh[key] = wh
        return self._wh[key]

    def spectrum(self) -> dict:
        """Summary of the eigenvalues of C (those of the Gram matrix / N): label-free description of the reference keys."""
        if self.gram is None:
            return {}
        with single_thread():
            lam = torch.linalg.eigvalsh(self.gram).flip(0).clamp_min(0) / self.n
        tot = float(lam.sum())
        out = {"top_shares_of_trace": [float(x) / tot for x in lam[:5]], "lambda_max": float(lam[0]), "lambda_min": float(lam[-1]),
               "participation_ratio": tot ** 2 / float((lam ** 2).sum()), "per_kappa": {}}
        for kappa in KAPPAS:
            r = self.rho(kappa)
            a = lam / (lam + r)
            out["per_kappa"][kappa_tag(kappa)] = {
                "rho": r, "N_rho": self.n * r, "effective_dof": float(a.sum()), "n_eigenvalues_above_rho": int((lam > r).sum()),
                "cond_gram_plus_N_rho": float((lam[0] + r) / (lam[-1] + r)), "cond_M": float((lam[0] + r) / r),
                "mean_xMx_of_reference_keys": float(a.sum()), "mean_cancellation_of_reference_keys": self.trace / r / float(a.sum())}
        return out


# ----------------------------------------------------------------------------- the stored BEN-W statistics
def out_dir(backbone: str, directory: str | None = None) -> str:
    return os.path.join(C.WHITEN_DIR if directory is None else directory, backbone)


def benw_keys(backbone: str, verify_cache: bool = False):
    """(sids, key rows, keys (N, d) float32, key cache): the prompt-only key at the last prompt position of every BEN-W row = the
    cache's decode key of the row (keys_decode.pt[row_decode]), in pool order."""
    from rescope import KeyCache
    kc = KeyCache(backbone, verify=verify_cache)
    P = benparts.load(backbone)
    rows = P.rows(PART, kc)
    sids = [kc.index[j]["sid"] for j in rows]
    if sids != P.sids(PART):
        raise RuntimeError("BEN-W rows of the key cache are not the BEN-W rows of the parts file")
    keys = kc.keys(WHICH, rows).clone()
    if keys.dtype != torch.float32 or not bool(torch.isfinite(keys).all()):
        raise RuntimeError("the cached BEN-W keys must be finite float32")
    return sids, kc.key_rows(WHICH, rows), keys, kc, P


def build(backbone: str) -> tuple:
    """(blob for stats.pt, meta dict) of one backbone."""
    sids, key_rows, keys, kc, P = benw_keys(backbone, verify_cache=True)
    st = KeyStats.from_keys(keys, name=f"{PART}:{backbone}", sids=sids)
    blob = {"backbone": backbone, "part": PART, "which": WHICH, "sids": sids, "key_rows": [int(r) for r in key_rows], "keys": keys, "mu": st.mu,
            "gram": st.gram, "trace": st.trace, "n": st.n, "d": st.d, "kappa": {}}
    for kappa in KAPPAS:
        rho = st.rho(kappa)
        with single_thread():
            A = st.gram + st.n * rho * torch.eye(st.n, dtype=torch.float64)
            L = torch.linalg.cholesky(A)
            linv = torch.linalg.solve_triangular(L, torch.eye(st.n, dtype=torch.float64), upper=False)
            resid = float((linv @ A @ linv.T - torch.eye(st.n, dtype=torch.float64)).abs().max())
        blob["kappa"][kappa_tag(kappa)] = {"kappa": float(kappa), "rho": rho, "linv": linv, "max_abs_linv_A_linvT_minus_I": resid}
    nrm = keys.double().norm(dim=1)
    ptype = [kc.index[kc.by_sid(s)]["ptype"] for s in sids]
    meta = {"what": "statistics of the BEN-W keys: mean key mu, Gram matrix of the uncentred second moment C_W, Woodbury factors of (C_W + rho I)^-1",
            "protocol": "10_final/PROTOCOL.md, section 6, amendment 'Review round 1', item R3",
            "backbone": backbone, "layer": kc.layer, "part": PART,
            "key": "prompt-only key at the last prompt position of every BEN-W row = the key cache's decode key (keys_decode.pt[row_decode]), float32 "
                   "image of the module dtype",
            "n": st.n, "d": st.d, "rows_by_ptype": {p: ptype.count(p) for p in ("image", "text")},
            "trace_C_W": st.trace, "mean_eigenvalue_trace_over_d": st.trace / st.d, "norm_mu": float(st.mu.norm()),
            "share_of_trace_in_mean_direction": float(st.mu.norm() ** 2) / st.trace,
            "key_norm_min_median_max": [float(nrm.min()), float(nrm.median()), float(nrm.max())],
            "kappas": [float(k) for k in KAPPAS], "rho": {kappa_tag(k): st.rho(k) for k in KAPPAS}, "rho_rule": "rho = kappa tr(C_W) / d",
            "spectrum": st.spectrum(),
            "linv_check": {t: v["max_abs_linv_A_linvT_minus_I"] for t, v in blob["kappa"].items()},
            "sources": {"benign_parts": os.path.relpath(P.path, C.FINAL), "benign_parts_md5": C.md5_file(P.path),
                        "pool_md5": P.obj["source"]["md5"], "key_cache": os.path.relpath(kc.dir, C.FINAL), "key_cache_index_md5": kc.meta.get("index_md5"),
                        "keys_decode_md5": (kc.meta.get("keys_decode") or {}).get("md5"), "key_files_md5_verified": True},
            "sha1": {"keys": sha1_tensor(keys), "mu": sha1_tensor(st.mu), "gram": sha1_tensor(st.gram),
                     **{f"linv_{t}": sha1_tensor(v["linv"]) for t, v in blob["kappa"].items()}},
            "precision": "float64 statistics of float32 keys; factorisations single-threaded (deterministic)",
            "torch": torch.__version__}
    return blob, meta


def _save_bytes(blob) -> bytes:
    buf = io.BytesIO()
    torch.save(blob, buf)
    return buf.getvalue()


def write(backbone: str, directory: str | None = None) -> dict:
    blob, meta = build(backbone)
    d = out_dir(backbone, directory)
    data = _save_bytes(blob)
    meta["stats_pt_md5"] = C.md5_bytes(data)
    meta["stats_pt_bytes"] = len(data)
    C.write_atomic(os.path.join(d, "stats.pt"), data)
    C.write_atomic(os.path.join(d, "meta.json"), C.jdumps(meta))                 # written last: the reader checks stats.pt against it
    C.write_atomic(os.path.join(d, "README.txt"), readme(meta))
    return meta


def readme(meta: dict) -> str:
    sp = meta["spectrum"]
    lines = [f"10_final/cache2/whiten/{meta['backbone']} -- statistics of the BEN-W keys of backbone {meta['backbone']} (layer {meta['layer']})",
             "Written by `python -m white.wstats build` (code and reference text: 10_final/feat2/white/wstats.py).  PROTOCOL.md section 6, amendment",
             "'Review round 1', item R3.  No time stamp: a re-run reproduces stats.pt and meta.json byte for byte (feat2/white/tests/t_stats.py).",
             "",
             f"keys      {meta['n']} BEN-W rows ({meta['rows_by_ptype']['image']} image, {meta['rows_by_ptype']['text']} text), d = {meta['d']}: {meta['key']}",
             f"          key norm min / median / max = {meta['key_norm_min_median_max'][0]:.3f} / {meta['key_norm_min_median_max'][1]:.3f} / {meta['key_norm_min_median_max'][2]:.3f}",
             f"mu        mean key, |mu| = {meta['norm_mu']:.4f}; |mu|^2 / tr(C_W) = {meta['share_of_trace_in_mean_direction']:.4f}",
             f"C_W       uncentred second moment (1/N) sum k k', kept as its N x N Gram matrix K K'; tr(C_W) = {meta['trace_C_W']:.6f}, tr / d = {meta['mean_eigenvalue_trace_over_d']:.6e}",
             f"          eigenvalue shares of the trace (top 5): {', '.join(f'{x:.4f}' for x in sp['top_shares_of_trace'])}; participation ratio {sp['participation_ratio']:.2f}; "
             f"lambda_min / lambda_max = {sp['lambda_min'] / sp['lambda_max']:.2e}",
             "M         (C_W + rho I)^-1 = (I - R'R) / rho, R = L^-1 K, L L' = K K' + N rho I (Woodbury on the Gram matrix; no d x d inverse)",
             "          kappa      rho          N rho     effective dof   eigenvalues > rho   cond(KK' + N rho I)   cond(M)    mean x M x' (BEN-W keys)   cancellation there"]
    for t, v in sp["per_kappa"].items():
        lines.append(f"          {t:<8}   {v['rho']:.4e}   {v['N_rho']:.4e}   {v['effective_dof']:>10.1f}   {v['n_eigenvalues_above_rho']:>10d} of {meta['n']}   "
                     f"{v['cond_gram_plus_N_rho']:>14.3e}   {v['cond_M']:.3e}   {v['mean_xMx_of_reference_keys']:>14.1f}   {v['mean_cancellation_of_reference_keys']:>18.1f}")
    lines += ["          effective dof = sum of lambda / (lambda + rho) over the eigenvalues of C_W = mean of x M x' over the BEN-W keys themselves;",
              "          cancellation = |x|^2 / (rho x M x'): the digits that x M x' = (|x|^2 - |x R'|^2) / rho loses (float64 is required).",
              "",
              "files     stats.pt   torch.load -> dict: backbone, part, which, sids, key_rows, keys (N, d) float32, mu (d,) float64, gram (N, N) float64,",
              "                     trace, n, d, kappa: {'0.01' | '0.1' | '1': {kappa, rho, linv (N, N) float64 = L^-1}}",
              f"                     md5 {meta['stats_pt_md5']}, {meta['stats_pt_bytes']} bytes",
              "          meta.json  provenance, numbers above, sha1 of every tensor",
              "reader    white.wstats.load(backbone) -> KeyStats (mu, keys, gram, trace, rho(kappa), whitener(kappa)); see feat2/white/USAGE.txt",
              f"sources   {meta['sources']['benign_parts']} (md5 {meta['sources']['benign_parts_md5']}); {meta['sources']['key_cache']}/keys_decode.pt "
              f"(md5 {meta['sources']['keys_decode_md5']}, verified at build time)",
              ""]
    return "\n".join(lines)


def root_readme(directory: str | None = None) -> str:
    """The index of 10_final/cache2/whiten (from the meta.json files that exist)."""
    root = C.WHITEN_DIR if directory is None else directory
    L = ["10_final/cache2/whiten -- statistics of the BEN-W keys, per backbone (PROTOCOL.md section 6, amendment 'Review round 1', item R3)",
         "Written by `python -m white.wstats build` (cwd = 10_final/feat2; code and reference text: feat2/white/wstats.py; usage: feat2/white/USAGE.txt).",
         "",
         "  <backbone>/stats.pt    the 600 BEN-W keys (prompt-only key at the last prompt position = the key cache's decode key; float32), the mean key mu,",
         "                         the Gram matrix K K' of the uncentred second moment C_W = K'K / N, tr(C_W), and per kappa in {0.01, 0.1, 1}: rho = kappa tr(C_W) / d",
         "                         and L^-1 with L L' = K K' + N rho I, from which M = (C_W + rho I)^-1 = (I - R'R) / rho, R = L^-1 K (Woodbury; no d x d inverse)",
         "  <backbone>/meta.json   provenance (md5 of the parts file, the pool, the key file), numbers, sha1 of every tensor, md5 of stats.pt",
         "  <backbone>/README.txt  the numbers in words",
         "  Reader: white.wstats.load(backbone) -> KeyStats.  One statistics source only (BEN-W); the files carry no time stamp and are reproduced byte for",
         "  byte by a re-run, whatever the number of BLAS threads (python -m white.wstats check).",
         "",
         "  backbone   layer   N     d       tr(C_W)     |mu|     rho (kappa 0.01 / 0.1 / 1)                 stats.pt md5"]
    for bb in sorted(os.listdir(root)):
        mp = os.path.join(root, bb, "meta.json")
        if not os.path.exists(mp):
            continue
        with open(mp) as f:
            m = json.load(f)
        L.append(f"  {bb:<9}  {m['layer']:<5}   {m['n']:<4}  {m['d']:<6}  {m['trace_C_W']:<10.4f}  {m['norm_mu']:<7.4f}  "
                 + " / ".join(f"{m['rho'][kappa_tag(k)]:.4e}" for k in KAPPAS) + f"   {m['stats_pt_md5']}")
    return "\n".join(L) + "\n"


_LOADED = {}


def load(backbone: str, verify: bool = True, directory: str | None = None) -> KeyStats:
    """The stored BEN-W statistics of a backbone (cached per process).  verify: md5 of stats.pt against meta.json, and the stored
    keys against the keys of the BEN-W rows in the key cache as it is now (bit for bit); raises on any difference."""
    key = (backbone, directory)
    if key in _LOADED:
        return _LOADED[key]
    d = out_dir(backbone, directory)
    with open(os.path.join(d, "meta.json")) as f:
        meta = json.load(f)
    p = os.path.join(d, "stats.pt")
    if verify and C.md5_file(p) != meta["stats_pt_md5"]:
        raise RuntimeError(f"{p} is not the file that meta.json describes (rebuild: python -m white.wstats build)")
    blob = torch.load(p, map_location="cpu")
    if blob["backbone"] != backbone or tuple(blob["keys"].shape) != (meta["n"], meta["d"]):
        raise RuntimeError(f"{p}: statistics of another backbone or shape")
    if verify:
        sids, key_rows, keys, _, _ = benw_keys(backbone)
        if sids != blob["sids"] or not torch.equal(keys, blob["keys"]):
            raise RuntimeError(f"{backbone}: the BEN-W keys of the key cache are not the keys the stored statistics were computed from")
    st = KeyStats(blob["keys"], blob["mu"], blob["gram"], blob["trace"], name=f"{PART}:{backbone}", sids=blob["sids"],
                  linv={t: v["linv"] for t, v in blob["kappa"].items()}, meta=meta)
    _LOADED[key] = st
    return st


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="statistics of the BEN-W keys (R3)")
    ap.add_argument("cmd", choices=("build", "check", "show"))
    ap.add_argument("backbones", nargs="*")
    ap.add_argument("--dir", default=None, help="write / read another directory (tests)")
    a = ap.parse_args(argv)
    bbs = a.backbones or list(C.BACKBONES)
    ok = True
    for bb in bbs:
        if a.cmd == "build":
            meta = write(bb, a.dir)
            print(f"wrote {out_dir(bb, a.dir)}: stats.pt md5 {meta['stats_pt_md5']}")
            print(readme(meta))
        elif a.cmd == "check":                                             # byte-for-byte reproduction of the stored files
            blob, meta = build(bb)
            data = _save_bytes(blob)
            meta["stats_pt_md5"], meta["stats_pt_bytes"] = C.md5_bytes(data), len(data)
            d = out_dir(bb, a.dir)
            with open(os.path.join(d, "stats.pt"), "rb") as f:
                same_pt = f.read() == data
            with open(os.path.join(d, "meta.json"), "rb") as f:
                same_meta = f.read() == C.jdumps(meta).encode("utf-8")
            load(bb, True, a.dir)
            ok &= same_pt and same_meta
            print(f"{bb}: stats.pt {'reproduced byte for byte' if same_pt else 'DIFFERS'}; meta.json {'reproduced byte for byte' if same_meta else 'DIFFERS'}; "
                  f"threads {torch.get_num_threads()}")
        else:
            with open(os.path.join(out_dir(bb, a.dir), "README.txt")) as f:
                print(f.read())
    if a.cmd == "build":
        C.write_atomic(os.path.join(C.WHITEN_DIR if a.dir is None else a.dir, "README.txt"), root_readme(a.dir))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
