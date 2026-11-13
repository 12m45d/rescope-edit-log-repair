"""Own gate arithmetic (no driver function, no library calibration): normalised solver coordinates, guard caps, two-block rule."""
from vcommon import *
from math import ceil
from fractions import Fraction

def ncoord(log, K, chunk=2048):
    """r[q, i] = <W_i, k_q> / <W_i, k_i>  (fp64); K: (n, d) keys (float32 or float64)."""
    W = log.W.double(); s = (W * log.K.double()).sum(1)
    out = []
    for a in range(0, K.shape[0], chunk):
        out.append((K[a:a + chunk].double() @ W.T) / s[None, :])
    return torch.cat(out).numpy()

def keys_of(kc, rows, which):
    """float32 keys of index rows: 'tf' -> row_tf of keys_tf, 'decode' -> row_decode of keys_decode (prompt-only)."""
    f = "row_tf" if which == "tf" else "row_decode"
    idx = torch.tensor([int(kc.index[j][f]) for j in rows], dtype=torch.long)
    return kc.K("tf" if which == "tf" else "decode")[idx]

def guard_caps(L):
    """cap_i = (1 - 1e-9) x smallest coordinate of edit i on its guard keys: the stored solver key (r = 1 by definition) and the
    teacher-forced B = 1 key of its own Rel prompt."""
    rows = L.edit_rows["rel"]
    R = ncoord(L.log, keys_of(L.kc, rows, "tf"))
    own_b1 = R[np.arange(L.T), np.arange(L.T)]
    own_st = ncoord(L.log, L.log.K)[np.arange(L.T), np.arange(L.T)]
    return (1 - 1e-9) * np.minimum(own_b1, own_st), own_b1

def conformal_rank(n2, alpha_frac):
    """k = ceil((1 - alpha)(n2 + 1)) in exact arithmetic."""
    return ceil((1 - alpha_frac) * (n2 + 1))

def two_block(Q, perm, alpha_frac=Fraction(1, 20), rank=None):
    """Q (n, T) calibration coordinates in the order given; perm: permutation of range(n), first half = block 1.
    -> b (T,), eta, k"""
    n = Q.shape[0]; n1 = n // 2
    i1, i2 = perm[:n1], perm[n1:]
    b = Q[i1].max(0)
    sc = np.sort((Q[i2] - b[None, :]).max(1))
    n2 = len(i2)
    k = conformal_rank(n2, alpha_frac) if rank is None else rank
    eta = float("inf") if k > n2 else float(sc[k - 1])
    return b, eta, k

def fire(R, b, eta, cap=None):
    """literal firing (strict) and firing with the guard caps: a detector whose literal threshold b_j + eta exceeds its cap is capped."""
    lit = (R - b[None, :]).max(1) > eta
    if cap is None:
        return lit, lit, 0
    capped = (cap - b) <= eta                # the literal rule would silence the guard key
    g = lit | (R[:, capped] >= cap[None, capped]).any(1) if capped.any() else lit
    return lit, g, int(capped.sum())
