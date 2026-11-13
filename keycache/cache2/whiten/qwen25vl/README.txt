10_final/cache2/whiten/qwen25vl -- statistics of the BEN-W keys of backbone qwen25vl (layer 20)
Written by `python -m white.wstats build` (code and reference text: 10_final/feat2/white/wstats.py).  PROTOCOL.md section 6, amendment
'Review round 1', item R3.  No time stamp: a re-run reproduces stats.pt and meta.json byte for byte (feat2/white/tests/t_stats.py).

keys      600 BEN-W rows (309 image, 291 text), d = 11008: prompt-only key at the last prompt position of every BEN-W row = the key cache's decode key (keys_decode.pt[row_decode]), float32 image of the module dtype
          key norm min / median / max = 10.790 / 12.514 / 15.584
mu        mean key, |mu| = 9.2169; |mu|^2 / tr(C_W) = 0.5337
C_W       uncentred second moment (1/N) sum k k', kept as its N x N Gram matrix K K'; tr(C_W) = 159.174175, tr / d = 1.445986e-02
          eigenvalue shares of the trace (top 5): 0.5377, 0.1837, 0.0501, 0.0181, 0.0126; participation ratio 3.06; lambda_min / lambda_max = 7.60e-06
M         (C_W + rho I)^-1 = (I - R'R) / rho, R = L^-1 K, L L' = K K' + N rho I (Woodbury on the Gram matrix; no d x d inverse)
          kappa      rho          N rho     effective dof   eigenvalues > rho   cond(KK' + N rho I)   cond(M)    mean x M x' (BEN-W keys)   cancellation there
          0.01       1.4460e-04   8.6759e-02        585.9          600 of 600        1.076e+05   5.919e+05            585.9               1878.9
          0.1        1.4460e-03   8.6759e-01        502.3          574 of 600        4.082e+04   5.919e+04            502.3                219.1
          1          1.4460e-02   8.6759e+00        284.7          261 of 600        5.665e+03   5.920e+03            284.7                 38.7
          effective dof = sum of lambda / (lambda + rho) over the eigenvalues of C_W = mean of x M x' over the BEN-W keys themselves;
          cancellation = |x|^2 / (rho x M x'): the digits that x M x' = (|x|^2 - |x R'|^2) / rho loses (float64 is required).

files     stats.pt   torch.load -> dict: backbone, part, which, sids, key_rows, keys (N, d) float32, mu (d,) float64, gram (N, N) float64,
                     trace, n, d, kappa: {'0.01' | '0.1' | '1': {kappa, rho, linv (N, N) float64 = L^-1}}
                     md5 b3c8ea266c4f19e8b842ea2ee347f623, 38046997 bytes
          meta.json  provenance, numbers above, sha1 of every tensor
reader    white.wstats.load(backbone) -> KeyStats (mu, keys, gram, trace, rho(kappa), whitener(kappa)); see feat2/white/USAGE.txt
sources   cache2/benign_parts/qwen25vl.json (md5 bf7dbbaa25338e3fbd8990c07c67d008); cache/qwen25vl/keys_decode.pt (md5 9346386a801eff227b6f2a4abe4cddfe, verified at build time)
