10_final/cache2/whiten/llava -- statistics of the BEN-W keys of backbone llava (layer 16)
Written by `python -m white.wstats build` (code and reference text: 10_final/feat2/white/wstats.py).  PROTOCOL.md section 6, amendment
'Review round 1', item R3.  No time stamp: a re-run reproduces stats.pt and meta.json byte for byte (feat2/white/tests/t_stats.py).

keys      600 BEN-W rows (309 image, 291 text), d = 11008: prompt-only key at the last prompt position of every BEN-W row = the key cache's decode key (keys_decode.pt[row_decode]), float32 image of the module dtype
          key norm min / median / max = 8.074 / 10.340 / 13.971
mu        mean key, |mu| = 5.6091; |mu|^2 / tr(C_W) = 0.2867
C_W       uncentred second moment (1/N) sum k k', kept as its N x N Gram matrix K K'; tr(C_W) = 109.722768, tr / d = 9.967548e-03
          eigenvalue shares of the trace (top 5): 0.3089, 0.1059, 0.0548, 0.0366, 0.0232; participation ratio 8.79; lambda_min / lambda_max = 1.02e-06
M         (C_W + rho I)^-1 = (I - R'R) / rho, R = L^-1 K, L L' = K K' + N rho I (Woodbury on the Gram matrix; no d x d inverse)
          kappa      rho          N rho     effective dof   eigenvalues > rho   cond(KK' + N rho I)   cond(M)    mean x M x' (BEN-W keys)   cancellation there
          0.01       9.9675e-05   5.9805e-02        593.0          599 of 600        2.522e+05   3.401e+05            593.0               1856.4
          0.1        9.9675e-04   5.9805e-01        552.7          588 of 600        3.286e+04   3.401e+04            552.7                199.2
          1          9.9675e-03   5.9805e+00        409.7          457 of 600        3.390e+03   3.402e+03            409.7                 26.9
          effective dof = sum of lambda / (lambda + rho) over the eigenvalues of C_W = mean of x M x' over the BEN-W keys themselves;
          cancellation = |x|^2 / (rho x M x'): the digits that x M x' = (|x|^2 - |x R'|^2) / rho loses (float64 is required).

files     stats.pt   torch.load -> dict: backbone, part, which, sids, key_rows, keys (N, d) float32, mu (d,) float64, gram (N, N) float64,
                     trace, n, d, kappa: {'0.01' | '0.1' | '1': {kappa, rho, linv (N, N) float64 = L^-1}}
                     md5 0ea4b86a6a543a08df1a7a55ee4f987d, 38046997 bytes
          meta.json  provenance, numbers above, sha1 of every tensor
reader    white.wstats.load(backbone) -> KeyStats (mu, keys, gram, trace, rho(kappa), whitener(kappa)); see feat2/white/USAGE.txt
sources   cache2/benign_parts/llava.json (md5 d385e2d55e842d5da568874675d3c23d); cache/llava/keys_decode.pt (md5 973c1097b3adc6c8a95a951003ad5e18, verified at build time)
