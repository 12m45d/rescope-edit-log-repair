10_final/cache2/whiten -- statistics of the BEN-W keys, per backbone (PROTOCOL.md section 6, amendment 'Review round 1', item R3)
Written by `python -m white.wstats build` (cwd = 10_final/feat2; code and reference text: feat2/white/wstats.py; usage: feat2/white/USAGE.txt).

  <backbone>/stats.pt    the 600 BEN-W keys (prompt-only key at the last prompt position = the key cache's decode key; float32), the mean key mu,
                         the Gram matrix K K' of the uncentred second moment C_W = K'K / N, tr(C_W), and per kappa in {0.01, 0.1, 1}: rho = kappa tr(C_W) / d
                         and L^-1 with L L' = K K' + N rho I, from which M = (C_W + rho I)^-1 = (I - R'R) / rho, R = L^-1 K (Woodbury; no d x d inverse)
  <backbone>/meta.json   provenance (md5 of the parts file, the pool, the key file), numbers, sha1 of every tensor, md5 of stats.pt
  <backbone>/README.txt  the numbers in words
  Reader: white.wstats.load(backbone) -> KeyStats.  One statistics source only (BEN-W); the files carry no time stamp and are reproduced byte for
  byte by a re-run, whatever the number of BLAS threads (python -m white.wstats check).

  backbone   layer   N     d       tr(C_W)     |mu|     rho (kappa 0.01 / 0.1 / 1)                 stats.pt md5
  llava      16      600   11008   109.7228    5.6091   9.9675e-05 / 9.9675e-04 / 9.9675e-03   0ea4b86a6a543a08df1a7a55ee4f987d
  qwen25vl   20      600   11008   159.1742    9.2169   1.4460e-04 / 1.4460e-03 / 1.4460e-02   b3c8ea266c4f19e8b842ea2ee347f623
