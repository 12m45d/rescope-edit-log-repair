# Harness smoke: every method on one log through the shared harness

Rendered by `harness/smoke/render.py` from `results.json` (aggregate.py over rows.jsonl written by run_smoke.py). Do not edit by hand.

Log `TB-Q100-ae-l2_3`; complaints: 10 random D1 v2 complaints from the REPAIR pool (sampler seed 0, stratified image/text): image_111, image_125, image_226, image_281, image_438, text_240, text_281, text_295, text_402, text_459. Collateral is measured on the HELD-OUT pool only. One repair per complaint (batch config: one joint repair of all complaints).

## 1. Outcome

| config | method | D1 resolved | 95% CI (uid bootstrap) | img / txt | D2 res / n | edits broken (rel+tgen+igen) mean [CI] | rel broken | edits fixed | held-out non-complaints lost (of n) | held-out complaints fixed (of n) | dT-Loc | dI-Loc | median ||corr||_F (/||dW_E||) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C01 | NONE | 0/10 | [0.00, 0.00] | 0/5 / 0/5 | 0/1 | 0.00 [0.00, 0.00] | 0.00 | 0.00 | 0.00 (222) | 0.00 (268) | 0.00 | 0.00 | 0.000 (0.0000) |
| C02 | COMP[K_last|answer|adapter] | 7/10 | [0.44, 1.00] | 4/5 / 3/5 | 1/1 | 1.20 [0.56, 1.73] | 1.10 | 1.30 | 24.40 (222) | 23.80 (268) | 0.17 | 0.24 | 6.871 (0.0259) |
| C03 | COMP[K_last|answer|merged] | 7/10 | [0.44, 1.00] | 4/5 / 3/5 | 1/1 | 1.00 [0.44, 1.67] | 0.90 | 1.50 | 24.40 (222) | 24.80 (268) | 0.13 | 0.27 | 6.888 (0.0260) |
| C04 | COMP[K_last|all|adapter] | 10/10 | [1.00, 1.00] | 5/5 / 5/5 | 1/1 | 2.40 [1.89, 2.90] | 1.80 | 2.90 | 38.20 (222) | 27.60 (268) | -0.05 | -1.04 | 25.158 (0.0948) |
| C05 | COMP[K_ans|answer|adapter] | 7/10 | [0.44, 1.00] | 4/5 / 3/5 | 1/1 | 0.70 [0.22, 1.10] | 0.60 | 0.70 | 21.50 (222) | 22.00 (268) | 0.10 | 0.20 | 7.179 (0.0270) |
| C06 | COMP[K_seq:10|answer|adapter] | 7/10 | [0.44, 1.00] | 4/5 / 3/5 | 1/1 | 2.20 [0.44, 4.17] | 1.50 | 1.50 | 26.00 (222) | 25.70 (268) | 0.26 | 0.51 | 8.079 (0.0304) |
| C07 | COMP[K_last+probe:200|answer|adapter] | 9/10 | [0.67, 1.00] | 4/5 / 5/5 | 1/1 | 0.80 [0.33, 1.30] | 0.80 | 1.40 | 11.50 (222) | 12.00 (268) | 0.11 | 0.33 | 20.628 (0.0777) |
| C08 | COMP[none|answer|adapter] | 7/10 | [0.44, 1.00] | 4/5 / 3/5 | 1/1 | 16.40 [8.00, 25.42] | 11.70 | 3.20 | 49.20 (222) | 34.80 (268) | 0.07 | -1.17 | 6.150 (0.0232) |
| C09 | COMP[K_last|answer|adapter|batch] | 8/10 | [0.58, 1.00] | 4/5 / 4/5 | 1/1 | 4.00 [4.00, 4.00] | 2.00 | 1.00 | 29.00 (222) | 56.00 (268) | 2.96 | 1.72 | 26.652 (0.1004) |
| C10 | REEDIT[variant=k0+kp] | 9/10 | [0.75, 1.00] | 5/5 / 4/5 | 1/1 | 0.50 [0.00, 1.08] | 0.40 | 0.60 | 35.10 (222) | 16.30 (268) | -2.69 | -0.06 | 17.970 (0.0677) |
| C11 | ROLLBACK_FO[budget=10] | 8/10 | [0.50, 1.00] | 5/5 / 3/5 | 1/1 | 6.80 [4.33, 9.70] | 4.80 | 2.20 | 20.70 (222) | 25.40 (268) | 1.10 | 0.18 | 57.743 (0.2175) |
| C12 | ROLLBACK_FO[budget=20] | 10/10 | [1.00, 1.00] | 5/5 / 5/5 | 1/1 | 7.10 [4.33, 10.56] | 5.20 | 2.10 | 21.20 (222) | 26.40 (268) | 1.25 | 0.09 | 57.743 (0.2175) |
| C13 | ROLLBACK_BEH[budget=20] | 10/10 | [1.00, 1.00] | 5/5 / 5/5 | 1/1 | 6.50 [3.80, 9.64] | 4.80 | 2.60 | 18.80 (222) | 25.40 (268) | 0.79 | 0.39 | 49.947 (0.1882) |
| C14 | ROLLBACK_ALL | 10/10 | [1.00, 1.00] | 5/5 / 5/5 | 1/1 | 108.00 [108.00, 108.00] | 78.00 | 0.00 | 0.00 (222) | 268.00 (268) | 15.17 | 11.31 | 265.439 (1.0000) |

Edits: all logged edits x {rel, T-Gen (rephrase), I-Gen (image rephrase)}, teacher-forced exact match; broken = exact at W_E and not after. Baseline at W_E: rel exact 78.00, tgen exact 9.00, igen exact 22.00. Loc at W_E (held-out items): T-Loc 84.83, I-Loc 88.69.

## 2. Cost per repair (mean over repairs; setup excluded for every method)

| config | full fwd | prefix fwd (0..L) | zopt fwd+bwd | suffix fwd (L+1..N) | suffix fwd+bwd | # backward | FLOPs est. (model + solve) | solve FLOPs | wall-clock mean / median (s) | solve (s, median) | setup excluded (s) | extra |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C01 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.50e-05 / 1.28e-05 | - | 0.00 |  |
| C02 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.98e+12 | 4.97e+08 | 0.074 / 0.059 | 0.0083 | 1.00 | protect rank 100; infeasible 0; dropped dirs 0; P1 max 3.68e-15 |
| C03 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.98e+12 | 9.43e+08 | 0.134 / 0.103 | 0.0235 | 1.00 | protect rank 100; infeasible 0; dropped dirs 0; P1 max 3.68e-15 |
| C04 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 2.01e+12 | 2.39e+10 | 0.320 / 0.208 | 0.1335 | 1.00 | protect rank 100; infeasible 0; dropped dirs 0; P1 max 1.32e-14 |
| C05 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.98e+12 | 6.28e+08 | 0.069 / 0.043 | 0.0083 | 0.22 | protect rank 400; infeasible 0; dropped dirs 0; P1 max 3.35e-15 |
| C06 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.98e+12 | 1.68e+09 | 0.079 / 0.054 | 0.0098 | 14.78 | protect rank 2824; infeasible 0; dropped dirs 0; P1 max 3.66e-15 |
| C07 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.98e+12 | 1.31e+09 | 0.121 / 0.069 | 0.0095 | 22.80 | protect rank 1954; infeasible 0; dropped dirs 0; P1 max 3.36e-15 |
| C08 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.98e+12 | 4.53e+08 | 0.089 / 0.043 | 0.0059 | 0.00 | protect rank 0; infeasible 0; dropped dirs 0; P1 max 3.35e-15 |
| C09 | 0.00 | 10.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.97e+13 | 5.60e+09 | 0.613 / 0.613 | 0.0360 | 6.67 | protect rank 100; infeasible 0; dropped dirs 0; P1 max 2.47e-15 |
| C10 | 1.00 | 2.00 | 9.10 | 0.00 | 0.00 | 9.10 | 4.82e+13 | 7.61e+08 | 1.348 / 0.876 | 0.0067 | 1.35 | zopt attempts [[1], [1], [1], [1], [1], [1], [1], [1], [1], [1]]; not installed 0 |
| C11 | 1.00 | 0.00 | 0.00 | 4.80 | 4.80 | 4.80 | 1.08e+13 | 4.33e+10 | 0.452 / 0.291 | - | 0.00 | k rolled back mean 4.80 [5, 4, 4, 2, 3, 2, 10, 10, 4, 4]; internal==eval 10/10 |
| C12 | 1.00 | 0.00 | 0.00 | 5.60 | 5.60 | 5.60 | 1.10e+13 | 5.05e+10 | 0.352 / 0.287 | - | 0.00 | k rolled back mean 5.60 [5, 4, 4, 2, 3, 2, 14, 14, 4, 4]; internal==eval 10/10 |
| C13 | 1.00 | 0.00 | 0.00 | 463.20 | 0.00 | 0.00 | 1.89e+14 | 0.00 | 5.083 / 3.763 | - | 0.00 | k rolled back mean 4.80 [4, 3, 4, 2, 3, 2, 12, 13, 2, 3]; internal==eval 10/10 |
| C14 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.92e-04 / 1.43e-04 | - | 0.00 |  |

Clean wall-clock (`smoke/timing.py`: repairs only, the same 10 complaints, one NVIDIA GeForce RTX 5090, 1 process(es) on it, methods interleaved, warm-up and per-log setup excluded, 8 CPU threads):

| config | median s | p10 / p90 s | solve median s |
|---|---|---|---|
| C02 COMP[K_last|answer|adapter] | 0.045 | 0.023 / 0.119 | 0.0082 |
| C03 COMP[K_last|answer|merged] | 0.043 | 0.024 / 0.092 | 0.0092 |
| C04 COMP[K_last|all|adapter] | 0.120 | 0.025 / 0.341 | 0.0844 |
| C05 COMP[K_ans|answer|adapter] | 0.042 | 0.023 / 0.090 | 0.0083 |
| C06 COMP[K_seq:10|answer|adapter] | 0.044 | 0.024 / 0.092 | 0.0097 |
| C07 COMP[K_last+probe:200|answer|adapter] | 0.042 | 0.023 / 0.091 | 0.0092 |
| C08 COMP[none|answer|adapter] | 0.040 | 0.020 / 0.088 | 0.0058 |
| C10 REEDIT[variant=k0+kp] | 0.615 | 0.435 / 1.529 | 0.0023 |
| C11 ROLLBACK_FO[budget=10] | 0.252 | 0.191 / 0.461 | - |
| C12 ROLLBACK_FO[budget=20] | 0.251 | 0.200 / 0.635 | - |
| C13 ROLLBACK_BEH[budget=20] | 3.718 | 2.022 / 11.565 | - |
| C14 ROLLBACK_ALL | 6.59e-05 | 6.50e-05 / 6.82e-05 | - |

The wall-clock columns of the table above come from the smoke itself (two GPUs, other jobs running): use the clean timing for cost claims.

Smoke rows ran on two NVIDIA GeForce RTX 5090s (configs split across them, other jobs present). FLOPs are an estimate (2 x params per token per block + attention, ViT, LM head; backward = 2x the forward of the blocks above L). Suffix passes are exact (bit-identical to full forwards).

## 3. Paired comparison with C02 (McNemar, exact, D1 resolution on the same complaints)

| config | both | C02 only | other only | neither | p |
|---|---|---|---|---|---|
| C01 NONE | 0 | 7 | 0 | 3 | 0.016 |
| C03 COMP[K_last|answer|merged] | 7 | 0 | 0 | 3 | 1.000 |
| C04 COMP[K_last|all|adapter] | 7 | 0 | 3 | 0 | 0.250 |
| C05 COMP[K_ans|answer|adapter] | 7 | 0 | 0 | 3 | 1.000 |
| C06 COMP[K_seq:10|answer|adapter] | 7 | 0 | 0 | 3 | 1.000 |
| C07 COMP[K_last+probe:200|answer|adapter] | 7 | 0 | 2 | 1 | 0.500 |
| C08 COMP[none|answer|adapter] | 7 | 0 | 0 | 3 | 1.000 |
| C10 REEDIT[variant=k0+kp] | 7 | 0 | 2 | 1 | 0.500 |
| C14 ROLLBACK_ALL | 7 | 0 | 3 | 0 | 0.250 |
| C11 ROLLBACK_FO[budget=10] | 5 | 2 | 3 | 0 | 1.000 |
| C12 ROLLBACK_FO[budget=20] | 7 | 0 | 3 | 0 | 0.250 |
| C13 ROLLBACK_BEH[budget=20] | 7 | 0 | 3 | 0 | 0.250 |

n = 10 per method: a smoke of the harness, not a result; CIs are wide by construction.
