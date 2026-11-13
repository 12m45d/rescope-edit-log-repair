# P12 (linear repair streams) -- independent verification

Rendered by `verify/render_report.py` from `verify/out/*.json` (GPU: `gpu_verify.py`; CPU: `cpu_checks.py`, `cpu_extra.py`, `cpu_moddiag.py`, `leak_pixels.py`, `sandbox_render.sh`). Every number is read from those files. The experiment's files were not modified.

## 1. Verdict per headline

| headline | verdict | basis |
|---|---|---|
| Sanity: NONE reproduces the stored D1 sets and Rel; W_0 gives zero D1 | **CONFIRMED** | own loop: Qwen, LLaVA row for row |
| Unrepaired baselines and the two seed-0 tables | **CONFIRMED** | 44/44 cells equal the raw records; 10/10 recomputed conditions agree row for row |
| Best operating point = COMP-B at n = ALL on both logs, with its numbers | **CONFIRMED** | selection re-derived from the raw records; correction rebuilt and re-evaluated on: Qwen, LLaVA |
| Generalisation reading (COMP-B fixes and breaks at similar rates; COMP-B+ worse at ALL; REEDIT net negative at every n; rollback only by removing edits) | **CONFIRMED (numbers)** | rates and nets re-derived from the raw records; the reading is single-seed and has no same-norm control |
| Anomalies: split leak; harness z* fallback bug; sequential = batch; answer-position compensation; ALLPOS | **CONFIRMED** | twins recounted from the records (also by pixel content); bug located in the source; operators and weights compared from the saved corrections |
| Why (LLaVA, module level): rank, share, residual change; 'the damage travels through the edits' own key directions' | **CONFIRMED (numbers); reading supported only by my extra run** | the three numbers reproduce (my own key capture (GPU)); my behavioural run of that very weight (W_0 restored outside the span; no experiment record, one run): HELD-OUT D1 81.0% at Rel exact 89, against 61.5% unrepaired -- it does not lower D1 |
| Not run: D2, seeds 1-2, text head-to-head, Qwen REEDIT re-runs | **CONFIRMED** | no such record exists; every record has seed 0 |

## 2. What was recomputed, and how

- Own evaluation loop (`gpu_verify.py`): pools and the nested stream rebuilt with numpy from `probe_records_v2.jsonl` (`vcommon.my_split`, `my_stream`); B=1 batches built from the record fields and from the log's `edit_batch.jsonl`; one full teacher-forced forward per row; the correction applied by my own forward hook (fp64 adapter, gate 1e-8) or by writing the replacement weight. No call of `harness.evaluate`, `harness.baseline`, the suffix engine or `H.applied`.
- Corrections rebuilt through `harness.repair` with the stated options and an empty private cache (edit keys, protection bases, z* targets recomputed with fresh forwards).
- The two further cells per log were drawn with a fixed seed before any GPU number of mine existed (`draw_cells.py`, seed [2026, 11, 10, 12], 21 cells): Qwen: REEDIT-B n=100, RB-TOT n=100; LLaVA: COMP-B n=10, REEDIT-B n=1.

## 3. Table

| quantity | experiment | verifier | agree? |
|---|---|---|---|
| Qwen NONE (unrepaired): HELD-OUT D1 % (+fixed / -lost) | 54.7 (268/490) | 54.7 (268/490; +0 / -0) | yes, 490/490 rows |
| Qwen NONE (unrepaired): Rel exact | 78 | 78 | yes, 100/100 edits |
| Qwen ALL (W_0): HELD-OUT D1 % (+fixed / -lost) | 0.0 (0/490) | 0.0 (0/490; +268 / -0) | yes, 490/490 rows |
| Qwen ALL (W_0): Rel exact | 0 | 0 | yes, 100/100 edits |
| Qwen best point COMP-B n=ALL: HELD-OUT D1 % (+fixed / -lost) | 47.3 (232/490) | 47.3 (232/490; +97 / -61) | yes, 490/490 rows |
| Qwen best point COMP-B n=ALL: Rel exact | 78 | 78 | yes, 100/100 edits |
| Qwen best point COMP-B n=ALL: reported resolved | 168/297 | 168/297 | yes, 297/297 rows |
| Qwen best point COMP-B n=ALL: rebuilt operator vs the experiment's saved operator | rank 2841, norm/log 0.647 | rank 2841, norm/log 0.647; rel. Frobenius diff 0.0e+00 | yes |
| Qwen random cell REEDIT-B n=100 (first-pass record), experiment's z* re-installed by me: HELD-OUT D1 % (+fixed / -lost) | 58.8 (288/490) (first-pass record) | 58.8 (288/490; +51 / -71) | yes, 490/490 rows |
| Qwen random cell REEDIT-B n=100 (first-pass record), experiment's z* re-installed by me: Rel exact | 71 | 71 | yes, 100/100 edits |
| Qwen random cell REEDIT-B n=100 (first-pass record), experiment's z* re-installed by me: reported resolved | 93/100 | 93/100 | yes, 100/100 rows |
| Qwen random cell REEDIT-B n=100 (first-pass record), experiment's z* re-installed by me: rebuilt weight vs the experiment's saved weight | norm/log 0.718 | norm/log 0.718; 0 of 22544384 entries differ (rel. diff 0.0e+00) | yes |
| Qwen random cell REEDIT-B n=100 (first-pass record), my own z*, same installed set: HELD-OUT D1 % (+fixed / -lost) | 58.8 (288/490) (first-pass record) | 59.4 (291/490; +50 / -73) | differs only through z*: 479/490 rows equal, D1 0.6 points apart |
| Qwen random cell REEDIT-B n=100 (first-pass record), my own z*, same installed set: Rel exact | 71 | 72 | 99/100 edits equal |
| Qwen random cell REEDIT-B n=100 (first-pass record), my own z*, same installed set: reported resolved | 93/100 | 93/100 | 100/100 rows |
| Qwen random cell REEDIT-B n=100 (first-pass record), my own z*, same installed set: rebuilt weight vs the experiment's saved weight | norm/log 0.718 | norm/log 0.717; 21412708 of 22544384 entries differ (rel. diff 9.3e-02) | z* noise |
| Qwen random cell REEDIT-B n=100, stated protocol (my z*, fallback ladder; the experiment has no such record): HELD-OUT D1 % (+fixed / -lost) | no record | 58.4 (286/490; +53 / -71) | - |
| Qwen random cell REEDIT-B n=100, stated protocol (my z*, fallback ladder; the experiment has no such record): Rel exact | no record | 72 | - |
| Qwen random cell REEDIT-B n=100, stated protocol (my z*, fallback ladder; the experiment has no such record): reported resolved / installed / norm over log | no record | 93/100 / 100 / 0.725 | - |
| Qwen random cell RB-TOT n=100: HELD-OUT D1 % (+fixed / -lost) | 42.2 (207/490) | 42.2 (207/490; +92 / -31) | yes, 490/490 rows |
| Qwen random cell RB-TOT n=100: Rel exact | 53 | 53 | yes, 100/100 edits |
| Qwen random cell RB-TOT n=100: reported resolved | 47/100 | 47/100 | yes, 100/100 rows |
| Qwen random cell RB-TOT n=100: rebuilt weight vs the experiment's saved weight | norm/log 0.582 | norm/log 0.582; 0 of 22544384 entries differ (rel. diff 0.0e+00) | yes |
| Qwen sanity: NONE vs stored v2 flags, rows differing (HELD-OUT / REPAIR) | 0 / 0 | 0 / 0 (stored D1 268 / 297) | yes |
| Qwen sanity: D1 complaints at W_0 (HELD-OUT / REPAIR); weight == checkpoint | 0 / 0; True | 0 / 0; True (W_0 == checkpoint file: ['model.layers.20.mlp.down_proj.weight', True]) | yes |
| Qwen sanity: stored Rel exact / token on the testbed's batch-8 path | 78.0 / 91.58 | 78.0 / 91.58 | yes |
| Qwen best point: 95% CI / exact McNemar p (full pool) | [43.1, 51.8] / 0.0052 | [43.1, 51.8] / 0.0052 (my flags, my statistics) | yes |
| Qwen best point on the non-duplicated rows: D1 % vs unrepaired (p) | 49.0 vs 54.2 (0.057) | 49.0 vs 54.2 (0.057; 465 rows) | yes |
| LLaVA NONE (unrepaired): HELD-OUT D1 % (+fixed / -lost) | 61.5 (304/494) | 61.5 (304/494; +0 / -0) | yes, 494/494 rows |
| LLaVA NONE (unrepaired): Rel exact | 88 | 88 | yes, 100/100 edits |
| LLaVA ALL (W_0): HELD-OUT D1 % (+fixed / -lost) | 0.0 (0/494) | 0.0 (0/494; +304 / -0) | yes, 494/494 rows |
| LLaVA ALL (W_0): Rel exact | 1 | 1 | yes, 100/100 edits |
| LLaVA best point COMP-B n=ALL: HELD-OUT D1 % (+fixed / -lost) | 58.5 (289/494) | 58.5 (289/494; +72 / -57) | yes, 494/494 rows |
| LLaVA best point COMP-B n=ALL: Rel exact | 86 | 86 | yes, 100/100 edits |
| LLaVA best point COMP-B n=ALL: reported resolved | 164/305 | 164/305 | yes, 305/305 rows |
| LLaVA best point COMP-B n=ALL: rebuilt operator vs the experiment's saved operator | rank 1973, norm/log 0.461 | rank 1973, norm/log 0.461; rel. Frobenius diff 0.0e+00 | yes |
| LLaVA random cell COMP-B n=10: HELD-OUT D1 % (+fixed / -lost) | 61.5 (304/494) | 61.5 (304/494; +19 / -19) | yes, 494/494 rows |
| LLaVA random cell COMP-B n=10: Rel exact | 88 | 88 | yes, 100/100 edits |
| LLaVA random cell COMP-B n=10: reported resolved | 7/10 | 7/10 | yes, 10/10 rows |
| LLaVA random cell COMP-B n=10: rebuilt operator vs the experiment's saved operator | rank 57, norm/log 0.066 | rank 57, norm/log 0.066; rel. Frobenius diff 0.0e+00 | yes |
| LLaVA random cell REEDIT-B n=1, experiment's z* re-installed by me: HELD-OUT D1 % (+fixed / -lost) | 65.0 (321/494) | 65.0 (321/494; +20 / -37) | yes, 494/494 rows |
| LLaVA random cell REEDIT-B n=1, experiment's z* re-installed by me: Rel exact | 88 | 88 | yes, 100/100 edits |
| LLaVA random cell REEDIT-B n=1, experiment's z* re-installed by me: reported resolved | 0/1 | 0/1 | yes, 1/1 rows |
| LLaVA random cell REEDIT-B n=1, my own z*: HELD-OUT D1 % (+fixed / -lost) | 65.0 (321/494) | 65.0 (321/494; +20 / -37) | yes, 494/494 rows (z* reproduced) |
| LLaVA random cell REEDIT-B n=1, my own z*: Rel exact | 88 | 88 | yes, 100/100 edits |
| LLaVA random cell REEDIT-B n=1, my own z*: reported resolved | 0/1 | 0/1 | 1/1 rows |
| LLaVA sanity: NONE vs stored v2 flags, rows differing (HELD-OUT / REPAIR) | 0 / 0 | 0 / 0 (stored D1 304 / 305) | yes |
| LLaVA sanity: D1 complaints at W_0 (HELD-OUT / REPAIR); weight == checkpoint | 0 / 0; True | 0 / 0; True (W_0 == checkpoint file: ['language_model.model.layers.16.mlp.down_proj.weight', True]) | yes |
| LLaVA sanity: stored Rel exact / token on the testbed's batch-8 path | 88.0 / 97.12 | 88.0 / 97.12 | yes |
| LLaVA best point: T-Loc / I-Loc of the held-out items, unrepaired -> repaired | 77.3 -> 77.8 / 81.5 -> 76.7 | 77.3 -> 77.8 / 81.5 -> 76.7 (probes and comparison rebuilt; vendor batch-8 forwards) | yes |
| LLaVA best point: 95% CI / exact McNemar p (full pool) | [54.2, 63.0] / 0.22 | [54.2, 63.0] / 0.22 (my flags, my statistics) | yes |
| LLaVA best point on the non-duplicated rows: D1 % vs unrepaired (p) | 59.7 vs 62.3 (0.31) | 59.7 vs 62.3 (0.31; 469 rows) | yes |
| LLaVA verifier's extra (no experiment record): behaviour of W_0 + D P_ans, i.e. W_0 restored outside the edits' answer-key span -- HELD-OUT D1 % (+fixed / -lost); Rel exact; REPAIR-pool D1 | module-level statement only (0.87 of the module change left); unrepaired 61.5, Rel 88, REPAIR D1 305/494 | 81.0 (400/494; +49 / -145); Rel 89; REPAIR D1 402/494 (+33 / -130) | - |
| LLaVA module level: rank of the edits' answer-key span / share of a held-out complaint's key norm in it / module change left if W_0 is restored outside it (medians) | 483 / 0.43 / 0.87 | 483 / 0.43 / 0.87 (my own key capture (GPU)) | yes |
| LLaVA module level, the same two medians for held-out NON-complaints (not quoted in the report's summary) | 0.64 / 1.41 | 0.64 / 1.41 | yes |
| report's two tables: cells equal to my own summary of the raw records | 44 cells | 44 agree; dagger marks correct: True | yes |
| Qwen raw records: reported set = first n of MY stream / all valid REPAIR-pool D1 complaints / held-out rows = MY valid HELD-OUT rows | 57 records | 57 / 57 / 57; nested pairs 120/120 | yes |
| Qwen results.json vs my summary of the raw records (cells, CI, McNemar) | 741 values | 0 mismatches | yes |
| Qwen split leak: valid HELD-OUT rows with a REPAIR-pool twin / of them complaints | 25 / 16 | 25 / 16 (comparing image pixels instead of file names: 25 / 16) | yes |
| LLaVA raw records: reported set = first n of MY stream / all valid REPAIR-pool D1 complaints / held-out rows = MY valid HELD-OUT rows | 47 records | 47 / 47 / 47; nested pairs 83/83 | yes |
| LLaVA results.json vs my summary of the raw records (cells, CI, McNemar) | 585 values | 0 mismatches | yes |
| LLaVA split leak: valid HELD-OUT rows with a REPAIR-pool twin / of them complaints | 25 / 12 | 25 / 12 (comparing image pixels instead of file names: 25 / 12) | yes |
| COMP-S(+) vs COMP-B(+) operators, max rel. Frobenius diff (saved corrections) | <= 3e-8 | 2.9e-08 (12 pairs) | yes |
| REEDIT-S vs REEDIT-B weights (LLaVA), max rel. diff | 1.5e-3 | 1.5e-03 (2 pairs) | yes |
| results.md = render(results.json) and results.json = aggregate(raw records): sandbox re-run of copies of the experiment's scripts, byte comparison | identical | md identical: True; json identical: True | yes |

## 4. Code reading (check 1)

- Qwen: 57 raw records. Reported set = first n of my own stream in 57; every reported row a valid REPAIR-pool D1 complaint in 57; held-out rows identical to my valid HELD-OUT rows in 57 (no REPAIR row in any: 57); stored W_E flags used as the reference in 57. Nested along n: 120/120 pairs; n // 2 image + (n - n // 2) text: True. COMP: options as stated True; answer-query key count = sum of y0 tokens True; joint solves infeasible or zero: 0; every correction non-zero: True; zero-rank steps inside the sequential variants: 9 complaints, all repeating an earlier complaint's input (same prompt, image bytes, y0): True; largest restoration error 5e-12; protection ranks [['B', 400], ['B+', 2254]]; no backward pass True. REEDIT: l2 [3.0] (the log's: 3.0), variant ['k0+kp']. Rollback budgets: RB-TOT [30] (largest k 30), RB-UNION [10] per complaint (largest per-complaint k 10).
  Budgets: RB-TOT ends at its budget with reported complaints still unresolved at n = 10, 100, ALL; RB-UNION at n = ALL: 57 of 297 per-complaint searches end at the budget, 247 are resolved individually.
  Protection keys recomputed with fresh forwards: `probe:100000` selects 192 rows, all valid REPAIR-pool non-complaints (True; equal to my list: True; HELD-OUT rows among them: 0); fresh keys equal to the cache file the experiment read: True (max abs diff 0.0); own hook capture equal on 12/12 sampled rows; fresh edit keys equal to the cached ones: True; rank K_ans 400 (experiment 400), K_ans + probe 2254 (experiment 2254).
- LLaVA: 47 raw records. Reported set = first n of my own stream in 47; every reported row a valid REPAIR-pool D1 complaint in 47; held-out rows identical to my valid HELD-OUT rows in 47 (no REPAIR row in any: 47); stored W_E flags used as the reference in 47. Nested along n: 83/83 pairs; n // 2 image + (n - n // 2) text: True. COMP: options as stated True; answer-query key count = sum of y0 tokens True; joint solves infeasible or zero: 0; every correction non-zero: True; zero-rank steps inside the sequential variants: 6 complaints, all repeating an earlier complaint's input (same prompt, image bytes, y0): True; largest restoration error 7e-10; protection ranks [['B', 483], ['B+', 1355]]; no backward pass True. REEDIT: l2 [3.0] (the log's: 3.0), variant ['k0+kp']. Rollback budgets: RB-TOT [30] (largest k 30), RB-UNION [10] per complaint (largest per-complaint k 10).
  Budgets: RB-TOT ends at its budget with reported complaints still unresolved at n = 10, 100, ALL; RB-UNION at n = ALL: 68 of 305 per-complaint searches end at the budget, 247 are resolved individually.
  Protection keys recomputed with fresh forwards: `probe:100000` selects 189 rows, all valid REPAIR-pool non-complaints (True; equal to my list: True; HELD-OUT rows among them: 0); fresh keys equal to the cache file the experiment read: True (max abs diff 0.0); own hook capture equal on 12/12 sampled rows; fresh edit keys equal to the cached ones: True; rank K_ans 483 (experiment 483), K_ans + probe 1355 (experiment 1355).

## 5. Bugs

- `07_repair_v3/harness/methods.py`, lines 653-668: the z* oracle fallback loop (line 664, indent 16) is inside the `else:` branch of line 656 (indent 12), so it runs only when `chunk` is given; the test `if chunk is None:` on line 668 is dead code. `repair("REEDIT", ...)` with the default `chunk=None` makes one vendor attempt and does not install a non-oracle complaint. The report's claim is correct; the experiment works around it with `chunk=1` (`code/run_streams.py`, `compute_target`).
- First-pass records affected (kept in `out/<log>/superseded_zfallback/`): Qwen `REEDIT_B__n100__s0` not installed ['image_263', 'text_446']; Qwen `REEDIT_B__n10__s0` not installed ['image_263']; Qwen `REEDIT_B__nALL__s0` not installed ['image_263', 'text_446', 'text_488', 'text_80']; LLaVA `REEDIT_B__n100__s0` not installed ['text_424', 'text_227', 'text_481'].
- Corrected re-runs present: Qwen: none; LLaVA: REEDIT_B__n100__s0, REEDIT_B__n2__s0, REEDIT_B__nALL__s0, REEDIT_S__n100__s0, REEDIT_S__n2__s0, REEDIT_S__nALL__s0.

## 6. Residual risks

- z* is not reproducible on image complaints, so every REEDIT number has a run-to-run component that one seed does not show. Same process, same call, three repeats per complaint (Qwen): text 3/3 bitwise equal; image 0/6 bitwise equal, relative difference of the optimised offset between repeats 0.05-0.18 (the starting state h is bitwise equal in every repeat: True). My first-attempt z* against the experiment's cached first-attempt z* (first 100 Qwen stream complaints): text 49/49 bitwise equal; image 1/49, median relative difference 0.06 (max 0.43), a different number of optimisation steps in 7. Oracle status flips between runs in the experiment's own files: `image_263` not an oracle, then an oracle at attempt 1 (p12 logs/streams_TB-Q100-ae-l2_3.log); `image_20` not an oracle, then an oracle at attempt 1 (07_repair_v3/rq1 rows (default call, one attempt) vs p12 cache/targets). Effect on the Qwen REEDIT-B n=100 cell: with the experiment's z* my install gives the saved weight (0 of 22544384 entries differ) and its record row for row; with my z* and the same installed set 11 of 490 held-out rows flip (D1 59.4% against 58.8%, Rel exact 72 against 71); with the fallback ladder (100 installed, the stated protocol) D1 58.4%, Rel exact 72.
- Qwen best point: Rel exact 78 -> 78 (2 broken, 2 fixed); T-Loc 84.8 -> 89.1, I-Loc 88.7 -> 88.9.
- LLaVA best point: Rel exact 88 -> 86 (3 broken, 1 fixed), exactly on the 2-point tolerance; T-Loc 77.3 -> 77.8, I-Loc 81.5 -> 76.7.
- The 'why' paragraph is LLaVA only and quotes the complaint medians only. The quantities are not specific to complaints: held-out non-complaints have s = 0.64 (complaints 0.43) and a = 1.41 (complaints 0.87); the gap is the probe-type mix (image rows s = 0.73 / 0.72, text rows 0.39 / 0.39, complaints / non-complaints). My one behavioural run of W_0 + D P_ans (W_0 restored outside the span): HELD-OUT D1 81.0% (+49 / -145) against 61.5% unrepaired, Rel exact 89; REPAIR pool D1 402/494 against 305/494. The part of the log inside the edits' answer-key span alone damages more rows than the whole log, so the out-of-span part partly offsets it (a and b are not an orthogonal split).
- Tables use B=1 teacher forcing for the edits; the testbed's stored values are batch-8: Qwen Rel / T-Gen / I-Gen stored 78 / 9 / 23, tables 78 / 9 / 22.
- Single seed; the best point is the minimum over the configurations inside the tolerance, chosen on the evaluation pool (stated in the report). No same-norm control correction was run, so 'fix rate about equal to loss rate' is not shown to be specific to the reported complaints.
- Qwen REEDIT at n = 10 / 100 / ALL exists only as first-pass records (complaints not installed); marked with a dagger in the report.
- Split leak, beyond identical queries: Qwen: 10 more valid HELD-OUT image rows share the question and the image's entity folder with a REPAIR-pool row (another picture; 8 next to a REPAIR complaint). The best point fixes 12/16 of the identical-twin complaints and 83/249 of the other held-out complaints; without both kinds of twins: 49.7% against 54.7% (p = 0.065, 455 rows); LLaVA: 10 more valid HELD-OUT image rows share the question and the image's entity folder with a REPAIR-pool row (another picture; 6 next to a REPAIR complaint). The best point fixes 7/12 of the identical-twin complaints and 63/290 of the other held-out complaints; without both kinds of twins: 60.8% against 63.0% (p = 0.4, 459 rows).

- Provenance: `code/run_streams.py` and `p12_common.py` were last modified at 13:25:53 UTC, after the last raw record (12:10:23 UTC). 9 COMP records carry `extra.feasible_exact = false` although every criterion of the current code holds (they have dropped duplicate directions); the flag is not used by any reported number. The cells I rebuilt reproduce with the current harness and the stated options.
- Not recomputed by me: T-Gen / I-Gen at B=1 (my loop checks Rel; the batch-8 T-Gen / I-Gen of the unrepaired log equal the stored values), the cost columns, CIs other than the best point's, and every cell outside the conditions of section 3 (those were checked against the raw records only). My private copies of the edit-key cache were deleted after the bitwise comparison recorded in `out/<log>/PROTKEYS.json`.

## 7. Shared-GPU log of this verifier (operational; `out/_verifier_incidents.json`)

- 2026-11-10 14:56:53, GPU 0: My Qwen job (pid 13896, running since 14:45) left the card on its watchdog: a launcher of 09_testbed_scale (gpu_big.sh, threshold 16424 MiB) fitted on GPU 0 only without it. Impact on others: that launcher waited at most about 25 s for my process (two watchdog looks 12 s apart); no failure.
- 2026-11-10 15:25:40, GPU 1: My LLaVA job (pid 37950) ran out of memory under its OWN cap (14.4 GiB) while loading and exited; restarted later with a 14.8 GiB cap and W_E - W_0 computed on the CPU. Impact on others: none found (no out-of-memory message of another agent names this process).
- 2026-11-10 15:54:09-15:54:16, GPU 1: My LLaVA job (pid 51485; the two extra conditions LOC and WPROJ) reserved its memory at 15:54:10, one second after the launcher of another verifier (08_rescope/p11_retrofit/verify/run_gpu.sh, waiting for 16200 MiB since 15:52:39) had launched its LLaVA job on the same free card. Their load ran out of memory (their message names process 51485 with 14.26 GiB). Impact on others: one failed start of that job; its launcher retried after 120 s, launched at 15:57:46 and the job ended with exit 0 at 15:58:49: a delay of about 3.5 minutes, no output lost. Cause: my queue inspector (vcommon.launchers) only knew the launcher scripts that existed when I wrote it (gpu_wait.sh, wait_gpu.sh, gpu_run.sh, gpu_any.sh, gpu_fill.sh, gpu_big.sh, gpu_launch.sh); run_gpu.sh was not among them, so its 90 s wait was invisible to my two looks. Fix: 16:02: vcommon.launchers now recognises any shell script whose name contains gpu / wait / launch / queue, in every argument layout in use (unknown layouts count as a LLaVA-sized waiter on both cards), and can_start refuses to start on a card while ANY other agent's launcher is waiting for it.
- 2026-11-10 16:01:27, GPU 1: I stopped my own LLaVA job (pid 52777, LOC, running since 15:59:28) by hand because I believed two launchers of the reviewer (code/rescope/review/gpu_wait.sh) were waiting behind it. They were not: both had started their jobs on GPU 0 at about 15:54. The condition was run again later. Impact on others: none.
- 2026-11-10 15:41-15:53 and 16:10-16:11, GPU 1: Deviation from the card convention (Qwen on GPU 0): GPU 0 was held by other agents' jobs and queued launchers from 14:57 on, so the remaining Qwen conditions (the two random cells, protection keys, batch-8 Rel: pid 46703; the z* repeat test: pid 56782; cap 9.2 GiB) ran on GPU 1, under the same start rule and watchdog. Impact on others: none found.

## 8. Files

- `verify/gpu_verify.py`, `vcommon.py`, `cpu_checks.py`, `cpu_extra.py`, `cpu_moddiag.py`, `leak_pixels.py`, `sandbox_render.sh`, `draw_cells.py`, `render_report.py`
- `verify/out/<log>/*.json` (one file per recomputed condition, with my per-row flags), `verify/out/*.json`, `verify/logs/`

