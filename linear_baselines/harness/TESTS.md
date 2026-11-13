# Harness unit tests

Rendered by `harness/tests/render_tests.py` from `out/tests/*.json` (produced by `harness/tests/run_tests.py`). Do not edit by hand.

## zero — zero correction is bit-identical; exactness of the speed-ups

**pass = True** (242 s, NVIDIA GeForce RTX 5090)

- logits of 10 sequences (complaints + edit rel/igen) bitwise equal to W_E under: adapter_gated True, adapter_ungated True, merged True, weight_WE True, rollback_empty True
- full evaluate() record identical to NONE: adapter_gated True, merged True
- NONE record: edits broken 0, fixed 0, held-out rows changed 0, Loc items changed {'tloc': 0, 'iloc': 0}, per-edit em/acc identical True
- suffix engine == full forward (bitwise) at: WE True, W0 True, W(R=3,17,55) True, W(batch 2 + edit 40) True
- W(R = all edits) == W_0 bitwise: True; module-input keys identical at W_E and W_0: True
- vision-feature cache bitwise (miss and hit): True; prebuilt eval batches == vendor run_probes: {'rel': True, 'tgen': True, 'igen': True}
- load checks {'probe_uids_match_records': True, 'collapsed_working_point': False, 'WE_float_minus_W0_equals_stored_composite': True, 'C_equals_stored_composite': True, 'factor_uids_match_stream': True, 'rollback_empty_equals_WE': True}; baseline checks {'heldout_reproduced_equals_stored_v2': True, 'n_heldout_rows': 490, 'n_heldout_items': 250, 'Rel_token_WE': 91.57936528325081, 'Rel_exact_WE': 78.0, 'Rel_B1_token_WE': 91.19047638773918, 'Rel_B1_exact_WE': 78.0, 'edits_B1_vs_b8_em_flags_differing': {'rel': 2, 'tgen': 0, 'igen': 1}, 'Rel_matches_stored': True, 'fast_eval_equals_full': True}

## comp_w2 — COMP reproduces the old W2 H1 (K_last, answer positions, fp64 adapter, alpha 1)

**pass = True** (1274 s). 20 random P8 complaints (seed 2026, 10 image / 10 text), ci = [2, 7, 16, 34, 35, 36, 44, 60, 63, 77, 111, 115, 127, 159, 161, 168, 174, 187, 189, 195].

- resolved: new ungated 16/20, new gated 16/20, old cb_run H1 16/20 (old H1 on all 200: 0.795)
- per-complaint flag agreement with cb_run: ungated 20/20, gated 20/20; max rel. diff of ||correction||_F 1.38e-15
- rel edits broken (sum over the 20): new 1 (vendor batch-8 eval) vs old 1 (cb_run B=1 full forwards)
- load checks {'probe_uids_match_records': True, 'collapsed_working_point': False, 'old_w2_W0_entries_differing_from_checkpoint': 3148538}

## fo_w2 — ROLLBACK-FO reproduces P8's greedy-FO flag (budget 10, old W2)

**pass = True** (172 s). Same 20 complaints.

- resolved flag agreement 20/20 (new 19, P8 19); identical removal sequence 20/20; internal stop == evaluate() 20/20
- mean k removed new 4.60 vs P8 4.60; wall-clock per complaint new 7.40 s (suffix engine, factorised scores) vs P8 32.3 s
- load checks {'probe_uids_match_records': True, 'old_w2_total_equals_cb_cache': True, 'old_w2_W0_entries_differing_from_checkpoint': 3148538}

## reedit — REEDIT reproduces baselines/smoke (TB-Q100-ae, 8 complaints x {k0, k0+kp})

**pass = True** (382 s).

- agreement with the smoke: resolved 16/16, Rel exact after 16/16, collateral broken (smoke's 20 probes) 13/16, z* steps 16/16; max |dRel token| 0.708; max rel. diff dW/W0 8.99e-03
- factorised GPU solve vs vendor Filters.solve: max rel. diff of D 1.32e-14, W_after entries differing (max) 0
- wall-clock median: solve 0.003 s, repair total 0.55 s (smoke, vendor dense solve with diagnostics: 8.58 s)

| complaint | variant | resolved new/old | Rel exact new/old | coll broken new/old | dW/W0 new/old | z steps new/old |
|---|---|---|---|---|---|---|
| image_427 | k0 | True/True | 22/22 | 1/1 | 8.3913e-02/8.4675e-02 | 7/7 |
| image_427 | k0+kp | True/True | 22/22 | 0/1 | 8.3011e-02/8.3732e-02 | 7/7 |
| image_365 | k0 | True/True | 22/22 | 1/1 | 4.4868e-02/4.4743e-02 | 2/2 |
| image_365 | k0+kp | True/True | 22/22 | 0/0 | 4.4458e-02/4.4347e-02 | 2/2 |
| image_313 | k0 | True/True | 22/22 | 3/0 | 1.0556e-01/1.0544e-01 | 12/12 |
| image_313 | k0+kp | True/True | 22/22 | 1/1 | 1.0383e-01/1.0380e-01 | 12/12 |
| image_28 | k0 | True/True | 19/19 | 1/1 | 1.1370e-01/1.1362e-01 | 9/9 |
| image_28 | k0+kp | True/True | 21/21 | 0/1 | 1.0966e-01/1.0962e-01 | 9/9 |
| text_191 | k0 | True/True | 22/22 | 7/7 | 9.2932e-02/9.2853e-02 | 6/6 |
| text_191 | k0+kp | True/True | 22/22 | 6/6 | 9.2756e-02/9.2677e-02 | 6/6 |
| text_395 | k0 | True/True | 22/22 | 6/6 | 5.8766e-02/5.8640e-02 | 3/3 |
| text_395 | k0+kp | True/True | 22/22 | 7/7 | 5.8618e-02/5.8493e-02 | 3/3 |
| text_492 | k0 | True/True | 22/22 | 3/3 | 7.4878e-02/7.4762e-02 | 5/5 |
| text_492 | k0+kp | True/True | 22/22 | 3/3 | 7.4686e-02/7.4570e-02 | 5/5 |
| text_215 | k0 | True/True | 22/22 | 5/5 | 5.8138e-02/5.8006e-02 | 3/3 |
| text_215 | k0+kp | True/True | 22/22 | 5/5 | 5.8002e-02/5.7870e-02 | 3/3 |

## kseq — protecting every position of the 26 shortest edit sequences (K_seq:26)

**pass = True** (281 s). 7886 protected positions (d_in 11008; d_in / mean edit length = 16.6 sequences).

- repairs leaving ALL 26 protected sequences bit-identical (full logits, every position): gated adapter 6/6, ungated 6/6; complaints resolved 5/6

| complaint | gated identical | ungated identical | protect rank | ||corr||_F |
|---|---|---|---|---|
| image_424 | 26 | 26 | 7511 | 14.351 |
| image_303 | 26 | 26 | 7511 | 14.677 |
| image_6 | 26 | 26 | 7511 | 13.621 |
| text_290 | 26 | 26 | 7511 | 16.799 |
| text_241 | 26 | 26 | 7511 | 14.483 |
| text_137 | 26 | 26 | 7511 | 13.343 |

## fasteval — exact fast evaluation path (suffix engine) == full forwards

**pass = True** (1227 s): identical evaluate() records 16/16 (COMP adapter, COMP merged, ROLLBACK-FO, REEDIT on 4 complaints); median eval time fast 21.9 s vs full 35.8 s.

## llava — the harness on LLaVA-1.5-7B (TB-L100-ae-l2_3)

**pass = True** (291 s); suffix == full bitwise [True, True, True, True]; peak GPU memory allocated 16.0 GB; load checks {'probe_uids_match_records': True, 'collapsed_working_point': False, 'WE_float_minus_W0_equals_stored_composite': True, 'C_equals_stored_composite': True, 'factor_uids_match_stream': True, 'rollback_empty_equals_WE': True}; baseline checks {'heldout_reproduced_equals_stored_v2': True, 'n_heldout_rows': 494, 'n_heldout_items': 250, 'Rel_token_WE': 97.12261933088303, 'Rel_exact_WE': 88.0, 'Rel_B1_token_WE': 97.12261933088303, 'Rel_B1_exact_WE': 88.0, 'edits_B1_vs_b8_em_flags_differing': {'rel': 0, 'tgen': 0, 'igen': 0}, 'Rel_matches_stored': True, 'fast_eval_equals_full': True}

| method | cfg | resolved | edits broken | held-out non-complaints lost | ||corr||_F | passes | s |
|---|---|---|---|---|---|---|---|
| NONE | {} | False | 0 | 0 | 0.000 | {'full': 0, 'prefix': 0, 'fwd_bwd': 0, 'suffix': 0, 'suffix_grad': 0} | 0.00 |
| COMP | {} | True | 0 | 7 | 4.887 | {'full': 0, 'prefix': 1, 'fwd_bwd': 0, 'suffix': 0, 'suffix_grad': 0} | 0.07 |
| COMP | {'form': 'merged', 'query': 'all'} | True | 8 | 30 | 57.174 | {'full': 0, 'prefix': 1, 'fwd_bwd': 0, 'suffix': 0, 'suffix_grad': 0} | 0.50 |
| REEDIT | {} | True | 0 | 5 | 15.313 | {'full': 1, 'prefix': 2, 'fwd_bwd': 3, 'suffix': 0, 'suffix_grad': 0} | 0.53 |
| ROLLBACK_FO | {'budget': 10} | True | 2 | 4 | 16.875 | {'full': 1, 'prefix': 0, 'fwd_bwd': 0, 'suffix': 1, 'suffix_grad': 1} | 0.24 |
| ROLLBACK_BEH | {'budget': 3} | True | 2 | 4 | 16.875 | {'full': 1, 'prefix': 0, 'fwd_bwd': 0, 'suffix': 100, 'suffix_grad': 0} | 2.65 |
| ROLLBACK_ALL | {} | True | 180 | 0 | 266.480 | {'full': 0, 'prefix': 0, 'fwd_bwd': 0, 'suffix': 0, 'suffix_grad': 0} | 0.00 |

## reedit_noise — run-to-run noise of z* (original baselines/reedit.py vs harness, 2 runs each)

W_E identical to reedit.py's: True; relative spread of the applied ||delta||/||W_0|| over the 4 runs: image_427 9.83e-02, text_191 0.00e+00

reedit.py reports dW/W0 with CPU fp32 norms (biased by ~1e-3 vs fp64); the fp64 column is comparable across rows.

| complaint | implementation | run | dW/W0 as reported | dW/W0 fp64 | W_after == harness (same run index) | z* ||delta||/||h|| | z* NLL | z* steps |
|---|---|---|---|---|---|---|---|---|
| image_427 | baselines/reedit.py | 0 | 7.71705e-02 | 7.72784e-02 | False | 1.06370 | 0.04543 | 6 |
| image_427 | harness | 0 | 8.47837e-02 | 8.47837e-02 | - | 1.16711 | 0.02803 | 7 |
| image_427 | baselines/reedit.py | 1 | 8.47457e-02 | 8.48380e-02 | False | 1.16785 | 0.02788 | 7 |
| image_427 | harness | 1 | 8.48756e-02 | 8.48756e-02 | - | 1.16838 | 0.02693 | 7 |
| text_191 | baselines/reedit.py | 0 | 9.28527e-02 | 9.29317e-02 | True | 0.99281 | 0.02834 | 6 |
| text_191 | harness | 0 | 9.29317e-02 | 9.29317e-02 | - | 0.99281 | 0.02834 | 6 |
| text_191 | baselines/reedit.py | 1 | 9.28527e-02 | 9.29317e-02 | True | 0.99281 | 0.02834 | 6 |
| text_191 | harness | 1 | 9.29317e-02 | 9.29317e-02 | - | 0.99281 | 0.02834 | 6 |

## kpc — protect "K_pc:<r>" (top-r principal directions of K_all; RQ3 extension, TB-Q100-ae-l2_3)

**pass = True** (47 s). K_all = 66426 keys, d_in 11008; eigenbasis orthonormality err 1.1e-14, eigen-residual 6.9e-16, descending True, lambda_min/lambda_max 1.47e-05 (K_all full rank); energy of U_64 from the keys 0.402348546992 vs from the eigenvalues 0.402348546992; union K_last + K_pc:64 rank 164 (independent matrix_rank 164); pre-existing specs give the bitwise same basis as before the extension: {'K_last': True, 'K_ans': True, 'K_seq:3': True, 'K_last+probe:20': True}

| complaint | A | K_pc:0 == none | K_pc:d_in infeasible | ||corr|| at d_in | module out == W_E | rank kept at d_in - A | P1 | P2 | ||corr|| at d_in - A | ||corr|| none |
|---|---|---|---|---|---|---|---|---|---|---|
| image_479 | 10 | True | True | 0.0 | True | 10 | 1.1e-14 | 5.2e-12 | 7652.7 | 4.85 |
| text_290 | 10 | True | True | 0.0 | True | 10 | 6.6e-15 | 1.5e-12 | 7393.3 | 8.23 |
| text_335 | 10 | True | True | 0.0 | True | 10 | 3.7e-15 | 4.4e-13 | 2066.3 | 6.34 |

## rq2cells — RQ2 options of COMP (solve="alphaedit") and REEDIT (solve="exact", form="adapter", targets=) (TB-Q100-ae-l2_3)

**pass = True** (56 s). 4 random complaints; K_ans protection rank 400. Defaults unchanged (COMP() == COMP(solve='exact') bitwise; REEDIT() == vendor solve, W_after bitwise; reusing REEDIT's own z* reproduces W_after bitwise and counts no zopt pass).

| complaint | COMP default bitwise | COMP-AE vs vendor (rel D) | COMP-AE attainment | REEDIT vs vendor (W entries diff) | reused z* W bitwise | REEDIT adapter vs vendor (rel D) | REEDIT-exact attainment | REEDIT-exact max abs R^T Q | protected K_ans outputs bitwise |
|---|---|---|---|---|---|---|---|---|---|
| image_274 | True | 1.3e-14 | 0.801 | 0 | True | 1.3e-14 | 0.999999999998 | 5.2e-13 | True |
| image_106 | True | 1.2e-14 | 0.839 | 0 | True | 9.1e-15 | 0.999999999999 | 2.7e-13 | True |
| text_32 | True | 1.9e-14 | 0.921 | 0 | True | 6.3e-15 | 1.000000000000 | 9.0e-14 | True |
| text_479 | True | 2.7e-14 | 0.943 | 0 | True | 6.5e-15 | 1.000000000000 | 8.2e-14 | True |

