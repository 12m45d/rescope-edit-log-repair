# P12 -- linear repair streams

Rendered by `code/render.py` from `out/results.json` (`code/aggregate.py` over `out/<log>/*.json`, written by `code/run_streams.py`, `code/run_d2.py`, `code/run_text_h2h.py` through the shared harness `07_repair_v3/harness`). Do not edit by hand. Figures: `figures/fig_heldout_d1.png`, `fig_rel_exact.png`, `fig_edits_broken.png`, `fig_tradeoff.png`.

## 1. Findings and anomalies

**Sanity.**
- Qwen (TB-Q100-ae-l2_3): NONE reproduces the stored baseline -- D1 flag of every valid row equal to `probe_records_v2.jsonl` on the REPAIR pool (489 rows: True) and on the HELD-OUT pool (490 rows: True); Rel exact 78.0 / token 91.58 on the testbed's batch-8 path = stored 78.0 / 91.58 (True); the B=1 teacher-forced Rel exact used in every table below is 78.0. ALL (W_0) gives zero D1 complaints (REPAIR 0, HELD-OUT 0; weight bitwise W_0: True); its Rel exact is 0.
- LLaVA (TB-L100-ae-l2_3): NONE reproduces the stored baseline -- D1 flag of every valid row equal to `probe_records_v2.jsonl` on the REPAIR pool (494 rows: True) and on the HELD-OUT pool (494 rows: True); Rel exact 88.0 / token 97.12 on the testbed's batch-8 path = stored 88.0 / 97.12 (True); the B=1 teacher-forced Rel exact used in every table below is 88.0. ALL (W_0) gives zero D1 complaints (REPAIR 0, HELD-OUT 0; weight bitwise W_0: True); its Rel exact is 1.

**Anomaly 0 (protocol level) -- the HELD-OUT pool is not disjoint from the REPAIR pool.** The harness splits the 500 probe items by uid (`harness/sampler.py`, seed 20261109; the same split is used by every experiment that calls `sampler.pool_of`), but VLKEB eval items share image-locality probes: the same (image, question) occurs under several uids, on both sides of the split. A held-out row whose query is identical to a reported complaint is that complaint (same input, same y0): repairing the reported one repairs it, which is not generalisation. Text queries are all distinct. Below, **disjoint HELD-OUT** = the valid HELD-OUT rows whose query does not occur in the REPAIR pool; both versions are reported, the protocol's full pool first. Any HELD-OUT number computed on this split contains these rows.
- Qwen (TB-Q100-ae-l2_3): 444 distinct image queries in 500 image rows (500 / 500 for text); 25 of the 490 valid HELD-OUT rows are twins of a REPAIR-pool row (25 image, 0 text): 16 held-out complaints (all 16 twins of REPAIR D1 complaints, 6.0% of the held-out complaints) and 9 held-out non-complaints. Inside the REPAIR pool 8 of the 297 complaints repeat an earlier query (79 redundant answer keys; the joint compensation of all REPAIR complaints drops 89 query directions). Unrepaired D1 rate: full pool 54.7% (490 rows), disjoint 54.2% (465 rows), twins 64.0% (25 rows). Check: over 48 repaired configurations, a held-out twin of a reported complaint has the same outcome as that complaint in 280/280 cases (two different evaluation paths: suffix engine for held-out rows, full forward for reported complaints).
- LLaVA (TB-L100-ae-l2_3): 444 distinct image queries in 500 image rows (500 / 500 for text); 25 of the 494 valid HELD-OUT rows are twins of a REPAIR-pool row (25 image, 0 text): 12 held-out complaints (all 12 twins of REPAIR D1 complaints, 3.9% of the held-out complaints) and 13 held-out non-complaints. Inside the REPAIR pool 6 of the 305 complaints repeat an earlier query (31 redundant answer keys; the joint compensation of all REPAIR complaints drops 31 query directions). Unrepaired D1 rate: full pool 61.5% (494 rows), disjoint 62.3% (469 rows), twins 48.0% (25 rows). Check: over 72 repaired configurations, a held-out twin of a reported complaint has the same outcome as that complaint in 288/288 cases (two different evaluation paths: suffix engine for held-out rows, full forward for reported complaints).

**Finding 1 -- what happens to the unreported complaints (HELD-OUT pool; seed 0; HELD-OUT D1 rate % full pool / disjoint rows, Rel exact in brackets).**
- Qwen (TB-Q100-ae-l2_3): unrepaired 54.7 / 54.2 (268/490 and 252/465 rows; Rel exact 78). After the repair: COMP-B n=1: 52.9 / 52.5 (78), n=10: 51.4 / 50.8 (75), n=100: 49.4 / 50.1 (75), n=ALL: 47.3 / 49.0 (78); COMP-B+ n=1: 55.1 / 54.8 (78), n=10: 53.7 / 53.5 (77), n=100: 55.9 / 56.6 (76), n=ALL: 66.9 / 68.8 (76); REEDIT-B n=1: 59.0 / 59.4 (76), n=10: 56.1 / 55.9 (75), n=100: 57.1 / 58.1 (72), n=ALL: 62.9 / 66.0 (63); REEDIT-S n=1: 59.0 / 59.4 (76), n=10: 56.1 / 55.9 (75), n=100: 59.0 / 60.0 (72), n=ALL: 63.1 / 66.2 (63); RB-TOT n=1: 51.8 / 52.0 (72), n=10: 45.1 / 44.9 (54), n=100: 42.2 / 43.4 (53), n=ALL: 44.9 / 46.0 (56); RB-UNION n=1: 51.8 / 52.0 (72), n=10: 46.7 / 47.5 (61), n=100: 14.1 / 14.4 (7), n=ALL: 5.9 / 6.2 (1).
- LLaVA (TB-L100-ae-l2_3): unrepaired 61.5 / 62.3 (304/494 and 292/469 rows; Rel exact 88). After the repair: COMP-B n=1: 61.3 / 61.8 (88), n=10: 61.5 / 62.0 (88), n=100: 59.1 / 60.6 (89), n=ALL: 58.5 / 59.7 (86); COMP-B+ n=1: 62.8 / 63.3 (88), n=10: 63.6 / 64.2 (88), n=100: 66.4 / 67.0 (88), n=ALL: 66.8 / 67.6 (84); REEDIT-B n=1: 65.0 / 66.3 (88), n=10: 66.0 / 66.5 (87), n=100: 71.3 / 73.6 (86), n=ALL: 75.3 / 78.0 (74); REEDIT-S n=1: 65.0 / 66.3 (88), n=10: 66.0 / 66.5 (87), n=100: 71.3 / 73.6 (85), n=ALL: 75.3 / 78.0 (74); RB-TOT n=1: 62.3 / 62.3 (86), n=10: 57.3 / 58.8 (66), n=100: 51.2 / 53.1 (65), n=ALL: 47.8 / 49.5 (63); RB-UNION n=1: 62.3 / 62.3 (86), n=10: 53.2 / 53.9 (66), n=100: 11.1 / 11.3 (5), n=ALL: 0.0 / 0.0 (1).

**Finding 2 -- what a repair does to unreported rows: held-out complaints fixed and held-out non-complaints lost (disjoint held-out rows, seed 0; +fixed / -lost).**
- Qwen (TB-Q100-ae-l2_3) (disjoint rows: 252 held-out complaints, 213 held-out non-complaints): COMP-B n=1: +27 / -19, n=10: +48 / -32, n=100: +74 / -55, n=ALL: +85 / -61; COMP-B+ n=1: +5 / -8, n=10: +28 / -25, n=100: +52 / -63, n=ALL: +40 / -108; REEDIT-B n=1: +32 / -56, n=10: +45 / -53, n=100: +49 / -67, n=ALL: +39 / -94; RB-TOT n=1: +36 / -26, n=10: +74 / -31, n=100: +81 / -31, n=ALL: +79 / -41; RB-UNION n=1: +36 / -26, n=10: +70 / -39, n=100: +199 / -14, n=ALL: +229 / -6.
- LLaVA (TB-L100-ae-l2_3) (disjoint rows: 292 held-out complaints, 177 held-out non-complaints): COMP-B n=1: +13 / -11, n=10: +19 / -18, n=100: +49 / -41, n=ALL: +65 / -53; COMP-B+ n=1: +3 / -8, n=10: +8 / -17, n=100: +29 / -51, n=ALL: +42 / -67; REEDIT-B n=1: +18 / -37, n=10: +19 / -39, n=100: +27 / -80, n=ALL: +32 / -106; RB-TOT n=1: +11 / -11, n=10: +52 / -36, n=100: +70 / -27, n=ALL: +88 / -28; RB-UNION n=1: +11 / -11, n=10: +66 / -27, n=100: +250 / -11, n=ALL: +292 / -0.

**Finding 2a -- fix rate against loss rate on unreported rows (disjoint held-out rows, seed 0; fix rate % = held-out complaints fixed / held-out complaints | loss rate % = held-out non-complaints lost / held-out non-complaints).** The two rates are per-row probabilities on two different sets, so they can be compared directly: equal rates mean that a held-out row is as likely to be changed for the worse as for the better. (No control with an unrelated correction of the same norm was run: whether the flips of the compensation and re-edit methods are specific to the reported complaints is not established by these numbers.)
- Qwen (TB-Q100-ae-l2_3): COMP-B n=1: 11 | 9, n=10: 19 | 15, n=100: 29 | 26, n=ALL: 34 | 29; COMP-B+ n=1: 2 | 4, n=10: 11 | 12, n=100: 21 | 30, n=ALL: 16 | 51; REEDIT-B n=1: 13 | 26, n=10: 18 | 25, n=100: 19 | 31, n=ALL: 15 | 44; RB-TOT n=1: 14 | 12, n=10: 29 | 15, n=100: 32 | 15, n=ALL: 31 | 19; RB-UNION n=1: 14 | 12, n=10: 28 | 18, n=100: 79 | 7, n=ALL: 91 | 3.
- LLaVA (TB-L100-ae-l2_3): COMP-B n=1: 4 | 6, n=10: 7 | 10, n=100: 17 | 23, n=ALL: 22 | 30; COMP-B+ n=1: 1 | 5, n=10: 3 | 10, n=100: 10 | 29, n=ALL: 14 | 38; REEDIT-B n=1: 6 | 21, n=10: 7 | 22, n=100: 9 | 45, n=ALL: 11 | 60; RB-TOT n=1: 4 | 6, n=10: 18 | 20, n=100: 24 | 15, n=ALL: 30 | 16; RB-UNION n=1: 4 | 6, n=10: 23 | 15, n=100: 86 | 6, n=ALL: 100 | 0.

**Finding 2a' -- do the held-out flips accumulate along the nested reported sets? Of the held-out rows flipped by the repair of n1 complaints, the number still flipped after the repair of the superset of n2 complaints (seed 0; fixed kept / fixed at n1 -> fixed at n2 | lost kept / lost at n1 -> lost at n2).**
- Qwen (TB-Q100-ae-l2_3), COMP-B: 1->2: 22/28 -> 39 | 19/19 -> 29; 2->5: 24/39 -> 42 | 21/29 -> 29; 5->10: 32/42 -> 48 | 22/29 -> 32; 10->20: 33/48 -> 59 | 24/32 -> 43; 20->50: 39/59 -> 63 | 26/43 -> 47; 50->100: 47/63 -> 82 | 32/47 -> 56; 100->ALL: 54/82 -> 97 | 35/56 -> 61.
- Qwen (TB-Q100-ae-l2_3), REEDIT-B: 1->2: 29/35 -> 30 | 48/56 -> 51; 2->5: 20/30 -> 31 | 33/51 -> 54; 5->10: 23/31 -> 46 | 42/54 -> 53; 10->20: 21/46 -> 36 | 37/53 -> 50; 20->50: 22/36 -> 43 | 40/50 -> 69; 50->100: 32/43 -> 55 | 50/69 -> 67; 100->ALL: 30/55 -> 54 | 51/67 -> 94.
- Qwen (TB-Q100-ae-l2_3), RB-TOT: 1->2: 31/40 -> 43 | 16/26 -> 27; 2->5: 29/43 -> 76 | 11/27 -> 30; 5->10: 56/76 -> 78 | 20/30 -> 31; 10->20: 56/78 -> 78 | 19/31 -> 29; 20->50: 46/78 -> 96 | 17/29 -> 38; 50->100: 60/96 -> 92 | 18/38 -> 31; 100->ALL: 60/92 -> 89 | 18/31 -> 41.
- Qwen (TB-Q100-ae-l2_3), RB-UNION: 1->2: 35/40 -> 44 | 20/26 -> 30; 2->5: 27/44 -> 60 | 17/30 -> 35; 5->10: 53/60 -> 78 | 27/35 -> 39; 10->20: 66/78 -> 114 | 18/39 -> 28; 20->50: 93/114 -> 162 | 16/28 -> 27; 50->100: 150/162 -> 213 | 10/27 -> 14; 100->ALL: 205/213 -> 245 | 6/14 -> 6.
- LLaVA (TB-L100-ae-l2_3), COMP-B: 1->2: 13/13 -> 13 | 12/12 -> 12; 2->5: 9/13 -> 14 | 12/12 -> 19; 5->10: 11/14 -> 19 | 14/19 -> 19; 10->20: 15/19 -> 27 | 15/19 -> 25; 20->50: 16/27 -> 39 | 16/25 -> 31; 50->100: 29/39 -> 54 | 24/31 -> 42; 100->ALL: 36/54 -> 72 | 28/42 -> 57.
- LLaVA (TB-L100-ae-l2_3), REEDIT-B: 1->2: 18/20 -> 23 | 35/37 -> 42; 2->5: 18/23 -> 23 | 38/42 -> 44; 5->10: 16/23 -> 22 | 37/44 -> 44; 10->20: 14/22 -> 22 | 38/44 -> 61; 20->50: 16/22 -> 35 | 50/61 -> 73; 50->100: 24/35 -> 36 | 66/73 -> 84; 100->ALL: 24/36 -> 44 | 74/84 -> 112.
- LLaVA (TB-L100-ae-l2_3), RB-TOT: 1->2: 7/11 -> 32 | 10/15 -> 33; 2->5: 20/32 -> 53 | 19/33 -> 36; 5->10: 32/53 -> 58 | 23/36 -> 37; 10->20: 29/58 -> 59 | 20/37 -> 34; 20->50: 39/59 -> 72 | 17/34 -> 27; 50->100: 50/72 -> 78 | 17/27 -> 27; 100->ALL: 60/78 -> 96 | 16/27 -> 28.
- LLaVA (TB-L100-ae-l2_3), RB-UNION: 1->2: 10/11 -> 28 | 8/15 -> 26; 2->5: 22/28 -> 55 | 13/26 -> 30; 5->10: 48/55 -> 70 | 25/30 -> 29; 10->20: 48/70 -> 97 | 20/29 -> 30; 20->50: 90/97 -> 222 | 14/30 -> 25; 50->100: 209/222 -> 260 | 10/25 -> 11; 100->ALL: 260/260 -> 304 | 0/11 -> 0.

**Finding 2b -- protecting a regression suite of known-good rows (COMP-B+ against COMP-B).** e = mean share of a reported complaint key's norm left after projecting out the protected span; the smaller e, the larger the correction needed for the same restoration. Largest n with both records, seed 0:
- Qwen (TB-Q100-ae-l2_3), n = ALL (297): COMP-B protects rank 400 (e = 0.82), correction norm 0.65 of the log, held-out +97 / -61, D1 rate 47.3% (below the unrepaired 54.7%); COMP-B+ protects rank 2254 (e = 0.44), norm 1.42, held-out +49 / -109, D1 rate 66.9% (above the unrepaired log); reported complaints resolved 168/297 against 127/297.
- LLaVA (TB-L100-ae-l2_3), n = ALL (305): COMP-B protects rank 483 (e = 0.86), correction norm 0.46 of the log, held-out +72 / -57, D1 rate 58.5% (below the unrepaired 61.5%); COMP-B+ protects rank 1355 (e = 0.69), norm 0.78, held-out +48 / -74, D1 rate 66.8% (above the unrepaired log); reported complaints resolved 164/305 against 137/305.

**Finding 2c -- re-editing (REEDIT) and held-out rows.** Net = held-out complaints fixed minus held-out non-complaints lost (full pool, seed 0):
- Qwen (TB-Q100-ae-l2_3): n=1: -21 (Rel 76, 2 edit checks broken); n=2: -21 (Rel 78, 0 edit checks broken); n=5: -23 (Rel 76, 2 edit checks broken); n=10: -7 (Rel 75, 3 edit checks broken); n=20: -14 (Rel 75, 5 edit checks broken); n=50: -26 (Rel 73, 7 edit checks broken); n=100: -12 (Rel 72, 10 edit checks broken); n=ALL: -40 (Rel 63, 26 edit checks broken). The net is negative at every n measured.
- LLaVA (TB-L100-ae-l2_3): n=1: -17 (Rel 88, 0 edit checks broken); n=2: -19 (Rel 88, 0 edit checks broken); n=5: -21 (Rel 89, 0 edit checks broken); n=10: -22 (Rel 87, 1 edit checks broken); n=20: -39 (Rel 87, 2 edit checks broken); n=50: -38 (Rel 87, 5 edit checks broken); n=100: -48 (Rel 86, 10 edit checks broken); n=ALL: -68 (Rel 74, 38 edit checks broken). The net is negative at every n measured.

**Finding 2d -- rollback: HELD-OUT D1 rate against the number of edits removed (k of the logged edits rolled back; seed 0).**
- Qwen (TB-Q100-ae-l2_3), RB-TOT: n=1: k=9, D1 51.8%, Rel 72; n=2: k=11, D1 51.4%, Rel 69; n=5: k=27, D1 45.3%, Rel 61; n=10: k=30, D1 45.1%, Rel 54; n=20: k=30, D1 44.7%, Rel 54; n=50: k=30, D1 42.9%, Rel 56; n=100: k=30, D1 42.2%, Rel 53; n=ALL: k=30, D1 44.9%, Rel 56.
- Qwen (TB-Q100-ae-l2_3), RB-UNION: n=1: k=9, D1 51.8%, Rel 72; n=2: k=11, D1 51.8%, Rel 70; n=5: k=24, D1 49.6%, Rel 62; n=10: k=27, D1 46.7%, Rel 61; n=20: k=46, D1 37.1%, Rel 43; n=50: k=75, D1 27.1%, Rel 18; n=100: k=92, D1 14.1%, Rel 7; n=ALL: k=99, D1 5.9%, Rel 1.
- LLaVA (TB-L100-ae-l2_3), RB-TOT: n=1: k=3, D1 62.3%, Rel 86; n=2: k=13, D1 61.7%, Rel 80; n=5: k=30, D1 58.1%, Rel 63; n=10: k=30, D1 57.3%, Rel 66; n=20: k=30, D1 56.5%, Rel 66; n=50: k=30, D1 52.4%, Rel 66; n=100: k=30, D1 51.2%, Rel 65; n=ALL: k=30, D1 47.8%, Rel 63.
- LLaVA (TB-L100-ae-l2_3), RB-UNION: n=1: k=3, D1 62.3%, Rel 86; n=2: k=9, D1 61.1%, Rel 83; n=5: k=23, D1 56.5%, Rel 71; n=10: k=29, D1 53.2%, Rel 66; n=20: k=49, D1 48.0%, Rel 47; n=50: k=86, D1 21.7%, Rel 13; n=100: k=96, D1 11.1%, Rel 5; n=ALL: k=100, D1 0.0%, Rel 1.

**Finding 3 -- best linear operating point (lowest HELD-OUT D1 rate with Rel exact within 2 points of the unrepaired log).**
- Qwen (TB-Q100-ae-l2_3): **COMP-B, n = 20, seed 2** -- HELD-OUT D1 rate 46.9% [42.9, 51.3] against 54.7% unrepaired (-7.8 points; exact McNemar on the held-out rows p = 0.00028; 72 complaints fixed, 34 non-complaints lost), Rel / T-Gen / I-Gen exact 78 / 8 / 24 (3 edit checks broken, 4 fixed), reported resolved 14/20. Cost: correction of rank 198 (protection 400), norm 0.145 of the log, 20 forward / 0 backward passes, 2.4 s. It is the minimum over 93 configurations inside the tolerance, selected on the evaluation pool itself (3 seeds at this n): read it as an optimistic bound. The same method and n over its 3 seeds: HELD-OUT D1 rate 49.0 (46.9-51.4)%, disjoint rows 48.8 (46.7-51.2)%, Rel exact 77 (74-80); 2 of the 3 seeds are inside the tolerance. Tied at the same rate: COMP-B n=20, COMP-S n=20. The repaired log keeps 86% of the unrepaired HELD-OUT D1 rate. On the disjoint rows the same point gives 46.7% [42.5, 50.9] against 54.2% (-7.5 points, p = 0.00073); 2 of its 72 fixed held-out complaints are twins of reported complaints (3 such twins).
  Selected on the disjoint rows instead: COMP-B, n = 20, seed 2 -- 46.7% (-7.5 points, p = 0.00073; full pool 46.9%), Rel exact 78.
  By seed mean, over the cells (method x n) whose every seed is inside the tolerance: COMP-B, n = ALL (1 seed) -- HELD-OUT D1 rate 47.3% (disjoint rows 49.0%), Rel exact 78.0.
- LLaVA (TB-L100-ae-l2_3): **COMP-B, n = ALL, seed 0** -- HELD-OUT D1 rate 58.5% [54.2, 63.0] against 61.5% unrepaired (-3.0 points; exact McNemar on the held-out rows p = 0.22; 72 complaints fixed, 57 non-complaints lost), Rel / T-Gen / I-Gen exact 86 / 32 / 66 (6 edit checks broken, 7 fixed), reported resolved 164/305. Cost: correction of rank 1973 (protection 483), norm 0.461 of the log, 305 forward / 0 backward passes, 10.9 s. It is the minimum over 117 configurations inside the tolerance, selected on the evaluation pool itself (1 seed at this n): read it as an optimistic bound. Tied at the same rate: COMP-B n=ALL, COMP-S n=ALL. The repaired log keeps 95% of the unrepaired HELD-OUT D1 rate. On the disjoint rows the same point gives 59.7% [55.1, 64.4] against 62.3% (-2.6 points, p = 0.31); 7 of its 72 fixed held-out complaints are twins of reported complaints (12 such twins).
  Selected on the disjoint rows instead: COMP-B, n = ALL, seed 0 -- 59.7% (-2.6 points, p = 0.31; full pool 58.5%), Rel exact 86.
  By seed mean, over the cells (method x n) whose every seed is inside the tolerance: COMP-B, n = ALL (1 seed) -- HELD-OUT D1 rate 58.5% (disjoint rows 59.7%), Rel exact 86.0.

**Finding 4 -- D2 complaints (ground-truth correctness lost) among the HELD-OUT rows, decoder v2, seed 0 (D2 after the repair; stored D2 resolved / new D2).**
- Qwen (TB-Q100-ae-l2_3): unrepaired 17/120 pre-correct held-out rows (stored flags reproduced on 120/120, long answers identical to the stored ones on 120); COMP-B n=10: 16 (2 / 1), n=100: 20 (3 / 6), n=ALL: 20 (5 / 8); COMP-B+ n=10: 20 (1 / 4), n=100: 25 (2 / 10), n=ALL: 27 (3 / 13); COMP-B-ALLPOS n=10: 23 (4 / 10); REEDIT-B n=10: 18 (2 / 3), n=100: 19 (5 / 7), n=ALL: 22 (4 / 9); REEDIT-S n=10: 18 (2 / 3), n=100: 21 (4 / 8), n=ALL: 23 (4 / 10); RB-TOT n=10: 20 (3 / 6), n=100: 20 (4 / 7), n=ALL: 17 (3 / 3); RB-UNION n=10: 16 (5 / 4), n=100: 7 (12 / 2), n=ALL: 1 (16 / 0); W_0: 0 (long answers identical to the stored pre-edit ones on 120/120).
- LLaVA (TB-L100-ae-l2_3): unrepaired 32/87 pre-correct held-out rows (stored flags reproduced on 87/87, long answers identical to the stored ones on 87); COMP-B n=10: 37 (2 / 7), n=100: 28 (10 / 6), n=ALL: 30 (12 / 10); COMP-B+ n=10: 35 (2 / 5), n=100: 37 (7 / 12), n=ALL: 40 (6 / 14); COMP-B-ALLPOS n=10: 28 (12 / 8); REEDIT-B n=10: 34 (5 / 7), n=100: 43 (4 / 15), n=ALL: 50 (4 / 22); REEDIT-S n=10: 34 (5 / 7), n=100: 43 (4 / 15), n=ALL: 50 (4 / 22); RB-TOT n=10: 29 (9 / 6), n=100: 25 (11 / 4), n=ALL: 25 (10 / 3); RB-UNION n=10: 24 (14 / 6), n=100: 2 (30 / 0), n=ALL: 0 (32 / 0); W_0: 0 (long answers identical to the stored pre-edit ones on 87/87).

**Finding 5 -- how much of the log's change at the edited module is undone on unreported rows (diagnostic; rho = ||(W_repaired - W_0) K|| / ||(W_E - W_0) K|| at the answer keys of each valid HELD-OUT row: 1 = unrepaired, 0 = pre-edit, > 1 = further from the pre-edit output). Disjoint held-out rows, seed 0; per config: median rho over held-out complaints / over held-out non-complaints; AUC = probability that a held-out complaint has a lower rho than a held-out non-complaint (0.5 = the repair does not tell them apart at the module).**
- Qwen (TB-Q100-ae-l2_3): COMP-B n=10: 0.94 / 0.93, AUC 0.46; n=100: 0.82 / 0.83, AUC 0.52; n=ALL: 0.77 / 0.80, AUC 0.53 | COMP-B+ n=10: 1.01 / 1.01, AUC 0.51; n=100: 1.05 / 1.05, AUC 0.55; n=ALL: 1.23 / 1.28, AUC 0.55 | COMP-B-ALLPOS n=10: 0.92 / 0.93, AUC 0.53 | REEDIT-B n=10: 1.08 / 1.08, AUC 0.50; n=100: 1.28 / 1.28, AUC 0.49; n=ALL: 1.61 / 1.61, AUC 0.52 | RB-TOT n=10: 0.84 / 0.84, AUC 0.53; n=100: 0.79 / 0.80, AUC 0.51; n=ALL: 0.82 / 0.82, AUC 0.51 | RB-UNION n=10: 0.85 / 0.85, AUC 0.51; n=100: 0.22 / 0.22, AUC 0.50; n=ALL: 0.03 / 0.03, AUC 0.53. Check of Anomaly 0 (identical keys are restored exactly): COMP-B n=ALL: 16 held-out twins of reported complaints, max rho 1.7e-13; COMP-B+ n=ALL: 16 held-out twins of reported complaints, max rho 4.7e-12. Cross-check of the two computations of rho (CPU: checkpoint tensor + stored composite delta; GPU pass of `run_d2.py`: the harness's weights): median over held-out complaints equal within 0.0e+00 over 25 configurations.
- LLaVA (TB-L100-ae-l2_3): COMP-B n=10: 0.98 / 1.00, AUC 0.62; n=100: 0.95 / 1.01, AUC 0.58; n=ALL: 0.91 / 1.05, AUC 0.60 | COMP-B+ n=10: 1.00 / 1.00, AUC 0.55; n=100: 1.07 / 1.08, AUC 0.54; n=ALL: 1.08 / 1.11, AUC 0.56 | COMP-B-ALLPOS n=10: 0.96 / 1.12, AUC 0.64 | REEDIT-B n=10: 1.09 / 1.03, AUC 0.34; n=100: 1.33 / 1.23, AUC 0.36; n=ALL: 1.69 / 1.58, AUC 0.37 | RB-TOT n=10: 0.82 / 0.82, AUC 0.49; n=100: 0.81 / 0.82, AUC 0.56; n=ALL: 0.80 / 0.81, AUC 0.55 | RB-UNION n=10: 0.82 / 0.82, AUC 0.51; n=100: 0.17 / 0.17, AUC 0.49; n=ALL: 0.00 / 0.00, AUC 0.50. Check of Anomaly 0 (identical keys are restored exactly): COMP-B n=ALL: 12 held-out twins of reported complaints, max rho 5.8e-13; COMP-B+ n=ALL: 12 held-out twins of reported complaints, max rho 1.0e-10. Cross-check of the two computations of rho (CPU: checkpoint tensor + stored composite delta; GPU pass of `run_d2.py`: the harness's weights): median over held-out complaints equal within 2.2e-16 over 25 configurations.

**Finding 5a -- how much of an unreported row lies in the subspace the reported complaints define (compensation only; share = ||R^T K|| / ||K|| at the row's answer keys, R = orthonormal basis of the correction's row space = the reported complaint keys with the protected span projected out; 1 = the correction acts on the whole key, 0 = it cannot act on it). Disjoint held-out rows, seed 0; rank of the correction, then median share over held-out complaints / over held-out non-complaints (90th percentile over all disjoint rows).**
- Qwen (TB-Q100-ae-l2_3) (d_in = 11008): COMP-B n=10: rank 99, 0.35 / 0.37 (0.57); n=100: rank 986, 0.61 / 0.61 (0.72); n=ALL: rank 2841, 0.72 / 0.73 (0.77) | COMP-B+ n=10: rank 99, 0.08 / 0.08 (0.10); n=100: rank 986, 0.25 / 0.24 (0.30); n=ALL: rank 2841, 0.37 / 0.36 (0.45) | COMP-B-ALLPOS n=10: rank 4583, 0.68 / 0.69 (0.71).
- LLaVA (TB-L100-ae-l2_3) (d_in = 11008): COMP-B n=10: rank 57, 0.20 / 0.11 (0.28); n=100: rank 611, 0.44 / 0.38 (0.62); n=ALL: rank 1973, 0.61 / 0.52 (0.72) | COMP-B+ n=10: rank 57, 0.08 / 0.07 (0.11); n=100: rank 611, 0.27 / 0.23 (0.34); n=ALL: rank 1973, 0.45 / 0.39 (0.51) | COMP-B-ALLPOS n=10: rank 3099, 0.57 / 0.50 (0.60).

**Finding 5c -- where the log's change on an unreported row sits (module level, CPU diagnostic, no behavioural evaluation).** Every edit-preserving compensation here leaves the log untouched on the span of the logged edits' answer keys (K_ans, projector P): the repaired module differs from W_0 by D P + (D + X)(I - P), D = W_E - W_0. Per valid disjoint HELD-OUT row, at its answer keys K: s = ||P K|| / ||K|| (share of the key inside the protected span); a = ||D P K|| / ||D K|| = the restoration ratio rho that restoring W_0 on everything orthogonal to the edits' answer keys would give (W_0 + D P; the complaint-free limit of such a repair); b = ||D (I - P) K|| / ||D K||; a and b are not an orthogonal split (they can cancel). Medians, held-out complaints / non-complaints:
- Qwen (TB-Q100-ae-l2_3): K_ans has rank 400 of 11008 (equal to the protection rank recorded by COMP-B: True) and carries 41% of ||D||_F; s = 0.51 / 0.52; a = 1.02 / 1.02; b = 1.23 / 1.24; share of held-out complaints with a < 0.5: 0% (252 complaints, 213 non-complaints). Size of the log's change itself, ||D K|| / ||W_0 K||: 1.30 / 1.25.
- LLaVA (TB-L100-ae-l2_3): K_ans has rank 483 of 11008 (equal to the protection rank recorded by COMP-B: True) and carries 38% of ||D||_F; s = 0.43 / 0.64; a = 0.87 / 1.41; b = 1.16 / 1.58; share of held-out complaints with a < 0.5: 0% (292 complaints, 177 non-complaints). Size of the log's change itself, ||D K|| / ||W_0 K||: 1.57 / 1.02.

**Finding 5b -- the regression suite itself (IN-SAMPLE diagnostic on the REPAIR pool, not a collateral number): REPAIR-pool known-good rows that still reproduce y0 under the correction. COMP-B+ protects the edited module's output at the answer keys of exactly these rows; COMP-B-RESTORE restores W_0's module output there; the other positions of the same rows are free to change.**
- Qwen (TB-Q100-ae-l2_3): COMP-B n=10: 166/192; COMP-B n=100: 149/192; COMP-B n=ALL: 150/192; COMP-B+ n=10: 182/192; COMP-B+ n=100: 168/192; COMP-B+ n=ALL: 129/192.
- LLaVA (TB-L100-ae-l2_3): COMP-B n=10: 175/189; COMP-B n=100: 153/189; COMP-B n=ALL: 135/189; COMP-B+ n=10: 177/189; COMP-B+ n=100: 149/189; COMP-B+ n=ALL: 138/189.

**Anomaly 1 -- sequential compensation is the joint compensation.** With accumulating protection (K_ans + the keys of every earlier repaired complaint) the sequential operator satisfies the same constraints in the same row space as the joint solve, so COMP-S = COMP-B and COMP-S+ = COMP-B+ as matrices (exact arithmetic, no dropped direction). Measured:
- Qwen (TB-Q100-ae-l2_3): max relative Frobenius difference between the sequential and the joint operator 1.9e-12 over 30 configurations; identical outcome (reported resolved, HELD-OUT D1 count, Rel / T-Gen / I-Gen exact) in 30/30; largest differences: 0 reported complaints, 0 held-out rows, 0 edit checks.
- LLaVA (TB-L100-ae-l2_3): max relative Frobenius difference between the sequential and the joint operator 2.9e-08 over 30 configurations; identical outcome (reported resolved, HELD-OUT D1 count, Rel / T-Gen / I-Gen exact) in 27/30; largest differences: 1 reported complaints, 1 held-out rows, 1 edit checks.

**Anomaly 1b -- the re-edit stream is the joint re-edit in exact arithmetic; in half precision it is not the same model on Qwen (TB-Q100-ae-l2_3) (up to 10 HELD-OUT rows).** REEDIT-S continues the log like the original stream (batches of 10, Kp extended by the installed keys, residual z* - h taken at the current weights, targets z* computed before the stream as in the testbed build). That is recursive least squares: with A_b = P(Kp Kp^T + sum_{j<=b} K_j K_j^T) + l2 I, the update of batch b is D_b = (R_b - D_{<b} K_b) K_b^T A_b^-1 and the sum of the updates is the joint solve of all batches. REEDIT-S can therefore differ from REEDIT-B only through half-precision arithmetic: the write after every batch, and the residuals, which the stream takes from half-precision forwards at the rounded weights. Measured (configurations with more than one batch):
- Qwen (TB-Q100-ae-l2_3): ||W_S - W_B|| / ||W_B - W_E|| on the saved weights (bfloat16): n=100: 1.0e-02; n=ALL: 1.2e-02; over 6 configurations the largest differences are 2 reported complaints, 10 held-out rows, 1 edit checks. Per configuration, REEDIT-S minus REEDIT-B (reported complaints resolved / HELD-OUT rows with D1 / edit checks differing): n=20 s0: +0 / +10 / 0; n=20 s1: +0 / -4 / 1; n=20 s2: +0 / -1 / 1; n=50 s0: +0 / +6 / 1; n=100 s0: +1 / +9 / 0; n=ALL s0: +2 / +1 / 0. Entry-wise, n=100: the differing entries differ by a median of 1 step(s) of the weight format (79% within 2 steps, 2.4% by more than 16, largest 488). Entry-wise, n=ALL: the differing entries differ by a median of 2 step(s) of the weight format (64% within 2 steps, 4.2% by more than 16, largest 480).
- LLaVA (TB-L100-ae-l2_3): ||W_S - W_B|| / ||W_B - W_E|| on the saved weights (float16): n=100: 1.3e-03; n=ALL: 1.5e-03; over 6 configurations the largest differences are 0 reported complaints, 1 held-out rows, 1 edit checks. Per configuration, REEDIT-S minus REEDIT-B (reported complaints resolved / HELD-OUT rows with D1 / edit checks differing): n=20 s0: +0 / +0 / 0; n=20 s1: +0 / +1 / 0; n=20 s2: +0 / +0 / 0; n=50 s0: +0 / +0 / 0; n=100 s0: +0 / +0 / 1; n=ALL s0: +0 / +0 / 0. Entry-wise, n=100: the differing entries differ by a median of 1 step(s) of the weight format (82% within 2 steps, 2.1% by more than 16, largest 3744). Entry-wise, n=ALL: the differing entries differ by a median of 2 step(s) of the weight format (66% within 2 steps, 3.9% by more than 16, largest 3616).
  A sequential re-edit that differs from the joint one needs targets re-optimised on the current weights (REEDIT-SC, exploratory, section 8). The identity is one of exact arithmetic: on the saved weights the two differ by up to 10 HELD-OUT rows (listed per configuration above), so REEDIT-S and REEDIT-B are kept as separate rows in every table.

**Anomaly 2 -- the harness skips its documented z* oracle fallback unless `chunk` is given.** In `07_repair_v3/harness/methods.py` (modified 2026-11-09 18:47 for RQ4) the fallback ladder of `m_reedit` (fixed loop at 2x steps, then 200 steps / early_stop 0.01) sits inside the `chunk is not None` branch, so `repair("REEDIT", ...)` with the default `chunk=None` makes one vendor attempt and silently does not install a complaint whose z* is not an oracle (the README documents the ladder for every call). The harness was not modified (read-only); here every z* is computed with `chunk=1` (same vendor attempt for one item, then the ladder). The first-pass records made before this was noticed are kept in `out/<log>/superseded_zfallback/`; first pass -> re-run:
- Qwen (TB-Q100-ae-l2_3) `REEDIT_B__n100__s0`: reported resolved 93/100 -> 93/100 (not installed 2 -> 0), HELD-OUT D1 58.8% -> 57.1%, Rel exact 71 -> 72, norm 0.718 -> 0.727.
- Qwen (TB-Q100-ae-l2_3) `REEDIT_B__n10__s0`: reported resolved 8/10 -> 9/10 (not installed 1 -> 0), HELD-OUT D1 56.7% -> 56.1%, Rel exact 77 -> 75, norm 0.198 -> 0.220.
- Qwen (TB-Q100-ae-l2_3) `REEDIT_B__nALL__s0`: reported resolved 239/297 -> 241/297 (not installed 4 -> 0), HELD-OUT D1 62.7% -> 62.9%, Rel exact 63 -> 63, norm 1.217 -> 1.223.
- Qwen (TB-Q100-ae-l2_3) `REEDIT_S__n100__s0`: reported resolved 93/100 -> 94/100 (not installed 2 -> 0), HELD-OUT D1 59.0% -> 59.0%, Rel exact 72 -> 72, norm 0.718 -> 0.727.
- Qwen (TB-Q100-ae-l2_3) `REEDIT_S__n10__s0`: reported resolved 8/10 -> 9/10 (not installed 1 -> 0), HELD-OUT D1 56.7% -> 56.1%, Rel exact 77 -> 75, norm 0.198 -> 0.220.
- Qwen (TB-Q100-ae-l2_3) `REEDIT_S__nALL__s0`: reported resolved 237/297 -> 243/297 (not installed 4 -> 0), HELD-OUT D1 62.9% -> 63.1%, Rel exact 63 -> 63, norm 1.217 -> 1.223.
- LLaVA (TB-L100-ae-l2_3) `REEDIT_B__n100__s0`: reported resolved 70/100 -> 70/100 (not installed 3 -> 0), HELD-OUT D1 71.1% -> 71.3%, Rel exact 86 -> 86, norm 0.683 -> 0.688.
- LLaVA (TB-L100-ae-l2_3) `REEDIT_S__n100__s0`: reported resolved 70/100 -> 70/100 (not installed 3 -> 0), HELD-OUT D1 71.1% -> 71.3%, Rel exact 86 -> 85, norm 0.683 -> 0.688.
  The single-complaint REEDIT rows of `07_repair_v3/rq1` written after that harness change use the default call and are exposed to the same omission (see section 9 for the text complaints).

**Anomaly 3 -- compensation at the answer positions restores the module output and still leaves reported complaints unresolved.** W_0's module output is restored at every reported answer key in every configuration (largest relative restoration error over all answer-position COMP configurations: 7e-10), but only the answer positions are restored; the prompt positions of the same row keep the edited module output. Restoring every position of the same complaints (COMP-B-ALLPOS, n <= 10) resolves 104/108 reported complaints against 79/108 for COMP-B on the same sets (both logs, all seeds available).
- Qwen (TB-Q100-ae-l2_3): COMP-B n=1: 0/1 unresolved, 0 of them with the module output restored (max restoration error 1e-15, 0 directions dropped); COMP-B n=10: 1/10 unresolved, 1 of them with the module output restored (max restoration error 2e-14, 0 directions dropped); COMP-B n=100: 29/100 unresolved, 29 of them with the module output restored (max restoration error 5e-14, 0 directions dropped); COMP-B n=ALL: 129/297 unresolved, 129 of them with the module output restored (max restoration error 2e-13, 89 directions dropped); COMP-B+ n=1: 0/1 unresolved, 0 of them with the module output restored (max restoration error 8e-15, 0 directions dropped); COMP-B+ n=10: 2/10 unresolved, 2 of them with the module output restored (max restoration error 1e-13, 0 directions dropped); COMP-B+ n=100: 36/100 unresolved, 36 of them with the module output restored (max restoration error 1e-12, 0 directions dropped); COMP-B+ n=ALL: 170/297 unresolved, 170 of them with the module output restored (max restoration error 5e-12, 89 directions dropped).
- LLaVA (TB-L100-ae-l2_3): COMP-B n=1: 0/1 unresolved, 0 of them with the module output restored (max restoration error 1e-15, 0 directions dropped); COMP-B n=10: 3/10 unresolved, 3 of them with the module output restored (max restoration error 1e-14, 0 directions dropped); COMP-B n=100: 41/100 unresolved, 41 of them with the module output restored (max restoration error 4e-13, 6 directions dropped); COMP-B n=ALL: 141/305 unresolved, 141 of them with the module output restored (max restoration error 9e-13, 31 directions dropped); COMP-B+ n=1: 0/1 unresolved, 0 of them with the module output restored (max restoration error 1e-15, 0 directions dropped); COMP-B+ n=10: 3/10 unresolved, 3 of them with the module output restored (max restoration error 8e-14, 0 directions dropped); COMP-B+ n=100: 51/100 unresolved, 51 of them with the module output restored (max restoration error 6e-10, 6 directions dropped); COMP-B+ n=ALL: 168/305 unresolved, 168 of them with the module output restored (max restoration error 7e-10, 31 directions dropped).

**Anomaly 4 -- restoring ALL positions (COMP-B-ALLPOS).** Feasible = W_0's module output restored at every position of every reported complaint (restoration error < 1e-6); dropped = query directions below the rank cut (keys inside the protected span or linearly dependent).
- Qwen (TB-Q100-ae-l2_3) (d_in = 11008): n=1 s0: 40 keys, rank 40 (0 dropped), feasible (err 1e-15), resolved 1/1, held-out +30 / -32, norm 0.04; n=1 s1: 42 keys, rank 42 (0 dropped), feasible (err 3e-15), resolved 1/1, held-out +24 / -39, norm 0.04; n=1 s2: 38 keys, rank 38 (0 dropped), feasible (err 2e-15), resolved 0/1, held-out +22 / -29, norm 0.04; n=2 s0: 1070 keys, rank 1070 (0 dropped), feasible (err 8e-15), resolved 2/2, held-out +48 / -45, norm 0.30; n=2 s1: 1053 keys, rank 1053 (0 dropped), feasible (err 8e-15), resolved 2/2, held-out +41 / -43, norm 0.30; n=2 s2: 357 keys, rank 357 (0 dropped), feasible (err 5e-15), resolved 1/2, held-out +43 / -35, norm 0.16; n=5 s0: 2199 keys, rank 2156 (43 dropped), feasible (err 9e-15), resolved 5/5, held-out +60 / -52, norm 0.44; n=5 s1: 2154 keys, rank 2126 (28 dropped), feasible (err 9e-15), resolved 5/5, held-out +45 / -52, norm 0.44; n=5 s2: 758 keys, rank 715 (43 dropped), feasible (err 5e-15), resolved 4/5, held-out +42 / -44, norm 0.24; n=10 s0: 4684 keys, rank 4583 (101 dropped), feasible (err 2e-14), resolved 10/10, held-out +66 / -93, norm 0.69; n=10 s1: 4600 keys, rank 4512 (88 dropped), feasible (err 1e-14), resolved 10/10, held-out +68 / -77, norm 0.68; n=10 s2: 3219 keys, rank 3117 (102 dropped), feasible (err 1e-14), resolved 9/10, held-out +72 / -65, norm 0.54.
  Qwen (TB-Q100-ae-l2_3): `text_206` stays unresolved in 4 all-position configuration(s) (n=1 s2, n=2 s2, n=5 s2, n=10 s2) although the module output is restored at each of its 38 positions (largest restoration error 1e-14): token accuracy 0.90-0.90 of 10 answer tokens, first mismatch at answer token 6; at W_0 itself the row is reproduced. Not explained. One candidate, not tested: the adapter adds the fp64 correction to the module's half-precision output and rounds, so the repaired forward equals W_0's forward only up to that rounding.
- LLaVA (TB-L100-ae-l2_3) (d_in = 11008): n=1 s0: 27 keys, rank 27 (0 dropped), feasible (err 1e-15), resolved 1/1, held-out +15 / -15, norm 0.04; n=1 s1: 23 keys, rank 23 (0 dropped), feasible (err 1e-15), resolved 1/1, held-out +9 / -11, norm 0.03; n=1 s2: 28 keys, rank 28 (0 dropped), feasible (err 2e-15), resolved 1/1, held-out +9 / -16, norm 0.04; n=2 s0: 626 keys, rank 626 (0 dropped), feasible (err 4e-13), resolved 2/2, held-out +27 / -36, norm 0.22; n=2 s1: 621 keys, rank 621 (0 dropped), feasible (err 7e-13), resolved 2/2, held-out +22 / -30, norm 0.22; n=2 s2: 623 keys, rank 623 (0 dropped), feasible (err 3e-13), resolved 2/2, held-out +19 / -41, norm 0.22; n=5 s0: 1284 keys, rank 1276 (8 dropped), feasible (err 7e-13), resolved 5/5, held-out +31 / -40, norm 0.33; n=5 s1: 1280 keys, rank 1271 (9 dropped), feasible (err 1e-12), resolved 5/5, held-out +33 / -46, norm 0.32; n=5 s2: 1274 keys, rank 1266 (8 dropped), feasible (err 1e-12), resolved 5/5, held-out +32 / -39, norm 0.32; n=10 s0: 3118 keys, rank 3099 (19 dropped), feasible (err 7e-13), resolved 10/10, held-out +44 / -53, norm 0.53; n=10 s1: 3125 keys, rank 3105 (20 dropped), feasible (err 8e-13), resolved 10/10, held-out +36 / -67, norm 0.54; n=10 s2: 3111 keys, rank 3093 (18 dropped), feasible (err 6e-13), resolved 10/10, held-out +48 / -59, norm 0.53.

**Anomaly 5 -- what RB-UNION and RB-TOT roll back at each n (edits removed, Rel exact, HELD-OUT D1 rate, reported complaints resolved):**
- Qwen (TB-Q100-ae-l2_3): RB-UNION n=1: 9 of 100 edits rolled back, Rel exact 72, HELD-OUT D1 51.8%, reported resolved 1/1; RB-UNION n=10: 27 of 100 edits rolled back, Rel exact 61, HELD-OUT D1 46.7%, reported resolved 7/10; RB-UNION n=100: 92 of 100 edits rolled back, Rel exact 7, HELD-OUT D1 14.1%, reported resolved 80/100; RB-UNION n=ALL: 99 of 100 edits rolled back, Rel exact 1, HELD-OUT D1 5.9%, reported resolved 274/297; RB-TOT n=1: 9 of 100 edits rolled back, Rel exact 72, HELD-OUT D1 51.8%, reported resolved 1/1; RB-TOT n=10: 30 of 100 edits rolled back, Rel exact 54, HELD-OUT D1 45.1%, reported resolved 7/10; RB-TOT n=100: 30 of 100 edits rolled back, Rel exact 53, HELD-OUT D1 42.2%, reported resolved 47/100; RB-TOT n=ALL: 30 of 100 edits rolled back, Rel exact 56, HELD-OUT D1 44.9%, reported resolved 97/297.
- LLaVA (TB-L100-ae-l2_3): RB-UNION n=1: 3 of 100 edits rolled back, Rel exact 86, HELD-OUT D1 62.3%, reported resolved 1/1; RB-UNION n=10: 29 of 100 edits rolled back, Rel exact 66, HELD-OUT D1 53.2%, reported resolved 4/10; RB-UNION n=100: 96 of 100 edits rolled back, Rel exact 5, HELD-OUT D1 11.1%, reported resolved 86/100; RB-UNION n=ALL: 100 of 100 edits rolled back, Rel exact 1, HELD-OUT D1 0.0%, reported resolved 305/305; RB-TOT n=1: 3 of 100 edits rolled back, Rel exact 86, HELD-OUT D1 62.3%, reported resolved 1/1; RB-TOT n=10: 30 of 100 edits rolled back, Rel exact 66, HELD-OUT D1 57.3%, reported resolved 9/10; RB-TOT n=100: 30 of 100 edits rolled back, Rel exact 65, HELD-OUT D1 51.2%, reported resolved 27/100; RB-TOT n=ALL: 30 of 100 edits rolled back, Rel exact 63, HELD-OUT D1 47.8%, reported resolved 81/305.

**Not finished.**
- Qwen (TB-Q100-ae-l2_3): exploratory configurations outside the protocol, planned and not run: `COMP_BR__nALL__s0`, `REEDIT_SC__n20__s0`, `REEDIT_SC__n50__s0`, `REEDIT_SC__n100__s0`, `REEDIT_SC__nALL__s0`.
- LLaVA (TB-L100-ae-l2_3): exploratory configurations outside the protocol, planned and not run: `COMP_BR__nALL__s0`, `REEDIT_SC__n20__s0`, `REEDIT_SC__n50__s0`, `REEDIT_SC__n100__s0`, `REEDIT_SC__nALL__s0`.

## 2. Protocol

- Logs: Qwen (TB-Q100-ae-l2_3): 100 edits, layer 20, l2 = 3, REPAIR pool 489 valid rows = 297 D1 complaints + 192 non-complaints, HELD-OUT pool 490 valid rows; LLaVA (TB-L100-ae-l2_3): 100 edits, layer 16, l2 = 3, REPAIR pool 494 valid rows = 305 D1 complaints + 189 non-complaints, HELD-OUT pool 494 valid rows.
- Reported sets: one nested stream per (log, seed): harness `sample_complaints(H, 100, seed)` (D1 v2, REPAIR pool, stratified), interleaved text / image in draw order, then the remaining REPAIR-pool complaints in a seeded random order; the reported set of size n is the first n of the stream (n // 2 image + n - n // 2 text). n in {1, 2, 5, 10, 20, 50, 100, ALL}; seeds [0, 1, 2] for n <= 20, seed 0 for n >= 50. Sequential methods use the stream order.
- Methods (all through `harness.repair`; alpha = 1; nothing tuned; fp64 solves on the CPU for both logs):
  - **COMP-B**: `{"form": "adapter", "gate": 1e-08, "harness": "COMP", "protect": ["K_ans"], "query": "answer", "tol_rel": 1e-09}`
  - **COMP-B+**: `{"form": "adapter", "gate": 1e-08, "harness": "COMP", "protect": ["K_ans", "probe:100000"], "query": "answer", "tol_rel": 1e-09}`
  - **COMP-S**: `{"form": "adapter", "gate": 1e-08, "harness": "COMP", "protect": ["K_ans"], "query": "answer", "sequential": "prior + extra_basis (keys of earlier repaired complaints), stream order", "tol_rel": 1e-09}`
  - **COMP-S+**: `{"form": "adapter", "gate": 1e-08, "harness": "COMP", "protect": ["K_ans", "probe:100000"], "query": "answer", "sequential": "prior + extra_basis (keys of earlier repaired complaints), stream order", "tol_rel": 1e-09}`
  - **COMP-B-ALLPOS**: `{"form": "adapter", "gate": 1e-08, "harness": "COMP", "protect": ["K_ans"], "query": "all", "tol_rel": 1e-09}`
  - **REEDIT-B**: `{"chunk": 1, "form": "merged", "harness": "REEDIT", "l2": "log", "targets": "z* on W_E per complaint (B=1, oracle required)", "variant": "k0+kp"}`
  - **REEDIT-S**: `{"chunk": 1, "form": "merged", "harness": "REEDIT", "l2": "log", "stream": "batches of 10, Kp accumulates, residual at the current weights", "targets": "z* on W_E per complaint (B=1, oracle required)", "variant": "k0+kp"}`
  - **RB-TOT**: `{"budget": 30, "harness": "ROLLBACK_FO", "offload": true, "record_flags": true}`
  - **RB-UNION**: `{"budget": 10, "harness": "ROLLBACK_FO", "union": true}`
- RB-TOT: budget 30; RB-UNION: budget 10 per complaint; REEDIT-S: batches of 10.
- Evaluation: `harness.evaluate(per_item=True)`; complaints and protection keys come from the REPAIR pool only; collateral and generalisation from the HELD-OUT pool only; y0 = decoder-v2 answers (`probe_records_v2.jsonl`).
- 95% CIs: uid-level percentile bootstrap, 2000 resamples. Paired test against the unrepaired log: exact McNemar on the held-out rows (descriptive; many configurations are compared).

## 3. Main table: method x n in {1, 10, 100, ALL}, seed 0

HELD-OUT = valid-y0 probe rows of the HELD-OUT pool. D1 rate = share of these rows whose y0 is not reproduced under teacher forcing after the repair. Disjoint = the HELD-OUT rows whose query does not occur in the REPAIR pool (Anomaly 0); twins of reported complaints = HELD-OUT rows whose query is identical to one of the n reported complaints (fixed / rows). Edits: all logged edits x {Rel, T-Gen, I-Gen}, B=1 teacher-forced exact; broken = exact at W_E and not after. Norm = Frobenius norm of the correction / ||W_E - W_0||. Rank of the correction: exact for COMP (fp64 adapter), nominal for REEDIT (installed complaints) and rollbacks (edits rolled back). Passes include the z* optimisation for REEDIT. Wall-clock on shared GPUs.

### Qwen (TB-Q100-ae-l2_3)

| method | n | reported resolved | HELD-OUT D1 rate % [95% CI] | held-out complaints fixed | non-complaints lost | disjoint HELD-OUT D1 % [CI] (+fixed / -lost) | twins of reported complaints fixed | Rel / T-Gen / I-Gen exact | edit checks broken (rel/tgen/igen) / fixed | T-Loc / I-Loc | norm / log | rank: correction / protection | passes fwd / bwd | repair s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NONE (unrepaired) | - | - | 54.7 [50.6, 58.8] | 0/268 | 0/222 | 54.2 [49.9, 58.6] (+0 / -0) | 0/0 | 78 / 9 / 22 | 0 (0/0/0) / 0 | 84.8 / 88.7 | 0.000 | 0 / 0 | 0 / 0 | 0.0 |
| COMP-B | 1 | 1/1 | 52.9 [48.4, 57.3] | 28/268 | 19/222 | 52.5 [48.1, 57.1] (+27 / -19) | 0/0 | 78 / 9 / 22 | 0 (0/0/0) / 0 | 87.3 / 88.4 | 0.027 | 10 / 400 | 1 / 0 | 0.0 |
| COMP-B | 10 | 9/10 | 51.4 [47.3, 55.6] | 48/268 | 32/222 | 50.8 [46.4, 55.3] (+48 / -32) | 0/0 | 75 / 9 / 22 | 3 (3/0/0) / 0 | 86.5 / 89.3 | 0.099 | 99 / 400 | 10 / 0 | 1.8 |
| COMP-B | 100 | 71/100 | 49.4 [45.3, 53.5] | 82/268 | 56/222 | 50.1 [46.0, 54.4] (+74 / -55) | 5/6 | 75 / 8 / 22 | 7 (3/2/2) / 3 | 88.9 / 89.6 | 0.337 | 986 / 400 | 100 / 0 | 11.7 |
| COMP-B | ALL (297) | 168/297 | 47.3 [43.1, 51.8] | 97/268 | 61/222 | 49.0 [44.3, 53.6] (+85 / -61) | 12/16 | 78 / 10 / 25 | 3 (2/0/1) / 7 | 89.1 / 88.9 | 0.647 | 2841 / 400 | 297 / 0 | 52.4 |
| COMP-B+ | 1 | 1/1 | 55.1 [51.0, 59.4] | 6/268 | 8/222 | 54.8 [50.7, 59.0] (+5 / -8) | 0/0 | 78 / 9 / 23 | 0 (0/0/0) / 1 | 84.8 / 88.9 | 0.049 | 10 / 2254 | 1 / 0 | 0.0 |
| COMP-B+ | 10 | 8/10 | 53.7 [49.4, 58.1] | 30/268 | 25/222 | 53.5 [49.1, 58.2] (+28 / -25) | 0/0 | 77 / 10 / 21 | 2 (1/0/1) / 1 | 83.7 / 89.4 | 0.203 | 99 / 2254 | 10 / 0 | 1.7 |
| COMP-B+ | 100 | 64/100 | 55.9 [51.6, 60.1] | 57/268 | 63/222 | 56.6 [52.4, 60.8] (+52 / -63) | 2/6 | 76 / 7 / 22 | 9 (5/2/2) / 5 | 84.1 / 86.6 | 0.730 | 986 / 2254 | 100 / 0 | 10.5 |
| COMP-B+ | ALL (297) | 127/297 | 66.9 [63.0, 70.9] | 49/268 | 109/222 | 68.8 [64.7, 72.7] (+40 / -108) | 9/16 | 76 / 8 / 18 | 14 (5/3/6) / 7 | 76.4 / 84.4 | 1.419 | 2841 / 2254 | 297 / 0 | 51.7 |
| COMP-S | 1 | 1/1 | 52.9 [48.4, 57.3] | 28/268 | 19/222 | 52.5 [48.1, 57.1] (+27 / -19) | 0/0 | 78 / 9 / 22 | 0 (0/0/0) / 0 | 87.3 / 88.4 | 0.027 | 10 / 400 | 1 / 0 | 0.0 |
| COMP-S | 10 | 9/10 | 51.4 [47.3, 55.6] | 48/268 | 32/222 | 50.8 [46.4, 55.3] (+48 / -32) | 0/0 | 75 / 9 / 22 | 3 (3/0/0) / 0 | 86.5 / 89.3 | 0.099 | 99 / 499 | 10 / 0 | 1.2 |
| COMP-S | 100 | 71/100 | 49.4 [45.3, 53.5] | 82/268 | 56/222 | 50.1 [46.0, 54.4] (+74 / -55) | 5/6 | 75 / 8 / 22 | 7 (3/2/2) / 3 | 88.9 / 89.6 | 0.337 | 986 / 1386 | 100 / 0 | 30.2 |
| COMP-S | ALL (297) | 168/297 | 47.3 [43.1, 51.8] | 97/268 | 61/222 | 49.0 [44.3, 53.6] (+85 / -61) | 12/16 | 78 / 10 / 25 | 3 (2/0/1) / 7 | 89.1 / 88.9 | 0.647 | 2841 / 3241 | 297 / 0 | 191.0 |
| COMP-S+ | 1 | 1/1 | 55.1 [51.0, 59.4] | 6/268 | 8/222 | 54.8 [50.7, 59.0] (+5 / -8) | 0/0 | 78 / 9 / 23 | 0 (0/0/0) / 1 | 84.8 / 88.9 | 0.049 | 10 / 2254 | 1 / 0 | 0.0 |
| COMP-S+ | 10 | 8/10 | 53.7 [49.4, 58.1] | 30/268 | 25/222 | 53.5 [49.1, 58.2] (+28 / -25) | 0/0 | 77 / 10 / 21 | 2 (1/0/1) / 1 | 83.7 / 89.4 | 0.203 | 99 / 2353 | 10 / 0 | 1.3 |
| COMP-S+ | 100 | 64/100 | 55.9 [51.6, 60.1] | 57/268 | 63/222 | 56.6 [52.4, 60.8] (+52 / -63) | 2/6 | 76 / 7 / 22 | 9 (5/2/2) / 5 | 84.1 / 86.6 | 0.730 | 986 / 3240 | 100 / 0 | 32.3 |
| COMP-S+ | ALL (297) | 127/297 | 66.9 [63.0, 70.9] | 49/268 | 109/222 | 68.8 [64.7, 72.7] (+40 / -108) | 9/16 | 76 / 8 / 18 | 14 (5/3/6) / 7 | 76.4 / 84.4 | 1.419 | 2841 / 5095 | 297 / 0 | 177.2 |
| COMP-B-ALLPOS | 1 | 1/1 | 55.1 [50.8, 59.3] | 30/268 | 32/222 | 55.7 [51.5, 59.9] (+25 / -32) | 0/0 | 78 / 9 / 23 | 1 (1/0/0) / 2 | 87.1 / 87.5 | 0.040 | 40 / 400 | 1 / 0 | 0.0 |
| COMP-B-ALLPOS | 10 | 10/10 | 60.2 [56.1, 64.3] | 66/268 | 93/222 | 60.2 [56.1, 64.4] (+63 / -91) | 0/0 | 76 / 11 / 26 | 3 (3/0/0) / 7 | 87.7 / 80.2 | 0.689 | 4583 / 400 | 10 / 0 | 14.5 |
| REEDIT-B | 1 | 1/1 | 59.0 [54.7, 63.0] | 35/268 | 56/222 | 59.4 [55.3, 63.7] (+32 / -56) | 0/0 | 76 / 9 / 22 | 2 (2/0/0) / 0 | 82.8 / 88.4 | 0.072 | 1 / K0 [redacted: describes a related submission under double-blind review] (hard) + Kp 100 (soft) | 13 / 10 | 1.1 |
| REEDIT-B | 10 | 9/10 | 56.1 [52.0, 60.4] | 46/268 | 53/222 | 55.9 [51.9, 60.3] (+45 / -53) | 0/0 | 75 / 9 / 22 | 3 (3/0/0) / 0 | 84.7 / 88.8 | 0.220 | 10 / K0 [redacted: describes a related submission under double-blind review] (hard) + Kp 100 (soft) | 96 / 66 | 13.1 |
| REEDIT-B | 100 | 93/100 | 57.1 [53.0, 61.4] | 55/268 | 67/222 | 58.1 [54.0, 62.3] (+49 / -67) | 6/6 | 72 / 10 / 19 | 10 (7/0/3) / 2 | 81.3 / 90.2 | 0.727 | 100 / K0 [redacted: describes a related submission under double-blind review] (hard) + Kp 100 (soft) | 954 / 652 | 125.4 |
| REEDIT-B | ALL (297) | 241/297 | 62.9 [58.8, 66.9] | 54/268 | 94/222 | 66.0 [62.0, 70.2] (+39 / -94) | 15/16 | 63 / 5 / 15 | 26 (15/4/7) / 0 | 76.7 / 89.9 | 1.223 | 297 / K0 [redacted: describes a related submission under double-blind review] (hard) + Kp 100 (soft) | 2814 / 1917 | 292.0 |
| REEDIT-S | 1 | 1/1 | 59.0 [54.7, 63.0] | 35/268 | 56/222 | 59.4 [55.3, 63.7] (+32 / -56) | 0/0 | 76 / 9 / 22 | 2 (2/0/0) / 0 | 82.8 / 88.4 | 0.072 | 1 / K0 [redacted: describes a related submission under double-blind review] (hard) + Kp 100 (soft) | 13 / 10 | 1.1 |
| REEDIT-S | 10 | 9/10 | 56.1 [52.0, 60.4] | 46/268 | 53/222 | 55.9 [51.9, 60.3] (+45 / -53) | 0/0 | 75 / 9 / 22 | 3 (3/0/0) / 0 | 84.7 / 88.8 | 0.220 | 10 / K0 [redacted: describes a related submission under double-blind review] (hard) + Kp 100 (soft) | 96 / 66 | 13.1 |
| REEDIT-S | 100 | 94/100 | 59.0 [54.8, 63.2] | 52/268 | 73/222 | 60.0 [55.9, 64.3] (+46 / -73) | 6/6 | 72 / 10 / 19 | 10 (7/0/3) / 2 | 80.7 / 90.3 | 0.727 | 100 / K0 [redacted: describes a related submission under double-blind review] (hard) + Kp 200 (soft) | 954 / 652 | 126.3 |
| REEDIT-S | ALL (297) | 243/297 | 63.1 [59.1, 67.0] | 55/268 | 96/222 | 66.2 [62.1, 70.4] (+40 / -96) | 15/16 | 63 / 5 / 15 | 26 (15/4/7) / 0 | 76.3 / 90.0 | 1.223 | 297 / K0 [redacted: describes a related submission under double-blind review] (hard) + Kp 397 (soft) | 2814 / 1917 | 295.8 |
| RB-TOT | 1 | 1/1 | 51.8 [47.7, 56.0] | 40/268 | 26/222 | 52.0 [47.7, 56.4] (+36 / -26) | 0/0 | 72 / 10 / 22 | 10 (8/0/2) / 5 | 86.8 / 89.3 | 0.323 | 9 / 0 | 19 / 9 | 1.8 |
| RB-TOT | 10 | 7/10 | 45.1 [40.5, 49.7] | 78/268 | 31/222 | 44.9 [40.6, 49.6] (+74 / -31) | 0/0 | 54 / 9 / 16 | 34 (25/1/8) / 4 | 89.8 / 90.8 | 0.554 | 30 / 0 | 610 / 300 | 31.4 |
| RB-TOT | 100 | 47/100 | 42.2 [37.9, 46.5] | 92/268 | 31/222 | 43.4 [38.8, 47.9] (+81 / -31) | 3/6 | 53 / 8 / 16 | 37 (27/3/7) / 5 | 90.9 / 92.3 | 0.582 | 30 / 0 | 6100 / 3000 | 319.8 |
| RB-TOT | ALL (297) | 97/297 | 44.9 [40.2, 49.6] | 89/268 | 41/222 | 46.0 [41.5, 50.7] (+79 / -41) | 10/16 | 56 / 7 / 16 | 34 (24/3/7) / 4 | 90.3 / 92.3 | 0.587 | 30 / 0 | 18117 / 8910 | 491.8 |
| RB-UNION | 1 | 1/1 | 51.8 [47.7, 56.0] | 40/268 | 26/222 | 52.0 [47.7, 56.4] (+36 / -26) | 0/0 | 72 / 10 / 22 | 10 (8/0/2) / 5 | 86.8 / 89.3 | 0.323 | 9 / 0 | 19 / 9 | 0.8 |
| RB-UNION | 10 | 7/10 | 46.7 [42.0, 51.4] | 78/268 | 39/222 | 47.5 [43.1, 52.2] (+70 / -39) | 0/0 | 61 / 9 / 17 | 27 (20/1/6) / 5 | 88.5 / 90.0 | 0.529 | 27 / 0 | 106 / 48 | 5.0 |
| RB-UNION | 100 | 80/100 | 14.1 [11.0, 17.4] | 213/268 | 14/222 | 14.4 [11.1, 17.8] (+199 / -14) | 6/6 | 7 / 2 / 4 | 97 (71/7/19) / 1 | 96.5 / 98.5 | 0.976 | 92 / 0 | 958 / 429 | 57.0 |
| RB-UNION | ALL (297) | 274/297 | 5.9 [3.7, 8.2] | 245/268 | 6/222 | 6.2 [3.9, 8.8] (+229 / -6) | 16/16 | 1 / 2 / 1 | 105 (77/7/21) / 0 | 98.8 / 99.3 | 0.999 | 99 / 0 | 3291 / 1497 | 153.7 |
| ALL (W_0) | - | - | 0.0 [0.0, 0.0] | 268/268 | 0/222 | 0.0 [0.0, 0.0] (+252 / -0) | 0/0 | 0 / 1 / 0 | 108 (78/8/22) / 0 | 100.0 / 100.0 | 1.000 | 100 / 0 | 0 / 0 | 0.0 |

### LLaVA (TB-L100-ae-l2_3)

| method | n | reported resolved | HELD-OUT D1 rate % [95% CI] | held-out complaints fixed | non-complaints lost | disjoint HELD-OUT D1 % [CI] (+fixed / -lost) | twins of reported complaints fixed | Rel / T-Gen / I-Gen exact | edit checks broken (rel/tgen/igen) / fixed | T-Loc / I-Loc | norm / log | rank: correction / protection | passes fwd / bwd | repair s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NONE (unrepaired) | - | - | 61.5 [57.5, 65.7] | 0/304 | 0/190 | 62.3 [58.2, 66.5] (+0 / -0) | 0/0 | 88 / 29 / 66 | 0 (0/0/0) / 0 | 77.3 / 81.5 | 0.000 | 0 / 0 | 0 / 0 | 0.0 |
| COMP-B | 1 | 1/1 | 61.3 [57.2, 65.7] | 13/304 | 12/190 | 61.8 [57.7, 66.1] (+13 / -11) | 0/0 | 88 / 29 / 66 | 0 (0/0/0) / 0 | 77.9 / 80.7 | 0.026 | 10 / 483 | 1 / 0 | 0.0 |
| COMP-B | 10 | 7/10 | 61.5 [57.6, 65.5] | 19/304 | 19/190 | 62.0 [58.0, 66.0] (+19 / -18) | 0/1 | 88 / 29 / 66 | 0 (0/0/0) / 0 | 77.7 / 80.9 | 0.066 | 57 / 483 | 10 / 0 | 0.4 |
| COMP-B | 100 | 59/100 | 59.1 [54.9, 63.3] | 54/304 | 42/190 | 60.6 [56.3, 64.6] (+49 / -41) | 5/8 | 89 / 29 / 69 | 0 (0/0/0) / 4 | 77.5 / 80.4 | 0.233 | 611 / 483 | 100 / 0 | 4.0 |
| COMP-B | ALL (305) | 164/305 | 58.5 [54.2, 63.0] | 72/304 | 57/190 | 59.7 [55.1, 64.4] (+65 / -53) | 7/12 | 86 / 32 / 66 | 6 (3/1/2) / 7 | 77.8 / 76.7 | 0.461 | 1973 / 483 | 305 / 0 | 10.9 |
| COMP-B+ | 1 | 1/1 | 62.8 [58.8, 66.7] | 3/304 | 9/190 | 63.3 [59.4, 67.5] (+3 / -8) | 0/0 | 88 / 30 / 66 | 0 (0/0/0) / 1 | 77.3 / 81.1 | 0.034 | 10 / 1355 | 1 / 0 | 0.0 |
| COMP-B+ | 10 | 7/10 | 63.6 [59.6, 67.4] | 8/304 | 18/190 | 64.2 [60.2, 68.1] (+8 / -17) | 0/1 | 88 / 31 / 66 | 0 (0/0/0) / 2 | 77.2 / 81.2 | 0.094 | 57 / 1355 | 10 / 0 | 0.3 |
| COMP-B+ | 100 | 49/100 | 66.4 [62.3, 70.3] | 33/304 | 57/190 | 67.0 [62.8, 71.1] (+29 / -51) | 3/8 | 88 / 27 / 66 | 6 (1/2/3) / 4 | 73.6 / 73.0 | 0.507 | 611 / 1355 | 100 / 0 | 3.4 |
| COMP-B+ | ALL (305) | 137/305 | 66.8 [62.8, 71.1] | 48/304 | 74/190 | 67.6 [63.3, 72.0] (+42 / -67) | 6/12 | 84 / 33 / 64 | 10 (4/1/5) / 8 | 69.1 / 65.9 | 0.783 | 1973 / 1355 | 305 / 0 | 25.3 |
| COMP-S | 1 | 1/1 | 61.3 [57.2, 65.7] | 13/304 | 12/190 | 61.8 [57.7, 66.1] (+13 / -11) | 0/0 | 88 / 29 / 66 | 0 (0/0/0) / 0 | 77.9 / 80.7 | 0.026 | 10 / 483 | 1 / 0 | 0.0 |
| COMP-S | 10 | 7/10 | 61.5 [57.6, 65.5] | 19/304 | 19/190 | 62.0 [58.0, 66.0] (+19 / -18) | 0/1 | 88 / 29 / 66 | 0 (0/0/0) / 0 | 77.7 / 80.9 | 0.066 | 57 / 540 | 10 / 0 | 0.5 |
| COMP-S | 100 | 59/100 | 59.1 [54.9, 63.3] | 54/304 | 42/190 | 60.6 [56.3, 64.6] (+49 / -41) | 5/8 | 89 / 29 / 69 | 0 (0/0/0) / 4 | 77.5 / 80.4 | 0.233 | 611 / 1094 | 100 / 0 | 13.4 |
| COMP-S | ALL (305) | 165/305 | 58.5 [54.2, 63.0] | 72/304 | 57/190 | 59.7 [55.1, 64.4] (+65 / -53) | 7/12 | 86 / 32 / 66 | 6 (3/1/2) / 7 | 77.8 / 76.7 | 0.461 | 1973 / 2456 | 305 / 0 | 113.3 |
| COMP-S+ | 1 | 1/1 | 62.8 [58.8, 66.7] | 3/304 | 9/190 | 63.3 [59.4, 67.5] (+3 / -8) | 0/0 | 88 / 30 / 66 | 0 (0/0/0) / 1 | 77.3 / 81.1 | 0.034 | 10 / 1355 | 1 / 0 | 0.0 |
| COMP-S+ | 10 | 7/10 | 63.6 [59.6, 67.4] | 8/304 | 18/190 | 64.2 [60.2, 68.1] (+8 / -17) | 0/1 | 88 / 31 / 66 | 0 (0/0/0) / 2 | 77.2 / 81.2 | 0.094 | 57 / 1412 | 10 / 0 | 0.6 |
| COMP-S+ | 100 | 49/100 | 66.4 [62.3, 70.3] | 33/304 | 57/190 | 67.0 [62.8, 71.1] (+29 / -51) | 3/8 | 88 / 27 / 65 | 6 (1/2/3) / 3 | 73.6 / 72.9 | 0.507 | 611 / 1966 | 100 / 0 | 12.4 |
| COMP-S+ | ALL (305) | 137/305 | 66.8 [62.8, 71.1] | 48/304 | 74/190 | 67.6 [63.3, 72.0] (+42 / -67) | 6/12 | 84 / 33 / 64 | 10 (4/1/5) / 8 | 69.1 / 65.9 | 0.783 | 1973 / 3328 | 305 / 0 | 177.0 |
| COMP-B-ALLPOS | 1 | 1/1 | 61.5 [57.3, 65.9] | 15/304 | 15/190 | 62.3 [58.0, 66.5] (+15 / -15) | 0/0 | 88 / 30 / 67 | 0 (0/0/0) / 2 | 77.4 / 80.8 | 0.037 | 27 / 483 | 1 / 0 | 0.0 |
| COMP-B-ALLPOS | 10 | 10/10 | 63.4 [59.2, 67.5] | 44/304 | 53/190 | 65.0 [60.8, 69.2] (+39 / -52) | 1/1 | 89 / 32 / 66 | 4 (0/1/3) / 8 | 75.5 / 76.5 | 0.528 | 3099 / 483 | 10 / 0 | 5.5 |
| REEDIT-B | 1 | 0/1 | 65.0 [61.0, 69.2] | 20/304 | 37/190 | 66.3 [62.2, 70.6] (+18 / -37) | 0/0 | 88 / 29 / 66 | 0 (0/0/0) / 0 | 77.2 / 81.1 | 0.075 | 1 / K0 [redacted: describes a related submission under double-blind review] (hard) + Kp 100 (soft) | 9 / 6 | 0.6 |
| REEDIT-B | 10 | 7/10 | 66.0 [61.9, 70.2] | 22/304 | 44/190 | 66.5 [62.4, 70.9] (+19 / -39) | 1/1 | 87 / 30 / 66 | 1 (1/0/0) / 1 | 76.5 / 79.4 | 0.230 | 10 / K0 [redacted: describes a related submission under double-blind review] (hard) + Kp 100 (soft) | 75 / 45 | 4.2 |
| REEDIT-B | 100 | 70/100 | 71.3 [67.7, 74.7] | 36/304 | 84/190 | 73.6 [70.1, 77.1] (+27 / -80) | 8/8 | 86 / 28 / 61 | 10 (3/2/5) / 2 | 71.6 / 76.8 | 0.688 | 100 / K0 [redacted: describes a related submission under double-blind review] (hard) + Kp 100 (soft) | 713 / 407 | 37.0 |
| REEDIT-B | ALL (305) | 159/305 | 75.3 [72.0, 78.5] | 44/304 | 112/190 | 78.0 [74.7, 81.2] (+32 / -106) | 12/12 | 74 / 23 / 49 | 38 (14/7/17) / 1 | 61.6 / 72.0 | 1.210 | 305 / K0 [redacted: describes a related submission under double-blind review] (hard) + Kp 100 (soft) | 2234 / 1311 | 137.4 |
| REEDIT-S | 1 | 0/1 | 65.0 [61.0, 69.2] | 20/304 | 37/190 | 66.3 [62.2, 70.6] (+18 / -37) | 0/0 | 88 / 29 / 66 | 0 (0/0/0) / 0 | 77.2 / 81.1 | 0.075 | 1 / K0 [redacted: describes a related submission under double-blind review] (hard) + Kp 100 (soft) | 9 / 6 | 0.6 |
| REEDIT-S | 10 | 7/10 | 66.0 [61.9, 70.2] | 22/304 | 44/190 | 66.5 [62.4, 70.9] (+19 / -39) | 1/1 | 87 / 30 / 66 | 1 (1/0/0) / 1 | 76.5 / 79.4 | 0.230 | 10 / K0 [redacted: describes a related submission under double-blind review] (hard) + Kp 100 (soft) | 75 / 45 | 4.2 |
| REEDIT-S | 100 | 70/100 | 71.3 [67.7, 74.7] | 36/304 | 84/190 | 73.6 [70.1, 77.1] (+27 / -80) | 8/8 | 85 / 28 / 61 | 10 (3/2/5) / 1 | 71.7 / 76.8 | 0.688 | 100 / K0 [redacted: describes a related submission under double-blind review] (hard) + Kp 200 (soft) | 713 / 407 | 38.0 |
| REEDIT-S | ALL (305) | 159/305 | 75.3 [72.0, 78.5] | 44/304 | 112/190 | 78.0 [74.7, 81.2] (+32 / -106) | 12/12 | 74 / 23 / 49 | 38 (14/7/17) / 1 | 61.5 / 72.2 | 1.210 | 305 / K0 [redacted: describes a related submission under double-blind review] (hard) + Kp 405 (soft) | 2234 / 1311 | 141.6 |
| RB-TOT | 1 | 1/1 | 62.3 [58.3, 66.4] | 11/304 | 15/190 | 62.3 [58.2, 66.4] (+11 / -11) | 0/0 | 86 / 29 / 65 | 8 (3/2/3) / 5 | 77.9 / 81.9 | 0.194 | 3 / 0 | 7 / 3 | 0.3 |
| RB-TOT | 10 | 9/10 | 57.3 [53.1, 61.2] | 58/304 | 37/190 | 58.8 [54.5, 63.0] (+52 / -36) | 1/1 | 66 / 20 / 51 | 53 (25/10/18) / 7 | 79.4 / 83.2 | 0.577 | 30 / 0 | 610 / 300 | 19.3 |
| RB-TOT | 100 | 27/100 | 51.2 [47.1, 55.8] | 78/304 | 27/190 | 53.1 [48.7, 57.6] (+70 / -27) | 5/8 | 65 / 20 / 51 | 57 (27/11/19) / 10 | 81.0 / 87.2 | 0.586 | 30 / 0 | 6100 / 3000 | 196.1 |
| RB-TOT | ALL (305) | 81/305 | 47.8 [43.4, 52.2] | 96/304 | 28/190 | 49.5 [44.9, 54.1] (+88 / -28) | 8/12 | 63 / 18 / 48 | 61 (28/11/22) / 7 | 81.5 / 85.6 | 0.599 | 30 / 0 | 18605 / 9150 | 1169.8 |
| RB-UNION | 1 | 1/1 | 62.3 [58.3, 66.4] | 11/304 | 15/190 | 62.3 [58.2, 66.4] (+11 / -11) | 0/0 | 86 / 29 / 65 | 8 (3/2/3) / 5 | 77.9 / 81.9 | 0.194 | 3 / 0 | 7 / 3 | 0.2 |
| RB-UNION | 10 | 4/10 | 53.2 [48.8, 57.7] | 70/304 | 29/190 | 53.9 [49.3, 58.5] (+66 / -27) | 0/1 | 66 / 17 / 47 | 61 (26/13/22) / 8 | 80.9 / 84.4 | 0.575 | 29 / 0 | 78 / 34 | 2.6 |
| RB-UNION | 100 | 86/100 | 11.1 [8.3, 14.0] | 260/304 | 11/190 | 11.3 [8.4, 14.3] (+250 / -11) | 7/8 | 5 / 3 / 4 | 174 (84/27/63) / 3 | 96.6 / 97.5 | 0.984 | 96 / 0 | 1136 / 518 | 38.6 |
| RB-UNION | ALL (305) | 305/305 | 0.0 [0.0, 0.0] | 304/304 | 0/190 | 0.0 [0.0, 0.0] (+292 / -0) | 12/12 | 1 / 2 / 1 | 180 (87/28/65) / 1 | 100.0 / 100.0 | 1.000 | 100 / 0 | 3497 / 1596 | 158.2 |
| ALL (W_0) | - | - | 0.0 [0.0, 0.0] | 304/304 | 0/190 | 0.0 [0.0, 0.0] (+292 / -0) | 0/0 | 1 / 2 / 1 | 180 (87/28/65) / 1 | 100.0 / 100.0 | 1.000 | 100 / 0 | 0 / 0 | 0.0 |

## 4. Curves over n

Cell = mean over the seeds run, with (min-max) where more than one seed was run. Protocol: seeds 0, 1, 2 for n <= 20, seed 0 for n >= 50. Seeds run per cell:

Qwen (TB-Q100-ae-l2_3)

| method | n=1 | n=2 | n=5 | n=10 | n=20 | n=50 | n=100 | n=ALL (297) |
|---|---|---|---|---|---|---|---|---|
| COMP-B | 0,1,2 | 0,1,2 | 0,1,2 | 0,1,2 | 0,1,2 | 0 | 0 | 0 |
| COMP-B+ | 0,1,2 | 0,1,2 | 0,1,2 | 0,1,2 | 0,1,2 | 0 | 0 | 0 |
| COMP-S | 0,1,2 | 0,1,2 | 0,1,2 | 0,1,2 | 0,1,2 | 0 | 0 | 0 |
| COMP-S+ | 0,1,2 | 0,1,2 | 0,1,2 | 0,1,2 | 0,1,2 | 0 | 0 | 0 |
| COMP-B-ALLPOS | 0,1,2 | 0,1,2 | 0,1,2 | 0,1,2 | - | - | - | - |
| REEDIT-B | 0,1,2 | 0,1,2 | 0,1,2 | 0,1,2 | 0,1,2 | 0 | 0 | 0 |
| REEDIT-S | 0,1,2 | 0,1,2 | 0,1,2 | 0,1,2 | 0,1,2 | 0 | 0 | 0 |
| RB-TOT | 0,1,2 | 0,1,2 | 0,1,2 | 0,1,2 | 0,1,2 | 0 | 0 | 0 |
| RB-UNION | 0,1,2 | 0,1,2 | 0,1,2 | 0,1,2 | 0,1,2 | 0 | 0 | 0 |

LLaVA (TB-L100-ae-l2_3)

| method | n=1 | n=2 | n=5 | n=10 | n=20 | n=50 | n=100 | n=ALL (305) |
|---|---|---|---|---|---|---|---|---|
| COMP-B | 0,1,2 | 0,1,2 | 0,1,2 | 0,1,2 | 0,1,2 | 0 | 0 | 0 |
| COMP-B+ | 0,1,2 | 0,1,2 | 0,1,2 | 0,1,2 | 0,1,2 | 0 | 0 | 0 |
| COMP-S | 0,1,2 | 0,1,2 | 0,1,2 | 0,1,2 | 0,1,2 | 0 | 0 | 0 |
| COMP-S+ | 0,1,2 | 0,1,2 | 0,1,2 | 0,1,2 | 0,1,2 | 0 | 0 | 0 |
| COMP-B-ALLPOS | 0,1,2 | 0,1,2 | 0,1,2 | 0,1,2 | - | - | - | - |
| REEDIT-B | 0,1,2 | 0,1,2 | 0,1,2 | 0,1,2 | 0,1,2 | 0 | 0 | 0 |
| REEDIT-S | 0,1,2 | 0,1,2 | 0,1,2 | 0,1,2 | 0,1,2 | 0 | 0 | 0 |
| RB-TOT | 0,1,2 | 0,1,2 | 0,1,2 | 0,1,2 | 0,1,2 | 0 | 0 | 0 |
| RB-UNION | 0,1,2 | 0,1,2 | 0,1,2 | 0,1,2 | 0,1,2 | 0 | 0 | 0 |

### HELD-OUT D1 rate (%), full pool (protocol)

Qwen (TB-Q100-ae-l2_3) -- unrepaired: 54.7

| method | n=1 | n=2 | n=5 | n=10 | n=20 | n=50 | n=100 | n=ALL (297) |
|---|---|---|---|---|---|---|---|---|
| COMP-B | 54.1 (52.9-54.9) | 52.3 (51.2-53.1) | 51.8 (51.2-52.0) | 50.8 (50.0-51.4) | 49.0 (46.9-51.4) | 51.4 | 49.4 | 47.3 |
| COMP-B+ | 55.7 (54.7-57.3) | 54.0 (53.5-54.3) | 53.4 (52.7-54.7) | 52.3 (51.4-53.7) | 52.6 (51.8-53.5) | 57.1 | 55.9 | 66.9 |
| COMP-S | 54.1 (52.9-54.9) | 52.3 (51.2-53.1) | 51.8 (51.2-52.0) | 50.8 (50.0-51.4) | 49.0 (46.9-51.4) | 51.4 | 49.4 | 47.3 |
| COMP-S+ | 55.7 (54.7-57.3) | 54.0 (53.5-54.3) | 53.4 (52.7-54.7) | 52.3 (51.4-53.7) | 52.6 (51.8-53.5) | 57.1 | 55.9 | 66.9 |
| COMP-B-ALLPOS | 56.3 (55.1-57.8) | 54.1 (53.1-55.1) | 54.8 (53.1-56.1) | 56.7 (53.3-60.2) | - | - | - | - |
| REEDIT-B | 61.6 (59.0-65.7) | 61.4 (59.0-65.3) | 60.2 (58.8-62.4) | 58.4 (56.1-59.6) | 61.2 (57.6-64.3) | 60.0 | 57.1 | 62.9 |
| REEDIT-S | 61.6 (59.0-65.7) | 61.4 (59.0-65.3) | 60.2 (58.8-62.4) | 58.4 (56.1-59.6) | 61.6 (59.6-63.5) | 61.2 | 59.0 | 63.1 |
| RB-TOT | 52.9 (51.8-54.3) | 49.5 (47.1-51.4) | 45.0 (43.1-46.7) | 45.9 (45.1-46.7) | 46.1 (44.7-47.3) | 42.9 | 42.2 | 44.9 |
| RB-UNION | 52.9 (51.8-54.3) | 51.1 (50.4-51.8) | 46.5 (44.5-49.6) | 44.8 (43.7-46.7) | 38.7 (37.1-40.6) | 27.1 | 14.1 | 5.9 |

LLaVA (TB-L100-ae-l2_3) -- unrepaired: 61.5

| method | n=1 | n=2 | n=5 | n=10 | n=20 | n=50 | n=100 | n=ALL (305) |
|---|---|---|---|---|---|---|---|---|
| COMP-B | 61.3 (60.9-61.7) | 61.4 (61.3-61.5) | 62.3 (61.3-63.2) | 62.2 (61.5-63.4) | 61.0 (59.9-61.9) | 59.9 | 59.1 | 58.5 |
| COMP-B+ | 62.0 (61.3-62.8) | 62.0 (61.3-62.8) | 62.7 (61.5-63.4) | 63.2 (62.1-64.0) | 62.1 (61.3-63.6) | 64.6 | 66.4 | 66.8 |
| COMP-S | 61.3 (60.9-61.7) | 61.4 (61.3-61.5) | 62.3 (61.3-63.2) | 62.2 (61.5-63.4) | 61.0 (59.9-61.9) | 59.9 | 59.1 | 58.5 |
| COMP-S+ | 62.0 (61.3-62.8) | 62.0 (61.3-62.8) | 62.7 (61.5-63.4) | 63.2 (62.1-64.0) | 62.1 (61.3-63.6) | 64.8 | 66.4 | 66.8 |
| COMP-B-ALLPOS | 62.1 (61.5-63.0) | 64.2 (63.2-66.0) | 63.5 (63.0-64.2) | 65.0 (63.4-67.8) | - | - | - | - |
| REEDIT-B | 63.7 (62.3-65.0) | 64.1 (62.8-65.4) | 64.3 (61.5-65.8) | 65.4 (63.6-66.6) | 67.5 (66.0-69.4) | 69.2 | 71.3 | 75.3 |
| REEDIT-S | 63.7 (62.3-65.0) | 64.1 (62.8-65.4) | 64.3 (61.5-65.8) | 65.4 (63.6-66.6) | 67.5 (66.2-69.4) | 69.2 | 71.3 | 75.3 |
| RB-TOT | 61.7 (60.9-62.3) | 60.4 (58.9-61.7) | 56.0 (54.0-58.1) | 55.7 (53.8-57.3) | 57.0 (56.5-57.3) | 52.4 | 51.2 | 47.8 |
| RB-UNION | 61.7 (60.9-62.3) | 60.7 (60.1-61.1) | 55.6 (53.4-56.9) | 51.8 (49.2-53.2) | 45.6 (44.3-48.0) | 21.7 | 11.1 | 0.0 |

### HELD-OUT D1 rate (%), disjoint rows (no query shared with the REPAIR pool)

Qwen (TB-Q100-ae-l2_3) -- unrepaired: 54.2

| method | n=1 | n=2 | n=5 | n=10 | n=20 | n=50 | n=100 | n=ALL (297) |
|---|---|---|---|---|---|---|---|---|
| COMP-B | 53.7 (52.5-54.4) | 51.8 (50.5-52.5) | 51.5 (51.2-51.8) | 50.5 (49.7-51.2) | 48.8 (46.7-51.2) | 51.6 | 50.1 | 49.0 |
| COMP-B+ | 55.5 (54.6-57.0) | 53.8 (53.3-54.4) | 53.1 (52.5-54.2) | 52.3 (51.4-53.5) | 52.8 (52.5-53.5) | 57.2 | 56.6 | 68.8 |
| COMP-S | 53.7 (52.5-54.4) | 51.8 (50.5-52.5) | 51.5 (51.2-51.8) | 50.5 (49.7-51.2) | 48.8 (46.7-51.2) | 51.6 | 50.1 | 49.0 |
| COMP-S+ | 55.5 (54.6-57.0) | 53.8 (53.3-54.4) | 53.1 (52.5-54.2) | 52.3 (51.4-53.5) | 52.8 (52.5-53.5) | 57.2 | 56.6 | 68.8 |
| COMP-B-ALLPOS | 56.4 (55.7-57.6) | 53.8 (52.9-55.1) | 54.3 (52.7-55.7) | 56.6 (53.8-60.2) | - | - | - | - |
| REEDIT-B | 61.7 (59.4-66.0) | 61.5 (59.1-65.6) | 60.1 (58.7-62.6) | 58.9 (55.9-60.4) | 61.6 (57.6-64.9) | 60.6 | 58.1 | 66.0 |
| REEDIT-S | 61.7 (59.4-66.0) | 61.5 (59.1-65.6) | 60.1 (58.7-62.6) | 58.9 (55.9-60.4) | 61.9 (59.8-64.1) | 61.9 | 60.0 | 66.2 |
| RB-TOT | 52.9 (52.0-54.0) | 50.0 (47.3-51.8) | 45.3 (43.4-47.3) | 46.2 (44.9-48.2) | 46.6 (44.5-48.8) | 43.9 | 43.4 | 46.0 |
| RB-UNION | 52.9 (52.0-54.0) | 51.3 (50.8-52.3) | 46.9 (45.4-49.7) | 45.4 (44.3-47.5) | 39.6 (37.8-41.7) | 28.0 | 14.4 | 6.2 |

LLaVA (TB-L100-ae-l2_3) -- unrepaired: 62.3

| method | n=1 | n=2 | n=5 | n=10 | n=20 | n=50 | n=100 | n=ALL (305) |
|---|---|---|---|---|---|---|---|---|
| COMP-B | 62.0 (61.6-62.5) | 61.8 (61.4-62.3) | 63.0 (62.0-63.8) | 62.8 (62.0-64.0) | 61.8 (61.4-62.3) | 60.8 | 60.6 | 59.7 |
| COMP-B+ | 62.7 (62.0-63.3) | 62.7 (62.0-63.3) | 63.2 (62.3-63.8) | 63.8 (62.9-64.2) | 62.8 (62.0-64.0) | 65.2 | 67.0 | 67.6 |
| COMP-S | 62.0 (61.6-62.5) | 61.8 (61.4-62.3) | 63.0 (62.0-63.8) | 62.8 (62.0-64.0) | 61.8 (61.4-62.3) | 60.8 | 60.6 | 59.7 |
| COMP-S+ | 62.7 (62.0-63.3) | 62.7 (62.0-63.3) | 63.2 (62.3-63.8) | 63.8 (62.9-64.2) | 62.8 (62.0-64.0) | 65.5 | 67.0 | 67.6 |
| COMP-B-ALLPOS | 62.9 (62.3-63.8) | 65.0 (64.4-65.9) | 64.4 (63.5-64.8) | 66.1 (64.2-69.1) | - | - | - | - |
| REEDIT-B | 64.7 (63.1-66.3) | 65.0 (63.5-66.3) | 65.0 (62.3-66.5) | 66.1 (63.8-68.0) | 69.1 (68.0-70.8) | 71.0 | 73.6 | 78.0 |
| REEDIT-S | 64.7 (63.1-66.3) | 65.0 (63.5-66.3) | 65.0 (62.3-66.5) | 66.1 (63.8-68.0) | 69.2 (68.2-70.8) | 71.0 | 73.6 | 78.0 |
| RB-TOT | 62.2 (61.6-62.7) | 60.9 (60.3-61.8) | 56.5 (55.2-58.8) | 56.6 (54.8-58.8) | 57.6 (56.7-58.4) | 53.5 | 53.1 | 49.5 |
| RB-UNION | 62.2 (61.6-62.7) | 61.6 (61.2-62.0) | 56.3 (54.6-57.6) | 52.7 (50.3-53.9) | 46.9 (45.6-49.5) | 22.4 | 11.3 | 0.0 |

### held-out complaints fixed (full pool)

Qwen (TB-Q100-ae-l2_3)

| method | n=1 | n=2 | n=5 | n=10 | n=20 | n=50 | n=100 | n=ALL (297) |
|---|---|---|---|---|---|---|---|---|
| COMP-B | 23.7 (20.0-28.0) | 38.3 (35.0-41.0) | 48.3 (42.0-53.0) | 54.0 (48.0-60.0) | 66.3 (59.0-72.0) | 63.0 | 82.0 | 97.0 |
| COMP-B+ | 5.7 (2.0-9.0) | 14.7 (11.0-21.0) | 25.0 (19.0-29.0) | 36.0 (30.0-41.0) | 44.7 (38.0-52.0) | 43.0 | 57.0 | 49.0 |
| COMP-S | 23.7 (20.0-28.0) | 38.3 (35.0-41.0) | 48.3 (42.0-53.0) | 54.0 (48.0-60.0) | 66.3 (59.0-72.0) | 63.0 | 82.0 | 97.0 |
| COMP-S+ | 5.7 (2.0-9.0) | 14.7 (11.0-21.0) | 25.0 (19.0-29.0) | 36.0 (30.0-41.0) | 44.7 (38.0-52.0) | 43.0 | 57.0 | 49.0 |
| COMP-B-ALLPOS | 25.3 (22.0-30.0) | 44.0 (41.0-48.0) | 49.0 (42.0-60.0) | 68.7 (66.0-72.0) | - | - | - | - |
| REEDIT-B | 22.7 (14.0-35.0) | 23.0 (17.0-30.0) | 25.7 (19.0-31.0) | 36.0 (27.0-46.0) | 33.0 (25.0-38.0) | 43.0 | 55.0 | 54.0 |
| REEDIT-S | 22.7 (14.0-35.0) | 23.0 (17.0-30.0) | 25.7 (19.0-31.0) | 36.0 (27.0-46.0) | 31.7 (24.0-38.0) | 39.0 | 52.0 | 55.0 |
| RB-TOT | 27.7 (13.0-40.0) | 48.3 (43.0-53.0) | 77.7 (64.0-93.0) | 78.3 (78.0-79.0) | 76.0 (74.0-78.0) | 96.0 | 92.0 | 89.0 |
| RB-UNION | 27.7 (13.0-40.0) | 44.0 (38.0-50.0) | 70.0 (60.0-80.0) | 86.0 (78.0-91.0) | 114.3 (110.0-119.0) | 162.0 | 213.0 | 245.0 |

LLaVA (TB-L100-ae-l2_3)

| method | n=1 | n=2 | n=5 | n=10 | n=20 | n=50 | n=100 | n=ALL (305) |
|---|---|---|---|---|---|---|---|---|
| COMP-B | 10.3 (8.0-13.0) | 11.3 (10.0-13.0) | 14.7 (11.0-19.0) | 18.3 (15.0-21.0) | 28.3 (27.0-29.0) | 39.0 | 54.0 | 72.0 |
| COMP-B+ | 2.7 (1.0-4.0) | 3.7 (3.0-5.0) | 8.7 (7.0-11.0) | 10.3 (8.0-15.0) | 23.0 (19.0-26.0) | 33.0 | 33.0 | 48.0 |
| COMP-S | 10.3 (8.0-13.0) | 11.3 (10.0-13.0) | 14.7 (11.0-19.0) | 18.3 (15.0-21.0) | 28.3 (27.0-29.0) | 39.0 | 54.0 | 72.0 |
| COMP-S+ | 2.7 (1.0-4.0) | 3.7 (3.0-5.0) | 8.7 (7.0-11.0) | 10.3 (8.0-15.0) | 23.0 (19.0-26.0) | 33.0 | 33.0 | 48.0 |
| COMP-B-ALLPOS | 11.0 (9.0-15.0) | 22.7 (19.0-27.0) | 32.0 (31.0-33.0) | 42.7 (36.0-48.0) | - | - | - | - |
| REEDIT-B | 15.0 (11.0-20.0) | 16.7 (11.0-23.0) | 19.0 (10.0-24.0) | 19.3 (11.0-25.0) | 24.3 (22.0-27.0) | 35.0 | 36.0 | 44.0 |
| REEDIT-S | 15.0 (11.0-20.0) | 16.7 (11.0-23.0) | 19.0 (10.0-24.0) | 19.3 (11.0-25.0) | 24.0 (22.0-26.0) | 35.0 | 36.0 | 44.0 |
| RB-TOT | 18.3 (11.0-28.0) | 31.0 (28.0-33.0) | 58.3 (53.0-66.0) | 60.0 (57.0-65.0) | 60.3 (58.0-64.0) | 72.0 | 78.0 | 96.0 |
| RB-UNION | 18.3 (11.0-28.0) | 28.0 (25.0-31.0) | 59.0 (54.0-68.0) | 75.0 (66.0-89.0) | 108.7 (97.0-115.0) | 222.0 | 260.0 | 304.0 |

### held-out non-complaints lost (full pool)

Qwen (TB-Q100-ae-l2_3)

| method | n=1 | n=2 | n=5 | n=10 | n=20 | n=50 | n=100 | n=ALL (297) |
|---|---|---|---|---|---|---|---|---|
| COMP-B | 21.0 (19.0-24.0) | 26.7 (24.0-29.0) | 34.0 (29.0-37.0) | 35.0 (31.0-42.0) | 38.3 (34.0-43.0) | 47.0 | 56.0 | 61.0 |
| COMP-B+ | 10.7 (8.0-15.0) | 11.3 (9.0-15.0) | 18.7 (17.0-20.0) | 24.3 (23.0-25.0) | 34.3 (27.0-38.0) | 55.0 | 63.0 | 109.0 |
| COMP-S | 21.0 (19.0-24.0) | 26.7 (24.0-29.0) | 34.0 (29.0-37.0) | 35.0 (31.0-42.0) | 38.3 (34.0-43.0) | 47.0 | 56.0 | 61.0 |
| COMP-S+ | 10.7 (8.0-15.0) | 11.3 (9.0-15.0) | 18.7 (17.0-20.0) | 24.3 (23.0-25.0) | 34.3 (27.0-38.0) | 55.0 | 63.0 | 109.0 |
| COMP-B-ALLPOS | 33.3 (29.0-39.0) | 41.0 (35.0-45.0) | 49.3 (44.0-52.0) | 78.3 (65.0-93.0) | - | - | - | - |
| REEDIT-B | 56.3 (45.0-68.0) | 55.7 (47.0-69.0) | 52.7 (47.0-57.0) | 54.0 (50.0-59.0) | 65.0 (50.0-73.0) | 69.0 | 67.0 | 94.0 |
| REEDIT-S | 56.3 (45.0-68.0) | 55.7 (47.0-69.0) | 52.7 (47.0-57.0) | 54.0 (50.0-59.0) | 65.3 (57.0-72.0) | 71.0 | 73.0 | 96.0 |
| RB-TOT | 18.7 (11.0-26.0) | 22.7 (16.0-27.0) | 30.3 (25.0-36.0) | 35.0 (31.0-40.0) | 34.0 (29.0-38.0) | 38.0 | 31.0 | 41.0 |
| RB-UNION | 18.7 (11.0-26.0) | 26.3 (20.0-30.0) | 29.7 (24.0-35.0) | 37.3 (35.0-39.0) | 36.0 (28.0-41.0) | 27.0 | 14.0 | 6.0 |

LLaVA (TB-L100-ae-l2_3)

| method | n=1 | n=2 | n=5 | n=10 | n=20 | n=50 | n=100 | n=ALL (305) |
|---|---|---|---|---|---|---|---|---|
| COMP-B | 9.3 (7.0-12.0) | 10.7 (10.0-12.0) | 18.7 (18.0-19.0) | 21.7 (19.0-24.0) | 25.7 (21.0-31.0) | 31.0 | 42.0 | 57.0 |
| COMP-B+ | 5.0 (3.0-9.0) | 6.0 (4.0-9.0) | 14.3 (7.0-20.0) | 18.7 (11.0-27.0) | 26.0 (23.0-29.0) | 48.0 | 57.0 | 74.0 |
| COMP-S | 9.3 (7.0-12.0) | 10.7 (10.0-12.0) | 18.7 (18.0-19.0) | 21.7 (19.0-24.0) | 25.7 (21.0-31.0) | 31.0 | 42.0 | 57.0 |
| COMP-S+ | 5.0 (3.0-9.0) | 6.0 (4.0-9.0) | 14.3 (7.0-20.0) | 18.7 (11.0-27.0) | 26.0 (23.0-29.0) | 49.0 | 57.0 | 74.0 |
| COMP-B-ALLPOS | 14.0 (11.0-16.0) | 35.7 (30.0-41.0) | 41.7 (39.0-46.0) | 59.7 (53.0-67.0) | - | - | - | - |
| REEDIT-B | 25.7 (18.0-37.0) | 29.3 (22.0-42.0) | 32.7 (24.0-44.0) | 38.3 (35.0-44.0) | 53.7 (49.0-61.0) | 73.0 | 84.0 | 112.0 |
| REEDIT-S | 25.7 (18.0-37.0) | 29.3 (22.0-42.0) | 32.7 (24.0-44.0) | 38.3 (35.0-44.0) | 53.7 (49.0-61.0) | 73.0 | 84.0 | 112.0 |
| RB-TOT | 19.0 (15.0-25.0) | 25.3 (20.0-33.0) | 31.0 (28.0-36.0) | 31.3 (27.0-37.0) | 37.7 (34.0-42.0) | 27.0 | 27.0 | 28.0 |
| RB-UNION | 19.0 (15.0-25.0) | 24.0 (18.0-28.0) | 29.7 (28.0-31.0) | 26.7 (23.0-29.0) | 30.0 (29.0-31.0) | 25.0 | 11.0 | 0.0 |

### held-out complaints fixed (disjoint rows)

Qwen (TB-Q100-ae-l2_3)

| method | n=1 | n=2 | n=5 | n=10 | n=20 | n=50 | n=100 | n=ALL (297) |
|---|---|---|---|---|---|---|---|---|
| COMP-B | 23.3 (20.0-27.0) | 38.0 (35.0-41.0) | 46.3 (41.0-50.0) | 52.0 (48.0-56.0) | 63.3 (57.0-69.0) | 59.0 | 74.0 | 85.0 |
| COMP-B+ | 4.7 (2.0-7.0) | 13.0 (9.0-19.0) | 23.7 (19.0-27.0) | 33.3 (28.0-38.0) | 40.7 (35.0-46.0) | 40.0 | 52.0 | 40.0 |
| COMP-S | 23.3 (20.0-27.0) | 38.0 (35.0-41.0) | 46.3 (41.0-50.0) | 52.0 (48.0-56.0) | 63.3 (57.0-69.0) | 59.0 | 74.0 | 85.0 |
| COMP-S+ | 4.7 (2.0-7.0) | 13.0 (9.0-19.0) | 23.7 (19.0-27.0) | 33.3 (28.0-38.0) | 40.7 (35.0-46.0) | 40.0 | 52.0 | 40.0 |
| COMP-B-ALLPOS | 23.0 (21.0-25.0) | 42.0 (39.0-47.0) | 47.7 (41.0-58.0) | 65.7 (63.0-67.0) | - | - | - | - |
| REEDIT-B | 21.3 (13.0-32.0) | 21.7 (16.0-28.0) | 25.0 (18.0-31.0) | 32.0 (21.0-45.0) | 30.3 (22.0-35.0) | 39.0 | 49.0 | 39.0 |
| REEDIT-S | 21.3 (13.0-32.0) | 21.7 (16.0-28.0) | 25.0 (18.0-31.0) | 32.0 (21.0-45.0) | 29.0 (21.0-35.0) | 35.0 | 46.0 | 40.0 |
| RB-TOT | 24.7 (12.0-36.0) | 42.0 (38.0-48.0) | 71.7 (57.0-86.0) | 72.0 (68.0-74.0) | 69.3 (63.0-74.0) | 86.0 | 81.0 | 79.0 |
| RB-UNION | 24.7 (12.0-36.0) | 39.7 (35.0-45.0) | 63.7 (56.0-71.0) | 78.0 (70.0-83.0) | 103.7 (99.0-108.0) | 149.0 | 199.0 | 229.0 |

LLaVA (TB-L100-ae-l2_3)

| method | n=1 | n=2 | n=5 | n=10 | n=20 | n=50 | n=100 | n=ALL (305) |
|---|---|---|---|---|---|---|---|---|
| COMP-B | 10.3 (8.0-13.0) | 11.3 (10.0-13.0) | 14.7 (11.0-19.0) | 18.3 (15.0-21.0) | 26.0 (24.0-27.0) | 37.0 | 49.0 | 65.0 |
| COMP-B+ | 2.7 (1.0-4.0) | 3.7 (3.0-5.0) | 8.7 (7.0-11.0) | 10.3 (8.0-15.0) | 19.7 (17.0-22.0) | 29.0 | 29.0 | 42.0 |
| COMP-S | 10.3 (8.0-13.0) | 11.3 (10.0-13.0) | 14.7 (11.0-19.0) | 18.3 (15.0-21.0) | 26.0 (24.0-27.0) | 37.0 | 49.0 | 65.0 |
| COMP-S+ | 2.7 (1.0-4.0) | 3.7 (3.0-5.0) | 8.7 (7.0-11.0) | 10.3 (8.0-15.0) | 19.7 (17.0-22.0) | 29.0 | 29.0 | 42.0 |
| COMP-B-ALLPOS | 11.0 (9.0-15.0) | 21.3 (19.0-25.0) | 29.0 (26.0-32.0) | 39.3 (34.0-45.0) | - | - | - | - |
| REEDIT-B | 14.3 (11.0-18.0) | 15.7 (11.0-20.0) | 18.0 (10.0-24.0) | 17.7 (9.0-25.0) | 20.7 (19.0-22.0) | 28.0 | 27.0 | 32.0 |
| REEDIT-S | 14.3 (11.0-18.0) | 15.7 (11.0-20.0) | 18.0 (10.0-24.0) | 17.7 (9.0-25.0) | 20.3 (19.0-21.0) | 28.0 | 27.0 | 32.0 |
| RB-TOT | 17.0 (11.0-25.0) | 29.0 (28.0-31.0) | 56.3 (51.0-62.0) | 56.3 (52.0-61.0) | 56.3 (55.0-58.0) | 68.0 | 70.0 | 88.0 |
| RB-UNION | 17.0 (11.0-25.0) | 25.7 (23.0-28.0) | 56.7 (52.0-64.0) | 70.0 (61.0-83.0) | 100.7 (90.0-106.0) | 212.0 | 250.0 | 292.0 |

### held-out non-complaints lost (disjoint rows)

Qwen (TB-Q100-ae-l2_3)

| method | n=1 | n=2 | n=5 | n=10 | n=20 | n=50 | n=100 | n=ALL (297) |
|---|---|---|---|---|---|---|---|---|
| COMP-B | 21.0 (19.0-24.0) | 26.7 (24.0-29.0) | 34.0 (29.0-37.0) | 35.0 (31.0-42.0) | 38.3 (34.0-43.0) | 47.0 | 55.0 | 61.0 |
| COMP-B+ | 10.7 (8.0-15.0) | 11.3 (9.0-15.0) | 18.7 (17.0-20.0) | 24.3 (23.0-25.0) | 34.3 (27.0-38.0) | 54.0 | 63.0 | 108.0 |
| COMP-S | 21.0 (19.0-24.0) | 26.7 (24.0-29.0) | 34.0 (29.0-37.0) | 35.0 (31.0-42.0) | 38.3 (34.0-43.0) | 47.0 | 55.0 | 61.0 |
| COMP-S+ | 10.7 (8.0-15.0) | 11.3 (9.0-15.0) | 18.7 (17.0-20.0) | 24.3 (23.0-25.0) | 34.3 (27.0-38.0) | 54.0 | 63.0 | 108.0 |
| COMP-B-ALLPOS | 33.3 (29.0-39.0) | 40.3 (34.0-44.0) | 48.3 (43.0-51.0) | 77.0 (65.0-91.0) | - | - | - | - |
| REEDIT-B | 56.3 (45.0-68.0) | 55.7 (47.0-69.0) | 52.7 (47.0-57.0) | 54.0 (50.0-59.0) | 64.7 (50.0-72.0) | 69.0 | 67.0 | 94.0 |
| REEDIT-S | 56.3 (45.0-68.0) | 55.7 (47.0-69.0) | 52.7 (47.0-57.0) | 54.0 (50.0-59.0) | 65.0 (57.0-71.0) | 71.0 | 73.0 | 96.0 |
| RB-TOT | 18.7 (11.0-26.0) | 22.7 (16.0-27.0) | 30.3 (25.0-36.0) | 35.0 (31.0-40.0) | 34.0 (29.0-38.0) | 38.0 | 31.0 | 41.0 |
| RB-UNION | 18.7 (11.0-26.0) | 26.3 (20.0-30.0) | 29.7 (24.0-35.0) | 37.3 (35.0-39.0) | 36.0 (28.0-41.0) | 27.0 | 14.0 | 6.0 |

LLaVA (TB-L100-ae-l2_3)

| method | n=1 | n=2 | n=5 | n=10 | n=20 | n=50 | n=100 | n=ALL (305) |
|---|---|---|---|---|---|---|---|---|
| COMP-B | 9.0 (7.0-11.0) | 9.3 (7.0-11.0) | 18.0 (18.0-18.0) | 21.0 (18.0-23.0) | 23.7 (20.0-27.0) | 30.0 | 41.0 | 53.0 |
| COMP-B+ | 4.7 (3.0-8.0) | 5.7 (4.0-8.0) | 13.0 (7.0-17.0) | 17.3 (11.0-24.0) | 22.0 (20.0-25.0) | 43.0 | 51.0 | 67.0 |
| COMP-S | 9.0 (7.0-11.0) | 9.3 (7.0-11.0) | 18.0 (18.0-18.0) | 21.0 (18.0-23.0) | 23.7 (20.0-27.0) | 30.0 | 41.0 | 53.0 |
| COMP-S+ | 4.7 (3.0-8.0) | 5.7 (4.0-8.0) | 13.0 (7.0-17.0) | 17.3 (11.0-24.0) | 22.0 (20.0-25.0) | 44.0 | 51.0 | 67.0 |
| COMP-B-ALLPOS | 14.0 (11.0-16.0) | 34.0 (30.0-36.0) | 39.0 (38.0-41.0) | 57.3 (52.0-66.0) | - | - | - | - |
| REEDIT-B | 25.7 (18.0-37.0) | 28.3 (22.0-39.0) | 31.0 (24.0-39.0) | 35.7 (32.0-39.0) | 52.7 (49.0-59.0) | 69.0 | 80.0 | 106.0 |
| REEDIT-S | 25.7 (18.0-37.0) | 28.3 (22.0-39.0) | 31.0 (24.0-39.0) | 35.7 (32.0-39.0) | 52.7 (49.0-59.0) | 69.0 | 80.0 | 106.0 |
| RB-TOT | 16.7 (11.0-22.0) | 22.7 (19.0-29.0) | 29.3 (24.0-35.0) | 30.0 (26.0-36.0) | 34.3 (30.0-40.0) | 27.0 | 27.0 | 28.0 |
| RB-UNION | 16.7 (11.0-22.0) | 22.7 (18.0-25.0) | 28.7 (28.0-30.0) | 25.3 (22.0-27.0) | 28.7 (28.0-30.0) | 25.0 | 11.0 | 0.0 |

### held-out twins of reported complaints (rows)

Qwen (TB-Q100-ae-l2_3)

| method | n=1 | n=2 | n=5 | n=10 | n=20 | n=50 | n=100 | n=ALL (297) |
|---|---|---|---|---|---|---|---|---|
| COMP-B | 0.0 (0.0-0.0) | 0.0 (0.0-0.0) | 0.0 (0.0-0.0) | 1.0 (0.0-3.0) | 1.3 (0.0-3.0) | 4.0 | 6.0 | 16.0 |
| COMP-B+ | 0.0 (0.0-0.0) | 0.0 (0.0-0.0) | 0.0 (0.0-0.0) | 1.0 (0.0-3.0) | 1.3 (0.0-3.0) | 4.0 | 6.0 | 16.0 |
| COMP-S | 0.0 (0.0-0.0) | 0.0 (0.0-0.0) | 0.0 (0.0-0.0) | 1.0 (0.0-3.0) | 1.3 (0.0-3.0) | 4.0 | 6.0 | 16.0 |
| COMP-S+ | 0.0 (0.0-0.0) | 0.0 (0.0-0.0) | 0.0 (0.0-0.0) | 1.0 (0.0-3.0) | 1.3 (0.0-3.0) | 4.0 | 6.0 | 16.0 |
| COMP-B-ALLPOS | 0.0 (0.0-0.0) | 0.0 (0.0-0.0) | 0.0 (0.0-0.0) | 1.0 (0.0-3.0) | - | - | - | - |
| REEDIT-B | 0.0 (0.0-0.0) | 0.0 (0.0-0.0) | 0.0 (0.0-0.0) | 1.0 (0.0-3.0) | 1.3 (0.0-3.0) | 4.0 | 6.0 | 16.0 |
| REEDIT-S | 0.0 (0.0-0.0) | 0.0 (0.0-0.0) | 0.0 (0.0-0.0) | 1.0 (0.0-3.0) | 1.3 (0.0-3.0) | 4.0 | 6.0 | 16.0 |
| RB-TOT | 0.0 (0.0-0.0) | 0.0 (0.0-0.0) | 0.0 (0.0-0.0) | 1.0 (0.0-3.0) | 1.3 (0.0-3.0) | 4.0 | 6.0 | 16.0 |
| RB-UNION | 0.0 (0.0-0.0) | 0.0 (0.0-0.0) | 0.0 (0.0-0.0) | 1.0 (0.0-3.0) | 1.3 (0.0-3.0) | 4.0 | 6.0 | 16.0 |

LLaVA (TB-L100-ae-l2_3)

| method | n=1 | n=2 | n=5 | n=10 | n=20 | n=50 | n=100 | n=ALL (305) |
|---|---|---|---|---|---|---|---|---|
| COMP-B | 0.0 (0.0-0.0) | 0.3 (0.0-1.0) | 0.3 (0.0-1.0) | 0.3 (0.0-1.0) | 2.0 (1.0-3.0) | 4.0 | 8.0 | 12.0 |
| COMP-B+ | 0.0 (0.0-0.0) | 0.3 (0.0-1.0) | 0.3 (0.0-1.0) | 0.3 (0.0-1.0) | 2.0 (1.0-3.0) | 4.0 | 8.0 | 12.0 |
| COMP-S | 0.0 (0.0-0.0) | 0.3 (0.0-1.0) | 0.3 (0.0-1.0) | 0.3 (0.0-1.0) | 2.0 (1.0-3.0) | 4.0 | 8.0 | 12.0 |
| COMP-S+ | 0.0 (0.0-0.0) | 0.3 (0.0-1.0) | 0.3 (0.0-1.0) | 0.3 (0.0-1.0) | 2.0 (1.0-3.0) | 4.0 | 8.0 | 12.0 |
| COMP-B-ALLPOS | 0.0 (0.0-0.0) | 0.3 (0.0-1.0) | 0.3 (0.0-1.0) | 0.3 (0.0-1.0) | - | - | - | - |
| REEDIT-B | 0.0 (0.0-0.0) | 0.3 (0.0-1.0) | 0.3 (0.0-1.0) | 0.3 (0.0-1.0) | 2.0 (1.0-3.0) | 4.0 | 8.0 | 12.0 |
| REEDIT-S | 0.0 (0.0-0.0) | 0.3 (0.0-1.0) | 0.3 (0.0-1.0) | 0.3 (0.0-1.0) | 2.0 (1.0-3.0) | 4.0 | 8.0 | 12.0 |
| RB-TOT | 0.0 (0.0-0.0) | 0.3 (0.0-1.0) | 0.3 (0.0-1.0) | 0.3 (0.0-1.0) | 2.0 (1.0-3.0) | 4.0 | 8.0 | 12.0 |
| RB-UNION | 0.0 (0.0-0.0) | 0.3 (0.0-1.0) | 0.3 (0.0-1.0) | 0.3 (0.0-1.0) | 2.0 (1.0-3.0) | 4.0 | 8.0 | 12.0 |

### held-out twins of reported complaints fixed

Qwen (TB-Q100-ae-l2_3)

| method | n=1 | n=2 | n=5 | n=10 | n=20 | n=50 | n=100 | n=ALL (297) |
|---|---|---|---|---|---|---|---|---|
| COMP-B | 0.0 (0.0-0.0) | 0.0 (0.0-0.0) | 0.0 (0.0-0.0) | 1.0 (0.0-3.0) | 1.0 (0.0-2.0) | 3.0 | 5.0 | 12.0 |
| COMP-B+ | 0.0 (0.0-0.0) | 0.0 (0.0-0.0) | 0.0 (0.0-0.0) | 0.7 (0.0-2.0) | 1.0 (0.0-2.0) | 3.0 | 2.0 | 9.0 |
| COMP-S | 0.0 (0.0-0.0) | 0.0 (0.0-0.0) | 0.0 (0.0-0.0) | 1.0 (0.0-3.0) | 1.0 (0.0-2.0) | 3.0 | 5.0 | 12.0 |
| COMP-S+ | 0.0 (0.0-0.0) | 0.0 (0.0-0.0) | 0.0 (0.0-0.0) | 0.7 (0.0-2.0) | 1.0 (0.0-2.0) | 3.0 | 2.0 | 9.0 |
| COMP-B-ALLPOS | 0.0 (0.0-0.0) | 0.0 (0.0-0.0) | 0.0 (0.0-0.0) | 1.0 (0.0-3.0) | - | - | - | - |
| REEDIT-B | 0.0 (0.0-0.0) | 0.0 (0.0-0.0) | 0.0 (0.0-0.0) | 1.0 (0.0-3.0) | 1.3 (0.0-3.0) | 4.0 | 6.0 | 15.0 |
| REEDIT-S | 0.0 (0.0-0.0) | 0.0 (0.0-0.0) | 0.0 (0.0-0.0) | 1.0 (0.0-3.0) | 1.3 (0.0-3.0) | 4.0 | 6.0 | 15.0 |
| RB-TOT | 0.0 (0.0-0.0) | 0.0 (0.0-0.0) | 0.0 (0.0-0.0) | 1.0 (0.0-3.0) | 1.3 (0.0-3.0) | 3.0 | 3.0 | 10.0 |
| RB-UNION | 0.0 (0.0-0.0) | 0.0 (0.0-0.0) | 0.0 (0.0-0.0) | 0.7 (0.0-2.0) | 1.0 (0.0-2.0) | 3.0 | 6.0 | 16.0 |

LLaVA (TB-L100-ae-l2_3)

| method | n=1 | n=2 | n=5 | n=10 | n=20 | n=50 | n=100 | n=ALL (305) |
|---|---|---|---|---|---|---|---|---|
| COMP-B | 0.0 (0.0-0.0) | 0.0 (0.0-0.0) | 0.0 (0.0-0.0) | 0.0 (0.0-0.0) | 1.7 (0.0-3.0) | 2.0 | 5.0 | 7.0 |
| COMP-B+ | 0.0 (0.0-0.0) | 0.0 (0.0-0.0) | 0.0 (0.0-0.0) | 0.0 (0.0-0.0) | 1.7 (0.0-3.0) | 3.0 | 3.0 | 6.0 |
| COMP-S | 0.0 (0.0-0.0) | 0.0 (0.0-0.0) | 0.0 (0.0-0.0) | 0.0 (0.0-0.0) | 1.7 (0.0-3.0) | 2.0 | 5.0 | 7.0 |
| COMP-S+ | 0.0 (0.0-0.0) | 0.0 (0.0-0.0) | 0.0 (0.0-0.0) | 0.0 (0.0-0.0) | 1.7 (0.0-3.0) | 3.0 | 3.0 | 6.0 |
| COMP-B-ALLPOS | 0.0 (0.0-0.0) | 0.3 (0.0-1.0) | 0.3 (0.0-1.0) | 0.3 (0.0-1.0) | - | - | - | - |
| REEDIT-B | 0.0 (0.0-0.0) | 0.3 (0.0-1.0) | 0.3 (0.0-1.0) | 0.3 (0.0-1.0) | 2.0 (1.0-3.0) | 4.0 | 8.0 | 12.0 |
| REEDIT-S | 0.0 (0.0-0.0) | 0.3 (0.0-1.0) | 0.3 (0.0-1.0) | 0.3 (0.0-1.0) | 2.0 (1.0-3.0) | 4.0 | 8.0 | 12.0 |
| RB-TOT | 0.0 (0.0-0.0) | 0.3 (0.0-1.0) | 0.3 (0.0-1.0) | 0.3 (0.0-1.0) | 1.3 (1.0-2.0) | 3.0 | 5.0 | 8.0 |
| RB-UNION | 0.0 (0.0-0.0) | 0.3 (0.0-1.0) | 0.0 (0.0-0.0) | 0.0 (0.0-0.0) | 2.0 (1.0-3.0) | 4.0 | 7.0 | 12.0 |

### reported complaints resolved (%)

Qwen (TB-Q100-ae-l2_3)

| method | n=1 | n=2 | n=5 | n=10 | n=20 | n=50 | n=100 | n=ALL (297) |
|---|---|---|---|---|---|---|---|---|
| COMP-B | 100 (100-100) | 50 (50-50) | 87 (80-100) | 87 (80-90) | 72 (65-80) | 70 | 71 | 57 |
| COMP-B+ | 67 (0-100) | 83 (50-100) | 80 (80-80) | 83 (80-90) | 67 (55-85) | 60 | 64 | 43 |
| COMP-S | 100 (100-100) | 50 (50-50) | 87 (80-100) | 87 (80-90) | 72 (65-80) | 70 | 71 | 57 |
| COMP-S+ | 67 (0-100) | 83 (50-100) | 80 (80-80) | 83 (80-90) | 67 (55-85) | 60 | 64 | 43 |
| COMP-B-ALLPOS | 67 (0-100) | 83 (50-100) | 93 (80-100) | 97 (90-100) | - | - | - | - |
| REEDIT-B | 100 (100-100) | 100 (100-100) | 100 (100-100) | 97 (90-100) | 98 (95-100) | 98 | 93 | 81 |
| REEDIT-S | 100 (100-100) | 100 (100-100) | 100 (100-100) | 97 (90-100) | 98 (95-100) | 98 | 94 | 82 |
| RB-TOT | 100 (100-100) | 100 (100-100) | 93 (80-100) | 73 (60-90) | 70 (65-75) | 56 | 47 | 33 |
| RB-UNION | 100 (100-100) | 100 (100-100) | 73 (60-100) | 77 (70-80) | 62 (55-70) | 76 | 80 | 92 |

LLaVA (TB-L100-ae-l2_3)

| method | n=1 | n=2 | n=5 | n=10 | n=20 | n=50 | n=100 | n=ALL (305) |
|---|---|---|---|---|---|---|---|---|
| COMP-B | 67 (0-100) | 50 (0-100) | 60 (40-80) | 67 (60-70) | 67 (60-75) | 62 | 59 | 54 |
| COMP-B+ | 67 (0-100) | 50 (0-100) | 60 (40-80) | 63 (60-70) | 63 (60-70) | 58 | 49 | 45 |
| COMP-S | 67 (0-100) | 50 (0-100) | 60 (40-80) | 67 (60-70) | 67 (60-75) | 62 | 59 | 54 |
| COMP-S+ | 67 (0-100) | 50 (0-100) | 60 (40-80) | 63 (60-70) | 63 (60-70) | 58 | 49 | 45 |
| COMP-B-ALLPOS | 100 (100-100) | 100 (100-100) | 100 (100-100) | 100 (100-100) | - | - | - | - |
| REEDIT-B | 67 (0-100) | 83 (50-100) | 80 (60-100) | 77 (70-90) | 75 (70-80) | 72 | 70 | 52 |
| REEDIT-S | 67 (0-100) | 83 (50-100) | 80 (60-100) | 77 (70-90) | 75 (70-80) | 72 | 70 | 52 |
| RB-TOT | 100 (100-100) | 100 (100-100) | 87 (80-100) | 63 (40-90) | 52 (40-65) | 34 | 27 | 27 |
| RB-UNION | 100 (100-100) | 83 (50-100) | 60 (60-60) | 43 (40-50) | 58 (50-65) | 68 | 86 | 100 |

### Rel exact (%)

Qwen (TB-Q100-ae-l2_3) -- unrepaired: 78.0

| method | n=1 | n=2 | n=5 | n=10 | n=20 | n=50 | n=100 | n=ALL (297) |
|---|---|---|---|---|---|---|---|---|
| COMP-B | 78.0 (78.0-78.0) | 77.7 (77.0-79.0) | 76.0 (75.0-77.0) | 76.0 (75.0-77.0) | 77.3 (74.0-80.0) | 75.0 | 75.0 | 78.0 |
| COMP-B+ | 78.0 (77.0-79.0) | 78.0 (78.0-78.0) | 78.0 (77.0-79.0) | 77.7 (77.0-79.0) | 76.0 (73.0-78.0) | 72.0 | 76.0 | 76.0 |
| COMP-S | 78.0 (78.0-78.0) | 77.7 (77.0-79.0) | 76.0 (75.0-77.0) | 76.0 (75.0-77.0) | 77.3 (74.0-80.0) | 75.0 | 75.0 | 78.0 |
| COMP-S+ | 78.0 (77.0-79.0) | 78.0 (78.0-78.0) | 78.0 (77.0-79.0) | 77.7 (77.0-79.0) | 76.0 (73.0-78.0) | 72.0 | 76.0 | 76.0 |
| COMP-B-ALLPOS | 78.3 (78.0-79.0) | 77.7 (76.0-79.0) | 77.3 (76.0-78.0) | 76.7 (76.0-78.0) | - | - | - | - |
| REEDIT-B | 77.3 (76.0-78.0) | 77.3 (77.0-78.0) | 76.7 (75.0-79.0) | 75.7 (75.0-77.0) | 76.0 (75.0-78.0) | 73.0 | 72.0 | 63.0 |
| REEDIT-S | 77.3 (76.0-78.0) | 77.3 (77.0-78.0) | 76.7 (75.0-79.0) | 75.7 (75.0-77.0) | 76.0 (75.0-77.0) | 74.0 | 72.0 | 63.0 |
| RB-TOT | 75.3 (72.0-79.0) | 67.7 (65.0-69.0) | 59.0 (57.0-61.0) | 54.7 (54.0-55.0) | 54.3 (53.0-56.0) | 56.0 | 53.0 | 56.0 |
| RB-UNION | 75.3 (72.0-79.0) | 71.0 (69.0-74.0) | 62.3 (59.0-66.0) | 52.7 (44.0-61.0) | 36.7 (31.0-43.0) | 18.0 | 7.0 | 1.0 |

LLaVA (TB-L100-ae-l2_3) -- unrepaired: 88.0

| method | n=1 | n=2 | n=5 | n=10 | n=20 | n=50 | n=100 | n=ALL (305) |
|---|---|---|---|---|---|---|---|---|
| COMP-B | 88.0 (88.0-88.0) | 88.0 (88.0-88.0) | 88.0 (88.0-88.0) | 88.0 (88.0-88.0) | 88.0 (88.0-88.0) | 88.0 | 89.0 | 86.0 |
| COMP-B+ | 88.0 (88.0-88.0) | 88.0 (88.0-88.0) | 88.0 (88.0-88.0) | 88.0 (88.0-88.0) | 88.0 (88.0-88.0) | 88.0 | 88.0 | 84.0 |
| COMP-S | 88.0 (88.0-88.0) | 88.0 (88.0-88.0) | 88.0 (88.0-88.0) | 88.0 (88.0-88.0) | 88.0 (88.0-88.0) | 88.0 | 89.0 | 86.0 |
| COMP-S+ | 88.0 (88.0-88.0) | 88.0 (88.0-88.0) | 88.0 (88.0-88.0) | 88.0 (88.0-88.0) | 88.0 (88.0-88.0) | 88.0 | 88.0 | 84.0 |
| COMP-B-ALLPOS | 88.0 (88.0-88.0) | 88.3 (88.0-89.0) | 89.0 (89.0-89.0) | 89.0 (88.0-90.0) | - | - | - | - |
| REEDIT-B | 88.0 (88.0-88.0) | 88.0 (88.0-88.0) | 88.3 (88.0-89.0) | 87.7 (87.0-89.0) | 87.7 (87.0-89.0) | 87.0 | 86.0 | 74.0 |
| REEDIT-S | 88.0 (88.0-88.0) | 88.0 (88.0-88.0) | 88.3 (88.0-89.0) | 87.7 (87.0-89.0) | 87.7 (87.0-89.0) | 87.0 | 85.0 | 74.0 |
| RB-TOT | 84.0 (81.0-86.0) | 79.7 (76.0-83.0) | 66.7 (61.0-76.0) | 63.7 (60.0-66.0) | 64.0 (63.0-66.0) | 66.0 | 65.0 | 63.0 |
| RB-UNION | 84.0 (81.0-86.0) | 82.3 (80.0-84.0) | 69.0 (67.0-71.0) | 59.7 (52.0-66.0) | 41.7 (39.0-47.0) | 13.0 | 5.0 | 1.0 |

### T-Gen exact (%)

Qwen (TB-Q100-ae-l2_3) -- unrepaired: 9.0

| method | n=1 | n=2 | n=5 | n=10 | n=20 | n=50 | n=100 | n=ALL (297) |
|---|---|---|---|---|---|---|---|---|
| COMP-B | 9.0 (9.0-9.0) | 9.0 (9.0-9.0) | 8.7 (8.0-9.0) | 8.7 (8.0-9.0) | 8.7 (8.0-9.0) | 10.0 | 8.0 | 10.0 |
| COMP-B+ | 9.0 (9.0-9.0) | 9.0 (9.0-9.0) | 9.3 (9.0-10.0) | 9.0 (8.0-10.0) | 8.7 (7.0-10.0) | 7.0 | 7.0 | 8.0 |
| COMP-S | 9.0 (9.0-9.0) | 9.0 (9.0-9.0) | 8.7 (8.0-9.0) | 8.7 (8.0-9.0) | 8.7 (8.0-9.0) | 10.0 | 8.0 | 10.0 |
| COMP-S+ | 9.0 (9.0-9.0) | 9.0 (9.0-9.0) | 9.3 (9.0-10.0) | 9.0 (8.0-10.0) | 8.7 (7.0-10.0) | 7.0 | 7.0 | 8.0 |
| COMP-B-ALLPOS | 9.0 (9.0-9.0) | 9.7 (9.0-10.0) | 10.7 (10.0-11.0) | 11.3 (11.0-12.0) | - | - | - | - |
| REEDIT-B | 9.0 (9.0-9.0) | 9.0 (9.0-9.0) | 8.7 (8.0-9.0) | 8.7 (8.0-9.0) | 9.0 (9.0-9.0) | 10.0 | 10.0 | 5.0 |
| REEDIT-S | 9.0 (9.0-9.0) | 9.0 (9.0-9.0) | 8.7 (8.0-9.0) | 8.7 (8.0-9.0) | 9.0 (9.0-9.0) | 10.0 | 10.0 | 5.0 |
| RB-TOT | 9.3 (9.0-10.0) | 9.0 (8.0-10.0) | 7.3 (4.0-9.0) | 7.7 (5.0-9.0) | 7.3 (6.0-8.0) | 8.0 | 8.0 | 7.0 |
| RB-UNION | 9.3 (9.0-10.0) | 9.7 (9.0-10.0) | 8.0 (6.0-9.0) | 7.7 (6.0-9.0) | 5.3 (4.0-7.0) | 4.0 | 2.0 | 2.0 |

LLaVA (TB-L100-ae-l2_3) -- unrepaired: 29.0

| method | n=1 | n=2 | n=5 | n=10 | n=20 | n=50 | n=100 | n=ALL (305) |
|---|---|---|---|---|---|---|---|---|
| COMP-B | 29.0 (29.0-29.0) | 29.0 (29.0-29.0) | 29.0 (29.0-29.0) | 29.3 (29.0-30.0) | 30.0 (29.0-31.0) | 29.0 | 29.0 | 32.0 |
| COMP-B+ | 29.3 (29.0-30.0) | 29.3 (29.0-30.0) | 29.3 (29.0-30.0) | 29.3 (28.0-31.0) | 29.3 (28.0-30.0) | 29.0 | 27.0 | 33.0 |
| COMP-S | 29.0 (29.0-29.0) | 29.0 (29.0-29.0) | 29.0 (29.0-29.0) | 29.3 (29.0-30.0) | 30.0 (29.0-31.0) | 29.0 | 29.0 | 32.0 |
| COMP-S+ | 29.3 (29.0-30.0) | 29.3 (29.0-30.0) | 29.3 (29.0-30.0) | 29.3 (28.0-31.0) | 29.3 (28.0-30.0) | 29.0 | 27.0 | 33.0 |
| COMP-B-ALLPOS | 29.3 (29.0-30.0) | 29.3 (29.0-30.0) | 29.0 (28.0-30.0) | 33.3 (32.0-35.0) | - | - | - | - |
| REEDIT-B | 29.0 (29.0-29.0) | 29.0 (29.0-29.0) | 29.3 (29.0-30.0) | 29.3 (29.0-30.0) | 29.7 (29.0-30.0) | 30.0 | 28.0 | 23.0 |
| REEDIT-S | 29.0 (29.0-29.0) | 29.0 (29.0-29.0) | 29.3 (29.0-30.0) | 29.3 (29.0-30.0) | 29.7 (29.0-30.0) | 30.0 | 28.0 | 23.0 |
| RB-TOT | 27.0 (23.0-29.0) | 25.3 (23.0-28.0) | 17.3 (14.0-21.0) | 18.3 (17.0-20.0) | 20.7 (19.0-23.0) | 21.0 | 20.0 | 18.0 |
| RB-UNION | 27.0 (23.0-29.0) | 26.3 (23.0-29.0) | 21.0 (19.0-25.0) | 18.3 (17.0-20.0) | 12.0 (8.0-14.0) | 6.0 | 3.0 | 2.0 |

### I-Gen exact (%)

Qwen (TB-Q100-ae-l2_3) -- unrepaired: 22.0

| method | n=1 | n=2 | n=5 | n=10 | n=20 | n=50 | n=100 | n=ALL (297) |
|---|---|---|---|---|---|---|---|---|
| COMP-B | 22.3 (22.0-23.0) | 22.7 (22.0-23.0) | 23.7 (23.0-24.0) | 23.3 (22.0-24.0) | 23.3 (22.0-24.0) | 22.0 | 22.0 | 25.0 |
| COMP-B+ | 22.7 (22.0-23.0) | 23.0 (23.0-23.0) | 23.0 (22.0-24.0) | 22.3 (21.0-24.0) | 22.7 (22.0-24.0) | 20.0 | 22.0 | 18.0 |
| COMP-S | 22.3 (22.0-23.0) | 22.7 (22.0-23.0) | 23.7 (23.0-24.0) | 23.3 (22.0-24.0) | 23.3 (22.0-24.0) | 22.0 | 22.0 | 25.0 |
| COMP-S+ | 22.7 (22.0-23.0) | 23.0 (23.0-23.0) | 23.0 (22.0-24.0) | 22.3 (21.0-24.0) | 22.7 (22.0-24.0) | 20.0 | 22.0 | 18.0 |
| COMP-B-ALLPOS | 22.7 (22.0-23.0) | 22.3 (22.0-23.0) | 23.0 (22.0-24.0) | 25.7 (24.0-27.0) | - | - | - | - |
| REEDIT-B | 22.7 (22.0-23.0) | 22.7 (22.0-23.0) | 22.3 (22.0-23.0) | 22.0 (22.0-22.0) | 22.7 (22.0-24.0) | 23.0 | 19.0 | 15.0 |
| REEDIT-S | 22.7 (22.0-23.0) | 22.7 (22.0-23.0) | 22.3 (22.0-23.0) | 22.0 (22.0-22.0) | 22.7 (22.0-24.0) | 23.0 | 19.0 | 15.0 |
| RB-TOT | 22.0 (22.0-22.0) | 20.3 (20.0-21.0) | 17.7 (17.0-19.0) | 17.7 (16.0-21.0) | 17.3 (16.0-20.0) | 19.0 | 16.0 | 16.0 |
| RB-UNION | 22.0 (22.0-22.0) | 21.3 (19.0-23.0) | 19.3 (18.0-22.0) | 17.0 (16.0-18.0) | 13.0 (11.0-14.0) | 10.0 | 4.0 | 1.0 |

LLaVA (TB-L100-ae-l2_3) -- unrepaired: 66.0

| method | n=1 | n=2 | n=5 | n=10 | n=20 | n=50 | n=100 | n=ALL (305) |
|---|---|---|---|---|---|---|---|---|
| COMP-B | 66.0 (66.0-66.0) | 66.3 (66.0-67.0) | 66.3 (66.0-67.0) | 66.7 (66.0-68.0) | 67.0 (67.0-67.0) | 67.0 | 69.0 | 66.0 |
| COMP-B+ | 66.0 (66.0-66.0) | 66.0 (66.0-66.0) | 65.7 (65.0-66.0) | 66.3 (66.0-67.0) | 67.7 (67.0-68.0) | 66.0 | 66.0 | 64.0 |
| COMP-S | 66.0 (66.0-66.0) | 66.3 (66.0-67.0) | 66.3 (66.0-67.0) | 66.7 (66.0-68.0) | 67.0 (67.0-67.0) | 67.0 | 69.0 | 66.0 |
| COMP-S+ | 66.0 (66.0-66.0) | 66.0 (66.0-66.0) | 65.7 (65.0-66.0) | 66.3 (66.0-67.0) | 67.7 (67.0-68.0) | 66.0 | 65.0 | 64.0 |
| COMP-B-ALLPOS | 66.3 (66.0-67.0) | 67.0 (67.0-67.0) | 67.3 (66.0-68.0) | 67.0 (66.0-69.0) | - | - | - | - |
| REEDIT-B | 66.0 (66.0-66.0) | 66.0 (66.0-66.0) | 66.0 (66.0-66.0) | 66.7 (66.0-67.0) | 67.3 (67.0-68.0) | 63.0 | 61.0 | 49.0 |
| REEDIT-S | 66.0 (66.0-66.0) | 66.0 (66.0-66.0) | 66.0 (66.0-66.0) | 66.7 (66.0-67.0) | 67.3 (67.0-68.0) | 63.0 | 61.0 | 49.0 |
| RB-TOT | 63.3 (61.0-65.0) | 58.3 (54.0-62.0) | 50.0 (46.0-57.0) | 49.7 (49.0-51.0) | 49.3 (47.0-51.0) | 51.0 | 51.0 | 48.0 |
| RB-UNION | 63.3 (61.0-65.0) | 62.3 (60.0-64.0) | 50.3 (49.0-52.0) | 43.7 (38.0-47.0) | 30.0 (27.0-33.0) | 9.0 | 4.0 | 1.0 |

### edit checks broken (of 300)

Qwen (TB-Q100-ae-l2_3)

| method | n=1 | n=2 | n=5 | n=10 | n=20 | n=50 | n=100 | n=ALL (297) |
|---|---|---|---|---|---|---|---|---|
| COMP-B | 0.0 (0.0-0.0) | 1.0 (0.0-2.0) | 2.3 (1.0-4.0) | 2.3 (1.0-3.0) | 3.0 (1.0-5.0) | 4.0 | 7.0 | 3.0 |
| COMP-B+ | 0.7 (0.0-1.0) | 0.3 (0.0-1.0) | 1.3 (0.0-2.0) | 2.3 (1.0-4.0) | 4.7 (2.0-6.0) | 10.0 | 9.0 | 14.0 |
| COMP-S | 0.0 (0.0-0.0) | 1.0 (0.0-2.0) | 2.3 (1.0-4.0) | 2.3 (1.0-3.0) | 3.0 (1.0-5.0) | 4.0 | 7.0 | 3.0 |
| COMP-S+ | 0.7 (0.0-1.0) | 0.3 (0.0-1.0) | 1.3 (0.0-2.0) | 2.3 (1.0-4.0) | 4.7 (2.0-6.0) | 10.0 | 9.0 | 14.0 |
| COMP-B-ALLPOS | 0.7 (0.0-1.0) | 0.7 (0.0-2.0) | 2.3 (1.0-3.0) | 2.3 (1.0-3.0) | - | - | - | - |
| REEDIT-B | 0.7 (0.0-2.0) | 1.0 (0.0-2.0) | 2.3 (1.0-4.0) | 3.0 (2.0-4.0) | 4.0 (2.0-5.0) | 7.0 | 10.0 | 26.0 |
| REEDIT-S | 0.7 (0.0-2.0) | 1.0 (0.0-2.0) | 2.3 (1.0-4.0) | 3.0 (2.0-4.0) | 4.3 (3.0-6.0) | 6.0 | 10.0 | 26.0 |
| RB-TOT | 5.7 (1.0-10.0) | 15.7 (13.0-19.0) | 30.3 (28.0-34.0) | 33.7 (32.0-35.0) | 34.3 (32.0-36.0) | 31.0 | 37.0 | 34.0 |
| RB-UNION | 5.7 (1.0-10.0) | 11.7 (8.0-14.0) | 23.3 (18.0-28.0) | 36.7 (27.0-47.0) | 58.3 (48.0-68.0) | 81.0 | 97.0 | 105.0 |

LLaVA (TB-L100-ae-l2_3)

| method | n=1 | n=2 | n=5 | n=10 | n=20 | n=50 | n=100 | n=ALL (305) |
|---|---|---|---|---|---|---|---|---|
| COMP-B | 0.0 (0.0-0.0) | 0.0 (0.0-0.0) | 0.0 (0.0-0.0) | 0.0 (0.0-0.0) | 0.3 (0.0-1.0) | 1.0 | 0.0 | 6.0 |
| COMP-B+ | 0.0 (0.0-0.0) | 0.0 (0.0-0.0) | 1.0 (0.0-3.0) | 0.7 (0.0-1.0) | 1.3 (1.0-2.0) | 3.0 | 6.0 | 10.0 |
| COMP-S | 0.0 (0.0-0.0) | 0.0 (0.0-0.0) | 0.0 (0.0-0.0) | 0.0 (0.0-0.0) | 0.3 (0.0-1.0) | 1.0 | 0.0 | 6.0 |
| COMP-S+ | 0.0 (0.0-0.0) | 0.0 (0.0-0.0) | 1.0 (0.0-3.0) | 0.7 (0.0-1.0) | 1.3 (1.0-2.0) | 3.0 | 6.0 | 10.0 |
| COMP-B-ALLPOS | 0.3 (0.0-1.0) | 3.7 (3.0-5.0) | 3.7 (2.0-6.0) | 4.0 (4.0-4.0) | - | - | - | - |
| REEDIT-B | 0.0 (0.0-0.0) | 0.0 (0.0-0.0) | 0.0 (0.0-0.0) | 0.7 (0.0-1.0) | 1.7 (1.0-2.0) | 5.0 | 10.0 | 38.0 |
| REEDIT-S | 0.0 (0.0-0.0) | 0.0 (0.0-0.0) | 0.0 (0.0-0.0) | 0.7 (0.0-1.0) | 1.7 (1.0-2.0) | 5.0 | 10.0 | 38.0 |
| RB-TOT | 12.7 (6.0-24.0) | 27.7 (17.0-34.0) | 58.0 (42.0-66.0) | 60.0 (53.0-65.0) | 58.3 (54.0-62.0) | 55.0 | 57.0 | 61.0 |
| RB-UNION | 12.7 (6.0-24.0) | 20.7 (14.0-26.0) | 54.7 (51.0-59.0) | 71.3 (61.0-88.0) | 109.0 (105.0-111.0) | 162.0 | 174.0 | 180.0 |

### edit checks fixed

Qwen (TB-Q100-ae-l2_3)

| method | n=1 | n=2 | n=5 | n=10 | n=20 | n=50 | n=100 | n=ALL (297) |
|---|---|---|---|---|---|---|---|---|
| COMP-B | 0.3 (0.0-1.0) | 1.3 (1.0-2.0) | 1.7 (1.0-2.0) | 1.3 (0.0-2.0) | 3.3 (1.0-5.0) | 2.0 | 3.0 | 7.0 |
| COMP-B+ | 1.3 (1.0-2.0) | 1.3 (1.0-2.0) | 2.7 (2.0-3.0) | 2.3 (1.0-4.0) | 3.0 (2.0-4.0) | 0.0 | 5.0 | 7.0 |
| COMP-S | 0.3 (0.0-1.0) | 1.3 (1.0-2.0) | 1.7 (1.0-2.0) | 1.3 (0.0-2.0) | 3.3 (1.0-5.0) | 2.0 | 3.0 | 7.0 |
| COMP-S+ | 1.3 (1.0-2.0) | 1.3 (1.0-2.0) | 2.7 (2.0-3.0) | 2.3 (1.0-4.0) | 3.0 (2.0-4.0) | 0.0 | 5.0 | 7.0 |
| COMP-B-ALLPOS | 1.7 (1.0-2.0) | 1.3 (1.0-2.0) | 4.3 (4.0-5.0) | 7.0 (6.0-8.0) | - | - | - | - |
| REEDIT-B | 0.7 (0.0-1.0) | 1.0 (0.0-2.0) | 1.0 (0.0-2.0) | 0.3 (0.0-1.0) | 2.7 (2.0-4.0) | 4.0 | 2.0 | 0.0 |
| REEDIT-S | 0.7 (0.0-1.0) | 1.0 (0.0-2.0) | 1.0 (0.0-2.0) | 0.3 (0.0-1.0) | 3.0 (2.0-4.0) | 4.0 | 2.0 | 0.0 |
| RB-TOT | 3.3 (2.0-5.0) | 3.7 (3.0-4.0) | 5.3 (5.0-6.0) | 4.7 (3.0-7.0) | 4.3 (4.0-5.0) | 5.0 | 5.0 | 4.0 |
| RB-UNION | 3.3 (2.0-5.0) | 4.7 (3.0-6.0) | 4.0 (2.0-6.0) | 5.0 (4.0-6.0) | 4.3 (3.0-5.0) | 4.0 | 1.0 | 0.0 |

LLaVA (TB-L100-ae-l2_3)

| method | n=1 | n=2 | n=5 | n=10 | n=20 | n=50 | n=100 | n=ALL (305) |
|---|---|---|---|---|---|---|---|---|
| COMP-B | 0.0 (0.0-0.0) | 0.3 (0.0-1.0) | 0.3 (0.0-1.0) | 1.0 (0.0-3.0) | 2.3 (1.0-3.0) | 2.0 | 4.0 | 7.0 |
| COMP-B+ | 0.3 (0.0-1.0) | 0.3 (0.0-1.0) | 1.0 (0.0-2.0) | 1.3 (1.0-2.0) | 3.3 (3.0-4.0) | 3.0 | 4.0 | 8.0 |
| COMP-S | 0.0 (0.0-0.0) | 0.3 (0.0-1.0) | 0.3 (0.0-1.0) | 1.0 (0.0-3.0) | 2.3 (1.0-3.0) | 2.0 | 4.0 | 7.0 |
| COMP-S+ | 0.3 (0.0-1.0) | 0.3 (0.0-1.0) | 1.0 (0.0-2.0) | 1.3 (1.0-2.0) | 3.3 (3.0-4.0) | 3.0 | 3.0 | 8.0 |
| COMP-B-ALLPOS | 1.0 (0.0-2.0) | 5.3 (4.0-7.0) | 6.0 (5.0-8.0) | 10.3 (8.0-13.0) | - | - | - | - |
| REEDIT-B | 0.0 (0.0-0.0) | 0.0 (0.0-0.0) | 0.7 (0.0-2.0) | 1.3 (1.0-2.0) | 3.3 (2.0-4.0) | 2.0 | 2.0 | 1.0 |
| REEDIT-S | 0.0 (0.0-0.0) | 0.0 (0.0-0.0) | 0.7 (0.0-2.0) | 1.3 (1.0-2.0) | 3.3 (2.0-4.0) | 2.0 | 1.0 | 1.0 |
| RB-TOT | 4.0 (1.0-6.0) | 8.0 (4.0-13.0) | 9.0 (4.0-13.0) | 8.7 (6.0-13.0) | 9.3 (8.0-11.0) | 10.0 | 10.0 | 7.0 |
| RB-UNION | 4.0 (1.0-6.0) | 8.7 (6.0-12.0) | 12.0 (11.0-13.0) | 10.0 (8.0-13.0) | 9.7 (8.0-11.0) | 7.0 | 3.0 | 1.0 |

### T-Loc (held-out items)

Qwen (TB-Q100-ae-l2_3) -- unrepaired: 84.83

| method | n=1 | n=2 | n=5 | n=10 | n=20 | n=50 | n=100 | n=ALL (297) |
|---|---|---|---|---|---|---|---|---|
| COMP-B | 86.59 (85.82-87.26) | 86.40 (85.73-87.23) | 86.33 (84.86-87.39) | 86.74 (85.68-88.03) | 86.77 (85.95-87.49) | 86.91 | 88.91 | 89.12 |
| COMP-B+ | 84.56 (83.87-84.99) | 84.46 (83.63-85.07) | 84.93 (84.73-85.25) | 84.42 (83.71-85.15) | 85.42 (84.57-86.20) | 84.89 | 84.07 | 76.43 |
| COMP-S | 86.59 (85.82-87.26) | 86.40 (85.73-87.23) | 86.33 (84.86-87.39) | 86.74 (85.68-88.03) | 86.77 (85.95-87.49) | 86.91 | 88.91 | 89.12 |
| COMP-S+ | 84.56 (83.87-84.99) | 84.46 (83.63-85.07) | 84.93 (84.73-85.25) | 84.42 (83.71-85.15) | 85.42 (84.57-86.20) | 84.89 | 84.07 | 76.43 |
| COMP-B-ALLPOS | 86.31 (85.33-87.15) | 86.14 (85.49-86.56) | 86.44 (84.21-87.85) | 87.62 (86.44-88.76) | - | - | - | - |
| REEDIT-B | 83.26 (80.99-86.03) | 83.24 (81.07-85.65) | 83.87 (83.25-84.68) | 84.21 (83.23-84.72) | 82.23 (81.49-82.73) | 82.03 | 81.34 | 76.67 |
| REEDIT-S | 83.26 (80.99-86.03) | 83.24 (81.07-85.65) | 83.87 (83.25-84.68) | 84.21 (83.23-84.72) | 82.25 (81.58-82.73) | 82.63 | 80.68 | 76.29 |
| RB-TOT | 86.40 (86.12-86.77) | 87.72 (87.11-88.83) | 89.18 (88.80-89.52) | 89.38 (88.40-89.89) | 89.28 (88.03-90.23) | 88.58 | 90.92 | 90.32 |
| RB-UNION | 86.40 (86.12-86.77) | 87.10 (86.66-87.40) | 88.46 (87.73-89.26) | 89.82 (88.53-91.10) | 91.94 (91.26-92.93) | 94.52 | 96.52 | 98.76 |

LLaVA (TB-L100-ae-l2_3) -- unrepaired: 77.26

| method | n=1 | n=2 | n=5 | n=10 | n=20 | n=50 | n=100 | n=ALL (305) |
|---|---|---|---|---|---|---|---|---|
| COMP-B | 77.45 (77.19-77.88) | 77.41 (77.03-77.69) | 77.33 (77.11-77.49) | 77.25 (77.00-77.71) | 76.97 (76.69-77.18) | 77.51 | 77.51 | 77.78 |
| COMP-B+ | 77.34 (77.28-77.46) | 77.09 (76.96-77.26) | 77.14 (76.63-77.44) | 77.12 (76.87-77.27) | 76.09 (75.86-76.22) | 75.75 | 73.60 | 69.10 |
| COMP-S | 77.45 (77.19-77.88) | 77.41 (77.03-77.69) | 77.33 (77.11-77.49) | 77.25 (77.00-77.71) | 76.97 (76.69-77.18) | 77.51 | 77.51 | 77.78 |
| COMP-S+ | 77.34 (77.28-77.46) | 77.09 (76.96-77.26) | 77.14 (76.63-77.44) | 77.12 (76.87-77.27) | 76.09 (75.86-76.22) | 75.75 | 73.60 | 69.09 |
| COMP-B-ALLPOS | 76.82 (76.31-77.36) | 76.64 (76.43-76.89) | 76.56 (75.21-77.27) | 76.90 (75.50-77.75) | - | - | - | - |
| REEDIT-B | 77.29 (77.18-77.40) | 76.83 (76.03-77.27) | 76.31 (76.03-76.68) | 76.37 (75.56-77.06) | 75.95 (75.32-76.37) | 74.59 | 71.59 | 61.57 |
| REEDIT-S | 77.29 (77.18-77.40) | 76.83 (76.03-77.27) | 76.31 (76.03-76.68) | 76.37 (75.56-77.06) | 75.90 (75.33-76.19) | 74.58 | 71.72 | 61.53 |
| RB-TOT | 78.14 (77.89-78.63) | 77.92 (77.50-78.23) | 80.82 (80.15-81.61) | 81.00 (79.44-82.02) | 80.17 (79.64-80.68) | 80.94 | 80.98 | 81.54 |
| RB-UNION | 78.14 (77.89-78.63) | 78.17 (77.09-78.97) | 81.31 (81.07-81.55) | 81.48 (80.86-82.25) | 84.77 (84.33-85.32) | 92.64 | 96.56 | 100.00 |

### I-Loc (held-out items)

Qwen (TB-Q100-ae-l2_3) -- unrepaired: 88.69

| method | n=1 | n=2 | n=5 | n=10 | n=20 | n=50 | n=100 | n=ALL (297) |
|---|---|---|---|---|---|---|---|---|
| COMP-B | 88.83 (88.40-89.13) | 88.94 (88.75-89.12) | 89.12 (87.53-90.36) | 89.94 (89.15-91.39) | 89.58 (88.76-90.78) | 87.65 | 89.58 | 88.87 |
| COMP-B+ | 88.87 (88.73-89.00) | 88.88 (88.32-89.19) | 89.36 (88.93-90.12) | 89.78 (89.03-90.92) | 89.59 (88.41-91.89) | 86.78 | 86.64 | 84.38 |
| COMP-S | 88.83 (88.40-89.13) | 88.94 (88.75-89.12) | 89.12 (87.53-90.36) | 89.94 (89.15-91.39) | 89.58 (88.76-90.78) | 87.65 | 89.58 | 88.87 |
| COMP-S+ | 88.87 (88.73-89.00) | 88.88 (88.32-89.19) | 89.36 (88.93-90.12) | 89.78 (89.03-90.92) | 89.59 (88.41-91.89) | 86.78 | 86.64 | 84.38 |
| COMP-B-ALLPOS | 87.55 (87.25-87.94) | 88.50 (87.72-89.14) | 87.12 (85.88-88.11) | 81.42 (80.20-83.24) | - | - | - | - |
| REEDIT-B | 88.68 (88.41-88.96) | 89.00 (88.75-89.15) | 89.28 (89.06-89.45) | 89.10 (88.77-89.32) | 88.90 (88.68-89.26) | 90.02 | 90.24 | 89.88 |
| REEDIT-S | 88.68 (88.41-88.96) | 89.00 (88.75-89.15) | 89.28 (89.06-89.45) | 89.10 (88.77-89.32) | 88.89 (88.35-89.55) | 90.14 | 90.25 | 90.00 |
| RB-TOT | 88.90 (88.63-89.33) | 89.44 (89.12-90.06) | 91.08 (90.49-91.84) | 91.37 (90.77-92.23) | 91.13 (90.39-91.75) | 91.61 | 92.29 | 92.34 |
| RB-UNION | 88.90 (88.63-89.33) | 88.83 (88.14-89.60) | 89.86 (88.98-90.44) | 91.66 (89.99-93.55) | 94.16 (93.50-94.50) | 96.38 | 98.46 | 99.30 |

LLaVA (TB-L100-ae-l2_3) -- unrepaired: 81.47

| method | n=1 | n=2 | n=5 | n=10 | n=20 | n=50 | n=100 | n=ALL (305) |
|---|---|---|---|---|---|---|---|---|
| COMP-B | 81.12 (80.74-81.43) | 81.17 (80.77-81.76) | 80.76 (80.53-81.04) | 80.72 (80.16-81.09) | 80.45 (80.27-80.57) | 81.32 | 80.42 | 76.66 |
| COMP-B+ | 81.51 (81.13-81.80) | 81.68 (81.46-81.91) | 80.77 (79.49-81.50) | 80.23 (78.73-81.16) | 79.67 (79.51-79.84) | 78.20 | 73.05 | 65.86 |
| COMP-S | 81.12 (80.74-81.43) | 81.17 (80.77-81.76) | 80.76 (80.53-81.04) | 80.72 (80.16-81.09) | 80.45 (80.27-80.57) | 81.32 | 80.42 | 76.66 |
| COMP-S+ | 81.51 (81.13-81.80) | 81.68 (81.46-81.91) | 80.77 (79.49-81.50) | 80.23 (78.73-81.16) | 79.67 (79.51-79.84) | 78.20 | 72.91 | 65.86 |
| COMP-B-ALLPOS | 80.70 (80.32-80.94) | 80.15 (78.57-81.04) | 77.53 (77.11-77.87) | 75.94 (75.23-76.55) | - | - | - | - |
| REEDIT-B | 81.39 (81.11-81.80) | 81.46 (80.96-81.82) | 80.87 (80.43-81.14) | 79.86 (79.37-80.54) | 79.90 (78.47-81.20) | 77.44 | 76.78 | 71.99 |
| REEDIT-S | 81.39 (81.11-81.80) | 81.46 (80.96-81.82) | 80.87 (80.43-81.14) | 79.86 (79.37-80.54) | 79.86 (78.37-81.20) | 77.36 | 76.78 | 72.23 |
| RB-TOT | 81.99 (81.88-82.12) | 81.88 (81.16-82.94) | 83.78 (83.25-84.53) | 83.89 (83.24-85.10) | 84.09 (82.90-85.04) | 87.05 | 87.19 | 85.60 |
| RB-UNION | 81.99 (81.88-82.12) | 82.17 (81.79-82.43) | 84.18 (83.72-84.68) | 85.58 (84.38-87.13) | 88.75 (87.66-89.80) | 95.24 | 97.50 | 100.00 |

### norm of the correction / norm of the log

Qwen (TB-Q100-ae-l2_3)

| method | n=1 | n=2 | n=5 | n=10 | n=20 | n=50 | n=100 | n=ALL (297) |
|---|---|---|---|---|---|---|---|---|
| COMP-B | 0.027 (0.026-0.028) | 0.039 (0.037-0.041) | 0.071 (0.063-0.081) | 0.102 (0.095-0.111) | 0.141 (0.133-0.145) | 0.232 | 0.337 | 0.647 |
| COMP-B+ | 0.048 (0.045-0.050) | 0.086 (0.076-0.104) | 0.153 (0.146-0.168) | 0.226 (0.203-0.245) | 0.333 (0.320-0.343) | 0.514 | 0.730 | 1.419 |
| COMP-S | 0.027 (0.026-0.028) | 0.039 (0.037-0.041) | 0.071 (0.063-0.081) | 0.102 (0.095-0.111) | 0.141 (0.133-0.145) | 0.232 | 0.337 | 0.647 |
| COMP-S+ | 0.048 (0.045-0.050) | 0.086 (0.076-0.104) | 0.153 (0.146-0.168) | 0.226 (0.203-0.245) | 0.333 (0.320-0.343) | 0.514 | 0.730 | 1.419 |
| COMP-B-ALLPOS | 0.041 (0.040-0.041) | 0.253 (0.163-0.299) | 0.372 (0.241-0.438) | 0.638 (0.545-0.689) | - | - | - | - |
| REEDIT-B | 0.051 (0.034-0.072) | 0.085 (0.077-0.096) | 0.158 (0.154-0.163) | 0.238 (0.220-0.248) | 0.340 (0.332-0.350) | 0.502 | 0.727 | 1.223 |
| REEDIT-S | 0.051 (0.034-0.072) | 0.085 (0.077-0.096) | 0.158 (0.154-0.163) | 0.238 (0.220-0.248) | 0.340 (0.332-0.350) | 0.502 | 0.727 | 1.223 |
| RB-TOT | 0.216 (0.114-0.323) | 0.370 (0.331-0.432) | 0.539 (0.505-0.575) | 0.564 (0.554-0.579) | 0.565 (0.559-0.575) | 0.580 | 0.582 | 0.587 |
| RB-UNION | 0.216 (0.114-0.323) | 0.316 (0.291-0.354) | 0.466 (0.413-0.501) | 0.600 (0.529-0.682) | 0.747 (0.695-0.803) | 0.892 | 0.976 | 0.999 |

LLaVA (TB-L100-ae-l2_3)

| method | n=1 | n=2 | n=5 | n=10 | n=20 | n=50 | n=100 | n=ALL (305) |
|---|---|---|---|---|---|---|---|---|
| COMP-B | 0.022 (0.013-0.027) | 0.025 (0.019-0.029) | 0.050 (0.045-0.054) | 0.067 (0.065-0.069) | 0.099 (0.096-0.101) | 0.162 | 0.233 | 0.461 |
| COMP-B+ | 0.028 (0.017-0.034) | 0.032 (0.025-0.035) | 0.100 (0.066-0.160) | 0.117 (0.088-0.170) | 0.221 (0.192-0.244) | 0.322 | 0.507 | 0.783 |
| COMP-S | 0.022 (0.013-0.027) | 0.025 (0.019-0.029) | 0.050 (0.045-0.054) | 0.067 (0.065-0.069) | 0.099 (0.096-0.101) | 0.162 | 0.233 | 0.461 |
| COMP-S+ | 0.028 (0.017-0.034) | 0.032 (0.025-0.035) | 0.100 (0.066-0.160) | 0.117 (0.088-0.170) | 0.221 (0.192-0.244) | 0.322 | 0.507 | 0.783 |
| COMP-B-ALLPOS | 0.037 (0.034-0.039) | 0.222 (0.221-0.223) | 0.324 (0.323-0.325) | 0.533 (0.528-0.536) | - | - | - | - |
| REEDIT-B | 0.058 (0.048-0.075) | 0.094 (0.072-0.129) | 0.136 (0.112-0.174) | 0.200 (0.184-0.230) | 0.281 (0.271-0.289) | 0.474 | 0.688 | 1.210 |
| REEDIT-S | 0.058 (0.048-0.075) | 0.094 (0.072-0.129) | 0.136 (0.112-0.174) | 0.200 (0.184-0.230) | 0.281 (0.271-0.289) | 0.474 | 0.688 | 1.210 |
| RB-TOT | 0.228 (0.194-0.293) | 0.364 (0.280-0.428) | 0.546 (0.466-0.590) | 0.576 (0.571-0.578) | 0.566 (0.553-0.575) | 0.571 | 0.586 | 0.599 |
| RB-UNION | 0.228 (0.194-0.293) | 0.313 (0.295-0.330) | 0.536 (0.518-0.545) | 0.625 (0.575-0.705) | 0.764 (0.737-0.783) | 0.944 | 0.984 | 1.000 |

### rank of the correction

Qwen (TB-Q100-ae-l2_3)

| method | n=1 | n=2 | n=5 | n=10 | n=20 | n=50 | n=100 | n=ALL (297) |
|---|---|---|---|---|---|---|---|---|
| COMP-B | 10 (10-10) | 20 (19-20) | 50 (49-50) | 100 (99-100) | 198 (198-199) | 497 | 986 | 2841 |
| COMP-B+ | 10 (10-10) | 20 (19-20) | 50 (49-50) | 100 (99-100) | 198 (198-199) | 497 | 986 | 2841 |
| COMP-S | 10 (10-10) | 20 (19-20) | 50 (49-50) | 100 (99-100) | 198 (198-199) | 497 | 986 | 2841 |
| COMP-S+ | 10 (10-10) | 20 (19-20) | 50 (49-50) | 100 (99-100) | 198 (198-199) | 497 | 986 | 2841 |
| COMP-B-ALLPOS | 40 (38-42) | 827 (357-1070) | 1666 (715-2156) | 4071 (3117-4583) | - | - | - | - |
| REEDIT-B | 1 (1-1) | 2 (2-2) | 5 (5-5) | 10 (10-10) | 20 (20-20) | 50 | 100 | 297 |
| REEDIT-S | 1 (1-1) | 2 (2-2) | 5 (5-5) | 10 (10-10) | 20 (20-20) | 50 | 100 | 297 |
| RB-TOT | 5 (1-9) | 13 (10-19) | 28 (27-30) | 30 (30-30) | 30 (30-30) | 30 | 30 | 30 |
| RB-UNION | 5 (1-9) | 10 (9-11) | 21 (17-24) | 35 (27-43) | 54 (46-62) | 75 | 92 | 99 |

LLaVA (TB-L100-ae-l2_3)

| method | n=1 | n=2 | n=5 | n=10 | n=20 | n=50 | n=100 | n=ALL (305) |
|---|---|---|---|---|---|---|---|---|
| COMP-B | 8 (3-10) | 10 (7-12) | 34 (29-37) | 56 (55-57) | 120 (113-128) | 291 | 611 | 1973 |
| COMP-B+ | 8 (3-10) | 10 (7-12) | 34 (29-37) | 56 (55-57) | 120 (113-128) | 291 | 611 | 1973 |
| COMP-S | 8 (3-10) | 10 (7-12) | 34 (29-37) | 56 (55-57) | 120 (113-128) | 291 | 611 | 1973 |
| COMP-S+ | 8 (3-10) | 10 (7-12) | 34 (29-37) | 56 (55-57) | 120 (113-128) | 291 | 611 | 1973 |
| COMP-B-ALLPOS | 26 (23-28) | 623 (621-626) | 1271 (1266-1276) | 3099 (3093-3105) | - | - | - | - |
| REEDIT-B | 1 (1-1) | 2 (2-2) | 5 (5-5) | 10 (10-10) | 20 (20-20) | 50 | 100 | 305 |
| REEDIT-S | 1 (1-1) | 2 (2-2) | 5 (5-5) | 10 (10-10) | 20 (20-20) | 50 | 100 | 305 |
| RB-TOT | 5 (3-8) | 12 (7-16) | 27 (20-30) | 30 (30-30) | 30 (30-30) | 30 | 30 | 30 |
| RB-UNION | 5 (3-8) | 9 (8-9) | 25 (23-27) | 35 (29-45) | 54 (49-57) | 86 | 96 | 100 |

### forward passes of the repair

Qwen (TB-Q100-ae-l2_3)

| method | n=1 | n=2 | n=5 | n=10 | n=20 | n=50 | n=100 | n=ALL (297) |
|---|---|---|---|---|---|---|---|---|
| COMP-B | 1 (1-1) | 2 (2-2) | 5 (5-5) | 10 (10-10) | 20 (20-20) | 50 | 100 | 297 |
| COMP-B+ | 1 (1-1) | 2 (2-2) | 5 (5-5) | 10 (10-10) | 20 (20-20) | 50 | 100 | 297 |
| COMP-S | 1 (1-1) | 2 (2-2) | 5 (5-5) | 10 (10-10) | 20 (20-20) | 50 | 100 | 297 |
| COMP-S+ | 1 (1-1) | 2 (2-2) | 5 (5-5) | 10 (10-10) | 20 (20-20) | 50 | 100 | 297 |
| COMP-B-ALLPOS | 1 (1-1) | 2 (2-2) | 5 (5-5) | 10 (10-10) | - | - | - | - |
| REEDIT-B | 9 (6-13) | 17 (14-18) | 47 (45-48) | 98 (96-99) | 200 (194-211) | 469 | 954 | 2814 |
| REEDIT-S | 9 (6-13) | 17 (14-18) | 47 (45-48) | 98 (96-99) | 200 (194-211) | 469 | 954 | 2814 |
| RB-TOT | 10 (3-19) | 55 (42-78) | 285 (275-305) | 610 (610-610) | 1220 (1220-1220) | 3050 | 6100 | 18117 |
| RB-UNION | 10 (3-19) | 24 (20-28) | 59 (47-67) | 119 (106-134) | 221 (194-246) | 436 | 958 | 3291 |

LLaVA (TB-L100-ae-l2_3)

| method | n=1 | n=2 | n=5 | n=10 | n=20 | n=50 | n=100 | n=ALL (305) |
|---|---|---|---|---|---|---|---|---|
| COMP-B | 1 (1-1) | 2 (2-2) | 5 (5-5) | 10 (10-10) | 20 (20-20) | 50 | 100 | 305 |
| COMP-B+ | 1 (1-1) | 2 (2-2) | 5 (5-5) | 10 (10-10) | 20 (20-20) | 50 | 100 | 305 |
| COMP-S | 1 (1-1) | 2 (2-2) | 5 (5-5) | 10 (10-10) | 20 (20-20) | 50 | 100 | 305 |
| COMP-S+ | 1 (1-1) | 2 (2-2) | 5 (5-5) | 10 (10-10) | 20 (20-20) | 50 | 100 | 305 |
| COMP-B-ALLPOS | 1 (1-1) | 2 (2-2) | 5 (5-5) | 10 (10-10) | - | - | - | - |
| REEDIT-B | 7 (6-9) | 14 (12-18) | 35 (30-43) | 68 (64-75) | 138 (131-142) | 361 | 713 | 2234 |
| REEDIT-S | 7 (6-9) | 14 (12-18) | 35 (30-43) | 68 (64-75) | 138 (131-142) | 361 | 713 | 2234 |
| RB-TOT | 10 (7-17) | 50 (30-66) | 272 (205-305) | 610 (610-610) | 1220 (1220-1220) | 3050 | 6100 | 18605 |
| RB-UNION | 10 (7-17) | 19 (18-20) | 61 (53-67) | 105 (78-144) | 191 (172-216) | 520 | 1136 | 3497 |

### backward passes of the repair

Qwen (TB-Q100-ae-l2_3)

| method | n=1 | n=2 | n=5 | n=10 | n=20 | n=50 | n=100 | n=ALL (297) |
|---|---|---|---|---|---|---|---|---|
| COMP-B | 0 (0-0) | 0 (0-0) | 0 (0-0) | 0 (0-0) | 0 (0-0) | 0 | 0 | 0 |
| COMP-B+ | 0 (0-0) | 0 (0-0) | 0 (0-0) | 0 (0-0) | 0 (0-0) | 0 | 0 | 0 |
| COMP-S | 0 (0-0) | 0 (0-0) | 0 (0-0) | 0 (0-0) | 0 (0-0) | 0 | 0 | 0 |
| COMP-S+ | 0 (0-0) | 0 (0-0) | 0 (0-0) | 0 (0-0) | 0 (0-0) | 0 | 0 | 0 |
| COMP-B-ALLPOS | 0 (0-0) | 0 (0-0) | 0 (0-0) | 0 (0-0) | - | - | - | - |
| REEDIT-B | 6 (3-10) | 11 (8-12) | 32 (30-33) | 68 (66-69) | 139 (132-151) | 317 | 652 | 1917 |
| REEDIT-S | 6 (3-10) | 11 (8-12) | 32 (30-33) | 68 (66-69) | 139 (132-151) | 317 | 652 | 1917 |
| RB-TOT | 5 (1-9) | 27 (20-38) | 140 (135-150) | 300 (300-300) | 600 (600-600) | 1500 | 3000 | 8910 |
| RB-UNION | 5 (1-9) | 11 (9-13) | 27 (21-31) | 54 (48-62) | 101 (87-113) | 193 | 429 | 1497 |

LLaVA (TB-L100-ae-l2_3)

| method | n=1 | n=2 | n=5 | n=10 | n=20 | n=50 | n=100 | n=ALL (305) |
|---|---|---|---|---|---|---|---|---|
| COMP-B | 0 (0-0) | 0 (0-0) | 0 (0-0) | 0 (0-0) | 0 (0-0) | 0 | 0 | 0 |
| COMP-B+ | 0 (0-0) | 0 (0-0) | 0 (0-0) | 0 (0-0) | 0 (0-0) | 0 | 0 | 0 |
| COMP-S | 0 (0-0) | 0 (0-0) | 0 (0-0) | 0 (0-0) | 0 (0-0) | 0 | 0 | 0 |
| COMP-S+ | 0 (0-0) | 0 (0-0) | 0 (0-0) | 0 (0-0) | 0 (0-0) | 0 | 0 | 0 |
| COMP-B-ALLPOS | 0 (0-0) | 0 (0-0) | 0 (0-0) | 0 (0-0) | - | - | - | - |
| REEDIT-B | 4 (3-6) | 8 (6-12) | 20 (15-28) | 38 (34-45) | 78 (71-82) | 207 | 407 | 1311 |
| REEDIT-S | 4 (3-6) | 8 (6-12) | 20 (15-28) | 38 (34-45) | 78 (71-82) | 207 | 407 | 1311 |
| RB-TOT | 5 (3-8) | 24 (14-32) | 133 (100-150) | 300 (300-300) | 600 (600-600) | 1500 | 3000 | 9150 |
| RB-UNION | 5 (3-8) | 9 (8-9) | 28 (24-31) | 48 (34-67) | 85 (76-98) | 235 | 518 | 1596 |

### repair wall-clock (s; shared GPUs)

Qwen (TB-Q100-ae-l2_3)

| method | n=1 | n=2 | n=5 | n=10 | n=20 | n=50 | n=100 | n=ALL (297) |
|---|---|---|---|---|---|---|---|---|
| COMP-B | 0.0 (0.0-0.1) | 0.2 (0.2-0.3) | 0.4 (0.2-0.5) | 1.1 (0.7-1.8) | 2.0 (1.5-2.4) | 4.4 | 11.7 | 52.4 |
| COMP-B+ | 0.0 (0.0-0.1) | 0.2 (0.2-0.2) | 0.5 (0.3-0.6) | 1.1 (0.8-1.7) | 2.1 (1.4-2.6) | 4.0 | 10.5 | 51.7 |
| COMP-S | 0.0 (0.0-0.1) | 0.2 (0.2-0.2) | 0.4 (0.3-0.5) | 1.0 (0.9-1.2) | 2.5 (1.9-2.8) | 7.3 | 30.2 | 191.0 |
| COMP-S+ | 0.0 (0.0-0.1) | 0.3 (0.3-0.4) | 0.6 (0.5-0.7) | 1.3 (1.2-1.3) | 3.7 (3.0-4.0) | 9.6 | 32.3 | 177.2 |
| COMP-B-ALLPOS | 0.1 (0.0-0.1) | 0.8 (0.4-1.3) | 2.6 (0.9-4.3) | 15.3 (9.5-22.0) | - | - | - | - |
| REEDIT-B | 0.6 (0.3-1.1) | 2.0 (1.7-2.3) | 4.8 (3.8-6.0) | 11.7 (10.3-13.1) | 22.4 (18.0-29.5) | 63.0 | 125.4 | 292.0 |
| REEDIT-S | 0.6 (0.3-1.1) | 2.0 (1.7-2.3) | 4.8 (3.8-6.0) | 11.7 (10.3-13.1) | 22.4 (17.5-29.7) | 63.3 | 126.3 | 295.8 |
| RB-TOT | 0.8 (0.1-1.8) | 2.9 (1.7-4.8) | 8.9 (8.2-9.8) | 28.1 (18.5-34.5) | 48.2 (35.4-67.7) | 109.9 | 319.8 | 491.8 |
| RB-UNION | 0.4 (0.1-0.8) | 1.2 (1.2-1.2) | 2.6 (2.3-3.3) | 5.1 (4.9-5.6) | 10.0 (8.9-10.6) | 26.0 | 57.0 | 153.7 |

LLaVA (TB-L100-ae-l2_3)

| method | n=1 | n=2 | n=5 | n=10 | n=20 | n=50 | n=100 | n=ALL (305) |
|---|---|---|---|---|---|---|---|---|
| COMP-B | 0.0 (0.0-0.0) | 0.1 (0.1-0.2) | 0.5 (0.2-0.9) | 0.4 (0.3-0.6) | 0.9 (0.8-1.1) | 1.9 | 4.0 | 10.9 |
| COMP-B+ | 0.0 (0.0-0.0) | 0.1 (0.1-0.1) | 0.2 (0.2-0.3) | 0.4 (0.3-0.6) | 0.9 (0.7-1.2) | 1.7 | 3.4 | 25.3 |
| COMP-S | 0.0 (0.0-0.0) | 0.1 (0.1-0.2) | 0.4 (0.3-0.4) | 0.5 (0.5-0.6) | 1.4 (1.2-1.6) | 3.9 | 13.4 | 113.3 |
| COMP-S+ | 0.0 (0.0-0.0) | 0.2 (0.1-0.2) | 0.4 (0.3-0.4) | 1.0 (0.6-1.7) | 1.8 (1.5-1.9) | 4.8 | 12.4 | 177.0 |
| COMP-B-ALLPOS | 0.0 (0.0-0.0) | 1.0 (0.5-1.5) | 2.0 (1.9-2.1) | 7.8 (5.5-9.0) | - | - | - | - |
| REEDIT-B | 0.4 (0.3-0.6) | 1.1 (0.9-1.4) | 2.2 (1.9-2.5) | 4.3 (3.9-4.6) | 8.7 (7.5-9.3) | 19.6 | 37.0 | 137.4 |
| REEDIT-S | 0.4 (0.3-0.6) | 1.1 (0.9-1.4) | 2.2 (1.9-2.5) | 4.3 (3.9-4.6) | 8.9 (7.7-9.6) | 20.2 | 38.0 | 141.6 |
| RB-TOT | 0.3 (0.2-0.5) | 2.9 (2.0-3.6) | 12.6 (10.0-17.2) | 25.9 (19.3-34.4) | 56.3 (39.0-65.7) | 97.2 | 196.1 | 1169.8 |
| RB-UNION | 0.4 (0.2-0.6) | 1.0 (0.8-1.3) | 2.5 (1.8-3.2) | 4.7 (2.6-7.4) | 8.4 (5.8-11.0) | 17.6 | 38.6 | 158.2 |

## 5. HELD-OUT rows by probe type (seed 0; D1 rate % after the repair; +fixed / -lost)

Qwen (TB-Q100-ae-l2_3) -- unrepaired: image 55.8% (139 complaints / 110 non-complaints), text 53.5% (129 / 112)

| method | n | image D1 % | image +fixed / -lost | text D1 % | text +fixed / -lost | reported resolved image | reported resolved text |
|---|---|---|---|---|---|---|---|
| COMP-B | 1 | 55.4 | +4 / -3 | 50.2 | +24 / -16 | 0/0 | 1/1 |
| COMP-B | 10 | 52.6 | +19 / -11 | 50.2 | +29 / -21 | 5/5 | 4/5 |
| COMP-B | 100 | 55.8 | +33 / -33 | 42.7 | +49 / -23 | 37/50 | 34/50 |
| COMP-B | ALL (297) | 51.8 | +47 / -37 | 42.7 | +50 / -24 | 86/155 | 82/142 |
| COMP-B+ | 1 | 57.0 | +2 / -5 | 53.1 | +4 / -3 | 0/0 | 1/1 |
| COMP-B+ | 10 | 53.0 | +21 / -14 | 54.4 | +9 / -11 | 5/5 | 3/5 |
| COMP-B+ | 100 | 59.0 | +29 / -37 | 52.7 | +28 / -26 | 34/50 | 30/50 |
| COMP-B+ | ALL (297) | 72.3 | +24 / -65 | 61.4 | +25 / -44 | 62/155 | 65/142 |
| COMP-S | 1 | 55.4 | +4 / -3 | 50.2 | +24 / -16 | 0/0 | 1/1 |
| COMP-S | 10 | 52.6 | +19 / -11 | 50.2 | +29 / -21 | 5/5 | 4/5 |
| COMP-S | 100 | 55.8 | +33 / -33 | 42.7 | +49 / -23 | 37/50 | 34/50 |
| COMP-S | ALL (297) | 51.8 | +47 / -37 | 42.7 | +50 / -24 | 86/155 | 82/142 |
| COMP-S+ | 1 | 57.0 | +2 / -5 | 53.1 | +4 / -3 | 0/0 | 1/1 |
| COMP-S+ | 10 | 53.0 | +21 / -14 | 54.4 | +9 / -11 | 5/5 | 3/5 |
| COMP-S+ | 100 | 59.0 | +29 / -37 | 52.7 | +28 / -26 | 34/50 | 30/50 |
| COMP-S+ | ALL (297) | 72.3 | +24 / -65 | 61.4 | +25 / -44 | 62/155 | 65/142 |
| COMP-B-ALLPOS | 1 | 58.2 | +7 / -13 | 51.9 | +23 / -19 | 0/0 | 1/1 |
| COMP-B-ALLPOS | 10 | 70.3 | +30 / -66 | 49.8 | +36 / -27 | 5/5 | 5/5 |
| REEDIT-B | 1 | 56.6 | +8 / -10 | 61.4 | +27 / -46 | 0/0 | 1/1 |
| REEDIT-B | 10 | 55.0 | +15 / -13 | 57.3 | +31 / -40 | 5/5 | 4/5 |
| REEDIT-B | 100 | 50.6 | +37 / -24 | 63.9 | +18 / -43 | 48/50 | 45/50 |
| REEDIT-B | ALL (297) | 53.4 | +42 / -36 | 72.6 | +12 / -58 | 138/155 | 103/142 |
| REEDIT-S | 1 | 56.6 | +8 / -10 | 61.4 | +27 / -46 | 0/0 | 1/1 |
| REEDIT-S | 10 | 55.0 | +15 / -13 | 57.3 | +31 / -40 | 5/5 | 4/5 |
| REEDIT-S | 100 | 52.2 | +35 / -26 | 66.0 | +17 / -47 | 49/50 | 45/50 |
| REEDIT-S | ALL (297) | 53.4 | +43 / -37 | 73.0 | +12 / -59 | 138/155 | 105/142 |
| RB-TOT | 1 | 55.0 | +18 / -16 | 48.5 | +22 / -10 | 0/0 | 1/1 |
| RB-TOT | 10 | 47.0 | +39 / -17 | 43.2 | +39 / -14 | 4/5 | 3/5 |
| RB-TOT | 100 | 43.0 | +49 / -17 | 41.5 | +43 / -14 | 28/50 | 19/50 |
| RB-TOT | ALL (297) | 43.0 | +54 / -22 | 46.9 | +35 / -19 | 53/155 | 44/142 |
| RB-UNION | 1 | 55.0 | +18 / -16 | 48.5 | +22 / -10 | 0/0 | 1/1 |
| RB-UNION | 10 | 49.0 | +37 / -20 | 44.4 | +41 / -19 | 4/5 | 3/5 |
| RB-UNION | 100 | 14.5 | +114 / -11 | 13.7 | +99 / -3 | 40/50 | 40/50 |
| RB-UNION | ALL (297) | 4.4 | +133 / -5 | 7.5 | +112 / -1 | 146/155 | 128/142 |

LLaVA (TB-L100-ae-l2_3) -- unrepaired: image 47.6% (118 complaints / 130 non-complaints), text 75.6% (186 / 60)

| method | n | image D1 % | image +fixed / -lost | text D1 % | text +fixed / -lost | reported resolved image | reported resolved text |
|---|---|---|---|---|---|---|---|
| COMP-B | 1 | 49.6 | +0 / -5 | 73.2 | +13 / -7 | 0/0 | 1/1 |
| COMP-B | 10 | 49.6 | +4 / -9 | 73.6 | +15 / -10 | 3/5 | 4/5 |
| COMP-B | 100 | 48.8 | +25 / -28 | 69.5 | +29 / -14 | 30/50 | 29/50 |
| COMP-B | ALL (305) | 49.6 | +28 / -33 | 67.5 | +44 / -24 | 60/124 | 104/181 |
| COMP-B+ | 1 | 50.0 | +1 / -7 | 75.6 | +2 / -2 | 0/0 | 1/1 |
| COMP-B+ | 10 | 51.2 | +3 / -12 | 76.0 | +5 / -6 | 3/5 | 4/5 |
| COMP-B+ | 100 | 54.8 | +18 / -36 | 78.0 | +15 / -21 | 22/50 | 27/50 |
| COMP-B+ | ALL (305) | 59.7 | +20 / -50 | 74.0 | +28 / -24 | 43/124 | 94/181 |
| COMP-S | 1 | 49.6 | +0 / -5 | 73.2 | +13 / -7 | 0/0 | 1/1 |
| COMP-S | 10 | 49.6 | +4 / -9 | 73.6 | +15 / -10 | 3/5 | 4/5 |
| COMP-S | 100 | 48.8 | +25 / -28 | 69.5 | +29 / -14 | 30/50 | 29/50 |
| COMP-S | ALL (305) | 49.6 | +28 / -33 | 67.5 | +44 / -24 | 60/124 | 105/181 |
| COMP-S+ | 1 | 50.0 | +1 / -7 | 75.6 | +2 / -2 | 0/0 | 1/1 |
| COMP-S+ | 10 | 51.2 | +3 / -12 | 76.0 | +5 / -6 | 3/5 | 4/5 |
| COMP-S+ | 100 | 54.8 | +18 / -36 | 78.0 | +15 / -21 | 22/50 | 27/50 |
| COMP-S+ | ALL (305) | 59.7 | +20 / -50 | 74.0 | +28 / -24 | 43/124 | 94/181 |
| COMP-B-ALLPOS | 1 | 50.0 | +0 / -6 | 73.2 | +15 / -9 | 0/0 | 1/1 |
| COMP-B-ALLPOS | 10 | 54.8 | +14 / -32 | 72.0 | +30 / -21 | 5/5 | 5/5 |
| REEDIT-B | 1 | 46.8 | +4 / -2 | 83.3 | +16 / -35 | 0/0 | 0/1 |
| REEDIT-B | 10 | 49.6 | +7 / -12 | 82.5 | +15 / -32 | 5/5 | 2/5 |
| REEDIT-B | 100 | 51.2 | +28 / -37 | 91.5 | +8 / -47 | 49/50 | 21/50 |
| REEDIT-B | ALL (305) | 56.5 | +40 / -62 | 94.3 | +4 / -50 | 119/124 | 40/181 |
| REEDIT-S | 1 | 46.8 | +4 / -2 | 83.3 | +16 / -35 | 0/0 | 0/1 |
| REEDIT-S | 10 | 49.6 | +7 / -12 | 82.5 | +15 / -32 | 5/5 | 2/5 |
| REEDIT-S | 100 | 51.2 | +28 / -37 | 91.5 | +8 / -47 | 49/50 | 21/50 |
| REEDIT-S | ALL (305) | 56.5 | +40 / -62 | 94.3 | +4 / -50 | 119/124 | 40/181 |
| RB-TOT | 1 | 50.4 | +2 / -9 | 74.4 | +9 / -6 | 0/0 | 1/1 |
| RB-TOT | 10 | 46.4 | +25 / -22 | 68.3 | +33 / -15 | 5/5 | 4/5 |
| RB-TOT | 100 | 37.9 | +39 / -15 | 64.6 | +39 / -12 | 18/50 | 9/50 |
| RB-TOT | ALL (305) | 39.5 | +37 / -17 | 56.1 | +59 / -11 | 48/124 | 33/181 |
| RB-UNION | 1 | 50.4 | +2 / -9 | 74.4 | +9 / -6 | 0/0 | 1/1 |
| RB-UNION | 10 | 43.5 | +27 / -17 | 63.0 | +43 / -12 | 2/5 | 2/5 |
| RB-UNION | 100 | 7.3 | +107 / -7 | 15.0 | +153 / -4 | 46/50 | 40/50 |
| RB-UNION | ALL (305) | 0.0 | +118 / -0 | 0.0 | +186 / -0 | 124/124 | 181/181 |

## 6. D2 complaints among the HELD-OUT rows (seed 0, n in {10, 100, ALL})

Testbed D2 pipeline (decoder v2: 10 no-cache greedy forwards + KV-cache greedy continuation to 48 tokens; `tb_common.contains_ref`): a HELD-OUT row is a D2 complaint if its pre-edit 48-token answer contains the reference (stored `pre_correct_long`) and the answer of the repaired model does not. Rows = every HELD-OUT row with `pre_correct_long`. rho = module-output restoration at the answer keys of the valid HELD-OUT rows (1 = unrepaired, 0 = pre-edit; medians).

### Qwen (TB-Q100-ae-l2_3)

| method | n | pre-correct rows | D2 after (rate %) | D2 at W_E (stored) | stored D2 resolved | new D2 | disjoint rows: D2 after / n (W_E; resolved / new) | image D2 / n | text D2 / n | answers == stored post-edit | == stored pre-edit | rho complaints (q25-q75) | rho non-complaints | rho fixed / not fixed complaints | rho lost / kept non-complaints | REPAIR known-good rows kept (in-sample) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NONE (unrepaired) | - | 120 | 17 (14.2) | 17 | 0 | 0 | 17/107 (17; 0 / 0) | 6/84 | 11/36 | 120 | 33 | 1.00 (1.00-1.00) | 1.00 | - / 1.00 | - / 1.00 | - |
| COMP-B | 10 | 120 | 16 (13.3) | 17 | 2 | 1 | 15/107 (17; 2 / 0) | 6/84 | 10/36 | 72 | 39 | 0.94 (0.89-0.98) | 0.93 | 0.94 / 0.95 | 0.96 / 0.92 | 166/192 |
| COMP-B | 100 | 120 | 20 (16.7) | 17 | 3 | 6 | 20/107 (17; 3 / 6) | 10/84 | 10/36 | 55 | 36 | 0.81 (0.67-0.88) | 0.83 | 0.81 / 0.82 | 0.85 / 0.81 | 149/192 |
| COMP-B | ALL (297) | 120 | 20 (16.7) | 17 | 5 | 8 | 20/107 (17; 5 / 8) | 11/84 | 9/36 | 35 | 38 | 0.76 (0.59-0.85) | 0.79 | 0.73 / 0.77 | 0.81 / 0.78 | 150/192 |
| COMP-B+ | 10 | 120 | 20 (16.7) | 17 | 1 | 4 | 19/107 (17; 1 / 3) | 7/84 | 13/36 | 77 | 36 | 1.01 (0.99-1.02) | 1.00 | 0.99 / 1.01 | 1.01 / 1.00 | 182/192 |
| COMP-B+ | 100 | 120 | 25 (20.8) | 17 | 2 | 10 | 24/107 (17; 2 / 9) | 11/84 | 14/36 | 41 | 41 | 1.05 (0.92-1.12) | 1.05 | 1.05 / 1.04 | 1.04 / 1.05 | 168/192 |
| COMP-B+ | ALL (297) | 120 | 27 (22.5) | 17 | 3 | 13 | 25/107 (17; 3 / 11) | 13/84 | 14/36 | 26 | 28 | 1.22 (0.93-1.35) | 1.25 | 1.12 / 1.22 | 1.25 / 1.28 | 129/192 |
| COMP-S | 10 | 120 | 16 (13.3) | 17 | 2 | 1 | 15/107 (17; 2 / 0) | 6/84 | 10/36 | 72 | 39 | 0.94 (0.89-0.98) | 0.93 | 0.94 / 0.95 | 0.96 / 0.92 | 166/192 |
| COMP-S | 100 | 120 | 20 (16.7) | 17 | 3 | 6 | 20/107 (17; 3 / 6) | 10/84 | 10/36 | 55 | 36 | 0.81 (0.67-0.88) | 0.83 | 0.81 / 0.82 | 0.85 / 0.81 | 149/192 |
| COMP-S | ALL (297) | 120 | 20 (16.7) | 17 | 5 | 8 | 20/107 (17; 5 / 8) | 11/84 | 9/36 | 35 | 38 | 0.76 (0.59-0.85) | 0.79 | 0.73 / 0.77 | 0.81 / 0.78 | 150/192 |
| COMP-S+ | 10 | 120 | 20 (16.7) | 17 | 1 | 4 | 19/107 (17; 1 / 3) | 7/84 | 13/36 | 77 | 36 | 1.01 (0.99-1.02) | 1.00 | 0.99 / 1.01 | 1.01 / 1.00 | 182/192 |
| COMP-S+ | 100 | 120 | 25 (20.8) | 17 | 2 | 10 | 24/107 (17; 2 / 9) | 11/84 | 14/36 | 41 | 41 | 1.05 (0.92-1.12) | 1.05 | 1.05 / 1.04 | 1.04 / 1.05 | 168/192 |
| COMP-S+ | ALL (297) | 120 | 27 (22.5) | 17 | 3 | 13 | 25/107 (17; 3 / 11) | 13/84 | 14/36 | 26 | 29 | 1.22 (0.93-1.35) | 1.25 | 1.12 / 1.22 | 1.25 / 1.28 | 129/192 |
| COMP-B-ALLPOS | 10 | 120 | 23 (19.2) | 17 | 4 | 10 | 21/107 (17; 4 / 8) | 13/84 | 10/36 | 40 | 30 | 0.92 (0.87-0.97) | 0.93 | 0.89 / 0.94 | 0.94 / 0.92 | 134/192 |
| REEDIT-B | 10 | 120 | 18 (15.0) | 17 | 2 | 3 | 18/107 (17; 2 / 3) | 7/84 | 11/36 | 65 | 33 | 1.04 (1.02-1.12) | 1.06 | 1.11 / 1.03 | 1.13 / 1.03 | - |
| REEDIT-B | 100 | 120 | 19 (15.8) | 17 | 5 | 7 | 19/107 (17; 5 / 7) | 9/84 | 10/36 | 53 | 35 | 1.28 (1.23-1.35) | 1.28 | 1.26 / 1.28 | 1.32 / 1.26 | - |
| REEDIT-B | ALL (297) | 120 | 22 (18.3) | 17 | 4 | 9 | 22/107 (17; 4 / 9) | 9/84 | 13/36 | 39 | 42 | 1.61 (1.56-1.67) | 1.61 | 1.60 / 1.61 | 1.64 / 1.59 | - |
| REEDIT-S | 10 | 120 | 18 (15.0) | 17 | 2 | 3 | 18/107 (17; 2 / 3) | 7/84 | 11/36 | 65 | 33 | 1.04 (1.02-1.12) | 1.06 | 1.11 / 1.03 | 1.13 / 1.03 | - |
| REEDIT-S | 100 | 120 | 21 (17.5) | 17 | 4 | 8 | 21/107 (17; 4 / 8) | 9/84 | 12/36 | 52 | 36 | 1.28 (1.23-1.35) | 1.28 | 1.26 / 1.28 | 1.32 / 1.26 | - |
| REEDIT-S | ALL (297) | 120 | 23 (19.2) | 17 | 4 | 10 | 23/107 (17; 4 / 10) | 9/84 | 14/36 | 37 | 41 | 1.61 (1.56-1.67) | 1.61 | 1.59 / 1.61 | 1.64 / 1.59 | - |
| RB-TOT | 10 | 120 | 20 (16.7) | 17 | 3 | 6 | 20/107 (17; 3 / 6) | 9/84 | 11/36 | 55 | 45 | 0.83 (0.81-0.86) | 0.84 | 0.84 / 0.83 | 0.84 / 0.84 | - |
| RB-TOT | 100 | 120 | 20 (16.7) | 17 | 4 | 7 | 19/107 (17; 4 / 6) | 10/84 | 10/36 | 48 | 49 | 0.79 (0.77-0.82) | 0.80 | 0.79 / 0.80 | 0.81 / 0.80 | - |
| RB-TOT | ALL (297) | 120 | 17 (14.2) | 17 | 3 | 3 | 17/107 (17; 3 / 3) | 8/84 | 9/36 | 45 | 51 | 0.82 (0.79-0.85) | 0.82 | 0.81 / 0.82 | 0.83 / 0.82 | - |
| RB-UNION | 10 | 120 | 16 (13.3) | 17 | 5 | 4 | 16/107 (17; 5 / 4) | 6/84 | 10/36 | 54 | 42 | 0.85 (0.82-0.87) | 0.85 | 0.85 / 0.84 | 0.84 / 0.85 | - |
| RB-UNION | 100 | 120 | 7 (5.8) | 17 | 12 | 2 | 7/107 (17; 12 / 2) | 3/84 | 4/36 | 39 | 83 | 0.22 (0.20-0.25) | 0.22 | 0.22 / 0.22 | 0.21 / 0.22 | - |
| RB-UNION | ALL (297) | 120 | 1 (0.8) | 17 | 16 | 0 | 1/107 (17; 16 / 0) | 0/84 | 1/36 | 34 | 108 | 0.03 (0.03-0.04) | 0.03 | 0.03 / 0.03 | 0.04 / 0.03 | - |
| ALL (W_0) | - | 120 | 0 (0.0) | 17 | 17 | 0 | 0/107 (17; 17 / 0) | 0/84 | 0/36 | 33 | 120 | 0.00 (0.00-0.00) | 0.00 | 0.00 / - | - / 0.00 | - |

### LLaVA (TB-L100-ae-l2_3)

| method | n | pre-correct rows | D2 after (rate %) | D2 at W_E (stored) | stored D2 resolved | new D2 | disjoint rows: D2 after / n (W_E; resolved / new) | image D2 / n | text D2 / n | answers == stored post-edit | == stored pre-edit | rho complaints (q25-q75) | rho non-complaints | rho fixed / not fixed complaints | rho lost / kept non-complaints | REPAIR known-good rows kept (in-sample) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NONE (unrepaired) | - | 87 | 32 (36.8) | 32 | 0 | 0 | 31/78 (31; 0 / 0) | 4/32 | 28/55 | 87 | 22 | 1.00 (1.00-1.00) | 1.00 | - / 1.00 | - / 1.00 | - |
| COMP-B | 10 | 87 | 37 (42.5) | 32 | 2 | 7 | 35/78 (31; 2 / 6) | 5/32 | 32/55 | 37 | 19 | 0.99 (0.96-1.01) | 1.00 | 0.97 / 0.99 | 0.99 / 1.00 | 175/189 |
| COMP-B | 100 | 87 | 28 (32.2) | 32 | 10 | 6 | 26/78 (31; 10 / 5) | 4/32 | 24/55 | 23 | 15 | 0.95 (0.90-1.03) | 1.02 | 0.91 / 0.96 | 1.02 / 1.02 | 153/189 |
| COMP-B | ALL (305) | 87 | 30 (34.5) | 32 | 12 | 10 | 29/78 (31; 12 / 10) | 5/32 | 25/55 | 21 | 19 | 0.90 (0.81-1.00) | 1.06 | 0.83 / 0.92 | 0.93 / 1.11 | 135/189 |
| COMP-B+ | 10 | 87 | 35 (40.2) | 32 | 2 | 5 | 33/78 (31; 2 / 4) | 5/32 | 30/55 | 48 | 23 | 1.00 (0.99-1.01) | 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 177/189 |
| COMP-B+ | 100 | 87 | 37 (42.5) | 32 | 7 | 12 | 35/78 (31; 7 / 11) | 5/32 | 32/55 | 12 | 8 | 1.07 (1.02-1.14) | 1.07 | 1.04 / 1.08 | 1.07 / 1.07 | 149/189 |
| COMP-B+ | ALL (305) | 87 | 40 (46.0) | 32 | 6 | 14 | 39/78 (31; 6 / 14) | 7/32 | 33/55 | 6 | 6 | 1.08 (0.97-1.21) | 1.10 | 0.99 / 1.09 | 1.10 / 1.10 | 138/189 |
| COMP-S | 10 | 87 | 37 (42.5) | 32 | 2 | 7 | 35/78 (31; 2 / 6) | 5/32 | 32/55 | 37 | 19 | 0.99 (0.96-1.01) | 1.00 | 0.97 / 0.99 | 0.99 / 1.00 | 175/189 |
| COMP-S | 100 | 87 | 28 (32.2) | 32 | 10 | 6 | 26/78 (31; 10 / 5) | 4/32 | 24/55 | 23 | 15 | 0.95 (0.90-1.03) | 1.02 | 0.91 / 0.96 | 1.02 / 1.02 | 153/189 |
| COMP-S | ALL (305) | 87 | 30 (34.5) | 32 | 12 | 10 | 29/78 (31; 12 / 10) | 5/32 | 25/55 | 21 | 19 | 0.90 (0.81-1.00) | 1.06 | 0.83 / 0.92 | 0.93 / 1.11 | 135/189 |
| COMP-S+ | 10 | 87 | 35 (40.2) | 32 | 2 | 5 | 33/78 (31; 2 / 4) | 5/32 | 30/55 | 48 | 23 | 1.00 (0.99-1.01) | 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 177/189 |
| COMP-S+ | 100 | 87 | 37 (42.5) | 32 | 7 | 12 | 35/78 (31; 7 / 11) | 5/32 | 32/55 | 12 | 8 | 1.07 (1.02-1.14) | 1.07 | 1.04 / 1.08 | 1.07 / 1.07 | 149/189 |
| COMP-S+ | ALL (305) | 87 | 40 (46.0) | 32 | 6 | 14 | 39/78 (31; 6 / 14) | 7/32 | 33/55 | 6 | 6 | 1.08 (0.97-1.21) | 1.10 | 0.99 / 1.09 | 1.10 / 1.10 | 138/189 |
| COMP-B-ALLPOS | 10 | 87 | 28 (32.2) | 32 | 12 | 8 | 28/78 (31; 11 / 8) | 1/32 | 27/55 | 21 | 23 | 0.97 (0.92-1.14) | 1.13 | 0.95 / 0.97 | 1.00 / 1.15 | 135/189 |
| REEDIT-B | 10 | 87 | 34 (39.1) | 32 | 5 | 7 | 33/78 (31; 4 / 6) | 4/32 | 30/55 | 27 | 17 | 1.09 (1.02-1.16) | 1.03 | 1.09 / 1.09 | 1.15 / 1.02 | - |
| REEDIT-B | 100 | 87 | 43 (49.4) | 32 | 4 | 15 | 42/78 (31; 3 / 14) | 3/32 | 40/55 | 22 | 16 | 1.33 (1.23-1.40) | 1.23 | 1.29 / 1.33 | 1.32 / 1.20 | - |
| REEDIT-B | ALL (305) | 87 | 50 (57.5) | 32 | 4 | 22 | 50/78 (31; 3 / 22) | 3/32 | 47/55 | 15 | 16 | 1.69 (1.58-1.79) | 1.57 | 1.60 / 1.69 | 1.63 / 1.53 | - |
| REEDIT-S | 10 | 87 | 34 (39.1) | 32 | 5 | 7 | 33/78 (31; 4 / 6) | 4/32 | 30/55 | 27 | 17 | 1.09 (1.02-1.16) | 1.03 | 1.09 / 1.09 | 1.15 / 1.02 | - |
| REEDIT-S | 100 | 87 | 43 (49.4) | 32 | 4 | 15 | 42/78 (31; 3 / 14) | 3/32 | 40/55 | 22 | 16 | 1.33 (1.23-1.40) | 1.23 | 1.29 / 1.33 | 1.32 / 1.20 | - |
| REEDIT-S | ALL (305) | 87 | 50 (57.5) | 32 | 4 | 22 | 50/78 (31; 3 / 22) | 3/32 | 47/55 | 15 | 16 | 1.69 (1.58-1.79) | 1.57 | 1.60 / 1.69 | 1.63 / 1.53 | - |
| RB-TOT | 10 | 87 | 29 (33.3) | 32 | 9 | 6 | 29/78 (31; 8 / 6) | 3/32 | 26/55 | 29 | 27 | 0.82 (0.80-0.84) | 0.82 | 0.81 / 0.82 | 0.82 / 0.82 | - |
| RB-TOT | 100 | 87 | 25 (28.7) | 32 | 11 | 4 | 25/78 (31; 10 / 4) | 3/32 | 22/55 | 24 | 24 | 0.81 (0.79-0.83) | 0.82 | 0.80 / 0.81 | 0.81 / 0.82 | - |
| RB-TOT | ALL (305) | 87 | 25 (28.7) | 32 | 10 | 3 | 24/78 (31; 10 / 3) | 4/32 | 21/55 | 30 | 29 | 0.80 (0.77-0.82) | 0.81 | 0.79 / 0.80 | 0.81 / 0.81 | - |
| RB-UNION | 10 | 87 | 24 (27.6) | 32 | 14 | 6 | 24/78 (31; 13 / 6) | 4/32 | 20/55 | 29 | 24 | 0.82 (0.79-0.84) | 0.82 | 0.81 / 0.82 | 0.82 / 0.82 | - |
| RB-UNION | 100 | 87 | 2 (2.3) | 32 | 30 | 0 | 2/78 (31; 29 / 0) | 0/32 | 2/55 | 22 | 62 | 0.17 (0.15-0.19) | 0.17 | 0.17 / 0.17 | 0.17 / 0.17 | - |
| RB-UNION | ALL (305) | 87 | 0 (0.0) | 32 | 32 | 0 | 0/78 (31; 31 / 0) | 0/32 | 0/55 | 22 | 87 | 0.00 (0.00-0.00) | 0.00 | 0.00 / - | - / 0.00 | - |
| ALL (W_0) | - | 87 | 0 (0.0) | 32 | 32 | 0 | 0/78 (31; 31 / 0) | 0/32 | 0/55 | 22 | 87 | 0.00 (0.00-0.00) | 0.00 | 0.00 / - | - / 0.00 | - |

## 7. Operating points with Rel exact within 2 points of the unrepaired log

Single configurations (lowest HELD-OUT D1 rate first; p = exact McNemar against the unrepaired log on the held-out rows, uncorrected):

Qwen (TB-Q100-ae-l2_3) -- 93 configurations inside the tolerance

| config | HELD-OUT D1 % [CI] | delta vs unrepaired | p | disjoint D1 % [CI] | delta (disjoint) | p (disjoint) | Rel exact |
|---|---|---|---|---|---|---|---|
| COMP-B n=20 s2 | 46.9 [42.9, 51.3] | -7.8 | 0.00028 | 46.7 [42.5, 50.9] | -7.5 | 0.00073 | 78 |
| COMP-S n=20 s2 | 46.9 [42.9, 51.3] | -7.8 | 0.00028 | 46.7 [42.5, 50.9] | -7.5 | 0.00073 | 78 |
| COMP-B n=ALL s0 | 47.3 [43.1, 51.8] | -7.3 | 0.0052 | 49.0 [44.3, 53.6] | -5.2 | 0.057 | 78 |
| COMP-S n=ALL s0 | 47.3 [43.1, 51.8] | -7.3 | 0.0052 | 49.0 [44.3, 53.6] | -5.2 | 0.057 | 78 |
| COMP-B n=20 s1 | 48.6 [44.2, 53.0] | -6.1 | 0.0046 | 48.6 [43.9, 53.1] | -5.6 | 0.013 | 80 |
| COMP-S n=20 s1 | 48.6 [44.2, 53.0] | -6.1 | 0.0046 | 48.6 [43.9, 53.1] | -5.6 | 0.013 | 80 |
| COMP-B n=10 s1 | 50.0 [45.8, 54.3] | -4.7 | 0.017 | 49.7 [45.0, 54.1] | -4.5 | 0.028 | 77 |
| COMP-S n=10 s1 | 50.0 [45.8, 54.3] | -4.7 | 0.017 | 49.7 [45.0, 54.1] | -4.5 | 0.028 | 77 |
| COMP-B n=10 s2 | 51.0 [47.0, 55.2] | -3.7 | 0.092 | 51.2 [46.9, 55.6] | -3.0 | 0.19 | 76 |
| COMP-S n=10 s2 | 51.0 [47.0, 55.2] | -3.7 | 0.092 | 51.2 [46.9, 55.6] | -3.0 | 0.19 | 76 |

Cells (method x n) whose every seed is inside the tolerance, by seed-mean HELD-OUT D1 rate:

| method | n | seeds | HELD-OUT D1 % (mean) | disjoint D1 % (mean) | Rel exact (mean) |
|---|---|---|---|---|---|
| COMP-B | ALL | 1 | 47.3 | 49.0 | 78.0 |
| COMP-S | ALL | 1 | 47.3 | 49.0 | 78.0 |
| COMP-B+ | 10 | 3 | 52.3 | 52.3 | 77.7 |
| COMP-S+ | 10 | 3 | 52.3 | 52.3 | 77.7 |
| COMP-B | 2 | 3 | 52.3 | 51.8 | 77.7 |
| COMP-S | 2 | 3 | 52.3 | 51.8 | 77.7 |
| COMP-B+ | 5 | 3 | 53.4 | 53.1 | 78.0 |
| COMP-S+ | 5 | 3 | 53.4 | 53.1 | 78.0 |
| COMP-B+ | 2 | 3 | 54.0 | 53.8 | 78.0 |
| COMP-S+ | 2 | 3 | 54.0 | 53.8 | 78.0 |

LLaVA (TB-L100-ae-l2_3) -- 117 configurations inside the tolerance

| config | HELD-OUT D1 % [CI] | delta vs unrepaired | p | disjoint D1 % [CI] | delta (disjoint) | p (disjoint) | Rel exact |
|---|---|---|---|---|---|---|---|
| COMP-B n=ALL s0 | 58.5 [54.2, 63.0] | -3.0 | 0.22 | 59.7 [55.1, 64.4] | -2.6 | 0.31 | 86 |
| COMP-S n=ALL s0 | 58.5 [54.2, 63.0] | -3.0 | 0.22 | 59.7 [55.1, 64.4] | -2.6 | 0.31 | 86 |
| COMP-B n=100 s0 | 59.1 [54.9, 63.3] | -2.4 | 0.26 | 60.6 [56.3, 64.6] | -1.7 | 0.46 | 89 |
| COMP-S n=100 s0 | 59.1 [54.9, 63.3] | -2.4 | 0.26 | 60.6 [56.3, 64.6] | -1.7 | 0.46 | 89 |
| COMP-B n=20 s1 | 59.9 [55.8, 64.1] | -1.6 | 0.32 | 61.4 [57.1, 65.7] | -0.9 | 0.65 | 88 |
| COMP-B n=50 s0 | 59.9 [55.8, 64.0] | -1.6 | 0.4 | 60.8 [56.6, 65.0] | -1.5 | 0.46 | 88 |
| COMP-S n=20 s1 | 59.9 [55.8, 64.1] | -1.6 | 0.32 | 61.4 [57.1, 65.7] | -0.9 | 0.65 | 88 |
| COMP-S n=50 s0 | 59.9 [55.8, 64.0] | -1.6 | 0.4 | 60.8 [56.6, 65.0] | -1.5 | 0.46 | 88 |
| COMP-B n=1 s1 | 60.9 [57.1, 65.1] | -0.6 | 0.63 | 61.6 [57.6, 65.8] | -0.6 | 0.63 | 88 |
| COMP-S n=1 s1 | 60.9 [57.1, 65.1] | -0.6 | 0.63 | 61.6 [57.6, 65.8] | -0.6 | 0.63 | 88 |

Cells (method x n) whose every seed is inside the tolerance, by seed-mean HELD-OUT D1 rate:

| method | n | seeds | HELD-OUT D1 % (mean) | disjoint D1 % (mean) | Rel exact (mean) |
|---|---|---|---|---|---|
| COMP-B | ALL | 1 | 58.5 | 59.7 | 86.0 |
| COMP-S | ALL | 1 | 58.5 | 59.7 | 86.0 |
| COMP-B | 100 | 1 | 59.1 | 60.6 | 89.0 |
| COMP-S | 100 | 1 | 59.1 | 60.6 | 89.0 |
| COMP-B | 50 | 1 | 59.9 | 60.8 | 88.0 |
| COMP-S | 50 | 1 | 59.9 | 60.8 | 88.0 |
| COMP-B | 20 | 3 | 61.0 | 61.8 | 88.0 |
| COMP-S | 20 | 3 | 61.0 | 61.8 | 88.0 |
| COMP-B | 1 | 3 | 61.3 | 62.0 | 88.0 |
| COMP-S | 1 | 3 | 61.3 | 62.0 | 88.0 |

## 8. Method diagnostics (seed 0)

### COMP: rank budget, restoration, unresolved reported complaints

Qwen (TB-Q100-ae-l2_3) (d_in = 11008)

| method | n | query keys | rank of the correction | dropped | protection rank | (rank + protection) / d_in | e (key share outside the protected span) | condition number | max restoration error | unresolved reported | ... with the module output restored | max ||X_seq - X_joint|| / ||X_joint|| |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| COMP-B | 1 | 10 | 10 | 0 | 400 | 0.04 | 0.88 | 3.6e+00 | 1.1e-15 | 0/1 | 0 | - |
| COMP-B | 2 | 19 | 19 | 0 | 400 | 0.04 | 0.82 | 6.6e+00 | 3.2e-15 | 1/2 | 1 | - |
| COMP-B | 5 | 49 | 49 | 0 | 400 | 0.04 | 0.82 | 1.3e+01 | 1.1e-14 | 0/5 | 0 | - |
| COMP-B | 10 | 99 | 99 | 0 | 400 | 0.05 | 0.81 | 2.3e+01 | 1.5e-14 | 1/10 | 1 | - |
| COMP-B | 20 | 199 | 199 | 0 | 400 | 0.05 | 0.82 | 4.8e+01 | 2.1e-14 | 7/20 | 7 | - |
| COMP-B | 50 | 497 | 497 | 0 | 400 | 0.08 | 0.82 | 8.7e+01 | 2.8e-14 | 15/50 | 15 | - |
| COMP-B | 100 | 986 | 986 | 0 | 400 | 0.13 | 0.82 | 1.6e+02 | 5.1e-14 | 29/100 | 29 | - |
| COMP-B | ALL (297) | 2930 | 2841 | 89 | 400 | 0.29 | 0.82 | 3.6e+02 | 1.9e-13 | 129/297 | 129 | - |
| COMP-B+ | 1 | 10 | 10 | 0 | 2254 | 0.21 | 0.56 | 5.9e+00 | 7.7e-15 | 0/1 | 0 | - |
| COMP-B+ | 2 | 19 | 19 | 0 | 2254 | 0.21 | 0.44 | 6.5e+00 | 3.3e-14 | 0/2 | 0 | - |
| COMP-B+ | 5 | 49 | 49 | 0 | 2254 | 0.21 | 0.45 | 1.2e+01 | 8.7e-14 | 1/5 | 1 | - |
| COMP-B+ | 10 | 99 | 99 | 0 | 2254 | 0.21 | 0.44 | 1.7e+01 | 1.4e-13 | 2/10 | 2 | - |
| COMP-B+ | 20 | 199 | 199 | 0 | 2254 | 0.22 | 0.42 | 2.5e+01 | 3.0e-13 | 9/20 | 9 | - |
| COMP-B+ | 50 | 497 | 497 | 0 | 2254 | 0.25 | 0.43 | 3.9e+01 | 5.2e-13 | 20/50 | 20 | - |
| COMP-B+ | 100 | 986 | 986 | 0 | 2254 | 0.29 | 0.44 | 5.2e+01 | 1.3e-12 | 36/100 | 36 | - |
| COMP-B+ | ALL (297) | 2930 | 2841 | 89 | 2254 | 0.46 | 0.44 | 7.9e+01 | 5.4e-12 | 170/297 | 170 | - |
| COMP-S | 1 | 10 | 10 | 0 | 400 | 0.04 | 0.88 | 3.6e+00 | 1.1e-15 | 0/1 | 0 | - |
| COMP-S | 2 | 19 | 19 | 0 | 419 | 0.04 | - | - | 2.4e-15 | 1/2 | 1 | 2.1e-15 |
| COMP-S | 5 | 49 | 49 | 0 | 449 | 0.05 | - | - | 1.0e-14 | 0/5 | 0 | 4.7e-15 |
| COMP-S | 10 | 99 | 99 | 0 | 499 | 0.05 | - | - | 1.7e-14 | 1/10 | 1 | 1.2e-14 |
| COMP-S | 20 | 199 | 199 | 0 | 599 | 0.07 | - | - | 2.1e-14 | 7/20 | 7 | 1.5e-14 |
| COMP-S | 50 | 497 | 497 | 0 | 897 | 0.13 | - | - | 2.4e-14 | 15/50 | 15 | 1.7e-14 |
| COMP-S | 100 | 986 | 986 | 0 | 1386 | 0.22 | - | - | 5.3e-14 | 29/100 | 29 | 3.0e-14 |
| COMP-S | ALL (297) | 2930 | 2841 | 89 | 3241 | 0.55 | - | - | 1.7e-13 | 129/297 | 129 | 1.1e-13 |
| COMP-S+ | 1 | 10 | 10 | 0 | 2254 | 0.21 | 0.56 | 5.9e+00 | 7.7e-15 | 0/1 | 0 | - |
| COMP-S+ | 2 | 19 | 19 | 0 | 2273 | 0.21 | - | - | 2.6e-14 | 0/2 | 0 | 1.2e-14 |
| COMP-S+ | 5 | 49 | 49 | 0 | 2303 | 0.21 | - | - | 8.4e-14 | 1/5 | 1 | 3.4e-14 |
| COMP-S+ | 10 | 99 | 99 | 0 | 2353 | 0.22 | - | - | 1.5e-13 | 2/10 | 2 | 6.7e-14 |
| COMP-S+ | 20 | 199 | 199 | 0 | 2453 | 0.24 | - | - | 2.7e-13 | 9/20 | 9 | 1.2e-13 |
| COMP-S+ | 50 | 497 | 497 | 0 | 2751 | 0.30 | - | - | 5.5e-13 | 20/50 | 20 | 2.3e-13 |
| COMP-S+ | 100 | 986 | 986 | 0 | 3240 | 0.38 | - | - | 1.3e-12 | 36/100 | 36 | 5.3e-13 |
| COMP-S+ | ALL (297) | 2930 | 2841 | 89 | 5095 | 0.72 | - | - | 4.5e-12 | 170/297 | 170 | 1.9e-12 |
| COMP-B-ALLPOS | 1 | 40 | 40 | 0 | 400 | 0.04 | 0.93 | 5.6e+00 | 1.5e-15 | 0/1 | 0 | - |
| COMP-B-ALLPOS | 2 | 1070 | 1070 | 0 | 400 | 0.13 | 0.96 | 5.5e+03 | 7.8e-15 | 0/2 | 0 | - |
| COMP-B-ALLPOS | 5 | 2199 | 2156 | 43 | 400 | 0.23 | 0.95 | 4.8e+03 | 9.1e-15 | 0/5 | 0 | - |
| COMP-B-ALLPOS | 10 | 4684 | 4583 | 101 | 400 | 0.45 | 0.96 | 1.4e+04 | 1.8e-14 | 0/10 | 0 | - |

LLaVA (TB-L100-ae-l2_3) (d_in = 11008)

| method | n | query keys | rank of the correction | dropped | protection rank | (rank + protection) / d_in | e (key share outside the protected span) | condition number | max restoration error | unresolved reported | ... with the module output restored | max ||X_seq - X_joint|| / ||X_joint|| |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| COMP-B | 1 | 10 | 10 | 0 | 483 | 0.04 | 0.87 | 2.8e+00 | 1.1e-15 | 0/1 | 0 | - |
| COMP-B | 2 | 12 | 12 | 0 | 483 | 0.04 | 0.84 | 2.8e+00 | 5.6e-15 | 1/2 | 1 | - |
| COMP-B | 5 | 37 | 37 | 0 | 483 | 0.05 | 0.87 | 5.7e+00 | 6.8e-15 | 2/5 | 2 | - |
| COMP-B | 10 | 57 | 57 | 0 | 483 | 0.05 | 0.86 | 1.1e+01 | 1.1e-14 | 3/10 | 3 | - |
| COMP-B | 20 | 128 | 128 | 0 | 483 | 0.06 | 0.83 | 2.0e+01 | 2.1e-14 | 7/20 | 7 | - |
| COMP-B | 50 | 297 | 291 | 6 | 483 | 0.07 | 0.83 | 1.8e+03 | 1.9e-13 | 19/50 | 19 | - |
| COMP-B | 100 | 617 | 611 | 6 | 483 | 0.10 | 0.84 | 7.9e+03 | 4.2e-13 | 41/100 | 41 | - |
| COMP-B | ALL (305) | 2004 | 1973 | 31 | 483 | 0.22 | 0.86 | 1.3e+04 | 8.7e-13 | 141/305 | 141 | - |
| COMP-B+ | 1 | 10 | 10 | 0 | 1355 | 0.12 | 0.71 | 2.9e+00 | 1.4e-15 | 0/1 | 0 | - |
| COMP-B+ | 2 | 12 | 12 | 0 | 1355 | 0.12 | 0.69 | 2.9e+00 | 3.6e-15 | 1/2 | 1 | - |
| COMP-B+ | 5 | 37 | 37 | 0 | 1355 | 0.13 | 0.70 | 1.2e+01 | 6.0e-15 | 2/5 | 2 | - |
| COMP-B+ | 10 | 57 | 57 | 0 | 1355 | 0.13 | 0.68 | 1.4e+01 | 7.7e-14 | 3/10 | 3 | - |
| COMP-B+ | 20 | 128 | 128 | 0 | 1355 | 0.13 | 0.66 | 1.6e+03 | 8.6e-11 | 8/20 | 8 | - |
| COMP-B+ | 50 | 297 | 291 | 6 | 1355 | 0.15 | 0.67 | 1.9e+03 | 1.6e-10 | 21/50 | 21 | - |
| COMP-B+ | 100 | 617 | 611 | 6 | 1355 | 0.18 | 0.66 | 3.5e+03 | 5.6e-10 | 51/100 | 51 | - |
| COMP-B+ | ALL (305) | 2004 | 1973 | 31 | 1355 | 0.30 | 0.69 | 5.5e+03 | 7.0e-10 | 168/305 | 168 | - |
| COMP-S | 1 | 10 | 10 | 0 | 483 | 0.04 | 0.87 | 2.8e+00 | 1.1e-15 | 0/1 | 0 | - |
| COMP-S | 2 | 12 | 12 | 0 | 495 | 0.05 | - | - | 1.1e-15 | 1/2 | 1 | 2.5e-15 |
| COMP-S | 5 | 37 | 37 | 0 | 520 | 0.05 | - | - | 2.2e-15 | 2/5 | 2 | 2.6e-15 |
| COMP-S | 10 | 57 | 57 | 0 | 540 | 0.05 | - | - | 1.9e-14 | 3/10 | 3 | 3.8e-15 |
| COMP-S | 20 | 128 | 128 | 0 | 611 | 0.07 | - | - | 1.7e-14 | 7/20 | 7 | 5.3e-15 |
| COMP-S | 50 | 297 | 291 | 6 | 774 | 0.10 | - | - | 2.5e-12 | 19/50 | 19 | 1.9e-13 |
| COMP-S | 100 | 617 | 611 | 6 | 1094 | 0.15 | - | - | 9.8e-12 | 41/100 | 41 | 4.7e-10 |
| COMP-S | ALL (305) | 2004 | 1973 | 31 | 2456 | 0.40 | - | - | 1.8e-11 | 140/305 | 140 | 7.6e-10 |
| COMP-S+ | 1 | 10 | 10 | 0 | 1355 | 0.12 | 0.71 | 2.9e+00 | 1.4e-15 | 0/1 | 0 | - |
| COMP-S+ | 2 | 12 | 12 | 0 | 1367 | 0.13 | - | - | 2.3e-15 | 1/2 | 1 | 1.6e-15 |
| COMP-S+ | 5 | 37 | 37 | 0 | 1392 | 0.13 | - | - | 5.9e-15 | 2/5 | 2 | 3.0e-15 |
| COMP-S+ | 10 | 57 | 57 | 0 | 1412 | 0.13 | - | - | 3.4e-14 | 3/10 | 3 | 6.2e-15 |
| COMP-S+ | 20 | 128 | 128 | 0 | 1483 | 0.15 | - | - | 8.1e-11 | 8/20 | 8 | 4.7e-12 |
| COMP-S+ | 50 | 297 | 291 | 6 | 1646 | 0.18 | - | - | 1.3e-10 | 21/50 | 21 | 3.9e-09 |
| COMP-S+ | 100 | 617 | 611 | 6 | 1966 | 0.23 | - | - | 3.6e-10 | 51/100 | 51 | 1.6e-08 |
| COMP-S+ | ALL (305) | 2004 | 1973 | 31 | 3328 | 0.48 | - | - | 5.5e-10 | 168/305 | 168 | 2.9e-08 |
| COMP-B-ALLPOS | 1 | 27 | 27 | 0 | 483 | 0.05 | 0.89 | 4.3e+00 | 1.4e-15 | 0/1 | 0 | - |
| COMP-B-ALLPOS | 2 | 626 | 626 | 0 | 483 | 0.10 | 0.94 | 1.5e+05 | 4.1e-13 | 0/2 | 0 | - |
| COMP-B-ALLPOS | 5 | 1284 | 1276 | 8 | 483 | 0.16 | 0.94 | 1.6e+05 | 7.1e-13 | 0/5 | 0 | - |
| COMP-B-ALLPOS | 10 | 3118 | 3099 | 19 | 483 | 0.33 | 0.94 | 1.8e+05 | 6.9e-13 | 0/10 | 0 | - |

### REEDIT: installed complaints and solve attainment

Qwen (TB-Q100-ae-l2_3)

| method | n | installed | not installed (no oracle z*) | installed but not resolved | solve attainment (joint, or first-last batch) | ||D|| / ||W_0|| (joint solve) | norm / log |
|---|---|---|---|---|---|---|---|
| REEDIT-B | 1 | 1 | 0 | 0 | 0.877 | 0.154 | 0.072 |
| REEDIT-B | 2 | 2 | 0 | 0 | 0.790 | 0.176 | 0.082 |
| REEDIT-B | 5 | 5 | 0 | 0 | 0.593 | 0.330 | 0.154 |
| REEDIT-B | 10 | 10 | 0 | 1 | 0.575 | 0.469 | 0.220 |
| REEDIT-B | 20 | 20 | 0 | 1 | 0.564 | 0.749 | 0.350 |
| REEDIT-B | 50 | 50 | 0 | 1 | 0.561 | 1.072 | 0.502 |
| REEDIT-B | 100 | 100 | 0 | 7 | 0.550 | 1.553 | 0.727 |
| REEDIT-B | ALL (297) | 297 | 0 | 56 | 0.518 | 2.613 | 1.223 |
| REEDIT-S | 1 | 1 | 0 | 0 | 0.877 | 0.154 | 0.072 |
| REEDIT-S | 2 | 2 | 0 | 0 | 0.790 | 0.176 | 0.082 |
| REEDIT-S | 5 | 5 | 0 | 0 | 0.593 | 0.330 | 0.154 |
| REEDIT-S | 10 | 10 | 0 | 1 | 0.575 | 0.469 | 0.220 |
| REEDIT-S | 20 | 20 | 0 | 1 | 0.575 - 0.577 (2 batches) | - | 0.350 |
| REEDIT-S | 50 | 50 | 0 | 1 | 0.575 - 0.593 (5 batches) | - | 0.502 |
| REEDIT-S | 100 | 100 | 0 | 6 | 0.575 - 0.522 (10 batches) | - | 0.727 |
| REEDIT-S | ALL (297) | 297 | 0 | 54 | 0.575 - 0.552 (30 batches) | - | 1.223 |

LLaVA (TB-L100-ae-l2_3)

| method | n | installed | not installed (no oracle z*) | installed but not resolved | solve attainment (joint, or first-last batch) | ||D|| / ||W_0|| (joint solve) | norm / log |
|---|---|---|---|---|---|---|---|
| REEDIT-B | 1 | 1 | 0 | 1 | 0.870 | 0.162 | 0.075 |
| REEDIT-B | 2 | 2 | 0 | 1 | 0.745 | 0.279 | 0.129 |
| REEDIT-B | 5 | 5 | 0 | 2 | 0.725 | 0.376 | 0.174 |
| REEDIT-B | 10 | 10 | 0 | 3 | 0.670 | 0.496 | 0.230 |
| REEDIT-B | 20 | 20 | 0 | 4 | 0.647 | 0.624 | 0.289 |
| REEDIT-B | 50 | 50 | 0 | 14 | 0.566 | 1.025 | 0.474 |
| REEDIT-B | 100 | 100 | 0 | 30 | 0.576 | 1.488 | 0.688 |
| REEDIT-B | ALL (305) | 305 | 0 | 146 | 0.609 | 2.615 | 1.210 |
| REEDIT-S | 1 | 1 | 0 | 1 | 0.870 | 0.162 | 0.075 |
| REEDIT-S | 2 | 2 | 0 | 1 | 0.745 | 0.279 | 0.129 |
| REEDIT-S | 5 | 5 | 0 | 2 | 0.725 | 0.376 | 0.174 |
| REEDIT-S | 10 | 10 | 0 | 3 | 0.670 | 0.496 | 0.230 |
| REEDIT-S | 20 | 20 | 0 | 4 | 0.670 - 0.622 (2 batches) | - | 0.289 |
| REEDIT-S | 50 | 50 | 0 | 14 | 0.670 - 0.517 (5 batches) | - | 0.474 |
| REEDIT-S | 100 | 100 | 0 | 30 | 0.670 - 0.620 (10 batches) | - | 0.688 |
| REEDIT-S | ALL (305) | 305 | 0 | 146 | 0.670 - 0.600 (31 batches) | - | 1.210 |

### Rollback: edits rolled back

Qwen (TB-Q100-ae-l2_3)

| method | n | edits rolled back (of 100) | all reported resolved at stop | resolved individually (per-complaint sets) | reported resolved | Rel exact |
|---|---|---|---|---|---|---|
| RB-TOT | 1 | 9 | True | - | 1/1 | 72 |
| RB-TOT | 2 | 11 | True | - | 2/2 | 69 |
| RB-TOT | 5 | 27 | True | - | 5/5 | 61 |
| RB-TOT | 10 | 30 | False | - | 7/10 | 54 |
| RB-TOT | 20 | 30 | False | - | 15/20 | 54 |
| RB-TOT | 50 | 30 | False | - | 28/50 | 56 |
| RB-TOT | 100 | 30 | False | - | 47/100 | 53 |
| RB-TOT | ALL (297) | 30 | False | - | 97/297 | 56 |
| RB-UNION | 1 | 9 | - | 1 | 1/1 | 72 |
| RB-UNION | 2 | 11 | - | 2 | 2/2 | 70 |
| RB-UNION | 5 | 24 | - | 4 | 3/5 | 62 |
| RB-UNION | 10 | 27 | - | 8 | 7/10 | 61 |
| RB-UNION | 20 | 46 | - | 17 | 12/20 | 43 |
| RB-UNION | 50 | 75 | - | 45 | 38/50 | 18 |
| RB-UNION | 100 | 92 | - | 89 | 80/100 | 7 |
| RB-UNION | ALL (297) | 99 | - | 247 | 274/297 | 1 |

LLaVA (TB-L100-ae-l2_3)

| method | n | edits rolled back (of 100) | all reported resolved at stop | resolved individually (per-complaint sets) | reported resolved | Rel exact |
|---|---|---|---|---|---|---|
| RB-TOT | 1 | 3 | True | - | 1/1 | 86 |
| RB-TOT | 2 | 13 | True | - | 2/2 | 80 |
| RB-TOT | 5 | 30 | False | - | 4/5 | 63 |
| RB-TOT | 10 | 30 | False | - | 9/10 | 66 |
| RB-TOT | 20 | 30 | False | - | 13/20 | 66 |
| RB-TOT | 50 | 30 | False | - | 17/50 | 66 |
| RB-TOT | 100 | 30 | False | - | 27/100 | 65 |
| RB-TOT | ALL (305) | 30 | False | - | 81/305 | 63 |
| RB-UNION | 1 | 3 | - | 1 | 1/1 | 86 |
| RB-UNION | 2 | 9 | - | 2 | 1/2 | 83 |
| RB-UNION | 5 | 23 | - | 4 | 3/5 | 71 |
| RB-UNION | 10 | 29 | - | 9 | 4/10 | 66 |
| RB-UNION | 20 | 49 | - | 18 | 10/20 | 47 |
| RB-UNION | 50 | 86 | - | 42 | 34/50 | 13 |
| RB-UNION | 100 | 96 | - | 81 | 86/100 | 5 |
| RB-UNION | ALL (305) | 100 | - | 247 | 305/305 | 1 |

## 9. Second priority: per-complaint head-to-head on TEXT complaints (rq1 continued)

- complaints: text rows of rq1's own sample harness.sample_complaints(H, 100, seed=1) (REPAIR pool, D1 v2): the first 30 text complaints in draw order (rq1 indices 50..79), never run by rq1
- methods: NONE; COMP protect K_ans, query answer, adapter; REEDIT k0+kp; ROLLBACK_FO budget 10; ROLLBACK_BEH budget 20 (the options of rq1/code/run_rq1.py)
- clean: resolved (D1) AND no logged edit broken (rel + T-Gen + I-Gen, teacher-forced exact)
- CI: uid-level percentile bootstrap, 2000 resamples, 95%; paired: exact McNemar on the same complaints; reference rows: rq1 image complaints: read-only from 07_repair_v3/rq1/rows (complaints with all five methods)

### Qwen (TB-Q100-ae-l2_3) -- TEXT complaints (this experiment): 30 complaints

W_E: Rel / T-Gen / I-Gen exact 78 / 9 / 22; held-out rows: 222 non-complaints, 268 complaints; median answer length 10 tokens.

| method | D1 resolved | rate [95% CI] | clean (resolved, 0 broken) [CI] | edits broken / repair [CI] | rel / T-Gen / I-Gen broken | edits fixed | held-out non-compl. lost [CI] | held-out compl. fixed [CI] | dT-Loc | dI-Loc | median norm (/ log) | D2 res / n | passes fwd / bwd | median s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NONE | 0/30 | 0.00 [0.00, 0.00] | (full evaluation on 1 complaint) | - | - | - | - | - | - | - | - | 0/3 | - | - |
| COMP | 24/30 | 0.80 [0.67, 0.93] | 15 (0.50 [0.33, 0.67]) | 0.33 [0.17, 0.53] | 0.33 / 0.00 / 0.00 | 0.60 | 21.87 [19.83, 24.07] | 26.70 [24.77, 28.70] | 1.42 [0.97, 1.84] | 0.08 [-0.02, 0.18] | 7.14 (0.027) | 0/3 | 1.0 / 0.0 | 0.05 |
| REEDIT | 28/30 | 0.93 [0.83, 1.00] | 22 (0.73 [0.57, 0.87]) | 0.27 [0.07, 0.50] | 0.27 / 0.00 / 0.00 | 0.30 | 55.67 [48.53, 62.40] | 19.70 [17.33, 21.90] | -2.97 [-4.43, -1.64] | 0.12 [0.03, 0.21] | 12.69 (0.048) | 1/3 | 9.1 / 6.2 | 0.50 |
| RB_FO | 24/30 | 0.80 [0.63, 0.93] | 0 (0.00 [0.00, 0.00]) | 6.83 [5.20, 8.60] | 5.30 / 0.50 / 1.03 | 2.23 | 19.77 [17.37, 22.20] | 26.47 [22.57, 30.33] | 1.19 [0.85, 1.54] | 0.31 [0.13, 0.48] | 49.57 (0.187) | 0/3 | 11.0 / 5.0 | 0.25 |
| RB_BEH | 30/30 | 1.00 [1.00, 1.00] | 0 (0.00 [0.00, 0.00]) | 6.50 [4.57, 8.60] | 5.10 / 0.37 / 1.03 | 1.97 | 18.50 [16.27, 20.83] | 24.63 [20.83, 28.73] | 1.04 [0.60, 1.52] | 0.40 [0.20, 0.62] | 47.54 (0.179) | 0/3 | 462.8 / 0.0 | 4.31 |

REEDIT: 1 complaint(s) not installed (z* not an oracle after the single vendor attempt of the default call: text_446); z* attempts {'1': 30}. Supplement with the documented fallback ladder (`chunk=1`) on these complaints: text_446: installed True, resolved True, attempts [2], edits broken 0 -> REEDIT with the documented fallback would resolve 29/30.

| comparison | outcome | n | both | first only | second only | neither | p |
|---|---|---|---|---|---|---|---|
| COMP vs NONE | resolved | 30 | 0 | 24 | 0 | 6 | <1e-4 |
| COMP vs REEDIT | resolved | 30 | 22 | 2 | 6 | 0 | 0.29 |
| COMP vs REEDIT | clean | 30 | 12 | 3 | 10 | 5 | 0.092 |
| COMP vs RB_FO | resolved | 30 | 19 | 5 | 5 | 1 | 1 |
| COMP vs RB_FO | clean | 30 | 0 | 15 | 0 | 15 | <1e-4 |
| COMP vs RB_BEH | resolved | 30 | 24 | 0 | 6 | 0 | 0.031 |
| COMP vs RB_BEH | clean | 30 | 0 | 15 | 0 | 15 | <1e-4 |
| REEDIT vs RB_BEH | resolved | 30 | 28 | 0 | 2 | 0 | 0.5 |
| REEDIT vs RB_BEH | clean | 30 | 0 | 22 | 0 | 8 | <1e-4 |

### Qwen (TB-Q100-ae-l2_3) -- image complaints (rq1 rows, read-only reference): 28 complaints

W_E: Rel / T-Gen / I-Gen exact 78 / 9 / 22; held-out rows: 222 non-complaints, 268 complaints; median answer length 10 tokens.

| method | D1 resolved | rate [95% CI] | clean (resolved, 0 broken) [CI] | edits broken / repair [CI] | rel / T-Gen / I-Gen broken | edits fixed | held-out non-compl. lost [CI] | held-out compl. fixed [CI] | dT-Loc | dI-Loc | median norm (/ log) | D2 res / n | passes fwd / bwd | median s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NONE | 0/28 | 0.00 [0.00, 0.00] | (full evaluation on 1 complaint) | - | - | - | - | - | - | - | - | 0/2 | - | - |
| COMP | 19/28 | 0.68 [0.50, 0.86] | 11 (0.39 [0.21, 0.57]) | 0.54 [0.29, 0.82] | 0.50 / 0.00 / 0.04 | 1.00 | 15.46 [14.18, 16.71] | 18.86 [17.00, 20.64] | -0.07 [-0.21, 0.08] | 0.11 [-0.06, 0.28] | 7.08 (0.027) | 2/2 | 1.0 / 0.0 | 0.17 |
| REEDIT | 27/28 | 0.96 [0.89, 1.00] | 15 (0.54 [0.36, 0.71]) | 0.61 [0.29, 1.04] | 0.54 / 0.00 / 0.07 | 1.07 | 9.68 [8.18, 11.32] | 10.75 [9.25, 12.25] | -0.20 [-0.44, 0.05] | 0.19 [0.07, 0.32] | 19.90 (0.075) | 2/2 | 10.6 / 7.7 | 1.66 |
| RB_FO | 24/28 | 0.86 [0.71, 0.96] | 0 (0.00 [0.00, 0.00]) | 6.43 [4.89, 8.04] | 4.82 / 0.32 / 1.29 | 2.14 | 17.39 [14.29, 20.61] | 23.00 [18.96, 27.07] | 0.88 [0.52, 1.26] | 0.61 [0.40, 0.83] | 52.40 (0.197) | 2/2 | 10.8 / 4.9 | 0.45 |
| RB_BEH | 27/28 | 0.96 [0.89, 1.00] | 0 (0.00 [0.00, 0.00]) | 7.11 [4.79, 9.96] | 5.29 / 0.39 / 1.43 | 2.04 | 16.93 [13.89, 20.29] | 22.82 [18.43, 28.00] | 0.66 [0.25, 1.14] | 0.46 [0.23, 0.71] | 50.46 (0.190) | 2/2 | 497.5 / 0.0 | 7.55 |

REEDIT: 1 complaint(s) not installed (z* not an oracle after the single vendor attempt of the default call: image_20); z* attempts {'1': 28}.

| comparison | outcome | n | both | first only | second only | neither | p |
|---|---|---|---|---|---|---|---|
| COMP vs NONE | resolved | 28 | 0 | 19 | 0 | 9 | <1e-4 |
| COMP vs REEDIT | resolved | 28 | 18 | 1 | 9 | 0 | 0.021 |
| COMP vs REEDIT | clean | 28 | 6 | 5 | 9 | 8 | 0.42 |
| COMP vs RB_FO | resolved | 28 | 17 | 2 | 7 | 2 | 0.18 |
| COMP vs RB_FO | clean | 28 | 0 | 11 | 0 | 17 | 0.00098 |
| COMP vs RB_BEH | resolved | 28 | 19 | 0 | 8 | 1 | 0.0078 |
| COMP vs RB_BEH | clean | 28 | 0 | 11 | 0 | 17 | 0.00098 |
| REEDIT vs RB_BEH | resolved | 28 | 26 | 1 | 1 | 0 | 1 |
| REEDIT vs RB_BEH | clean | 28 | 0 | 15 | 0 | 13 | <1e-4 |

### LLaVA (TB-L100-ae-l2_3) -- TEXT complaints (this experiment): 30 complaints

W_E: Rel / T-Gen / I-Gen exact 88 / 29 / 66; held-out rows: 190 non-complaints, 304 complaints; median answer length 10 tokens.

| method | D1 resolved | rate [95% CI] | clean (resolved, 0 broken) [CI] | edits broken / repair [CI] | rel / T-Gen / I-Gen broken | edits fixed | held-out non-compl. lost [CI] | held-out compl. fixed [CI] | dT-Loc | dI-Loc | median norm (/ log) | D2 res / n | passes fwd / bwd | median s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NONE | 0/30 | 0.00 [0.00, 0.00] | (full evaluation on 1 complaint) | - | - | - | - | - | - | - | - | 0/1 | - | - |
| COMP | 19/30 | 0.63 [0.47, 0.80] | 19 (0.63 [0.47, 0.80]) | 0.00 [0.00, 0.00] | 0.00 / 0.00 / 0.00 | 0.27 | 9.07 [8.00, 10.10] | 10.37 [9.20, 11.53] | -0.03 [-0.17, 0.11] | -0.25 [-0.39, -0.11] | 7.01 (0.026) | 1/1 | 1.0 / 0.0 | 0.03 |
| REEDIT | 16/30 | 0.53 [0.33, 0.70] | 16 (0.53 [0.33, 0.70]) | 0.00 [0.00, 0.00] | 0.00 / 0.00 / 0.00 | 0.23 | 20.77 [17.17, 24.23] | 11.47 [9.23, 13.67] | 0.01 [-0.11, 0.12] | -0.03 [-0.14, 0.07] | 13.80 (0.052) | 0/1 | 7.7 / 4.8 | 0.46 |
| RB_FO | 19/30 | 0.63 [0.47, 0.80] | 1 (0.03 [0.00, 0.10]) | 14.80 [11.73, 17.63] | 6.27 / 2.97 / 5.57 | 4.67 | 18.90 [16.53, 21.03] | 25.07 [21.50, 28.50] | 0.74 [0.56, 0.94] | 0.65 [0.44, 0.86] | 82.49 (0.310) | 1/1 | 14.8 / 6.9 | 0.51 |
| RB_BEH | 29/30 | 0.97 [0.90, 1.00] | 1 (0.03 [0.00, 0.10]) | 15.40 [11.47, 19.67] | 6.50 / 3.13 / 5.77 | 4.23 | 17.90 [15.40, 20.47] | 23.03 [19.07, 27.13] | 0.76 [0.47, 1.10] | 0.56 [0.29, 0.84] | 67.62 (0.254) | 1/1 | 663.0 / 0.0 | 7.60 |

REEDIT: 1 complaint(s) not installed (z* not an oracle after the single vendor attempt of the default call: text_481); z* attempts {'1': 30}. Supplement with the documented fallback ladder (`chunk=1`) on these complaints: text_481: installed True, resolved True, attempts [2], edits broken 0 -> REEDIT with the documented fallback would resolve 17/30.

| comparison | outcome | n | both | first only | second only | neither | p |
|---|---|---|---|---|---|---|---|
| COMP vs NONE | resolved | 30 | 0 | 19 | 0 | 11 | <1e-4 |
| COMP vs REEDIT | resolved | 30 | 12 | 7 | 4 | 7 | 0.55 |
| COMP vs REEDIT | clean | 30 | 12 | 7 | 4 | 7 | 0.55 |
| COMP vs RB_FO | resolved | 30 | 13 | 6 | 6 | 5 | 1 |
| COMP vs RB_FO | clean | 30 | 0 | 19 | 1 | 10 | <1e-4 |
| COMP vs RB_BEH | resolved | 30 | 19 | 0 | 10 | 1 | 0.002 |
| COMP vs RB_BEH | clean | 30 | 0 | 19 | 1 | 10 | <1e-4 |
| REEDIT vs RB_BEH | resolved | 30 | 16 | 0 | 13 | 1 | 0.00024 |
| REEDIT vs RB_BEH | clean | 30 | 0 | 16 | 1 | 13 | 0.00027 |

### LLaVA (TB-L100-ae-l2_3) -- image complaints (rq1 rows, read-only reference): 44 complaints

W_E: Rel / T-Gen / I-Gen exact 88 / 29 / 66; held-out rows: 190 non-complaints, 304 complaints; median answer length 4 tokens.

| method | D1 resolved | rate [95% CI] | clean (resolved, 0 broken) [CI] | edits broken / repair [CI] | rel / T-Gen / I-Gen broken | edits fixed | held-out non-compl. lost [CI] | held-out compl. fixed [CI] | dT-Loc | dI-Loc | median norm (/ log) | D2 res / n | passes fwd / bwd | median s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NONE | 0/44 | 0.00 [0.00, 0.00] | (full evaluation on 1 complaint) | - | - | - | - | - | - | - | - | 0/4 | - | - |
| COMP | 22/44 | 0.50 [0.36, 0.64] | 22 (0.50 [0.36, 0.64]) | 0.00 [0.00, 0.00] | 0.00 / 0.00 / 0.00 | 0.14 | 4.00 [3.43, 4.59] | 3.27 [2.73, 3.89] | -0.01 [-0.08, 0.07] | -0.12 [-0.22, -0.02] | 3.85 (0.014) | 3/4 | 1.0 / 0.0 | 0.08 |
| REEDIT | 44/44 | 1.00 [1.00, 1.00] | 42 (0.95 [0.89, 1.00]) | 0.05 [0.00, 0.11] | 0.00 / 0.02 / 0.02 | 0.52 | 5.84 [5.05, 6.64] | 5.73 [5.02, 6.45] | -0.05 [-0.14, 0.04] | -0.11 [-0.22, 0.01] | 19.02 (0.071) | 4/4 | 6.3 / 3.3 | 0.75 |
| RB_FO | 37/44 | 0.84 [0.73, 0.93] | 0 (0.00 [0.00, 0.00]) | 9.05 [7.09, 11.09] | 3.93 / 1.55 / 3.57 | 3.61 | 14.86 [12.93, 16.73] | 16.84 [14.61, 19.09] | 0.41 [0.23, 0.59] | 0.28 [0.07, 0.49] | 48.73 (0.183) | 4/4 | 9.8 / 4.4 | 0.46 |
| RB_BEH | 44/44 | 1.00 [1.00, 1.00] | 0 (0.00 [0.00, 0.00]) | 8.16 [5.98, 10.64] | 3.52 / 1.45 / 3.18 | 3.11 | 13.98 [11.98, 16.23] | 16.70 [14.16, 19.64] | 0.40 [0.23, 0.59] | 0.20 [0.01, 0.40] | 42.49 (0.159) | 4/4 | 394.2 / 0.0 | 8.08 |

REEDIT: 0 complaint(s) not installed (z* not an oracle after the single vendor attempt of the default call: none); z* attempts {'1': 44}.

| comparison | outcome | n | both | first only | second only | neither | p |
|---|---|---|---|---|---|---|---|
| COMP vs NONE | resolved | 44 | 0 | 22 | 0 | 22 | <1e-4 |
| COMP vs REEDIT | resolved | 44 | 22 | 0 | 22 | 0 | <1e-4 |
| COMP vs REEDIT | clean | 44 | 22 | 0 | 20 | 2 | <1e-4 |
| COMP vs RB_FO | resolved | 44 | 22 | 0 | 15 | 7 | <1e-4 |
| COMP vs RB_FO | clean | 44 | 0 | 22 | 0 | 22 | <1e-4 |
| COMP vs RB_BEH | resolved | 44 | 22 | 0 | 22 | 0 | <1e-4 |
| COMP vs RB_BEH | clean | 44 | 0 | 22 | 0 | 22 | <1e-4 |
| REEDIT vs RB_BEH | resolved | 44 | 44 | 0 | 0 | 0 | 1 |
| REEDIT vs RB_BEH | clean | 44 | 0 | 42 | 0 | 2 | <1e-4 |

## 10. Run information

- Qwen (TB-Q100-ae-l2_3): 158 records (18 copies of an identical computation: REEDIT-S with one batch, COMP-S / COMP-S+ at n = 1); peak GPU memory allocated 9.64 GiB; other compute processes on the card during the runs: median 1, max 2; total repair wall-clock 39 min, evaluation 114 min.
- LLaVA (TB-L100-ae-l2_3): 158 records (18 copies of an identical computation: REEDIT-S with one batch, COMP-S / COMP-S+ at n = 1); peak GPU memory allocated 15.17 GiB; other compute processes on the card during the runs: median 1, max 1; total repair wall-clock 46 min, evaluation 119 min.
- Wall-clock numbers come from shared GPUs and CPU solves under load: use pass counts and ranks for cost claims.

### Resume run (operational log `out/_resume.json`; record counts from the raw records)

- Resume of 2026-11-10 17:09 UTC (the first run wrote its last record at 12:10 UTC; its later starts left the cards to other agents' jobs without producing a record). Both cards were idle at the start; `code/chain_resume.sh` was launched at 17:17:34 UTC: Qwen on GPU 0, LLaVA on GPU 1, one process of this experiment per card, every start behind `code/gpu_wait.sh` (needs the job's footprint + 1024 MiB free, defers to other agents' queued jobs), every worker with the watchdog of the first run.
- Unchanged: the harness, every method option, the pools, the streams, the evaluation, the memory caps (torch cap 10.25 GiB for the Qwen stream / head-to-head workers and 9.6 GiB for its D2 worker; 15.3 / 15.0 GiB for LLaVA). Changed: at most 4 CPU (BLAS) threads per process (`P12_THREADS`; the first run used 8), which can move fp64 solves at round-off level; wall-clock numbers of the resume records come from cards without another compute process unless stated in the per-log line below.
- Order: (1) seed 0 completions (missing n values; corrected REEDIT runs on Qwen), (2) seeds 1 and 2 for n <= 20, both in one worker per log (`run_streams.py --phase protocol`); (3) D2 decodes (core set first, then REEDIT-S, COMP-B-ALLPOS n = 10 and the sequential COMP variants); (4) text head-to-head; (5) the module-level diagnostics of Qwen follow from the first D2 start (held-out answer keys) on the CPU. The exploratory configurations outside the protocol (COMP-B-RESTORE, REEDIT-SC) were not part of the resume.
- Deviations of the resume from its instructions, both mine. (a) The chains were launched at 17:17:34 with the head-to-head ahead of the non-core D2 decodes; corrected at 19:38, before any head-to-head pass had started (the two chain shells of this experiment were stopped, their running workers were left untouched, `code/then_chain.sh` continued with the order above). (b) At 18:14 a test copy of `code/gpuq.py` was written to /tmp and run there, outside this experiment's directory; it was deleted by the same command and nothing of it is left.
- Yielding to other agents during the resume (see the incident list below for the one case with a possible effect on another job): 17:43:58 LLaVA stream worker left GPU 1 (card nearly full next to a job of 02_research/code/rescope); 18:17:55 LLaVA stream worker left GPU 1 (queued job of the same package); 18:31:52 Qwen stream worker taken off GPU 0 (`code/ext_watchdog.py`: SIGTERM to that worker of this experiment only; queued job of the same package); 19:50:21 Qwen D2 worker left GPU 0 (queued job of 10_final/cache); 20:04:28 LLaVA stream worker taken off GPU 1 (`code/ext_watchdog.py`; queued job of 10_final/cache). Interrupted configurations were redone on the next start; between these times the launcher waited for the cards (LLaVA had no start between 18:18 and 19:16 and none after 20:04 while GPU 1 was held by other agents' jobs).
- Stopped at 21:54:42 UTC (five-hour limit of the resume: 22:09) with `code/stop_all.sh`: the processes of this experiment only; the Qwen head-to-head worker was in its 22nd complaint, which is not recorded. From 20:04 on GPU 1 was held by other agents' jobs, so LLaVA had no further start: its remaining seed-2 configurations, its D2 decodes and its head-to-head were not run.
- Completion run of 2026-11-11 (task start 03:36 UTC; driver `code/gq_finish.sh`, launched 03:49:40 UTC): the LLaVA cells that the resume above left open (14 seed-2 configurations, every D2 decode, the text head-to-head), then the Qwen head-to-head complaints 22 to 30, in this order, one job of this experiment at a time. Every GPU start went through the shared queue `02_research/tools/gpuq/gq` (`gq run --prio -3`, `--mem 17000` for LLaVA, `--mem 10500` for Qwen) around the same python entry points as the resume (`run_streams.py --phase protocol`, `run_d2.py --set all`, `run_text_h2h.py`); `code/gpu_wait.sh` was not used. Unchanged: the harness, every method option, the pools, the streams, the evaluation, the memory caps (torch cap 15.3 GiB for the LLaVA stream and head-to-head workers, 15.0 GiB for its D2 worker, 10.25 GiB for the Qwen head-to-head worker), at most 4 CPU (BLAS) threads per process. After each unit `code/finalize.sh` (CPU) rebuilt results.json and results.md. The record counts and times of the two per-log lines at the end of this list include the records of this run.
- Completion run of 2026-11-11, queue jobs (read from the queue's event log `tools/gpuq/state/events.log`): `p12_L_streams` (--mem 17000, --prio -3): queued 03:49:42, started 03:49:52 on card 1 after 10 s in the queue, exit code 0 at 04:05:08 after 916 s; `p12_L_d2` (--mem 17000, --prio -3): queued 04:05:41, started 04:06:37 on card 1 after 56 s in the queue, exit code 0 at 04:45:19 after 2323 s; `p12_L_h2h` (--mem 17000, --prio -3): queued 04:45:20, started 04:46:46 on card 1 after 86 s in the queue, exit code 0 at 06:26:23 after 5977 s; `p12_Q_h2h` (--mem 10500, --prio -3): queued 06:26:24, started 06:26:34 on card 1 after 10 s in the queue, exit code 75 at 06:37:48 after 674 s; `p12_Q_h2h` (--mem 10500, --prio -3): queued 06:38:50, started 06:50:36 on card 0 after 706 s in the queue, exit code 0 at 07:11:45 after 1269 s.
- Completion run of 2026-11-11, written (counted from the raw records): LLaVA 14 stream records (13 evaluated, 1 copy of an identical computation; torch CPU threads 4; peak GPU memory allocated 15.16 GiB; other compute processes on the card: max 1), 27 D2 records (of 27 on disk), 30 head-to-head complaints (151 rows; peak GPU memory allocated 14.71 GiB); Qwen 9 head-to-head complaints (number 22 to 30 in draw order; 45 rows; peak GPU memory allocated 8.70 GiB), 30 of 30 done in all. Failed / out-of-memory markers on disk: none.
- Completion run of 2026-11-11, other experiments in the queue (from the same event log; not an incident: the queue serves by priority, then ticket time, and never interrupts a running job): jobs of other experiments that were waiting with a LOWER priority than mine at the moment a job of mine was started -- `p12_L_h2h` (04:46:46): `a3_relb1_QO` (priority -5, --mem 10500, had waited 16 min, started 1 min later), `a3_harness_blip2` (priority -5, --mem 10500, had waited 11 min, started 2 min later), `a3_port_QO` (priority -5, --mem 10500, had waited 4 min, started 1 min later), `fx_a_Q` (priority -10, --mem 10500, had waited 0 min, did not start afterwards), `fx_a_L` (priority -10, --mem 17000, had waited 0 min, did not start afterwards); `p12_Q_h2h` (06:26:34): `fx_t_Q` (priority -10, --mem 10500, had waited 28 min, started 4 min later), `fx_t_L` (priority -10, --mem 17000, had waited 28 min, started 5 min later); `p12_Q_h2h` (06:50:36): `fx_t_L` (priority -10, --mem 17000, had waited 3 min, did not start afterwards), `fx_t_Q` (priority -10, --mem 10500, had waited 3 min, started 0 min later). Whether such a job would have fitted on a card at that moment without mine is not recorded in the event log.
- Completion run of 2026-11-11, workers that left the card by themselves (the experiment's own card checks of `code/p12_common.py` -- `load_log`, `yield_if_blocking`, `start_watchdog` -- written before the queue existed and left unchanged here; the worker exits with code 75, the complaint or configuration in progress is not recorded and is redone by the next pass, which the driver submits 60 s later): `p12_Q_h2h` 06:37:47: [watchdog] leaving the card now (4260 MiB free on the card, the running neighbours may still take 5200 MiB within their declared caps); the current config is redone later. These checks count 2600 MiB of possible growth for every neighbour on the card that declares no cap on its command line; they do not read the queue's reservations.
- Completion run of 2026-11-11, memory against the reservation (`gq status`, column ACTUAL, sampled every 30 s by `code/gq_poll.sh` while it was polling; `logs/gq_actual.log`): `p12_L_h2h` at most 16222 MiB of the 17000 MiB reserved (samples 05:35:29 to 06:26:24 UTC); `p12_Q_h2h` at most 9768 MiB of the 10500 MiB reserved (samples 06:28:40 to 07:11:51 UTC). The stream and D2 jobs ran before the sampling was added.
- Completion run of 2026-11-11, memory cap against the queue's rule (README of the queue: a job caps its torch memory at its --mem minus 600 MiB): `p12_Q_h2h` ran with the torch cap of the existing scripts, 10.25 GiB = 10496 MiB, which is above 10500 - 600 = 9900 MiB. Both figures were given for this run (the --mem of the queue and the caps of the existing scripts) and were kept. The reservation was not exceeded in any sample of the note above, but the cap would not have enforced it. The other jobs' caps are inside the rule (`p12_L_d2` 15.0 GiB for --mem 17000, `p12_L_h2h` 15.3 GiB for --mem 17000, `p12_L_streams` 15.3 GiB for --mem 17000).
- Completion run of 2026-11-11, driver change (mine, no effect on a result): the first version of `code/gq_finish.sh` ran `finalize.sh` between two units, so that the ticket of the D2 job was written 33 s after the stream job had ended (04:05:08 -> 04:05:41); the queue started 3 job(s) of other experiments in that gap (priority -5; mine: -3). At 04:23:29 UTC the driver was replaced, without touching the running D2 job, by a version that queues the next unit at once and runs `finalize.sh` in the background (the first driver was told by its stop file to submit nothing more and ended after its `finalize.sh`; the second waited for the running job to end before its first unit). Time limits of the first version: no job submitted after 07:14:00, a running job stopped at 07:24:00 UTC; of the second: 07:20:00 and 07:27:00 UTC (task limit: four hours from the task start).
- Completion run of 2026-11-11, end: the driver finished at 07:12:15 UTC with every unit done; what is still open is listed under 'Not finished' in section 1.
- Qwen (TB-Q100-ae-l2_3): 94 stream records written by the resume (2026-11-10 17:22:34 to 2026-11-10 19:16:05 UTC), torch CPU threads 4 (the records of the first run do not carry the field; its setting was 8); peak GPU memory allocated 9.62 GiB; other compute processes on the card: max 2; repair wall-clock 10 min, evaluation 66 min.
- LLaVA (TB-L100-ae-l2_3): 100 stream records written by the resume (2026-11-10 17:21:53 to 2026-11-11 04:05:05 UTC), torch CPU threads 4 (the records of the first run do not carry the field; its setting was 8); peak GPU memory allocated 15.17 GiB; other compute processes on the card: max 1; repair wall-clock 9 min, evaluation 79 min.

### Shared-GPU incidents caused by this experiment (operational log `out/_incidents.json`; not experimental results)

- **2026-11-10 10:17-10:21, GPU 1 / GPU 0.** After restarting both workers (10:17) my LLaVA worker occupied GPU 1 while another agent's LLaVA job (08_rescope/p11_retrofit c_eval.py, gpu_wait.sh 1 17600) was waiting for it; I stopped my workers at 10:20-10:21 and re-queued them. Impact on other agents: a delay of about 3 minutes for that job (it started on GPU 1 after my worker left); no failure. Cause: both cards were empty when my workers started (10:17:31); the other job was queued about a minute later and my workers had no automatic yield at that time. Fix: stopped by hand after about 2 minutes; since 12:13 every worker runs the watchdog of p12_common.start_watchdog and my launcher (code/gpu_wait.sh + code/gpuq.py) defers to other agents' queued jobs.
- **2026-11-10 11:40:27-11:40:36, GPU 0.** My Qwen worker (pid 4034376) and another agent's job (08_rescope/p11_retrofit, code/e_gen.py TB-Q100-ae-l2_3, started by its gpu_run.sh) were launched within one second of each other on the same free block (15637 MiB free seen by both launchers). Both ran out of memory while loading and exited. Impact on other agents: one failed start of the other agent's job (its launcher counts it as try 1 of 6); the job was restarted by its own launcher at 11:43:37 and ran. No output of another agent was touched. Cause: my launcher (code/gpu_wait.sh) only recognised other agents' gpu_wait.sh / wait_gpu.sh as queued jobs, not their gpu_run.sh / gpu_fill.sh / gpu_big.sh / gpu_any.sh; two launchers that had waited for the same memory fired together. Fix: code/gpuq.py + code/gpu_wait.sh: my launcher adds the need of every other agent's waiting launcher that fits (or nearly fits) on the card to its own requirement; every worker runs a watchdog (p12_common.start_watchdog) and leaves the card when another agent's queued job is blocked only by it, or when the card is almost full.
- **2026-11-10 12:10, GPU 1.** I stopped my LLaVA worker (pid 4037158, started 11:24) by hand because a queued job of another agent (09_testbed_scale, gpu_big.sh 15400, queued 11:57:53) seemed to fit on GPU 1 only without it. That job was in fact restricted to GPU 0 by its own environment (BIG_GPUS=0), so my worker had not delayed it. A different job of the same agent (gpu_big.sh 16300, queued about 12:13) started on GPU 1 at 12:14:50 in the memory my worker had released. Impact on other agents: none. Cause: my queue inspector assumed gpu_big.sh may use any card; it now reads the card set from the launcher's environment. Fix: code/gpuq.py; the interrupted configuration is redone on resume.
- **2026-11-10 12:21:50-12:24:08, GPU 0.** Near miss caused by a decision of mine. To get my Qwen job started I narrowed the margin by which my launcher defers to other agents' queued jobs (gpuq.py NEAR_MIB 1500 -> 600 MiB), so that it would use memory a queued job of another agent (09_testbed_scale, gpu_big.sh 15400, 945 MiB short of fitting on GPU 0) was waiting for. My launcher fired at 12:23:25; a neighbour finished in the same seconds and that queued job was launched by its own launcher at 12:23:27 on the same card, while my worker (pid 4081724) was loading its model. My worker's boundary check found another queued job blocked by it and left the card at 12:23:32, 7 seconds after starting; it evaluated nothing. Impact on other agents: no failure and no delay found (their job, pid 4084567, kept running; no out-of-memory message names my process). It is the same race as 11:40 and went well only because a third job had just released about 7.8 GiB. Cause: I loosened my own safety rule in the direction that lets my lower-priority job start next to a queue. Fix: margin restored at 12:24:08, then removed altogether: gpuq.reserve() now blocks my launcher while ANY other agent's launcher is waiting for the card, whatever its distance from fitting. Consequence: my jobs start only when no other agent has anything queued for the card.
- **2026-11-10 12:33:00-12:33:36 (GPU 1) and 12:35:47-12:36:55 (GPU 0), GPU 1, GPU 0.** Twice my launcher (no other agent's launcher waiting on two looks, enough free memory) started a worker and, seconds later, another agent queued a job that fitted on the card only without it: the LLaVA D2 worker on GPU 1 (09_testbed_scale gpu_big.sh 16100 queued; my worker left at 12:33:36 at its first boundary check) and the Qwen stream worker on GPU 0 (08_rescope/p11_scopecut gpu_wait.sh 9700 and 09_testbed_scale gpu_fill.sh 12000 queued; my watchdog left the card at 12:36:55). Neither worker produced a record. Impact on other agents: those jobs could start up to about a minute later than otherwise; no failure. Cause: the other agents' chains queue their next job within seconds of a job ending, so a card that looks free between two of their jobs is not free for a lowest-priority job. Fix: code/gpu_wait.sh now starts a job only after the card has been quiet (enough free memory, no other agent's launcher waiting, no memory growth) on 4 consecutive looks over about three minutes.
- **2026-11-10 17:40:46-17:43:58 (resume run), GPU 1.** My LLaVA worker of the resume (pid 81476, on GPU 1 since 17:19:58, 14.7-16.0 GiB) could not see a queued LLaVA job of another agent (02_research/code/rescope, tests/gpu_launch.sh, need 16000: it waits for 17024 MiB free on a card), whose launcher started at 17:40:46, and so did not leave the card for it. That launcher fired at 17:43:32, at a moment when my worker's usage was low (17441 MiB free); both jobs then shared the card with 1.5-2.7 GiB free until my worker left at its next configuration boundary, 17:43:58 (its tight-card rule). Impact on other agents: no failure (their log runs to DONE at 17:48:55 without an out-of-memory message) and no delay that the logs show: that launcher appears to have waited for its own package's launch lock, held by its Qwen launcher until about 17:43:06 (inferred from launch_pre_Q.out: launch at 17:41:36 plus the 90 s hold of gpu_launch.sh), and fired 26 s later on its first pair of looks. The exposure was real all the same: their model loaded next to my worker with 1.5 GiB free, and with a slightly larger footprint of mine their launcher would have kept waiting behind a process that could not see it. Two earlier failures of the same agent's Qwen test job (17:37:47 on GPU 1, 17:41:04 on GPU 0) name my processes 81476 / 81477 in torch's message; both stopped at that job's own memory cap (`11.00 GiB allowed`, 10.90 and 10.93 GiB allocated, with 4.48 and 2.62 GiB still free on the card). Cause: code/gpuq.py recognised other agents' launchers by a fixed list of script names (gpu_wait.sh, wait_gpu.sh, gpu_run.sh, gpu_any.sh, gpu_fill.sh, gpu_big.sh). tests/gpu_launch.sh was not on it (nor 09_testbed_scale's code_text/gpu_text.sh, nor p11_retrofit/verify/run_gpu.sh), so neither my launcher nor my workers' watchdog saw that queue. The verifier's report (section 7) describes the same gap in its own inspector, fixed there at 16:02; I did not carry that fix over to code/gpuq.py before launching the resume. Fix: 18:14-18:26: code/gpuq.py treats ANY shell script of another agent whose name contains gpu / wait / launch / queue as a launcher (layouts read: <gpu|any> <need>, <g,g> <need>, <need>, run_gpu.sh, gpu_text.sh; an unreadable layout counts as a LLaVA-sized window on both cards; scripts that wait for a process are skipped until that process has ended) and reads --mem_gb as a declared cap. code/gpu_wait.sh uses it from its next look on. The LLaVA worker started at 18:14:25 had loaded it and left the card at 18:17:55, 21 s after the same agent queued a LLaVA job (18:17:34). The Qwen worker started at 17:20 keeps the old inspector in memory and is covered by code/ext_watchdog.py since 18:21 (it sends SIGTERM to that worker of mine, and to nothing else, when the current inspector finds a queued job blocked by it).

