Per-row records of the behavioural and the first-order rollback (configurations RB_BEH, RB_FO) on image complaints of the REPAIR pool of the two
development logs.  They are from an EARLIER, EXPLORATORY PHASE of the work (a head-to-head of linear repairs on single complaints, run before the
protocol was frozen), not from a registered experiment.  experiments/e9_maintenance/code/e9_render.py reads them for the numbers 'rollback on image
complaints' of Table 4 (b) of the paper (how many edits a rollback deletes for one complaint).  Only the rollback rows of that phase are released.
Fields: log, seed, config, method, cfg (budget), group / uid / probe_idx (the complaint), cost (passes, tokens, time), meta (R = the deleted edits,
k = their number, the trajectory of the search), eval (the complaint after the repair, edits broken / fixed, collateral rows, locality).
rescope_paths.py mounts this directory at the path of the working tree (02_research/experiments/07_repair_v3/rq1/rows).
