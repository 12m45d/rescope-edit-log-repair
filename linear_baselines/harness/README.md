# harness — the one repair harness for every 07_repair_v3 RQ

```python
import sys; sys.path.insert(0, ".../07_repair_v3")
from harness import load, sample_complaints, repair, evaluate, baseline
H  = load("TB-Q100-ae-l2_3")                  # model at W_E; exact H.W0 / H.WE; H.checks
cs = sample_complaints(H, n=10, seed=0)        # D1 v2, REPAIR pool, 5 image + 5 text (kind="D2" possible)
corr, cost = repair("COMP", H, cs[:1], protect=("K_last",), query="answer", form="adapter")
rec = evaluate(H, corr, cs[:1])                # uniform metrics; model back at W_E
```

| module | content |
|---|---|
| `hlog.py` | `load(name)` (all testbed-v3 logs + `old_W2`), exact W_0/W_E (checked vs stored composite), `rollback_weight(R)`, `fo_scores`, `Correction` (none / fp64 gated adapter / weight), `applied(corr)`, edit keys cache |
| `suffix.py` | exact suffix engine: re-runs only blocks L+1..N; logits bit-identical to full forwards (tests) |
| `sampler.py` | fixed REPAIR / HELD-OUT split of the 500 probe items (seed 20261109); seeded stratified random complaint sampler |
| `methods.py` | `repair(method, H, complaints, **cfg)` for NONE, COMP, REEDIT, ROLLBACK_FO, ROLLBACK_BEH, ROLLBACK_ALL; pass counter, FLOPs estimate, CUDA-synced wall-clock, setup excluded |
| `evaluation.py` | `evaluate(...)`: D1 (+D2 when the complaint is D2) resolution; edit preservation over all edits x {rel, T-Gen, I-Gen} (B=1 TF exact/token, broken/fixed); held-out collateral (non-complaints lost, complaints fixed, T-/I-Loc vs W_0); ||corr||_F; `baseline(H)` cached + checked against stored Rel / reproduced flags |
| `stats.py` | uid-level bootstrap CI, exact McNemar |
| `tests/run_tests.py` | unit tests -> `out/tests/*.json`; `tests/render_tests.py` -> `TESTS.md` |
| `smoke/` | `run_smoke.py` -> `out/smoke/rows_*.jsonl`; `aggregate.py` -> `results.json`; `render.py` -> `results.md` |

Method options (all alpha = 1, no tuning):

- `COMP`: `protect` any union of `"none"`, `"K_last"`, `"K_ans"`, `"K_seq:<n>"` (every position of the n
  shortest edit sequences), `"probe:<n>"` / `"probe_last:<n>"` (known-good REPAIR-pool probe rows,
  answer / last-prompt positions; never held-out rows), `"K_pc:<r>"` (top-r principal directions of
  K_all = every position of every logged edit sequence, uncentred, fp64 eigh cached as
  `cache/pcbasis_*.pt`; added for RQ3, unit test `kpc`); `query` `"answer"|"all"`; `form`
  `"adapter"|"merged"`; `gate` (1e-8 default, `None` = ungated); several complaints = one joint solve.
- `REEDIT`: `variant` `"k0+kp"` (default) | `"k0"`, `l2` (default the log's). z* must be an oracle:
  vendor zopt, then the fixed loop (early stop on the evaluated delta) at 2x steps, then 200 steps /
  early_stop 0.01; never-oracle rows are not installed (`meta.not_installed_rows`, `zopt_attempts`).
  Solve in factorised rank-1 form (fp64, on `H.solve_dev`) == vendor `Filters.solve` (W_after bitwise).
- RQ2 options (backwards compatible; test `rq2cells`): `COMP(solve="alphaedit", variant, l2)` installs the
  log-derived target (K_q, V) with the log's AlphaEdit solve ([redacted: describes a related submission under double-blind review] + Kp soft term + l2; `protect` ignored);
  `REEDIT(solve="exact", protect=("K_ans",))` installs the z* target (k1 -> z* - h) with COMP's exact null-space
  solve; `REEDIT(form="adapter", gate)` = fp64 side path instead of the bf16 merge; `REEDIT(targets=...)` reuses
  z* from an earlier `REEDIT(return_targets=True)` (pops `corr.meta["_targets"]`); REEDIT cost records
  `passes_zopt` / `flops_zopt` / `n_backward_zopt` (the z* phase).
- `ROLLBACK_FO`: `budget` (10 / 20). `ROLLBACK_BEH`: `budget` (20); criterion (# y0 tokens, mean log p).
- Rollback semantics (B=10 logs): W(R) = bf16(W_0 + (C - X(R))), C = W_E - W_0 (fp32, == stored composite);
  X(R) = exact applied delta of every batch entirely in R + the rank-1 solver term of every other rolled-back
  edit (no re-solve; PREP_VERIFY's vc_behav convention); W(empty) == W_E bitwise, W(all) := W_0.

Pass kinds in `cost["passes"]`: `full`, `prefix` (layers 0..L), `fwd_bwd` (vendor zopt), `suffix`,
`suffix_grad`. Protocol notes (PREP_VERIFY fix 2) are in `hlog.py`'s docstring; TB-L100-ae-l2_1 is
flagged `H.checks["collapsed_working_point"]`.

## PREP_VERIFY fixes — where they live

1. v2 everywhere: the harness reads only `probe_records_v2.jsonl` (complaints = `is_complaint_D1` /
   `is_complaint_D2`, `y0_valid`). `baselines/smoke_reedit.py`, `rollback_opt/code/{capture_keys,run_ropt,
   refine_open,render}.py` now read the `_v2` files (env `ROPT_SUFFIX`, default `_v2`; `.bak_v1` copies
   kept). v2 complaint keys are captured for TB-Q100-ae l2 10/3/1 and TB-L100-ae l2 10/3
   (`rollback_opt/cache/keys_<log>_v2.pt`). The Track C optimum re-run is CPU-only (~50 min per 330
   complaints at 22 workers, ~6 h for the five logs) and resumable:
   `cd rollback_opt/code && ROPT_SUFFIX=_v2 python run_ropt.py --logs TB-Q100-ae,TB-Q100-ae-l2_3,TB-Q100-ae-l2_1,TB-L100-ae,TB-L100-ae-l2_3 --workers 22 --tl 90 && ROPT_SUFFIX=_v2 python render.py`.
2. Protocol text: `hlog.py` docstring (re-tokenised y0 under TF; LLaVA leading-space caveat; D2 pooled
   only); `H.checks["collapsed_working_point"]` for TB-L100-ae-l2_1.
3. z* oracle before install, attempts recorded: `methods.m_reedit`.
4. Cost: pass counts by kind + FLOPs estimate; REEDIT solve in factorised rank-1 form, no diagnostics;
   per-log setup cached outside the timer for every method; COMP and REEDIT solve on the same device
   (`H.solve_dev`: GPU for Qwen, CPU fp64 for LLaVA to stay < 16 GB); random complaint samples.
5. Surrogate: the harness never uses the linear surrogate as a rollback criterion; ROLLBACK_BEH is the
   behavioural (teacher-forced D1) greedy, its per-step trajectory is stored in `corr.meta`.

RQ4 options (backwards compatible; defaults unchanged; test `tests/test_rq4_ext.py` -> `out/tests/rq4_ext.json`):
`sampler.set_split(H, frac, seed)` (REPORTED share `frac`, item level, nested for one seed; drops pool-dependent
caches; non-default splits get their own baseline cache file), `evaluate(..., per_item=True)` (per-edit /
per-held-out-row / per-item Loc values), `COMP` `prior=` / `extra_basis=` / `keep_keys=` (sequential repair with
accumulating protection), `ROLLBACK_FO` `offload=` / `record_flags=` (many complaints), `REEDIT` `chunk=`
(memory; groups image / text rows, which the vendor z* batching requires; still one joint solve).
