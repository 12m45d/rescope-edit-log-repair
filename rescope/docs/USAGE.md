# rescope — scoped replay of an edit log

Reference implementation for the final experiments. One edited linear module; the model keeps the pre-edit
weight `W_0` and a forward hook adds the scoped log. Everything about detectors, thresholds and values is fp64;
the hook casts once to the module dtype. Definitions live in the module docstrings (`scoped.py` first).

Import: `sys.path.insert(0, "$RESCOPE_ROOT/02_research/code")`, then `import rescope`
(python: `$RESCOPE_PYTHON`; cap BLAS threads, e.g. `OMP_NUM_THREADS=4`).

## Example

```python
import json, torch
from rescope import *                                        # load, Evaluator, ScopedLog, operators, gates, splits, ...

H = load("TB-Q100-ae-l2_3", device="cuda:0", mem_gb=11)      # harness handle: model at W_E, exact H.W0 / H.WE
ev = Evaluator(H)                                            # the ONE evaluation path: gates, weights, corrections
log = from_harness(H).to(H.dev)                              # U_i, W_i, stored keys k_i, batch of every edit (fp64)
S = ScopedLog.from_log(log, role="last", fire="top1", response="repl", W0=H.W0)   # detectors W_i, rebased values
S, rec = add_guards(S, *ev.own_keys("rel", answer="all"))    # B = 1 keys of the edits' own prompts: guards, BEFORE calibrating
S, rec = calibrate_log_only(S, q=0.999)                      # 1. thresholds from the log alone
benign = splits.sample(ev.probe_items("repair"), 100, seed=0)       # benign calibration queries: a SEEDED DRAW from REPAIR
keys = [ev.eligible_keys(it, S, answer="all") for it in benign]    # every key a query can present: teacher forcing + decode
S, rec = calibrate_conformal(S, benign_scores(S, keys), alpha=0.1)    # 2. P(fresh benign query fires) <= 0.1 (over draws)
print(rec["eta"], rec["n_own_keys_silenced_by_literal_rule"], rec["guarantee"])

rows = ev.probes({"scoped": S, "deployed": None, "pre-edit": H.W0}, pool="heldout_disjoint")
print({k: summarise(v)["probe"] for k, v in rows.items()})   # D1 % / firing % per method; rows keep per-row flags
rows_p = ev.probes(S, pool="heldout_disjoint", lp="prompt")  # the same rows, gate at the decode's position (differs if lp_shift)
print(summarise(ev.edits(S)))                                # Rel / T-Gen / I-Gen exact %, own_fired_last %
ses = Session(S)                                             # complaint-driven repairs: pure operators + bookkeeping
for it, r in zip(benign, ev.tf_rows(S, benign)):
    if r["D1"] and r["fired"]:                               # 3. wrong-answer complaint -> least threshold repair
        rec = ses.complain(ev.eligible_keys(it, ses.S, answer="all"))     # "all": the repair holds in a decode too
        print(it["sid"], "raised", rec["n_raised"], "refused", rec["refused"], rec["conflicts"])
tgen = ev.edit_items(("tgen",), edits=range(H.T // 2))       #    (edit half A: never repair on the reported half)
for it, r in zip(tgen, ev.tf_rows(ses.S, tgen)):
    if not r["own_fired_last"]:                              # 5. why-not complaint: the rephrase should fire its edit
        rec = ses.why_not(it["edit"], ev.eligible_keys(it, ses.S), mode="threshold")
        if rec["conflict"]:                                  #    a recorded benign query would fire: add a detector
            rec = ses.why_not(it["edit"], ev.eligible_keys(it, ses.S), mode="detector", gamma=0.95)
S_null, rec = null_repair(ses.S, torch.cat(keys), ridge=0.0)  # 4. detectors with zero coordinate on benign keys

hz = harness()                                               # the shared repair harness (07_repair_v3), private cache
cs = hz.pkg.sample_complaints(H, n=10, seed=0)               # (REEDIT: corr, info = reedit(H, cs); never chunk=None)
methods = {"repaired": ses.S, "null": S_null, "cosine": cosine_gate(log, 0.95, W0=H.W0), "exact": exact_gate(log, W0=H.W0),
           "ball": ball_gate(log, 0.5 * float(key_distances(log).min()), W0=H.W0),
           "router": router_gate(log, H.WE, H.W0, q=0.25), "zscore": zscore_gate(H.WE, H.W0, alpha=2.2),
           "COMP": hz.pkg.repair("COMP", H, cs, protect=("K_last",), query="answer")[0],     # a harness Correction
           "ROLLBACK_ALL": H.W0}                             # any full replacement weight
out = {"probes": ev.probes(methods, pool="heldout_disjoint"), "edits": ev.edits(methods),
       "hard_negatives": ev.hard_negatives(methods), "d2": ev.d2(methods, pool="heldout"),
       "free_generation": ev.generate(methods, ev.edit_items(("rel",)))}
rows_json = json.dumps(out)                                  # plain dict rows: write them, bootstrap later
```

The code block above is executed by the GPU test `b8_usage` (on a subsample of the items).

## Final experiments: default calibration, baselines, locality, scale, cached keys

Additions for `10_final/PROTOCOL.md` (amendments of 2026-11-10). Nothing above changes its behaviour; the default
method of the protocol is now built like this (executed by the GPU test `b14_usage2`):

```python
import torch
from rescope import *

register_logs()                                              # scale / part-2 / text registries, in-process (optional: load() does it on demand)
H = load("TB-Q100-ae-l2_3", device="cuda:0", mem_gb=9.8)     # any registered log; T >= 500: lean loading (no batch deltas in RAM)
ev = SparseEvaluator(H, cache="auto")                        # the Evaluator's rows; the model runs only where something fires
log = from_harness(H).to(H.dev)                              # gate on the model's device: calibration and hook compute the same bits

def calibrated(feature, rule=calibrate_two_block, alpha=0.05, n=200, seed=0):
    S = feature_gate(log, feature, W0=H.W0)                  # 'ncoord' = ReScope | 'coord' | 'cos' | 'negdist' (same rule for the baselines)
    S, _ = add_guards(S, *ev.own_keys("rel"))                # B = 1 keys of the edits' own prompts: guards, BEFORE calibrating
    benign = splits.sample(ev.probe_items("repair"), n, seed)        # benign calibration queries: a seeded draw from REPAIR
    return rule(S, [ev.eligible_keys(it, S) for it in benign], alpha, **({"seed": seed} if rule is calibrate_two_block else {}))

S, rec = calibrated("ncoord")                                # ReScope(0.05): two-block rule = the default calibration
C, rec_c = calibrated("cos")                                 # the cosine gate under the same calibration
A, rec_a = calibrated("ncoord", rule=calibrate_shared)       # ablation: one shared scale (Section 0 of the protocol)
print(rec["eta"], rec["n_edits_capped"], rec["bound_false_firing"], rec["guarantee"])

methods = {"rescope": S, "cosine": C, "shared": A, "deployed": None}
rows = ev.probes(methods, pool="heldout_disjoint")           # D1, firing flags and tok_loc (token-level locality) per row
loc = {k: locality(v) for k, v in rows.items()}              # T-Loc* / I-Loc* per method
dep = ev.token_locality(None, "heldout_disjoint", rows=rows["deployed"])     # + stored benchmark locality of the log, signed difference
edits = ev.edits(methods)                                    # Rel / T-Gen / I-Gen, teacher forced
gen = ev.generate({"deployed": None, "dual": ev.dual("eq"), "dual_contains": ev.dual("contains"), "rescope": S},
                  ev.probe_items("heldout_disjoint")[:20] + ev.edit_items(("rel", "tgen"), range(10)))      # DUAL: generated answers only
stored = stored_metrics(H.name)                              # the deployed log's own metrics, from the testbed's files (no forward)
out2 = {"loc": loc, "deployed": {k: v for k, v in dep.items() if k != "per_row"}, "edits": {k: summarise(v) for k, v in edits.items()},
        "dual": summarise(gen["dual"]), "stored": stored["edits"], "sparse": dict(ev.stats)}
```

Without a model, from the per-backbone key cache (`10_final/cache/<backbone>/`, format: `FORMAT.txt` there and the docstring
of `keycache.py`):

```python
kc = KeyCache("qwen25vl")                                    # index.jsonl + keys_tf.pt / keys_decode.pt / keys_decode_steps.pt (float32)
L = from_testbed("TB-Q2000-ae-l2_3")                         # ANY log of the backbone: the keys do not depend on the log
G = feature_gate(L, "ncoord")
G, _ = add_guards(G, *kc.own_keys(L.meta["uids"]))           # B = 1 own-prompt keys from the cache
ben = splits.sample(kc.log_rows(L.meta["uids"], kind="probe", pool="repair", valid=True), 200, 0)
G, rec = calibrate_two_block(G, coords=kc.coords(G, ben), alpha=0.05, seed=0)
dec = kc.decide(G, "tf", log=L)                              # fired / top_edit / own_top / margin / near for every cached sequence of the log
```

### 1. Two-block conformal calibration (`calibrate_two_block`), the default calibration

For a scoped log `S` every key `k` has one FEATURE per detector, `r_j(k) = S.scores(k)[j]`; `Q[q, j]` = the largest `r_j`
over the eligible keys of benign query `q` (`S.query_max`).

1. The `n` benign calibration queries are split at random (seeded) into block 1 (`n1 = n // 2`; `split=(a, b)`:
   `n1 = n a // (a + b)`) and block 2 (`n2 = n - n1`).
2. Baseline `b_j = max over block 1 of Q[q, j]`.
3. Score of a query `s(q) = max_j (Q[q, j] - b_j) / u_j`, `u_j = 1` (option `u=`: any fixed positive vector; it must not
   depend on the calibration queries).
4. `eta` = the `k`-th smallest block-2 score, `k = ceil((1 - alpha)(n2 + 1))` in exact integer arithmetic (`alpha` is read as
   a fraction: `0.05` -> 1/20; `conformal_rank_exact`); `eta = +inf` if `k > n2` (nothing fires: `n2 >= 1/alpha - 1` is needed).
5. Detector `j` is a candidate at a key iff `(r_j - b_j) / u_j > eta` (STRICT).
6. `fire="top1"`: the candidate with the largest excess `(r_j - b_j - eta u_j) / u_j` releases, i.e. the largest margin
   `(r_j - b_j) / u_j` (the two orders are the same; an exact tie goes to the lowest detector index; with `eta = +inf` only
   capped detectors can be candidates and the margin still orders them). `rank="r"`: the largest `r_j`.

Guarantee of the uncapped rule, for benign queries exchangeable with the calibration queries and any `n1 >= 1`:
`P(a fresh benign query has a candidate) <= floor(alpha (n2 + 1)) / (n2 + 1) <= alpha`, marginal over the draw and the
split, ties included (`rec["bound_false_firing"]`). It is a bound on the calibrated event only: same feature, same role,
same forward path as the calibration queries (rule 6 above).

**Stored form.** The thresholds are not `b_j + eta u_j` (under that rounded sum the query that defines `eta` can fire:
referee C1). `det_tau[j] = succ(theta_j)`, `theta_j` = the largest fp64 number `r` with `fl(fl(r - b_j) / u_j) <= eta`
(`float_safe_threshold`: exponential bracket + bisection on the fp64 lattice), and `eta = 1`. Subtraction of a constant and
division by a positive constant are monotone under correct rounding, hence for every finite `r`:
`r >= det_tau[j]` <=> `fl(fl(r - b_j) / u_j) > eta`, bit for bit: the package's ordinary non-strict gate IS the strict
conformal event. The baselines and scales of the ranking are kept in `S.det_b` / `S.det_u` (`ScopedLog.rank_key`); they
persist through the repair operators and through `calibrate_log_only` / `calibrate_conformal` (which only change
thresholds): recalibrate from a fresh `feature_gate`, or with `calibrate_shared` (it resets the ranking to `r`).
In exact arithmetic the candidate condition is `r_j - b_j > eta u_j`; in floating point the library takes the score form
`(r_j - b_j) / u_j > eta` as THE definition, because `eta` is itself such a score (with the product form the query that
defines `eta` can fire); for `u = 1` the two are the same expression.
The identity is about the comparison: it holds for the coordinate the gate computes. The coordinate itself moves in its
last bits with the device and the batch shape (rule 2): build the gate on the model's device (`from_harness(H).to(H.dev)`,
as in the example) and pass each query's keys as the hook sees them (one key per role-`last` query); calibration and
hook then compute the same bits, and a test query identical to a calibration query is decided as the rule says.
A gate calibrated on the CPU (e.g. from a `KeyCache`) decides such exact ties up to round-off (`near`).

**Guards.** Default: an edit's guard keys (its stored solver key and the keys added with `add_guards`, i.e. the B = 1 keys of
its own prompts) keep firing: a detector that the literal rule silences on a guard is capped at `(1 - 1e-9) x` its smallest
guard coordinate. `rec["n_edits_capped"]`, `rec["capped_edits"]`, `rec["guarantee"]` say for how many edits the guarantee
is void; `allow_silence=True` returns the uncapped rule (`rec["n_own_keys_silenced"]` guards then do not fire).

**Any feature.** `feature_gate(log, feature)` builds the un-thresholded gate: `ncoord` (normalised solver coordinate =
ReScope), `coord` (raw solver coordinate), `cos` (raw cosine to the stored solver key), `negdist` (minus the Euclidean
distance to the stored key). `calibrate_shared(S, queries, alpha, u=None | "tau" | vector)` is the shared-scale rule in the
same exact form (no block 1: `b = 0`, bound `floor(alpha (n + 1)) / (n + 1)`); with `u=None` it is one conformal threshold
on the feature (a conformal cosine threshold, a conformal radius). `calibrate_dense(gate, scores, alpha, two_block=False)`
does the same for the router / z-score gates from one scalar score per query. `two_block_rule` / `shared_rule` work on
plain matrices. `calibrate_conformal` (multiplier with a 1e-9 guard) is unchanged and kept for the earlier experiments.

### 2. Baseline DUAL (answer match)

`ev.dual("eq" | "contains", on="line" | "y")` (or `Dual(targets, ...)`) is a method of `ev.generate` / `ev.d2` only: the deployed
and the pre-edit model both decode the query (decoder v2, greedy); the deployed answer is returned iff, after
`tb_common.norm_answer`, it EQUALS the target of a logged edit (`eq`: first line of the 48-token answer, the comparison behind
the row field `eq`; `on="y"`: the 10-token answer) or CONTAINS one (`contains`: `tb_common.contains_ref` on the 48-token
answer); otherwise the pre-edit answer. Rows are the decode row of the returned answer plus `fired` (= released),
`dual` (`match`, `n_matching_edits`, `edits`, `own`, `source`), `n_generations = 2`, `n_new_tokens = [deployed, pre-edit]`;
`summarise(rows)["dual"]`. Teacher-forced functions refuse it (`TypeError`). The deployed / pre-edit decodes of an item are
computed once and shared by every method of one call. For tables that need D1 / exact / `tok_loc` of DUAL at B = 1,
`ev.dual_tf_rows(dual, items)` returns the teacher-forced row of the MODEL that DUAL selects for each query (a labelled
composition, not part of the baseline's definition: two generations + one teacher-forced forward per query).

### 3. Token-level locality at B = 1

Every teacher-forced probe / hard-negative row carries `tok_loc`: the share of answer tokens whose argmax equals the
pre-edit model's under teacher forcing of the pre-edit answer. For a valid `y0` the pre-edit argmax is `y0` itself, so
`tok_loc == token_acc` (no extra forward); rows with an invalid `y0` (`probe_items(valid_only=False)`) cost one forward
at `W_0` per item (`Evaluator.pre_ids`). A row with a non-finite logit under the method counts 0 (it is counted:
`n_*_nonfinite`); an invalid-`y0` row whose PRE-EDIT logits are not finite has no value (`n_*_undefined`). `locality(rows)` -> `T-Loc*` / `I-Loc*` (mean over
text / image probe rows, percent), `*_em` (= 100 - D1 rate), `*_finite`, counts. `ev.token_locality(method, pool)` adds, for
the deployed log, the stored benchmark locality and the SIGNED difference `T-Loc* - T-Loc`, `I-Loc* - I-Loc`. The two are
different quantities (decoder-v2 answer at B = 1 on valid rows; benchmark reference answer in padded batches of 8 on all
500 items): the difference is reported, not a check.

### 4. Scale-safe use

- `register_logs()` registers, in-process, the logs of `09_testbed_scale/code/scale_registry.py`, of its part 2 (names
  appended to `registry.json`, read again at every call, and names of `out_part2/plan.json` whose directory is complete)
  and of `code_text/text_registry.py`. Every source is optional (`register_logs()["sources"]`); `load` / `load_rows` /
  `stored_metrics` register on demand, so a name that appears later is picked up.
- `load(name)` reads logs with `T >= 500` LEAN (default): the applied batch deltas are summed one at a time and not kept
  (`H.A[b]` reads one from disk on demand); `W_0`, `W_E`, factors and load-time checks are those of the harness loader bit
  for bit (test `d1_scale`). `load(name, reuse=H_old)` reuses the model of another log of the same backbone.
  `H.load_info` holds seconds and the peak resident memory.
- For `T >= 500` the harness functions that hold every position of every edit sequence are REFUSED: `harness.baseline`,
  `harness.evaluate`, its evalcache, `HLog.edit_keys` (hence COMP protections built on all positions). Use this package's
  B = 1 evaluation; `rescope.evaluate.ALLOW_HARNESS_AT_SCALE = True` lifts the guard.
- `stored_metrics(name)`: the deployed log's own metrics from `metrics.json`, `per_edit_metrics.jsonl`,
  `probe_records_v2.jsonl` (edits and benchmark locality: the testbed's vendor pass in padded batches; probes: decoder-v2
  rows by ptype and pool; per-edit and per-row flags).
- Hard negatives: `Evaluator(H, hn_dir=HN_DIR_FINAL)` reads the final set (existing + new rows, cross rows `HN-X` only
  if both edits are in the log); its items carry `rule`, `group`, `new`, `src_related_to_log` (the row's source item is
  related to an edit of THIS log: not a clean negative) and `q_edit` (an edit of this log asks the row's question: a
  cross-like row); such rows are kept and flagged, not dropped. `hn_dir=text_registry.HN_DIR` for the text logs.

### 5. Cached keys

`KeyCache(backbone, verify=False)` reads `10_final/cache/<backbone>/` (index and key files are opened together and checked
against each other and `meta.json`; `verify=True` also checks the md5 sums of the key files; `kc.is_final`, `kc.summary()`):
`rows(kind=, family=, T=, pool=, valid=)`, `log_rows(uids)`,
`keys(which)`, `coords(gate, rows, which)` (per-query maxima for `calibrate_*(coords=...)`), `own_keys(uids)`, and
`decide(gate, which, rows | log=)` -> `fired`, `near`, `n_cand`, `top_edit`, `margin`, `own`, `own_cand`, `own_top` for
every sequence. `which`: `tf` (teacher-forced key), `prompt` (same sequence at the decode's token), `decode` (prompt-only
key), `steps` (the 11 forwards of a decode), `both`, `all` (as `eligible_keys(answer=...)`). Gates: scoped logs with role
`last` and the fp64 gate (any detector kind) and the router. `near` marks sequences with a coordinate within
`1e-9 x (|threshold| + |w||k|/s)` of a threshold: there the GPU hook may decide otherwise (round-off of the fp64 products
is about `1e-13` of that scale).

`SparseEvaluator(H, cache="auto" | KeyCache | None)` is an `Evaluator` whose teacher-forced functions (`edits`, `probes`,
`hard_negatives`, `tf_rows`; `lp="tf" | "prompt"`) return, for scoped gates with role `last`, exactly the Evaluator's rows
while running the model only where a gate fires or is `near`: a silent sequence is the pre-edit model bit for bit, so its
row is the cached pre-edit outcome (`pre_exact`, `pre_token_acc`, ... of the cache; or a local store filled on first use
by one truncated forward and one forward at `W_0` per item, `save_local` / `load_local`) plus the flags of a silent hook.
Items are matched to the cache by content (format, question, image, teacher-forced text), so the cache of a backbone
serves every log. Other methods, other roles, free generation and D2 take the Evaluator's path unchanged.
`verify_cache(items, n)` re-captures a sample and compares bit for bit. Tests: `a22_keycache`, `b13_sparse`.

### Where the additions live

| addition | file: functions | tests |
|---|---|---|
| two-block / shared conformal rules, exact rank, floating-point-safe thresholds, any feature | `calibrate.py`: `calibrate_two_block`, `calibrate_shared`, `calibrate_dense`, `two_block_rule`, `shared_rule`, `float_safe_threshold`, `conformal_rank_exact`; `baselines.py`: `feature_gate`; `scoped.py`: `det_b`, `det_u`, `rank_key` | `a17` - `a20` |
| baseline DUAL | `evaluate.py`: `Dual`, `Evaluator.dual`, `generate` / `d2` | `b11_dual` |
| token-level locality | `evaluate.py`: row field `tok_loc`, `Evaluator.pre_ids`, `locality`, `Evaluator.token_locality` | `b12_tokloc` |
| scale-safe use | `evaluate.py`: `register_logs`, `ensure_registered`, `load(lean=, reuse=)`, `stored_metrics`, the guards of `harness()`, `Evaluator(hn_dir=)`, `HN_DIR_FINAL` | `a21_scale_registry`, `d1_scale` |
| cached keys | `keycache.py`: `KeyCache`, `write_cache`, `decision_band`; `sparse.py`: `SparseEvaluator` | `a22_keycache`, `b13_sparse`, `b14_usage2` |

## Objects

| object | what it is |
|---|---|
| `EditLog` | `U (T, m)`, `W (T, d)`, `K (T, d)`, `batch (T,)`; `coords`, `own`, `crosstalk`, `delta`; `from_testbed(name)`, `from_harness(H)` |
| `rebase(log)` | `v_i = u_i + sum_{batch(j) < batch(i)} u_j <w_j, k_i>` (log quantities only); `rebase_exact` adds the write round-off |
| `ScopedLog` | detectors `(w_j, s_j, tau_j, edit_j, kind_j, key_j)`, `values`, `eta`, `role`, `fire`, `response`; guards = keys that MUST fire |
| `Hook(module, gate)` | the forward hook; `begin(lp, gen=False, seq_len=None)` per prompt (one `lp` per row; `seq_len` masks PAD positions); `summary()` = firing flags |
| `SparseEvaluator(H, cache)` | an `Evaluator` that runs the model only where a scoped gate fires (section 5 below) |
| `KeyCache(backbone)` | per-backbone cached keys -> `decide(gate, which, log=)` on the CPU, `coords`, `own_keys` |
| `Dual(targets, variant)` | baseline DUAL for `generate` / `d2`; `ev.dual("eq" \| "contains")` |
| `Evaluator(H)` | `edits`, `probes`, `hard_negatives`, `d2`, `generate`, `tf_rows` (teacher-forced ones take `lp="tf" \| "prompt"`); `eligible_keys(it, gate, answer=True \| False \| "both" \| "decode" \| "all")`, `decode_keys(it)`, `own_keys(metric, answer=True \| False \| "both" \| "all")`, `lp_shift(it)`, `predict`, `applied(method)` |
| `splits` | `select(rows, pool_of_item, pool)`, `annotate`, `summary`, `sample(items, n, seed)`; pools `repair \| heldout \| heldout_disjoint \| heldout_strict \| all` |
| `reedit(H, complaints)` | the harness REEDIT correction built correctly (z* at B = 1, `chunk=1`, one joint install) |

Roles `last | last+answer | all`; fire `top1 | all` (the candidate with the largest `r` releases, or the largest margin
`(r - b) / u` if the log carries `det_b` / `det_u`; an exact top-1 tie goes to the lowest detector index); response
`repl | gain | lin`. A detector is a candidate at an
eligible position iff `r_j(k) >= eta * tau_j` (non-strict, fp64). Where nothing fires the hook returns the module's
own output tensor: a sequence on which nothing fires is bitwise the pre-edit model.

## Operators (each returns `(new scoped log, record)`; the input is never modified)

| operator | effect |
|---|---|
| `calibrate_two_block(S, queries, alpha)` | DEFAULT: per-edit baselines from block 1, exact conformal `eta` from block 2, strict, floating-point-safe thresholds (section 1 below) |
| `calibrate_shared(S, queries, alpha, u)` | one shared conformal threshold / scale in the same exact form |
| `calibrate_log_only(S, q)` | `tau` = q-quantile of the pooled normalised cross-talk of the log's own solver keys |
| `calibrate_conformal(S, scores, alpha)` | `eta` = the `ceil((1 - alpha)(n + 1))`-th smallest benign score, times `1 + 1e-9` (strict conformal event) |
| `least_repair(S, keys)` | coordinatewise-maximum thresholds that silence a wrong-answer complaint; refuses if a guard would be silenced |
| `null_repair(S, keys, ridge)` | minimum-change detectors with zero (ridge: shrunk) coordinate on benign keys, own coordinate kept |
| `witness_repair(S, i, key, mode)` | `threshold`: lower `tau` to the witness; `detector`: add a cosine detector on the witness key |
| `scope_cut(S, rows, witnesses, lam)` | exact optimum of `w(clean rows) - lam * w(uncovered witnesses)` by one minimum cut |
| `add_guards(S, keys, edits, allow_nonfiring=False)` | further must-fire keys of an edit (e.g. its B = 1 own key, `ev.own_keys`); refuses keys that do not fire now |

Baselines with the same interface: `cosine_gate`, `exact_gate`, `ball_gate` (scoped logs with other detector kinds),
`router_gate`, `zscore_gate` (dense gates that need `W_E`). Deviations from the papers (full list: docstring of
`baselines.py`): `cosine_gate` (NeuralDB) defaults to role `last`, the published gate acts at every token position
(`role="all"`); `zscore_gate` (CoachHooK; the fixed `alpha=2.2` is also the value of its dynamic rule on both primary
logs) scores a KV-cache step with the statistics of the last full forward (`kv="stale"`), whereas the official hook
standardises per call and therefore never swaps in a one-token step (`kv="call"`); `ball_gate` (GRACE Eq. 1, strict
`<`, final prompt token) uses one global radius and adds the log's value instead of replacing the layer output;
`exact_gate` thresholds the cosine to the stored B = 10 key at 0.999 and can miss an edit's own B = 1 prompt (test `b10`).

## Rules that keep numbers trustworthy

1. **Splits.** Calibrate, repair and select on the REPAIR pool and on edit half A only; report on HELD-OUT and on
   HELD-OUT-DISJOINT (`pool="heldout_disjoint"`: 25 valid HELD-OUT rows per log repeat a REPAIR query). Two weaker
   overlaps remain inside HELD-OUT-DISJOINT and are stated next to its numbers: 90 valid image rows per log ask a
   question that a REPAIR row asks about another image (`pool="heldout_strict"` removes them; rows carry
   `question_in_repair`), and 29 valid rows per log repeat another disjoint row (rows carry `query_id`: bootstrap by
   `query_id` clusters, not by row).
2. **Keys of a query** are the module inputs at its eligible positions (`Evaluator.eligible_keys`); pass them as one
   tensor per query (a list of tensors = several queries). Coordinates are evaluated per query, exactly as the hook
   does; guard keys are always scored one by one.
3. **Two paths, two positions (decode metrics; numbers: test `b9`).** Teacher forcing puts the last prompt position
   at the harness index of the joint sequence. D2 and free generation put it on the last token of the prompt alone
   and re-evaluate it in up to 11 full forwards of growing length (10 no-cache forwards + the first forward of the
   KV-cache continuation). Two effects follow.
   *Same token, other sequence length:* the low-precision forward gives a slightly different key (up to 3 distinct
   keys within one decode), which moves a coordinate by up to 0.019 (Qwen) / 0.001 (LLaVA).
   *Another token:* where the tokenizer leaves a bare space between prompt and answer (LLaVA, answers that start with
   a digit: 46 of the 1685 sequences of the primary log, `lp_shift = 1` in every row; none of 1673 on Qwen) the
   teacher-forced position is one token later than the decode's, |dr| reaches 0.80, a gate with thresholds 0.25 fires
   at exactly one of the two positions on 22 of the 43 shifted benign rows, and D1 differs on 15 of them.
   A least repair computed from the teacher-forced keys with the default margin therefore does not hold in a decode
   (repaired complaints still firing in their decode: 1 of 30 on Qwen, 14 of 31 on LLaVA), and a larger margin on
   the teacher-forced keys does not help where the token differs (margin 0.05: still 13 of 31 on LLaVA; the sample
   contains every shifted REPAIR row that fires on either path, 19 complaints). Protocol:
   (a) complaints, calibration queries and own-prompt guards that must hold for D2 / free generation take EVERY key
   the query can present to the gate: `eligible_keys(it, S, answer="all")` / `own_keys(metric, answer="all")` = the
   teacher-forced key, the teacher-forced sequence's key at the decode's token where that is another token, and the
   keys of the 11 full forwards of a decode (`decode_keys`: bitwise the keys of the real decode, test `b9` (4)).
   Exact with the default margin; role `last` only; 12 truncated forwards per item. Cheaper approximation, and the
   only one for the roles `last+answer` / `all` (whose later positions depend on the generated tokens):
   `answer="both"` (teacher-forced + prompt-only key) with `margin=DECODE_MARGIN` (0.05); both leave 0 repaired
   complaints firing on either path in test `b9`, the measured need above the keys of both paths is 0.002;
   (b) on LLaVA report teacher-forced D1 at both positions: `ev.probes(S, pool, lp="tf")` and `lp="prompt"`;
   (c) an edit whose own sequence is shifted (LLaVA edit 39) was written at the teacher-forced position and cannot
   fire under role `last` in a decode: count it, do not tune for it.
4. **Margins.** "Just above" a coordinate means `m (1 + 1e-6) + 1e-12` by default (`margin`, `abs_margin`): enough for
   device / batch-shape round-off of the fp64 gate on the SAME key, NOT for another key of the same query (rule 3) or
   for a padded batch (decisions within a few per cent of a threshold are not stable across those).
5. **Own keys / guards.** A guard is a MUST-FIRE key. Operators keep every guard firing (the edits' stored solver
   keys, granted witnesses, `add_guards`) and report how many the literal rule would have silenced; a capped conformal
   threshold voids the guarantee for that edit (`rec["guarantee"]`). The stored solver key is the B = 10 batched key:
   the B = 1 key of the edit's own prompt has r in [0.950, 1.022] on Qwen (65 of 100 below 1), so thresholds capped
   at the stored keys silence own prompts (test `b10`: 35 of 100 fire on Qwen, 50 of 100 on LLaVA). Add the B = 1
   keys as guards BEFORE calibrating: `S, _ = add_guards(S, *ev.own_keys("rel"))` (`answer="all"` if the own prompt
   must also fire in every forward of a free generation). `add_guards` refuses a key on which its edit does not
   fire now, because the next calibration would lower that threshold to the key's coordinate (calibration records:
   `n_guards_not_firing_before`). Evaluate at B = 1 only.
6. **Conformal guarantee.** Scores must be `benign_scores(S, keys)` of the SAME scoped log (thresholds, role) that is
   then scaled, on queries exchangeable with the future benign queries and through the same forward path (rule 3:
   `answer="all"` if decode metrics are reported). The guarantee is MARGINAL over the calibration draw: the mean
   false-firing rate over draws is <= alpha, a single draw can exceed it (test `a14`, P11b's protocol, n = 100,
   alpha = 0.1: mean 9.4 % / 9.0 % on HELD-OUT, 95th percentile of the single draws 13.9 % / 14.0 %). Calibrate on
   seeded random subsets (`splits.sample(items, n, seed)`; `items[:n]` is in file order, not a random subset), use
   several seeds and report the mean and the spread over draws.
7. **y0** is the multi-token decoder-v2 answer (the harness reads `probe_records_v2.jsonl`); D1 uses valid rows only.
8. `precision="module"` / `"fp32"` are cost options, not the reference: the bf16 gate changes decisions (test
   `c_timing`); every reported number uses the fp64 gate.
9. Per-row firing flags come from the hook itself (`fired`, `fired_last`, `own_fired_last`, `top_last`, ...); they
   are `None` for methods that are not gates. A sequence with `fired == False` is bitwise the pre-edit model.
10. The harness is imported with a private cache directory (`rescope/_hcache`); nothing is written under `experiments/`.
11. **REEDIT** corrections come from `reedit(H, complaints)` (or `harness.repair("REEDIT", ..., chunk=1)`): with the
    harness default `chunk=None` the z* oracle fallback is skipped and some complaints are silently not installed
    (P12 Anomaly 2).
    A REEDIT correction is NOT reproducible from one build to the next: the z* optimisation of IMAGE complaints
    gives another target every time, in the same process and with the same seed (text complaints are bitwise
    reproducible; `tests/out/extra_reedit_repro_<log>.json`, 10 complaints, two builds: Qwen z* differs by 0.8 - 15.7 %
    of |z* - h| on the 5 image complaints, the correction by 4.9 % of its norm, Rel exact 78 vs 76 of 100, 8 of 490
    HELD-OUT flags differ; LLaVA 0.4 - 3.7 %, 1.6 %, no flag differs in that pair, one HELD-OUT flag against the
    build of test `b6`). The cause was not established. Build a REEDIT correction ONCE per configuration, save
    `corr.W` and evaluate the saved weight everywhere (a plain tensor is a method), or report the spread over builds.
12. A missing hard-negative file is an error (`ev.hard_negatives`, `ev.hn_items`), never an empty table.

## Known limitations

- One edited module, B = 1 evaluation. In a padded batch, role `last` is exact per row (same firing set as B = 1 on
  190 - 192 of 192 rows in the reviewer's test; the keys themselves move by |dr| <= 0.03); roles `last+answer` / `all`
  also evaluate PAD positions unless the hook is given the row lengths (`Hook.begin(lp, seq_len=...)`), and the
  `zscore` baseline includes PAD positions in its statistics (as the official hook does).
- The last prompt position is not the same token on the teacher-forced path and on the decode path for LLaVA answers
  that start with a digit (rule 3). The package reports it per row (`lp_shift`), scores either position
  (`lp="prompt"`) and returns every key of both paths (`answer="all"`); it does not make the two paths agree, and an
  edit written at the later token cannot fire under role `last` in a decode. `answer="all"` is exact for role `last`
  only; `DECODE_MARGIN = 0.05` (with `answer="both"`) is a protocol constant validated on the sampled complaints of
  test `b9`, not a guarantee.
- `response="lin"` reproduces the log WITHOUT the write round-off `E` (0.5 % / 0.06 % of the log in Frobenius norm
  on the two primary logs): it matches the deployed `W_E` forward to the round-off floor of the forward, not bitwise
  (test `b1_linear`). P10's primary configs kept `E` at firing positions; this package never does.
- Rebased values use log quantities only (factor form) and miss the round-off of the earlier writes (relative
  error <= 1.9 % / 0.5 % against the cached clean-model targets; `rebase_exact` needs the stored batch deltas and is
  not better against those targets, whose own batch composition differs).
- `witness_repair(mode="detector")` uses the normalised witness key with a cosine threshold: the log-style read-out
  vector of a new key needs the editor's protected-subspace projector, which the log does not store (pass it as
  `detector=` if available). A granted witness makes the edit a candidate; under `fire="top1"` another edit may still
  release there (`rec["edit_is_top1_on_witness"]`).
- `scope_cut` optimises one threshold per edit (no added detectors); its optimum is exact for the integer-scaled
  weights (`rec["scale"]`), i.e. exact for the stated objective when `lam * weight * scale` is integral.
- `null_repair` changes detectors only; thresholds calibrated before it refer to the old coordinates (recalibrate),
  and guards that are not detector keys (witnesses, B = 1 own keys) may stop firing (`rec["n_guards_lost"]`): the
  next calibration then lowers those thresholds until they fire again (`n_guards_not_firing_before`).
- Baselines are same-log analogues, not the published editors (keys, values and thresholds come from this log; see
  `baselines.py`). The `zscore` gate reaches Rel 15 % on the Qwen log (W_E 75 %; reviewer's 100-item test): on that
  log the edit's own position is rarely a z-score outlier, which is a property of that gate, not of the port.
- Rel under `response="repl"` is a keyed lookup by construction (the edit's own key releases its rebased value):
  report it as memorisation, next to T-Gen / I-Gen.
- Teacher-forced D1 under-counts free-running drift on firing rows (harness caveat, P10 verification item 8).
- The harness REEDIT correction is not reproducible between builds (rule 11): differences of 1 - 2 Rel points and of
  up to 8 of 490 HELD-OUT flags between two builds of the same 10 complaints are build noise, not method effects.
- Timings are load-dependent on the shared GPUs: the end-to-end overhead of role `last` (fp64 gate) was +2.0 % / +1.7 %
  in the reviewed run, +5.3 % (LLaVA) in the reviewer's re-run and +3.6 % / +3.6 % in the final run; the gate-only
  times (minimum of 10 runs: 0.60 ms / 0.94 ms per forward) are the more stable numbers.
- Not implemented: T-Loc / I-Loc on the vendor's padded B = 8 batches (use `harness.evaluate` for weights and
  adapters; for gates the hook supports per-row `lp` and `seq_len`, see `Hook.begin`), multi-module logs, sampling
  decoders.
- `rescope/_hcache` holds copies of the harness' baseline files and SYMLINKS to its large read-only caches (edit
  keys, PC bases, probe keys). The harness only reads them; if a link target were deleted by its owner the harness
  would recreate it through the link. Set `RESCOPE_CACHE` to another directory to relocate the private cache.

Limitations of the additions for the final experiments:

- Two-block / shared rule: the bound holds for the UNCAPPED rule, for queries exchangeable with the calibration queries and
  evaluated through the same forward path (teacher-forced keys calibrate teacher-forced rows; decode metrics need
  `eligible_keys(answer="all")`). Capped edits (guards) are outside it and are counted. `eta = +inf` (nothing but the guards
  fires) whenever `n2 < 1/alpha - 1`: with `n = 200` halves this is `alpha < 1/101`; with the block ratio 3:1 (`n2 = 50`) it
  is `alpha < 1/51`. The split costs data: block 1 only sets baselines, block 2 only sets `eta`.
- Bit-for-bit means: stored decision == defining comparison ON THE SAME computed coordinate. A coordinate changes in its last
  bits with the device and the batch shape; the query that defines `eta` (and an identical test query) then sits within
  one unit in the last place of the threshold (the decision band of `keycache.py`).
- `KeyCache.decide` and the sparse path cover gates that decide at the last prompt position (role `last`, fp64 gate;
  router for `decide`). Roles `last+answer` / `all`, the z-score gate, free generation and D2 always run the model.
  `near` sequences are not decided on the CPU by `SparseEvaluator` (they run through the hook); `KeyCache.decide` reports
  its own decision and the flag.
- `SparseEvaluator` trusts the cached pre-edit outcome of a sequence (`verify_cache` re-captures a sample). A cache built
  with another tokenizer / model revision would be wrong silently: the item is matched by content, not by revision
  (a cache of the other backbone or layer is refused; both backbones have keys of width 11008).
- `KeyCache` opens index and key files together and checks them against each other and `meta.json` (index md5, shapes;
  `verify=True`: md5 of the key files). A cache that is re-assembled with the SAME shapes and index but other key values
  is only caught by `verify=True`; a process that holds a cache open keeps reading the files it opened.
- DUAL is defined on generated answers only; `eq` compares the first line of the 48-token decode (or the 10-token answer),
  after `norm_answer`; targets that normalise to the empty string never match. Its teacher-forced rows do not exist.
- `tok_loc` / `T-Loc*` / `I-Loc*` are not the benchmark's T-Loc / I-Loc (other answer text, B = 1, valid rows): the stored
  values are reported next to them, with the signed difference.
- `stored_metrics`: edit metrics and benchmark locality are the testbed's vendor pass (padded batches of 8); B = 1 flags of
  this package differ on a few near-tie edits (test `d1_scale`: sampled edits at T = 2000).
- Lean handles read a batch delta from disk at every `H.A[b]` (harness rollbacks get slower, not different). For
  `T >= 500` the harness baseline / evaluate / evalcache and all-position edit keys are refused; the COMP protections
  that need every position of every edit sequence are therefore not available there without lifting the guard.
- `register_logs` registers a part-2 log as soon as its directory holds every file of the testbed layout, even before its
  builder has appended it to `registry.json` (`source = "part2-pending"`): check `register_logs()["logs"][name]`.

## Changes after the independent review (findings F1 - F8 of `review/`)

| finding | change (file: function) | test |
|---|---|---|
| F1 the last prompt position is another token in teacher forcing than in the decode (LLaVA, digit-leading answers); repairs on teacher-forced keys do not hold in a decode | `evaluate.py`: `Evaluator.lp_shift`, `lp="tf" \| "prompt"` in `tf` / `tf_rows` / `edits` / `probes` / `hard_negatives` / `capture` / `predict`, `decode_keys`, `eligible_keys(answer="both" \| "decode" \| "all")`, rows carry `lp`, `lp_shift`; `repair.py`: `DECODE_MARGIN` | `b9_paths` |
| F2 `add_guards` documented a non-firing key as harmless although every calibration lowers the threshold to it | `repair.py`: `add_guards(..., allow_nonfiring=False)` raises; `witness_repair` refuses a repair that leaves the witness silent; `calibrate.py` / `repair.scope_cut`: record field `n_guards_not_firing_before` | `a16_guard_contract`, `a10_witness_repair` |
| F3 a gate without `W0` ran on the deployed weight | `scoped.py`: `Hook.__enter__` raises unless `set_weight=False` | `a15_hook_guards` |
| F4 an `lp` vector shorter than the batch disabled the remaining rows | `scoped.py`: `ScopedLog.eligible`, `Hook._hook` (every gate); `baselines.py`: `DenseGate.firing` (router) | `a15_hook_guards` |
| F5 guards are the stored B = 10 keys; capped thresholds silence own B = 1 prompts; `exact_gate` misses Qwen edit 24 | `evaluate.py`: `Evaluator.own_keys` (guards from the B = 1 sequences, optionally every forward of a decode); documented for `exact_gate` | `b10_own` |
| F6 undocumented deviations of the baselines; REEDIT needs `chunk=1` | `baselines.py`: docstring (cosine role, GRACE checked against the paper and the official code, CoachHooK), `zscore_gate(kv="stale" \| "call")`; `evaluate.py`: `reedit` | `a8_baselines`, `b6_harness` |
| F7 the conformal guarantee is marginal over the calibration draw; `[:100]` is not a random subset | `splits.py`: `sample(items, n, seed)`; rule 6; the example draws with a seed | `a14_real_keys_operators` (spread over draws), `a12_real_logs` |
| F8 HELD-OUT-DISJOINT rows that share their question with a REPAIR row, duplicate rows | `splits.py`: pool `heldout_strict`, row fields `question_in_repair`, `query_id`, `n_same_query` | `a12_real_logs`, `b10_own` |
| style: top-1 ties, `agree=None`, missing hard-negative file | `scoped.py` docstring (lowest index); `evaluate.py` docstring; `load_hard_negatives` raises, `Evaluator.hn_items` raises | `a15_hook_guards`, `a12_real_logs`, `b10_own` |
| residual: PAD positions fire for the roles `last+answer` / `all` | `scoped.py`: `Hook.begin(lp, seq_len=...)` | `a15_hook_guards` |

Default behaviour is unchanged: with `lp="tf"` every per-row record of the regression pass is identical to the
reviewed version (last line of `tests/out/table.md`; the reviewed version's outputs are kept in `tests/out_prev/`).
The reviewer's independent naive re-implementation test (`review/t_naive.py`, copied unchanged to
`tests/review_rerun/`) gives on the final package the output it gave on the reviewed one (1440 models).
The GRACE paper and the official adaptor code consulted for `ball_gate` are in `docs/refs/grace/`.

## Tests

```
GQ=$RESCOPE_ROOT/02_research/tools/gpuq/gq         # the shared GPU queue (README.txt there)
python tests/run_tests.py cpu                               # synthetic logs + CPU checks on the real logs
$GQ run --mem 10800 -- python tests/run_tests.py gpu --log TB-Q100-ae-l2_3 --mem_gb 10.1
$GQ run --mem 17000 -- python tests/run_tests.py gpu --log TB-L100-ae-l2_3 --mem_gb 16.1
$GQ run --mem 10500 -- python tests/run_tests.py scale --mem_gb 9.8       # T = 2000 log in a fresh process (peak RAM)
python tests/run_tests.py table                             # pass / fail + regression tables from tests/out/*.json
```

Peak GPU memory of the suite: 9.5 GB (Qwen) / 15.0 GB (LLaVA); the peak is the z* optimisation of the REEDIT correction
in `b6_harness` (backward passes, one complaint at a time). GPU jobs go through the shared queue (`gq run --mem MiB --`;
the command sees its card as `cuda:0`); `--mem_gb` caps the process itself. `tests/gpu_launch.sh` is superseded.
Test files: `t_core.py`, `t_ops.py` (a1 - a16), `t_cal.py` (a17 - a20: calibration), `t_final.py` (a21, a22: registries,
key cache), `t_gpu.py` (b1 - b10, c), `t_gpu2.py` (b11 - b14, d1). The key cache used by `a22` / `b13` / `b14` is the one
under `10_final/cache/<backbone>/` (or `RESCOPE_KEYCACHE_ROOT`); without it they test the local store and say so.
The delivered table was produced with the complete caches of both backbones in their final location. If a cache is
still being built, `python tests/extra/snapshot_cache.py` makes a fixed state under `tests/out/keycache_snapshot/` (a
symbolic link to a final cache, a copy of the newest partial assembly otherwise; verified with md5) and the tests are run
with `RESCOPE_KEYCACHE_ROOT` pointing there; every table row that reads a cache says which one it read.
The whole GPU part, as launched for the delivered table: `nohup tests/extra/final_runs_Q.sh &`, `nohup tests/extra/final_runs_L.sh &`
(`tests/extra/drop_stale.py` first removes GPU results of another code version, so the resumable commands redo exactly those).
Extra scripts (`tests/extra/`, rendered in the table's section "Extra checks"): `load_families.py` (one log of every
registry family through `load(reuse=)` and a B = 1 sample), `load_peak.py <log> --lean 0|1` (peak RAM of one load in a
fresh process), `cache_vs_p11.py` (CPU: cached keys against the keys P11b captured independently),
`regression_check.py` (regression table and b345 rows against the outputs before the additions, kept in `tests/out_v1/`).
