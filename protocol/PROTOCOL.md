```text
PREFACE TO THE RELEASED COPY. This block is NOT part of the protocol; the protocol starts at the line "# Final experiments ...".

How to read this file. Sections 0 to 5 are the protocol as frozen on 2026-11-10, before the final runs. Section 6
("Amendments") is append-only: every later change is one dated entry, in the order in which it was written, with its
time stamp (UTC, clock of the machine; one entry explains the stamps of the first night) and with the state of the runs
at that moment ("before any final run", "after the run and the independent verification of ..."). Where an amendment and
an earlier text differ, the amendment holds; an amendment that was withdrawn stays in the file together with its
withdrawal (R18, registered 2026-11-11 07:50 UTC and withdrawn in full at 08:03 UTC, before anything of it was built).
The experiments that were run are those of the protocol after the amendments R19 and R21 to R25; the README of the
repository maps each of them to its directory, command and table.

Who wrote it. The protocol and every amendment were written by the AI assistant (Claude, Anthropic) that coordinated
the work, under the supervision of the first author; decisions of the first author are recorded as such in the entries
("decision of the first author", "the author decides"). "Independent verification", "independent audit", "review" and
"referee" in this file denote passes by separate instances of the AI assistant (and, for "Review round 1", a simulated
review by language models), each with its own code; they are not checks by other people. See the README, section
"Use of AI tools".

Words for roles. "Principal investigator", "PI" and "coordinator" in this file and in the released records denote the
coordinating instance of the AI assistant. "Prover", "referee", "reviewer" (of the driver or of the theory), "verifier",
"auditor" and "judge" denote separate instances of the assistant; pronouns in the records refer to these instances.
"External referee" and "external critic" denote a second language model. "The first author" is the human who supervised
the work; the records attribute to the first author the decisions that the first author took.

Registration. Protocol frozen before the final runs; amendments carry the time stamps of the authors' machine; no external
registry was used; "registered" in the paper and in the records means written into this protocol before the run.

Redactions. This copy differs from the working file only by marked replacements, made in place, and by the replaced tag in
the names of the logs (see "Log names"); no sentence was rewritten, reordered or dropped, and the line structure is
unchanged (line n of the protocol is line n + 45 here):
  [redacted: describes a related submission under double-blind review]   a passage about the editor configuration that
        wrote the vision-language logs, or about the paper that describes it (a related submission of the authors that
        is under review; the logs are released as data: README, section "What can be reproduced")
  [redacted: local path], [redacted: account name], [redacted: related project]   locations and names of the machine; names of other projects of the authors
Log names. The protocol uses the directory names of the working tree (TB-..., TX-...); LOG_NAMES.json at the root of
the repository maps them to the names of the paper (Q-A100-r3, ..., G-R100, ...).
One tag in the working names of the logs of the first editor family was replaced by 'ae' in every released file (file names
and contents); the replacement is mechanical and changes nothing else. The paper's names are those of LOG_NAMES.json.
Paths such as 10_final/e1_main/out/... are paths of the working tree; rescope_paths.py at the root of the repository
lists where each of them is in this repository (10_final/<experiment> = experiments/<experiment>, 10_final/common =
driver, 09_testbed_scale/text_public = text_logs, 07_repair_v3/harness and 08_rescope/p12_linear_streams =
linear_baselines).
```

# Final experiments of the v4 paper (ReScope): protocol, frozen before the runs

Frozen by the principal investigator on 2026-11-10, after P10 / P11 / P12 (exploratory, T = 100 logs only) and
before any run on the scale logs, the MEMIT log, the MMKE log or any text log. Target: PVLDB 2027, deadline
2026-12-01. Every experiment below is implemented against the library `02_research/code/rescope/` and the shared
harness `07_repair_v3/harness`; no other code path may produce a number that enters the paper.

Changes to this file after 2026-11-10 must be appended at the end with a date and a reason. Results that
contradict an expectation stated here are reported as they are.

## 0. Method under test (fixed; no tuning on any evaluation data)

**ReScope(alpha)** = scoped replay of the edit log.
- role: only the last prompt position of a sequence is eligible (the position type at which the log wrote every edit);
- gate: normalised solver coordinate r_i(k) = <W_i, k> / s_i, s_i = <W_i, k_i>; top-1 edit j = argmax_i r_i(k);
  the edit fires iff r_j(k) >= eta;
- response: REPL: module output = W_0 k + V_j, with the rebased value V_j = U_j + sum over edits l of strictly
  earlier batches of U_l <W_l, k_j> (computed from the log alone);
- calibration: eta = split-conformal quantile of the scores max_i r_i(k_last) of n benign calibration queries at
  miscoverage alpha (label-free: only keys are needed). Default alpha = 0.05, n = 200.
- Every non-firing sequence is the pre-edit model, bit for bit.

Complaint-driven operators (applied on top, never tuned):
least repair (raise tau_i just above a reported complaint's coordinate), null repair (remove reported benign keys
from the detectors, own coordinate kept), witness repair (why-not complaint: add a detector for the edit).

Level-1 variant (**Rescope-WHERE**, the pure predicate repair studied by the theory): role = all positions,
response = the original log terms c_i(k) U_i of the firing edits, dense round-off term E kept at firing positions,
so that thresholds -infinity give the deployed model exactly and +infinity give W_0. Thresholds: log-only rule
(5 % above the largest coordinate on the other logged sequences) and least repair from complaints.

Ablations (reported, not selected): role in {last, last+answer, all}; response in {REPL, GAIN, LIN};
rebase on / off; top-1 / all; gate feature in {normalised coordinate, raw coordinate, raw cosine};
calibration in {conformal, log-only quantile q = 1, per-edit running maxima}.

## 1. Data, pools, splits

- Logs. Primary (method development happened here): `TB-Q100-ae-l2_3`, `TB-L100-ae-l2_3`.
  Working points of the same streams: `TB-Q100-ae`, `TB-Q100-ae-l2_1`, `TB-L100-ae`.
  **Unseen logs (never used for any design decision)**: `TB-Q{200,500,1000,2000}-ae-l2_3`,
  `TB-L{200,500,1000}-ae-l2_3`, `TB-Q{100,500}-memit`, `TB-Q{100,500}-mmke-l2_3`, and any text log. The method,
  alpha and n are applied to them unchanged.
- Probe pools (harness split by item, seed 20261109): REPAIR pool = calibration rows and reported complaints;
  **HELD-OUT-DISJOINT** = held-out rows whose query does not occur in the REPAIR pool = the only pool on which
  locality numbers are reported (the full HELD-OUT pool contains 25 rows that duplicate REPAIR queries).
- Edits: Rel, T-Gen (text rephrase), I-Gen (image rephrase) of every logged edit; teacher-forced exact, B = 1.
  Nothing in ReScope(alpha) is fitted on edit prompts or rephrases, so all T edits are test edits. Whenever an
  operator IS fitted on witnesses (witness repair, SCOPE-CUT), fit and test witnesses are disjoint by type
  (fit T-Gen -> test I-Gen, and the reverse) and by edit parity, as in P11b.
- Hard negatives (`08_rescope/p11_retrofit/out/hard_negatives`): HN-I (edit image, unrelated question), HN-Q
  (edit question, unrelated image), and CROSS (image of edit a, question of edit b; to be added by E7). They are
  outside the conformal guarantee (distribution shift) and are reported separately.
- y0: decoder-v2 multi-token answers only. D1 = y0 not reproduced under teacher forcing; D2 = pre-correct row whose
  48-token answer loses the reference; free generation = decoder-v2 greedy.

## 2. Compared methods

| family | method | evidence it uses |
|---|---|---|
| deployed | LINEAR = W_E | - |
| editor frontier | the same stream at ridge l2 = 10, 3, 1 | - |
| linear repair | COMP-B (joint closed-form compensation, K_ans protected), REEDIT-B, RB-UNION, RB-TOT (harness; options of P12) | n reported complaints |
| post-hoc gate | COSINE (NeuralDB-style: top-1 raw cosine to the stored keys), BALL (GRACE-style distance), ROUTER (WISE-style activation norm), ZSCORE (CoachHooK-style), EXACT (cosine >= 0.999) | published default threshold AND the same conformal calibration as ReScope |
| ours | ReScope(alpha); Rescope-WHERE; + complaint operators | n benign queries / n complaints |

Fairness rules. (i) Every post-hoc gate receives the retrofit that the log makes possible (role typing where its
definition allows it, rebased values); its published form is reported as well. (ii) Every gate with a scalar
score is calibrated by the same conformal rule at the same alpha and n, on the same calibration rows.
(iii) Linear repairs are run at their best: all reported complaints, the options of P12, the z* oracle ladder
enabled.

## 3. Experiments

**E1 Main comparison at a matched false-firing budget** (primary logs + working points).
alpha in {0.01, 0.02, 0.05, 0.10, 0.20}, n = 200, 20 random calibration draws from the REPAIR pool.
Report on HELD-OUT-DISJOINT: firing rate, D1 rate, D2 count, T-Loc, I-Loc; on all edits: Rel / T-Gen / I-Gen exact
and token-level, share of rephrases in scope; on hard negatives: firing, D1, wrong-target rate. Free generation
and D2 for LINEAR, ReScope(0.05), COSINE(0.05), COMP-B(all).
*Primary endpoint*: at alpha = 0.05, (T-Gen + I-Gen) / 2 exact of ReScope versus each calibrated baseline gate,
paired bootstrap over edits, per log. *Secondary*: Rel exact; hard-negative D1 (expected to favour COSINE);
D1 rate versus LINEAR and versus the best linear repair.

**E2 Evidence curves** (primary logs). Held-out D1 rate, Rel / T-Gen / I-Gen versus the amount of evidence
n in {1, 2, 5, 10, 20, 50, 100, all}: (a) linear repairs of n reported complaints (P12, re-scored on
HELD-OUT-DISJOINT through the library's evaluation); (b) Rescope-WHERE from the deployed model with least repair
of the same n complaints; (c) ReScope calibrated on n benign queries; (d) ReScope(0.05) followed by least / null
repair of the complaints it still receives; (e) hard-negative stream: n reported hard negatives, null repair,
tested on held-out hard negatives of the same and of other edits; (f) why-not stream: witness repair, cross-type.
3 seeds for n <= 20, 1 seed above (as P12); order independence of least repair is checked on every stream.

**E3 Scale** (unseen logs). T in {100, 200, 500, 1000, 2000} (Qwen), {100, 200, 500, 1000} (LLaVA):
LINEAR, ReScope(0.05), COSINE(0.05), COSINE(default), Rescope-WHERE(log-only). Edit metrics on all T edits (and on
the first 100 edits, to separate the effect of T from the effect of the item set), locality metrics, D2, own keys
silenced, top-1 confusions among logged keys, rebase error, gate latency and storage versus T.

**E4 Other editors and data** (unseen logs). The E1 row set at alpha = 0.05 on the MEMIT-style log, the MMKE-Bench
log and, if built, a text-only log (CounterFact / ZsRE style: paraphrase prompts are witnesses, neighbourhood
prompts are near-scope negatives).

**E5 Checks of the theory on real logs.** (a) rebase lemma: error of V with and without rebasing; exact target
attainment of scoped replay versus the linear log's attainment, as a function of T; (b) lineage width: number of
firing edits per query under each rule; (c) conformal validity: empirical false-firing rate versus alpha for
n in {20, 50, 100, 200, 500}, 500 draws, with the per-edit running-maxima rule and its T / (n + 1) bound, on a
large label-free benign pool (keys only; to be captured); crossover between one shared scale and per-edit
thresholds as n grows; (d) least repair: uniqueness / order independence / no regression on real streams;
(e) SCOPE-CUT: exactness and run time (done in P11b; re-reported), and its out-of-sample comparison with the
shared scale as the evidence grows (the P11b result that it does not win at n of about 500 rows is reported).

**E6 Ablations** (primary logs, alpha = 0.05 where a calibration applies): the list of Section 0.

**E7 Limits.** Near-scope negatives (HN-I, HN-Q, CROSS) for every E1 method; behaviour of the complaint operators
on them (E2e); portability-style queries if the benchmark provides them; LLaVA is expected to be the hard case.

**E8 Cost.** Per-query latency overhead and storage of each gate versus T (clean GPU, methods interleaved);
repair time per complaint for every repair method (pass counts and wall-clock); calibration time.

## 4. Statistics and reporting

- Intervals: percentile bootstrap, 2000 resamples; unit = probe item (uid) for probe metrics, edit for edit
  metrics; calibration-draw variability reported separately (mean and 5th-95th percentile over draws).
- Paired comparisons: paired bootstrap of per-edit / per-row differences; exact McNemar for binary outcomes.
- Every table and figure is rendered by a script from JSON written by the experiment code; per-row flags are kept
  so that any number can be recomputed. No hand-typed number anywhere, including the paper (LaTeX macros are
  generated).
- Each experiment is followed by an independent verification agent that recomputes the headline cells with its
  own code.
- Negative and mixed outcomes stay in the paper: SCOPE-CUT versus the shared scale, hard negatives on LLaVA,
  the baselines' advantages wherever they occur.

## 5. Boundaries

- No analysis [redacted: describes a related submission under double-blind review]
  [redacted: describes a related submission under double-blind review]: the logs are a testbed built with a fixed, existing editor configuration.
- No [redacted: related project] content.

## 6. Amendments

- 2026-11-10, before any final run. **Target cycle fixed by the author: full paper 2026-12-01 17:00 PT (abstract
  2026-11-25), VLDB 2027 research track.** Consequence for this protocol: none of the experiments is dropped;
  E1, E2, E3 and E5 are the must-haves, E4 (other editor / other data) and the text-only log are run as far as
  the GPUs allow and are reported with whatever logs exist on 2026-11-22 (results freeze).
- 2026-11-10, before any final run. **Added baseline DUAL (answer match)** to the post-hoc family of E1 / E3:
  run the pre-edit and the deployed model on the query and return the deployed model's answer only if it equals
  the target of a logged edit, otherwise the pre-edit answer. It needs the list of edit targets (not only the
  factors) and two forward passes per query; it is the simplest way to obtain locality on unrelated queries and
  bounds what a gate must add (reliability and generalisation beyond the deployed model's).
- 2026-11-10. Theory status used by E5: parts A (semantics) and B1 (minimum cut, frontier) are proved with
  checks and are being refereed; part C (conformal bounds) is being written; nothing in E5 depends on the
  outcome of the referees.
- 2026-11-10 17:10 UTC, before any run on an unseen log. **Default calibration of ReScope changed to the
  two-block (split) per-edit conformal rule.** Reason: the referee of theory part C showed (stored keys of the
  two development logs and of TB-Q500-ae, alpha = 5 %) that it keeps the distribution-free bound of the shared
  scale while adapting to each edit: rows firing 5.1 / 2.8 / 3.3 % with text-rephrase coverage 67 / 47 / 37 %,
  against 4.7 / 3.6 / 4.3 % and 64 / 35 / 29 % for the shared scale. Definition (normalised coordinates
  r_i = <W_i, k> / s_i at the eligible position): the n benign calibration queries are split at random into
  two blocks of n/2; block 1 gives the per-edit baseline b_i = largest r_i over its queries; block 2 gives
  eta = conformal quantile at level alpha of the scores max_i (r_i - b_i) (strict gate, exact rank, +infinity
  when the rank exceeds the block size); edit i fires iff it is the top-1 edit by r_i - b_i and
  r_i - b_i > eta. Thresholds are stored in the floating-point-safe form recommended by the referee. An edit
  whose own solver key would be silenced keeps firing on its own key and is counted. Default n = 200, alpha =
  0.05. The shared-scale rule of Section 0 becomes an ablation of E6, next to the raw-coordinate variant
  (margin not divided by s_i) and block ratios 1:3 and 3:1. No other part of the method changes.
- 2026-11-10 17:10 UTC. Reading of the P11 verdicts that enters the paper (verified by independent agents):
  the retrofit strictly improves on the deployed log on both development logs but misses the pre-registered
  text-rephrase bound on LLaVA (PARTIAL); the comparison with the cosine gate under the pre-registered
  selection rule is NO ADVANTAGE (different operating points; at matched probe firing the coordinate gate
  covers far more rephrases, the cosine gate is safer on near-scope negatives); exact per-edit threshold
  optimisation by minimum cut does not beat one shared threshold out of sample at the available sample sizes.
  E1 compares all gates at matched, conformally calibrated false-firing budgets for this reason.
- 2026-11-10 17:40 UTC, before any run on an unseen log. **Log families and the boundary with the author's
  earlier paper.** The first scale-up built AlphaEdit-style logs with [redacted: describes a related submission under double-blind review]
  up to T = 2000 (Qwen) and T = 1000 (LLaVA), and MEMIT-style logs [redacted: describes a related submission under double-blind review]. The
  earlier paper [redacted: describes a related submission under double-blind review]
  [redacted: describes a related submission under double-blind review]. Therefore, unless the author decides otherwise:
  (i) AlphaEdit-style logs with the [redacted: describes a related submission under double-blind review] enter the paper only for T <= 500 (development logs,
  working points, T = 200 and 500 snapshots); (ii) scale beyond T = 500 is reported on a MEMIT-style family in
  the public default configuration (Wikipedia-text statistics, lam_hat = 100 fixed in advance, same edit
  streams: TB-Q{100..2000}-memitw, TB-L{100..1000}-memitw, being built); (iii) no table or figure [redacted: describes a related submission under double-blind review]
  [redacted: describes a related submission under double-blind review], and the legacy log [redacted: describes a related submission under double-blind review] is not used; (iv) the T = 1000 / 2000 logs
  with the [redacted: describes a related submission under double-blind review] are evaluated internally as a consistency check only. All of these logs are
  unseen for the purposes of Section 1.
- 2026-11-10 17:40 UTC. Testbed facts that the experiments must respect (from the build and its audit):
  new logs load only after `import scale_registry` (09_testbed_scale/code); TB-L1000-ae-l2_3 has two probe
  rows with non-finite logits at the deployed weight (excluded, counted); the MMKE-Bench logs have Rel exact 0
  and irreproducible targets and are reported, if at all, as a separate stress test; the harness baseline
  needs 34 GB of RAM at T = 200, so runs at T >= 500 must not call it and must keep only last-position keys in
  memory; D1 saturates on long linear logs (3 to 58 valid held-out rows are not complaints), so locality at
  scale is reported as complaint rate, not as a ratio of fixed to lost rows.
- 2026-11-10 22:10 UTC, before any final run. **Clarifications that follow from the reviewed library and from
  the testbed audit; none changes the method.**
  (1) *Rescope-WHERE without the round-off term.* The library replays a log from its factors only. Rescope-WHERE
  is therefore W_0 k + sum over firing edits of <W_i, k> U_i, without the dense write round-off term E; with
  all thresholds at -infinity it equals the deployed model up to that round-off (the number of probe and edit
  flags that differ from the deployed weight is reported for every log).
  (2) *Evidence curves at small n (E2).* Conformal calibration cannot let any edit fire below n = 1/alpha - 1
  benign queries (two-block: n_2 >= 1/alpha - 1). The ReScope curve of E2(c) therefore starts from the log-only
  rule at n = 0 (thresholds = the largest pooled normalised cross-talk among the log's own solver keys, the
  rule called POOL(q = 1) in P11), applies the least repair to the queries received while n < 40, and switches
  to the two-block rule at alpha = 0.05 from n = 40 on. The Rescope-WHERE curve of E2(b) starts from the
  deployed log (all thresholds -infinity) and applies the least repair of the same n complaints, paired with
  the complaint sets of the linear repairs (P12 records).
  (3) *Recalibration after detector repairs.* Operators that change detectors (null repair, witness repair) use
  reported complaints only; the calibration sample is kept disjoint from the complaints, and the two-block rule
  is re-run after every batch of such operators, so that the false-firing bound holds again at each step.
  (4) *Additional ablations of E6* (reported, not selected): the block-1 baseline b_i as the maximum (default)
  or as the 0.98-quantile of the block-1 coordinates; the source of the calibration queries (REPAIR-pool
  probes, the default, or the label-free benign pool of 10_final/cache, a shifted source); contamination of
  the calibration sample by in-scope queries (0, 1, 5, 20 percent of rephrase prompts of half of the edits),
  reporting false firing on benign held-out queries and coverage on the rephrases of the other edits.
  (5) *Probes that share a question string with a logged edit* (677 of the 2000 edits of the long Qwen stream
  reuse a probe question) are near-scope queries for that log. E3 reports false firing and D1 separately for
  held-out probes that share and that do not share a question string with an edit of the evaluated log; the
  calibration sample is drawn from all REPAIR-pool probes, as before.
  (6) *DUAL at scale* is computed on the first 100 edits, 200 sampled later edits and all held-out probes (two
  generations per query make the full pass too expensive at T = 2000).
- 2026-11-10 22:10 UTC. P12 (linear repair streams) is complete on Qwen (3 seeds, D2) and nearly complete on
  LLaVA (seed 2 at n = 10, 20 partly missing; no D2); its records are the linear side of E2(a). Held-out D1
  on rows without duplicated queries, mean over seeds: COMP-B never below 48.8 % (Qwen, unrepaired 54.2 %) and
  59.7 % (LLaVA, 62.3 %); REEDIT-B above the unrepaired rate at every n; rollback below 40 % only with Rel
  exact below 40.
- 2026-11-10 23:55 UTC, before any final run. **The text-only logs are dropped from E4.** Two families were
  built on the same 500 CounterFact edits and audited: with the project's AlphaEdit configuration
  (TB-Q{100,500}-cf-l2_3) and with the public MEMIT-style configuration (TB-Q{100,500}-cfw). Both are degenerate
  deployed logs: locality on unrelated text probes is about 30 after the first ten edits and 99 % of the probes
  are complaints at T = 100, [redacted: describes a related submission under double-blind review]. The principal investigator's earlier reading (the
  [redacted: describes a related submission under double-blind review]) is therefore not supported; the cause was not
  investigated (candidate: this pipeline writes at the last prompt token, which for a chat-templated text
  prompt is a template token shared by all prompts; text editors write at the subject token and gate at every
  position). No number of these logs enters the paper and no method decision was or will be based on them. The
  paper states as a limitation that the evaluation covers vision-language edit logs written at the last
  prompt position.
- 2026-11-11 00:55 UTC, after the sanity comparison on the two development logs (10_final/e0_sanity/out/
  table.txt) and before any final run. **Nothing in the method changes.** The sanity numbers at alpha = 0.05,
  n = 200 (default method M1 / shared scale M2 / cosine gate with the same two calibrations M3, M4):
  held-out false firing 4.65 / 4.63 / 4.72 / 3.89 % (Qwen) and 3.09 / 3.39 / 4.91 / 3.07 % (LLaVA), all below
  the bound; coverage of text rephrases 62 / 60 / 12 / 2 % and 43 / 33 / 13 / 0 %; firing on near-scope
  negatives is higher for the solver coordinate than for the cosine (HN-I 38 / 29 / 11 / 4 % on Qwen). The
  two-block default covers more text rephrases on LLaVA (+11 points) at a larger spread over calibration draws
  and more firing on near-scope negatives; it stays the default as registered, the shared scale stays in
  every table next to it.
  Two descriptive additions follow from reading these numbers; they select nothing:
  (E1b) *Frontier tracing.* For each gate feature (normalised solver coordinate, raw cosine), one global
  threshold is swept over a fine grid and, at the gate level, the held-out false-firing rate, the coverage of
  text and image rephrases and the firing rate on each kind of near-scope negative are recorded (key cache,
  no model). Behaviour is measured at the calibrated points only. The figure shows the calibrated points on
  these curves.
  (E1c) *Matched-coverage read-out.* Because calibrated gates sit at different coverage, the comparison of
  near-scope firing between two gate features is also reported at matched coverage of rephrases (the threshold
  of each feature that reaches 25, 50 and 75 % coverage of the union of text and image rephrases, where
  reachable), next to the matched-budget comparison of the primary endpoint.
- 2026-11-11 00:55 UTC. A scout is asked whether a third backbone or a further short-answer benchmark can be
  added with the existing pipeline. Any log it builds is an unseen log in the sense of Section 1 and is
  evaluated with the frozen row set of E1 at alpha = 0.05 only; its working point is fixed by a rule stated
  before the build (the largest ridge in {10, 3, 1} with Rel exact >= 70 at T = 100; otherwise ridge 1).
- 2026-11-11 01:35 UTC, before any run on an unseen log and before the variant was run on any log.
  **Registered variant ReScope-N (`rescope_null`): detectors cleared of benign directions before calibration.**
  Motivation, stated before the fact: the default method takes its detectors from the log, so their quality
  depends on the statistics the editor used; the unseen MEMIT-style logs were written with other statistics
  than the development logs, and near-scope firing is the known weak point. Definition: a seeded draw of
  n_A = 100 benign queries from the REPAIR pool, disjoint from the n = 200 calibration queries; every detector
  is replaced by its minimum-change version whose coordinate is zero on the last-prompt keys of these queries
  while its coordinate on the edit's own key is kept (the null repair of Section 0, ridge 0); then the default
  two-block calibration runs unchanged on the calibration queries. The false-firing bound holds as before,
  because the detectors depend on a sample that is separate from the calibration and test queries. The variant
  is reported next to the default method on every log (development and unseen); no choice between the two is
  made on any unseen log, and the default method of Section 0 remains the default.
- 2026-11-11 01:35 UTC. Real-log statistics of the linear side were re-measured on the development logs
  (10_final/e5_linear, verification running). They replace the older numbers in the paper. Reading: the
  logs are well conditioned at a complaint key (ratio of the extreme eigenvalues of the scaled Gram matrix
  57.5 and 18.1 at one key; 10.1 and 5.0 over the stacked answer keys), no complaint is cancellation-free
  (half of the pairs of contributions have a negative inner product), greedy rollback is within a few
  percent of the certified optimum, and the optimum itself is large: about 70 of 100 edits at tolerance 0.1
  and 15 to 18 at tolerance 0.5. The paper says so: on real logs rollback is not hard to compute, it is
  expensive. The capacity law is exact on the Qwen development log: 37 of 100 whole request sequences can be
  protected, as predicted with the shared-prefix correction (35 without it).
- 2026-11-11 01:50 UTC, before any final run. **Independence of the recalibration sample (correction of the
  amendment of 2026-11-10 22:10, item 3).** The false-firing bound needs detectors and reference thresholds
  that do not depend on the calibration queries. Complaints are selected by thresholds that were fitted on
  the first calibration sample, so detectors repaired from these complaints depend on it; keeping the samples
  disjoint is not enough. In every experiment with detector repairs (E2 d-f) three disjoint seeded subsets of
  the REPAIR pool are used: C0 calibrates the starting thresholds, S is the stream from which wrong-answer
  complaints arise (for E2 e-f the complaints are near-scope negatives and rephrases, and S is not needed),
  and C1, never looked at before, recalibrates after the detector repairs. The bound is claimed for the
  thresholds calibrated on C1 only. Sizes: |C0| = |C1| = 160 and |S| = 160 for E2 d; |C0| = |C1| = 200 for
  E2 e-f.
- 2026-11-11 01:50 UTC. **Null repair on near-scope complaints (E2 e).** Removing the key of a near-scope
  negative from a detector whose own key is close to it can inflate the detector (the paper's method section
  says so). E2 e therefore reports, without selecting between them, the least repair alone, the null repair at
  ridge 0, and the null repair at the single ridge value 0.1 (the library's regularised form), each followed
  by recalibration on C1, with the number of capped edits and the norm inflation of the detectors.
- Note on time stamps (written 2026-11-11 01:29 UTC by the clock of the machine). The amendments above that
  carry the stamps "2026-11-11 00:55", "01:35" and "01:50" were stamped by estimate; they were appended between
  00:53 and 01:27 UTC. The order of the amendments and the fact that all of them precede every final run are
  not affected; the modification history of this file is authoritative.
- 2026-11-11 03:36 UTC. **Correction of the reading of the linear real-log statistics** (amendment stamped 01:35; the
  measurement was verified independently: 10_final/e5_linear/verify/verdicts.json, no numerical discrepancy).
  (1) "Well conditioned" is withdrawn. At ONE complaint key the ratio of the extreme eigenvalues of the scaled
  Gram matrix is a constant of the log (57.5 and 18.1; it is the cosine matrix of the write vectors), not a
  median over complaints, and the tolerance factor of the conditioning proposition is then 0.13 and 0.24: the
  proposition is loose there. Over the stacked answer keys the medians are 10.1 and 5.0. (2) What the data
  support: greedy rollback is close to the certified optimum (median ratio at most 1.08 where certified, 1.02
  at tolerance 0.1), and the optimum is large (about 70 of 100 edits at tolerance 0.1; 15 to 18 at 0.5); no
  complaint is cancellation-free. The paper states: on these logs minimum rollback is easy to approximate and
  expensive. (3) The lower bounds 59 and 62 quoted for tolerance 0.1 come from the strong form of the
  proposition (the count function mu of the technical report); the compressed form printed in Sec. 3 yields 8
  and 30 only. The printed statement is to be restored to the strong form before these numbers are quoted.
  (4) The capacity figure of the Qwen log (37 protectable request sequences, as predicted) uses a prefix of
  15 keys that is shared approximately, not bit for bit (the plain law predicts 35); the text must say so.
- 2026-11-11 04:26 UTC, before any run on these logs (their independent audit is still running; a log that fails the audit
  is dropped). **Two further families of unseen logs, built by a scout with rules fixed in advance**
  (09_testbed_scale/out_part3/part3_table.txt).
  (a) *One-hop family* TB-Q{100,500}-onehop-l2_3: Qwen2.5-VL-3B, the configuration and ridge of the Qwen
  development log, edits taken from the one-hop (portability) split of VLKEB; every edit comes with one
  one-hop question about its image whose answer changes under the edited fact. The frozen row set of E3 is
  run unchanged (these logs have their own probes and no near-scope negatives or benign pool; calibration
  queries are drawn from the REPAIR pool of their probes). In addition, as part of E7 (limits), *portability*
  is measured for the deployed log, the pre-edit model, rescope, rescope_shared, rescope_null, the two
  calibrated cosine gates and dual: (i) teacher-forced exact and token-level match of the one-hop answer (the
  testbed's stored metric: 3.0 % exact for the deployed log at T = 100, 0.0 % before editing); (ii) the share
  of one-hop questions on which an edit is released, and on which the edit of the same image is released;
  (iii) from generated answers: the share that equals the one-hop answer, the edit's own target (wrong-target
  rate), and the pre-edit answer. Expectation, stated before the run: a scoped edit releases a stored value
  and does not propagate to dependent facts, so its portability cannot exceed the deployed log's, and a
  release on a one-hop question yields the edit's target; the paper reports this as a limit whatever the
  numbers are.
  (b) *Third backbone, failed installation* TB-B{100,200,500}-ae: BLIP2-OPT-2.7B, layer 12, the project's
  AlphaEdit rule; the working-point rule fixed in advance (largest ridge in {10, 3, 1} with Rel exact >= 70 at
  T = 100, otherwise ridge 1) ended in its fallback, ridge 1: the solver attains 3.6 % of the residuals and
  the deployed log has Rel exact 2.0 % with 63 % of the probes changed. These logs are reported as a stress
  test in the table of the MMKE-Bench logs, not in the main tables: a log whose edits were written but did
  not take effect. The frozen row set of E3 is run unchanged. Because the solver's attainment is the scale
  s_i of the normalised coordinate, small s_i may make the default gate noisy on these logs; whatever happens
  is reported.
- 2026-11-11 04:29 UTC, before any final run. **E9: maintenance of a scoped log (revocation and lineage); development
  logs.** Motivation: the frame rule and the lineage statement of Sec. 3 say that, once edits are scoped,
  deleting an edit changes only the queries on which it is released, and that restoring a complaint needs only
  the edits released on it; in a linear log every deletion moves every key. This is measured.
  (a) *Selective revocation.* alpha = 0.05; default method with calibration draws 0 to 2; revocation sets R of
  m in {1, 5, 10, 20, 50} edits, 5 seeded sets per m, the same sets for both logs of a comparison.
  Scoped log: the edits of R are masked (threshold plus infinity), all other thresholds unchanged (raise-only,
  so the false-firing bound still holds). Linear log: the rank-one terms of R are subtracted from the
  deployed weight (the harness's rollback convention). Reported for both: Rel / T-Gen / I-Gen exact of the
  revoked edits (the pre-edit level is the target) and of the kept edits; number of kept-edit sequences and
  of heldout_disjoint probe rows whose flags change, and, for the scoped log, whose logits are not bit for bit
  those before the revocation; violations of the frame rule (a sequence on which no revoked edit was released
  and whose logits changed): expected 0.
  (b) *Lineage.* For every heldout_disjoint probe and every near-scope negative on which the calibrated
  default method releases an edit: masking exactly that edit; share of rows that then equal the pre-edit model
  bit for bit, share on which another edit is released instead (lineage width at least 2), and the number of
  masked edits needed until nothing is released. Next to it, from existing records, the number of edits a
  rollback of the linear log needs for one complaint (behavioural greedy of P12 / rq1; layer-level optimum of
  E5).
  (c) *Retention under continued editing* is read from E3: Rel / T-Gen / I-Gen exact of the first 100 edits
  of a stream as T grows, for the deployed log and for the scoped log.
- 2026-11-11 04:40 UTC, before any log of this family exists. **E10: public text logs (generality beyond vision-language
  logs).** Motivation: every log of E1 to E9 was written by this project's testbed on vision-language prompts,
  at the last prompt position. E10 tests the log abstraction in the canonical text setting, on logs written by a
  public reference implementation: ROME (one module, one rank-one update per request, requests applied
  sequentially; the implementation returns the two factors of every update, which are the log) on GPT-2 XL and,
  if a card can be spared, GPT-J, with CounterFact requests. That long sequential ROME logs degrade the model is
  documented in the literature; whether a scoped replay of the same log removes the degradation is the question.
  *Logs.* Public code and public hyper-parameters, unchanged; second-moment statistics as published by the
  authors if downloadable, otherwise recomputed with the public script (stated in the paper). Reported logs:
  requests in the order of the data set from record 0, T in {100, 500, 1000, 2000} (one stream; the shorter logs
  are prefixes). Development log: T = 100 on a disjoint slice of records (from record 15000), the only log of the
  family on which anything may be looked at before the row set runs.
  *Conditions of use (independent audit, before any method is run on a reported log).* (1) The deployed weight
  equals the pre-edit weight plus the logged rank-one terms up to float round-off. (2) Single, non-sequential
  edits on the first 200 records reproduce the published efficacy / paraphrase / neighbourhood scores of the
  editor within 3 points. (3) The data traps of the handover (multi-token decode, degenerate outputs, signs).
  The family is reported if the audit passes, whatever the methods achieve on it; if it fails, the reason is
  stated in the paper's limitations.
  *Scoped log.* The editor writes at the last subject token, which a served prompt does not mark; the role set
  is therefore all positions of the prompt and of the decode. Everything else is the default method, unchanged:
  detector = logged right factor, normalised to 1 at the solver key; top-1 per position; rebased values;
  two-block calibration, n = 200, alpha = 0.05, the score of a query being its maximum over positions; guards
  on the own request as served.
  *Pools.* Records that are in no log, split by record (seed 20261109) into a REPAIR pool (calibration) and a
  HELD-OUT pool (evaluation). Benign prompts: the request prompt and the paraphrase prompts of these records;
  records that share a subject string with a logged request are flagged and reported separately. Near-scope
  negatives: the neighbourhood prompts of the logged requests (the standard specificity prompts).
  *Rows.* The frozen row set as far as it is defined on these logs (deployed, pre-edit, default, shared scale,
  null repair, cosine gate calibrated, cosine gate at 0.65 at all positions, exact match); undefined rows are
  listed with the reason.
  *Metrics.* (i) Exact and token-level match of the new object on the request and on the paraphrase prompts
  (greedy decode; multi-token). (ii) The public scores (efficacy, paraphrase, neighbourhood success: comparison
  of the probabilities of the new and the true object), computed by the public evaluation code, in separate
  columns, never mixed with (i). (iii) D1 on HELD-OUT benign prompts and on the neighbourhood prompts: the
  pre-edit greedy continuation of 10 tokens is not reproduced (teacher-forced; decode variant next to it).
  (iv) Gate level over 20 calibration draws: false firing on benign prompts, coverage of paraphrases, firing
  on neighbourhood prompts. (v) Degeneration: the n-gram entropy of generations of the public code, for the
  deployed log, the pre-edit model and the default method. (vi) Retention as in E9(c): the first 100 requests
  as T grows. Complaint operators are not run on this family unless registered here later.
  Boundary: text-only language models and ROME are outside Papers 1 and 2.
- 2026-11-11 04:40 UTC, before any log of the E10 family exists: correction of audit condition (2) of E10. On 200 records the
  standard error of the neighbourhood score is about 2 points, so a 3-point tolerance would fail by chance about
  one time in six. The single-edit check uses the first 1000 records (tolerance 3 points per score; the
  published scores are read from the editor's paper, not from memory).
- 2026-11-11 05:06 UTC. **Audit outcome of the one-hop family and the BLIP2 logs** (09_testbed_scale/audit_part3/verdicts.json:
  all seven checks pass for the five logs and both key caches). Conditions of use that follow from the audit:
  (1) One-hop logs: calibration and null-repair queries come only from the REPAIR pool of the logs' own probe
  rows; the Qwen benign pool is not used with these logs (574 of their 1000 probe rows are in it). The stored
  edit flags of these logs were computed in padded batches of 8; every number of the paper is evaluated at
  batch size 1 (the audit's values of the deployed log: Rel exact 70/100 and 236/500, stored 72 and 250), and the
  table of logs is regenerated from the batch-size-1 values of E4 for these two logs. The family is a split of
  VLKEB, not an independent benchmark: 82 of its edits share entity and answer with the main edit stream; the
  paper says so. (2) BLIP2 logs: the editor does not install the edits at any ridge of the scout rule (Rel exact
  0, 1, 2 of 100 at ridge 10, 3, 1), so these are stress logs of collateral damage only; edit metrics are
  reported but no claim about edit preservation is made on them; degenerate pre-edit answers (empty, one
  character, None / No) are counted, and D1 is reported on valid rows with and without the one-token answers.
- 2026-11-11 06:14 UTC, before any final run (no READY.json, no directory of E1 to E9 exists). **Review round 1.** A simulated
  PVLDB review of the partial draft (five seats and a meta-review; record:
  03_analysis/review_round1/) asked for the following. Everything below is added to the registered experiments;
  nothing registered is removed; where an earlier text and this amendment differ, this amendment holds.
  **R1 Roles.** *Repair* = Rescope-WHERE (paper: ReScope-W): the deployed scoping with thresholds as the only
  repair variable; it keeps the deployed edits as they are. *Migration* = ReScope(alpha): scoped replay with
  rebased values at the last prompt position. The paper's headline claim is about the repair at equal
  evidence (E2 a, b); the migration is compared with gates (E1).
  **R2 Parts of the benign pool.** cache/benign/<backbone> (3000 rows) is split by source item, seed 20261111,
  into BEN-W (20 percent of the items: statistics of baselines, extra null-repair keys), BEN-S (45 percent:
  complaint streams, second calibration source) and BEN-T (35 percent: fresh test rows, used for nothing else;
  D1 on its rows with a cached pre-edit answer, firing on all its rows). Not used with the one-hop logs.
  **R3 Gates added to the row set of E1** (role last, top-1, rebased values, guards and both calibrations exactly
  as the registered cosine gate, same draws): cos_centred (cosine of k - mu and k_i - mu, mu = mean BEN-W key);
  cos_null (the cosine detector after the library's null repair, ridge 0, on the 100 keys that rescope_null
  uses in the same draw); white(kappa) (coordinate k_i' M k / k_i' M k_i, M = (C_W + rho I)^-1, C_W = uncentred
  second moment of the BEN-W keys, rho = kappa tr(C_W) / d_in, kappa in {0.01, 0.1, 1}); whitecos(kappa) (the
  cosine in that whitened space). Per backbone, "the best gate of the cosine family" is the variant and
  calibration with the largest value of read-out H3 below (a second one for H2 if it differs), chosen at gate
  level on the development log, i.e. in favour of the baseline, and carried unchanged to every other log; all
  variants are tabulated. Equal-data companion of ours: rescope_nullW = rescope_null with all BEN-W keys in
  place of its 100 REPAIR keys. One statistics source only (BEN-W); no comparison of sources.
  **R4 Reference systems that use the logged requests instead of the log's factors.** retr: per edit the
  embedding of its image (CLIP vision tower of the local LLaVA-1.5 checkpoint, pooled output, for both
  backbones) and of its question (local all-MiniLM-L6-v2); an edit is released iff the image cosine and the
  question cosine both pass thresholds calibrated per attribute by the two-block rule on the same draw at the
  same alpha (queries without an image release nothing); top-1 by the sum of the two margins; the same rebased
  values at the last prompt position. lookup: same image content (md5) and at least one shared content word
  with the edit's question; no calibration. Both are reported in every table that reports a gate.
  **R5 Variant rescope_x (log-derived near-scope calibration; no label, no complaint).** For every edit i,
  m = 4 recombinations of each kind: its image with the image-locality question of a partner item; its
  question with the entity image of a partner item; its image with the question of another edit j. Partner
  items come from the source pool of the hard negatives; a recombination is discarded if its question equals
  the question or rephrase of edit i, or if its (image content, question string) is the query of any test row
  (hard negative, probe, benign). Only the last-prompt key of each recombination is captured. In the two-block
  rule the baseline of edit i becomes the maximum of its block-1 baseline and of its coordinates on ITS OWN
  recombinations; block 2, the quantile and the guards are unchanged, so the bound of the calibration theorem
  holds as before. The same construction is applied to the best gate of the cosine family (<gate>_x).
  Expectation written down now: near-scope firing falls on all three kinds; coverage of text rephrases falls
  (the detector at the last prompt position does not separate a rephrased from an unrelated question well);
  more edits are capped. The result is reported whatever it is.
  **R6 Endpoints.** H1 (repair at equal evidence; E2 a, b): held-out D1 after n = 10 reported complaints,
  ReScope-W (least repair from the log-only thresholds) against the best linear repair of the same complaints;
  it "helps" if D1 is at most 10 percent with Rel exact within 2 points of the deployed log, on the
  development logs and on at least one mild prefix (R12). H2 = the registered primary endpoint of E1, now
  against the best gate of the cosine family, retr and lookup as well. H3 (near-scope-constrained): the hard
  negatives of a log are split by parity of the stream position of the edit that owns the image (even = fit,
  odd = test); for a calibrated gate at alpha = 0.05 the common scale is raised by the smallest amount at
  which the firing rate on the fit rows, averaged over the three kinds, is at most 10 percent; read on the
  odd edits: (T-Gen + I-Gen) / 2 coverage and exact, firing and wrong-target rate per kind on the test rows,
  false firing on heldout_disjoint. Decision rule for the text: "dominates" only if ReScope exceeds the best
  gate of the cosine family by at least 10 points of H3 on both backbones with an interval that excludes 0;
  otherwise the paper says "trade-off" and prints near-scope firing next to every far rate. H1 to H3 are
  the only confirmatory comparisons; Holm correction within each over backbones and baselines; everything
  else is descriptive.
  **R7 Read-outs added to every behavioural row.** Edits kept (exact under the deployed log and under the
  method) and gained (not exact under the deployed log, exact under the method), for Rel, T-Gen, I-Gen;
  D2 as a rate of the pre-correct rows with interval; decoder-v2 generation for every row at draw 0. The main
  table carries the LIN-response ablation and the linear repairs COMP-B and REEDIT-B at all reported
  complaints (records of E2 a).
  **R8 Unseen logs.** Edit metrics also on later edits only (stream position >= 500 for Qwen, >= 100 for
  LLaVA); far locality also on BEN-T; second read-out of the scale series with calibration and test rows
  restricted to rows whose question string is not a question or rephrase of any edit of the longest stream of
  the backbone. The paper discloses that all logs of a backbone share the probe items and the first edits and
  that TB-Q500-ae was read when the default calibration was chosen.
  **R9 Validity (E5 c extended).** Calibrate on REPAIR, test on heldout_disjoint and on BEN-T; calibrate on
  BEN-S, test on both; per modality (image / text rows); distribution of the realised false-firing rate over
  500 draws against the Beta law of the two-block rule. Registered variant rescope_pac: the rank of the
  quantile is the smallest k with P[Beta(n_2 + 1 - k, k) <= alpha] >= 0.9 (calibration-conditional form,
  delta = 0.1); gate level on all logs, behaviour for draws 0 to 2 on the development logs.
  **R10 Statistics.** Intervals by a crossed bootstrap over calibration draws and units (2000 resamples);
  units of probe metrics are query clusters (query_id), of edit metrics edits. Behavioural draws at
  alpha = 0.05 remain 5; this is stated.
  **R11 Do complaints carry information? (E2 g, h).** (g) Stream = BEN-S rows in a seeded order (3 orders);
  complaints = rows on which the default method (calibration draw c, c = 0..2) releases an edit. For n in
  {10, 25, 50}: arm C = null repair (ridge 0) of every detector on the keys of the first n complaints; arm R =
  the same on n random rows of the same stream prefix; arm 0 = no repair; each followed by calibration on a
  fresh REPAIR sample. Read-out at equal coverage: the common scale is moved until the mean coverage of text
  and image rephrases equals that of arm 0 at alpha = 0.05; false firing on heldout_disjoint and on BEN-T.
  Criterion: firing of arm C at most 0.7 times that of arm R on both backbones (mean over orders and draws,
  interval of the ratio below 1). If it fails, the title loses "complaint-driven" and the text says that
  complaints resolve themselves with a certificate but teach the detectors no more than benign queries.
  (h) Blocklist line in every evidence curve: the deployed model plus an exact-match list of the n reported
  queries, which are answered by the pre-edit model; its held-out D1 is measured, not assumed.
  **R12 Mild regime (E2 a, b on prefixes).** Prefix logs of the two development streams after 10, 20 and 50
  edits (pre-edit weight plus the stored batch updates of the first 1, 2, 5 batches) are described like every
  log and then receive E2 (a, b, c) with n in {1, 10, 20, all}, 3 seeds for n <= 20: linear repairs through
  the shared harness with the options of P12, ReScope-W, ReScope(0.05), blocklist. If the harness runs on
  them within 12 GPU-hours per log, E2 (a, b) with n in {10, 100}, one seed, also on TB-Q200-ae-l2_3 and
  TB-L200-ae-l2_3; otherwise recorded as not run.
  **R13 Cost and solver.** E8 adds synthetic logs with T up to 100000 (random factors with the norms of a real
  log): gate latency on GPU and CPU, storage, calibration and least-repair time against T. E5 (e) adds the
  integer program of the same threshold-repair instances (HiGHS) next to the minimum cut: equal optimum,
  run time against instance size.
  **R14 E10 (public text logs), additions before any method has run on a log of the family.** Rows: ReScope-W
  with log-only thresholds and with two-block thresholds (all positions, logged values, linear response);
  the gates of R3 with statistics from 500 REPAIR-pool benign prompts that are in no calibration draw (keys
  at all positions). Equal evidence, n in {1, 10, 100} reported D1 complaints from the REPAIR pool (3 seeds for
  n <= 10): comp_min (the minimum-norm correction of the module that restores its pre-edit output at every
  key of the reported complaints and leaves it unchanged at the solver keys of all edits; when these
  constraints exceed the input dimension this is recorded as infeasible), rb_suffix (point-in-time recovery:
  the longest prefix of the log under which every reported complaint is resolved), rb_lineage (deletion of the
  edits that the log-only thresholds release on the reported complaints), blocklist, ReScope-W with least
  repair. Relational read-out: prompts grouped by relation of the data set into selection views; share of
  views with at least one changed tuple before and after each repair.
  **R15 E11: multi-module logs written by public code (registered before any such log exists).** Editors:
  the public MEMIT code and the public AlphaEdit code with their published hyper-parameters for GPT-2 XL
  (several edited layers), CounterFact in data order, sequential batches of 100 as in the published AlphaEdit
  protocol; one stream of 2000 requests per editor with snapshots at 100, 500, 1000, 2000; development log:
  500 requests from record 15000. The log holds, per edit and edited module, the two factors of its rank-one
  term (columns of the residual matrix and of the solved key matrix of the batch update), the solver key and
  the batch. Audit conditions as E10 (1) and (3); (2) is replaced by: the deployed logs at T = 2000 reproduce
  the efficacy / paraphrase / neighbourhood scores that the AlphaEdit paper reports for both editors under
  this protocol within 5 points (published values read from the paper); a failure of (2) is reported and
  does not exclude the logs if (1) and (3) pass. Method: the repair only (ReScope-W at every edited module:
  each rank-one term is gated by its normalised coordinate at its module, all positions; log-only thresholds,
  and two-block thresholds whose score is the maximum over modules and positions on the pre-edit trace;
  guards on the own requests evaluated on the scoped trace in computation order). The certificate for several
  modules (nothing is released on the pre-edit trace at any module, hence the pre-edit model bit for bit) is
  checked on every certified query. The migration is not defined on these logs (values for several modules
  are open) and is not run. Rows: deployed, pre-edit, ReScope-W (two threshold rules), rb_suffix, blocklist.
  Metrics as E10 (i) to (vi). Reported if the audit passes, whatever the outcome; time box: logs audited by
  2026-11-17, otherwise the family is dropped and the paper restricts every claim to single-module logs.
  Boundary: text-only language models and public code with its own public statistics; [redacted: describes a related submission under double-blind review]
  [redacted: describes a related submission under double-blind review].
- 2026-11-11 06:24 UTC, before any final run. Additions to R11 and R5 of "Review round 1". (1) E2 (g) reports a further arm L
  next to arms C, R and 0: the least repair (thresholds only, no recalibration) of the same n complaints.
  (2) E2 (e) reports, next to the stream of real near-scope complaints, the variant rescope_x (no complaint) on
  the same test rows, and rescope_x followed by the same stream. (3) E9 (c), retention, is read for the deployed
  log, the thresholds-only repair and the default method. (4) Organisation only: the driver is declared ready in
  two stages (stage 1: everything except the gates of R3 to R5; stage 2: with them); E6, E9, E2 (a-c), E2 on
  prefixes, E5 and E8 start at stage 1; E1 and E2 (d-g) at stage 2; the unseen logs after E1 has written the
  selection of R3 (e1_main/out/best_cosine_gate.json). No experiment looks at the result of another except
  through the files this protocol names.
- 2026-11-11 06:27 UTC, before any E11 log exists. Convention for E11 (R15): a logged term whose own coordinate <w_i, k_i> at its
  module is not positive cannot be normalised; the repair silences it at that module and the number of such terms is
  reported per log and layer. The E11 pools are those of E10 without the records of the E11 logs (counted).
- 2026-11-11 06:31 UTC, before any method has run on a reported text log. **R16 Query-level read-out on the text logs (E10, E11;
  replaces the last sentence of R14).** The edited model is read as a relation Facts(subject, relation, value):
  value = the first token of the greedy continuation of a record's request prompt. VIEW pool: 5000 records that
  are in no log and in neither the REPAIR nor the HELD-OUT pool (by record, seed 20261109), split by record into
  a report half and a test half. Views: V_r = the tuples of relation r of the data set, per half; views with fewer
  than 20 tuples are dropped and counted. For every row of E10 / E11, on the test half: (a) tuple level: share of
  tuples whose value differs from the pre-edit value; (b) view level: share of views with at least one changed
  tuple; (c) aggregate level: for SELECT value, COUNT(*) FROM V_r GROUP BY value, the total-variation distance
  between the result before editing and under the row, and the share of views whose result is restored exactly.
  (d) Query-level complaints: a complaint is one view of the report half whose aggregate result differs from the
  pre-edit result; its lineage, the tuples of the view whose value changed, is what every repair receives
  (m in {1, 3, 10} complained views, 3 seeds; the repairs of R14: comp_min, rb_suffix, rb_lineage, blocklist,
  ReScope-W with least repair); read-outs (a) to (c) on the test half after the repair, and the edit metrics.
  Descriptive; no confirmatory test.
- 2026-11-11 06:34 UTC, before any final run. **Clarification of H3 (R6) and R17.**
  (1) H3 is evaluated for two pairs, each on both backbones: (rescope, best gate of the cosine family) and
  (rescope_x, the same gate with the construction of R5). "Dominates" is said of a pair only if the rule of R6
  holds for that pair on both backbones; a claim without qualification refers to the first pair, a claim about
  the second pair says "with log-derived negatives". retr is compared in the first pair's form only. Holm
  correction within H3 over backbones, baselines and the two pairs.
  (2) R17 Out-of-benchmark benign workload. The MME benchmark (local copy; yes / no questions about images of
  other sources) is captured under the pre-edit model of both backbones like the benign pool (key at the last
  prompt position, decoder-v2 pre-edit answer, validity) and split by image, seed 20261111, into MME-C and
  MME-T. Read-outs (E5 c extended; E1 at alpha = 0.05 for every calibrated gate): calibrated on REPAIR, false
  firing and D1 on MME-T; calibrated on MME-C, false firing on heldout_disjoint, BEN-T and MME-T. The bound of
  the calibration theorem is claimed only where calibration and test queries are exchangeable; the other cells
  measure what a shift of the workload costs, for every gate alike. MME images may show entities that an edit
  concerns; such queries are not benign, cannot be identified from the data, and are part of the measured rate.
  (3) No linear rebuild from the pre-edit weight with benign keys as preservation set is run ([redacted: describes a related submission under double-blind review]
  [redacted: describes a related submission under double-blind review]); the paper answers the reviewers' request with
  the capacity statement for linear certification, which covers every linear repair including rebuilds, and
  with COMP-B+ (compensation that also protects known-good rows; P12).
- 2026-11-11 06:38 UTC, before any final run and before any retrieval score has been looked at. **R4 extended: the retrieval
  reference is a family, selected in favour of the baseline.** Calibrating each attribute at level alpha makes the
  conjunction needlessly conservative, which would handicap the reference. Three fusion rules of the image cosine a
  and the question cosine c of a query to an edit, each calibrated by the two-block rule on the same draws at the
  same alpha (rows without an image release nothing): retr_conj (R4 as written: per-attribute thresholds, release
  iff both pass); retr_sum (one per-edit score a + c); retr_min (one per-edit score min(a', c'), a' and c' being a
  and c standardised per edit by the mean and standard deviation of that edit's block-1 scores; the block-1
  statistics are fixed before block 2 is read, so the bound holds). Top-1 by the fused score (retr_conj: by the sum
  of the two margins); the same rebased values at the last prompt position; guards on the own request. Per
  backbone "retr" in every confirmatory comparison is the rule with the largest H3 coverage at gate level on the
  development log (a second one for H2 if it differs), carried unchanged to every other log; all three are
  tabulated. The selection file e1_main/out/best_cosine_gate.json holds this selection as well. The paper states
  the deployment class of every row: the retrieval family needs the requests of the edits (image and question)
  and two external encoders; the gates of the cosine family need the solver keys; the methods of this paper need
  the factors of the log.
- 2026-11-11 07:50 UTC, before any log of this amendment exists and before any final run. **R18 Testbed v4: every
  vision-language log of the paper is built from public resources.** Decision of the first author (2026-11-11):
  the paper neither cites nor depends on unpublished work of the authors. Consequences, binding from now on;
  where an earlier text and this amendment differ, this amendment holds.
  (1) *Pilot logs.* Every log that exists today for the vision-language backbones (all logs named TB-...: the
  logs with [redacted: describes a related submission under double-blind review], their working points, the -memit and -memitw logs, the MMKE-Bench,
  one-hop and BLIP2 logs) is a pilot log: it may be used for software tests of the library and of the driver,
  and nowhere else. No number measured on a pilot log enters the paper (this includes P10 to P12, the sanity run
  e0, e5_linear and the logs' audits). The method, its defaults and this protocol were fixed on pilot logs; the
  paper says so in one sentence ("fixed on pilot logs of the same edit streams, built with other editor settings
  and not part of this paper; every log reported here was built after the freeze").
  (2) *Statistics.* One preservation statistic per backbone and edited module: [redacted: describes a related submission under double-blind review]
  [redacted: describes a related submission under double-blind review]
  [redacted: describes a related submission under double-blind review]
  [redacted: describes a related submission under double-blind review]
  [redacted: describes a related submission under double-blind review]
  computed in this project with its own script. [redacted: describes a related submission under double-blind review]
  [redacted: describes a related submission under double-blind review]
  (3) *Editors* (one module per backbone, layer and edit position of the testbed unchanged: Qwen2.5-VL-3B layer
  20, LLaVA-1.5-7B layer 16, last prompt token, batches of 10, VLKEB, targets optimised once on the pre-edit
  model). ME: the published MEMIT closed form, update = R K1' (lambda C + K1 K1')^-1 per batch, applied
  sequentially, lambda = 15000 as published (GPT-J; also the value of the AlphaEdit repository's MEMIT baseline
  for Llama 3), no further term. AE: the published AlphaEdit closed form, update' = solve(P (K1 K1' + Kp Kp') +
  L2 I, P K1 R'), Kp the keys of all earlier batches, P the projector onto the eigenvectors of C with eigenvalue
  below tau. Two pre-stated adaptations of AE, decided from editor-side quantities only, before any repair method
  is run on any v4 log: (a) the published tau = 0.02 presupposes the activation scale of the published models;
  tau is therefore the eigenvalue quantile at which the null space has the same relative dimension as in the
  published GPT-2 XL configuration at its last edited layer (public statistics and public projector code); the
  relative dimension that the literal value 0.02 would give is reported; (b) L2 by the scout rule already
  registered: the largest value of {10 (published), 3, 1} with Rel exact of at least 70 percent at T = 100 at
  batch size 1, else 1. Both closed forms are validated against the functions of the public repositories on
  random inputs. Nothing else of the editors is varied or studied.
  (4) *Streams, probes, caches.* The edit streams, probe items, pools, hard negatives, benign pool, recombinations
  and key caches are those already built (they depend on the items and on the pre-edit model only).
  (5) *Logs* (names TW-...). AE: TW-Q{100,200,500,1000,2000}-ae, TW-L{100,200,500,1000}-ae (prefixes of one
  stream per backbone). The T = 100 logs of the two ridges not selected by rule (b) are working points
  (TW-Q100-ae-r<ridge>, TW-L100-ae-r<ridge>). ME: TW-Q{100,500,2000}-me, TW-L{100,500,1000}-me. Other data, AE on
  Qwen: TW-QK{100,500}-ae (MMKE-Bench stream), TW-QO{100,500}-ae (one-hop split; conditions of the audit of the
  pilot one-hop logs apply). Conditional on GPU time after every other registered family is built and audited,
  decided before any method is run on them: TW-Q{100,500}-ae-cw (targets optimised on the current weights batch
  after batch, as the public code does; expectation written now: the migration loses reliability on them, the
  repair does not depend on the targets) and TW-B{100,500}-ae (BLIP2-OPT-2.7B, layer 12); if they are not built
  the paper has two vision-language backbones and clean-model targets on them, and says so.
  (6) *Audit.* An independent audit before any method is run: weights equal the pre-edit weight plus the logged
  factors; factor identities; streams and batches; the two closed forms and the statistics recipe against the
  public code; own batch-size-1 evaluation of the stored flags; and independence: no code path of the v4 build
  reads anything from another project of the authors except models, data sets and the Python environment.
  (7) *Roles in the registered experiments.* "Development logs" is replaced, for experiments, by the primary logs
  TW-Q100-ae and TW-L100-ae, and, for software tests, by the pilot logs. The primary logs are held-out logs like
  every v4 log; nothing is tuned on them. The selections "in favour of the baseline" (R3, R4) are made at gate
  level on the primary logs. E1, E2, E6, E9 and the prefix logs of R12 use the primary logs and their working
  points; E3 uses the AE logs of both backbones (no limit at 500 edits: [redacted: describes a related submission under double-blind review]);
  E4 uses the ME, MMKE-Bench and one-hop logs; "later edits" (R8) are unchanged. The internal consistency logs
  are dropped. The linear repair streams (E2 a) and the linear real-log statistics (E5, linear side) are re-run
  on the primary logs with the code and options of P12 and of e5_linear, unchanged. The working points serve as a
  robustness check of the repair, not as a study of the editor.
- 2026-11-11 07:53 UTC, before any v4 statistic or log exists. Clarifications of R18. (a) [redacted: describes a related submission under double-blind review]
  [redacted: describes a related submission under double-blind review]
  [redacted: describes a related submission under double-blind review]
  [redacted: describes a related submission under double-blind review] (b) One ridge L2 per backbone, selected
  by rule (3b) on the VLKEB stream (TW-Q100-ae, TW-L100-ae), is used for every AE log of that backbone, including
  the MMKE-Bench and one-hop logs. (c) Published values read from the public hyper-parameter files on disk:
  MEMIT mom2_update_weight 15000 (GPT-J, Llama 3, Phi-1.5) and 20000 (GPT-2 XL); AlphaEdit nullspace_threshold
  0.02 and L2 10 (all models). The registered lambda = 15000 stands. (d) Stored edit flags of v4 logs are
  evaluated at batch size 1.
- 2026-11-11 08:03 UTC. **R18 and the "Clarifications of R18" are withdrawn in full** (decision of the first author on being told
  the cost: the paper cites the authors' earlier paper, marked "under submission", and declares it in the
  submission form, as the PVLDB rule on related submissions requires). Testbed v4 is not built: its workflow was
  stopped about fifteen minutes after its launch, before any statistic, log or file of it existed; nothing of it
  was computed or looked at. Everything registered before R18 is in force again as written: the logs TB-... with
  their roles (development logs TB-Q100-ae-l2_3 and TB-L100-ae-l2_3, working points, unseen logs, stress logs);
  the boundary amendment of 2026-11-10 17:40 (AlphaEdit-style logs with at most 500 edits in the paper, scale on
  the MEMIT-style -memitw logs, [redacted: describes a related submission under double-blind review]); P10 to P12, the sanity run e0 and e5_linear
  as registered; the amendments "Review round 1" (R1 to R17) with their follow-ups. The words "pilot log" and
  "primary log" have no meaning outside R18.
- 2026-11-11 08:17 UTC, before any method has run on an MME row. Conventions for the MME pools (R17), from the capture
  (cache2/mme/<backbone>/README.txt, verify.json: 2374 rows, 1177 images, 14 categories; MME-C 1188 rows, MME-T
  1186 rows; one invalid pre-edit answer on Qwen, none on LLaVA; no image and no question string shared with
  the caches). (a) The pre-edit answers of this workload are "Yes" / "No" by construction; the degenerate-answer
  filter (bare None / No) is NOT applied to MME rows: every row with a valid pre-edit answer counts. (b) D1 on
  MME-T is, as everywhere, "the pre-edit answer is not reproduced under teacher forcing"; because the answers
  are one or two tokens long, the paper reports it next to the firing rate and says so. (c) Prompts are slightly
  longer than the longest benign image row on 1004 LLaVA rows and 40 Qwen rows; nothing is truncated.
- 2026-11-11 08:49 UTC, before any final run (no READY.json; no directory of E1 to E10 holds a result). **R19 Reduction of the
  programme** (decision of the first author: wind the research down and write; cost). What is cut is cut before
  anything of it was run or seen; the paper makes no claim that depended on a cut item. Where an earlier text and
  this amendment differ, this amendment holds; everything not named here stands (in particular H1, H2, H3 with
  their decision rules, R2 to R8, R10, R16, R17).
  **Not run.** (1) E11 (R15, multi-module logs by public code): its build was stopped at 08:44 UTC, about two
  hours after its launch; the partial logs are unaudited and are not used. Every claim of the paper is restricted
  to single-module logs; statements about several modules are theory only (technical report). (2) E4 on the
  one-hop family and on the BLIP2 logs, the MMKE-Bench logs, the three internal consistency logs. The paper's
  limitation on one-hop questions quotes only the audited description of the deployed one-hop log. (3) E2 (d),
  (e), (f) and (g) (complaint streams on top of the calibrated method, including the test of R11). Consequence,
  fixed now: the title does not contain "complaint-driven"; about complaints the paper claims only what E2 (a, b),
  R12 and E10 measure (the least repair of reported complaints at equal evidence, with its certificate); detector
  repair and why-not repair are described as extensions, with the verified pilot outcomes of P11 reported as what
  they are (registered criteria missed). (4) R13: the integer program; E5 (e) re-reports the verified P11b record
  and runs nothing new. (5) The quantile-baseline ablation of E6. (6) The answer-match baseline on unseen logs
  (it stays in E1). (7) GPT-J in E10. (8) The T = 200 logs of the MEMIT-style family in E3.
  **Reduced.** (a) E1: behaviour at alpha = 0.05 for calibration draws 0 to 2; the other four levels at gate level
  only (20 draws); working points at gate level only; generated answers and D2 at draw 0 for the rows of the main
  table (deployed, pre-edit, ReScope, rescope_x, ReScope-W with log-only thresholds, the best gate of the cosine
  family, retr, COMP-B and REEDIT-B at all reported complaints); rescope_nullW and rescope_pac at gate level only.
  (b) E3 logs: TB-Q200-ae-l2_3, TB-Q500-ae-l2_3, TB-Q{100,500,1000,2000}-memitw; TB-L200-ae-l2_3,
  TB-L500-ae-l2_3, TB-L{100,500,1000}-memitw; row set and draws as registered. (c) E9: calibration draw 0; three
  seeded revocation sets per m. (d) R12: the prefix logs only. (e) The parts BEN-T, BEN-S of the benign pool and
  the MME pools are used at gate level only (false firing); no behavioural evaluation on benign-pool rows.
  (f) E10: gate level on all four reported logs; behaviour, equal evidence and the query-level read-out on the
  logs with 500 and 2000 requests; of the default method's variants only the default; of the gates of R3 all at
  gate level and, at behaviour, the one with the largest paraphrase coverage on the development log;
  query-level complaints with rb_suffix, blocklist and ReScope-W. (g) Verification: one independent verification
  each for E1, E2 (a-c), E3 and E10, and one for the remaining experiments together.
- 2026-11-11 09:03 UTC, before any final run. Two clarifications that follow from the mechanics checks of the feature layer on
  the development logs (feat2/*/out/mechanics.txt; draw 0; not results). (1) *Null-repair sample.* The registered
  text of ReScope-N says "disjoint from the calibration queries"; the pilot code separated rows, and because the
  REPAIR pool repeats queries, 5 (Qwen) and 3 (LLaVA) of the 200 calibration queries of draw 0 were also
  null-sample queries (their score is 0 after the repair, which biases the threshold). The driver separates the
  null-repair sample from the calibration draw by QUERY (query_id) for rescope_null and cos_null; the calibration
  theorem requires detectors that do not depend on the calibration queries. (2) *Recombinations and test
  negatives share a source pool.* The benchmark repeats question strings and entity images: most recombination
  rows reuse, on another image or with another question, a string or image that also occurs in a test hard
  negative (no recombination equals a test query). For every _x method the near-scope read-outs (firing per kind,
  H3) are therefore also reported on the subset of test hard negatives whose foreign part (the question of an
  HN-I row, the image of an HN-Q row, the partner edit of an HN-X row) occurs in no recombination row of the log,
  with the size of the subset; the paper states the overlap.
- 2026-11-11 10:46 UTC, before any final run. **R20 Capacity curve of linear certification (E5, part f; development logs; CPU after
  one capture).** Round 4 of the theory (03_analysis/theory/round4_review/P1b_capacity; refereed there) states: a
  linear repair (rollback, compensation, re-editing, any other weight) that restores n complaints on their whole
  pre-edit traces can preserve at most cap(n) = rank[K_Q(n), K_E] - rank K_Q(n) edits at their solver keys (K_Q(n):
  the keys of the edited module at all positions of the pre-edit traces of the n complaints; K_E: the solver keys),
  with equality when the edits' values are linearly independent; from rank K_Q(n) = d_in on, the only such weight is
  the pre-edit weight. From ranks recorded at n <= 10 the theory PREDICTS, before the measurement: all T = 100 edits
  can survive up to 23 to 34 complaints on the Qwen development log and about 35 on the LLaVA one, and from about 25
  to 36 complaints on only the pre-edit weight restores them (predictions_P1b.json). Measurement: D1 complaints of
  the REPAIR pool under the deployed log in 3 seeded orders; for n in {1, 2, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60}:
  rank K_Q(n), cap(n), the number of edits that the certifying compensation of smallest norm preserves at their
  solver keys (module level, to round-off), and, for the same complaints, the number of edits that the least
  threshold repair of the deployed scoping silences on their own requests and on their solver keys. Whatever comes
  out is reported next to the prediction; a miss of the prediction is reported as such.
- 2026-11-11 11:38 UTC, before any method has run on a reported text log. **R21 E10 after the audit of the text logs.** The
  independent audit (09_testbed_scale/text_public/audit/verdicts.json, verdicts_table.txt: all seven checks pass,
  all five logs safe; single edits reproduce the published scores: ES 100.0 / 100.0, PS 96.7 / 96.4, NS 75.4 /
  75.4) describes the DEPLOYED logs: the reported stream collapses at update 173, a request whose subject is the
  first token of its prompt, for which the public code divides by a coordinate of 0.0218 and writes an update of
  norm 5902 against a weight of norm 117; 8 of the 2000 updates carry 99.62 percent of the squared update norm;
  T100 is the only reported log before the collapse (ES / PS / NS 100.0 / 98.5 / 66.8); T500, T1000 and T2000 are
  collapse logs (ES 57 / 53 / 50, D1 100 percent; 1827 of the 2000 targets were computed on collapsed weights).
  R19 (f) had placed the behavioural evaluation of E10 on the logs with 500 and 2000 requests, i.e. on collapse
  logs only. Changed here, on the basis of the description of the deployed models alone (no method has run on a
  reported text log; the adapter's mechanics run is on the development log):
  (a) Behaviour, equal evidence and the query-level read-out run on TX-G2XL-rome-T100 (the clean reported log) and
  on TX-G2XL-rome-T500 (a collapse log), with the rows of R19 (f). T1000 and T2000: gate level, and the screening
  of (c).
  (b) Collapse log. The split position is 173, from the audit's table of the deployed model around that update,
  fixed here. Expectations written now, from the algebra of the public code, before any run: (1) in the log's
  normalised coordinates an exploding update has a read-out vector of large norm, so that benign keys have
  coordinates above 1, the coordinate of its own request; the frozen default, whose guards keep every own
  request in scope, will therefore cap such updates, and they will keep firing on most benign prompts: the
  calibrated rows are expected to MISS their bound on this log through capped edits; whatever happens is
  reported; (2) every calibrated row is therefore also run on T500 with guards off (the registered ablation of
  E6): an update whose own request is not separated from benign queries is then silenced on its own request and
  counted as lost; (3) the requests after the split have targets computed on collapsed weights: no method is
  expected to recover them, and the deployed log does not have them either; every read-out on T500 is split into
  requests before and after position 173, and claims concern the requests before it and the benign prompts.
  (c) Screening from the log alone (no query; all four reported logs; descriptive): the self-separation margin of
  the retrofit proposition, 1 minus the largest normalised coordinate of edit i at the solver key of another
  edit. Reported: the edits with a non-positive margin, against the updates of exploding norm (the audit's list)
  and against the requests whose subject is the first prompt token: precision and recall.
  (d) The least repair of reported complaints on the text logs is reported in two forms: *sacrificing* (no
  must-cover pair: every reported complaint is certified; an edit whose own request is thereby silenced is counted
  as lost; the form of the theory, in which the least repair loses exactly the edits whose own coordinate does
  not exceed the complaints') and *protecting* (the library's rule: a complaint that cannot be separated from an
  own request is refused and counted).
  (e) D1 on 10-token continuations is near its ceiling already on the logs that did not collapse (93.5 percent at
  T100): the share of pre-edit tokens kept and the share of changed first tokens are reported next to every D1.
  Prompts of the pools whose text also occurs in a logged request or shares its subject are flagged and reported
  separately, as in the adapter's tables. Everything else of E10 is unchanged.
- 2026-11-11 11:51 UTC, before any method has run on a reported text log. **R22 Relational maintenance workload on the text logs
  (replaces R16 and the last sentence of R14; logs: TX-G2XL-rome-T100 and -T500).** An outside reading of the
  scope case found that R16, as registered, could be satisfied by restoring the pre-edit model everywhere (its
  views held unedited records only), called "lineage" what is a diagnosis, and did not say what evidence a repair
  receives. Redesigned before any run:
  (a) *Relation.* Facts(subject, relation, value) over a finite set of records D: the requests of the log (edited
  tuples) and a VIEW pool of 5000 records that are in no log and in neither the REPAIR nor the HELD-OUT pool (by
  record, seed 20261109; records that share a subject string with a logged request are excluded and counted). The
  value of a tuple under a served model is the first token of its greedy continuation of the record's request
  prompt. Three states: the pre-edit state R0 (pre-edit model); the deployed state; the INTENDED state R*: the
  first token of the new object for edited tuples, the pre-edit value for all other tuples. Every state and every
  repaired model is materialised as a table, and the read-outs are computed by SQL on it (sqlite3).
  (b) *Queries.* For every relation r with at least 20 tuples: the selection view V_r and the aggregate
  SELECT value, COUNT(*) FROM Facts WHERE relation = r GROUP BY value. The VIEW-pool records are split by record
  into a report half and a test half; the edited tuples of relation r belong to the view of either half.
  (c) *Scores, against the intended state R** (so that restoring the pre-edit model everywhere is not a success):
  reference preservation (share of unedited test tuples whose value equals the pre-edit value); intended-update
  retention (share of edited tuples whose value is the new object; on T500 separately before and after position
  173); query level (share of test-half queries whose result equals the result on R* exactly; mean total-variation
  distance to it).
  (d) *Complaints and the evidence contract.* A complaint is a report-half query whose result on the deployed
  model differs from its result on R*; the complainant supplies the query, nothing else. The operator holds the
  log (pre-edit weight, factors, solver keys) and diagnoses, at a cost that is charged: every tuple of the
  complained view is evaluated under the pre-edit model and under the deployed model (two forward passes per
  tuple); the tuples whose value changed and whose prompt is not the own request of a logged edit (recognised
  from the log: a position of the prompt carries a logged solver key up to round-off) become tuple-level
  complaints. Every repair receives exactly these complaints; any further model evaluation that a repair needs
  (replays of prefixes for point-in-time recovery, keys for a compensation) is counted in forward passes.
  m in {1, 3, 10} complained queries, 3 seeds.
  (e) *Repairs and references.* The least threshold repair of ReScope-W in its two forms (R21 d); rb_suffix;
  rb_lineage; comp_min; blocklist. Without complaints: the deployed log; restore-base (the pre-edit model);
  ReScope-W and the migration with two-block thresholds; rb_screen (rollback of the updates that the log-only
  screening of R21 c flags); on T500 also the oracle "delete the update at the split position". Reference of
  another deployment class, labelled as such: lookup_subject, a keyed store that needs the REQUESTS: an edit is
  released iff the prompt contains its subject string, at the last subject token, with the migration's value.
  (f) *Reported per repair:* the scores of (c) on the test half; forward passes for diagnosis and for the repair;
  repair time; statements or thresholds changed; edits lost; serving overhead per forward pass. lookup_subject is
  also a row of the behavioural tables of E10. Descriptive; no confirmatory test.
- 2026-11-11 12:37 UTC, before any final run. **Outcome of the independent review of the shared driver** (common/review/patches.txt,
  review/out/review_table.txt; driver fingerprint b92d0532498f). (1) One bug was found and patched before any
  experiment used the driver: the behavioural session cached the signature of a replacement weight (and of released
  values) by object identity, so that a tensor modified in place was served with the rows of its first contents; the
  signature is now a hash of the present contents. No result of this protocol was produced with the bug (the sanity
  run e0 and all pilots used the library directly). (2) The reviewer rebuilt 536 gates by its own rule (thresholds
  equal to 1e-14, 0 of 2.98 million decisions differ) and reproduced the behavioural cells of the default method and
  of the cosine gate on both development logs with the library's plain evaluator, row for row. (3) Confirmed without
  change: the log-only thresholds of Rescope-WHERE use the negatives of P10 as section 0 defines them: every position
  of the other edits' request sequences and the non-solver positions of the edit's own request sequence (the
  reviewer notes that the latter set 65 of the 100 thresholds on the LLaVA development log; the paper states the
  rule). (4) Experiment agents assert in every job that the hook and the key cache agree (hook_mismatch = 0) and
  that the memo reports no conflict; one log per process; the number of capped edits is reported for every
  calibrated gate (the two-block cosine gate caps 23 to 58 edits on the LLaVA development log, for which its bound
  is void).
- 2026-11-11 15:25 UTC, before any run on an unseen vision-language log. **R23 Further reduction of E3 (cost).** On the logs with
  1000 or more edits (TB-Q1000-memitw, TB-Q2000-memitw, TB-L1000-memitw) the behavioural evaluation of the row set
  uses calibration draw 0 only (the driver's builder measured up to 7 hours per draw for the row set at 2000
  edits); gate level stays at 20 draws; the logs with at most 500 edits keep draws 0 to 2. The thresholds-only
  repair with log-only thresholds is run on these three logs only if one pass fits in 40 minutes (as registered).
  Nothing else changes.
- 2026-11-11 16:41 UTC, after the run and the independent verification of E10 (e10_text/run/verify/verdicts.json: 12 of 14 checks
  pass). Reporting clarifications that the verification asked for; no number of the logs with 100, 500 or 1000
  requests changes. (1) *Screening (R21 c), ties.* Two logged requests with the same request prompt have the same
  solver key, so each has margin exactly 0 against the other, and round-off decided whether such an edit was
  counted (29 flagged by the experiment, 31 by the verifier at 2000 requests). Rule: a margin of absolute value
  below 1e-9 is a tie; ties (duplicate solver keys) are reported separately from negative margins. (2) *Least
  repair (R21 d).* With the library's margins a repaired complaint can remain inside the decision band of the gate
  as served; the paper reports the measured number of reported complaints that are certified as served and does
  not write "every". (3) *Guards off on the collapsed log.* The outcome of the two-block rule is bimodal over
  calibration draws; the paper reports the draws separately (behaviour: draws 0 to 2; gate level: the distribution
  over the 20 draws), not only their mean. (4) *Point-in-time recovery.* The deployed model after the first 173
  requests (the last prefix before the collapse) was described by the audit of the logs (its own decode of 200
  HELD-OUT prompts and of the first 100 requests); the paper quotes that description as the outcome of a rollback
  to the last prefix before the collapse; no further run.
- 2026-11-11 18:31 UTC, before any final run on a vision-language log. **R24 ReScope-W with thresholds from benign queries on
  the vision-language logs (E2, part c'; development logs and their prefix logs).** The paper gives the repair
  three sources of thresholds: the log alone, benign queries (the calibration algorithm), reported complaints
  (least repair). On the public text logs all three were registered (R14) and run; on the vision-language logs
  only the first and the third were. Added: ReScope-W (all positions, logged values, linear response) with the
  two-block rule at alpha = 0.05 on n = 200 benign queries of the REPAIR pool; the score of a query for edit i is
  its largest normalised coordinate over all positions of the pre-edit trace of the query (prompt and pre-edit
  answer, teacher-forced); guards on the own requests as in the default method. These per-query vectors of maxima
  are captured once per row of the REPAIR and heldout_disjoint pools and per edit sequence; gate level over 20
  calibration draws from them (false firing on heldout_disjoint, own requests and rephrases released, capped
  edits); behaviour for draws 0 to 2 with the metrics of E2 (b). It is a row of the main table (deployment class:
  the log and benign queries) and of the prefix-log table. Expectation written now: on these logs the log-only
  thresholds already leave few changed answers (pilot P10: 0.7 and 4.1 percent of the held-out rows), so the
  two-block thresholds are expected to add the bound and little else; whatever comes out is reported. Not run on
  the unseen logs of E3 (cost); the paper says so.
- 2026-11-11 19:48 UTC, before any final run on a vision-language log. **R25 Decisions after the independent review of the extended
  driver** (common/review_lean/verdicts.json: all ten items pass, no line of the driver changed by the review;
  fingerprint 6a71f0d2cd71; READY.json: stage 2, reviewed). The review left two decisions and several reporting
  rules to the coordinator; they are fixed here, before the selection of R3 / R4 is made and before any behavioural
  run. (1) *Top-1 of the retrieval rules.* retr_sum and retr_min release, among the edits that pass, the one with
  the largest margin of its fused score over its block-1 baseline, as every two-block gate of this protocol does
  (the text of "R4 extended", "top-1 by the fused score", is read so; firing is unaffected). (2) *The _x form of
  a shared-scale gate* of the cosine family is the two-block _x form on the same detectors (R5 is defined inside
  the two-block rule). (3) *Intervals and the word "dominates".* Behavioural intervals rest on three calibration
  draws; the crossed bootstrap is then too narrow when the draw-to-draw component matters (coverage 0.86 at
  nominal 0.95 in the review's simulation). The number of draws is printed next to every interval, the draws are
  listed separately where they disagree, and the decision rule of R6 becomes stricter: "dominates" additionally
  requires that the same difference in its gate-level coverage form (coverage of text and image rephrases; for H3
  under the near-scope constraint), over the 20 calibration draws, has a crossed-bootstrap interval that excludes 0
  with the same sign. (4) *H3.* Rows of the kind HN-Q are split by the edit whose question they ask; the raise of
  the scale keeps the guard caps; the number of edits at their cap is reported with every H3 number. (5) *Keys of
  the validity cells.* A gate is calibrated on teacher-forced keys of the REPAIR pool by default; the rows of
  BEN-T, BEN-S, MME-C and MME-T have prompt-only keys. In the cells of E5 (c) that involve one of these pools,
  calibration and test both read the prompt-only key of every row (REPAIR and heldout_disjoint included); the
  registered cells REPAIR to heldout_disjoint are also reported at teacher-forced keys. The BEN-T and MME-T columns
  of E1 and E3 (calibration at teacher-forced REPAIR keys, test at prompt-only keys) say so in the table.
  (6) *Reporting.* Next to every calibrated row: the number of capped edits and the false-firing rate of the
  literal rule without guards; all 20 gate-level draws are kept in the result files. The retrieval family and
  lookup are not defined on the MME pools and have no _x form. The pool guard of the driver sees sequences, not
  tensors: where an experiment passes key matrices itself, it keeps BEN-T and MME-T on the test side, and the
  verification checks this.
- 2026-11-12 00:48 UTC, after the run and the independent verification of E1 (e1_main/verify/verdicts.json: every claim of the
  experiment's summary confirmed; 410 table cells and 392 paper numbers reproduced; own forward passes: 0 of 73,164
  flags differ; analysis and rendering re-run identically). **Registered outcome of H2 and H3 on the development
  logs: "trade-off" for both pairs of H3 and for the retrieval reference** (decision.json; rule of R6 with R25 (3),
  applied literally): H3, ReScope minus the best gate of the cosine family: -1.3 [-7.7, +5.3] (Qwen) and -4.0
  [-9.7, +1.7] (LLaVA) points; with log-derived negatives: +2.0 [-3.0, +7.0] and -0.3 [-6.0, +5.3]; against the
  retrieval reference: -22.3 [-36.0, -9.0] and -12.7 [-25.0, -0.3]. H2 (all edits): against the gate selected for
  H2, a coordinate whitened with benign keys under the shared-scale calibration: -7.0 [-12.0, -2.0] (Qwen; Holm
  p = 0.028) and -1.3 [-8.0, +4.7] (LLaVA); against the retrieval reference: -27.8 and -29.0; against the registered
  cosine gate, ball, router, lookup and the answer-match baseline: +25 to +49 points. The paper therefore does not
  claim that its detector is better than a detector that uses benign key statistics or the requests.
  Decisions on the protocol problems that the verification lists. (1) The exact McNemar test of section 4 is added
  for the registered binary paired comparisons (D1 on heldout_disjoint and Rel exact, ReScope against the deployed
  log and against each row of the main table), from the stored per-row flags, per calibration draw; no model run.
  (2) The calibrated z-score gate and the published forms of the ball and router gates are gate level only
  (reduced programme): H2 has 18 comparisons, which is stated. (3) Ties of the selection of R3 / R4 are broken by
  the order in which the driver lists the family (the registered gate first), the rule the experiment applied; on
  the Qwen log the H3 read-out of two gates of the cosine family ties exactly, and the decision is "trade-off" with
  either. (4) Reporting: the three behavioural draws are listed next to every near-scope cell (they differ by up to
  25 points); wrong-target rates, D2 rates and generation metrics are added to the paper numbers; Holm-adjusted
  values at the resolution of the bootstrap are printed as upper bounds; the rows taken from E2 are re-rendered
  from the final, verified E2 records.
- 2026-11-12 01:00 UTC, after the run of E2 (parts a to c') on the development logs, BEFORE its independent verification has
  reported and BEFORE the prefix logs of e2_prefix are complete (status.json there: the 20- and 50-edit prefixes
  are missing or partial). **Reading of the rule of H1 (R6): both readings are reported; the rule is not re-worded.**
  The rule says "D1 at most 10 percent with Rel exact within 2 points of the deployed log". The code fixed the
  two-sided reading (|Rel - Rel deployed| <= 2) before any cell of the repair existed. Outcome of the run
  (h1.json; to be confirmed by the verification): D1 after 10 reported complaints 1.0 percent (Qwen) and 2.6 percent
  (LLaVA); Rel exact of the repair against the deployed log +0.0 and +4.0 points. On the LLaVA log the two-sided
  wording is therefore NOT met, because the repair answers more requests than the deployed log (86 of 100 kept,
  6 gained); the one-sided reading (not more than 2 points below the deployed log), which is what the criterion
  was written to protect (no loss of edits), is met on both logs. The paper states the rule as registered, says
  that its two-sided wording fails on the LLaVA log and why, and gives the one-sided reading next to it, marked
  as a reading adopted after the run. The prefix part of the rule (R12: at least one mild prefix) is evaluated
  under both readings in the same way, whatever it shows. The differences of D1 to the linear repairs of the same
  complaints (Holm over backbone and the two choices of the best linear repair) are reported as computed.
- 2026-11-12 02:55 UTC, after the independent verification of E2 (e2_curves/verify/verdicts.json, written 2026-11-12 01:32 UTC:
  nine of ten claims confirmed; the summary's list of unfavourable items "correct but incomplete"). **Correction of
  the previous amendment and reporting decisions for E2.** (1) The previous amendment says that the code fixed the
  two-sided reading of "within 2 points" before any cell of the repair existed. The verification could not confirm
  this from the files (the file that holds the rule was modified after the last H1 cell was written). The sentence
  is withdrawn; the paper does not make it. Nothing else changes: the rule is reported as registered, with both
  readings and the numbers behind them, and the reading that fails (two-sided, LLaVA development log, Rel exact
  4.0 points ABOVE the deployed log) is named. (2) Reported next to the H1 row, as the verification asks: the share
  of held-out rows and of near-scope negatives on which the repaired log still releases an edit (18.0 and 23.7
  percent of the held-out rows; 9 to 52 percent of the near-scope negatives), and the edits kept, gained and lost
  against the deployed log (the repair retains every recorded term and does not keep every deployed answer).
  (3) The least repair that starts from the deployed log (all thresholds minus infinity) still releases an edit on
  52 and 97 percent of the held-out rows after 10 complaints: its low D1 is a measurement, not a certificate, and
  the paper says so. (4) The read-out of part c' on the REPAIR rows outside the calibration draw (cw_check.py) was
  added after the result on heldout_disjoint had been seen: it is labelled post hoc wherever it is quoted, with an
  interval over 200 draws (to be computed from the captured vectors, no model run). On the LLaVA log its mean
  (5.24 percent) is above the bound (4.95 percent); the paper reports this and does not explain it away.
  (5) False firing of the label-free migration on the out-of-benchmark workload (MME-T, prompt-only keys) exceeds
  the level on the Qwen log at n = 40, 50, 200 and ALL: reported in RQ5 with the other shifted pairs.
  (6) The running example: the default method releases no edit on the example's complaint; "the edit" of the
  example is the one with the largest coordinate, and the why-not rephrase belongs to another edit: the text
  presents them as two edits. (7) Prefix part of H1 as of this time (h1_prefix.json, five of six prefix logs
  complete at n = 10; the 50-edit Qwen prefix is still running): the rule is met in its two-sided wording on four
  prefixes (10 and 20 edits, both backbones) and in its one-sided reading only on the 50-edit LLaVA prefix (Rel
  exact 6.0 points above the deployed log); the final file is read when the run has finished.
- 2026-11-12 04:14 UTC, after the prefix-log run of E2 finished (e2_prefix/out/h1_prefix.json of 04:11 UTC: six of six prefix logs
  complete, no failed cell). **Final outcome of H1.** Held-out D1 after 10 reported complaints, ReScope-W (least
  repair from the log-only thresholds), deployed -> repaired, with Rel exact deployed -> repaired:
  Qwen development log 54.2 -> 1.0, 78 -> 78; LLaVA development log 62.3 -> 2.6, 88 -> 92; Qwen prefixes of 10 / 20 /
  50 edits 23.9 -> 0.6 (80 -> 80), 29.9 -> 1.1 (75 -> 75), 43.0 -> 0.9 (82 -> 78); LLaVA prefixes 26.4 -> 0.5
  (90 -> 90), 35.4 -> 0.9 (90 -> 90), 50.5 -> 1.8 (86 -> 92). The rule as registered (D1 at most 10 percent with Rel
  exact within 2 points of the deployed log, on the development logs and on at least one mild prefix): NOT met,
  because its development-log part fails on the LLaVA log (Rel exact 4.0 points above the deployed log); its prefix
  part is met (four of six prefixes). Under the one-sided reading (not more than 2 points below; adopted after the
  run, see the two amendments before this one) the rule is met: both development logs and five of six prefixes.
  The 50-edit Qwen prefix fails under either reading (Rel exact 4.0 points BELOW the deployed log); the sentence
  of the amendment of 02:55 UTC about the prefixes was provisional and is replaced by this one. The paper reports
  all of this: the registered verdict first, the one-sided reading as such, and the range of the change of Rel exact
  (-4 to +6 points) with kept / gained / lost. Observation of the same run, to be checked by the secondary
  verification before it is quoted: on the 50-edit Qwen prefix the two-block thresholds of ReScope-W release an
  edit on 6.97 percent of the heldout_disjoint rows (mean of 20 draws; 13 of 20 draws above the bound of 4.95
  percent); heldout_disjoint is a second pool, not exchangeable with the calibration pool row by row.
- 2026-11-12 04:51 UTC, after the independent verification of the secondary experiments (e5_rescope/verify_secondary/out/
  verdicts.json, written 04:48 UTC: checks 0 to 6 pass with own captures and own forward passes; 100 of 100 paper
  numbers of E5, E8, E6, E9 and the prefix logs reproduced from the per-row files; the additions of the fix-up
  reproduced; no defect of the two-block rule). **Deviations that the experiments' reports did not state, and
  decisions.** (1) E8 times the paper's gate (last prompt position, all positions, the thresholds-only form) and
  the answer-match baseline; section 3 asks for "each gate": the other gates (cosine, ball, router, z-score) are
  not timed, and the paper makes no latency claim about them. (2) E6: behavioural cells exist for 8 of the 28
  ablation rows (all positions, linear response, logged values, shared scale, log-only thresholds, running maxima,
  guards off, contamination 20 percent). The list was fixed in the task text of the experiment (written
  2026-11-11 19:48 UTC, before the run, under the cost reduction decided with R19) and is NOT part of the text of
  R19: the table's attribution "R19" is wrong and is not repeated in the paper; rows without a behavioural cell
  are reported at gate level only, and a row that changes only what is released (GAIN response, selection = all)
  equals the default row at gate level by construction, which is said where such a row is quoted. (3) E5 (c):
  validity cells across calibration sources exist for 8 of the 13 logs of the experiment's table; for the other
  five (the 200- and 500-edit logs of the first family, one 500-edit log of the second) false firing on
  heldout_disjoint is reported from E3 (gate level, 20 draws), and the paper's statements about shifted
  workloads are restricted to the logs that have the cells. (4) E5 (f): the thresholds-only repair of the
  capacity experiment starts from the log-only thresholds, where R20 says "of the deployed scoping"; the reported
  conclusion (no edit is silenced on its own request) holds a fortiori from thresholds minus infinity; the paper
  names the start. (5) Retention of the first 100 edits as T grows (E9 c) is read from E3, as registered.
  (6) Keys of the out-of-benchmark pool: E1 and E2 evaluate MME-T at the last prompt position of the teacher-
  forced sequence, E5 (c) at the prompt-only key of every row on both sides (R25 (5)); for a gate at the last
  prompt position the two coincide by causality (identical firing in all 20 draws on both logs), so no number
  changes; the label "prompt-only keys" of E2's table and of the amendment of 02:55 UTC is wrong for MME-T and is
  corrected in the paper to "at the last prompt position". The paper's statements on calibration under workload
  shift (RQ5) quote E5 (c). (7) E1 table (2b): the near-scope cells of the row "ReScope-W, two-block thresholds"
  are calibration draw 0 only (the label says draws 0-2); the paper takes these cells from E2 (gate level: the
  20-draw record; behaviour: draw 0, labelled). (8) Post hoc calibration read-out (amendment of 02:55, item 4):
  reported as mean and Monte Carlo standard error over 200 draws on the REPAIR rows outside the draw: 5.16 and
  5.03 percent (fix-up) and 4.95 +- 0.18 and 5.02 +- 0.18 (verification, own draws) against the bound 4.95;
  50-edit Qwen prefix 5.01 +- 0.17. The excess on heldout_disjoint of that prefix (6.8 percent) and of the Qwen
  development log (5.9 percent) is a property of a second pool (three edits carry 30 percent of its releases); the
  bound is not claimed for a second pool row by row, and the paper prints both numbers. (9) Out-of-benchmark
  workload on the Qwen log (calibrated on benchmark queries): three estimates of the mean false-firing rate,
  6.1, 6.3 and 7.7 percent, all above the level, dominated by a few draws (95th percentile about 24 percent):
  reported as a failure of the level under workload shift.
- 2026-11-12 06:16 UTC, after the run and the independent verification of E3 on both backbones (e3_qwen/verify/verdicts.json,
  written 06:12 UTC: every reported number reproduces; own forward passes at draw 0: 0 of 6,804 probe flags and
  7,200 edit flags differ; own two-block rule over 20 draws on 13 logs and 3 methods: 0 of 364,140 firing flags
  differ; 4,674 of 4,700 paper numbers recomputed, 0 disagree; three sentences of the experiments' summaries are
  wrong and are not used). **Outcome of the registered read-outs on the unseen logs: H2 is not confirmed, and
  nothing "dominates" in H3.** The method, its variants, the level and n were applied unchanged. Rephrase exact,
  (T-Gen + I-Gen) / 2, ReScope minus the gate of the cosine family selected for H2 on the development log
  (Qwen: the experiment's values; LLaVA: the verification's own paired bootstrap, the experiment computed none):
  first editor family (unseen T = 200, 500): Qwen +2.8 [-4.0, +10.8] and -0.4 [-3.0, +2.1]; LLaVA +1.2
  [-5.2, +8.5] and -1.1 [-4.7, +2.2]: no difference. Second editor family (public MEMIT-style configuration;
  T = 100 to 2000): Qwen -34.5, -14.6, -10.0, -8.1; LLaVA -31.0 [-39.7, -22.0], -17.8 [-21.0, -14.7], -10.2
  [-11.7, -8.7]: ReScope is worse on every log; on later edits only (R8) the same (Qwen T = 2000: -7.3
  [-8.5, -6.2]; LLaVA T = 1000: -8.9 [-10.4, -7.4]). H3, pair 1: positive only on the first family of Qwen
  (+1.2; +5.1 [2.1, 8.0]) and negative on every other unseen log (-2.7 to -23.3); pair 2: +3.1 to +5.8 on the
  first family, -2.9 to -14.7 on the second; against the retrieval reference -12.5 to -37.4 on every log.
  The paper reports this as the outcome of the replication: on the first family the trade-off of the development
  logs is reproduced; on the second family the coordinate detector is worse than a detector on the solver keys
  with benign key statistics. The thresholds-only repair on the unseen logs is reported in the same way: on the
  first family held-out D1 is 0.9 to 1.1 percent (Qwen) and 3.4 and 10.4 percent (LLaVA, T = 200, 500) with Rel
  exact above the deployed log; on the second family D1 is 4.7 to 17.5 percent and on the Qwen logs the repair
  answers fewer requests than the deployed log (26 against 53 at T = 100), because a logged value alone is far
  from its target there (relative error up to 118 percent).
  Decisions on what the verification lists. (1) The LLaVA paired differences are computed from the stored flags
  with the method of the Qwen analysis by a fix-up (new files only) and compared with the verification's values.
  (2) The confirmatory family stays that of R6 (development logs). The read-outs on unseen logs are
  replications: reported per log with paired intervals; Holm within each read-out over the unseen logs of both
  backbones and the baselines is added and labelled as such; exact McNemar tests of D1 and Rel exact against the
  deployed log for the repair and the migration, per log. (3) Tables by editor family, never side by side; rows
  of methods that read only solver keys or requests and release rebased values are equal in both families at
  equal T by construction (the two families of a backbone share requests and solver keys bitwise; verified), and
  the paper says so. (4) Printed next to far locality on unseen logs: the release rate on held-out probes that ask
  the question of an edit (3 to 23 percent), the number of calibration draws above the level (0 to 9 of 20 per
  log) and the number of capped edits (for them the bound is void: up to 60 and 149 for the default method on the
  second family). (5) The smallest self-separation margin on the LLaVA log with 1000 edits is -0.107 (two edits
  with a non-positive margin); the experiment's summary sentence ("0.02") is wrong. (6) Latency and storage
  against T are read from E8, not from E3. (7) The paper's table of logs prints the deployed log's edit metrics
  as evaluated here (one sequence per forward pass), which differ from the metrics stored at build time on 0 to
  13 edits per metric. (8) The second read-out of R8 (question-disjoint calibration and test rows) restricts
  the far rows only; edits and near-scope negatives are read on all rows under that calibration (deviation of
  the Qwen experiment, stated by its agent).
- 2026-11-12 12:35 UTC, after all experiments and verifications, while the paper is finished. **Presentation of H1 in the paper (no
  change of any outcome).** The amendments of 2026-11-12 01:00, 02:55 and 04:14 UTC record the registered verdict
  (not met as worded) and a one-sided reading adopted after the run. The paper prints the registered verdict and,
  instead of naming a second reading, the facts from which a reader can form it: on which of the eight logs the
  requests answered lie more than 2 points below the deployed log (one: the 50-edit Qwen prefix, -4) and on which
  more than 2 points above (two: the LLaVA development log, +4, and its 50-edit prefix, +6). An interpretation
  adopted after a run is not offered as evidence. The one-sided reading stays recorded here and in
  e2_curves/out/fixup/h1_registered.json and e2_prefix/out/h1_prefix.json, which are part of the artifact.
  Also recorded: the read-outs that were added after the verifications (e1_main/out/fixup, e2_curves/out/fixup,
  e3_qwen/out/fixup, e3_llava/out/fixup, x_release/out) are recomputed independently from the stored rows
  (x_release/verify/); the paper's sentence on independent verification covers them only if that recomputation
  confirms them.
