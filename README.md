# Rollback, Compensate, or Rescope? Repairing Deployed Logs of Rank-One Model Edits

Artifact of the manuscript of the same title, prepared for submission to PVLDB (Vol. 20): code, edit logs, per-row
results, and the scripts that regenerate every number, every generated table and the data figure of the paper (Figure 1 is
an illustration and is not part of the artifact).

- **Technical report with the full proofs** (reference [11] of the paper): [`paper/techreport.pdf`](paper/techreport.pdf)
- **Protocol, frozen before the final runs, with its time-stamped amendments** (Sec. 5.1 of the paper): [`protocol/PROTOCOL.md`](protocol/PROTOCOL.md);
  "registered" in the paper means written into this protocol before the run (see "The protocol")
- **Verification passes** (Sec. 1 and 5.1 of the paper: each experiment and each later read-out was verified by a separate
  automated pass with its own code): the `verify/` directory of each experiment with its `verdicts.json`, and
  `experiments/x_release/verify/` for the later read-outs; the table of experiments below names each of them.
- **Data**: one release of this repository carries the archives (edit logs, pools, per-row results); see "Data archives".
- The authors' statement on the use of generative AI tools, which were used extensively in this work: see "Use of AI tools".

## What the paper is about

Deployed models outlive the factual updates written into their weights. The paper studies the repair, from reported
complaints, of an already deployed, frozen, single-module history of rank-one updates: deleting recorded updates (rollback),
linear compensation, and predicate-scoped replay (rescoping), under one pre-edit restoration specification. It bounds what
rollback and linear repairs preserve; scoped replay has a locality certificate, a unique least threshold repair and a
calibrated firing bound for exchangeable benign queries. The repairs are compared on the same complaints on real edit logs.

## Layout

| Path | Content |
|---|---|
| `rescope_paths.py`, `LOG_NAMES.json` | the one configuration point for every location (see "Configuration"); the one map between the log names of the paper and the directory names the code uses |
| `rescope/` | the repair library (scoped replay, calibration rules, complaint operators, baseline gates, evaluator, key cache reader), `docs/USAGE.md`, `tests/` |
| `driver/` | the experiment driver `fx` (registry of logs, seeded draws, method constructors, gate-level and behavioural evaluation, statistics, result format), its `USAGE.txt`, tests, the review record `review_lean/`, and the feature packages `feat2/` (whitened and retrieval detectors, recombined negatives) with their verification |
| `experiments/` | one directory per experiment: `code/`, `out/` (results.json, paper_numbers.json, table.txt, fix-up read-outs), `verify/` (the independent verification pass: its code, outputs, `verdicts.json`); `inputs/` in `e5_rescope` and `e8_cost` (two small inputs from outside the experiment directories) |
| `text_logs/` | the build of the public-code text logs: driver scripts around the public ROME code, `out/DEVIATIONS.txt`, the audit (`audit/`: code, outputs, `verdicts.json`) |
| `linear_baselines/` | `harness/` (compensation, rollbacks, evaluation of linear repairs), `streams/` (the linear repair streams: code, per-row records, verification) and `rollback_image_exploratory/` (per-row records of the two rollbacks on image complaints; they are from an earlier, exploratory phase, before the protocol: its `README.txt`) |
| `keycache/` | the capture code of the key caches and their tables (the key tensors are regenerated: "What is not in the repository") |
| `evalkit/` | model and benchmark loaders, prompt building, decoder, registries of the logs (evaluation side only) |
| `paper/` | `techreport.pdf`; `proof_gaps.txt` (the record of the proof markers that the technical report names); `scripts/` (numbers manifest, generators of numbers, tables and figures, statement checks); `checks/`; `expected/` (the generated files the paper was built from); `compare_numbers.py` |
| `protocol/PROTOCOL.md` | the protocol, with a preface on how to read it |
| `tools/gpuq/` | the GPU job queue used for every GPU run; `tools/logs/`: extraction of the pre-edit weight, rebuild of the applied batch updates and of the text logs' weights |
| `examples/` | `quickstart.py` (CPU), `gate_smoke.py` (GPU, 3 minutes) |
| `THIRD_PARTY_NOTICES.md`, `third_party_licenses/` | the terms of the third-party material in the artifact (model-derived files, benchmark items), the licence texts and `Notice.txt` (see "Licence") |
| `FINGERPRINTS_AT_RUN_TIME.json` | the stored hashes and fingerprints whose values the release changed, with their values at run time (see "What can be reproduced", further limits) |

Directory names inside the code are those of the authors' working tree (`02_research/experiments/10_final/e1_main/...`).
`rescope_paths.py` rebuilds that layout as a tree of symbolic links into this repository and into your data, so the
released code runs unchanged; the table `LAYOUT` in that file is the complete map (`10_final/<experiment>` =
`experiments/<experiment>`, `10_final/common` = `driver`, `10_final/cache` = `keycache` + the unpacked pools,
`09_testbed_scale/text_public` = `text_logs`, `07_repair_v3/harness` and `08_rescope/p12_linear_streams` = `linear_baselines/`).

### Names of the logs

The paper's names (Q-A100-r3, L-M1000, G-R100, ...) are the directory names in the archives; the code and the result files
use working names (TB-Q100-ae-l2_3, ..., TX-G2XL-rome-T100); `LOG_NAMES.json` is the map.

One tag in the working names of the logs of the first editor family was replaced by 'ae' in every released file (file names and contents); the replacement is mechanical and changes nothing else. The paper's names are those of LOG_NAMES.json.

Result files carry method digests that were computed under the working names; the field `name_at_run_time` of
`LOG_NAMES.json` keeps these names and the digest functions read them through `rescope_paths.name_at_run_time`
(seeded draws do not depend on a log's name).

## Requirements

Python 3.11.16 with: torch 2.8.0 (CUDA 12.8), numpy 2.4.6, transformers 5.17.0, tokenizers 0.23.2, safetensors 0.8.0,
accelerate 1.15.0, Pillow 12.3.0, scipy 1.17.1; for the retrieval detectors sentence-transformers 6.1.0 and
scikit-learn 1.9.1; for the figures matplotlib 3.11.2. These are the versions the results were produced with; nothing else
was tested. The text logs were written and evaluated with the public ROME code under transformers 4.23.1,
tokenizers 0.13.3, huggingface_hub 0.11.1 and nltk, installed into separate directories (`pip --target`;
`text_logs/code/tp_common.py`, `$RESCOPE_EXT/pydeps_tf4`, `$RESCOPE_EXT/pydeps`).

GPU memory per process (the reservations used with `tools/gpuq/gq`): Qwen2.5-VL-3B 10.5 GB, LLaVA-1.5-7B 17 GB
(12.5 GB and 19.5 GB for the capacity curve of E5), GPT-2 XL in float32 13 GB. All runs were made on two RTX 5090 (32 GB).
QUICK START and the regeneration of the paper's numbers need no GPU, no checkpoint and no benchmark.

## Configuration

`rescope_paths.py` is the only place where locations are defined. Environment variables (all optional when the data is
under `<clone>/data`):

    RESCOPE_HOME    the clone                          (must be set for the shell scripts, e.g. tools/gpuq/gq)
    RESCOPE_DATA    default <clone>/data
    RESCOPE_LOGS    the unpacked archives              default $RESCOPE_DATA/logs
    RESCOPE_EXT     public checkpoints and benchmarks  default $RESCOPE_DATA/ext
    RESCOPE_WORK    caches, memo stores, link tree     default $RESCOPE_DATA/work   (safe to delete)
    RESCOPE_PYTHON  interpreter of queued jobs         default: the interpreter that runs rescope_paths.py

`python rescope_paths.py` prints the configuration and which logs are present. Shell scripts take the interpreter from
`RESCOPE_PYTHON`; set it to your environment's python (or have that python first on `PATH` as `python3`).

## Data archives

The release `v1.0` of this repository holds (SHA-256 in `SHA256SUMS`; each archive has a README inside):

| Archive | Size | Content |
|---|---|---|
| `logs_vl_QA.tar` | 0.65 GB | vision-language logs Q-A100-r3 (development), Q-A100-r10, Q-A100-r1, Q-A200, Q-A500 |
| `logs_vl_QM.tar` | 1.07 GB | Q-M100, Q-M500, Q-M1000, Q-M2000 |
| `logs_vl_LA.tar` | 0.92 GB | L-A100-r3 (development), L-A100-r10, L-A200, L-A500 |
| `logs_vl_LM.tar` | 0.88 GB | L-M100, L-M500, L-M1000 |
| `logs_vl_prefix.tar` | 0.85 GB | the prefix logs of the two development logs after 10, 20, 50 edits |
| `pools_vl.tar` | 0.08 GB | rows of the key caches (every evaluated sequence with its pre-edit answer), the selection files of the key capture, benign pool, hard negatives. The key-cache rows of Qwen2.5-VL-3B also list 2500 rows of a log family (an MMKE-Bench stress log) that is neither reported in the paper nor released |
| `logs_text.tar` | 1.32 GB | the five public-code text logs (G-R100 ... G-R2000 and the development log): factors, solver keys, requests, pools, pre-edit continuations |
| `results_rows.tar.gz` | 0.17 GB | per-row and per-draw result files of the experiments, the captured vectors of E2 and the per-record gate flags of E10 (unpacks into the clone) |

Per edit a vision-language log holds the write vector, the read-out vector, the solver key and the batch; per log the
deployed update, the edit stream and the probe records with the benchmark's item identifiers and prompts. No image, no
checkpoint and no weight is included: the pre-edit weight is extracted from the public checkpoint
(`tools/logs/extract_w0.py`; hashes in `tools/logs/W0_SHA256.json`).

    mkdir -p data/logs
    for f in logs_*.tar pools_vl.tar; do tar -xf $f -C data/logs; done
    tar -xzf results_rows.tar.gz -C .

## Checkpoints and benchmarks

Nothing of this is needed for QUICK START. Place (or link) the public resources under `$RESCOPE_EXT`:

| Resource | Directory under `$RESCOPE_EXT` | Version used |
|---|---|---|
| Qwen2.5-VL-3B (Hugging Face `Qwen/Qwen2.5-VL-3B-Instruct`) | `models/Qwen2.5-VL-3B` | weight files identical to those of the repository at commit 66285546d2b821cf421d4f5eb2576359d3770cd3 (SHA-256 of the local files = the LFS object ids that the hub's API reports for that commit, the head of `main` on 2026-11-12; the revision was not recorded at download time): `model-00001-of-00002.safetensors` 41a8895c164b4d32bae6b302f4603fcbc1797f32dafa45c7e9bcda23c6755df8, `model-00002-of-00002.safetensors` 365531ff8752420e89dee707b79d021fb2d6e25abafe486f080555a4fe6972e4 |
| LLaVA-1.5-7B (Hugging Face `llava-hf/llava-1.5-7b-hf`) | `models/llava-1.5-7b` | weight files identical to those of the repository at commit b234b804b114d9e37bb655e11cbbb5f5e971b7a9 (checked in the same way; the revision was not recorded at download time): `model-00001-of-00003.safetensors` c11dbf016ee7d35ee130c19b67e20eb04873996006f424b7bcbd453b6517ee66, `-00002-` 46df6c6e5fad297fe7fbca4963dcf180cb1d0528cabc884d1f603311fed01328, `-00003-` 4f06177c37ca13944e73b57dd6d051767b950dffc12400c700dfa2509c24c697 |
| GPT-2 XL (Hugging Face `openai-community/gpt2-xl`) | `models/gpt2-xl` | revision 15ea56dee5df4983c59b2538573817e1667135e2 (recorded at download time); `model.safetensors` 0f8b28eb05a8075f48b61b6f35332978c74fc7763fa9fb4051a1c30511736a6a (= the LFS object id of that revision) |
| VLKEB (Huang et al., 2024): `train.json`, `eval.json`, `eval_edit_onehop.json`, `mmkb_images/` | `datasets/VLKEB` | the three JSON files are identical to those of the Hugging Face data set `HymanH/VLKEB-data` at commit cf2d0abe73cb638fcf368d8bc5f9ac3caa204d5a (git blob ids of the local files = the ids that the hub's API reports; the revision was not recorded at download time; the images come from `VLKEB_images.tar.gz` of that repository, not compared). SHA-256: `train.json` 6e40e0e9a28dc3e7523f8d48f032755fe050c5b5728344633f13113dbd6fbc01, `eval.json` 72c47efe00773bb16cd83258aff394a0894b8c29a6b02e6250717a30d1b18f13, `eval_edit_onehop.json` f32256e7c317d5a18c9fe480f3d373050303e93aeaedeef9c9b4afb5c024a670 |
| MME (the shifted benign pool MME-C / MME-T): 4 parquet files | `datasets/MME/data` | file names and sizes in `keycache/cache2/mme/extract_summary.json` |
| ROME code, `https://github.com/kmeng01/rome` | `third_party/rome` | commit 0874014cd9837e4365f3e6f3c71400ef11509e04, unedited (`text_logs/out/provenance.json`) |
| CounterFact as published with ROME (`https://rome.baulab.info/data/dsets/counterfact.json`) and the published second-moment statistics of GPT-2 XL | `data/rome_data`, `data/stats` | `counterfact.json` SHA-256 d017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f (21,919 records); statistics file named in `text_logs/out/provenance.json` |

The checkpoints were downloaded through a mirror of the hub (`https://hf-mirror.com`; the hub itself was not reachable
from the authors' machine), and the identities above were checked through that mirror's API. The released download script
(`text_logs/code/dl_models.py`) uses the endpoint in the environment variable `HF_ENDPOINT` if it is set, otherwise the
official hub.

The pre-edit weight of the edited module identifies the checkpoint the logs were written on:
`python tools/logs/extract_w0.py --backbone qwen25vl --checkpoint $RESCOPE_EXT/models/Qwen2.5-VL-3B` prints its SHA-256 and
compares it with `tools/logs/W0_SHA256.json` (the same for `llava`, `gpt2-xl`).

## QUICK START (10 minutes, no GPU)

Needs the clone, `logs_vl_QA.tar` and a Python with torch and numpy.

    # 1. verify the release checksums (in the directory with the downloaded archives)
    sha256sum -c --ignore-missing SHA256SUMS

    # 2. unpack one archive and look at the configuration
    mkdir -p data/logs && tar -xf logs_vl_QA.tar -C data/logs
    python rescope_paths.py

    # 3. load a released log, check that its deployed update equals the sum of its recorded terms,
    #    and rebuild the thresholds of the repair from the log alone
    python examples/quickstart.py Q-A100-r3

    # 4. regenerate one table of the paper from the released results (Table 2, the main comparison: tab_vlm_main.tex),
    #    then every number and every generated table, and compare with the files the paper was built from
    python paper/scripts/gen_vlm.py
    cat paper/tab_vlm_main.tex
    python paper/scripts/gen_tab_e10.py
    python paper/scripts/gen_numbers.py --no-collect
    python paper/compare_numbers.py

Step 3 prints the log's shapes, the relative deviation between the deployed update and the sum of the logged rank-one
terms (0.0051 for Q-A100-r3: the round-off of the low-precision write, recorded in the log's `metrics.json`), and the
log-only threshold (0.287182 for Q-A100-r3 at q = 0.999). Step 4 writes the table bodies `paper/tab_*.tex` and
`paper/numbers.tex` from `experiments/` and `text_logs/`; `compare_numbers.py` must report 570 identical macros,
0 different, 0 placeholders and 17 of 17 identical table bodies. With `--no-collect` the facts of Table 1 are read from the stored copy
`paper/scripts/data/setup_facts.json`; `python paper/scripts/gen_numbers.py` (without the flag) re-collects them from the
logs themselves and needs every `logs_vl_*.tar`, `pools_vl.tar` and `logs_text.tar` unpacked; without them it stops, names the
archives that are missing and changes nothing.

The library's CPU tests: `python rescope/tests/run_tests.py cpu` (about 3 minutes; 15 tests run on synthetic logs and
pass; 7 tests are reported as NOT RUN because an input of theirs is not released: the editor's cached targets (1 test), files
of the exploratory phase (2), files of the analysis notes (3), and a registry of internal logs that the paper does not report
(1): "What can be reproduced").

A small end-to-end check on a GPU (3 minutes; needs the Qwen2.5-VL-3B checkpoint, VLKEB, `pools_vl.tar`):

    python tools/logs/rebuild_batch_deltas.py Q-A100-r3
    tools/gpuq/gq run --mem 10500 -- python examples/gate_smoke.py Q-A100-r3 --n 20

It loads the checkpoint and the released log (the five equality checks of the loader must be true; the entry
`collapsed_working_point` of the same printed block is `false` for this log, which is the good value), calibrates the
thresholds-only repair on 100 benign queries and evaluates 20 held-out rows; the deployed model's flags must equal the
flags stored in the log (20/20).

## Experiments

One row per experiment of the paper (protocol after the amendments R19 and R21 to R25). Directories are under
`experiments/`. "gate" = gate-level computation on the CPU from the key cache; "behave" = forward passes on a GPU through
the queue (`GQ=tools/gpuq/gq`, `--mem 10500` for a Qwen log, `17000` for a LLaVA log); "analysis" = CPU, from the
per-row files (`results_rows.tar.gz`), rebuilds the summary files of `out/` (`paper_numbers.json`, `results.json`,
`table.txt`). `<log>` is a working name (`LOG_NAMES.json`). Run times are the wall-clock of our runs on two RTX 5090 that were shared
between experiments (first to last time stamp of the run logs); they are not benchmarks. Scripts are run from the
experiment's `code/` directory (E10: from `e10_text`, with the paths as written; the linear streams: from `linear_baselines/streams`).

### Levels of reproduction

The column "Levels" says what can be reproduced for an experiment. A level is stated as tested only where it was run in a
fresh copy of this repository with the archives unpacked, before the release.

- **[S] from the shipped summaries** (no GPU, no checkpoint, no benchmark): the paper's scripts read the summary files that
  are in git and regenerate every number (570 macros), the 17 generated table bodies and the data figure of the paper
  (QUICK START, step 4; row "Numbers, tables and figures"). Tested: identical to the files the paper was built from; for the
  figure this means its data file (`paper/figs/evidence_ab.json`). The figure scripts draw with the paper's fonts only if a TeX
  tree with the Libertine fonts is given in the environment variable `ACMFONTS`; without it they say so and fall back to
  DejaVu Sans (the same data, another typography, so the PDF is not byte-identical to the paper's).
- **[R] analysis from the per-row files** (CPU; needs `results_rows.tar.gz` unpacked into the clone and the log archives
  under `data/logs`): the commands marked "analysis" (and the "read-outs") rebuild the experiment's summary files. Tested
  for every command so marked: exit code 0, and every file of paper numbers (`paper_numbers.json`, `paper_numbers_fixup.json`)
  is rewritten byte for byte as shipped (one value of E2's read-outs, `EtwoFxHonePrefixFileTime`, is the modification time of a
  file and so depends on the copy; the description strings of `experiments/e9_maintenance/out/paper_numbers_doc.json`, which
  holds no number, name the directory of the copy where the shipped file has the mark `[redacted: local path]`); the other
  summary files (`results.json`, `table.txt`, the files of `out/fixup/`) come out with the same records and differ from the
  shipped ones only in what depends on the run (time stamps, code fingerprints, directories) and in the places that the
  column "Levels" names (a labelled cross-check of E5, the check of modification times of the read-outs). **Before the analysis of E1, E2 and E3 the applied
  batch updates must be rebuilt from the public checkpoints**: `python tools/logs/rebuild_batch_deltas.py` (CPU; 2 minutes
  on our machine; 53 GB under `data/logs/batch_deltas`; needs the two vision-language checkpoints under `$RESCOPE_EXT`);
  without them these scripts stop with "log ... does not exist yet". The other analysis commands need no checkpoint.
  The analysis of E3 on the LLaVA logs also reads the benchmark's records: the three JSON files of VLKEB under
  `$RESCOPE_EXT/datasets/VLKEB` (no image); without them it stops and names the file.
- **[G] re-running the experiment on a GPU** from the released logs, pools and code, after the key caches have been
  captured (`keycache/code`) and the batch updates rebuilt. Tested in a copy: only the short end-to-end check
  `examples/gate_smoke.py` and a 20-sequence key capture; the GPU runs of the experiments themselves were not repeated
  from the artifact. [G] is not available where the table says so (the editor is not released).

A renderer or generator that does not find one of its inputs that holds results stops and leaves every existing output
file as it was; where a script stops by itself, the message names the input. The status records of finished jobs
(`job_*.json`, `behave_*_job_*.json`) and single stored test results are optional: without one the table omits or replaces
its line; after a stop by the postcondition (check (3) below) the message lists the summaries that would have shrunk.
`rescope_paths.guard_outputs` enforces this in three ways: (1) a script that aggregates
per-row directories first checks that they are unpacked, and otherwise runs nothing (exit code 2; the message names the
directories and `results_rows.tar.gz`); (2) the work runs in a child process and the output files (for the analysis of E3 on Qwen2.5-VL-3B also its per-log cache
`out/cache`) are put back if it fails;
(3) if it ends without an error but a JSON summary of the output directory is gone or would hold fewer entries, records or
non-null values than before the run, everything is put back as well and the exit code is 3 (`RESCOPE_ALLOW_SHRINK=1` allows
a deliberate shrink, for instance after re-running an experiment on fewer logs). `gen_numbers.py` ends with exit code 1 when
a macro is left without a source and `compare_numbers.py` when a placeholder is left; `gen_numbers.py` without
`--no-collect` stops before it writes anything when a log archive is not unpacked. No command of the column ends
successfully after dropping numbers. Tested for every command of the column in a clone in which `results_rows.tar.gz` was
not unpacked (a plain clone, and the state that QUICK START leaves): a command that needs the archive or the logs ends with
a non-zero exit code and every file of the repository is byte for byte as before; a command that needs neither (the
read-outs of E2, the linear side of E5, E9, the figure scripts) runs and rewrites its summaries with the same numbers;
QUICK START, step 4, gives 570 identical macros afterwards. Also tested in a full copy with single inputs moved away. What
the check (3) cannot see: a number that changes while the count of values stays the same.

Numbers of the paper that come from shipped files only, at every level: the three numbers of the registered capacity
prediction (`experiments/e5_rescope/inputs/capacity_predictions.json`, an extract of the authors' analysis file; macros
`thCapPredQlo`, `thCapPredQhi`, `thCapPredL` and the keys `capacity.*.predicted_*` of E5); the published figures of the
gated key-value editor in the cost table (`experiments/e8_cost/inputs/published_figures.json`: quoted, not measured); the
rollback-on-image-complaints numbers of Table 4 (b), which are computed from records of the exploratory phase
(`linear_baselines/rollback_image_exploratory/`); the numbers of the linear re-editing baseline (per-row records of
`linear_baselines/streams`; the editor is not released); and every number whose source file is written by a gate-level or
GPU script and not by an analysis command: `paper/NUMBERS_SOURCES.txt` lists, for each of the 570 macros, its source and
whether an analysis command of this table rewrites that source ([R]), whether it is collected from the released logs ([C]),
or whether it is used as shipped ([S] only). For 48 macros that the table generator computes from several files the list
names the facts file they pass through instead of the single files and counts them as [S], which is the safe side (most of
them are the deployed scores of Table 1 from `experiments/x_release/out/logs_b1.json`, which the read-outs command rewrites).

The table (column "Levels": what the section above defines):

| Experiment | Directory | Commands | Run time | Output | Verification | In the paper | Levels |
|---|---|---|---|---|---|---|---|
| Sanity run on the development logs | `e0_sanity`, `e0_verify` | `python e0_gate.py <log>`; `$GQ run ... -- python e0_behaviour.py <log>`; `python e0_render.py` | 0.9 h | `out/results.json`, `out/table.txt` | `e0_verify/verdicts.json` | Sec. 5.1 (the registered sanity comparison; no table) | [S]. [R] `python e0_render.py` counts the stored results of the library's tests, which are not in the repository: without them it stops and says so. Tested after `python rescope/tests/run_tests.py cpu` in the copy: exit code 0; its line on the library then reports the tests of that run (22 of 22 results), so its table differs from the shipped one in that line. [G] |
| E1 main comparison at a matched false-firing budget (development logs, working points) | `e1_main` | gate: `python e1_gate.py <log>`; behave: `$GQ run --mem 10500\|17000 -- python e1_behave.py <log>`; `python e1_frontier.py`; analysis: `python e1_analyze.py && python e1_render.py`; read-outs: `python fixup_compute.py && python fixup_render.py` | 2.9 h | `out/paper_numbers.json`, `out/table.txt`, `out/fixup/` | `verify/verdicts.json` | Table 2 (Sec. 5.3); technical report App. C (full table, draws, McNemar tests, frontier figure) | [S]. [R] tested, after the batch-update rebuild; one check of the read-outs counts 1762 compared E2 records where the shipped file says 1726 (E2's record file is the final one; 0 differ in both). [G] |
| E2 evidence curves: repairs of n reported complaints (development logs) | `e2_curves` | `$GQ run --mem 10500 -- bash code/run_log.sh <log>`; `python c_gate.py`; analysis: `python render.py`; read-outs: `python fixup_compute.py && python fixup_render.py` | 4.7 h | `out/paper_numbers.json`, `out/running_example.json`, `out/fixup/` | `verify/verdicts.json` | Figure 2 (a), (b); the running example of Sec. 2; Sec. 5.2 | [S]. [R] tested, after the batch-update rebuild (reads the captured vectors `out/capture/` of `results_rows.tar.gz`). [G] for parts b and c; part a is the row "Linear repair streams" |
| E2 on prefix logs (10, 20, 50 edits) | `e2_prefix` | `bash code/chain.sh <mem> 0 <prefix log> ...`; `python gate.py <prefix log>`; analysis: `python aggregate.py && python render.py` | 8.2 h | `out/h1_prefix.json`, `out/paper_numbers.json` | `e5_rescope/verify_secondary/out/verdicts.json` (check 6: the prefix logs) | Sec. 5.2 (H1 on prefixes); technical report App. C (prefix table) | [S]. [R] tested. [G] |
| E3 scale and second editor family, Qwen2.5-VL-3B | `e3_qwen` | gate: `python e3_gate.py <log>`; `python e3_factor.py <log>`; behave: `bash run_chain.sh <log> ...`; analysis: `python e3_analyze.py && python e3_render.py`; read-outs: `python fixup_compute.py && python fixup_render.py` | 4.7 h | `out/paper_numbers.json`, `out/scale_series.json`, `out/retention_series.json`, `out/fixup/` | `verify/verdicts.json` (both backbones) | Table 3 (a) (Sec. 5.4); technical report App. C (per log and method) | [S]. [R] tested, after the batch-update rebuild. [G] |
| E3, LLaVA-1.5-7B | `e3_llava` | `bash code/run_all.sh` (gate: `python e3_gate.py <log>`; behave: `python e3_behave.py <log>`); analysis: `python e3_analyze.py && python e3_render.py`; read-outs: `python fixup_compute.py && python fixup_render.py` | 3.3 h | as above | `e3_qwen/verify/verdicts.json` | Table 3 (a) | [S]. [R] tested, after the batch-update rebuild. [G] |
| E5 checks of the theory on real logs (rebase, lineage size, validity of the calibration, least repair, re-reporting, capacity curve) | `e5_rescope` | `bash code/run_a.sh`, `run_c.sh`, `run_d.sh` (CPU); `$GQ run --mem 12500 -- bash code/gpu_all.sh qwen25vl TB-Q100-ae-l2_3` (19500: llava); analysis: `python render.py` | 2.7 h | `out/paper_numbers.json`, `out/capacity_curve.json`, `out/validity_distribution.json` | `verify_secondary/out/verdicts.json` (E5, E6, E8, E9 and the prefix logs) | Sec. 5.2 (capacity), Sec. 5.6; technical report App. C (validity figure, lineage size) | [S]. [R] tested. [G] |
| E5, linear side (real-log statistics of the linear theory) | `e5_linear` | `bash code/run_opt.sh`, `python e5_capacity.py`; analysis: `python e5_aggregate.py && python e5_render.py` | 0.8 h | `out/paper_numbers.json` | `verify/verdicts.json` | technical report App. A | [S]. [R] tested; one cross-check with a file of the exploratory phase is kept from the shipped summary and labelled "from the shipped summary; input not released". [G] |
| E6 ablations of the method (development logs) | `e6_ablations` | `python e6_gate.py <log>`; `$GQ run ... -- bash code/run_gpu.sh <log>`; analysis: `python e6_render.py` | 0.7 h | `out/paper_numbers.json`, `out/table.txt` | `e5_rescope/verify_secondary` | Sec. 5.7; technical report App. C (ablation table) | [S]. [R] tested. [G] |
| E8 cost (storage, CPU cost of a repair, latency) | `e8_cost` | `bash code/run_cpu.sh`; `$GQ run --mem 10500 -- python lat.py qwen25vl` (17000: llava); analysis: `python render.py` | 2.5 h | `out/paper_numbers.json`, `out/scale_series.json` | `e5_rescope/verify_secondary` | Table 4 (c) (Sec. 5.7); technical report App. C (cost table) | [S]. [R] tested. [G] |
| E9 maintenance of a scoped log (revocation, lineage) | `e9_maintenance` | `bash code/run_gpu.sh <log> <mem> <tag>`; analysis: `python e9_render.py` | 0.5 h | `out/paper_numbers.json`, `out/table.txt` | `e5_rescope/verify_secondary` | Table 4 (b) (Sec. 5.7) | [S]. [R] tested. [G] |
| E10 public text logs (GPT-2 XL, ROME): equal evidence, scale, relational maintenance workload | `e10_text` (run in `e10_text/run`) | see `e10_text/USAGE.txt`: `python scripts/build_keycache.py --log <log>`; `python scripts/run_rowset.py --log <log> --parts gate\|behave,public,retention\|gen`; `run/code/run_*.py`; analysis: `python run/code/summarise.py && python run/code/render.py` | 2.4 h (development log) + 3.2 h (reported logs) | `run/out/paper_numbers.json`, `run/out/summary/`, `run/out/table.txt` | `run/verify/verdicts.json`; audit of the logs: `text_logs/audit/verdicts.json` | Figure 2 (c); Table 3 (b); Table 4 (a); Sec. 5.5 and 5.7 | [S]. [R] tested. [G] |
| Read-outs added after the verifications (multiplicity, deployed scores at batch size 1, release against change) | `x_release` | analysis: `python release_compute.py && python release_render.py` | minutes | `out/paper_numbers_fixup.json`, `out/logs_b1.json`, `out/holm.json` | `x_release/verify/` (independent recomputation: `out/`, `code/`) | Table 1 (deployed scores), Table 2, Sec. 5.2 to 5.4; technical report App. C | [S]. [R] tested; the check that compares file modification times with the start of that work is meaningful in the authors' working tree only: in a copy it is skipped with a printed note and recorded with the status `skipped`, not `ok` (regenerated `checks.json` and `FIXUP.txt` of `x_release`, `e3_qwen`, `e3_llava`; the shipped files are those of the working tree, where the check ran and passed) |
| Linear repair streams (the records behind E2 a) | `linear_baselines/streams` | `python code/run_streams.py --log <log>`; analysis: `python code/aggregate.py` | 21 h (including a resumed run) | `out/<log>/*.json`, `out/results.json`, `results.md` | `verify/VERIFY_REPORT.md` | Figure 2 (a), (b), Table 2 (rollback and compensation rows) | [S]. [R] tested. [G] for rollback and compensation only; the re-editing baseline needs the editor, which is not released |
| Build and audit of the text logs | `text_logs` | `python code/run_stream.py --name stream`; `python code/build_logs.py`; audit: `audit/*.py` | 4.1 h | `out/text_logs.json`, `out/DEVIATIONS.txt` | `audit/verdicts.json` | Table 1 (text rows); Sec. 5.1 | [G] not tested in a copy (public ROME code, public checkpoint, public data); the audit's verdicts are shipped |
| Numbers, tables and figures of the paper | `paper` | `python paper/scripts/gen_numbers.py`; `python paper/scripts/figs/fig_evidence.py`; `python paper/scripts/figs/fig_report.py`; `python paper/compare_numbers.py` | 1 min | `paper/numbers.tex`, `paper/tab_*.tex`, `paper/figs/` | compared with `paper/expected/` | every number; Tables 1 to 4; Figure 2 (a), (b); the figures of the technical report's App. C | [S] tested: 570 of 570 macros and 17 of 17 table bodies identical to `paper/expected/` |

Figure 1 of the paper is an illustration (drawn by the first author; its draft was made with an image model); it is not
generated from data and has no script here. The statement checks `paper/scripts/check_vlm_statements.py` and
`check_e10_statements.py` compare the wording of the paper with the experiments' JSON and read the paper's section files;
`paper/checks/check_tab_logs_rows.py` reads the run logs of the log builds. Neither input is released, so these three are
a record of what was checked; `paper/scripts/final_spot_checks.py`, `check_method_pseudocode.py`,
`paper/checks/check_two_modules.py` and `check_lost_least_repairs.py` run as they are.

## What can be reproduced

**From this repository and its release alone, given the public checkpoints and benchmarks:**

- everything on the public-code text logs: the logs themselves from the public ROME code, a public checkpoint and a public
  data set (`text_logs/`), their audit, and experiment E10;
- every gate-level and behavioural experiment on the released vision-language logs (E1, E2 part b and c, the prefix logs,
  E3, E5, E6, E8, E9), after the key caches have been captured (`keycache/code`) and the batch updates rebuilt
  (`tools/logs/rebuild_batch_deltas.py`);
- the rollback and compensation repairs of the linear baselines (`linear_baselines/harness`: `COMP`, `ROLLBACK_FO`,
  `ROLLBACK_BEH`, `ROLLBACK_ALL`);
- every number of the paper's text, every generated table and the data figure, from the shipped summary files
  (`paper/scripts`; no GPU; level [S] above), and the summary files themselves from the released per-row results where the
  table of experiments says [R] tested.

The first three items say what the released code, logs and pools allow (level [G]); from a copy of the artifact only a short
GPU check was run ("Levels of reproduction").

**Not from this repository:**

- re-writing the vision-language logs;
- the linear re-editing baseline (`REEDIT` in `linear_baselines`, and `COMP` with `solve="alphaedit"`), which re-runs the
  editor: its per-row records and the hashes of its repaired weights are released instead
  (`linear_baselines/streams/out/<log>/REEDIT_*.json`, `corrections_sha256.json` there and in `experiments/e2_curves/out`,
  `experiments/e2_prefix/out`); the entry points raise `NotImplementedError` with this explanation;
- checks that compare with the editor's cached targets (the rebase check against stored targets in E5 a, E3 and library
  test a12; their stored outcomes are in the released result files).

The reason: the editor configuration that wrote these logs is described in a related submission of the authors that is under review; the logs are released as data.

For the same reason some fields of the released metadata hold the string `see README (related submission)`, and
passages of the protocol and of result files are replaced by the mark
`[redacted: describes a related submission under double-blind review]`. Nothing that a repair method of this paper reads
is redacted.

Further limits, stated plainly:

- The numbers that Appendix A of the technical report quotes from exact-arithmetic checks of the linear theory come from
  scripts of the internal analysis material that is not part of this repository. Two small files of that material are
  released as inputs because renderers read them: the registered capacity prediction
  (`experiments/e5_rescope/inputs/capacity_predictions.json`) and the published figures quoted in the cost table
  (`experiments/e8_cost/inputs/published_figures.json`).
- Seven CPU tests of the library and some comparison sections of the renderers refer to files of the exploratory phase
  (before the protocol); the tests are reported as not run, and one comparison of E5 (linear side) keeps its stored outcome
  with a label. The sanity run E0 can be rendered again only after the library's tests have been run.
- The code fingerprints stored in the result files and in `driver/READY.json` (SHA-1 of the library and driver sources) are
  those of the authors' working tree, with one kind of exception. Where a stored hash was the hash of a single file whose
  bytes changed in the release (the replaced tag in the names of the logs), it was replaced by the hash of the released file. Among
  code fingerprints this concerns exactly: `methods.py`, `prefix.py` and `review1.py` in `source_files` of
  `driver/READY.json`, and `e0_gate.py` and `e0_render.py` in five files of `experiments/e0_sanity/out` (`checks.json`,
  the two `gate_<log>.json`, `referee.json`, `results.json`). Every other code fingerprint is the working tree's, including
  the combined ones (`6a71f0d2cd71` for the driver, `c76c4da16bd6` for the library) and `__init__.py` and `logs.py` in
  `source_files`, with the exception of the regenerated E2 fix-up record (`experiments/e2_curves/out/fixup/checks.json`,
  `table_fixup.txt`), which carries the combined fingerprints of the released sources (`a7cfd4d450d1` for the driver,
  `ec18057f0f60` for the library); no released code compares the per-file fingerprints. The released sources differ from the working tree by
  the path configuration, by the marked redactions and by the replaced tag, so a re-run records other fingerprints (and the
  test table marks older records as produced by another version). Stored hashes of data files whose bytes changed (the
  indexes of the key caches, summary files that a verification pass had hashed) were replaced in the same way, so that the
  loaders' consistency checks hold, with the exceptions that the next item names.
- Stored hashes of files that the release changed were recomputed so that the released checks pass;
  `FINGERPRINTS_AT_RUN_TIME.json` (root of the repository) keeps the values at run time: 70 values in 23 files
  (50 data hashes, 20 code fingerprints), each with the released file, the key, the value at run time, the
  released value, the kind and the reason. Apart from these, result files carry the code fingerprints of the working tree
  at run time, unchanged. The same file lists, under `kept_at_run_time_value`, 14 stored values (85 places) that
  are hashes of a file as it was at run time and were not replaced although the released file has other bytes: 11 per-file
  code fingerprints (sources changed by the path configuration or by marked cuts) and 3 data hashes (the MD5 of the Qwen
  key-cache index, 13 places: the released index differs by the path token in 2000 rows and by the tag, the released
  `meta.json` holds the MD5 of the released index and keeps the run-time value as `index_md5_before_redaction`; and two
  files of `experiments/e5_linear`). These values are true for the run, not for the released file. One of them is compared by
  a loader: `driver/fx/pools.py` (the parts of the benign pool) checks the value in `keycache/cache2/benign_parts/qwen25vl.json`
  against the index of the key cache in use and stops because they differ, with the released index as with a regenerated one:
  before a gate-level run on a Qwen log, rebuild the parts file with `cd driver/feat2 && python -m white.benparts build` (the
  split is seeded and comes out identical; the file then differs in the recorded MD5 and in its list of backbones only).
  Tested with the released index, not with a regenerated cache (see the table "What is not in the repository").
- `driver/feat2/white` and `keycache/cache2/whiten` concern the second moments of benign keys that the whitened baseline
  detectors of this paper use (protocol, R3); they are not the statistics of the editor that wrote the logs.
- The revisions of the two vision-language checkpoints and of VLKEB were not recorded when they were downloaded. The files
  that were used are identical to those of the repositories and commits named under "Checkpoints and benchmarks" (hashes
  compared on 2026-11-12), and `tools/logs/extract_w0.py` checks that a checkpoint has the pre-edit weight the logs were
  written on.
- Texts that the models generated are stored as generated, except for marks. Texts generated by GPT-2 XL from CounterFact
  prompts name real persons and organisations; their content is invented by the model and states nothing about anyone.
  E-mail addresses, telephone numbers, identity-document numbers and similar strings that the model produced were replaced
  by marks; if you find one that was missed, or if you hold rights in material contained in the data and want it changed or
  removed, please open an issue or write to the first author. The marks are `[redacted: e-mail address in generated text]`
  (47 places), `[redacted: telephone number in generated text]` (6 places: five numbers that a text puts next to a named
  person) and `[redacted: identity-document number in generated text]` (11 places: passport numbers that a text gives for a
  named person). What stays: telephone numbers and addresses of organisations, offices and businesses, names, dates, web
  sites and social-media handles. Word counts stored in `text_logs/out/text_logs.json` ("mean_words"), the word
  statistics of the text-log audit (`text_logs/audit/out/traps.json`, block "generation_entropy") and the n-gram
  entropies stored next to the generated texts are those of the unmarked texts.

### Named in released documents, not released

Released records name files that are not part of the artifact: the workflow scripts that drove the AI assistant
(`02_research/tools/workflows/*.js`, named in `driver/USAGE.txt`), the analysis notes under `03_analysis/` (theory rounds,
literature notes; named in code comments, in result files and in `paper/proof_gaps.txt`), `CHECK_LOG.tex` and the other
check and review files of the paper's directory (named in `paper/proof_gaps.txt` and in the paper's scripts), the LaTeX
sources of the paper, and the builders of the vision-language logs. `paper/proof_gaps.txt` itself is released.

### What is not in the repository, and how it is regenerated

| Not released | Size in our tree | Regenerated by |
|---|---|---|
| key caches of the vision-language backbones (module-input keys of every evaluated sequence) | 2.0 GB (Qwen), 1.1 GB (LLaVA) | `keycache/code`: `python fc_capture.py qwen25vl Q` and `python fc_capture.py llava L` (GPU, through the queue; resumable; needs the batch updates of Q-M2000 / L-M1000) -> `fc_assemble.py <backbone> --partial` -> `fc_verify.py`; the selection of sequences, the rows and the pre-edit answers are in `pools_vl.tar`. The capture loads the released logs Q-M2000 / L-M1000, whose edit streams are those of the internal logs the build used; a 20-sequence capture reproduces the original keys bit for bit (smoke test). The 2000 sequences of an MMKE-Bench stress log that the paper does not report cannot be captured (that log is not released), so a regenerated Qwen cache is marked incomplete; whether every experiment script accepts such a cache was not tested |
| key cache and stores of the text logs | 15 GB + 2.2 GB | `experiments/e10_text/scripts/build_keycache.py`, `run_rowset.py` |
| feature caches of the feature packages (whitening statistics of benign keys, encoder embeddings, recombination keys, MME keys and images) | 4.1 GB | `driver/feat2/*/` (each package's `USAGE.txt`), `keycache/cache2/mme/code` |
| applied batch updates of the vision-language logs | 53 GB for all released logs (0.9 GB for Q-A100-r3) | `tools/logs/rebuild_batch_deltas.py` (CPU; bit for bit, checked against the deployed update) |
| weights of the text logs (pre-edit, deployed) | 0.2 GB | `tools/logs/rebuild_text_weights.py` (CPU; checked against the recorded SHA-256) |
| saved corrections of the linear repairs and of E2 (repaired weights, adapter factors) | 5.3 GB + 1.1 GB + 15 GB | `linear_baselines/streams/code/run_streams.py`, `experiments/e2_curves/code/job.py`, `experiments/e2_prefix/code/job.py`; SHA-256 of every tensor in `corrections_sha256.json` |
| memo stores of behavioural results, harness caches | 55 MB + 7 GB | created on use under `$RESCOPE_WORK` |
| run logs | 10 MB | written by the scripts into each `logs/` directory |
| model checkpoints, benchmark images | | public: "Checkpoints and benchmarks" |

## The protocol

Protocol frozen before the final runs; amendments carry the time stamps of the authors' machine; no external registry was used; "registered" in the paper and in the records means written into this protocol before the run.

`protocol/PROTOCOL.md` was frozen on 2026-11-10 before the final runs (Sections 0 to 5: method under test, data and pools,
compared methods, experiments, statistics, boundaries). Everything decided later is in Section 6 as an append-only list of
dated amendments, each with its time stamp (UTC) and the state of the runs at that moment; an amendment overrides earlier
text, and a withdrawn amendment stays in the file with its withdrawal. Read Sections 0 to 5 first, then the amendments in
order; R19 and R21 to R25 define what was finally run, and the amendments of 2026-11-12 record the registered verdicts
(including outcomes against the method) after each verification. The preface of the file explains the redaction marks.

## Use of AI tools

The manuscript's statement on the use of AI tools (its Sec. 7):

> The first author initiated and supervised this work and set its direction, scope and constraints. Under this supervision, generative AI was used extensively: an AI assistant (Claude, Anthropic) to develop the approach, draft proofs, code and text, and run the experiments and their independent verifications (separate instances with their own code); a second language model as a critic; an image model for the draft of Figure 1 of the paper. The authors reviewed the work and are responsible for its content.

**The first author's part.** The first author initiated the project, set its direction, scope and constraints (the target
venue, the boundary to the authors' related work, what is released), provided the editing setup and the computing
environment on which the vision-language logs build, supervised the work, took the decisions on venue, scope, disclosure
and release, drew the final version of Figure 1 of the paper, and reviewed the manuscript.

What this means for a user of the artifact:

- **Machine-written parts.** Under the first author's supervision, the code of this repository (library, driver,
  experiments, harness, tools, paper scripts), the experiment runs and their analyses, the verification passes under each
  experiment's `verify` directory (and `verify_secondary`, `x_release/verify`, `text_logs/audit`, `driver/review_lean`),
  the protocol's amendments, and the proofs of the technical report were written by the AI assistant. This README and the
  staging of the artifact (path configuration, redactions, archives) were also produced by the AI assistant.
- **What "independent verification" means.** "Independent verification" in the paper and `verify` / `verdicts` /
  "audit" / "review" / "referee" in this repository denote re-implementations and re-runs by separate instances of the
  AI assistant, each with its own code and without the code of the pass it checks. They are not checks by other people.
  Where a file speaks of an "agent", a "verification agent", a "coordinator" or a "referee pass", it means such an
  instance. "Review round 1" in the protocol is a simulated review by language models.
- **What you can check yourself.** Every number of the paper can be regenerated from the shipped summary files with the
  released scripts (`paper/scripts/gen_numbers.py`, `paper/compare_numbers.py`); the summary files can be rebuilt from the
  released per-row files with the analysis commands, and the experiments can be re-run from the released logs, to the
  extent that the table of experiments states and was tested ("Levels of reproduction").
- **Words for roles in the released records.** "Principal investigator", "PI" and "coordinator" denote the coordinating
  instance of the AI assistant. "Prover", "referee", "reviewer" (of the driver or of the theory), "verifier", "auditor" and
  "judge" denote separate instances of the assistant; pronouns in the records ("he", "his script", "my code") refer to these
  instances. "External referee" and "external critic" denote a second language model. "The first author" is the human who
  supervised the work; the records attribute to the first author the decisions that the first author took.

## Licence

MIT (see `LICENSE`) for the authors' code and documentation. Everything else (the recorded updates of model weights, the
outputs of the models and the items of the benchmarks, in this repository and in the archives) is under the terms that
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) states, with the licence texts and the notice file in
`third_party_licenses/`; every archive carries both at its top level. Among this material are items of MME and MMKE-Bench,
whose sources state no licence.

- Built with Qwen. Files derived from Qwen2.5-VL-3B (the logs and results whose names begin with Q or carry the family
  letter Q as a part of the name, as in `__Q__` and `c4_Q/`, and the Qwen parts of the pools) may be used for
  non-commercial (research or evaluation) purposes only (Qwen RESEARCH LICENSE AGREEMENT).
- Files derived from LLaVA-1.5-7B (names beginning with L or carrying the family letter L as a part of the name, as in
  `__L__` and `c4_L/`) are under the LLAMA 2 Community License and its Acceptable Use Policy.
- Texts sampled from GPT-2 XL were created using GPT-2 (OpenAI's Modified MIT License): the generated texts and decoded
  outputs that `THIRD_PARTY_NOTICES.md` names and, beyond its list, the context templates that the public ROME code samples
  from the model (the fields `context_templates`, `generate_fast_env` and `builder_templates_tf4` in the records of the
  text logs, in the repository and in the archives).

## Citation

    @misc{cao2026rescope,
      author = {Minghao Cao and Bei Hua},
      title  = {Rollback, Compensate, or Rescope? Repairing Deployed Logs of Rank-One Model Edits},
      year   = {2026},
      note   = {Manuscript prepared for submission to PVLDB (Vol. 20)}
    }

Contact: Minghao Cao (caominghao@mail.ustc.edu.cn), Bei Hua (bhua@ustc.edu.cn).
