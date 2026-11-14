# Rollback, Compensate, or Rescope? Repairing Deployed Logs of Rank-One Model Edits

Research artifact for the manuscript prepared for PVLDB (Vol. 20): repair code, edit logs, experiment results,
and scripts for regenerating the paper’s numbers, tables and data figures.

The paper studies how to repair a deployed log of rank-one model updates from user complaints while retaining
useful edits. It compares **rollback**, **linear compensation** and **rescoping** under a common pre-edit
restoration criterion. Scoped replay provides a locality certificate, a least threshold repair for fixed
detectors, and a firing bound for exchangeable benign queries.

[Technical report and proofs](paper/techreport.pdf) ·
[Data release v1.0](https://github.com/12m45d/rescope-edit-log-repair/releases/tag/v1.0) ·
[Reproduction guide](REPRODUCIBILITY.md) ·
[Experiment protocol](protocol/PROTOCOL.md)

## Quick start

Run from the repository root. Neither quick start below needs a GPU or model checkpoint. The log example
uses Python, PyTorch and NumPy; see the [tested environment](REPRODUCIBILITY.md#requirements).

### Check a released log

Download `logs_vl_QA.tar` and `SHA256SUMS` from the [release](https://github.com/12m45d/rescope-edit-log-repair/releases/tag/v1.0)
into the repository root, then run:

```bash
sha256sum -c --ignore-missing SHA256SUMS
mkdir -p data/logs
tar -xf logs_vl_QA.tar -C data/logs
python rescope_paths.py
python examples/quickstart.py Q-A100-r3
```

This checks the recorded update against its rank-one terms, allowing for recorded write round-off, and
rebuilds thresholds from the log. On macOS, use `shasum -a 256 --ignore-missing -c SHA256SUMS` for checksums.

### Regenerate the paper’s numbers and tables

These commands use the summary files included in the clone and need no data archives or model checkpoints:

```bash
python paper/scripts/gen_vlm.py
python paper/scripts/gen_tab_e10.py
python paper/scripts/gen_numbers.py --no-collect
python paper/compare_numbers.py
```

Expected result: **570 identical macros, 0 differences, 0 placeholders, and 17 identical table bodies**.
The guide also covers [CPU tests and a GPU smoke check](REPRODUCIBILITY.md#quick-start-10-minutes-no-gpu).

## Repository

| Path | Purpose |
|---|---|
| [`rescope/`](rescope/) | Scoped replay, calibration, repair operators and tests; [API usage](rescope/docs/USAGE.md) |
| [`driver/`](driver/), [`experiments/`](experiments/) | Experiment execution, results and verification records |
| [`linear_baselines/`](linear_baselines/) | Rollback and compensation baselines |
| [`text_logs/`](text_logs/) | Public ROME log construction and audit |
| [`keycache/`](keycache/), [`evalkit/`](evalkit/) | Key capture, model and benchmark loading |
| [`paper/`](paper/) | Technical report, result generators and expected outputs |
| [`examples/`](examples/) | CPU quick start and GPU smoke check |
| [`rescope_paths.py`](rescope_paths.py), [`LOG_NAMES.json`](LOG_NAMES.json) | Path configuration and paper-to-code log names |

By default, archives go under `data/logs`, public models and benchmarks under `data/ext`, and caches under
`data/work`. Override these locations with `RESCOPE_LOGS`, `RESCOPE_EXT` and `RESCOPE_WORK`; see
[configuration](REPRODUCIBILITY.md#configuration).

## Reproduction

- **From summaries:** regenerate the paper’s numbers, generated tables and data figures. The release checks
  matched the expected outputs.
- **From per-row results:** unpack `results_rows.tar.gz` and the required logs, then run the analysis commands
  in the [experiment table](REPRODUCIBILITY.md#experiments).
- **From model inference:** obtain the public checkpoints and benchmarks, rebuild batch updates and capture
  keys. The release was checked with a short GPU smoke run and key capture; full GPU experiments were not
  repeated in a fresh artifact copy.

The vision-language log builder and the editor needed for linear re-editing are not released because the
editor configuration belongs to a related submission under review. The logs and stored results are available;
some checks also depend on unreleased internal inputs. Metadata marked `see README (related submission)`
refers to the editor restrictions.
See [reproduction limits](REPRODUCIBILITY.md#what-can-be-reproduced),
[missing inputs and cache regeneration](REPRODUCIBILITY.md#what-is-not-in-the-repository-and-how-it-is-regenerated),
and [recorded fingerprints](FINGERPRINTS_AT_RUN_TIME.json).

The [protocol](protocol/PROTOCOL.md) records the methods, endpoints and decision rules fixed before the runs,
followed by time-stamped amendments. “Registered” refers to this internal protocol.

## Use of AI tools

The authors used AI agents to help write and debug code while designing the experiments. AI agents also helped the authors with the related work and references. Once the manuscript was complete, the authors used AI tools to polish the text and improve its clarity and readability. All work done with AI assistance was carefully reviewed by the authors, who take full responsibility for the code, experiments, references, and all other content submitted as part of this paper.

## Licence

The authors’ code and documentation use the [MIT licence](LICENSE). Model-derived files, generated outputs
and benchmark items follow [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the terms in
[`third_party_licenses/`](third_party_licenses/):

- **Built with Qwen:** Qwen2.5-VL-3B-derived material is for non-commercial research or evaluation under the
  Qwen RESEARCH LICENSE AGREEMENT.
- **LLaVA-1.5-7B-derived material:** LLAMA 2 Community License and Acceptable Use Policy.
- **GPT-2 XL-generated text and context templates:** OpenAI’s Modified MIT License.

MME and MMKE-Bench sources state no licence; consult the third-party notices before use.

## Citation

```bibtex
@misc{cao2026rescope,
  author = {Minghao Cao and Bei Hua},
  title  = {Rollback, Compensate, or Rescope? Repairing Deployed Logs of Rank-One Model Edits},
  year   = {2026},
  note   = {Manuscript prepared for submission to PVLDB (Vol. 20)}
}
```

Contact: Minghao Cao (caominghao@mail.ustc.edu.cn), Bei Hua (bhua@ustc.edu.cn).
