# Third-party notices

**Built with Qwen.**

This artifact (the repository and the eight archives of its release) contains material that derives from models and
benchmarks of other parties. The MIT licence of the repository (`LICENSE`) covers the authors' code and documentation only.
The licence texts are in `third_party_licenses/`, byte for byte as their publishers distribute them (fetched on 2026-11-13),
together with `third_party_licenses/Notice.txt`, which holds the attribution notices that the licences prescribe. Every
archive of the release carries this file and that directory at its top level.

The files that derive from these materials were made by the authors of the artifact: recorded updates of one weight matrix
per edit log (differences to the public checkpoints, which are not redistributed), outputs of the models, and items of the
benchmarks (text and identifiers; no image). They are modifications or derivatives, not the original materials. No
checkpoint, no benchmark image and no file of the ROME code is redistributed.

## Terms of the derived files

- **Files derived from Qwen2.5-VL-3B** (all logs and results whose names begin with Q or carry the family letter Q as a part
  of the name, as in `__Q__` and `c4_Q/`: the logs `Q-...` of the archives, the result files and directories that carry the
  working names `TB-Q...` or the backbone's name `qwen25vl`, and the Qwen parts of the pools) are distributed under the Qwen
  RESEARCH LICENSE AGREEMENT, for non-commercial (research or evaluation) purposes only.
- **Files derived from LLaVA-1.5-7B** (names beginning with L or carrying the family letter L as a part of the name, as in
  `__L__` and `c4_L/`: the logs `L-...`, the result files and directories with the working names `TB-L...` or the backbone's
  name `llava`, and the LLaVA parts of the pools) are distributed under the LLAMA 2 Community License and its Acceptable Use
  Policy. Under that licence the files and the outputs they hold may not be used to improve any other large language model
  (excluding Llama 2 or derivative works thereof).
- **Files derived from GPT-2 XL** (the text logs: `logs_text.tar`, the logs `G-...` / `TX-G2XL-...`, the generated texts) are
  distributed under OpenAI's Modified MIT License. The texts in the files `gen_*.json`, in the directories `gen/` and in the
  fields `y0_text` of the text logs were created using GPT-2, and so were the short decoded outputs of GPT-2 XL in the
  records of the text-log audit under `text_logs/audit/out/` of the repository (the fields `audit_text` and `audit_y0_text`
  of `eval_tf4_rows.json` and `eval_env_rows.json`, `example` and `most_common_continuations` of `traps.json`, and `text`
  under `ngram_entropy_formula` of `extra2.json`).
- **VLKEB items** (questions, answers, identifiers and image file names in the vision-language logs, pools and results) are
  distributed under the BSD 3-Clause licence of VLKEB.
- **CounterFact items** (requests and prompts of the text logs) and **the re-typed loop of the ROME code**
  (`text_logs/code/run_single.py`) are distributed under the MIT licence of the ROME repository. CounterFact is published by
  the authors of ROME with their code; the data file itself carries no licence statement of its own.
- **MME and MMKE-Bench state no licence.** Their items here: question texts and file names of the shifted benign pool (MME;
  `keycache/cache2/mme/` of the repository), and key-cache rows of a log family that is neither reported in the paper nor
  released (MMKE-Bench; 2500 rows of the Qwen2.5-VL-3B key cache in `pools_vl.tar`).
- A file that holds material of more than one of these sources (for instance a summary over the logs of both
  vision-language backbones) is under the terms of each of them.

## Material, place, source, licence

| Material | Where it is in the artifact | Source | Licence | What the licence asks, and how the artifact meets it |
|---|---|---|---|---|
| Qwen2.5-VL-3B (Alibaba Cloud): updates of one weight matrix, outputs of the model | `logs_vl_QA.tar`, `logs_vl_QM.tar`; the logs `Q-...` of `logs_vl_prefix.tar`; the `qwen25vl` parts of `pools_vl.tar`; in `results_rows.tar.gz` and in the repository (`experiments/`, `linear_baselines/`, `keycache/`, `driver/`) the result files of the logs `TB-Q...` and the files and directories that carry the family letter Q as a part of the name (as in the row files `...__Q__...` of `results_rows.tar.gz` and the directory `experiments/e0_verify/out/c4_Q/` of the repository) | Hugging Face `Qwen/Qwen2.5-VL-3B-Instruct`, commit 66285546d2b821cf421d4f5eb2576359d3770cd3 | Qwen RESEARCH LICENSE AGREEMENT (`third_party_licenses/QWEN25VL3B_LICENSE.txt`) | Non-commercial use only, that is, research or evaluation (1.i, 2.a): stated here and in the README. A copy of the agreement for every recipient (3.a): the licence file, in the repository and in every archive. Modified files carry notices of the change (3.b): no file of Qwen is redistributed; the derived files are declared as the authors' modifications here and in `Notice.txt`. The attribution notice in a "Notice" text file (3.c): `Notice.txt`, first line. "Built with Qwen" in the documentation of a model made with the materials or their outputs (4.b): displayed at the top of this file. |
| LLaVA-1.5-7B (according to its model card, trained by fine-tuning LLaMA/Vicuna): updates of one weight matrix, outputs of the model | `logs_vl_LA.tar`, `logs_vl_LM.tar`; the logs `L-...` of `logs_vl_prefix.tar`; the `llava` parts of `pools_vl.tar`; in `results_rows.tar.gz` and in the repository the result files of the logs `TB-L...` and the files and directories that carry the family letter L as a part of the name (as in the row files `...__L__...` of `results_rows.tar.gz` and the directory `experiments/e0_verify/out/c4_L/` of the repository) | Hugging Face `llava-hf/llava-1.5-7b-hf`, commit b234b804b114d9e37bb655e11cbbb5f5e971b7a9 (licence tag of the model card: `llama2`) | LLAMA 2 Community License (`LLAMA2_COMMUNITY_LICENSE.txt`) with the Llama 2 Acceptable Use Policy (`LLAMA2_USE_POLICY.md`). The LLaVA code base is under Apache-2.0 (`LLAVA_CODE_LICENSE.txt`); no LLaVA code is redistributed here | A copy of the agreement for every third party (1.b.i): the licence file. The attribution notice in a "Notice" text file (1.b.iii): `Notice.txt`, second line. Use in keeping with the law and with the Acceptable Use Policy (1.b.iv): the policy is included; it binds every user of these files. No use of the materials or of their outputs to improve any other large language model (1.b.v): stated above; it binds every user of these files. |
| GPT-2 XL (OpenAI): updates of one weight matrix, texts generated by the model | `logs_text.tar`; in the repository `text_logs/out/` (pre-edit continuations, `describe/*/gen_*.json`) and `experiments/e10_text/` (`run/out/gen/`); in `results_rows.tar.gz` the files under `experiments/e10_text/` | Hugging Face `openai-community/gpt2-xl`, revision 15ea56dee5df4983c59b2538573817e1667135e2 | Modified MIT License (`GPT2_LICENSE.txt`) | The copyright and permission notice in copies of the software (not required with content created by it): the licence file. "Clearly indicate your content was created using GPT-2": `Notice.txt`, the README and the statement above. |
| VLKEB (Huang et al., 2024): items (questions, answers, identifiers, image file names; no image) | every `logs_vl_*.tar` (edit streams, probe records, complaints), `pools_vl.tar`, the per-row results of the vision-language experiments | Hugging Face data set `HymanH/VLKEB-data`, commit cf2d0abe73cb638fcf368d8bc5f9ac3caa204d5a | BSD 3-Clause (`VLKEB_LICENSE.txt`) | The copyright notice, the list of conditions and the disclaimer with every redistribution: the licence file. No use of the names of the copyright holders to endorse or promote: none is made. |
| ROME code (Meng et al., 2022): one loop, re-typed | `text_logs/code/run_single.py` of the repository | `https://github.com/kmeng01/rome`, commit 0874014cd9837e4365f3e6f3c71400ef11509e04 | MIT (`ROME_LICENSE.txt`) | The copyright and permission notice in every copy or substantial portion: in the file itself and in the licence file. |
| CounterFact (Meng et al., 2022): requests and prompts | `logs_text.tar`; `text_logs/out/` and `experiments/e10_text/` of the repository; in `results_rows.tar.gz` the files under `experiments/e10_text/` | `https://rome.baulab.info/data/dsets/counterfact.json` (SHA-256 d017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f), published by the authors of ROME with their code | MIT, the licence of the ROME repository (`ROME_LICENSE.txt`); the data file carries no licence statement of its own | As for the ROME code. |
| MME: question texts and file names of the shifted benign pool | `keycache/cache2/mme/` of the repository | four parquet files of the public MME data set (names and sizes in `keycache/cache2/mme/extract_summary.json`) | none stated by the source (the data set's card in the local copy) | Nothing is stated. The items are named here as what they are. |
| MMKE-Bench: key-cache rows of a log family that is neither reported in the paper nor released | `pools_vl.tar` (2500 rows of the key cache of Qwen2.5-VL-3B) | Hugging Face data set `kailinjiang/MMKE-Bench-dataset` | none stated by the source (the data set's card, local copy and repository) | Nothing is stated. The rows are named here as what they are. |

If you hold rights in material contained in the artifact and want it changed or removed, please open an issue in the
repository or write to the first author (address at the end of the README).
