"""Render out/DEVIATIONS.txt: every deviation of the E10 build from the public ROME code and its hyper-parameter file.
The list is maintained here (one entry per deviation, with the reason); the facts it quotes (commit, hashes, versions,
templates, resume events) are read from out/provenance.json, the run meta files and logs_edit/registry.json."""
import os
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tp_common as C  # noqa: E402


def main():
    P = C.jload(os.path.join(C.OUT, "provenance.json"))
    runs = P.get("runs", {})
    V = P["versions"]
    resumed = {k: (v.get("resumed") or []) for k, v in runs.items()}
    tmpl = {k: v.get("context_templates") for k, v in runs.items()}
    same_tmpl = len({tuple(t) for t in tmpl.values() if t}) == 1
    st = P["stats"]
    import glob
    import json
    fb = {}
    for pat, lab in ((os.path.join(C.OUT, "public_results", "ROME_seq", "run_*", "case_*.json"), "description"),
                     (os.path.join(C.LOGS_EDIT, "*", "write_time", "case_*.json"), "write time")):
        for pth in glob.glob(pat):
            with open(pth) as fh:
                txt = fh.read()
            if '"prob_test_use_cache_false": true' in txt:
                fb.setdefault(f"{lab}, {os.path.basename(os.path.dirname(pth)) if lab == 'description' else pth.split(os.sep)[-3]}", []).append(
                    int(os.path.basename(pth)[5:-5]))
    FB = "; ".join(f"{k}: case(s) {sorted(v)}" for k, v in sorted(fb.items())) if fb else "none so far"
    uce = []
    for ip in glob.glob(os.path.join(C.OUT, "describe", "*", "info.json")):
        u = C.jload(ip).get("use_cache_equivalence_at_W0")
        if u:
            uce.append(u["n_identical_output"] == u["n_records"])
    UCE = ("identical in every run of the check" if uce and all(uce) else ("NOT identical in some run" if uce else "not run yet"))
    rc = []
    for k in ("dev", "stream"):
        mp = os.path.join(C.LOGS_EDIT, k, "meta.json")
        if os.path.exists(mp):
            for c in C.jload(mp).get("resume_checks", []):
                rc.append(f"{k} after {c['n_have']} updates: the direction of the last stored update, recomputed with the public compute_u in the "
                          f"new process, is bit-for-bit the stored one: {c['left_vector_bitwise_equal']}; recorded weight hashes reproduced by "
                          f"the replay: {c['n_recorded_hashes_reproduced']}")
    RC = ("Resume checks: " + "; ".join(rc) + ".") if rc else "No resume check was recorded (the check was added after the development log)."
    items = [
        ("Public code",
         f"github.com/kmeng01/rome, commit {P['rome_repo']['commit']} ({P['rome_repo']['commit_date']}), obtained with `git clone --depth 1` "
         f"(a shallow clone: github.com:443 timed out repeatedly); working tree clean: {P['rome_repo']['working_tree_clean']}; identical to the "
         "commit archive served by codeload.github.com (kept under ext/third_party/_tarballs).  No file of the clone is edited.  "
         f"github.com/kmeng01/memit (commit {P['memit_repo']['commit']}) is cloned for reference only: of the ROME files, compute_u.py, compute_v.py, "
         "rome_hparams.py, tok_dataset.py, util/generate.py, util/nethook.py, util/runningstats.py and hparams/ROME/gpt2-xl.json are byte-identical "
         "there; rome_main.py (adds an optional factor cache), repr_tools.py (batch 128 instead of 512), layer_stats.py (a force_recompute "
         "option) and the evaluation files experiments/py/eval_utils_counterfact.py and experiments/summarize.py (MEMIT adds accuracy-based "
         "metrics and drops the attribute prompts) differ.  The ROME repository is the one that is run, for editing and for scoring.",
         "none (not a deviation); recorded as required"),
        ("Hyper-parameters",
         f"hparams/ROME/gpt2-xl.json of the clone, sha256 {P['hparams']['sha256']}, read by the public ROMEHyperParams.from_json; unchanged "
         f"(tracked file unmodified: {P['hparams']['tracked_file_unmodified']}).",
         "none (not a deviation)"),
        ("Paths (globals.yml)",
         "The public util/globals.py reads ./globals.yml.  The drivers chdir to text_public/run, whose globals.yml has the keys of the public "
         "file with absolute paths: RESULTS_DIR = text_public/out/public_results, DATA_DIR = ext/data/rome_data, STATS_DIR = "
         "ext/data/rome_data/stats, HPARAMS_DIR = ext/third_party/rome/hparams, REMOTE_ROOT_URL unchanged.",
         "outputs and data must live under project_3; the clone must stay unedited"),
        ("Model source and loading",
         f"GPT-2 XL from the local directory ext/models/gpt2-xl (repo {P['model']['repo']}, revision {P['model']['revision']}, downloaded through "
         f"hf-mirror.com; model.safetensors sha256 equals the hub's LFS hash: {P['model']['model.safetensors_sha256_matches_hub_lfs']}) instead of the "
         "hub name 'gpt2-xl'.  Loaded as experiments/evaluate.py does: AutoModelForCausalLM.from_pretrained(...).cuda(), "
         "tok.pad_token = tok.eos_token; torch_dtype=torch.float32 is passed explicitly (it is the default).  model.config._name_or_path is set "
         "to 'gpt2-xl' after loading, because the public code builds the statistics path from it.",
         "huggingface.co is not reachable from this machine"),
        ("Device and dtype", "One GPU (cuda:0 of the queue's card), float32 everywhere, as in the public code.  The process caps its GPU memory "
                             "(torch.cuda.set_per_process_memory_fraction) and uses at most 4 BLAS threads.", "queue rules; no numerical effect"),
        ("Library stack",
         f"python {V['python']}, torch {V['torch']}, transformers {V['transformers']}, tokenizers {V['tokenizers']}, numpy {V['numpy']}, "
         f"scipy {V.get('scipy')}, scikit-learn {V.get('sklearn')}, nltk {V.get('nltk')}.  The public environment pins python 3.9.7, pytorch 1.10.2, "
         "transformers 4.15.0, nltk 3.6.5 (scripts/rome.yml); MEMIT, which ships the same ROME files, pins pytorch 1.12.1 and transformers 4.23.1.  "
         "transformers 4.23.1, tokenizers 0.13.3 and huggingface_hub 0.11.1 are installed under ext/pydeps_tf4 and put first on sys.path for the "
         "E10 scripts only; nltk under ext/pydeps (with the punkt_tab tokenizer data under ext/data/nltk_data).  The project environment is not "
         "modified.",
         "the project environment has transformers 5.17, under which the public util/generate.py (generate_fast) produces garbage text "
         "(logs/smoke_tf5.out: the sampled context templates were strings such as 'In the-\\n. {}'); transformers 4.15.0 has no build of its "
         "tokenizers for python 3.11; nltk is not installed in the environment"),
        ("`datasets` placeholder",
         "rome/layer_stats.py imports datasets.load_dataset at import time.  The environment's datasets package cannot be imported next to "
         "huggingface_hub 0.11.1, so the drivers register an empty placeholder module whose load_dataset raises.  It is never called: the "
         "statistics file exists, so the public layer_stats loads it from the cache.",
         "import-time dependency only"),
        ("Second-moment statistics",
         f"The authors' precomputed file, downloaded from {st['url']} (sha256 {st['sha256']}; {st['token_count']} tokens of {st['sample_size']} "
         f"Wikipedia samples, {st['mom2_dtype']}, {st['mom2_shape'][0]} x {st['mom2_shape'][1]}), placed where the public code caches it.  Not recomputed.  "
         "The inverse is taken by the public compute_u.get_inv_cov (torch.inverse on the GPU, cast to float32).",
         "the task asks for the authors' file if it downloads; curl is used because the public loader's own download (torch.hub) would write "
         "to the same place"),
        ("Data",
         f"counterfact.json from the URL of the public loader (https://rome.baulab.info/data/dsets/counterfact.json), sha256 {P['counterfact']['sha256']}, "
         f"{P['counterfact']['n_records']} records, downloaded with curl into ext/data and read with json.load, which is what the public "
         "CounterFactDataset (dsets/counterfact.py) does with the file.  idf.npy, tfidf_vocab.json and attribute_snippets.json were also downloaded "
         "(the last one after a resumed transfer) and linked into the data directory; none of the three is used by this build.",
         "same file from the same URL; the consistency score, the only user of the three auxiliary files, is not required"),
        ("Driver of the single-edit check",
         "experiments/evaluate.py cannot be imported here: it imports every baseline and fails at `import hydra` (baselines/mend/mend_main.py; "
         "checked on 2026-11-11).  Its loop (lines 105-146) "
         "is re-typed in run_single.py with the same public calls and arguments: apply_rome_to_model(model, tok, [record['requested_rewrite']], "
         "hparams, copy=False, return_orig_weights=True); compute_rewrite_quality_counterfact(edited_model, tok, record, None, None); restore the "
         "weight; the same call on the restored model; case_<id>.json in the public format.  snips = vec = None is the public "
         "--skip_generation_tests.  Scores: the public experiments/summarize.py, unchanged.",
         "the other baselines' dependencies are not installed and are not needed for ROME"),
        ("Weight not restored between requests (the sequential logs)",
         "run_stream.py calls apply_rome_to_model(model, tok, [request], hparams, copy=False, return_orig_weights=False) once per request and "
         "does not restore the weight afterwards.  The public function itself computes the factors on the current weight and adds their outer "
         "product in place; only the restore of the public evaluation driver is absent.",
         "this is the object of E10: one module, one rank-one update per request, applied sequentially"),
        ("Random seed and context templates",
         "torch.manual_seed(0) is called once before the first request of every run (the public driver sets no seed).  Public behaviour, kept: "
         "rome_main.get_context_templates samples the context templates ONCE per process, at the first request, from the model as it is at "
         "that moment (generate_fast with top-5 sampling from '<|endoftext|>': 10 texts of 5 tokens and 10 of 10 tokens, each followed by "
         "'. {}', plus the bare '{}'), and caches them in the module global CONTEXT_TEMPLATES_CACHE; every later request of the process reuses "
         "them, so in a sequential run they are samples of the PRE-EDIT model.  "
         + ("With seed 0 the development log, the reported stream and the single-edit check obtained the same 21 templates: " if same_tmpl else
            "The runs obtained different templates (see provenance.json); those of the first run: ")
         + repr(next((t for t in tmpl.values() if t), None)) + ".  "
         "When a run is resumed, the cache is restored from the run's meta file instead of being re-sampled from the edited model, and the "
         "weight is rebuilt by replaying the stored factors with the public write expression (bit-for-bit equality with the recorded sha256 is "
         "asserted).  Resume events: " + "; ".join(f"{k}: {len(v)}" + (f" (at {', '.join(str(x.get('n_have', x)) if isinstance(x, dict) else str(x) for x in v)} updates)" if v else "")
                                                   for k, v in resumed.items()) + ".  "
         "Causes: the development log was started as a 3-update test and continued; the reported stream stopped once, with a CUDA "
         "out-of-memory error in the write-time probability test of case 1531 (item on the memory fallback below), after 1531 updates.  "
         + RC,
         "reproducibility; a resumed run must behave as the uninterrupted public process would"),
        ("Observation of the public computation",
         "tp_common.Capture wraps rome.rome_main.execute_rome and rome.repr_tools.get_reprs_at_word_tokens at run time (attributes of the "
         "imported modules; no file is edited) and reads the local variables target, target_init, delta and the last loss terms of the running "
         "compute_v frame.  It records the returned factors, the module inputs at the last subject token under every context template (their "
         "mean is the vector compute_u multiplies by the inverse second moment), the module input and output for the request prompt alone "
         "(the key compute_v divides by and the output it subtracts) and the target.  Nothing the public code computes is changed.",
         "the log needs the keys and the target, which the public functions do not return"),
        ("Evaluation inside the sequential runs",
         "After each update the public compute_rewrite_quality_counterfact is run on the request's own record with the weight of that moment "
         "(forward passes only; no sampling, the weight is not touched) and stored under logs_edit/<run>/write_time.",
         "write-time success of the editor along the stream"),
        ("Memory fallback of the public probability test (model.config.use_cache)",
         "The public test_batch_prediction feeds the 46 sequences of a record as one padded batch through model(**batch); with the default "
         "config the forward also returns the key / value tensors of all 48 layers.  For a record with a very long paraphrase prompt this "
         "exceeds the memory cap of the queue job (case 1531: a 179-token paraphrase, 46 x 181 tokens; first seen as an out-of-memory failure, "
         "logs/describe_A_oom_case1531.out).  tp_common.public_prob_test calls the public function as is and, on a CUDA out-of-memory error "
         "only, calls the same public function again with model.config.use_cache = False (the forward then returns no key / value tensors; "
         "the logits are computed by the same operations).  Records evaluated that way carry prob_test_use_cache_false = true in their case "
         f"file: {FB}.  Equality of the public function's output under both settings was checked on three records at the pre-edit weight "
         f"(bit for bit: {UCE}).",
         "GPU memory is shared; the alternative was a reservation of about 15 GB for one record"),
        ("Generation metric of the description",
         "The public test_generation needs the attribute snippets and the tf-idf vectoriser for its consistency score.  The description calls "
         "the two public functions it is made of that do not need them: util.generate.generate_fast(model, tok, generation_prompts, "
         "n_gen_per_prompt=1, max_out_len=100) and eval_utils_counterfact.n_gram_entropy(texts of one record).  The generation prompts of 2 "
         "records (20 prompts) are passed to one generate_fast call instead of one call per record, with torch.manual_seed(77000 + index of "
         "the first record) before each call; rows of a batch do not interact in generate_fast (no pad position is fed), so only the random "
         "numbers drawn differ from per-record calls.  The consistency (reference) score and the essence score are not computed.  nltk.word_tokenize is "
         f"that of nltk {V.get('nltk')} (punkt_tab), not 3.6.5.",
         "the consistency score is not required; a seed makes the sampled generations reproducible and paired across weights; one call per "
         "record took about 15 minutes per 100 records on the shared cards (a first attempt was stopped for that reason)"),
        ("Not public code (this project's own measurements)",
         "The pools, the pre-edit continuations, the exact / token-level match of the new object, D1 and the degeneration flags are computed "
         "by text_public/code (tp_eval.py, describe.py, make_table.py): greedy decoding and teacher forcing in batches of one token length "
         "without padding.  They are reported in separate columns and never mixed with the public scores.",
         "required by the E10 protocol; listed so that no reader takes them for public metrics"),
    ]
    L = ["DEVIATIONS of the E10 build (ROME on GPT-2 XL, CounterFact) from the public reference implementation and its hyper-parameter file",
         f"rendered {C.now()} by make_deviations.py from out/provenance.json and the run meta files", ""]
    for i, (title, what, why) in enumerate(items, 1):
        L.append(f"{i}. {title}")
        for para, lab in ((what, "what"), (why, "why ")):
            w = textwrap.wrap(para, 116)
            L.append(f"   {lab}: {w[0]}")
            L += ["         " + x for x in w[1:]]
        L.append("")
    open(os.path.join(C.OUT, "DEVIATIONS.txt"), "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
