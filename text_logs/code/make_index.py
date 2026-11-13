"""Render out/INDEX.txt: where every deliverable of the E10 build lives (paths, sizes, one line each).  CPU."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tp_common as C  # noqa: E402

ITEMS = [
    ("PUBLIC CODE, MODEL, DATA (ext/)", [
        (f"{C.EXT}/third_party/rome", "public ROME repository (git clone, unedited); commit in out/provenance.json"),
        (f"{C.EXT}/third_party/memit", "public MEMIT repository (reference only)"),
        (f"{C.EXT}/third_party/_tarballs", "commit archives of both repositories (identical to the clones)"),
        (f"{C.EXT}/models/gpt2-xl", "GPT-2 XL float32 (model.safetensors, pytorch_model.bin, tokenizer), DOWNLOAD_RECORD.json"),
        (f"{C.EXT}/models/gpt-j-6b", "GPT-J 6B float32 main revision (downloaded for a later task; no GPU work done with it)"),
        (f"{C.EXT}/data/counterfact/counterfact.json", "CounterFact as published by the ROME authors (21,919 records)"),
        (f"{C.EXT}/data/stats/gpt2-xl/wikipedia_stats", "authors' precomputed second-moment statistics of transformer.h.17.mlp.c_proj"),
        (f"{C.EXT}/data/rome_data", "the DATA_DIR / STATS_DIR layout the public code expects (symbolic links)"),
        (f"{C.EXT}/data/paper_rome", "arXiv 2202.05262v5: e-print, extracted LaTeX source (tables/cf-results.tex), PDF"),
        (f"{C.EXT}/pydeps", "nltk (pip --target)"),
        (f"{C.EXT}/pydeps_tf4", "transformers 4.23.1, tokenizers 0.13.3, huggingface_hub 0.11.1 (pip --target; E10 scripts only)"),
    ]),
    ("EDIT LOGS (text_public/logs_edit/)", [
        (f"{C.LOGS_EDIT}/registry.json", "the five logs: source, T, records, files, sha256 of W_0 and W_E, reconstruction errors, key identity"),
        (f"{C.LOGS_EDIT}/W0_transformer.h.17.mlp.c_proj.weight.pt", "pre-edit Conv1D weight (6400, 1600) float32"),
        (f"{C.LOGS_EDIT}/dev/factors.pt", "development log (records 15000..15099): stacked factors, keys, targets, requests, scalars, templates"),
        (f"{C.LOGS_EDIT}/dev/snapshots", "deployed weight of the development log (W_T100.pt)"),
        (f"{C.LOGS_EDIT}/dev/updates", "one file per update: what execute_rome returned + keys, target, request, seed, checks"),
        (f"{C.LOGS_EDIT}/dev/write_time", "public probability test of each record right after its own update"),
        (f"{C.LOGS_EDIT}/stream/factors.pt", "reported stream (records 0..1999): the same; T100/T500/T1000/T2000 are prefixes"),
        (f"{C.LOGS_EDIT}/stream/snapshots", "deployed weights after 100, 500, 1000, 2000 requests"),
        (f"{C.LOGS_EDIT}/stream/updates", "one file per update"),
        (f"{C.LOGS_EDIT}/stream/meta.json", "seed, context templates, hyper-parameters, versions, resume events and checks"),
        (f"{C.LOGS_EDIT}/library_format", "each log as a directory that rescope.loglib.from_testbed(<dir>) reads (edit_factors.pt, metrics.json)"),
    ]),
    ("CODE (text_public/code/)", [
        (f"{C.CODE}/tx_public.py", "LOADER AND REGISTRY for the frozen library: from_public(name), weights(name), load_model(), prompts, pools"),
        (f"{C.CODE}/tp_common.py", "paths, set-up of the public repository, capture of the public computation, memory fallback"),
        (f"{C.CODE}/run_single.py", "single-edit check (public protocol)"),
        (f"{C.CODE}/run_stream.py", "sequential logs (dev, stream), resumable"),
        (f"{C.CODE}/build_logs.py", "assemble + audit the logs, write the registry"),
        (f"{C.CODE}/export_library_format.py", "library-format directories + check through the library's own constructor"),
        (f"{C.CODE}/check_library.py", "development log through rescope.EditLog / ScopedLog / Hook (format check)"),
        (f"{C.CODE}/pools.py", "REPAIR / HELD-OUT pools, prompt table, flags"),
        (f"{C.CODE}/pre_cont.py", "pre-edit 10-token continuations"),
        (f"{C.CODE}/tp_eval.py", "greedy decode / teacher forcing without padding, degeneration flags"),
        (f"{C.CODE}/describe.py", "raw measurements of the pre-edit and the deployed weights"),
        (f"{C.CODE}/selfcheck.py", "sample of rows recomputed one prompt at a time (selfcheck_cpu.py: on the CPU)"),
        (f"{C.CODE}/d1_curve_dev.py", "development log: D1 as a function of the number of updates"),
        (f"{C.CODE}/collapse_point.py", "reported stream: deployed prefixes around the first exploding update"),
        (f"{C.CODE}/first_token_subjects.py", "count of first-token-subject requests in CounterFact"),
        (f"{C.CODE}/score_single.py", "public summariser on the single-edit run + the paper's table"),
        (f"{C.CODE}/make_table.py", "aggregation -> out/text_logs.json; render_table.py -> out/text_logs_table.txt"),
        (f"{C.CODE}/provenance.py", "provenance record; make_deviations.py -> out/DEVIATIONS.txt; make_index.py -> this file"),
        (f"{C.CODE}/finalize.sh", "CPU: rebuilds every rendered output from the stored results"),
    ]),
    ("OUTPUTS (text_public/out/)", [
        (f"{C.OUT}/text_logs_table.txt", "THE TABLE: logs, deployed vs pre-edit model, retention, updates, pools, audit conditions"),
        (f"{C.OUT}/text_logs.json", "the numbers behind the table"),
        (f"{C.OUT}/text_logs_rows", "per-row flags behind the table (one file per log and model)"),
        (f"{C.OUT}/DEVIATIONS.txt", "every deviation from the public code and its hyper-parameter file, with the reason"),
        (f"{C.OUT}/single_edit_check.txt", "audit condition 2: our scores against the published table"),
        (f"{C.OUT}/single_edit_check.json", "the same, with intervals and the parsed table"),
        (f"{C.OUT}/log_audit.json", "per-update audit: attainment at both keys, norms, replay hashes; single vs stream factors"),
        (f"{C.OUT}/provenance.json", "commits, hashes, versions, run meta"),
        (f"{C.OUT}/pools.json", "pools: rule, seed, record lists, overlap counts"),
        (f"{C.OUT}/pool_records.json", "per pool record: subject, flags"),
        (f"{C.OUT}/prompts.jsonl", "one row per (record, kind, k): set, text id, flags"),
        (f"{C.OUT}/prompt_texts.json", "distinct prompt texts and their token ids"),
        (f"{C.OUT}/pre_continuations.pt", "pre-edit greedy 10-token continuations (token ids), validity"),
        (f"{C.OUT}/pre_continuations.jsonl", "the same, readable (ids and text, degeneration flags)"),
        (f"{C.OUT}/describe", "raw description results per weight (dec.pt, edit_tf.pt, gen_*.json)"),
        (f"{C.OUT}/public_results/ROME_single/run_000", "single-edit check: public-format case files + factors of every single edit"),
        (f"{C.OUT}/public_results/ROME_seq", "public probability test of every logged record under each weight (public case format)"),
        (f"{C.OUT}/library_check.json", "format check of the development log through the library"),
        (f"{C.OUT}/library_format.json", "library-format export check"),
        (f"{C.OUT}/selfcheck.json", "self-check of the batched description against single-prompt recomputation"),
        (f"{C.OUT}/selfcheck_cpu.json", "the same on the CPU"),
        (f"{C.OUT}/d1_curve_dev.txt", "development log: D1 on held-out prompts after 0, 1, 2, 5, 10, 20, 50, 100 updates"),
        (f"{C.OUT}/collapse_point.txt", "reported stream: the deployed model after 100, 150, 173, 174, 200 requests"),
        (f"{C.OUT}/first_token_subjects.json", "base rate of requests whose subject is the first token of the prompt (the exploding updates)"),
        (f"{C.OUT}/summary.txt", "one line per log (the table of the report)"),
        (f"{C.OUT}/smoke_tf5.json", "evidence: the public generation under the environment's transformers 5.17 (garbage templates)"),
    ]),
    ("RUN LOGS (text_public/logs/)", [
        (f"{C.LOGS}/stream_public_stdout.log", "everything the public code printed while writing the stream (its optimisation traces)"),
        (f"{C.LOGS}/dev_public_stdout.log", "the same for the development log"),
        (f"{C.LOGS}/single_run_000_public_stdout.log", "the same for the single-edit check"),
        (f"{C.LOGS}/setup", "download / clone logs, GPT-J checksum, pip cache entries removed"),
    ]),
]


def size(p):
    if os.path.isfile(p):
        return os.path.getsize(p)
    tot = 0
    for d, _, fs in os.walk(p):
        for f in fs:
            fp = os.path.join(d, f)
            if not os.path.islink(fp):
                tot += os.path.getsize(fp)
    return tot


def human(n):
    for u in ("B", "kB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return f"{n:.0f} {u}" if u == "B" else f"{n:.1f} {u}"
        n /= 1024


def main():
    L = ["E10 PUBLIC TEXT LOGS -- index of deliverables", f"rendered {C.now()} by make_index.py", ""]
    for title, items in ITEMS:
        L.append(title)
        for p, what in items:
            ex = os.path.exists(p)
            L.append(f"  {(human(size(p)) if ex else 'MISSING'):>10s}  {p}")
            L.append(f"              {what}")
        L.append("")
    open(os.path.join(C.OUT, "INDEX.txt"), "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
