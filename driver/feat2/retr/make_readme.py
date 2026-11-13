"""Writes <DATA>/README.txt (DATA = 10_final/cache2/retr) from the meta files of the data: what the directory holds, formats, counts,
md5 sums, the exact preprocessing of the two encoders.  Deterministic for a given state of the data (no time stamp).

    python make_readme.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rcommon as C  # noqa: E402


def main(write: bool = True, verbose: bool = True) -> str:
    em = C.jload(os.path.join(C.EMB, "meta.json"))
    km = em["keys"]
    at = {bb: C.jload(os.path.join(C.DATA, bb, "attributes_meta.json")) for bb in C.BACKBONES}
    ie, qe = em["image"]["encoder"], em["question"]["encoder"]
    L = ["10_final/cache2/retr -- attributes and embeddings of the retrieval reference (retr) and of the lookup rule (PROTOCOL.md section 6,",
         "\"Review round 1\", item R4).  Written by 10_final/feat2/retr (build_attributes.py, embed.py; this file: make_readme.py).  Reader: feat2/retr/retr.py",
         "(retr.store(<backbone>)); usage and tests: feat2/retr/USAGE.txt.  Label-free: no gate decision, no outcome of any log is stored here.",
         "",
         "FILES",
         "  <backbone>/attributes.jsonl       qwen25vl, llava: the two attributes of every sequence of the backbone's key cache and of every edit request",
         "  <backbone>/attributes_meta.json   md5 of the table, counts, the checks made while building it, one rendered prompt per family / row type",
         "  emb/image_keys.jsonl              one line per distinct image content (md5), sorted by md5: md5, image (representative file), n_paths, backbones",
         "  emb/question_keys.jsonl           one line per distinct question key (question.strip().lower()), sorted: question_key, backbones",
         "  emb/keys_meta.json                counts and md5 of the two key lists, the attribute tables they were built from",
         "  emb/image_emb.npy                 float32 (n_images, 1024): row i = embedding of image content i of image_keys.jsonl",
         "  emb/question_emb.npy              float32 (n_questions, 384): row i = embedding of question key i of question_keys.jsonl",
         "  emb/meta.json                     encoders, exact preprocessing, library versions, md5 of the matrices, norm statistics, GPU memory measured",
         "  emb/parts/                        the resumable parts the matrices were assembled from (256 keys each) and the record of every run",
         "The embeddings are shared by the backbones: an image content or a question string that occurs in both caches has ONE row.",
         "",
         "ATTRIBUTE TABLE (one JSON object per line, keys in this order)",
         "  " + ", ".join(at["qwen25vl"]["fields"]),
         "  lines 0 .. n - 1: table = 'sequence', line j = the sequence at position j of 10_final/cache/<backbone>/index.jsonl (row = j; kinds rel, tgen, igen,",
         "  probe, hn_i, hn_q, hn_x, benign).  The following lines: table = 'edit', kind = 'request': one line per edit of every edit stream of the cache, in",
         "  stream order (sid '<family>:request:<position>'; row = null).",
         "  image       path as in the benchmark (VLKEB: relative to " + km["image_root_of_relative_paths"] + "; MMKE-Bench: absolute); null for text-only rows",
         "  image_md5   md5 of the image FILE (the image content identifier); null where there is no image",
         "  question    the question string as the user asks it: no chat template, no answer",
         "  question_key   question.strip().lower(): the identity under which questions are embedded and compared",
         "  edit, edit_uid   edit sequences and requests: the edit (stream position, item uid); hard negatives: the edit whose image (HN-I, HN-X) or question (HN-Q)",
         "              the row uses; edit_b, edit_b_uid (HN-X): the edit whose question the row uses",
         "  How derived and checked (attributes_meta.json, field checks): from the data loaders the caches were built with (edit stream items, probe items,",
         "  hard-negative and benign row files), compared with the cache index for EVERY sequence; md5 recomputed from every image file; EVERY prompt rendered",
         "  with the backbone's tokenizer and prompt code: the rendered text is the template around exactly the question string and its token count (with the",
         "  image slots) equals the captured prompt length of the cache.",
         ""]
    rows = []
    for bb in C.BACKBONES:
        m = at[bb]
        fam = "; ".join(f"{f}: {v['master_log']} ({v['n_edits']} edits, {v['dataset']})" for f, v in m["families"].items())
        rows.append([bb, m["n_sequences"], m["n_edit_requests"], m["n_text_rows"], m["n_distinct_image_md5"], m["n_distinct_question_keys"], m["md5"], fam])
    L += [C.table(rows, ["backbone", "sequences", "edit requests", "text rows", "distinct image md5", "distinct question keys", "md5 of attributes.jsonl", "edit streams"]), ""]
    L += ["EMBEDDINGS",
          C.table([["image_emb.npy", "x".join(str(x) for x in em["image"]["shape"]), em["image"]["md5"], "image_keys.jsonl", em["image"]["keys_md5"]],
                   ["question_emb.npy", "x".join(str(x) for x in em["question"]["shape"]), em["question"]["md5"], "question_keys.jsonl", em["question"]["keys_md5"]]],
                  ["file", "shape", "md5", "key list", "md5 of the key list"]),
          f"  {km['n_images_shared_by_backbones']} image contents and {km['n_questions_shared_by_backbones']} question keys occur in both caches; "
          f"{km['n_image_contents_with_several_paths']} image contents exist under several file paths (one row each).",
          f"  Unit norm: max | |e| - 1 | = {em['image']['norm_minus_1']['max_abs']:.1e} (images), {em['question']['norm_minus_1']['max_abs']:.1e} (questions).",
          "  Storage form: x / |x| in float64, every component rounded to the nearest multiple of 2^-24, float32.  For two such rows the float64 dot product is",
          "  EXACT (all partial sums are multiples of 2^-48 below 2), so a cosine score is a pure function of the two rows: the same bits in every process.",
          "",
          "  IMAGES    " + ie["encoder"] + ": " + ie["checkpoint"],
          "            " + ie["tensors"],
          f"            {ie['n_parameters']} parameters, sha1 of the loaded weights {ie['weights_sha1']}; {ie['model_class']} built from the checkpoint's vision_config",
          "            " + str(ie["vision_config"]) + " + defaults " + str(ie["config_defaults_used"]),
          "            weights: " + ie["forward_dtype"] + "; eval mode; " + ie["batch"] + "; device of the build: GPU (" + ie["device"] + "), TF32 off, cudnn deterministic",
          "            decode: " + ie["decode"],
          "            image processor: " + ie["image_processor_class"] + " of the checkpoint's preprocessor_config.json",
          "            " + ie["preprocessing"],
          "            " + str(ie["image_processor"]),
          "            embedding: " + ie["embedding"],
          "  QUESTIONS " + qe["encoder"] + ": " + qe["checkpoint"] + " through " + qe["backend"] + " (" + " -> ".join(qe["modules"]) + ")",
          "            " + qe["recipe"],
          f"            max_seq_length {qe['max_seq_length']} (longest question of the build: {max(r['max_tokens_seen'] for r in em['question']['runs'])} tokens: nothing is truncated); "
          + qe["batch"] + "; " + qe["forward_dtype"] + ", " + qe["device"],
          "            input: " + qe["input"],
          "  versions  " + ", ".join(f"{k} {v}" for k, v in ie["versions"].items()),
          "  GPU memory of the image job (one process, B = 1): " + ", ".join(f"{r['gpu']['nvidia_smi_process_mib_peak_seen']} MiB (nvidia-smi), {r['gpu']['max_memory_allocated_mib']:.0f} MiB allocated"
                                                                                for r in em["image"]["runs"][-1:]) + "; gq run --mem 2000",
          "",
          "REPRODUCIBILITY (feat2/retr/tests.py)",
          "  attributes.jsonl, image_keys.jsonl, question_keys.jsonl: a re-run reproduces the files byte for byte (test repro).",
          "  question_emb.npy: a complete re-run reproduces the matrix bit for bit (test reembed_questions_all); the model card's recipe through transformers",
          "  agrees within one grid step.  image_emb.npy: 50 images re-embedded on the GPU are bit for bit the stored rows (test reembed_images_gpu); on the CPU",
          "  (another device, float32) they differ by up to about 2e-6 per component (test reembed_images_cpu): rebuild on a GPU, the stored file is the reference.",
          "",
          "REBUILD (python = " + C.PYTHON + "; cd 10_final/feat2/retr)",
          "  python build_attributes.py qwen25vl; python build_attributes.py llava           CPU, seconds",
          "  python embed.py keys                                                              CPU",
          "  python embed.py questions                                                         CPU, 40 s, resumable parts",
          "  $GQ run --mem 2000 --prio 4 --name retr_img -- python embed.py images             GPU, 4.5 min, resumable parts",
          "  python embed.py assemble; python make_readme.py; python tests.py",
          "NOT COVERED: the caches of the scout's logs (10_final/cache/QO: VLKEB one-hop family on Qwen; 10_final/cache/blip2) have no attribute table here."]
    txt = "\n".join(L)
    if write:
        C.text_write(txt, os.path.join(C.DATA, "README.txt"))
    if verbose:
        print(txt)
    return txt


if __name__ == "__main__":
    main()
