"""Plain-text descriptions written next to the cache files (FORMAT.txt, README.txt).  Counts come from the data."""
from __future__ import annotations

import collections

import fc_common as C


def cache_text(BB, meta):
    fams = "; ".join(f"{f} = {v['dataset']} stream of {v['master_log']}" for f, v in meta["families"].items())
    return f"""KEY CACHE of backbone {BB} (layer {meta['layer']}: {meta['module']}, d = {meta['d_in']})
Built by 02_research/experiments/10_final/cache/code (fc_select.py -> fc_capture.py -> fc_assemble.py); checked by
fc_verify.py (verify_gpu.json, verify_dev.json, verify_cpu.json in this directory); counts: ../cache_table.txt.
complete = {meta['complete']}.

The key of a sequence at a token position depends on the backbone and on the token sequence, NOT on the edit log:
this one cache serves every log of the backbone.  Everything was captured at B = 1 with the pre-edit model through
rescope.Evaluator (capture / decode_keys / lp_shift / tf) of the family's longest log.
Families (field `family`): {fams}.

FILES
  index.jsonl            one JSON object per cached sequence, {meta['n_sequences']} lines (order = row order of keys_decode.pt)
  keys_tf.pt             torch.FloatTensor {tuple(meta['keys_tf']['shape'])}: module input at the last prompt position of the
                         TEACHER-FORCED sequence (prompt + answer), float32.  Rows 0 .. {meta['keys_tf']['n_main_rows'] - 1}: position
                         `lp_tf` (the harness index answer_start - 1), addressed by `row_tf`.  The last
                         {meta['keys_tf']['n_shifted_extra_rows']} rows: the SAME teacher-forced sequence at the decode's token
                         (`lp_tf - lp_shift`), only for sequences with lp_shift != 0, addressed by `row_tf_prompt`
                         (= Evaluator.capture(it, True, lp='prompt')).
  keys_decode.pt         torch.FloatTensor {tuple(meta['keys_decode']['shape'])}: module input at the last token of the PROMPT-ONLY
                         sequence (position `lp_decode`), i.e. the first forward of a decode, float32; row = `row_decode`
                         (= line number in index.jsonl).  Exists for every sequence.
  keys_decode_steps.pt   torch.FloatTensor {tuple(meta['keys_decode_steps']['shape'])}: the DISTINCT keys that the prompt's last token takes
                         in the 11 full forwards of a decoder-v2 decode (prompt + k ids, k = 0 .. 10; = Evaluator.decode_keys).
                         `rows_decode_steps` (11 row numbers, k = 0 .. 10) addresses them; rows_decode_steps[0] holds the
                         same values as keys_decode.pt[row_decode].  Distinct keys per sequence (histogram):
                         {meta['keys_decode_steps']['distinct_keys_per_sequence_hist']}.
  pre_edit_flags.jsonl   pre-edit teacher-forced flags of every edit sequence (also in index.jsonl): is the edit target
                         already produced by the pre-edit model (pre_exact = every target token is the argmax;
                         pre_token_acc = share of target tokens).  A sequence on which no edit fires inherits them.
  meta.json              shapes, md5 sums, capture provenance.   verify_*.json: verification results.
  ../hard_negatives/{BB}/   all hard negatives (existing + new) in the file format of 08_rescope's hard negatives.
  ../benign/{BB}/           the benign pool (rows + decoder-v2 answers of the subset).

All keys are float32 images of the module-dtype values ({meta['module_dtype']}): exact.

INDEX FIELDS (absent or null where not applicable)
  sid            unique id of the sequence in this cache: '<family>:edit:<metric>:<pos>', '<family>:probe:<ptype>:<probe_idx>',
                 the hard-negative id 'HN-I|HN-Q:<edit uid>:<k>' / 'HN-X:<uid a>:<uid b>', 'benign:<ptype>:<n>'
  lib_sid        the sid of the same item in rescope.Evaluator for a log of the family ('edit:rel:17', 'probe:image:3', hn id)
  kind           rel | tgen | igen | probe | hn_i | hn_q | hn_x | benign
  family, dataset, master_log
  ptype, fmt, prompt, image, image_md5      the query: 'image' (image + question, fmt vqa) or 'text'; image path as in the
                 benchmark (VLKEB: relative to its image directory; MMKE-Bench: absolute); md5 of the image file
  query_id       same id <=> same model input up to the last prompt token = same (ptype, prompt string, image CONTENT);
                 n_same_query = number of cached sequences with that id (duplicate detection across all kinds / families)
  edit           stream position of the edit (rel / tgen / igen; hard negatives: the edit whose image (HN-I, HN-X) or
                 question (HN-Q) the row uses); edit_uid; metric.   HN-X: edit_b / edit_b_uid = the edit whose question is used.
                 A log with T edits contains the edits with edit < T of its family.
  probe_idx, uid, group, pool, disjoint, strict, question_in_repair, twin_of, probe_query_id
                 probe rows: uid = probe item; group = the probe record's row id ('image_17'); pool = harness split 'repair' |
                 'heldout' (seed 20261109); disjoint = HELD-OUT row whose query is not a REPAIR query (the reporting pool
                 HELD-OUT-DISJOINT); strict / question_in_repair / probe_query_id as rescope.splits.annotate (the library
                 compares image PATHS, query_id compares image CONTENT; verify_cpu.json, probe_pools_by_content, lists the
                 rows on which the two differ).  benign rows: uid = source item, pool = 'benign'.
  hn_id, hn_type, k, hn_group (first100 | later100; also in `group`), rule (p11_category_restricted | final_unrestricted), new, same_category,
  related_flag, src_uid, src_related_edit_first_pos, q_in_edit, q_in_edit_first_pos         hard negatives (see its README)
  benign_idx, answer_subset, q_in_edit, q_in_edit_first_pos, img_in_edit, q_in_probe, img_in_probe, query_in_probe,
  img_entity_in_edit, img_entity_in_probe, overlap      benign rows: does the question string (stripped, lower case) or
                 the image (md5) occur among the edits (src / rephrase; image / image_rephrase) or the probe rows of the
                 VLKEB family; `overlap` holds the same flags per family of the backbone
  answer, answer_from     the text that is teacher-forced: the edit target (edits) or the pre-edit answer; null: no answer
  y0_text, y0_valid, y0_invalid_reason      decoder-v2 pre-edit answer (multi-token) and its validity (non-empty AND reproduced
                 by the pre-edit model under teacher forcing) for probe rows, hard negatives and the benign subset
                 (answer_subset = true); null for edit sequences and for benign rows outside the subset
  deg_empty, deg_single_char, deg_qmark, deg_none_no     degenerate-answer flags of y0_text
  prompt_y0_text, prompt_y0_valid          edit sequences only, where it was decoded (the 200 edits of the VLKEB family that
                 have hard negatives): the decoder-v2 pre-edit answer of the sequence's own prompt and its validity.  It is
                 NOT the teacher-forced text of the sequence (that is the edit target, `answer`); no key belongs to it.
  pre_exact, pre_token_acc, pre_n_tok, pre_finite, pre_first_mismatch      pre-edit model on the teacher-forced sequence
                 (Evaluator.tf at W_0, B = 1): edits -> target already produced; others -> y0 reproduced (= y0_valid)
  lp_tf          index of the last prompt position in the teacher-forced sequence (null: no answer / no answer tokens)
  lp_decode      index of the last prompt position in the prompt-only sequence (decode path)
  lp_shift       lp_tf - lp_decode as Evaluator.lp_shift (tokenizer only): 0 = same token; 1 = the joint tokenisation leaves a
                 bare space token after the prompt (LLaVA, answers starting with a digit); null = no answer
  len_tf, len_prompt, n_image_tokens
  row_tf, row_tf_prompt   rows of keys_tf.pt (null where there is no teacher-forced sequence / no shift)
  row_decode              row of keys_decode.pt
  rows_decode_steps       11 rows of keys_decode_steps.pt (k = 0 .. 10); n_decode_distinct = number of distinct keys
  aliases                 edit_pos = edit; probe_uid = uid (probe rows); lp_prompt = lp_decode (the library's name of
                          the decode position, lp='prompt'); item_uid = the benchmark item of the query (edit item for
                          edit sequences and hard negatives, probe item, benign source item)

HOW TO USE WITH THE LIBRARY'S CONVENTIONS
  Evaluator.eligible_keys(it, S, answer=True)      keys_tf[row_tf]
  ... answer=True, lp='prompt'                     keys_tf[row_tf_prompt] if lp_shift else keys_tf[row_tf]
  ... answer=False                                 keys_decode[row_decode]
  ... answer='decode' (= decode_keys)              keys_decode_steps[rows_decode_steps]          (11 keys, some identical)
  ... answer='all'                                 keys_tf[row_tf], keys_tf[row_tf_prompt] (if lp_shift), keys_decode_steps[rows_decode_steps]
  Evaluator.own_keys('rel', answer=...)            the rows of kind 'rel' with edit < T; decode-path keys only where lp_shift == 0
  rows of a log                                    family == the log's family; kind in (rel, tgen, igen): edit < T;
                                                   hard negatives: edit < T (HN-X: also edit_b < T for a true cross row)
  benign calibration pools                         kind == 'probe' and pool == 'repair' (protocol); kind == 'benign' (label-free)
Caveats: (1) a teacher-forced key belongs to ONE answer text (the low-precision forward changes the last bits with the
sequence length; on LLaVA the position itself moves where lp_shift = 1); (2) rows with y0_valid = false are kept and
must be excluded from D1 statistics; (3) duplicates: use query_id; (4) a row without an answer (benign rows outside the
answer subset, rows with an empty y0) has NO teacher-forced key (row_tf = null): use row_decode / rows_decode_steps.
On the rows that HAVE an answer of n <= 10 tokens and the same last prompt token on both paths (lp_shift = 0), the
teacher-forced key is bit for bit keys_decode_steps[rows_decode_steps[n]] (same sequence length; the count is in
verify_cpu.json: tf_key_vs_decode_step_of_same_length).  For a row without an answer the same identity is expected
(the model is causal: the key depends on how many tokens follow, not on which), but it cannot be checked there.
"""


def hn_text(BB, rows, hn_edits):
    c = collections.Counter((r["hn_type"], r["group"], "new" if r["new"] else "existing") for r in rows)
    v = collections.Counter((r["hn_type"], bool(r["y0_valid"])) for r in rows)
    return f"""HARD NEGATIVES of backbone {BB}: the existing rows of 08_rescope/p11_retrofit/out/hard_negatives/{BB} (unchanged,
annotated) + the new rows of the final experiments.  Same file format: rescope.load_hard_negatives(H, hn_dir=<parent
of this directory>) reads rows.jsonl.  Built by 10_final/cache/code/fc_select.py (selection, CPU, seeded) and
fc_capture.py (decoder-v2 pre-edit answers, keys); counts: ../../cache_table.txt, select summary: work/{BB}/select.

Rows: {dict(sorted((' / '.join(k), n) for k, n in c.items()))}
Valid pre-edit answers (y0 reproduced by the pre-edit model): {dict(sorted((f'{a} valid={b}', n) for (a, b), n in v.items()))}

CONSTRUCTION (hn_build.py's rules, without its category restriction for the new rows)
  source pool   VLKEB eval items that are not probe items and share no (image, src, alt), no Freebase MID and no entity
                name with ANY edit of the project's VLKEB edit streams, including the long streams of both backbones.
  HN-I          the edit's image + an unrelated question: the image-locality question (m_loc_q) of a source item drawn
                uniformly (per-edit seeded permutation; a question equal to the edit's question / rephrase or already
                used for the edit is skipped).  rule = 'final_unrestricted': the category of the question is NOT
                restricted; `same_category` records whether it shares a category (lexicon of hn_build.py) with the
                edit's question or rephrase.  The existing rows (k = 0, 1 of the first 100 edits, rule =
                'p11_category_restricted') were drawn only from other categories (same_category = false for all).
  HN-Q          the edit's question + an unrelated image: the entity image of a source item (other MID, other names).
  HN-X          the image of edit a + the question of edit b (edit_uid = a, edit_b_uid = b): one seeded cyclic order per
                group of 100 edits, every edit gives its image once and its question once; b's question differs from
                a's question, rephrase and HN-I questions; the two edits share no MID and no name.
  groups        first100 = stream positions 0..99 (6 HN-I + 6 HN-Q + 1 HN-X row per edit as image owner);
                later100 = 100 positions >= 100 drawn with a fixed seed (2 HN-I + 2 HN-Q + 1 HN-X):
                {hn_edits['later100']}
                per prefix: {hn_edits['later100_per_prefix']}
  pre-edit answer   decoder v2 (tb_common.Prober.decode_v2, ptype 'image'): y0_text (10 greedy ids, first line), long_text
                (48 tokens); y0_valid = non-empty AND reproduced by the pre-edit model under teacher forcing; invalid rows
                are KEPT (y0_invalid_reason).  deg_* = degenerate answers (empty, single character, '?', bare None / No).

FIELDS added to the P11 fields: rule, group, new, edit (= edit_pos, stream position), edit_b (HN-X), src_in_final_pool,
src_related_edit_first_pos (existing rows: smallest stream position of an edit of THIS backbone's long stream that the
row's source item is related to by the pool rule; null = unrelated.  A row is a clean negative for a log with T edits
only if this is null or >= T), q_in_edit / q_in_edit_first_pos (the row's question string is also the question or
rephrase of an edit of the stream, first such position), q_in_probe, img_in_probe, query_in_probe, lp_tf, lp_decode,
lp_shift, pre_exact, query_id, n_same_query, deg_*; HN-X: edit_b_uid, edit_b_src, edit_b_alt, edit_b_pred,
long_contains_edit_b_pred / _alt, y0_equals_pre_rel_y0_b.  related_flag as hn_build.py (HN-X: either edit).
keys.pt: hn_ids, K_last (teacher-forced key, float32), last_prompt_idx.  edit_pre.jsonl: decoder-v2 pre-edit answers of
the edits' own prompts (rel / tgen / igen) for the 200 edits with hard negatives.
"""


def benign_text(BB, rows):
    n = collections.Counter(r["ptype"] for r in rows)
    s = collections.Counter(r["ptype"] for r in rows if r["answer_subset"])
    return f"""BENIGN POOL of backbone {BB}: label-free benign queries, {dict(n)} rows; decoder-v2 pre-edit answer for a
seeded subset {dict(s)} (answer_subset = true).  Built by 10_final/cache/code/fc_select.py + fc_capture.py.

Source: VLKEB eval items that are neither one of the 500 probe items nor an edit of the backbone's long stream.  Each
item offers the same two rows as a probe item: its image-locality row (m_loc image + m_loc_q, fmt vqa) and its
text-locality row (loc; text format of the backbone).  The probe items are a uniform sample of the eval split, so
the remaining ITEMS are exchangeable with the probe items.  The ROWS are not a like-for-like sample of the probe rows:
items are visited in one seeded order and a row is taken only if its query (image content + question string) is not
a probe query of the VLKEB family and was not taken before.  The benchmark repeats image-locality rows across items,
so the pool holds distinct queries only and leaves out the image rows that the probes also ask (counts: select
summary, field `exists`); the probe rows keep their repeats.  Text rows do not repeat.  Draws from this pool are
exchangeable with each other; whether they stand in for the probe rows is an assumption of the user, not a property
established here.  The pool is the same for both backbones (the flags against the edits differ).
rows.jsonl: sid ('benign:<ptype>:<n>'), ptype, uid (source item), prompt, image, fmt, ref_answer (benchmark reference,
never used for keys), answer_subset; flags: q_in_edit (question string, stripped lower case, is the question or
rephrase of an edit of the long stream; q_in_edit_first_pos = first such stream position), img_in_edit (image content
is an edit image or image rephrase), q_in_probe / img_in_probe / query_in_probe (against the probe rows),
img_entity_in_edit / img_entity_in_probe (same Freebase MID), `overlap` = the same flags per family of the backbone
(qwen25vl: also against the MMKE-Bench edits and probes); subset rows: y0_text, y0_valid, y0_invalid_reason, long_text,
deg_*, pre_correct_long; all rows: lp_decode, query_id, row_decode (row of ../../{BB}/keys_decode.pt; the 11 decode
forwards: index.jsonl -> rows_decode_steps), row_tf (subset rows with a teacher-forced sequence).
"""
