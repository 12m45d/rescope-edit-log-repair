MME WORKLOAD of backbone llava (edited module: input of layer 16, model.language_model.layers.16.mlp.down_proj, d = 11008): the out-of-benchmark
benign queries of 10_final/PROTOCOL.md, section 6, amendment "Clarification of H3 (R6) and R17", item (2) = R17, captured under the
PRE-EDIT model like the benign pool and split by image into MME-C (calibration source) and MME-T (test rows).
This directory is a capture and nothing else: no gate, no calibration and no edited model was run on these rows.
Written by cache2/mme/code/mme_finish.py; every number below is computed from the files of this directory.
Reader: ../mme_pool.py (ten-line example in its docstring; test ../test_mme_pool.py).

CONSTRUCTION
  1. data       $RESCOPE_EXT_VIEW/data/datasets/MME (read-only; 4 parquet files): 2374 rows = image + question + reference
                answer (Yes | No) + category; 2374 of 2374 questions end with "Please answer yes or no.".  1177 distinct images
                (content md5), 14 categories, 1187 question ids (two rows each: one reference Yes, one No); 10 images are asked
                in two categories (4 rows).  Images -> ../images/<md5>.<jpg|png> (the benchmark's bytes, unchanged); row table
                ../mme_rows.jsonl (code/mme_extract.py).
  2. selection  ALL rows.  Split BY IMAGE (content md5), stratified by category, seed 20261111: MME-C 589 images / 1188 rows, MME-T 588
                images / 1186 rows; the rows of an image are never separated (0 images in both parts).  Rule (code/mme_select.py):
                  BY IMAGE (unit = image content, md5): MME-C (calibration source) and MME-T (test rows), 50 percent of the images each,
                  stratified by category.  Rule:
                  1. stratum of an image = the sorted set of the categories of its rows, joined by '+' (ten images are asked in two
                  categories: their strata are the four pairs that occur);
                  2. per stratum: the image md5s sorted as strings; perm = numpy.random.default_rng(useed('split', stratum)).permutation(n),
                  useed = first 32 bits of sha1('20261111|split|<stratum>');
                  3. the first n_C images of perm are MME-C, the rest MME-T; n_C = n // 2, plus 1 in every second stratum of odd size
                  (odd strata in name order: the 1st, 3rd, 5th get the extra image in MME-C, the 2nd, 4th in MME-T);
                  4. a row belongs to the part of its image: the rows of an image are never separated.
                Deterministic: a second run in a fresh process reproduces selection.jsonl and both manifests byte for byte
                (../select/check_select.json: True).  Flags: see FLAGS below.
  3. capture    code/mme_capture.py calls 10_final/cache/code/fc_capture.main() -- the function that captured the benign pool -- on a
                manifest in which every MME row is a benign IMAGE row (kind 'benign', free image query, prompt format 'vqa') with
                decode = true.  Nothing about prompts, decoding, validity or keys is implemented in cache2/mme.  Per row, B = 1:
                  decoder v2 (tb_common.Prober.decode_v2): 10 greedy ids from full no-cache forwards, cut at the first newline =
                    y0_text; the same ids + KV-cache continuation to 48 tokens = long_text;
                  validity: y0 non-empty AND reproduced by the pre-edit model under teacher forcing (Prober.tf_state);
                  the key of the edited module at the last prompt position of the PROMPT-ONLY sequence, in the 11 full forwards of a
                    decode (Evaluator.decode_keys: prompt + k filler ids, k = 0..10; k = 0 is the prompt alone);
                  the key at the last prompt position of the TEACHER-FORCED sequence prompt + y0 (Evaluator.capture), and the
                    pre-edit flags of that sequence (Evaluator.tf).
                Model: rescope.load('TB-L1000-ae-l2_3') with the edited module at its PRE-EDIT weight W_0 for the whole job
                (Evaluator.applied(H.W0), asserted at the start of every part); keys do not depend on the edit log.
                48 resumable parts of 50 rows, 2135 GPU-seconds, NVIDIA GeForce RTX 5090, torch 2.8.0+cu128,
                2026-11-11 07:04:52 .. 2026-11-11 07:39:36 (clock of the machine).
                BEFORE the run (2026-11-11 07:03:37): 30 rows of the existing benign pool were captured again through the same call and
                compared with 10_final/cache/llava/: prompt-only key 30/30, decode-step keys 30/30 rows, teacher-forced key
                30/30 bit for bit; answers, ids, validity, positions equal on 30/30 rows.
  4. assembly   code/mme_assemble.py: the bookkeeping of 10_final/cache/code/fc_assemble.py for benign rows (index fields, key rows,
                float32 images of the module-dtype keys, degenerate-answer flags, pre_correct_*).
  5. checks     code/mme_verify.py, test_mme_pool.py -> verify.json (VERIFICATION below).

COUNTS
  2374 rows, 1177 images, 14 categories; teacher-forced key for 2374 rows; valid pre-edit answer: 2374 of 2374 rows (100.00 %).
  part   rows  images  valid y0  valid %  y0 yes  y0 no  y0 other  ref yes  ref no  y0 = ref %
  -----  ----  ------  --------  -------  ------  -----  --------  -------  ------  ----------
  MME-C  1188     589      1188   100.00     546    642         0      594     594       74.58
  MME-T  1186     588      1186   100.00     567    619         0      593     593       75.72
  all    2374    1177      2374   100.00    1113   1261         0     1187    1187       75.15

  per category (valid = pre-edit answer reproduced by the pre-edit model; yes / no / other = the pre-edit answer after the testbed's
  normalisation IS the word yes / no, or anything else; y0 = ref: it equals the benchmark's reference answer):
  category               C img  C rows  C valid  T img  T rows  T valid  rows  valid  valid %  yes   no  other  y0 = ref %
  ---------------------  -----  ------  -------  -----  ------  -------  ----  -----  -------  ---  ---  -----  ----------
  code_reasoning            10      20       20     10      20       20    40     40   100.00   35    5      0       47.50
  artwork                  100     200      200    100     200      200   400    400   100.00  197  203      0       66.75
  celebrity                 85     170      170     85     170      170   340    340   100.00  198  142      0       69.41
  numerical_calculation     10      20       20     10      20       20    40     40   100.00   10   30      0       50.00
  text_translation          10      20       20     10      20       20    40     40   100.00    7   33      0       57.50
  count                     15      30       30     15      30       30    60     60   100.00   14   46      0       66.67
  color                     15      30       30     15      30       30    60     60   100.00   30   30      0       93.33
  commonsense_reasoning     35      70       70     35      70       70   140    140   100.00   95   45      0       66.43
  position                  14      28       28     16      32       32    60     60   100.00   44   16      0       70.00
  OCR                       10      20       20     10      20       20    40     40   100.00   26   14      0       75.00
  landmark                 100     200      200    100     200      200   400    400   100.00  181  219      0       83.75
  scene                    100     200      200    100     200      200   400    400   100.00  152  248      0       82.50
  existence                 16      32       32     14      28       28    60     60   100.00   28   32      0       96.67
  posters                   74     148      148     73     146      146   294    294   100.00   96  198      0       79.93
  An image that is asked in two categories counts in both rows of this table (sum of the image columns 1187, images 1177); the split keeps
  such an image whole, in its own stratum (color+count, count+existence, count+position, existence+position),
  so a category that shares images is not split exactly in half:
  position 14 / 16 images; existence 16 / 14 images (MME-C / MME-T).

PRE-EDIT ANSWERS (decoder v2, pre-edit model)
  yes 1113, no 1261, other 0 of 2374 rows; among the valid rows: yes 1113, no 1261, other 0.
  answer strings:
    'No.' x 807, 'Yes.' x 739, 'No' x 454, 'Yes' x 374
  every pre-edit answer is yes or no.
  answer length in scored tokens (teacher-forced sequence): {'1': 828, '2': 1546};
  the 10-token decode stopped by: {'eos': 2374}; the 48-token answer equals the 10-token answer on 2374 of 2374 rows
  (stop of the long decode: {'eos': 2374}).
  no invalid row.
  degenerate-answer flags (fc_common.degenerate_flags): empty 0, one character 0, '?' 0, bare None / No 1261.
  NOTE  deg_none_no is true for 1261 of the 1261 answers "No": on this yes / no workload it marks a legitimate answer,
        not a degenerate one.  A read-out that sets deg_none_no rows apart (as for the VLKEB rows) would set apart 53.12 % of this pool.
  NOTE  2374 of 2374 answers are one or two tokens long: D1 on such a row is the flip of the first answer token(s); the 48-token
        answer differs from the 10-token answer on 0 rows.

PROMPTS AND KEYS
  prompt format 'vqa' of the backbone (the format of the benign pool's image rows); question string as in the benchmark, including
  "Please answer yes or no.".  Sequence lengths in tokens (min / 5 % / median / 95 % / max):
                                min  p05  median  p95  max
  ----------------------------  ---  ---  ------  ---  ---
  MME prompt-only sequence      595  599     603  610  624
    image tokens                576  576     576  576  576
    other tokens                 19   23      27   34   48
  benign pool, 1500 image rows  590  592     593  597  603
    image tokens                576  576     576  576  576
    other tokens                 14   16      17   21   27
  1004 MME rows are longer than the longest image row of the benign pool (603 tokens).
  last prompt position: teacher-forced index minus prompt-only index (lp_shift) = {'0': 2374} (0 = the same token on both paths);
  distinct keys among the 11 decode forwards of a row (the low-precision forward changes the last bits with the length): {'1': 2374}.
  keys (prompt-only, keys_decode.pt; descriptive, no gate and no edit log involved): 2374 distinct keys for 2374 rows.
  Euclidean norm and raw cosine between two keys (min / 5 % / median / 95 % / max):
                                                     min     p05   median      p95      max
  ----------------------------------------------  ------  ------  -------  -------  -------
  norm, MME                                       8.1596  8.5793   9.0499  10.1510  12.5296
  norm, benign pool (1500 image rows)             8.1705  8.9780  10.4681  12.1597  13.9706
  cosine, two MME rows (all pairs)                0.4194  0.7377   0.8794   0.9494   0.9996
  cosine, two MME rows of one image (1227 pairs)  0.5849  0.7896   0.9555   0.9939   0.9996
  cosine, two benign-pool image rows              0.0461  0.2379   0.3634   0.8376   0.9997
  cosine, an MME row and a benign-pool image row  0.0387  0.1123   0.1638   0.2650   0.6725
  cosine of the mean keys: MME-C and MME-T 0.999915; MME and the benign pool's image rows 0.278243.
  NOTE  (medians above; wording by rule: a difference of more than 0.1 in the median cosine) two MME rows are
        closer to each other than two image rows of the benign pool (0.8794 against 0.3634); an MME row and a
        benign-pool row are less similar than rows within either pool (0.1638).
        Every MME question ends with the same sentence.  MME-C and MME-T are exchangeable with each other by construction (seeded
        split by image); these numbers describe the keys only and say nothing about any gate.

FILES (the file format of 10_final/cache/<backbone>/, described in FORMAT.txt there; rescope.keycache.KeyCache reads this directory)
  index.jsonl           2374 lines, one per row, benchmark order (= row order of keys_decode.pt); kind = 'benign', pool = 'benign',
                        family = 'MME', dataset = 'MME', sid = 'mme:<row number in the benchmark>', uid = 'mme_img:<image md5>'.
  keys_tf.pt            float32 (2374, 11008): key at the last prompt position of the teacher-forced sequence prompt + y0; row = row_tf
                        (2374 rows + 0 rows for the decode's token where lp_shift != 0, row_tf_prompt).
  keys_decode.pt        float32 (2374, 11008): key at the last token of the prompt-only sequence; row = row_decode = line number.
  keys_decode_steps.pt  float32 (2374, 11008): the DISTINCT keys of the 11 full forwards of a decode; rows_decode_steps (11 row numbers).
                        All keys are float32 images of the module-dtype values (torch.float16): exact.
  rows.jsonl            the rows in the format of 10_final/cache/benign/<backbone>/rows.jsonl (below).
  meta.json             shapes, md5 sums, capture provenance (the reader checks index and key files against it).
  capture_parts.json    the capture parts.      verify.json   verification results.      README.txt   this file.

ROW FIELDS (rows.jsonl)
  as in the benign pool: sid, ptype ('image'), uid (source item: here the IMAGE), prompt, image (absolute path), fmt ('vqa'), ref_answer
    (benchmark reference Yes | No; never used for keys), ref_field, benign_idx (row number), answer_subset (true for EVERY row: all rows
    have a decoder-v2 answer); y0_text, y0_ids, y0_gen_ids, y0_stop, y0_raw, y0_roundtrip_ids_equal, w0_reproduced, w0_n_ans_tok,
    w0_first_mismatch, y0_valid, y0_invalid_reason, y0_note, long_text, long_ids, long_stop, deg_empty, deg_single_char, deg_qmark,
    deg_none_no, pre_correct_at10 / pre_correct_long (the 10- / 48-token answer contains the reference), lp_tf, lp_decode, lp_shift,
    pre_exact, query_id, n_same_query, row_tf, row_decode; flags q_in_edit, q_in_edit_first_pos, q_in_edit_exact, img_in_edit,
    img_in_edit_first_pos, q_in_probe, q_in_probe_exact, img_in_probe, query_in_probe (against the backbone's VLKEB family) and `overlap`
    (the same flags per stream family of the backbone's caches).
  added: part (MME-C | MME-T), category, stratum (categories of the image, the unit of the stratified split), question_id, mme_idx,
    image_md5, img_in_cache, y0_yn (yes | no | other: the normalised answer is that word), y0_yn_first (by its first word), long_yn_first,
    ref_yn, pre_token_acc, pre_n_tok, len_prompt, len_tf, n_image_tokens, n_decode_distinct.
  fields of the benign pool's rows that do not apply and are absent: img_entity_in_edit, img_entity_in_edit_first_pos, img_entity_in_probe, item_rank
    (item_rank: seeded visiting order of the VLKEB items; img_entity_*: Freebase MID of a VLKEB image path).
  index.jsonl: the fields of the caches' index for a benign row (absent: img_entity_in_edit, img_entity_in_edit_first_pos, img_entity_in_probe) plus
    capture_family, category, img_in_cache, long_text, long_yn_first, mme_idx, part, question_id, ref_yn, stratum, y0_yn, y0_yn_first.
    master_log = the log whose handle carried the model (its PRE-EDIT weight); capture_family = that log's stream family.

FLAGS (selection; expected and found: none)
  image content (md5) also an image of the project's caches: 0 of 1177 images.  Compared with (distinct image contents):
    benign:blip2 1500; benign:llava 1500; benign:qwen25vl 1500; hard_negatives:blip2 559; hard_negatives:llava 816;
    hard_negatives:qwen25vl 797; index:QO 1453; index:blip2 3402; index:llava 4556; index:qwen25vl 7720;
    md5_memo_of_the_caches 12736
    (index:<dir> = every sequence of 10_final/cache/<dir>/index.jsonl: edits, image rephrases, probes, hard negatives, benign rows;
    hard_negatives / benign:<backbone> = the rows files, image files hashed again; md5_memo = every image the caches' selection read.)
    Also compared at verification time (cache2/recomb, recombination rows): qwen25vl: 0 shared of 3051 contents; llava: 0 shared of 2028 contents.
  question string (stripped, lower case) equal to the question or text rephrase of an edit: 0 rows; to a probe question: 0;
  also without the suffix "Please answer yes or no.": 0; equal to a hard-negative or benign-pool question: 0.
  104 question strings occur in both parts (the benchmark asks the same question about several images): 213 rows of MME-C and
  195 rows of MME-T; the parts are disjoint by image, not by question string.

VERIFICATION (verify.json; ok = True)
  bitwise re-capture   50 seeded random rows captured again in a fresh process with the plain library calls, WITHOUT the vision-feature
                       memo, pre-edit weight: prompt-only key 50/50, the 11 decode-step keys 50/50 rows,
                       teacher-forced key 50/50 bit for bit; decoder-v2 answer, ids, 48-token answer, validity and pre-edit
                       flags reproduced on 50/50 rows.  Rows outside the sample, [bitwise, n]:
                       {'invalid_y0': None, 'answer_not_yes_no': None, 'shifted_position': None} (None: no such row exists).
  two forwards agree   teacher-forced key (prompt + answer) = decode-step key of the same sequence length (prompt + filler ids), two
                       separate forward passes of the capture: 2374/2374 rows bit for bit.
  selection            reproduced byte for byte by a second run: True; images in both parts: 0.
  images               shared with the caches: 0 (image files hashed again; md5 sets of every cache index).
                       1177 files decode to 1177 distinct RGB contents; groups of files with identical pixels: 0 (in both parts: 0).
  pre-edit model only  forwards of the edited module in the verification: 1635, of them with a weight other than W_0: 0.
  glue (mme_pool.py)   50 items found in this key cache by content: 50; Evaluator.tf_rows with the pre-edit weight as the method returns
                       the cached pre-edit outcome for 50 of them; CPU test (../test_mme_pool.json): True.

USE (R17) AND CAVEATS
  Read-outs registered by R17: calibrated on REPAIR, false firing and D1 on MME-T; calibrated on MME-C, false firing on
  heldout_disjoint, BEN-T and MME-T.  "The bound of the calibration theorem is claimed only where calibration and test queries are
  exchangeable; the other cells measure what a shift of the workload costs, for every gate alike.  MME images may show entities
  that an edit concerns; such queries are not benign, cannot be identified from the data, and are part of the measured rate."
  * keys: which = 'decode' (prompt-only) and 'steps' exist for every row; 'tf' for the rows with a non-empty answer.  D1 needs y0_valid.
  * every row has kind = 'benign' and pool = 'benign', like the benign pool: after a concatenation with the backbone's cache, tell the
    rows apart by family == 'MME' (or the sid prefix 'mme:').  query_id starts at 1000000 (no collision with the caches' ids).
  * the unit of resampling is the image (uid); MME-C and MME-T are the same for both backbones.
  * image paths are absolute (../images/); the keys were captured from these files.
