RECOMBINATIONS of backbone qwen25vl: log-derived near-scope calibration rows of the variant rescope_x / <gate>_x (PROTOCOL.md section 6,
amendment "Review round 1", item R5).  They are CALIBRATION INPUT (per-edit baselines), never test rows: no label, no answer, no complaint;
their queries are disjoint from every test query (check_select.json).  Built by 10_final/feat2/recomb: rc_select.py (selection, CPU,
seed 20261111) -> rc_check.py -> rc_capture.py (keys, GPU) -> rc_assemble.py.  Reader: recomb.store('qwen25vl') / recomb.log_rows(log)
(feat2/recomb/core.py, USAGE.txt).

Rows: 24000 = 4 of each kind for each of the 2000 edits of the long stream of family Q (TB-Q2000-ae-l2_3): {'X-I': 8000, 'X-Q': 8000, 'X-X': 8000}
Usable by a log with the first T edits (owner position < T; X-X partner position < T = row field min_T <= T):
    T =   10: X-I    40  X-Q    40  X-X    21   (10 edits with fewer than 4 X-X rows)
    T =   20: X-I    80  X-Q    80  X-X    46   (20 edits with fewer than 4 X-X rows)
    T =   50: X-I   200  X-Q   200  X-X   145   (37 edits with fewer than 4 X-X rows)
    T =  100: X-I   400  X-Q   400  X-X   400
    T =  200: X-I   800  X-Q   800  X-X   800
    T =  500: X-I  2000  X-Q  2000  X-X  2000
    T = 1000: X-I  4000  X-Q  4000  X-X  4000
    T = 2000: X-I  8000  X-Q  8000  X-X  8000

CONSTRUCTION (modelled on the hard negatives: 10_final/cache/hard_negatives/qwen25vl/README.txt)
  source pool   the source pool of the hard negatives, recomputed and checked against its record: VLKEB eval items that are not probe
                items and share no (image, src, alt), no Freebase MID and no entity name with ANY edit of the project's VLKEB edit streams
                (1051 items, 426 distinct image-locality questions).
  X-I           the edit's image + the image-locality question (m_loc_q) of a pool item (seeded permutation of the pool per edit).
  X-Q           the edit's question (src) + the entity image (`image`) of a pool item (seeded permutation per edit).
  X-X           the edit's image + the question (src) of another edit j of the same stream (edit_b_pos = j, edit_b_uid):
                stream position < 100: 2 partners from the edit's batch of 10 (scope 'batch'), then 2 from its group of 100 (scope 'group');
                later positions: 4 from its group of 100.  Counts: {'batch': 200, 'group': 7800}; batches that could not supply 2 partners: 0.
  discarded and redrawn (the next candidate of the permutation), in this order:
    own_question  the question string (stripped, lower case) is the question or the rephrase of the edit            [X-I, X-X]
    test_query    (image content = md5 of the file, question string) is the query of ANY row of a key cache under 10_final/cache
                  (qwen25vl, llava, QO, blip2: edit prompts and rephrases, probe rows, hard negatives of all kinds,
                  benign rows)
    related       X-X: the two edits share an entity (Freebase MID), a name or an answer (normalised pred / alt)
    repeat        (image content, question string) repeats an earlier recombination of the same edit (any kind)
  redraws       X-I: 250 discarded candidates for 8000 rows ({'own_question': 41, 'repeat': 131, 'test_query': 78}; 238 rows with a redraw, at most 3); X-Q: 111 discarded candidates for 8000 rows ({'test_query': 111}; 108 rows with a redraw, at most 2); X-X: 319 discarded candidates for 8000 rows ({'own_question': 90, 'related': 2, 'repeat': 189, 'test_query': 38}; 301 rows with a redraw, at most 3)

FLAGS (as the hard negatives' rows carry them)
  q_in_edit / q_in_edit_first_pos   the row's question string is the question or rephrase of an edit of the stream; first such stream position
                (a row is 'cross-like' for a log with T edits iff that position < T).  Rows with the flag: {'X-I': 4230, 'X-Q': 8000, 'X-X': 8000} (X-Q and X-X by construction);
                X-I rows that are cross-like by log length: {'10': 2, '20': 2, '50': 12, '100': 40, '200': 187, '500': 760, '1000': 1863, '2000': 4230}
  same_category the question shares a category of hn_build.py's lexicon with the edit's question or rephrase (X-Q: the partner item's own
                question with the edit's question): {'X-I': 2472, 'X-Q': 2354, 'X-X': 2490}
  q_in_probe {'X-I': 4775, 'X-Q': 2708, 'X-X': 2476}, img_in_probe {'X-I': 0, 'X-Q': 0, 'X-X': 0}, query_in_probe {'X-I': 0, 'X-Q': 0, 'X-X': 0} (against the probe rows of the family)
  n_redraws / redraws   candidates discarded before the row was accepted, by reason
  NOT a flag, but to be known: partner items come from the SAME source pool as the test hard negatives.  6736 X-I rows ask a question string
  that a test HN-I row asks about another image (320 distinct strings), 3804 X-Q rows show an image that a test HN-Q row shows with another question
  (597 distinct images).  No recombination QUERY (image content + question) is a test query.

OTHER FIELDS of rows.jsonl: rid ('<kind>:<edit uid>:<k>'), kind, family, edit_uid, edit_pos (= edit: the OWNING edit's stream position), k,
image, prompt, fmt ('vqa'), ptype ('image'), image_md5, src_uid, src_field, edit_b_uid, edit_b_pos (= edit_b), scope, min_T, q_cats,
edit_cats, edit_src, edit_rephrase, edit_alt, edit_pred, src_item_question / src_item_pred (X-Q), edit_b_src / edit_b_alt / edit_b_pred (X-X),
query_id / n_same_query (rows of this file with the same query: 23762 distinct queries; rows in duplicate groups {'X-I': 0, 'X-Q': 456, 'X-X': 0}: two edits with the same question drew the same pool image).
There is NO pre-edit answer, no teacher-forced sequence and no validity flag: only the last-prompt key is captured.

keys.pt: rids (the order of rows.jsonl), K_last FloatTensor [24000, 11008] float32, last_prompt_idx, seq_len, n_image_tokens, layer, backbone,
module_dtype (torch.bfloat16), rows_sha256, doc.  K_last[r] = input of the edited module (layer 20, mlp.down_proj) at the last prompt token of the
PROMPT-ONLY sequence (no answer tokens) of row r, pre-edit model, B = 1 (float32 image of the module-dtype values: exact).  This is the quantity
of keys_decode.pt[row_decode] of the key cache, NOT the teacher-forced key that hard_negatives/<backbone>/keys.pt holds (a recombination has
no answer).  md5 918b044f90ba2035b3bf61a3a1d99bc4; rows.jsonl sha256 59a5a489767f0bcd23b1e394b89b8d90f2771de65202df18c8f301ac5194f8f7.

CAPTURE PATH (rc_capture.py: fc_capture.make_evaluator / make_item, Evaluator.batch(it, answer=False), Evaluator.decode_keys(it, 0))
  proof before the run (verify_capture.json): 50 existing hard-negative rows ({'hn_i': 18, 'hn_q': 26, 'hn_x': 6}) and 50 probe rows ({'image': 21, 'text': 29})
  re-captured at B = 1: prompt-only key bit for bit equal to keys_decode.pt 50/50 and 50/50; teacher-forced key equal to keys_tf.pt
  50/50 and 50/50 (hard negatives also equal to hard_negatives/qwen25vl/keys.pt: 50/50).
  after the run (recheck_capture.json): 96/96 sampled recombination rows bit for bit identical on a second capture without
  the vision-feature memo (96 of them in the process of their first capture; the proof across processes is the
  comparison with the cached keys above).
  label-free consistency (capture_meta.json): last prompt index = last token of the sequence: True; duplicate-query groups
  with identical keys 218/218; X-I X-X rows with the image token count of the edit's own sequence 16000/16000; X-Q rows with the text token count of the edit's own prompt 8000/8000.
  120 parts, 3207.7 GPU-seconds (0.1337 s per row), ['NVIDIA GeForce RTX 5090'], torch ['2.8.0+cu128'], master log TB-Q2000-ae-l2_3.

SELECTION CHECK (rc_check.py -> check_select.json; ok = True): two re-runs in fresh interpreters reproduce rows.jsonl and
select_summary.json byte for byte; 24000 rows against 42400 test rows with an image (17238 distinct queries; image md5 recomputed
from 10491 files): 0 equal exactly, 0 equal after normalising the question; rule violations 0.
Test rows compared: every image sequence of the key caches QO, blip2, llava, qwen25vl, their hard-negative and benign row files, the raw edit stream and probe records of the family, the one-hop portability queries.

FILES  rows.jsonl, select_summary.json, check_select.json, keys.pt, capture_meta.json, verify_capture.json, recheck_capture.json, README.txt;
       ../work/<backbone>/parts (capture parts, resumable), ../logs (capture logs).
