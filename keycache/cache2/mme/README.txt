10_final/cache2/mme -- the MME benchmark as an out-of-benchmark benign workload (PROTOCOL.md section 6, amendment "Clarification of H3 (R6)
and R17", item (2)): captured under the PRE-EDIT model of both backbones like the benign pool, split by image into MME-C and MME-T.
Capture only: no gate, no calibration and no edited model was run on these rows.  Written by code/mme_finish.py.

  mme_pool.py           the reader (glue): mme_pool.load(backbone) -> rescope KeyCache + rows of a part + library items; ten-line example in
                        its docstring.  test_mme_pool.py / test_mme_pool.json: its test.
  <backbone>/           qwen25vl (layer 20), llava (layer 16): index.jsonl, keys_tf.pt, keys_decode.pt, keys_decode_steps.pt, rows.jsonl, meta.json,
                        capture_parts.json, README.txt (construction, counts, format, validity and answers per category), verify.json
  images/               1177 image files <content md5>.<jpg|png> (the benchmark's bytes); mme_rows.jsonl: the 2374 benchmark rows; extract_summary.json
  select/               selection.jsonl (rows, parts, flags), manifest_<backbone>.jsonl (what the capture read), select_summary.json, check_select.json
  summary.json          the counts below as JSON
  code/                 mme_extract.py -> mme_select.py -> mme_capture.py (calls 10_final/cache/code/fc_capture.main) -> mme_assemble.py ->
                        mme_verify.py -> mme_finish.py; run_bb.sh (the GPU chain of a backbone, through the queue); mme_precheck.py (CPU dry run
                        of the prompt construction); mme_common.py
  logs/, work/          logs; work/capture/<backbone>/parts (the raw capture parts), work/proof (the proof on the existing benign pool),
                        work/verify (verification results), work/select_recheck and work/extract_recheck (second runs), work/test_pool
                        (the synthetic copy of the test), work/hcache (private copy of the harness' cache directory)

2374 rows, 1177 images, 14 categories; split by image, stratified by category, seed 20261111: MME-C 589 images / 1188 rows; MME-T 588 images / 1186 rows.

backbone  part   rows  images  valid y0  valid %  y0 yes  y0 no  y0 other  y0 = ref %
--------  -----  ----  ------  --------  -------  ------  -----  --------  ----------
qwen25vl  MME-C  1188     589      1187    99.92     586    599         3       85.19
qwen25vl  MME-T  1186     588      1186   100.00     578    607         1       84.32
qwen25vl  all    2374    1177      2373    99.96    1164   1206         4       84.75
llava     MME-C  1188     589      1188   100.00     546    642         0       74.58
llava     MME-T  1186     588      1186   100.00     567    619         0       75.72
llava     all    2374    1177      2374   100.00    1113   1261         0       75.15

valid y0 = decoder-v2 pre-edit answer reproduced by the pre-edit model under teacher forcing; yes / no / other = the pre-edit answer; y0 = ref =
it equals the benchmark's reference answer.  Verification (per backbone, <backbone>/verify.json): qwen25vl ok = True; llava ok = True.
