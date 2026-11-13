10_final/cache -- key caches, benign pool and hard negatives of the final experiments (written by code/fc_table.py)

  <backbone>/                 qwen25vl (layer 20), llava (layer 16): index.jsonl, keys_tf.pt, keys_decode.pt, keys_decode_steps.pt,
                              pre_edit_flags.jsonl, FORMAT.txt (the format), meta.json, verify_{gpu,dev,cpu}.json
  hard_negatives/<backbone>/  rows.jsonl (existing + new rows: HN-I, HN-Q, HN-X), keys.pt, edit_pre.jsonl, README.txt
                              (the file format of 08_rescope's hard negatives; rescope.evaluate.HN_DIR_FINAL)
  benign/<backbone>/          rows.jsonl (label-free benign pool; decoder-v2 answers of the seeded subset), README.txt
  cache_table.json / .txt     what the caches contain, verification results (cache_table_short.txt: one compact table)
  code/                       fc_select.py (CPU selection) -> fc_capture.py (GPU, resumable parts) -> fc_assemble.py -> fc_verify.py -> fc_table.py
  work/<backbone>/select/     the selection in use: manifest.jsonl (every sequence), hn.jsonl, benign.jsonl, hn_edits.json, select_summary.json
  work/<backbone>/parts/      capture parts (one file per 100 sequences; resumable)
  logs/                       capture and verification logs

Reader: rescope.keycache.KeyCache('<backbone>').  One cache per backbone serves every log whose edit stream is a prefix of a cached
family (list: cache_table.txt, 'Logs served').  Rebuild: fc_select.py is deterministic (code/fc_check_select.sh compares a re-run byte
for byte); fc_capture.py skips finished parts; fc_assemble.py publishes each directory with one rename.

  qwen25vl: 14300 sequences, complete = True; keys_tf [11900, 11008], keys_decode [14300, 11008], keys_decode_steps [20777, 11008]
  llava: 8800 sequences, complete = True; keys_tf [6496, 11008], keys_decode [8800, 11008], keys_decode_steps [10082, 11008]

NOT in these caches: the text family (CounterFact logs TB-Q{100,500}-cf-l2_3).  Its keys already exist next to the logs and were not
captured again: 06_testbed_v3/<log>/scope_keys.pt + scope_rows.jsonl (edit prompts, paraphrase witnesses, neighbourhood negatives) and
probe_keys.pt.  They hold the TEACHER-FORCED key at the last prompt position only (no prompt-only key, no decode-step keys).
  TB-Q100-cf-l2_3: scope rows {'nbr': 1000, 'para': 200, 'rel': 100}
  TB-Q500-cf-l2_3: scope rows {'nbr': 5000, 'para': 1000, 'rel': 500}
