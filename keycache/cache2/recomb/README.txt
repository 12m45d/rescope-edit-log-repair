10_final/cache2/recomb -- log-derived recombinations and their last-prompt keys (PROTOCOL.md section 6, amendment "Review round 1", item R5:
variant rescope_x / <gate>_x).  Calibration input, never test rows.  Code, tests and manual: 10_final/feat2/recomb (USAGE.txt).

  <backbone>/            qwen25vl (long stream of TB-Q2000-ae-l2_3), llava (long stream of TB-L1000-ae-l2_3)
      rows.jsonl             the selection: 4 X-I + 4 X-Q + 4 X-X rows per edit, in stream order (rc_select.py; seed 20261111)
      select_summary.json    counts, redraw statistics, flags, sha256 of rows.jsonl
      check_select.json      rc_check.py: reproduction byte for byte, disjointness from every test query, the rules re-derived
                             (check_select_fast.json: the same with stored image md5, written by the tests)
      verify_capture.json    proof of the capture path before the run: cached hard-negative and probe keys re-captured bit for bit
      keys.pt                rids, K_last float32 (n, d), last_prompt_idx, seq_len, n_image_tokens, ...   (rc_capture.py + rc_assemble.py)
      recheck_capture.json   sampled recombination rows captured a second time, bit for bit
      capture_meta.json      md5 of keys.pt, capture parts, label-free consistency checks
      README.txt             the description of the backbone's rows and keys, with the numbers (rendered by rc_assemble.py)
  work/<backbone>/parts  capture parts of 200 rows (resumable; keys in the module dtype)
  work/hcache            the library's private harness cache of this build (copies / symbolic links; read-only use)
  logs/                  capture logs

Reader: recomb.store(backbone), recomb.log_rows(log), recomb.own_baseline(log, score), recomb.calibrate_two_block_x(...).
