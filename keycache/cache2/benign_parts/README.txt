10_final/cache2/benign_parts -- parts of the label-free benign pool (PROTOCOL.md section 6, amendment "Review round 1", item R2)
Written by `python -m white.benparts build` (code: 10_final/feat2/white/benparts.py; its module docstring is the reference).

  <backbone>.json   parts {sid: BEN-W | BEN-S | BEN-T}, items {uid: part}, summary; split by SOURCE ITEM (uid), seed 20261111:
                    BEN-W 20 % of the items (statistics of baselines, extra null-repair keys), BEN-S 45 % (complaint streams, second
                    calibration source), BEN-T 35 % (fresh test rows, used for nothing else).  The item split is the same for every
                    backbone (the pool holds the same rows for all of them; checked at build time).
  Reader: white.benparts.load(backbone) -> .part(sid), .sids(part), .rows(part, kc), .summary     (see feat2/white/USAGE.txt)
  The files carry no time stamp: a re-run reproduces them byte for byte (feat2/white/tests/t_parts.py).
  blip2.json is the same item split for the scout's backbone (same pool); the amendment names no use for it.

columns: answer subset = rows with a cached decoder-v2 pre-edit answer; valid y0 = of them reproduced by the pre-edit model (the rows on
which D1 is defined); tf key = rows with a teacher-forced key; q_in_edit = the row's question string is the question or text rephrase of
an edit of the backbone's long stream (near-scope benign rows; 'edit < T' = of one of the first T edits); q_in_probe = the question string
is also asked by a probe row; QK probe query = the whole query is a probe query of the MMKE-Bench family (Qwen only).

qwen25vl: 3000 rows of 2002 source items; pool md5 d5cb6171fbe2563c0ab2d328804f9423; same pool as blip2, llava: True
  part   items  rows  image  text  answer subset  valid y0  tf key  q_in_edit  of them image  edit < 100  < 500  q_in_probe  QK probe query
  BEN-W    400   600    309   291            112       108     112        169            169          40    132         167              42
  BEN-S    901  1375    679   696            291       282     291        383            383          81    263         347              95
  BEN-T    701  1025    512   513            197       197     197        260            260          53    190         250              78
  all     2002  3000   1500  1500            600       587     600        812            812         174    585         764             215
  BEN-T rows whose question string also occurs in BEN-W / BEN-S: 230 / 254 (image rows only: True); whose image content occurs in BEN-W / BEN-S: 0 / 0

llava: 3000 rows of 2002 source items; pool md5 34f40d9a3efd09e5eb67a7114a948e81; same pool as blip2, qwen25vl: True
  part   items  rows  image  text  answer subset  valid y0  tf key  q_in_edit  of them image  edit < 100  < 500  q_in_probe  QK probe query
  BEN-W    400   600    309   291            112       111     112        149            149          61    117         167               -
  BEN-S    901  1375    679   696            291       289     291        331            331         154    268         347               -
  BEN-T    701  1025    512   513            197       195     197        232            232          93    190         250               -
  all     2002  3000   1500  1500            600       595     600        712            712         308    575         764               -
  BEN-T rows whose question string also occurs in BEN-W / BEN-S: 230 / 254 (image rows only: True); whose image content occurs in BEN-W / BEN-S: 0 / 0

blip2: 3000 rows of 2002 source items; pool md5 1b65cbdbc4caaf6fd87eaf3a9073038a; same pool as llava, qwen25vl: True
  part   items  rows  image  text  answer subset  valid y0  tf key  q_in_edit  of them image  edit < 100  < 500  q_in_probe  QK probe query
  BEN-W    400   600    309   291            112       110     111        132            132          40    132         167               -
  BEN-S    901  1375    679   696            291       290     291        263            263          81    263         347               -
  BEN-T    701  1025    512   513            197       194     195        190            190          53    190         250               -
  all     2002  3000   1500  1500            600       594     597        585            585         174    585         764               -
  BEN-T rows whose question string also occurs in BEN-W / BEN-S: 230 / 254 (image rows only: True); whose image content occurs in BEN-W / BEN-S: 0 / 0
