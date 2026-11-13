"""Step 1: the MME benchmark (local parquet copy) -> image files named by their content md5 and the row table.

    python -I -B mme_extract.py [--out DIR]

Reads   $RESCOPE_EXT_VIEW/data/datasets/MME/data/*.parquet  (read-only; files in name order, rows in file
        order).  Every row = question_id, image {bytes, path}, question, answer (Yes | No), category.
Writes  images/<md5>.<ext>   the image bytes exactly as stored in the benchmark (md5 of the file = md5 of the bytes = the file's
                             name; <ext> = the lower-case extension of the benchmark's own file name, jpg | png);
        mme_rows.jsonl       the row table: one line per benchmark row, in benchmark order;
        extract_summary.json counts (rows, images, categories), image formats and sizes, how often an image is asked.
--out DIR writes the two files and the images under DIR instead; --rows-only writes no image file (determinism check: the
row table and the summary of a second run are compared with the files in use).  No time stamp is written: a re-run
reproduces both files byte for byte.  CPU only; no model.
The parquet content is data, not code: it is parsed by pyarrow, image bytes are only hashed, measured with PIL and copied.
"""
from __future__ import annotations

import argparse
import collections
import glob
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mme_common as M  # noqa: E402

EXT_OK = {"jpg": "jpg", "jpeg": "jpg", "png": "png"}
PIL_EXT = {"JPEG": "jpg", "MPO": "jpg", "PNG": "png"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=M.MME)
    ap.add_argument("--rows-only", action="store_true", help="do not write image files (determinism check of the row table and the summary)")
    args = ap.parse_args()
    import pyarrow.parquet as pq
    from PIL import Image
    img_dir = os.path.join(args.out, "images")
    os.makedirs(img_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(M.SRC, "data", "*.parquet")))
    if not files:
        raise SystemExit(f"no parquet files under {M.SRC}/data")
    rows, seen, n_written, n_kept = [], {}, 0, 0
    for f in files:
        pf = pq.ParquetFile(f)
        k = 0
        for batch in pf.iter_batches(batch_size=32):
            for r in batch.to_pylist():
                data = r["image"]["bytes"]
                md5 = M.md5_bytes(data)
                if md5 not in seen:
                    with Image.open(io.BytesIO(data)) as im:
                        fmt, (w, h), mode = im.format, im.size, im.mode
                        im.convert("RGB").load()                        # the conversion the model wrapper applies must succeed
                    ext = EXT_OK.get(os.path.splitext(r["image"].get("path") or "")[1].lstrip(".").lower()) or PIL_EXT.get(fmt)
                    if ext is None:
                        raise RuntimeError(f"{r['question_id']}: unknown image type {fmt!r}")
                    name = f"{md5}.{ext}"
                    p = os.path.join(img_dir, name)
                    if args.rows_only:
                        pass
                    elif os.path.exists(p) and os.path.getsize(p) == len(data) and M.md5_file(p) == md5:
                        n_kept += 1
                    else:
                        M.write_atomic(p, data)
                        n_written += 1
                    seen[md5] = {"file": name, "width": w, "height": h, "format": fmt, "mode": mode, "n_bytes": len(data)}
                s = seen[md5]
                q = r["question"]
                rows.append({"mme_idx": len(rows), "parquet_file": os.path.basename(f), "parquet_row": k, "question_id": r["question_id"],
                             "category": r["category"], "question": q, "ref_answer": r["answer"], "image_md5": md5, "image_file": s["file"],
                             "image_name_in_benchmark": r["image"].get("path"), "width": s["width"], "height": s["height"],
                             "format": s["format"], "mode": s["mode"], "n_bytes": s["n_bytes"], "ends_with_suffix": q.endswith(M.SUFFIX)})
                k += 1
    M.jsonl_write(rows, os.path.join(args.out, "mme_rows.jsonl"))
    cats = list(dict.fromkeys(r["category"] for r in rows))
    per_img = collections.defaultdict(list)
    for r in rows:
        per_img[r["image_md5"]].append(r)
    px = sorted(v[0]["width"] * v[0]["height"] for v in per_img.values())
    asp = sorted(max(v[0]["width"], v[0]["height"]) / min(v[0]["width"], v[0]["height"]) for v in per_img.values())
    summ = {
        "source": M.SRC, "parquet_files": [{"file": os.path.basename(f), "md5": M.md5_file(f), "n_rows": sum(r["parquet_file"] == os.path.basename(f) for r in rows)}
                                           for f in files],
        "n_rows": len(rows), "n_images": len(per_img), "n_categories": len(cats),
        "n_question_ids": len({r["question_id"] for r in rows}),
        "n_distinct_question_strings": len({r["question"] for r in rows}),
        "n_distinct_queries(image content, question string)": len({(r["image_md5"], r["question"]) for r in rows}),
        "n_questions_ending_with_suffix": sum(r["ends_with_suffix"] for r in rows), "suffix": M.SUFFIX,
        "reference_answers": dict(collections.Counter(r["ref_answer"] for r in rows)),
        "rows_per_image": {str(k): v for k, v in sorted(collections.Counter(len(v) for v in per_img.values()).items())},
        "images_in_two_categories": sorted(["+".join(sorted({x["category"] for x in v})), k] for k, v in per_img.items() if len({x["category"] for x in v}) > 1),
        "by_category": {c: {"rows": sum(r["category"] == c for r in rows), "images": len({r["image_md5"] for r in rows if r["category"] == c}),
                            "ref_yes": sum(r["category"] == c and r["ref_answer"] == "Yes" for r in rows),
                            "ref_no": sum(r["category"] == c and r["ref_answer"] == "No" for r in rows)} for c in cats},
        "image_formats": dict(collections.Counter(v[0]["format"] for v in per_img.values())),
        "image_modes": dict(collections.Counter(v[0]["mode"] for v in per_img.values())),
        "image_pixels": {"min": px[0], "median": px[len(px) // 2], "max": px[-1]},
        "image_aspect_ratio(long side / short side)": {"median": round(asp[len(asp) // 2], 3), "max": round(asp[-1], 3)},
        "image_bytes_total": sum(v[0]["n_bytes"] for v in per_img.values()),
        "question_chars": {"min": min(len(r["question"]) for r in rows), "max": max(len(r["question"]) for r in rows)},
        "row_table_md5": M.md5_file(os.path.join(args.out, "mme_rows.jsonl")),
        "image_file_name": "<md5 of the image bytes>.<jpg | png>; the bytes are those of the benchmark, unchanged",
    }
    M.jdump(summ, os.path.join(args.out, "extract_summary.json"))
    print(f"rows {summ['n_rows']}, images {summ['n_images']}, categories {summ['n_categories']}; images written {n_written}, already present {n_kept}; "
          f"row table md5 {summ['row_table_md5']}")
    print(M.text_table([(c, v["rows"], v["images"], v["ref_yes"], v["ref_no"]) for c, v in summ["by_category"].items()],
                       head=("category", "rows", "images", "ref yes", "ref no"), right=(1, 2, 3, 4)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
