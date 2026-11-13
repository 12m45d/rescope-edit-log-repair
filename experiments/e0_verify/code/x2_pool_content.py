"""Extra check X2 (CPU): the reporting pool HELD-OUT-DISJOINT is defined by image PATH.  By image CONTENT (md5 of the file, verified
in check 2) and question string: how many valid HELD-OUT-DISJOINT rows are the same model input as a REPAIR row (a calibration
candidate), and how many share only the question string.   python x2_pool_content.py -> out/x2_pool_content.json"""
import vcommon as V

res = {}
for bb, fams in (("qwen25vl", ("Q", "QK")), ("llava", ("L",))):
    idx = V.read_index(bb)
    for fam in fams:
        pr = [e for e in idx if e["family"] == fam and e["kind"] == "probe"]
        ann = V.annotate(pr)
        key = lambda e: (e["ptype"], e["prompt"], e["image_md5"] if e["image"] else None)      # noqa: E731
        rep = {key(e) for e, a in zip(pr, ann) if a["pool"] == "repair"}
        repq = {(e["ptype"], " ".join(e["prompt"].lower().split())) for e, a in zip(pr, ann) if a["pool"] == "repair"}
        hd = [(e, a) for e, a in zip(pr, ann) if a["disjoint"] and e["y0_valid"]]
        res[fam] = {"n_valid_heldout_disjoint": len(hd), "n_equal_to_a_repair_row_by_content": sum(key(e) in rep for e, a in hd),
                    "n_sharing_the_question_string_with_a_repair_row": sum((e["ptype"], " ".join(e["prompt"].lower().split())) in repq for e, a in hd),
                    "n_strict": sum(a["strict"] for e, a in hd), "pool_flags_equal_index": all(a["pool"] == e["pool"] and a["disjoint"] == bool(e["disjoint"]) and a["strict"] == bool(e["strict"]) for e, a in zip(pr, ann)),
                    "n_distinct_queries_by_content_in_heldout_disjoint": len({key(e) for e, a in hd})}
        print(fam, res[fam])
V.jdump(res, f"{V.OUT}/x2_pool_content.json")

# ---- duplicates inside the calibration pool, and held-out rows that ask the question of a logged edit (development logs)
extra = {}
for tag, d in V.DEV.items():
    idx = V.read_index(d["backbone"])
    log = V.read_log(d["log"])
    pr = [e for e in idx if e["family"] == d["family"] and e["kind"] == "probe"]
    ann = V.annotate(pr)
    key = lambda e: (e["ptype"], e["prompt"], e["image_md5"] if e["image"] else None)      # noqa: E731
    rep = [e for e, a in zip(pr, ann) if a["pool"] == "repair" and e["y0_valid"]]
    cnt = {}
    for e in rep:
        cnt[key(e)] = cnt.get(key(e), 0) + 1
    hd = [e for e, a in zip(pr, ann) if a["disjoint"] and e["y0_valid"]]
    eq = {" ".join(e["prompt"].lower().split()) for e in idx if e["family"] == d["family"] and e["kind"] in ("rel", "tgen") and e["edit"] < log["T"]}
    low = lambda e: " ".join(e["prompt"].lower().split())                                  # noqa: E731
    extra[tag] = {"n_repair_valid": len(rep), "n_repair_valid_rows_in_duplicate_query_groups": sum(c for c in cnt.values() if c > 1), "n_distinct_repair_queries": len(cnt),
                  "n_heldout_disjoint_valid": len(hd), "n_heldout_disjoint_image_rows_asking_the_question_of_a_logged_edit": sum(e["ptype"] == "image" and low(e) in eq for e in hd),
                  "n_repair_image_rows_asking_the_question_of_a_logged_edit": sum(e["ptype"] == "image" and low(e) in eq for e in rep)}
    print(tag, extra[tag])
res["development_logs"] = extra
V.jdump(res, f"{V.OUT}/x2_pool_content.json")
