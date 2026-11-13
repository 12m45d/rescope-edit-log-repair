"""Tests of white.benparts: the parts of the benign pool (R2)."""
from __future__ import annotations

import json
import os

import numpy as np

import tcommon as T
from tcommon import C, test
from white import benparts as BP


def _pool(bb):
    return C.jsonl_read(BP.pool_path(bb))


def _stored(bb):
    with open(BP.parts_path(bb)) as f:
        return json.load(f)


@test("parts", "the stored split is the documented rule (re-coded here): sorted uids, default_rng(20261111).permutation, 20 / 45 / 35 percent of the items")
def p1_split_rule():
    out, ok = {}, True
    for bb in BP.pool_backbones():
        uids = sorted({r["uid"] for r in _pool(bb)})
        n = len(uids)
        perm = np.random.default_rng(20261111).permutation(n)
        n_w, n_s = int(round(0.20 * n)), int(round(0.45 * n))
        mine = {}
        for rank, i in enumerate(perm):
            mine[uids[int(i)]] = "BEN-W" if rank < n_w else ("BEN-S" if rank < n_w + n_s else "BEN-T")
        obj = _stored(bb)
        cnt = {p: sum(v == p for v in mine.values()) for p in BP.PARTS}
        pct = {p: 100.0 * cnt[p] / n for p in BP.PARTS}
        good = mine == obj["items"] and cnt == obj["n_items"] and all(abs(pct[p] - BP.PERCENT[p]) <= 100.0 / n for p in BP.PARTS) and obj["seed"] == 20261111
        ok &= good
        out[bb] = {"n_items": n, "items_per_part": cnt, "percent": pct, "equal_to_file": mine == obj["items"]}
    q = out["qwen25vl"]
    return {"pass": ok, "per_backbone": out,
            "note": f"{q['n_items']} items -> W / S / T = {q['items_per_part']['BEN-W']} / {q['items_per_part']['BEN-S']} / {q['items_per_part']['BEN-T']} "
                    f"({q['percent']['BEN-W']:.2f} / {q['percent']['BEN-S']:.2f} / {q['percent']['BEN-T']:.2f} %); file == rule for {len(out)} backbones"}


@test("parts", "no source item lies in two parts; every pool row has exactly the part of its item; the parts cover the pool")
def p2_no_item_in_two_parts():
    out, ok = {}, True
    for bb in BP.pool_backbones():
        rows, obj = _pool(bb), _stored(bb)
        parts_of_item = {}
        for r in rows:
            parts_of_item.setdefault(r["uid"], set()).add(obj["parts"][r["sid"]])
        two = sum(len(v) > 1 for v in parts_of_item.values())
        wrong = sum(obj["parts"][r["sid"]] != obj["items"][r["uid"]] for r in rows)
        cover = set(obj["parts"]) == {r["sid"] for r in rows} and len(rows) == len(obj["parts"]) == sum(obj["n_rows"].values())
        uid_sets = {p: {r["uid"] for r in rows if obj["parts"][r["sid"]] == p} for p in BP.PARTS}
        inter = sum(len(uid_sets[a] & uid_sets[b]) for a in BP.PARTS for b in BP.PARTS if a < b)
        ok &= two == 0 and wrong == 0 and cover and inter == 0
        out[bb] = {"items_in_two_parts": two, "rows_with_another_part_than_their_item": wrong, "parts_cover_pool": cover, "uids_shared_between_parts": inter,
                   "rows": obj["n_rows"]}
    q = out["qwen25vl"]
    return {"pass": ok, "per_backbone": out, "note": f"items in two parts: {max(v['items_in_two_parts'] for v in out.values())}; rows W / S / T = "
            f"{q['rows']['BEN-W']} / {q['rows']['BEN-S']} / {q['rows']['BEN-T']} of {sum(q['rows'].values())}"}


@test("parts", "the pool holds the same rows for every backbone and the item split (and the row split) is the same")
def p3_same_split_all_backbones():
    bbs = BP.pool_backbones()
    objs = {bb: _stored(bb) for bb in bbs}
    ident = {bb: BP.pool_identity(_pool(bb)) for bb in bbs}
    same_pool = len(set(ident.values())) == 1
    same_items = all(objs[bb]["items"] == objs[bbs[0]]["items"] for bb in bbs)
    same_rows = all(objs[bb]["parts"] == objs[bbs[0]]["parts"] for bb in bbs)
    diff = {bb: sum(a.get("fmt") != b.get("fmt") for a, b in zip(_pool(bbs[0]), _pool(bb))) for bb in bbs[1:]}      # what DOES differ: the text format
    flag = {bb: sum(bool(a.get("q_in_edit")) != bool(b.get("q_in_edit")) for a, b in zip(_pool(bbs[0]), _pool(bb))) for bb in bbs[1:]}
    return {"pass": same_pool and same_items and same_rows and set(C.BACKBONES) <= set(bbs), "backbones": bbs, "pool_identity_sha1": ident,
            "rows_with_another_fmt_than_" + bbs[0]: diff, "rows_with_another_q_in_edit_flag_than_" + bbs[0]: flag,
            "note": f"{', '.join(bbs)}: same rows (sid, uid, ptype, prompt, image, answer subset): {same_pool}; same item split: {same_items}; "
                    f"differing fields: text format ({'/'.join(str(v) for v in diff.values())} rows), flags against the edit stream ({'/'.join(str(v) for v in flag.values())} rows)"}


@test("parts", "determinism: a re-run in a fresh process (other hash seeds) reproduces every stored file byte for byte")
def p4_determinism():
    res, ok = {}, True
    for hs in ("0", "987"):
        d = T.tmpdir(f"parts_h{hs}")
        rc, log = T.child("white.benparts", ["build", "--dir", d], env={"PYTHONHASHSEED": hs})
        files = sorted(f for f in os.listdir(C.PARTS_DIR))
        same = {f: os.path.exists(os.path.join(d, f)) and T.read_bytes(os.path.join(d, f)) == T.read_bytes(os.path.join(C.PARTS_DIR, f)) for f in files}
        ok &= rc == 0 and all(same.values()) and sorted(os.listdir(d)) == files
        res[f"PYTHONHASHSEED={hs}"] = {"exit": rc, "identical": same}
        T.rmtmp(f"parts_h{hs}")
    inproc = all(BP.render(bb).encode() == T.read_bytes(BP.parts_path(bb)) for bb in BP.pool_backbones())
    return {"pass": ok and inproc, "runs": res, "in_process_render_equals_file": inproc,
            "note": f"{len(os.listdir(C.PARTS_DIR))} files x 2 fresh processes (PYTHONHASHSEED 0, 987) + in-process render: all byte-identical: {ok and inproc}"}


@test("parts", "the split depends on the SET of uids only (row order, repeated uids)")
def p5_order_independence():
    rows = _pool("qwen25vl")
    uids = [r["uid"] for r in rows]
    ref = BP.item_split(uids)
    rng = np.random.default_rng(1)
    sh = [uids[int(i)] for i in rng.permutation(len(uids))]
    ok = BP.item_split(sh) == ref and BP.item_split(sorted(set(uids))) == ref and BP.item_split(uids + uids[:100]) == ref
    drop = BP.item_split([u for u in uids if u != uids[0]])                    # another set of items -> another permutation (documented: pool-specific)
    return {"pass": ok, "n_items_changing_part_when_one_item_is_removed": sum(drop[u] != ref[u] for u in drop),
            "note": "shuffled rows / de-duplicated uids / repeated uids give the same split: " + str(ok)}


@test("parts", "loader: verification passes; sids / rows of a part in pool order; rows(part, key cache) are the benign rows of the key cache")
def p6_loader():
    from white.common import use_library
    use_library()
    from rescope import KeyCache
    out, ok = {}, True
    for bb in C.BACKBONES:
        P = BP.load(bb)
        kc = KeyCache(bb)
        rows = _pool(bb)
        order = [r["sid"] for r in rows]
        good = True
        pos_all = []
        for p in BP.PARTS:
            sids = P.sids(p)
            good &= sids == [s for s in order if P.part(s) == p]
            pos = P.rows(p, kc)
            good &= [kc.index[j]["sid"] for j in pos] == sids and all(kc.index[j]["kind"] == "benign" for j in pos)
            good &= all(kc.index[j]["uid"] == r_uid for j, r_uid in zip(pos, [r["uid"] for r in rows if P.part(r["sid"]) == p]))
            pos_all += pos
        good &= sorted(pos_all) == kc.rows(kind="benign") and len(set(pos_all)) == len(rows)
        good &= P.sids(("BEN-W", "BEN-S")) == [s for s in order if P.part(s) in ("BEN-W", "BEN-S")]
        m = P.mask(order[:10])
        good &= all(sum(m[p][i] for p in BP.PARTS) == 1 for i in range(10))
        try:
            P.sids("BEN-X")
            good = False
        except ValueError:
            pass
        ok &= good
        out[bb] = {"ok": good, "n_rows": {p: len(P.sids(p)) for p in BP.PARTS}}
    return {"pass": ok, "per_backbone": out, "note": "sids(part) / rows(part, kc) / mask / unknown part refused: " + str(ok)}


@test("parts", "loader refuses a pool file that is not the file the split was made from")
def p7_loader_detects_changed_pool():
    d = T.tmpdir("pool")
    src = BP.pool_path("llava")
    rows = C.jsonl_read(src)
    rows[5]["uid"], rows[6]["uid"] = rows[6]["uid"], rows[5]["uid"]              # two rows swap their source items
    p = os.path.join(d, "rows.jsonl")
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    orig = BP.pool_path
    raised = None
    try:
        BP.pool_path = lambda bb: p
        try:
            BP.load("llava")
        except RuntimeError as e:
            raised = str(e)[:120]
    finally:
        BP.pool_path = orig
        T.rmtmp("pool")
    ok_after = BP.load("llava").n_rows == _stored("llava")["n_rows"]
    return {"pass": raised is not None and ok_after, "error": raised, "note": "changed pool file -> RuntimeError: " + str(raised is not None)}


@test("parts", "summary counts recomputed here from the pool rows (rows by ptype, answer subset, valid y0, teacher-forced key, q_in_edit, per-prefix)")
def p8_summary_recount():
    out, ok = {}, True
    for bb in BP.pool_backbones():
        rows, obj = _pool(bb), _stored(bb)
        bad = []
        for p in BP.PARTS:
            sub = [r for r in rows if obj["parts"][r["sid"]] == p]
            s = obj["summary"]["parts"][p]
            mine = {"rows.all": len(sub), "rows.image": sum(r["ptype"] == "image" for r in sub), "rows.text": sum(r["ptype"] == "text" for r in sub),
                    "n_items": len({r["uid"] for r in sub}),
                    "answer_subset.rows.all": sum(bool(r["answer_subset"]) for r in sub),
                    "answer_subset.rows.image": sum(bool(r["answer_subset"]) and r["ptype"] == "image" for r in sub),
                    "answer_subset.y0_valid.all": sum(bool(r["answer_subset"]) and bool(r.get("y0_valid")) for r in sub),
                    "answer_subset.with_teacher_forced_key.all": sum(bool(r["answer_subset"]) and r.get("row_tf") is not None for r in sub),
                    "q_in_edit.rows.all": sum(bool(r["q_in_edit"]) for r in sub), "q_in_edit.rows.text": sum(bool(r["q_in_edit"]) and r["ptype"] == "text" for r in sub),
                    "q_in_edit.first_edit_below_T.100": sum(r["q_in_edit_first_pos"] is not None and r["q_in_edit_first_pos"] < 100 for r in sub),
                    "q_in_edit.first_edit_below_T.2000": sum(r["q_in_edit_first_pos"] is not None and r["q_in_edit_first_pos"] < 2000 for r in sub),
                    "flags.q_in_probe.all": sum(bool(r["q_in_probe"]) for r in sub), "flags.query_in_probe.all": sum(bool(r["query_in_probe"]) for r in sub),
                    "flags.img_in_edit.all": sum(bool(r["img_in_edit"]) for r in sub)}
            for k, v in mine.items():
                x = s
                for part in k.split("."):
                    x = x[part]
                if x != v:
                    bad.append((p, k, x, v))
        ok &= not bad
        out[bb] = {"n_cells": 3 * 15, "differing": bad[:5],
                   "rows_tf_outside_answer_subset": sum(r.get("row_tf") is not None and not r["answer_subset"] for r in rows)}
    return {"pass": ok, "per_backbone": out, "note": f"{sum(v['n_cells'] for v in out.values())} summary cells recounted, differing: {sum(len(v['differing']) for v in out.values())}"}


@test("parts", "benign rows against the probe rows, edits and hard negatives of the key caches (same query = same image content + question): none in the VLKEB families; MMKE-Bench counted")
def p9_benign_vs_cached_queries():
    """The uses of the parts rest on this: BEN-T rows are 'fresh test rows' and BEN-W keys are disjoint from the calibration draws only if no
    benign query is a probe query of the evaluated log's family."""
    out, ok = {}, True
    for bb in C.BACKBONES:
        idx = C.jsonl_read(os.path.join(C.CACHE, bb, "index.jsonl"))
        obj = _stored(bb)
        by_q = {}
        for e in idx:
            by_q.setdefault(e["query_id"], []).append(e)
        vlkeb = {"qwen25vl": "Q", "llava": "L"}[bb]
        cnt = {}
        for e in idx:
            if e["kind"] != "benign":
                continue
            for o in by_q[e["query_id"]]:
                if o["kind"] == "benign":
                    continue
                k = (o["family"], "probe:" + str(o.get("pool")) if o["kind"] == "probe" else o["kind"], obj["parts"][e["sid"]])
                cnt[k] = cnt.get(k, 0) + 1
        dup_benign = sum(sum(o["kind"] == "benign" for o in v) > 1 for v in by_q.values())
        in_vlkeb = sum(n for (fam, _, _), n in cnt.items() if fam == vlkeb)
        ok &= in_vlkeb == 0 and dup_benign == 0
        out[bb] = {"benign_rows_equal_to_a_sequence_of_the_VLKEB_family": in_vlkeb, "benign_queries_occurring_twice_in_the_pool": dup_benign,
                   "other_families": {f"{fam} {kind} / {part}": n for (fam, kind, part), n in sorted(cnt.items()) if fam != vlkeb}}
    q = out["qwen25vl"]["other_families"]
    tot = {p: sum(n for k, n in q.items() if k.endswith(p)) for p in BP.PARTS}
    return {"pass": ok, "per_backbone": out,
            "note": f"no benign query is a probe / edit / hard-negative query of the VLKEB family (Qwen, LLaVA) and none repeats; on Qwen {sum(tot.values())} benign rows ARE probe queries "
                    f"of the MMKE-Bench family (BEN-W {tot['BEN-W']}, BEN-S {tot['BEN-S']}, BEN-T {tot['BEN-T']}; REPAIR and held-out): on TB-Q*-mmke logs the parts are not disjoint from the probes"}
