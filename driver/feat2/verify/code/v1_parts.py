"""Check 1: parts of the benign pool (R2).  Own code; reads only the pool files, the parts files and the whitening statistics.
   python v1_parts.py  -> out/v1_parts.json"""
import collections
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vcommon as V

out = {"checks": {}}
ok_all = True


def chk(name, ok, **info):
    global ok_all
    ok_all = ok_all and bool(ok)
    out["checks"][name] = dict(ok=bool(ok), **info)
    print(("ok  " if ok else "FAIL"), name, {k: (v if len(str(v)) < 300 else str(v)[:300] + "...") for k, v in info.items()}, flush=True)


BBS = ("qwen25vl", "llava", "blip2")
pool = {b: V.jl(f"{V.CACHE}/benign/{b}/rows.jsonl") for b in BBS}
parts = {b: V.jload(f"{V.CACHE2}/benign_parts/{b}.json") for b in BBS}

# ---- the pool is the same query set for the backbones (what "the same items for both backbones" can mean)
ident = {b: [(r["sid"], r["uid"], r["ptype"], r["prompt"], r["image"]) for r in pool[b]] for b in BBS}
chk("pool_same_rows_all_backbones", ident["qwen25vl"] == ident["llava"] == ident["blip2"], n_rows={b: len(pool[b]) for b in BBS})

for b in BBS:
    P, rows = parts[b], pool[b]
    part_of_sid, part_of_item = P["parts"], P["items"]
    # every row has a part; the part of a row is the part of its item
    miss = [r["sid"] for r in rows if r["sid"] not in part_of_sid]
    wrong = [r["sid"] for r in rows if part_of_sid.get(r["sid"]) != part_of_item.get(r["uid"])]
    extra = sorted(set(part_of_sid) - {r["sid"] for r in rows})
    by_item = collections.defaultdict(set)
    for r in rows:
        by_item[r["uid"]].add(part_of_sid[r["sid"]])
    two = [u for u, s in by_item.items() if len(s) > 1]
    n_items = collections.Counter(part_of_item.values())
    n_rows = collections.Counter(part_of_sid.values())
    n = len(by_item)
    share = {p: 100.0 * n_items[p] / n for p in n_items}
    chk(f"{b}:split_by_item", not miss and not wrong and not extra and not two and set(part_of_item) == set(by_item),
        n_items=dict(n_items), n_rows=dict(n_rows), percent_of_items={k: round(v, 3) for k, v in share.items()}, items_in_two_parts=len(two),
        rows_without_part=len(miss), rows_not_in_part_of_item=len(wrong), sids_not_in_pool=len(extra))
    # proportions: 20 / 45 / 35 percent of the ITEMS (largest deviation from the exact share, in items)
    dev = {p: n_items[p] - q * n / 100.0 for p, q in (("BEN-W", 20), ("BEN-S", 45), ("BEN-T", 35))}
    chk(f"{b}:proportions", all(abs(v) < 1.0 for v in dev.values()), deviation_in_items={k: round(v, 2) for k, v in dev.items()}, n_items_total=n)
    # own re-derivation of the rule that the file states (sorted distinct uids, numpy default_rng(20261111).permutation)
    items = sorted({r["uid"] for r in rows})
    perm = np.random.default_rng(20261111).permutation(len(items))
    nW, nS = int(round(0.20 * len(items))), int(round(0.45 * len(items)))
    mine = {}
    for rank, i in enumerate(perm.tolist()):
        mine[items[i]] = "BEN-W" if rank < nW else ("BEN-S" if rank < nW + nS else "BEN-T")
    chk(f"{b}:rule_rederived", mine == part_of_item, n_w=nW, n_s=nS, n_t=len(items) - nW - nS, seed=20261111)
    # what an "item" is: rows per item; the uid field against the row's own content
    per = collections.Counter(len([1 for r in rows if r["uid"] == u]) for u in list(by_item)[:0])  # placeholder (cheap count below)
    cnt = collections.Counter(collections.Counter(r["uid"] for r in rows).values())
    chk(f"{b}:rows_per_item", set(cnt) <= {1, 2}, histogram=dict(cnt),
        items_image_only=sum(1 for u in by_item if {r["ptype"] for r in rows if r["uid"] == u} == {"image"}) if b == "qwen25vl" else None)

# ---- the same items for both backbones
chk("same_item_split_all_backbones", parts["qwen25vl"]["items"] == parts["llava"]["items"] == parts["blip2"]["items"]
    and parts["qwen25vl"]["parts"] == parts["llava"]["parts"] == parts["blip2"]["parts"], n_items=len(parts["qwen25vl"]["items"]))

# ---- leakage between parts that an item split does not exclude (counted; own computation, image content = md5 recomputed from the files)
rows, P = pool["qwen25vl"], parts["qwen25vl"]["parts"]
md5 = {}
for r in rows:
    if r["image"] is not None and r["image"] not in md5:
        md5[r["image"]] = V.md5_file(V.img_path(r["image"]))
q = collections.defaultdict(set)
im = collections.defaultdict(set)
qu = collections.defaultdict(set)
ent = collections.defaultdict(set)
for r in rows:
    p = P[r["sid"]]
    q[p].add((r["ptype"], V.nq(r["prompt"])))
    if r["image"] is not None:
        im[p].add(md5[r["image"]])
        ent[p].add(r["image"].split("/")[0])
    qu[p].add((r["ptype"], V.nq(r["prompt"]), md5.get(r["image"])))
leak = {}
for a, b_ in (("BEN-T", "BEN-W"), ("BEN-T", "BEN-S"), ("BEN-S", "BEN-W")):
    sub = [r for r in rows if P[r["sid"]] == a]
    leak[f"{a} rows whose QUERY (ptype, question, image content) is in {b_}"] = sum((r["ptype"], V.nq(r["prompt"]), md5.get(r["image"])) in qu[b_] for r in sub)
    leak[f"{a} rows whose question string is in {b_} (image rows / text rows)"] = [sum(r["ptype"] == t and (t, V.nq(r["prompt"])) in q[b_] for r in sub) for t in ("image", "text")]
    leak[f"{a} image rows whose image content is in {b_}"] = sum(r["image"] is not None and md5[r["image"]] in im[b_] for r in sub)
    leak[f"{a} image rows whose image ENTITY directory (Freebase mid) is in {b_}"] = sum(r["image"] is not None and r["image"].split("/")[0] in ent[b_] for r in sub)
chk("leakage_between_parts(counts)", leak["BEN-T rows whose QUERY (ptype, question, image content) is in BEN-W"] == 0
    and leak["BEN-T rows whose QUERY (ptype, question, image content) is in BEN-S"] == 0, **leak,
    n_distinct_queries=len({(r["ptype"], V.nq(r["prompt"]), md5.get(r["image"])) for r in rows}), n_rows=len(rows))

# ---- the whitening statistics use BEN-W rows only, all of them, and nothing of BEN-T / BEN-S
torch = V.lib()
for b in ("qwen25vl", "llava"):
    blob = torch.load(f"{V.CACHE2}/whiten/{b}/stats.pt", map_location="cpu")
    sids = list(blob["sids"])
    Pb = parts[b]["parts"]
    want = [r["sid"] for r in pool[b] if Pb[r["sid"]] == "BEN-W"]
    idx = {e["sid"]: e for e in V.jl(f"{V.CACHE}/{b}/index.jsonl") if e["kind"] == "benign"}
    Kd = torch.load(f"{V.CACHE}/{b}/keys_decode.pt", map_location="cpu", mmap=True)
    rows_dec = torch.tensor([int(idx[s]["row_decode"]) for s in sids])
    same_keys = bool(torch.equal(Kd[rows_dec], blob["keys"]))
    chk(f"{b}:whitening_statistics_are_BEN-W_only", sorted(sids) == sorted(want) and len(set(sids)) == 600 and same_keys
        and all(Pb[s] == "BEN-W" for s in sids), n=len(sids), parts_of_sids=dict(collections.Counter(Pb[s] for s in sids)),
        keys_equal_cache_decode_keys_bitwise=same_keys, which=blob.get("which"), key_rows_equal=[int(x) for x in rows_dec] == [int(x) for x in blob["key_rows"]])

out["ok"] = ok_all
V.jdump(out, V.OUT + "/v1_parts.json")
print("ALL OK" if ok_all else "SOME FAILED")
