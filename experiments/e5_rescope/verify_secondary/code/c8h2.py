"""Check 8 (h), second part: does the composition of the pools explain heldout_disjoint - REPAIR(rest)?  Per-row firing frequencies over my own 200 draws."""
from c8h import *

def freq(name, ndraw=200):
    L, Rr, Rh, cr, ch, rep, hd = load(name)
    N = len(rep)
    fr, nr, fh = np.zeros(N), np.zeros(N), np.zeros(len(hd))
    topr, toph = np.zeros(L.T), np.zeros(L.T)
    for d in range(ndraw):
        perm = np.random.default_rng([977, 8, d]).permutation(N)
        cal, rest = perm[:200], perm[200:]
        b, eta, sc = rule(Rr[cal], np.arange(100), np.arange(100, 200))
        mr, mh = Rr - b[None, :], Rh - b[None, :]
        f, f2 = mr.max(1) > eta, mh.max(1) > eta
        fr[rest] += f[rest]; nr[rest] += 1; fh += f2
        np.add.at(topr, mr[rest][f[rest]].argmax(1), 1.0 / len(rest)); np.add.at(toph, mh[f2].argmax(1), 1.0 / len(hd))
    pr, ph = fr / np.maximum(nr, 1), fh / ndraw
    def ent(rows): return np.array([str(L.index[j].get("image") or "").split("/")[0] for j in rows])
    eents = {str(L.index[j].get("image") or "").split("/")[0] for m in ("rel", "igen") for j in L.edit_rows[m]}
    for c, rows in ((cr, rep), (ch, hd)):
        c["ent_edit"] = np.array([(L.index[j]["ptype"] == "image") and (e in eents) for j, e in zip(rows, ent(rows))])
        c["nimg"] = np.array([L.index[j].get("n_image_tokens") or 0 for j in rows])
        c["qir"] = np.array([bool(L.index[j].get("question_in_repair")) for j in rows])
        c["uid"] = np.array([L.index[j]["uid"] for j in rows])
    def cl_se(p, uid):                                              # SE of the mean of per-row frequencies, clusters = probe items (uid)
        u, inv = np.unique(uid, return_inverse=True)
        s = np.bincount(inv, weights=p - p.mean())
        return float(np.sqrt((s ** 2).sum()) / len(p) * np.sqrt(len(u) / (len(u) - 1)))
    out = {"log": name, "repair_rest": {"mean": 100 * float(pr.mean()), "se_items": 100 * cl_se(pr, cr["uid"])}, "heldout_disjoint": {"mean": 100 * float(ph.mean()), "se_items": 100 * cl_se(ph, ch["uid"])}}
    d = out["heldout_disjoint"]["mean"] - out["repair_rest"]["mean"]
    out["difference"] = {"value": d, "se": float(np.hypot(out["repair_rest"]["se_items"], out["heldout_disjoint"]["se_items"]))}
    out["difference"]["z"] = d / out["difference"]["se"]
    strata = {}
    img_r, img_h = cr["ptype"] == "image", ch["ptype"] == "image"
    qs = np.quantile(cr["len"][img_r], [1 / 3, 2 / 3])
    def strat(c, m):
        return {"text": c["ptype"] == "text", "image, near (question of an edit)": m & c["near"], "image, far, entity of an edit image": m & ~c["near"] & c["ent_edit"],
                "image, far, other entity": m & ~c["near"] & ~c["ent_edit"],
                "image, short (len tercile 1)": m & (c["len"] <= qs[0]), "image, len tercile 2": m & (c["len"] > qs[0]) & (c["len"] <= qs[1]), "image, long (tercile 3)": m & (c["len"] > qs[1]),
                "image, answer < 10 tokens": m & (c["ntok"] < 10), "image, answer = 10 tokens": m & (c["ntok"] >= 10)}
    sr, sh = strat(cr, img_r), strat(ch, img_h)
    for k in sr:
        strata[k] = {"repair_n": int(sr[k].sum()), "repair_rate": 100 * float(pr[sr[k]].mean()) if sr[k].any() else None, "heldout_n": int(sh[k].sum()), "heldout_rate": 100 * float(ph[sh[k]].mean()) if sh[k].any() else None}
    out["strata"] = strata
    # heldout_disjoint standardised to the REPAIR composition over (text | image near | image far entity | image far other) x length tercile
    def cells(c, m):
        t = np.where(c["len"] <= qs[0], 0, np.where(c["len"] <= qs[1], 1, 2))
        g = np.where(c["ptype"] == "text", 0, np.where(c["near"], 1, np.where(c["ent_edit"], 2, 3)))
        return g * 3 + np.where(c["ptype"] == "text", 0, t)
    gr, gh = cells(cr, img_r), cells(ch, img_h)
    std, wsum = 0.0, 0.0
    for g in np.unique(gr):
        w = (gr == g).mean()
        if (gh == g).any(): std += w * ph[gh == g].mean(); wsum += w
    out["heldout_standardised_to_repair_composition"] = {"rate": 100 * std / wsum, "covered_weight": float(wsum), "strata": "modality x near-scope x entity of an edit image x length tercile"}
    # concentration: how many heldout rows carry half of the firing mass; items
    o = np.sort(ph)[::-1]; cs = np.cumsum(o) / o.sum()
    o2 = np.sort(pr)[::-1]; cs2 = np.cumsum(o2) / o2.sum()
    out["concentration"] = {"heldout_rows_for_half_of_firing": int(np.searchsorted(cs, 0.5) + 1), "heldout_rows_firing_in_over_half_of_draws": int((ph > 0.5).sum()),
                            "repair_rows_for_half_of_firing": int(np.searchsorted(cs2, 0.5) + 1), "repair_rows_firing_in_over_half_of_draws": int((pr > 0.5).sum())}
    te = np.argsort(toph)[::-1][:5]
    out["top_edits_heldout"] = [{"edit": int(e), "share_of_heldout_firing": float(toph[e] / toph.sum()), "share_of_repair_firing": float(topr[e] / topr.sum())} for e in te]
    return out

if __name__ == "__main__":
    res = {n: freq(n) for n in ("TB-Q100-ae-l2_3-p50", QDEV)}
    jdump(res, OUT + "/c8h2.json")
    for n, o in res.items():
        print(n, o["repair_rest"], o["heldout_disjoint"], o["difference"], o["heldout_standardised_to_repair_composition"], o["concentration"])
        for k, v in o["strata"].items(): print("   ", k, v)
        print("   top edits", o["top_edits_heldout"])
