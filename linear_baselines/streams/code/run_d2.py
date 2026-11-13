"""P12 -- D2 complaints among the HELD-OUT rows after a linear repair (testbed D2 pipeline, decoder v2).

usage: CUDA_VISIBLE_DEVICES=k python run_d2.py --log TB-Q100-ae-l2_3 [--configs NAME,NAME] [--wait]

D2 definition of the testbed (06_testbed_v3/code/y0v2.py = the decoder-v2 version of d2_long.py /
recompute_d2.py): a probe row is a D2 complaint if the pre-edit 48-token decoder-v2 answer contains the
reference (stored `pre_correct_long`) and the 48-token decoder-v2 answer of the evaluated model does not
(tb_common.contains_ref; tb_common.Prober.decode_v2 = 10 no-cache greedy forwards + plain greedy KV-cache
continuation to 48 tokens).  Here the evaluated model is the repaired log (correction applied through the
harness: fp64 adapter hook or replacement weight), and only HELD-OUT rows are decoded: every HELD-OUT row
with pre_correct_long (valid y0 or not, as y0v2.py).  Configs: NONE (must reproduce the stored post-edit
answers and D2 flags), every method at n in {10, 100, ALL} of seed 0 (corrections saved by run_streams.py in
out/<log>/corr/), and ALL = W_0 (must reproduce the stored pre-edit answers: zero D2).

Output: out/<log>/d2/<config>.json (resumable).  --wait: poll until every expected correction file exists.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p12_common import *  # noqa: F401,F403,E402

import torch  # noqa: E402

# priority: end point, the batch methods of the summary table, then the rest; the sequential COMP variants
# (the same operator as the batch solve up to round-off, see extra.identity_vs_batch) come last


D2_CORE = ("COMP_B", "COMP_BP", "REEDIT_B", "RB_TOT", "RB_UNION")


def d2_plan(which="all"):
    """core: NONE, then the five distinct methods of the summary table at n = ALL, 100, 10 (a whole n at a time, so
    an interrupted pass leaves complete columns), then W_0.
    core ends with the exploratory COMP_BR (n = ALL) when its correction exists.
    all:  + REEDIT-S (= REEDIT-B up to the half-precision writes), COMP-B-ALLPOS (n = 10) and the sequential COMP
          variants (= the joint solve up to round-off)."""
    out = ["NONE"]
    for n in ("ALL", 100, 10):
        out += [config_name(m, n, D2_SEED) for m in D2_CORE]
    out.append("ALL")
    out.append(config_name("COMP_BR", "ALL", D2_SEED))
    if which == "core":
        return out
    for m in ("REEDIT_S", "COMP_B_ALLPOS", "COMP_SP", "COMP_S"):
        for n in ("ALL", 100, 10):
            if m == "COMP_B_ALLPOS" and n not in N_ALLPOS:
                continue
            out.append(config_name(m, n, D2_SEED))
    return out


def get_corr(H, odir, name):
    """Correction of a config: saved file, alias of a saved file, or an end point.  None = not there yet."""
    if name == "NONE":
        return Correction("none", meta={"method": "NONE"})
    if name == "ALL":
        return Correction("weight", W=H.W0.clone(), meta={"method": "ROLLBACK_ALL"})
    p = os.path.join(odir, "corr", f"{name}.pt")
    al = os.path.join(odir, "corr", f"{name}.alias")
    if not os.path.exists(p) and os.path.exists(al):
        p = os.path.join(odir, "corr", open(al).read().strip() + ".pt")
    if not os.path.exists(p):
        rec = os.path.join(odir, f"{name}.json")
        if os.path.exists(rec) and jload(rec)["eval"]["fro"] == 0.0:        # a repair that changed nothing
            return Correction("none", meta={"method": "NONE"})
        return None
    return load_corr(p, H.dev)


def decode_rows(H, corr, rows):
    out = []
    H.reset()
    with H.applied(corr), HE.vision_cache(H):
        for r in rows:
            y, lt, info = H.prober.decode_v2(H.probes[r["probe_idx"]], r["ptype"])
            out.append({"group": r["group"], "uid": r["uid"], "ptype": r["ptype"], "y0_valid": bool(r["y0_valid"]),
                        "stored_D1": bool(r["is_complaint_D1"]), "stored_D2": bool(r["is_complaint_D2"]),
                        "post_correct": bool(TB.contains_ref(lt, r["ref_answer"])),
                        "y_eq_y0": y == r["y0_text"], "long_eq_stored_post": lt == r["post_long_text"],
                        "long_eq_stored_pre": lt == r["pre_long_text"],
                        "degenerate": (not lt.strip()) or lt.strip() in ("?", "The"), "long": lt})
    H.reset()
    return out


def heldout_keys(H, fh):
    """Edited-module inputs at the answer-predicting positions of every valid HELD-OUT row.
    DIAGNOSTIC ONLY: read by module_restore() below to measure generalisation on the HELD-OUT pool; no
    method ever sees this file (its name is not one the harness looks for)."""
    p = os.path.join(MYCACHE, f"heldout_anskeys_{H.name}.pt")
    if os.path.exists(p):
        return torch.load(p)
    rows = HE.heldout_rows(H)
    ks, cnt = [], []
    H.reset()
    with HE.vision_cache(H):
        for r in rows:
            b = H.complaint_batch(r)
            K = H.capture_inputs(b)
            A = int(b["answer_len"][0])
            ks.append(K[b["pred_pos"][0, :A]].cpu())
            cnt.append(A)
    d = {"K": torch.cat(ks, 0), "counts": cnt, "groups": [r["group"] for r in rows],
         "stored_D1": [bool(r["is_complaint_D1"]) for r in rows], "ptype": [r["ptype"] for r in rows]}
    tmp = p + f".tmp{os.getpid()}"
    torch.save(d, tmp)
    os.replace(tmp, p)
    log(f"  held-out answer keys captured: {len(rows)} rows, {int(d['K'].shape[0])} keys (diagnostic only)", fh)
    return d


def module_restore(H, corr, hk):
    """Per valid HELD-OUT row: rho = ||(W_rep - W_0) K_r||_F / ||(W_E - W_0) K_r||_F at the row's answer keys,
    W_rep = the repaired module (W_E + correction).  rho = 1: the repair leaves the edited module's output at
    these positions where the log put it; 0: the pre-edit output is back; > 1: further from it.
    (Only the edited module at the answer positions: the prompt positions of the same row are not in it.)"""
    K = hk["K"].to(H.solve_dev).double().T.contiguous()            # (d, N)
    DK = H.D_act @ K
    if corr.kind == "adapter":
        XK = corr.Lf.to(H.solve_dev) @ (corr.Rf.to(H.solve_dev).T @ K)
    elif corr.kind == "weight":
        XK = (corr.W.to(H.solve_dev).double() - H.WE.to(H.solve_dev).double()) @ K
    else:
        XK = torch.zeros_like(DK)
    R = DK + XK
    out, j = [], 0
    for A in hk["counts"]:
        out.append(float(R[:, j:j + A].norm() / DK[:, j:j + A].norm().clamp_min(1e-30)))
        j += A
    del K, DK, XK, R
    return out


def regression_suite(H, corr, nonc):
    """IN-SAMPLE diagnostic (REPAIR pool; never reported as collateral or generalisation): how many of the
    REPAIR-pool known-good rows -- the rows whose answer keys COMP-B+ protects and COMP_BR restores -- still
    reproduce y0 under the correction (B=1 teacher-forced full forward: the D1 check of harness.evaluate).
    Protection / restoration acts on the edited module at the answer positions only; this checks the whole row."""
    out = {"n": len(nonc), "kept": 0, "image": {"n": 0, "kept": 0}, "text": {"n": 0, "kept": 0}}
    H.reset()
    with H.applied(corr), HE.vision_cache(H):
        for r in nonc:
            ok = HE._tf(H, r)[0] is True
            out["kept"] += int(ok)
            out[r["ptype"]]["n"] += 1
            out[r["ptype"]]["kept"] += int(ok)
    H.reset()
    return out


def summarise(dr):
    def s(sub):
        sd2 = [x["stored_D2"] for x in sub]
        d2 = [not x["post_correct"] for x in sub]
        return {"n_pre_correct": len(sub), "n_D2": int(sum(d2)), "D2_rate": 100.0 * sum(d2) / max(len(sub), 1),
                "n_D2_WE": int(sum(sd2)), "D2_rate_WE": 100.0 * sum(sd2) / max(len(sub), 1),
                "stored_D2_resolved": int(sum(a and not b for a, b in zip(sd2, d2))),
                "new_D2": int(sum(b and not a for a, b in zip(sd2, d2))),
                "D2_flags_equal_stored": int(sum(a == b for a, b in zip(sd2, d2))),
                "long_eq_stored_post": int(sum(x["long_eq_stored_post"] for x in sub)),
                "long_eq_stored_pre": int(sum(x["long_eq_stored_pre"] for x in sub)),
                "y_eq_y0": int(sum(x["y_eq_y0"] for x in sub)),
                "n_degenerate": int(sum(x["degenerate"] for x in sub))}
    out = s(dr)
    out["by_ptype"] = {pt: s([x for x in dr if x["ptype"] == pt]) for pt in ("image", "text")}
    out["valid_y0_only"] = s([x for x in dr if x["y0_valid"]])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--configs", default=None)
    ap.add_argument("--set", default="all", choices=("core", "all"))
    ap.add_argument("--cap_gb", type=float, default=None)
    ap.add_argument("--wait", action="store_true", help="poll for corrections that are not saved yet")
    ap.add_argument("--max_wait_h", type=float, default=10.0)
    a = ap.parse_args()
    odir = os.path.join(OUT, a.log)
    ddir = os.path.join(odir, "d2")
    os.makedirs(ddir, exist_ok=True)
    fh = open(os.path.join(LOGD, f"d2_{a.log}.log"), "a")
    names = d2_plan(a.set) if a.configs is None else a.configs.split(",")
    log(f"==== run_d2 {vars(a)} pid {os.getpid()}; {len(names)} configs", fh)
    if all(os.path.exists(os.path.join(ddir, f"{n}.json")) for n in names):
        log("all D2 configs done", fh)
        return
    H, cap = load_log(a.log, a.cap_gb)
    start_watchdog(fh)
    pl = SM.pool_of(H)
    rows = [r for r in H.rows if pl[r["probe_idx"]] == "heldout" and r["pre_correct_long"]]
    log(f"loaded {a.log}; cap {cap} GiB; HELD-OUT pre-correct rows {len(rows)} (stored D2 {sum(r['is_complaint_D2'] for r in rows)}; "
        f"image {sum(r['ptype'] == 'image' for r in rows)} text {sum(r['ptype'] == 'text' for r in rows)})", fh)
    hk = heldout_keys(H, fh)
    _, _, nonc = repair_rows(H)
    t_start = time.time()
    pending = list(names)
    skipped = []
    while pending:
        progressed = False
        for name in list(pending):
            path = os.path.join(ddir, f"{name}.json")
            if os.path.exists(path):
                pending.remove(name)
                continue
            if os.path.exists(os.path.join(odir, f"{name}.failed.json")):
                log(f"  {name}: the repair failed -> no D2", fh)
                pending.remove(name)
                continue
            yield_if_blocking(fh, f"(before D2 of {name})")
            lock = os.path.join(ddir, f".lock_{name}")
            corr = get_corr(H, odir, name)
            if corr is None or not claim(lock):
                continue
            try:
                t0 = time.time()
                try:
                    dr = decode_rows(H, corr, rows)
                    S = summarise(dr)
                    rho = module_restore(H, corr, hk)
                    # only where it says something about the method: the compensation variants (protection /
                    # restoration of exactly these rows) -- saves shared-GPU time for the other configs
                    suite = regression_suite(H, corr, nonc) if name.startswith("COMP_") else None
                except Exception as e:  # noqa: BLE001
                    if not is_oom(e):
                        raise
                    kind = oom_kind(str(e)) if isinstance(e, torch.OutOfMemoryError) else "card_full"
                    log(f"  {name}: CUDA out of memory ({kind}; cap {cap} GiB): {str(e)[:300]}", fh)
                    if kind != "own_cap":
                        release(lock)
                        leave_card(fh, f"card full during D2 of {name}")       # does not return
                    H.reset()                                   # does not fit under my cap: not done, not a result
                    skipped.append(name)
                    pending.remove(name)
                    continue
                atomic_jdump({"log": a.log, "config": name, "summary": S, "rows": dr, "seconds": time.time() - t0,
                              "regression_suite_in_sample": suite,
                              "module_restore": {"rho": rho, "groups": hk["groups"], "stored_D1": hk["stored_D1"],
                                                 "ptype": hk["ptype"],
                                                 "def": "||(W_rep - W_0) K_r||_F / ||(W_E - W_0) K_r||_F at the answer keys of "
                                                        "each valid HELD-OUT row (diagnostic; 1 = unrepaired, 0 = pre-edit)"},
                              "decoder": "tb_common.Prober.decode_v2 (10 no-cache + KV-cache greedy to 48 tokens)",
                              "rows_def": "HELD-OUT rows with stored pre_correct_long",
                              "time": time.strftime("%Y-%m-%d %H:%M:%S")}, path)
                log(f"  {name}: D2 {S['n_D2']}/{S['n_pre_correct']} (W_E stored {S['n_D2_WE']}; resolved "
                    f"{S['stored_D2_resolved']}, new {S['new_D2']}); long == stored post {S['long_eq_stored_post']}, "
                    f"== stored pre {S['long_eq_stored_pre']}"
                    + ("" if suite is None else f" | REPAIR known-good rows kept {suite['kept']}/{suite['n']}") + f" | {time.time() - t0:.0f}s", fh)
                progressed = True
                pending.remove(name)
            finally:
                release(lock)
                del corr
                torch.cuda.empty_cache()
        if pending and not progressed:
            if not a.wait or time.time() - t_start > a.max_wait_h * 3600:
                log(f"not available (no saved correction): {pending}", fh)
                break
            time.sleep(120)
    log(f"run_d2 finished" + (f"; NOT decoded (out of memory under the process cap {cap} GiB): {skipped}" if skipped else ""), fh)


if __name__ == "__main__":
    main()
