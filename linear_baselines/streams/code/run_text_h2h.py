"""P12, second priority -- the rq1 head-to-head on TEXT complaints (rq1 stopped with image complaints only).

usage: CUDA_VISIBLE_DEVICES=k python run_text_h2h.py --log TB-Q100-ae-l2_3 [--n 30] [--limit K]

An adapted copy of 07_repair_v3/rq1/code/run_rq1.py (which is not touched; its rows are not touched): the same
five methods with the same options, through the same shared harness, on TEXT complaints only:

  complaints  the text rows of rq1's own sample, harness.sample_complaints(H, 100, seed=1) (random D1 v2
              complaints of the REPAIR pool, 50 image + 50 text): the first --n (30) text complaints in draw
              order = a uniform random subset of the REPAIR-pool text D1 complaints.  rq1 would have reached
              exactly these rows next (it processes the 50 image rows first); complaint_index keeps rq1's
              numbering (50 + j), so the rows can be pooled with rq1's image rows.
  NONE        no correction (complaint-only evaluation, except the first complaint: full evaluation as a check)
  COMP        repair("COMP", [c], protect=("K_ans",), query="answer", form="adapter")
  REEDIT      repair("REEDIT", [c], variant="k0+kp")                     -- exactly rq1's call
  RB_FO       repair("ROLLBACK_FO", [c], budget=10)
  RB_BEH      repair("ROLLBACK_BEH", [c], budget=20)
  REEDIT_FB   (supplement, ONLY for complaints whose REEDIT z* was not an oracle and so was not installed)
              repair("REEDIT", [c], variant="k0+kp", chunk=1): in harness/methods.py (as of 2026-11-09 18:47)
              the documented z* oracle fallback ladder only runs when `chunk` is given, so rq1's call makes one
              vendor attempt; this row shows what the documented method does.  It is reported separately and
              never replaces the REEDIT row.
Evaluation: harness.evaluate(H, corr, [c]) with the defaults (D2 decode for complaints that are also D2; all
logged edits; HELD-OUT collateral; T-/I-Loc).  Solves on the CPU (fp64) for both logs, batch deltas on the CPU
(= rq1's --lowmem path).  Rows -> out/text_h2h/rows_<log>.jsonl; finished complaints are skipped on rerun.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p12_common import *  # noqa: F401,F403,E402

import torch  # noqa: E402

SEED = 1
N_RQ1 = 100
RQ1 = [("NONE", "NONE", {}),
       ("COMP", "COMP", {"protect": ("K_ans",), "query": "answer", "form": "adapter"}),
       ("REEDIT", "REEDIT", {"variant": "k0+kp"}),
       ("RB_FO", "ROLLBACK_FO", {"budget": 10}),
       ("RB_BEH", "ROLLBACK_BEH", {"budget": 20})]
REEDIT_FB = ("REEDIT_FB", "REEDIT", {"variant": "k0+kp", "chunk": 1})
TDIR = os.path.join(OUT, "text_h2h")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cap_gb", type=float, default=None)
    a = ap.parse_args()
    os.makedirs(TDIR, exist_ok=True)
    fh = open(os.path.join(LOGD, f"text_h2h_{a.log}.log"), "a")
    log(f"==== run_text_h2h {vars(a)} pid {os.getpid()} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", fh)
    H, cap = load_log(a.log, a.cap_gb)
    start_watchdog(fh)
    if H.checks.get("collapsed_working_point"):
        raise SystemExit("collapsed working point")
    B = baseline(H)
    full_sample = sample_complaints(H, N_RQ1, seed=SEED)
    text = [(ci, c) for ci, c in enumerate(full_sample) if c["ptype"] == "text"][: a.n]
    pl = SM.pool_of(H)
    assert all(pl[c["probe_idx"]] == "repair" and c["is_complaint_D1"] and c["y0_valid"] for _, c in text)
    log(f"loaded {a.log}; cap {cap} GiB; solve device {H.solve_dev}; checks {H.checks}; baseline {B['checks']}", fh)
    log(f"{len(text)} text complaints (rq1 sample seed {SEED}, indices {text[0][0]}..{text[-1][0]}): {[c['group'] for _, c in text]}", fh)
    path = os.path.join(TDIR, f"rows_{a.log}.jsonl")
    done = set()
    if os.path.exists(path):
        for l in open(path):
            done.add(json.loads(l)["group"])
    n_new, skipped = 0, []
    for j, (ci, c) in enumerate(text):
        if c["group"] in done:
            continue
        yield_if_blocking(fh, f"(before text complaint {j})")
        try:
            rows = one_complaint(H, a, j, ci, c, fh)
        except Exception as e:  # noqa: BLE001
            if not is_oom(e):
                raise
            kind = oom_kind(str(e)) if isinstance(e, torch.OutOfMemoryError) else "card_full"
            log(f"[{j}] {c['group']}: CUDA out of memory ({kind}; cap {cap} GiB): {str(e)[:300]}", fh)
            if kind != "own_cap":
                leave_card(fh, f"card full during text complaint {j}")      # does not return
            H.reset()                                       # does not fit under my cap: complaint not done
            torch.cuda.empty_cache()
            skipped.append(c["group"])
            continue
        with open(path, "a") as f:
            for row in rows:
                f.write(json.dumps(clean(row)) + "\n")
        torch.cuda.empty_cache()
        n_new += 1
        if a.limit and n_new >= a.limit:
            break
    log("DONE" + (f"; NOT done (out of memory under the process cap): {skipped}" if skipped else ""), fh)


def one_complaint(H, a, j, ci, c, fh):
    """The five RQ1 methods (+ the REEDIT fallback supplement if needed) on one complaint -> rows."""
    if True:
        rows = []
        configs = list(RQ1)
        k = 0
        while k < len(configs):
            cid, method, cfg = configs[k]
            k += 1
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            corr, cost = repair(method, H, [c], **dict(cfg))
            t1 = time.time()
            full = not (cid == "NONE" and j > 0)
            rec = evaluate(H, corr, [c], edits=full, collateral=full, loc=full)
            meta = {kk: v for kk, v in corr.meta.items() if kk not in ("attempt_log", "_targets")}
            row = {"log": a.log, "seed": SEED, "config": cid, "method": method, "cfg": cfg, "group": c["group"],
                   "uid": c["uid"], "ptype": c["ptype"], "probe_idx": c["probe_idx"], "complaint_index": ci,
                   "text_index": j, "n_ans_tok": c["n_ans_tok"],
                   "is_D2": bool(c.get("is_complaint_D2")), "full_eval": full,
                   "cost": cost, "meta": meta, "eval": rec, "sec_repair_wall": t1 - t0,
                   "sec_eval_wall": time.time() - t1, "peak_mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30,
                   "gpu_procs": gpu_procs()}
            rows.append(row)
            res = rec["complaints"][0]["D1_resolved"]
            log(f"[{j}] {c['group']} {cid}: D1 {res} broken {rec.get('edits_broken_total')} "
                f"lost {rec.get('collateral', {}).get('all', {}).get('noncomplaints_lost')} |corr| {rec['fro']:.3f} "
                f"passes {cost['passes']} {cost['sec']['total']:.2f}s (eval {time.time() - t1:.0f}s)", fh)
            if cid == "REEDIT" and meta.get("not_installed_rows"):
                configs.append(REEDIT_FB)                   # supplement: the documented fallback ladder
            del corr
        return rows


if __name__ == "__main__":
    main()
