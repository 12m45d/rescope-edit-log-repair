"""P12 verifier -- GPU part: my own evaluation loop for NONE, W_0, the reported best linear operating point and two
(method, n) cells drawn at random (out/draw.json), per primary log.

  * corrections are REBUILT here through harness.repair with the options stated in the experiment
    (COMP: protect=("K_ans",), query="answer", form="adapter"; REEDIT: z* per complaint with
    repair("REEDIT", [c], variant="k0+kp", return_targets=True, chunk=1), install with
    repair("REEDIT", cs, variant="k0+kp", targets=z*, chunk=1); RB_TOT: repair("ROLLBACK_FO", cs, budget=30,
    offload=True, record_flags=True)); nothing is read from the experiment's cache/ or out/.../corr (the saved
    corrections are only COMPARED with mine afterwards).
  * the evaluation does NOT use harness.evaluate / harness.baseline / the suffix engine / H.applied:
      - pools and the nested stream: vcommon.my_split / my_stream (numpy, from probe_records_v2.jsonl),
      - batches: vlm.build_batch from the record fields (prompt, image, fmt, y0_text) and from the log's
        edit_batch.jsonl (image, subject, alt), B=1,
      - one FULL forward per row, argmax over the answer positions == answer ids (teacher forcing; non-finite = not
        reproduced),
      - the correction is applied by my own forward hook (adapter: fp64 side path with the documented gate) or by
        writing the replacement weight into the edited module.
  * one JSON per (log, condition) under verify/out/<log>/ ; finished conditions are skipped (resumable).

usage: CUDA_VISIBLE_DEVICES=g python gpu_verify.py --log TB-Q100-ae-l2_3 --cap_gb 9.6 --need_mib 10300
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import glob
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vcommon as VC  # noqa: E402

EXIT_RETRY = 75
T0 = time.time()


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--cap_gb", type=float, required=True, help="torch memory cap of this process (GiB)")
    ap.add_argument("--need_mib", type=int, required=True, help="card memory this job needs (cap + CUDA context)")
    ap.add_argument("--conds", default=None)
    a = ap.parse_args()
    log = a.log
    g = int(os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0])
    odir = os.path.join(VC.VOUT, log)
    os.makedirs(odir, exist_ok=True)
    draw = [tuple(x) for x in VC.jload(os.path.join(VC.VOUT, "draw.json"))["draw"][log]]
    conds = ["NONE", "W0", "BEST__COMP_B__nALL"]
    for m, n in draw:
        if m == "REEDIT_B" and log == "TB-Q100-ae-l2_3" and n in (10, 100, "ALL"):
            # the experiment's Qwen record of this cell is a FIRST-PASS record (harness default call, 1/2/4
            # complaints not installed): "theirz" = its cached z* + its installed set (install + evaluation only),
            # "asrec" = my z* + its installed set, "ladder" = my z*, the stated protocol (oracle fallback ladder)
            conds += [f"{m}__n{n}__theirz", f"{m}__n{n}__asrec", f"{m}__n{n}__ladder"]
        elif m == "REEDIT_B":
            conds += [f"{m}__n{n}__theirz", f"{m}__n{n}"]
        else:
            conds.append(f"{m}__n{n}")
    conds.append("PROTKEYS")                           # protection keys recomputed with fresh forwards
    if log == "TB-L100-ae-l2_3":
        conds.append("MODDIAG")                        # the report's module-level "why" numbers (LLaVA only)
    conds.append("RELB8")                              # the testbed's own batch-8 Rel path at W_E (stored Rel)
    if log == "TB-Q100-ae-l2_3":
        conds.append("ZREPEAT")                        # is z* reproducible run to run? (same process, same inputs, 3 repeats)
    if log == "TB-L100-ae-l2_3":
        conds.append("LOC")                            # T-Loc / I-Loc of the unrepaired log and of the best point (report: I-Loc 81.5 -> 76.7)
        conds.append("WPROJ")                          # verifier's extra: behaviour of W_0 + D P_ans (the "why" paragraph)
    if a.conds:
        conds = [c for c in conds if c in set(a.conds.split(","))]
    todo = [c for c in conds if not os.path.exists(os.path.join(odir, c + ".json"))]
    say(f"==== gpu_verify {vars(a)} card {g} pid {os.getpid()} todo {todo}")
    if not todo:
        return
    ok, info = VC.can_start(g, a.need_mib)
    if not ok:
        say(f"not starting: {info}")
        sys.exit(EXIT_RETRY)

    import numpy as np
    import torch
    torch.cuda.init()                       # CUDA context only (about 0.5 GiB); nothing else is allocated yet
    time.sleep(max(0.0, 22.0 - (time.time() - T0)))
    ok2, info2 = VC.can_start(g, a.need_mib - VC.CTX_MIB)       # second look, >= 20 s later (my context is already counted)
    grew = info["free"] - info2["free"] - VC.CTX_MIB
    if not ok2 or grew > 400:
        say(f"not starting (second look; the card lost {grew} MiB besides my context): {info2}")
        os._exit(EXIT_RETRY)
    tot = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, a.cap_gb * 2 ** 30 / tot), 0)
    try:                                    # take my whole cap at once (kept in the torch pool): closes the start race
        x = torch.empty(int(a.cap_gb * 2 ** 30 * 0.93), dtype=torch.uint8, device="cuda:0")
        del x
    except RuntimeError as e:
        say(f"could not reserve my cap: {str(e)[:200]}")
        os._exit(EXIT_RETRY)
    say(f"reserved {torch.cuda.memory_reserved() / 2 ** 30:.2f} GiB in the torch pool (cap {a.cap_gb} GiB); looks: {info} | {info2}")
    VC.Watchdog(g, say)

    sys.path.insert(0, VC.V3)
    from harness import load, repair, sample_complaints
    from harness import evaluation as HE, hlog as HL, methods as HM, sampler as SM
    from harness.common import TB, VZ
    from harness.hlog import Correction
    import vendor_v7.data as VD
    mycache = os.path.join(VC.VER, "cache")
    os.makedirs(mycache, exist_ok=True)
    HM.CACHE = HE.CACHE = HL.CACHE = mycache          # the harness never writes under 07_repair_v3 from here
    VZ.ZCACHE = os.path.join(mycache, "zcache_unused")
    VD.FINITE_CACHE_DIR = os.path.join(mycache, "finite_unused")
    HL.GPU_DELTA_BUDGET = 0                           # exact batch deltas stay on the CPU (same values)
    torch.set_num_threads(8)
    try:
        H = load(log, solve_device="cpu")
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        say(f"CUDA OOM while loading: {str(e)[:200]}")
        os._exit(EXIT_RETRY)
    vlm, dev = H.vlm, H.dev
    if str(H.solve_dev) == "cpu":           # W_E - W_0 in fp64, computed on the CPU (identical values; no GPU fp64 temporaries)
        H._D_act = H.WE.cpu().double() - H.W0.cpu().double()
    say(f"loaded {log} in {time.time() - T0:.0f}s; checks {H.checks}; d_in {H.d_in} m_out {H.m_out} T {H.T} l2 {H.cfg['l2']}")

    # ------------------------------------------------------------------ data (mine) + cross-checks vs harness
    rows = VC.records(log)
    pool = VC.my_split(rows)
    hpool = SM.pool_of(H)
    stream = VC.my_stream(rows, pool, 0)
    hbase = sample_complaints(H, 100, seed=0)
    himg = [c for c in hbase if c["ptype"] == "image"]
    htxt = [c for c in hbase if c["ptype"] == "text"]
    hstream100 = [(htxt[k // 2] if k % 2 == 0 else himg[k // 2])["group"] for k in range(100)]
    setup = VC.jload(os.path.join(VC.P12, "out", log, "_setup.json"))
    ho_rows = [r for r in rows if pool[r["probe_idx"]] == "heldout" and r["y0_valid"]]
    rp_rows = [r for r in rows if pool[r["probe_idx"]] == "repair" and r["y0_valid"]]
    import json as _json
    edits = [_json.loads(l) for l in open(os.path.join(VC.TB_ROOT, log, "edit_batch.jsonl")) if l.strip()]
    text_fmt = TB.TEXT_FMT[H.backbone]
    xc = {"split_equals_harness": all(pool[i] == hpool[i] for i in hpool) and len(pool) == len(hpool),
          "stream100_equals_harness_sampler": [c["group"] for c in stream[:100]] == hstream100,
          "stream_equals_experiment_setup": [c["group"] for c in stream] == setup["streams"]["0"],
          "records_equal_harness_rows": rows == H.rows,
          "record_fields_match_probes": all(
              (r["prompt"] == H.probes[r["probe_idx"]]["m_loc_q"] and r["image"] == H.probes[r["probe_idx"]]["m_loc"] and r["fmt"] == "vqa")
              if r["ptype"] == "image" else
              (r["prompt"] == H.probes[r["probe_idx"]]["loc"] and r["image"] is None and r["fmt"] == text_fmt) for r in rows),
          "edit_stream_matches_items": len(edits) == H.T and all(
              e["image"] == it["image"] and e["subject"] == it["src"] and e["alt"] == it["alt"] and e["uid"] == it["uid"]
              for e, it in zip(edits, H.items)),
          "n_heldout_valid": len(ho_rows), "n_repair_valid": len(rp_rows), "n_stream": len(stream),
          "stream_all_repair_D1_valid": all(pool[c["probe_idx"]] == "repair" and c["is_complaint_D1"] and c["y0_valid"] for c in stream),
          "heldout_uids_disjoint_from_repair_uids": not ({r["uid"] for r in ho_rows} & {r["uid"] for r in rp_rows})}
    # W_0 against the checkpoint file on disk (not through the loaded model)
    try:
        from safetensors import safe_open
        from vendor_v7.vlm import MODEL_DIRS
        hit = None
        for fn in sorted(glob.glob(os.path.join(MODEL_DIRS[H.backbone], "*.safetensors"))):
            with safe_open(fn, framework="pt", device="cpu") as f:
                for k in f.keys():
                    if k.endswith(f"layers.{H.L}.mlp.down_proj.weight") and "visual" not in k and "vision" not in k:
                        t = f.get_tensor(k)
                        if tuple(t.shape) == (H.m_out, H.d_in):
                            hit = (k, bool(torch.equal(t.to(H.dtype), H.W0.cpu())))
        xc["W0_equals_checkpoint_file"] = hit
    except Exception as e:      # noqa: BLE001
        xc["W0_equals_checkpoint_file"] = f"not checked: {e!r}"
    xc["WE_differs_from_W0"] = int((H.WE != H.W0).sum())
    xc["fro_log"] = float(H.D_act.norm())
    say(f"cross-checks {xc}")
    VC.jdump({"log": log, "cross_checks": xc, "harness_checks": H.checks, "draw": [list(d) for d in draw]},
             os.path.join(odir, "_xcheck.json"))
    assert xc["split_equals_harness"] and xc["stream_equals_experiment_setup"] and xc["record_fields_match_probes"]

    # ------------------------------------------------------------------ my batches (CPU), built once
    KEYS = ("last_prompt_idx", "answer_start", "answer_len", "answer_ids", "answer_mask", "pred_pos", "labels", "seq_len",
            "n_image_tokens")

    def cpu(b):
        o = dict(b)
        o["inputs"] = {k: v.cpu() for k, v in b["inputs"].items()}
        for k in KEYS:
            o[k] = b[k].cpu()
        return o

    def gpu(b):
        o = dict(b)
        o["inputs"] = {k: v.to(dev) for k, v in b["inputs"].items()}
        for k in KEYS:
            o[k] = b[k].to(dev)
        return o

    def row_batch(r):
        return cpu(vlm.build_batch([r["image"]] if r["ptype"] == "image" else [None], [r["prompt"]],
                                   answers=[r["y0_text"]], fmt=r["fmt"]))

    t = time.time()
    ho_b = [row_batch(r) for r in ho_rows]
    rp_b = {r["group"]: row_batch(r) for r in rp_rows}
    ed_b = [cpu(vlm.build_batch([e["image"]], [e["subject"]], answers=[e["alt"]], fmt="vqa")) for e in edits]
    say(f"built {len(ho_b)} held-out + {len(rp_b)} repair + {len(ed_b)} edit batches in {time.time() - t:.0f}s")

    @torch.no_grad()
    def tf_ok(b):
        b = gpu(b)
        lg = vlm.forward(b, logits_to_keep="answer").logits
        n = int(b["answer_mask"][0].sum())
        lg = lg[0, :n]
        if n == 0 or not bool(torch.isfinite(lg).all()):
            return False
        return bool((lg.argmax(-1) == b["answer_ids"][0, :n]).all())

    @contextlib.contextmanager
    def my_apply(corr, rowwise=False):
        """rowwise (LOC only, batch-8 forwards): the fp64 side path is computed one batch row at a time (the same
        operations per row; fp64 temporaries 8x smaller, so the batch-8 path fits under my memory cap)."""
        mod = H.mod
        assert torch.equal(mod.weight.data, H.WE), "module not at W_E"
        if corr.kind == "none":
            yield
        elif corr.kind == "weight":
            mod.weight.data.copy_(corr.W.to(mod.weight.device))
            try:
                yield
            finally:
                mod.weight.data.copy_(H.WE)
        elif corr.kind == "adapter":
            Lf, Rf, gt = corr.Lf.to(dev), corr.Rf.to(dev), corr.gate

            def side(x, out):
                x = x.to(torch.float64)
                z = x @ Rf
                if gt is not None:
                    z = z.masked_fill(z.abs() < gt * x.norm(dim=-1, keepdim=True), 0.0)
                return (out.to(torch.float64) + z @ Lf.T).to(out.dtype)

            def hook(m, args, out):
                if not rowwise:
                    return side(args[0], out)
                res = torch.empty_like(out)
                for i_ in range(out.shape[0]):
                    res[i_] = side(args[0][i_], out[i_])
                return res
            h = mod.register_forward_hook(hook)
            try:
                yield
            finally:
                h.remove()
        else:
            raise ValueError(corr.kind)
        assert torch.equal(mod.weight.data, H.WE)

    def my_eval(corr, reported=None, repair_pool=False):
        t = time.time()
        out = {}
        with my_apply(corr):
            out["heldout"] = [tf_ok(b) for b in ho_b]
            out["rel_em"] = [tf_ok(b) for b in ed_b]
            if repair_pool:
                out["repair"] = [tf_ok(rp_b[r["group"]]) for r in rp_rows]
            if reported is not None:
                out["reported"] = [tf_ok(rp_b[c["group"]]) for c in reported]
        out["eval_sec"] = time.time() - t
        return out

    def exp_record(name, sub=""):
        p = os.path.join(VC.P12, "out", log, sub, name + ".json")
        return VC.jload(p) if os.path.exists(p) else None

    def compare(mine, rec, reported=None):
        """Row-level agreement of my flags with an experiment record."""
        if rec is None:
            return {"experiment_record": None}
        hr = rec["eval"]["heldout_rows"]
        assert [x["group"] for x in hr] == [r["group"] for r in ho_rows], "held-out rows differ"
        ea = [bool(x["reproduced_after"]) for x in hr]
        em = [bool(x) for x in rec["eval"]["edits"]["rel"]["em_list"]]
        c = {"experiment_record": rec["config"], "experiment_time": rec.get("time"),
             "heldout_D1_exp": 100.0 * sum(not x for x in ea) / len(ea), "heldout_n_D1_exp": int(sum(not x for x in ea)),
             "heldout_rows_agree": int(sum(x == y for x, y in zip(mine["heldout"], ea))), "heldout_rows": len(ea),
             "heldout_rows_differ": [r["group"] for r, x, y in zip(ho_rows, mine["heldout"], ea) if x != y],
             "rel_exact_exp": 100.0 * sum(em) / len(em), "rel_em_agree": int(sum(x == y for x, y in zip(mine["rel_em"], em))),
             "rel_em_differ": [i for i, (x, y) in enumerate(zip(mine["rel_em"], em)) if x != y],
             "fro_rel_to_log_exp": rec["eval"]["fro_rel_to_log"]}
        if reported is not None:
            er = {f"{x['ptype']}_{x['probe_idx']}": bool(x["D1_resolved"]) for x in rec["eval"]["complaints"]}
            c["reported_same_set"] = [c0["group"] for c0 in reported] == rec.get("reported_groups")
            c["reported_resolved_exp"] = int(sum(er.values()))
            c["reported_agree"] = int(sum(er.get(c0["group"]) == y for c0, y in zip(reported, mine["reported"])))
        return c

    def summarise(mine, reported=None):
        h = mine["heldout"]
        we = [bool(r["reproduced"]) for r in ho_rows]              # stored W_E flags (probe_records_v2)
        s = {"heldout_n": len(h), "heldout_n_D1": int(sum(not x for x in h)), "heldout_D1": 100.0 * sum(not x for x in h) / len(h),
             "heldout_complaints_fixed": int(sum(x and not w for x, w in zip(h, we))), "heldout_n_complaints_stored": int(sum(not w for w in we)),
             "heldout_noncomplaints_lost": int(sum(w and not x for x, w in zip(h, we))), "heldout_n_noncomplaints_stored": int(sum(we)),
             "rel_exact": 100.0 * sum(mine["rel_em"]) / len(mine["rel_em"]), "rel_n_exact": int(sum(mine["rel_em"])),
             "eval_sec": mine["eval_sec"]}
        if "repair" in mine:
            s["repair_n"] = len(mine["repair"])
            s["repair_n_D1"] = int(sum(not x for x in mine["repair"]))
        if reported is not None:
            s["reported_n"] = len(reported)
            s["reported_resolved"] = int(sum(mine["reported"]))
        return s

    def corr_compare(corr, path):
        """My rebuilt correction against the one the experiment saved (CPU, fp64)."""
        if not os.path.exists(path):
            return {"saved_correction": None}
        b = torch.load(path, map_location="cpu")
        o = {"saved_correction": os.path.relpath(path, VC.P12), "kind_saved": b["kind"], "kind_mine": corr.kind}
        if b["kind"] == "adapter" and corr.kind == "adapter":
            Xa = corr.Lf.cpu() @ corr.Rf.cpu().T
            Xb = b["Lf"] @ b["Rf"].T
            o.update(rel_fro_diff=float((Xa - Xb).norm() / Xb.norm()), fro_mine=float(Xa.norm()), fro_saved=float(Xb.norm()),
                     rank_mine=int(corr.Rf.shape[1]), rank_saved=int(b["Rf"].shape[1]), gate_mine=corr.gate, gate_saved=b["gate"])
        elif b["kind"] == "weight" and corr.kind == "weight":
            Wa, Wb, We = corr.W.cpu().double(), b["W"].double(), H.WE.cpu().double()
            o.update(entries_differing=int((corr.W.cpu() != b["W"]).sum()), n_entries=int(Wb.numel()),
                     rel_fro_diff=float((Wa - Wb).norm() / max(float((Wb - We).norm()), 1e-30)),
                     fro_mine=float((Wa - We).norm()), fro_saved=float((Wb - We).norm()))
        return o

    def finish(name, body):
        body.update(log=log, cond=name, peak_gpu_gib=torch.cuda.max_memory_allocated() / 2 ** 30,
                    time=time.strftime("%Y-%m-%d %H:%M:%S"))
        VC.jdump(body, os.path.join(odir, name + ".json"))
        s = body["mine"]
        cmpd = body.get("vs_experiment", {})
        say(f"  {name}: mine HELD-OUT D1 {s['heldout_D1']:.1f}% ({s['heldout_n_D1']}/{s['heldout_n']}; +{s['heldout_complaints_fixed']}/-{s['heldout_noncomplaints_lost']}) "
            f"Rel exact {s['rel_exact']:.0f} | experiment {cmpd.get('heldout_D1_exp')} / {cmpd.get('rel_exact_exp')} | rows agree "
            f"{cmpd.get('heldout_rows_agree')}/{cmpd.get('heldout_rows')} rel {cmpd.get('rel_em_agree')}/100 | eval {s['eval_sec']:.0f}s "
            f"peak {body['peak_gpu_gib']:.2f} GiB")
        why = VC.blocked_by_me(g)
        if why:
            say(f"  yielding the card: another agent's waiting launcher is blocked by this process: {why}")
            sys.exit(EXIT_RETRY)

    fro_log = float(H.D_act.norm())

    def cost_small(c):
        return {k: c[k] for k in ("passes", "sec", "n_backward", "n_forward", "n_complaints", "correction_fro", "cfg") if k in c}

    # ------------------------------------------------------------------ targets (z*) of mine, per complaint
    tpath = os.path.join(mycache, f"targets_{log}.pt")
    tmem = torch.load(tpath, map_location="cpu", weights_only=False) if os.path.exists(tpath) else {}

    def my_targets(cs):
        new = 0
        for c in cs:
            if c["group"] in tmem:
                continue
            c1, k1 = repair("REEDIT", H, [c], variant="k0+kp", return_targets=True, chunk=1)
            t1 = c1.meta.pop("_targets")[0]
            al = c1.meta.get("attempt_log", [])
            tmem[c["group"]] = {"blob": {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in t1.to_blob().items()},
                                "oracle": bool(t1.oracle), "attempts": int(c1.meta["zopt_attempts"][0]),
                                "vendor_attempt_oracle": bool(al[0]["oracle"][0]) if al else None,
                                "attempt_log": [{k: v for k, v in x.items()} for x in al],
                                "single_installed": c1.kind == "weight", "l2": c1.meta.get("l2"),
                                "passes_zopt": k1["passes_zopt"], "sec_zopt": k1["sec"]["zopt"]}
            new += 1
            if new % 25 == 0:
                torch.save(tmem, tpath + ".tmp")
                os.replace(tpath + ".tmp", tpath)
                say(f"    z* targets: {new} new")
        if new:
            torch.save(tmem, tpath + ".tmp")
            os.replace(tpath + ".tmp", tpath)
        return [tmem[c["group"]] for c in cs]

    def as_target(e, oracle=None):
        t1 = VZ.Target(**{k: v for k, v in e["blob"].items() if k in VZ.Target.__dataclass_fields__})
        return t1 if oracle is None else dataclasses.replace(t1, oracle=bool(oracle))

    # ------------------------------------------------------------------ conditions
    def run_cond(name):
        torch.cuda.reset_peak_memory_stats()
        say(f"-- {name}")
        if name == "NONE":
            mine = my_eval(Correction("none", meta={}), repair_pool=True)
            st_h = [bool(r["reproduced"]) for r in ho_rows]
            st_r = [bool(r["reproduced"]) for r in rp_rows]
            rec = exp_record("NONE")
            body = {"mine": summarise(mine), "flags": mine,
                    "sanity": {"heldout_flags_equal_stored_v2": mine["heldout"] == st_h,
                               "heldout_flags_differing_from_stored": int(sum(x != y for x, y in zip(mine["heldout"], st_h))),
                               "repair_flags_equal_stored_v2": mine["repair"] == st_r,
                               "repair_flags_differing_from_stored": int(sum(x != y for x, y in zip(mine["repair"], st_r))),
                               "stored_heldout_D1": int(sum(not x for x in st_h)), "stored_repair_D1": int(sum(not x for x in st_r)),
                               "stored_Rel_exact_metrics_json": H.tl.metrics["Rel_exact"]},
                    "vs_experiment": compare(mine, rec)}
            if rec is not None:
                er = {f"{x['ptype']}_{x['probe_idx']}": bool(x["D1_resolved"]) for x in rec["eval"]["complaints"]}
                body["vs_experiment"]["repair_rows_agree"] = int(sum(er.get(r["group"]) == y for r, y in zip(rp_rows, mine["repair"])))
                body["vs_experiment"]["repair_rows"] = len(rp_rows)
            finish(name, body)
        elif name == "W0":
            corr, cost = repair("ROLLBACK_ALL", H, [])
            mine = my_eval(corr, repair_pool=True)
            rec = exp_record("ALL")
            body = {"mine": summarise(mine), "flags": mine,
                    "sanity": {"weight_equals_W0": bool(torch.equal(corr.W, H.W0)),
                               "heldout_D1_at_W0": int(sum(not x for x in mine["heldout"])),
                               "repair_D1_at_W0": int(sum(not x for x in mine["repair"]))},
                    "vs_experiment": compare(mine, rec)}
            finish(name, body)
        elif name.startswith("BEST__COMP_B") or name.startswith("COMP_B__"):
            n = name.split("__n")[1]
            cs = stream if n == "ALL" else stream[:int(n)]
            t = time.time()
            corr, cost = repair("COMP", H, cs, protect=("K_ans",), query="answer", form="adapter")
            wall = time.time() - t
            mine = my_eval(corr, reported=cs)
            ename = f"COMP_B__n{n}__s0"
            body = {"mine": summarise(mine, cs), "flags": mine, "experiment_config": ename,
                    "repair": {"call": 'repair("COMP", H, stream[:n], protect=("K_ans",), query="answer", form="adapter")',
                               "wall_sec": wall, "cost": cost_small(cost), "fro_rel_to_log": corr.fro(H) / fro_log,
                               "meta": {k: v for k, v in corr.meta.items() if not k.startswith("_")}},
                    "vs_experiment": compare(mine, exp_record(ename), cs),
                    "vs_saved_correction": corr_compare(corr, os.path.join(VC.P12, "out", log, "corr", ename + ".pt"))}
            del corr
            finish(name, body)
        elif name.startswith("RB_TOT__"):
            n = name.split("__n")[1]
            cs = stream if n == "ALL" else stream[:int(n)]
            t = time.time()
            corr, cost = repair("ROLLBACK_FO", H, cs, budget=30, offload=True, record_flags=True)
            wall = time.time() - t
            mine = my_eval(corr, reported=cs)
            ename = f"RB_TOT__n{n}__s0"
            rec = exp_record(ename)
            body = {"mine": summarise(mine, cs), "flags": mine, "experiment_config": ename,
                    "repair": {"call": 'repair("ROLLBACK_FO", H, stream[:n], budget=30, offload=True, record_flags=True)',
                               "wall_sec": wall, "cost": cost_small(cost), "fro_rel_to_log": corr.fro(H) / fro_log,
                               "R": [int(i) for i in corr.meta["R"]], "k": int(corr.meta["k"]),
                               "resolved_internal": bool(corr.meta["resolved_internal"]),
                               "R_equals_experiment": None if rec is None else [int(i) for i in corr.meta["R"]] == rec["extra"]["R"],
                               "R_same_set_as_experiment": None if rec is None else set(int(i) for i in corr.meta["R"]) == set(rec["extra"]["R"])},
                    "vs_experiment": compare(mine, rec, cs),
                    "vs_saved_correction": corr_compare(corr, os.path.join(VC.P12, "out", log, "corr", ename + ".pt"))}
            del corr
            finish(name, body)
        elif name.startswith("REEDIT_B__"):
            # variants:  theirz  the experiment's cached z* targets + the installed set of its record: isolates the install
            #                    and the evaluation from the z* optimisation (expected: the saved weight, bit for bit)
            #            asrec   MY z* targets + the installed set of the experiment's first-pass record
            #            ladder  MY z* targets, stated protocol (oracle fallback ladder; every oracle installed)
            parts = name.split("__")
            n = parts[1][1:]
            variant = parts[2] if len(parts) > 2 else "ladder"
            cs = stream if n == "ALL" else stream[:int(n)]
            ename = f"REEDIT_B__n{n}__s0"
            sub, rec = "", exp_record(ename)
            if rec is None or variant == "asrec":
                sub = "superseded_zfallback"
                rec = exp_record(ename, sub)
            t = time.time()
            drop = None
            if variant == "theirz":
                their = torch.load(os.path.join(VC.P12, "cache", f"targets_{log}.pt"), map_location="cpu", weights_only=False)
                drop = set(rec["extra"]["not_installed"])
                ents = [their[c["group"]] for c in cs]
                tg = [as_target(e, oracle=(c["group"] not in drop)) for e, c in zip(ents, cs)]
                zinfo = {"targets": "experiment cache (cache/targets_<log>.pt), read-only",
                         "cache_oracle_flags_false": [c["group"] for e, c in zip(ents, cs) if not e["oracle"]]}
                del their
            else:
                ents = my_targets(cs)
                if variant == "asrec":
                    drop = set(rec["extra"]["not_installed"])
                    tg = [as_target(e, oracle=(e["oracle"] and c["group"] not in drop)) for e, c in zip(ents, cs)]
                else:
                    tg = [as_target(e) for e in ents]
                zinfo = {"targets": "mine (fresh z* per complaint, B=1, chunk=1)",
                         "my_z_oracle": int(sum(e["oracle"] for e in ents)),
                         "my_z_vendor_attempt_oracle": int(sum(bool(e["vendor_attempt_oracle"]) for e in ents)),
                         "my_z_vendor_attempt_not_oracle": [c["group"] for e, c in zip(ents, cs) if not e["vendor_attempt_oracle"]],
                         "my_z_attempts": [e["attempts"] for e in ents]}
            t_z = time.time() - t
            corr, cost = repair("REEDIT", H, cs, variant="k0+kp", targets=tg, chunk=1)
            wall = time.time() - t
            mine = my_eval(corr, reported=cs)
            m = corr.meta
            body = {"mine": summarise(mine, cs), "flags": mine, "experiment_config": ename, "variant": variant,
                    "experiment_record_is_first_pass": bool(sub),
                    "repair": {"call": 'z*: repair("REEDIT", H, [c], variant="k0+kp", return_targets=True, chunk=1) per complaint; '
                                       'install: repair("REEDIT", H, stream[:n], variant="k0+kp", targets=z*, chunk=1)',
                               "wall_sec": wall, "zopt_wall_sec": t_z, "cost": cost_small(cost),
                               "fro_rel_to_log": corr.fro(H) / fro_log if corr.kind != "none" else 0.0,
                               "l2": m.get("l2"), "variant": m.get("variant"), "installed": len(m.get("installed_rows", [])),
                               "not_installed": [cs[j]["group"] for j in m.get("not_installed_rows", [])],
                               "forced_not_installed": None if drop is None else sorted(drop),
                               "solve_attainment": m.get("solve_attainment"), "dW_over_W0": m.get("dW_over_W0"), **zinfo},
                    "vs_experiment": compare(mine, rec, cs),
                    "vs_saved_correction": corr_compare(corr, os.path.join(VC.P12, "out", log, sub if sub else "corr", ename + ".pt"))}
            del corr
            finish(name, body)
        elif name == "ZREPEAT":
            # z* of the same complaint computed three times in THIS process with the experiment's call
            # (repair("REEDIT", H, [c], variant="k0+kp", return_targets=True, chunk=1)): first 6 image and first 3 text
            # complaints of the stream.  Compared among themselves, with my earlier run (another process) and with the
            # experiment's cached target.
            sel = [c for c in stream[:100] if c["ptype"] == "image"][:6] + [c for c in stream[:100] if c["ptype"] == "text"][:3]
            their = torch.load(os.path.join(VC.P12, "cache", f"targets_{log}.pt"), map_location="cpu", weights_only=False)
            res = []
            for c in sel:
                runs = []
                for _k in range(3):
                    c1, _c = repair("REEDIT", H, [c], variant="k0+kp", return_targets=True, chunk=1)
                    t1 = c1.meta.pop("_targets")[0]
                    al = c1.meta.get("attempt_log", [])
                    runs.append({"delta": t1.delta.double().cpu(), "h": t1.h.cpu(), "oracle": bool(t1.oracle), "steps": int(t1.steps_used),
                                 "attempts": int(c1.meta["zopt_attempts"][0]), "vendor_oracle": bool(al[0]["oracle"][0]) if al else None})
                    del c1, t1

                def rel(a_, b_):
                    return float((a_ - b_).norm() / b_.norm())
                pair = [rel(runs[i]["delta"], runs[j]["delta"]) for i in range(3) for j in range(i + 1, 3)]
                e = {"group": c["group"], "ptype": c["ptype"], "h_bitwise_equal_across_runs": bool(all(torch.equal(runs[0]["h"], r_["h"]) for r_ in runs)),
                     "own_runs_bitwise_equal": bool(all(torch.equal(runs[0]["delta"], r_["delta"]) for r_ in runs)),
                     "own_runs_max_rel_diff": max(pair), "steps": [r_["steps"] for r_ in runs], "oracle": [r_["oracle"] for r_ in runs],
                     "attempts": [r_["attempts"] for r_ in runs], "vendor_attempt_oracle": [r_["vendor_oracle"] for r_ in runs]}
                if c["group"] in their:
                    b_ = their[c["group"]]
                    e["vs_experiment_rel_diff"] = [rel(r_["delta"], b_["blob"]["delta"].double()) for r_ in runs]
                    e["h_equal_experiment"] = bool(torch.equal(runs[0]["h"], b_["blob"]["h"]))
                if c["group"] in tmem:
                    e["vs_my_earlier_process_rel_diff"] = [rel(r_["delta"], tmem[c["group"]]["blob"]["delta"].double()) for r_ in runs]
                res.append(e)
            del their
            body = {"log": log, "cond": name, "rows": res, "card": g, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
            VC.jdump(body, os.path.join(odir, name + ".json"))
            say(f"  ZREPEAT: " + "; ".join(f"{e['group']} equal {e['own_runs_bitwise_equal']} max {e['own_runs_max_rel_diff']:.2e}" for e in res))
        elif name == "LOC":
            # T-Loc / I-Loc of the HELD-OUT items (vendor definition: argmax agreement with the W_0 predictions over the
            # reference-answer positions, vendor batch-8 forwards), at W_E and under the rebuilt best-point correction.
            # Probes are built here from the raw VLKEB item fields; the comparison is my own code.
            from vendor_v7 import metrics as VMM
            ho_items = [pr for i, pr in enumerate(H.probes) if pool[i] == "heldout"]
            pr_ = {"tloc": [{"item": i, "uid": d["uid"], "metric": "tloc", "prompt": d["loc"], "answer": d["loc_ans"], "image": None, "fmt": "plain"}
                            for i, d in enumerate(ho_items)],
                   "iloc": [{"item": i, "uid": d["uid"], "metric": "iloc", "prompt": d["m_loc_q"], "answer": d["m_loc_a"], "image": d["m_loc"], "fmt": "vqa"}
                            for i, d in enumerate(ho_items)]}

            def preds(corr_):
                with my_apply(corr_, rowwise=True):
                    return {mt: [(r["pred"], r["finite"]) for r in VMM.run_probes(vlm, pr_[mt], batch_size=8)] for mt in ("tloc", "iloc")}

            def agree(a_, b_):
                o = []
                for (pa, fa), (pb, fb) in zip(a_, b_):
                    n_ = min(len(pa), len(pb))
                    o.append(float((pa[:n_] == pb[:n_]).mean()) if (fb and n_) else 0.0)
                return o
            p0 = preds(Correction("weight", W=H.W0.clone(), meta={}))
            pe = preds(Correction("none", meta={}))
            cb, _c = repair("COMP", H, stream, protect=("K_ans",), query="answer", form="adapter")
            pb = preds(cb)
            del cb
            res = {mt: {"WE": 100.0 * float(np.mean(agree(p0[mt], pe[mt]))), "best": 100.0 * float(np.mean(agree(p0[mt], pb[mt]))),
                        "W0_self": 100.0 * float(np.mean(agree(p0[mt], p0[mt]))), "n": len(pr_[mt])} for mt in ("tloc", "iloc")}
            rec = exp_record("COMP_B__nALL__s0")["eval"]["loc"]
            body = {"log": log, "cond": name, "mine": res,
                    "experiment": {mt: {"WE": rec[mt]["WE"], "best": rec[mt]["after"]} for mt in ("tloc", "iloc")},
                    "agree": bool(all(abs(res[mt][k] - {"WE": rec[mt]["WE"], "best": rec[mt]["after"]}[k]) < 1e-6 for mt in ("tloc", "iloc") for k in ("WE", "best"))),
                    "n_heldout_items": len(ho_items), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
            VC.jdump(body, os.path.join(odir, name + ".json"))
            say(f"  LOC: {body}")
        elif name == "WPROJ":
            # VERIFIER'S EXTRA (not part of the experiment's protocol; one run): the report explains the lack of
            # generalisation at the module level -- "restoring W_0 everywhere outside the edits' answer-key span would
            # still leave 0.87 of the log's module-output change" -- without a behavioural evaluation.  This evaluates
            # exactly that weight, W_proj = W_0 + D P (D = W_E - W_0, P = projector onto span(K_ans)), with my loop.
            ek = H.edit_keys()
            Ka = torch.cat([ek["K"][i][ek["pp"][i]] for i in range(H.T)]).double().T.contiguous()
            Uq, Sq, _ = torch.linalg.svd(Ka, full_matrices=False)
            rk = int((Sq > max(Ka.shape) * float(np.finfo(np.float64).eps) * float(Sq[0])).sum())
            Qb = Uq[:, :rk].contiguous()
            D = H.WE.cpu().double() - H.W0.cpu().double()
            Wp64 = H.W0.cpu().double() + (D @ Qb) @ Qb.T
            Wp = Wp64.to(H.dtype)
            info = {"rank_K_ans": rk, "norm_Wproj_minus_W0_over_log": float((Wp.double() - H.W0.cpu().double()).norm() / D.norm()),
                    "norm_Wproj_minus_WE_over_log": float((Wp.double() - H.WE.cpu().double()).norm() / D.norm()),
                    "module_output_on_edit_answer_keys_rel_err": float(((Wp.double() - H.WE.cpu().double()) @ Ka).norm() / (D @ Ka).norm())}
            del D, Wp64, Uq, Qb
            corr = Correction("weight", W=Wp.to(dev), meta={})
            mine = my_eval(corr, repair_pool=True)
            sm = summarise(mine)
            st_r = [bool(r["reproduced"]) for r in rp_rows]
            sm["repair_complaints_fixed"] = int(sum(x and not w for x, w in zip(mine["repair"], st_r)))
            sm["repair_noncomplaints_lost"] = int(sum(w and not x for x, w in zip(mine["repair"], st_r)))
            body = {"mine": sm, "flags": mine, "what": "W_proj = W_0 + (W_E - W_0) P_ans: the log restricted to the span of the edits' answer keys "
                                                       "(verifier's extra diagnostic, no experiment record)", "weight": info, "vs_experiment": {"experiment_record": None}}
            del corr
            finish(name, body)
        elif name == "RELB8":
            # stored Rel of the log (metrics.json) was measured on the vendor batch-8 path: repeat exactly that path at W_E
            from vendor_v7 import metrics as VMM
            assert torch.equal(H.mod.weight.data, H.WE)
            sm = {}
            for mt in ("rel", "tgen", "igen"):
                res = VMM.run_probes(vlm, VMM.build_probes(H.items, mt), batch_size=8)
                sm[mt] = {"token": float(np.mean([r["acc"] if r["finite"] else 0.0 for r in res]) * 100),
                          "exact": float(np.mean([r["em"] if r["finite"] else 0.0 for r in res]) * 100), "n": len(res)}
            st = H.tl.metrics
            body = {"log": log, "cond": name, "mine_b8": sm,
                    "stored": {"Rel_exact": st["Rel_exact"], "Rel_token": st["Rel_token"], "TGen_exact": st["TGen_exact"], "IGen_exact": st["IGen_exact"]},
                    "rel_matches_stored": bool(abs(sm["rel"]["exact"] - st["Rel_exact"]) < 1e-6 and abs(sm["rel"]["token"] - st["Rel_token"]) < 1e-6),
                    "tgen_igen_match_stored": bool(abs(sm["tgen"]["exact"] - st["TGen_exact"]) < 1e-6 and abs(sm["igen"]["exact"] - st["IGen_exact"]) < 1e-6),
                    "time": time.strftime("%Y-%m-%d %H:%M:%S")}
            VC.jdump(body, os.path.join(odir, name + ".json"))
            say(f"  RELB8: {body}")
        elif name == "PROTKEYS":
            from harness.common import sha as hsha
            hcache = os.path.join(VC.V3, "harness", "cache")
            p12cache = os.path.join(VC.P12, "cache")
            rows_p = HM._probe_rows(H, 100000)                     # what "probe:100000" selects
            byg = {r["group"]: r for r in rows}
            sel_ok = all(pool[r["probe_idx"]] == "repair" and r["y0_valid"] and r["reproduced"] is True for r in rows_p)
            mine_nonc = sorted([r["group"] for r in rp_rows if r["reproduced"] is True])
            Kp = HM._probe_keys(H, rows_p, "ans")                  # fresh forwards -> my cache (empty before)
            tag = hsha([H.name, [(r["probe_idx"], r["ptype"], r["y0_text"]) for r in rows_p], "ans"])
            fn = f"probekeys_{H.name}_{tag}.pt"
            hp = os.path.join(hcache, fn)
            hk = torch.load(hp) if os.path.exists(hp) else None
            # my own capture (forward hook on the edited module, truncated forward) for 12 random rows
            offs = np.cumsum([0] + [int(r["n_ans_tok"]) for r in rows_p])
            pick = sorted(np.random.default_rng(7).choice(len(rows_p), 12, replace=False).tolist())
            own_equal = []
            for j in pick:
                b = gpu(rp_b[rows_p[j]["group"]])
                box = {}
                h = H.mod.register_forward_hook(lambda m, args, out: box.__setitem__("x", args[0].detach()))
                try:
                    with torch.no_grad():
                        vlm.forward(b, stop_after_layer=H.L)
                finally:
                    h.remove()
                A = int(b["answer_len"][0])
                own = box["x"][0][b["pred_pos"][0, :A]].cpu()
                own_equal.append(bool(A == int(rows_p[j]["n_ans_tok"]) and torch.equal(own, Kp[offs[j]:offs[j + 1]])))
            ek = H.edit_keys()                                     # mine (recomputed by the COMP rebuild, or now)
            he = [f for f in os.listdir(hcache) if f.startswith(f"editkeys_{H.backbone}_L{H.L}_")]
            ek_equal = None
            if he:
                hek = torch.load(os.path.join(hcache, he[0]))
                ek_equal = bool(len(hek["K"]) == len(ek["K"]) and all(torch.equal(x, y) for x, y in zip(hek["K"], ek["K"]))
                                and all(torch.equal(x, y) for x, y in zip(hek["pp"], ek["pp"])) and hek["last"] == ek["last"])
            Qa, sta = HM.protection_basis(H, ("K_ans",))
            Qp, stp = HM.protection_basis(H, ("K_ans", "probe:100000"))
            ra = VC.jload(os.path.join(VC.P12, "out", log, "COMP_B__nALL__s0.json"))["rank"]["protection"]
            rb = VC.jload(os.path.join(VC.P12, "out", log, "COMP_BP__nALL__s0.json"))["rank"]["protection"]
            body = {"log": log, "cond": name, "n_probe_rows": len(rows_p), "selection_repair_noncomplaints_only": bool(sel_ok),
                    "selection_equals_my_repair_noncomplaints": sorted(r["group"] for r in rows_p) == mine_nonc,
                    "n_heldout_rows_in_selection": int(sum(pool[r["probe_idx"]] == "heldout" for r in rows_p)),
                    "n_keys": int(Kp.shape[0]), "n_keys_expected": int(offs[-1]), "harness_cache_file": fn,
                    "harness_cache_file_exists": hk is not None,
                    "fresh_keys_equal_harness_cache": None if hk is None else bool(hk.shape == Kp.shape and torch.equal(hk, Kp)),
                    "fresh_vs_cache_max_abs_diff": None if hk is None or hk.shape != Kp.shape else float((hk.float() - Kp.float()).abs().max()),
                    "p12_cache_entry": os.path.realpath(os.path.join(p12cache, fn)) if os.path.lexists(os.path.join(p12cache, fn)) else None,
                    "own_hook_capture_equal": own_equal, "edit_keys_fresh_equal_harness_cache": ek_equal,
                    "n_edit_answer_keys": int(sum(len(x) for x in ek["pp"])),
                    "rank_K_ans": int(sta["rank"]), "rank_K_ans_plus_probe": int(stp["rank"]),
                    "experiment_protection_rank_COMP_B": ra, "experiment_protection_rank_COMP_BP": rb,
                    "time": time.strftime("%Y-%m-%d %H:%M:%S")}
            VC.jdump(body, os.path.join(odir, name + ".json"))
            say(f"  PROTKEYS: {body}")
            del Qp, Kp
        elif name == "MODDIAG":
            # the report: "the edits' answer-key span has rank 483 and holds 43% of a held-out complaint's answer-key
            # norm; restoring W_0 everywhere outside it would still leave 0.87 of the log's module-output change"
            ek = H.edit_keys()
            Ka = torch.cat([ek["K"][i][ek["pp"][i]] for i in range(H.T)]).double().T.contiguous()      # (d, n_ans) CPU
            Uq, Sq, _ = torch.linalg.svd(Ka, full_matrices=False)
            rk = int((Sq > max(Ka.shape) * float(np.finfo(np.float64).eps) * float(Sq[0])).sum())
            Qb = Uq[:, :rk].contiguous()
            D = (H.WE.double() - H.W0.double()).cpu()
            DQ = D @ Qb
            res = {"group": [], "ptype": [], "stored_D1": [], "s": [], "a": [], "b": [], "n_keys": []}
            for r, b0 in zip(ho_rows, ho_b):
                b = gpu(b0)
                box = {}
                h = H.mod.register_forward_hook(lambda m, args, out: box.__setitem__("x", args[0].detach()))
                try:
                    with torch.no_grad():
                        vlm.forward(b, stop_after_layer=H.L)
                finally:
                    h.remove()
                A = int(b["answer_len"][0])
                K = box["x"][0][b["pred_pos"][0, :A]].double().cpu().T          # (d, A)
                c = Qb.T @ K
                DK = D @ K
                DPK = DQ @ c
                res["group"].append(r["group"])
                res["ptype"].append(r["ptype"])
                res["stored_D1"].append(not bool(r["reproduced"]))
                res["s"].append(float(c.norm() / K.norm()))
                res["a"].append(float(DPK.norm() / DK.norm()))
                res["b"].append(float((DK - DPK).norm() / DK.norm()))
                res["n_keys"].append(A)
            body = {"log": log, "cond": name, "rank_K_ans": rk, "d_in": H.d_in, "n_edit_answer_keys": int(Ka.shape[1]),
                    "fro_D": float(D.norm()), "fro_D_on_K_ans_span": float(DQ.norm()), "rows": res,
                    "time": time.strftime("%Y-%m-%d %H:%M:%S")}
            VC.jdump(body, os.path.join(odir, name + ".json"))
            say(f"  MODDIAG: rank {rk}, {len(res['group'])} held-out rows")
            del D, DQ, Qb, Ka
        else:
            raise ValueError(name)

    import gc
    for name in todo:
        try:
            run_cond(name)
        except Exception as e:      # noqa: BLE001
            msg = str(e)
            oom = isinstance(e, torch.OutOfMemoryError) or "out of memory" in msg.lower() or "ALLOC_FAILED" in msg
            if not oom:
                raise
            H.mod.weight.data.copy_(H.WE)
            gc.collect()
            torch.cuda.empty_cache()
            free = VC.free_mib(g)
            own = "allowed;" in msg and free >= 1500
            say(f"  {name}: CUDA out of memory ({'own cap' if own else 'card full'}; {free} MiB free on the card): {msg[:260]}")
            if not own:
                os._exit(EXIT_RETRY)                   # the card filled up: leave, the queue starts me again
            VC.jdump({"log": log, "cond": name, "error": "CUDA OOM under my own cap", "cap_gb": a.cap_gb, "detail": msg[:400],
                      "time": time.strftime("%Y-%m-%d %H:%M:%S")}, os.path.join(odir, name + ".owncap_oom.txt.json"))
    say(f"gpu_verify finished {log}: {len(todo)} conditions in {time.time() - T0:.0f}s")


if __name__ == "__main__":
    main()
