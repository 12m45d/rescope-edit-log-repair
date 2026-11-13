"""GPU tests of e10t on the DEVELOPMENT log only (TX-G2XL-rome-dev100; no other log is opened).

    $GQ run --mem 12000 --prio -8 --name e10t_test -- $PY tests/t_gpu.py [names]      -> tests/out/gpu_<name>.json

    linear   identity (a): all thresholds -inf, linear response, logged values, all positions = the deployed weight up to
             round-off, on 200 prompts (full logits of padded batches; every real position fires, no PAD position does)
    silent   identity (b): a prompt on which no edit fires is served bit for bit as by the pre-edit model (teacher-forced
             logits and decode logits; mixed batches in which other rows fire; thresholds +inf: every row)
    target   identity (c): at its solver key every edit attains its module-output target with rebased values (served module
             output at the solver position of the request prompt against the logged target; contrast: logged values, deployed)
    null0    identity (d): the null-repair variant with n_A = 0 serves the same logits as the default method, bit for bit
    batch    a batched (padded, masked) pass equals the single pass: tokens, firing flags; logits to round-off
    hook     the hook's firing in real passes (teacher-forced and decode) = the gate-level decision from the key cache, on
             every prompt of the development log; the sparse evaluation = the full evaluation, row by row
    cache    cached keys = the keys the hook sees in a live teacher-forced pass and in a live decode (to round-off, in the band)
    public   the public evaluation code on the served model: deployed / pre-edit reproduce the build's case files and
             generations; a gate with all thresholds -inf reproduces the deployed scores; PAD rows of the public batch are masked
    store    a second pass through the on-disk store runs the model on nothing and returns the same rows
    guard    capped edits fire on their own request as served (live padded passes), with and without the margin
    usage    the examples of USAGE.txt (one per metric) run as written
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "code"))
import e10t  # noqa: E402

B = e10t.Backbone()
from e10t import backbone as BB  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
INF = float("inf")
L = e10t.open_log(e10t.DEV_LOG)
assert L.development
W0, WE = L.weights()
dev = B.dev


def sample(rows, n, seed=0):
    idx = np.sort(np.random.default_rng(seed).choice(len(rows), min(n, len(rows)), replace=False))
    return [rows[int(i)] for i in idx]


def mixed_batches(tids, bs=16, seed=0):
    """Batches of MIXED lengths (so that padding and masks are exercised): a seeded shuffle cut into chunks."""
    perm = np.random.default_rng(seed).permutation(len(tids))
    return [[tids[int(i)] for i in perm[a:a + bs]] for a in range(0, len(tids), bs)]


@torch.no_grad()
def full_logits(seqs):
    """Logits at every position of a padded batch (the model as served now) -> (logits (B, Tn, V), lens)."""
    ids, mask, lens = B._pad(seqs)
    lg = B.model(input_ids=ids, use_cache=False, **({} if mask is None else {"attention_mask": mask})).logits
    return lg, lens


def real(lg, lens):
    return torch.cat([lg[i, :int(n)] for i, n in enumerate(lens)])


def P200():
    rows = sample(L.benign("heldout"), 100, 1) + sample(L.rows["request"], 50, 2) + sample(L.rows["paraphrase"], 50, 3)
    return [r["text_id"] for r in rows]


def t_linear():
    M = e10t.methods.where(L, "none")
    tids = P200()
    mx, mx_pre, same_arg, n_pos, pad_fired, all_fire, n_pad = 0.0, [], 0, 0, 0, 0, 0
    dec_same = 0
    for bt in mixed_batches(tids):
        seqs = [L.ids(t) + L.y0(t) for t in bt]
        with B.served(WE):
            ld, lens = full_logits(seqs)
            dd, _ = B.decode([L.ids(t) for t in bt])
        with B.served(None):
            lp, _ = full_logits(seqs)
        with B.served(M.obj) as srv:
            lg, _ = full_logits(seqs)
            fl = srv.rows()
            dg, _ = B.decode([L.ids(t) for t in bt])
        a, d_, p = real(lg, lens), real(ld, lens), real(lp, lens)
        mx = max(mx, float((a - d_).abs().max()))
        mx_pre.append(float((d_ - p).abs().max()))
        same_arg += int((a.argmax(-1) == d_.argmax(-1)).sum())
        n_pos += a.shape[0]
        all_fire += sum(f["n_fire_pos"] == int(n) for f, n in zip(fl, lens))
        n_pad += int((lens.max() - lens).sum())
        dec_same += int((dg == dd).all(1).sum())
    d = {"n_prompts": len(tids), "n_positions": n_pos, "n_pad_positions_in_batches": n_pad, "max_abs_logit_diff_to_deployed": mx,
         "for_scale_median_max_abs_logit_diff_deployed_vs_pre_edit": float(np.median(mx_pre)), "n_positions_same_argmax": same_arg,
         "n_rows_firing_at_exactly_their_real_positions": all_fire, "n_decodes_equal_to_deployed": dec_same,
         "gate": {"role": M.obj.role, "fire": M.obj.fire, "response": M.obj.response, "thresholds": M.rec["thresholds"]}}
    ok = mx < 1e-3 and all_fire == len(tids) and same_arg >= n_pos - 2 and dec_same >= len(tids) - 1 and d["for_scale_median_max_abs_logit_diff_deployed_vs_pre_edit"] > 1.0 and n_pad > 0
    return ok, d, "identity (a): thresholds -inf, linear response, logged values, all positions = the deployed weight up to round-off (200 prompts, padded batches)"


def t_silent():
    M = e10t.methods.rescope(L, draw=0)
    rows = sample(L.benign("heldout"), 300, 4) + sample(L.rows["neighborhood"], 100, 5) + sample(L.rows["request"], 40, 6)
    tids = list(dict.fromkeys(r["text_id"] for r in rows))
    dec = e10t.gate.decide(L, M.obj, tids, with_positions=False)
    cache_silent = {t: bool(s) for t, s in zip(tids, dec["sure_silent"])}
    S_off = M.obj.replace(det_tau=torch.full_like(M.obj.det_tau, INF))
    st = {"n_rows": 0, "n_silent_hook": 0, "n_silent_bitwise_tf": 0, "n_silent_bitwise_dec": 0, "n_fired": 0, "n_fired_bitwise_tf": 0,
          "n_cache_silent": 0, "n_cache_silent_but_hook_fired": 0, "n_off_bitwise_tf": 0, "n_off_bitwise_dec": 0, "n_batches_mixed": 0}
    for bt in mixed_batches(tids):
        seqs, conts = [L.ids(t) for t in bt], [L.y0(t) for t in bt]
        full = [s + c for s, c in zip(seqs, conts)]
        with B.served(None):
            lp, lens = full_logits(full)
            _, _, dp = B.decode(seqs, return_logits=True)
        with B.served(M.obj) as srv:
            lg, _ = full_logits(full)
            f_tf = srv.rows()
            _, _, dg = B.decode(seqs, return_logits=True)
            f_dec = srv.rows()
        with B.served(S_off) as srv:
            lo, _ = full_logits(full)
            _, _, do = B.decode(seqs, return_logits=True)
        fired = [f["fired"] for f in f_tf]
        st["n_batches_mixed"] += int(any(fired) and not all(fired))
        for i, t in enumerate(bt):
            n = int(lens[i])
            eq_tf, eq_dec = bool(torch.equal(lg[i, :n], lp[i, :n])), bool(torch.equal(dg[i], dp[i]))
            st["n_rows"] += 1
            st["n_cache_silent"] += int(cache_silent[t])
            st["n_cache_silent_but_hook_fired"] += int(cache_silent[t] and (f_tf[i]["fired"] or f_dec[i]["fired"]))
            if not f_tf[i]["fired"]:
                st["n_silent_hook"] += 1
                st["n_silent_bitwise_tf"] += int(eq_tf)
                st["n_silent_bitwise_dec"] += int(eq_dec and not f_dec[i]["fired"])
            else:
                st["n_fired"] += 1
                st["n_fired_bitwise_tf"] += int(eq_tf)
            st["n_off_bitwise_tf"] += int(torch.equal(lo[i, :n], lp[i, :n]))
            st["n_off_bitwise_dec"] += int(torch.equal(do[i], dp[i]))
    ok = (st["n_silent_hook"] > 200 and st["n_silent_bitwise_tf"] == st["n_silent_hook"] == st["n_silent_bitwise_dec"] and st["n_fired"] >= 40
          and st["n_fired_bitwise_tf"] == 0 and st["n_cache_silent_but_hook_fired"] == 0 and st["n_off_bitwise_tf"] == st["n_rows"] == st["n_off_bitwise_dec"]
          and st["n_batches_mixed"] > 0)
    return ok, st, "identity (b): a prompt on which no edit fires is the pre-edit model bit for bit (teacher-forced and decode logits, mixed padded batches)"


def t_target():
    rq = L.rows["request"]
    out = {}
    ok = True
    for lab, M in (("rebased", e10t.methods.rescope(L, draw=0)), ("logged_values", e10t.methods.rescope(L, draw=0, rebase=False)), ("deployed", e10t.methods.deployed(L))):
        ses = e10t.behave.Session(L, B, store=False)
        errs, own_rel, other_at_p = [], 0, 0
        for b in B.batches([len(L.ids(r["text_id"])) for r in rq], extra=0):
            rb = [rq[i] for i in b]
            with B.served(ses.served_obj(M)) as srv:
                xs, os_ = B.keys([L.ids(r["text_id"]) for r in rb], output=True)
                fl = srv.rows()
            for i, r in enumerate(rb):
                p, e = L.solver_pos[r["pos"]], r["pos"]
                o = os_[i][p].double().cpu()
                t = L.target[e]
                errs.append(float((o - t).norm() / t.norm()))
                if fl is not None:
                    at = [ed for (q, ed) in fl[i]["rel"] if q == p]
                    own_rel += int(at == [e])
                    other_at_p += int(bool(at) and at != [e])
        errs = np.array(errs)
        out[lab] = {"n": len(errs), "max_rel_err": float(errs.max()), "median_rel_err": float(np.median(errs)), "n_within_1e-5": int((errs < 1e-5).sum()),
                    "own_edit_released_at_solver_position": own_rel if M.kind == "gate" else None, "other_edit_released_there": other_at_p if M.kind == "gate" else None}
    ok = (out["rebased"]["max_rel_err"] < 1e-5 and out["rebased"]["own_edit_released_at_solver_position"] == L.T and out["logged_values"]["median_rel_err"] > 1e-2
          and out["deployed"]["median_rel_err"] > 1e-3)
    return ok, out, "identity (c): at its solver key every edit attains its module-output target with rebased values (served module output vs logged target)"


def t_null0():
    A, Bm = e10t.methods.rescope(L, draw=0), e10t.methods.rescope_null(L, n_a=0, draw=0)
    C100 = e10t.methods.rescope_null(L, n_a=100, draw=0)
    tids = [r["text_id"] for r in sample(L.benign("heldout"), 60, 7) + sample(L.rows["request"], 20, 8) + sample(L.rows["paraphrase"], 20, 9)]
    n_eq = n_eq_dec = n = n_fired = n_c_diff = 0
    for bt in mixed_batches(tids):
        seqs = [L.ids(t) for t in bt]
        full = [s + L.y0(t) for s, t in zip(seqs, bt)]
        res = []
        for M in (A, Bm, C100):
            with B.served(M.obj) as srv:
                lg, lens = full_logits(full)
                fl = srv.rows()
                _, _, dg = B.decode(seqs, return_logits=True)
            res.append((lg, dg, fl))
        for i in range(len(bt)):
            k = int(lens[i])
            n += 1
            n_eq += int(torch.equal(res[0][0][i, :k], res[1][0][i, :k]) and res[0][2][i] == res[1][2][i])
            n_eq_dec += int(torch.equal(res[0][1][i], res[1][1][i]))
            n_fired += int(res[0][2][i]["fired"])
            n_c_diff += int(not torch.equal(res[0][0][i, :k], res[2][0][i, :k]))
    d = {"n_prompts": n, "n_fired_under_default": n_fired, "n_bitwise_equal_tf_logits_and_flags": n_eq, "n_bitwise_equal_decode_logits": n_eq_dec,
         "control_n_a_100_rows_with_other_logits": n_c_diff, "gate_sig_equal": A.rec["gate_sig"] == Bm.rec["gate_sig"]}
    ok = n_eq == n == n_eq_dec and n_fired >= 40 and n_c_diff > 0
    return ok, d, "identity (d): the null-repair variant with n_A = 0 serves the default method's logits bit for bit"


def t_batch():
    rows = sample(L.benign("heldout"), 60, 10) + sample(L.rows["request"], 30, 11) + sample(L.rows["paraphrase"], 30, 12) + sample(L.rows["neighborhood"], 30, 13)
    tids = list(dict.fromkeys(r["text_id"] for r in rows))
    out = {}
    ok = True
    for lab, obj in (("preedit", None), ("deployed", WE), ("rescope", e10t.methods.rescope(L, draw=0).obj), ("where_all", e10t.methods.where(L, "none").obj)):
        st = {"n": 0, "tf_tok_diff_rows": 0, "dec_tok_diff_rows": 0, "flag_diff_rows": 0, "max_abs_logit_diff_tf": 0.0, "max_abs_logit_diff_dec": 0.0,
              "tok_diff_min_top2_gap": None, "n_fired_rows": 0}
        with B.served(obj) as srv:
            single = {}
            for t in tids:
                p, f, lg = B.tf([L.ids(t)], [L.y0(t)], return_logits=True)
                ftf = srv.rows()
                dtok, _, dl = B.decode([L.ids(t)], return_logits=True)
                fd = srv.rows()
                single[t] = (p[0], lg[0], ftf[0] if ftf else None, dtok[0], dl[0], fd[0] if fd else None)
            for bt in mixed_batches(tids, bs=24, seed=3):
                seqs = [L.ids(t) for t in bt]
                p, f, lg = B.tf(seqs, [L.y0(t) for t in bt], return_logits=True)
                ftf = srv.rows()
                dtok, _, dl = B.decode(seqs, return_logits=True)
                fd = srv.rows()
                for i, t in enumerate(bt):
                    s = single[t]
                    st["n"] += 1
                    dtf, dd = not bool((p[i] == s[0]).all()), not bool((dtok[i] == s[3]).all())
                    st["tf_tok_diff_rows"] += int(dtf)
                    st["dec_tok_diff_rows"] += int(dd)
                    st["max_abs_logit_diff_tf"] = max(st["max_abs_logit_diff_tf"], float((lg[i] - s[1]).abs().max()))
                    if not dd:
                        st["max_abs_logit_diff_dec"] = max(st["max_abs_logit_diff_dec"], float((dl[i] - s[4]).abs().max()))
                    if dtf:
                        j = int((p[i] != s[0]).nonzero()[0])
                        top = s[1][j].topk(2).values
                        g = float(top[0] - top[1])
                        st["tok_diff_min_top2_gap"] = g if st["tok_diff_min_top2_gap"] is None else min(g, st["tok_diff_min_top2_gap"])
                    if ftf is not None:
                        st["flag_diff_rows"] += int(ftf[i] != s[2] or fd[i] != s[5])
                        st["n_fired_rows"] += int(ftf[i]["fired"])
        out[lab] = st
        ok &= st["flag_diff_rows"] == 0 and st["max_abs_logit_diff_tf"] < 5e-3 and st["max_abs_logit_diff_dec"] < 5e-3 and st["tf_tok_diff_rows"] == 0 and st["dec_tok_diff_rows"] == 0
    return ok, out, "a batched (right-padded, masked) pass equals the single pass: same tokens and firing flags; logits to float32 round-off"


def _cmp_rows(a, b, keys):
    return sum(any(x.get(k) != y.get(k) for k in keys) for x, y in zip(a, b))


def t_hook():
    out = {}
    ok = True
    rows = L.rows["request"] + L.rows["paraphrase"] + L.rows["neighborhood"] + L.benign("heldout")
    tids = [r["text_id"] for r in rows]
    own = [r["pos"] if r["set"] == "dev" else None for r in rows]
    for M in e10t.methods.rowset(L, draw=0, labels=("rescope", "cosine", "cosine_published", "where_twoblock")):
        t0 = time.time()
        full_s, sp_s = e10t.behave.Session(L, B, store=False), e10t.behave.Session(L, B, store=False)
        full = full_s.rows(M, rows, force_run=True)                      # every prompt through the hook, both passes
        sparse = sp_s.rows(M, rows)                                      # the sparse path
        d = e10t.gate.decide(L, M.obj, tids, own)
        mism = {"fired": 0, "n_fire_pos": 0, "rel_edits": 0, "n_cand_edits": 0, "own_released": 0}
        for i, r in enumerate(full):
            if d["near"][i]:
                continue
            mism["fired"] += int(r["fired"] != bool(d["fired"][i]))
            mism["n_fire_pos"] += int(r["n_fire_pos"] != int(d["n_fire_pos"][i]))
            if M.obj.fire == "top1":
                mism["rel_edits"] += int(r["rel_edits"] != BB.pack_edits({e for _, e in (d["rel"][i] or [])}))
            mism["n_cand_edits"] += int(r["n_cand_edits"] != int(d["n_cand_edits"][i]))
            if own[i] is not None:
                mism["own_released"] += int(bool(r["own_released"]) != bool(d["own_released"][i]))
        keys = ("tf_tok", "dec_tok", "D1_tf", "D1_dec", "tok_loc", "fired", "fired_dec", "n_fire_pos", "n_fire_pos_dec", "rel_edits", "rel_edits_dec",
                "n_cand_edits", "own_released", "own_released_dec", "other_released", "new_exact", "new_tok", "true_exact", "finite_tf", "finite_dec")
        o = {"n_rows": len(rows), "n_in_band": int(d["near"].sum()), "hook_tf_vs_cache_mismatch": mism,
             "hook_mismatch_counter_full": full_s.stats["hook_mismatch"] + full_s.stats["hook_mismatch_dec"],
             "hook_mismatch_counter_sparse": sp_s.stats["hook_mismatch"] + sp_s.stats["hook_mismatch_dec"], "sparse_vs_full_rows_differing": _cmp_rows(full, sparse, keys),
             "sequences_run_full": full_s.stats["sequences_run"], "sequences_run_sparse": sp_s.stats["sequences_run"], "silent_sparse": sp_s.stats["silent"],
             "n_rows_fired_tf": sum(bool(r["fired"]) for r in full), "n_rows_fired_dec": sum(bool(r["fired_dec"]) for r in full),
             "n_rows_fired_dec_but_not_tf": sum(bool(r["fired_dec"]) and not r["fired"] for r in full),
             "n_rows_fired_tf_but_not_dec": sum(bool(r["fired"]) and not r["fired_dec"] for r in full),
             "n_rows_D1_tf_differs_from_dec": sum(r["D1_tf"] != r["D1_dec"] for r in full),
             "seconds_full": full_s.stats["seconds_model"], "seconds_sparse": sp_s.stats["seconds_model"], "seconds": time.time() - t0}
        out[M.label] = o
        ok &= (sum(mism.values()) == 0 and o["hook_mismatch_counter_full"] == 0 and o["hook_mismatch_counter_sparse"] == 0 and o["sparse_vs_full_rows_differing"] == 0
               and o["n_rows_fired_dec_but_not_tf"] == 0 and o["n_rows_fired_tf_but_not_dec"] == 0 and o["sequences_run_sparse"] < o["sequences_run_full"])
    return ok, out, "hook firing in real passes = gate-level decision from the key cache on every prompt of the development log; sparse evaluation = full evaluation"


def t_cache():
    rows = sample(L.benign("heldout"), 120, 14) + sample(L.rows["request"], 40, 15) + sample(L.rows["neighborhood"], 40, 16)
    tids = list(dict.fromkeys(r["text_id"] for r in rows))
    S, _ = e10t.methods.base_gate(L, "ncoord")
    Sg = S.replace(guard_key=S.guard_key[:0], guard_edit=S.guard_edit[:0]).to(dev)
    wn = (Sg.det_w.norm(dim=1) / Sg.det_s)
    st = {"n": len(tids), "tf_max_rel_key": 0.0, "tf_max_dr_over_scale": 0.0, "dec_max_rel_key": 0.0, "dec_max_dr_over_scale": 0.0, "dec_tokens_equal_y0": 0,
          "f16_max_dr_over_scale": 0.0, "band": e10t.BAND}

    def cmp(live, cached):
        a, c = live.double(), cached.double()
        rel = float(((a - c).norm(dim=1) / c.norm(dim=1)).max())
        dr = (Sg.scores(a) - Sg.scores(c)).abs() / (c.norm(dim=1, keepdim=True) * wn[None, :])
        return rel, float(dr.max())
    got = []
    h = B.mod.register_forward_hook(lambda m, a, o: got.append(a[0].detach().clone()))
    try:
        for bt in mixed_batches(tids, bs=16, seed=5):
            seqs = [L.ids(t) for t in bt]
            got.clear()
            B.tf(seqs, [L.y0(t) for t in bt])
            x = got[0]
            for i, t in enumerate(bt):
                c = torch.from_numpy(np.array(L.kc.keys(t))).to(dev)
                r, d_ = cmp(x[i, :c.shape[0]], c)
                st["tf_max_rel_key"], st["tf_max_dr_over_scale"] = max(st["tf_max_rel_key"], r), max(st["tf_max_dr_over_scale"], d_)
                st["f16_max_dr_over_scale"] = max(st["f16_max_dr_over_scale"], cmp(c.half().float(), c)[1])
            got.clear()
            tok, _ = B.decode(seqs)
            for i, t in enumerate(bt):
                n = len(seqs[i])
                live = torch.cat([got[0][i, :n]] + [g[i] for g in got[1:]])            # prompt positions, then one position per step
                c = torch.from_numpy(np.array(L.kc.keys(t))).to(dev)
                r, d_ = cmp(live, c)
                st["dec_max_rel_key"], st["dec_max_dr_over_scale"] = max(st["dec_max_rel_key"], r), max(st["dec_max_dr_over_scale"], d_)
                st["dec_tokens_equal_y0"] += int(tok[i].tolist() == L.y0(t))
    finally:
        h.remove()
    # the cosine feature: live against cached key, in units of ITS decision band (|theta| + 1; theta >= 0 for a calibrated gate)
    Sc, _ = e10t.methods.base_gate(L, "cos")
    Scg = Sc.replace(guard_key=Sc.guard_key[:0], guard_edit=Sc.guard_edit[:0]).to(dev)
    got.clear()
    h = B.mod.register_forward_hook(lambda m, a, o: got.append(a[0].detach().clone()))
    try:
        mxc = 0.0
        for bt in mixed_batches(tids, bs=16, seed=5):
            got.clear()
            B.tf([L.ids(t) for t in bt], [L.y0(t) for t in bt])
            for i, t in enumerate(bt):
                c = torch.from_numpy(np.array(L.kc.keys(t))).to(dev)
                mxc = max(mxc, float((Scg.scores(got[0][i, :c.shape[0]].double()) - Scg.scores(c.double())).abs().max()))
    finally:
        h.remove()
    st["cos_max_abs_dr_live_vs_cached"] = mxc
    # features on the GPU against the CPU reference
    Fc = e10t.gate.features(L, S, tids, "cpu")["Q"]
    Fg = e10t.gate.features(L, S, tids, "cuda")["Q"]
    st["features_cuda_vs_cpu_max_abs_diff"] = float((Fc - Fg).abs().max())
    ok = (st["tf_max_dr_over_scale"] < e10t.BAND / 4 and st["dec_max_dr_over_scale"] < e10t.BAND / 4 and st["dec_tokens_equal_y0"] == len(tids)
          and st["cos_max_abs_dr_live_vs_cached"] < e10t.BAND / 2          # the cosine's band is BAND * (|theta| + 1) >= BAND
          and st["features_cuda_vs_cpu_max_abs_diff"] < 1e-9 and st["f16_max_dr_over_scale"] > e10t.BAND)
    return ok, st, "cached keys = live keys of a teacher-forced pass and of a decode (padded batches) far inside the decision band; float16 rounding would exceed it"


def t_public():
    import tx_public as TX
    data = e10t.data.counterfact()
    ses = e10t.behave.Session(L, B, store=False)
    scratch = os.path.join(HERE, "scratch", "public_test")
    out = {}
    ok = True
    n_rec = 30
    build_dir = {"deployed": "run_dev100", "preedit": "run_W0"}
    for M in (e10t.methods.deployed(L), e10t.methods.preedit(L), e10t.methods.where(L, "none"), e10t.methods.rescope(L, draw=0)):
        import shutil
        shutil.rmtree(os.path.join(scratch, L.name, M.label), ignore_errors=True)
        sc = e10t.public.scores(ses, M, scratch, first=n_rec)
        o = {k: sc[k] for k in ("n", "ES", "PS", "NS", "EM", "PM", "NM", "S", "rows_fired_pct", "n_records_with_nonfinite_prob")}
        if M.label in build_dir:                                         # against the build's case files (same public function, same weight)
            mx = 0.0
            for ri in L.records[:n_rec]:
                cid = data[ri]["case_id"]
                mine = e10t.data.jload(os.path.join(sc["case_files"], f"case_{cid}.json"))["post"]
                ref = e10t.data.jload(os.path.join(e10t.TXOUT, "public_results", "ROME_seq", build_dir[M.label], f"case_{cid}.json"))["post"]
                for key in ("rewrite_prompts_probs", "paraphrase_prompts_probs", "neighborhood_prompts_probs", "attribute_prompts_probs"):
                    for a, b in zip(mine[key], ref[key]):
                        mx = max(mx, abs(a["target_new"] - b["target_new"]), abs(a["target_true"] - b["target_true"]))
            o["max_abs_diff_to_build_case_files"] = mx
            ok &= mx == 0.0
        out[M.label] = o
    dep, wa = out["deployed"], out["where_all"]
    out["where_all_vs_deployed_max_abs_score_diff"] = max(abs(dep[k] - wa[k]) for k in ("ES", "PS", "NS", "EM", "PM", "NM"))
    ok &= out["where_all_vs_deployed_max_abs_score_diff"] < 0.05 and wa["rows_fired_pct"]["rewrite"] == 100.0 and wa["rows_fired_pct"]["neighborhood"] == 100.0
    # PAD rows of the public batch never fire: per-row fire positions of the all-firing gate = the row's real length
    rec = data[L.records[0]]
    prompts = [rec["requested_rewrite"]["prompt"].format(rec["requested_rewrite"]["subject"])] + rec["paraphrase_prompts"] + rec["neighborhood_prompts"] + rec["attribute_prompts"]
    lens = []
    for p in prompts:
        for tgt in (rec["requested_rewrite"]["target_new"]["str"], rec["requested_rewrite"]["target_true"]["str"]):
            lens.append(len(B.tok(f"{p} {tgt}")["input_ids"]))
    with B.served(e10t.methods.where(L, "none").obj) as srv:
        B.C.public_prob_test(B.model, B.tok, rec)
        fl = srv.rows()
    out["public_batch"] = {"n_rows": len(fl), "n_rows_expected": len(lens), "rows_firing_at_exactly_their_real_length": sum(f["n_fire_pos"] == n for f, n in zip(fl, lens)),
                           "n_pad_positions": int(sum(max(lens) - n for n in lens))}
    ok &= out["public_batch"]["rows_firing_at_exactly_their_real_length"] == len(lens) == len(fl) and out["public_batch"]["n_pad_positions"] > 0
    # generation: the build's texts are reproduced (same public function, same seeds) by weights and by a silent hook
    recs = L.records[:4]
    gen = {}
    Mo = e10t.methods.rescope(L, draw=0)
    off = e10t.methods.wrap(L, Mo.obj.replace(det_tau=torch.full_like(Mo.obj.det_tau, INF)), "all_off")
    for M, tag in ((e10t.methods.preedit(L), "W0"), (e10t.methods.deployed(L), "dev100"), (off, "W0")):
        p = os.path.join(scratch, f"gen_{M.label}.json")
        if os.path.exists(p):
            os.remove(p)
        g = e10t.public.generation(ses, M, p, records=recs)
        mine = e10t.data.jload(p)["rows"]
        ref = {r["record"]: r for r in e10t.data.jload(os.path.join(e10t.TXOUT, "describe", tag, "gen_dev_first100.json"))["rows"]}
        gen[M.label] = {"GE": g["GE"], "n_texts": g["n_texts"], "n_texts_equal_to_build": sum(a == b for r in mine for a, b in zip(r["text"], ref[r["record"]]["text"])),
                        "prompts_fired_pct": g.get("prompts_fired_pct")}
        ok &= gen[M.label]["n_texts_equal_to_build"] == g["n_texts"] == 40
    out["generation"] = gen
    return ok, out, "public evaluation code on the served model: weights reproduce the build's case files and generated texts; an all-firing linear gate reproduces the deployed scores; PAD positions never fire"


def t_store():
    import shutil
    root = os.path.join(HERE, "scratch", "store_test")
    shutil.rmtree(root, ignore_errors=True)
    rows = sample(L.benign("heldout"), 200, 20) + L.rows["request"][:30] + L.rows["paraphrase"][:30]
    out = {}
    ok = True
    for M in (e10t.methods.rescope(L, draw=0), e10t.methods.deployed(L)):
        s1 = e10t.behave.Session(L, B, store=True, store_root=root)
        a = s1.rows(M, rows)
        s2 = e10t.behave.Session(L, None, store=True, store_root=root)     # no model: everything must come from the store / the cache
        b = s2.rows(M, rows)
        keys = ("tf_tok", "dec_tok", "D1_tf", "D1_dec", "fired", "fired_dec", "rel_edits", "n_fire_pos", "new_exact", "own_released")
        o = {"n_rows": len(rows), "first_pass_run": s1.stats["sequences_run"], "first_pass_silent": s1.stats["silent"], "second_pass_run": s2.stats["sequences_run"],
             "second_pass_from_store": s2.stats["from_store"], "rows_differing": _cmp_rows(a, b, keys), "store_writes": s1.store.stats["writes"]}
        out[M.label] = o
        ok &= o["second_pass_run"] == 0 and o["rows_differing"] == 0 and o["first_pass_run"] > 0
    out["silent_rows_cost_no_forward"] = out["rescope"]["first_pass_silent"] > 0 and out["rescope"]["first_pass_run"] < 2 * len(rows)
    ok &= out["silent_rows_cost_no_forward"]
    # the store is keyed by the served OBJECT: two replacement weights with the SAME label never share outcomes
    s3 = e10t.behave.Session(L, B, store=True, store_root=root)
    Wa, Wb = e10t.data.replay_weight(e10t.DEV_LOG, 10), e10t.data.replay_weight(e10t.DEV_LOG, 60)
    Ma, Mb = e10t.methods.weight(L, Wa, "rollback"), e10t.methods.weight(L, Wb, "rollback")
    ra, rb = s3.rows(Ma, rows[:60]), s3.rows(Mb, rows[:60])
    g1 = e10t.methods.rescope(L, draw=0)
    g2 = e10t.methods.wrap(L, g1.obj.replace(det_tau=g1.obj.det_tau + 0.05), g1.label)            # same label, other thresholds
    out["same_label"] = {"weights_sig_differ": s3.sig(Ma) != s3.sig(Mb), "weights_rows_differing": _cmp_rows(ra, rb, ("tf_tok", "dec_tok")),
                         "second_weight_from_store": sum(r["src"] == "store" for r in rb), "gates_sig_differ": s3.sig(g1) != s3.sig(g2),
                         "replayed_prefix_sig_equals_prefix_log_deployed": s3.sig(e10t.methods.weight(L, e10t.data.replay_weight(e10t.DEV_LOG, 100), "x")) == s3.sig(e10t.methods.deployed(L))}
    ok &= (out["same_label"]["weights_sig_differ"] and out["same_label"]["weights_rows_differing"] > 0 and out["same_label"]["second_weight_from_store"] == 0
           and out["same_label"]["gates_sig_differ"] and out["same_label"]["replayed_prefix_sig_equals_prefix_log_deployed"])
    return ok, out, "memoisation: silent prompts cost no forward; a second pass runs the model on nothing and returns the same rows"


def t_guard():
    """Capped edits fire on their own request AS SERVED: for every calibrated gate of draws 0 .. 19 that has a capped edit, the capped
    edits' own request prompts run through the hook (padded batches mixed with other prompts; teacher forced and decode).  With the
    margin (default) the own edit is a candidate in every pass; with the library's thresholds alone (served_guards=False) the cap
    sits within float32 round-off of the served coordinate and some do not fire: that count is the reason for the margin."""
    cases = []
    for d in range(e10t.N_DRAWS):
        for lab in ("rescope", "rescope_null", "cosine", "where_twoblock"):
            M = e10t.methods.build(lab, L, draw=d)
            for e in M.rec["cal"]["capped_edits"]:
                cases.append((lab, d, int(e)))
    M = e10t.methods.where(L, "logonly")
    cases += [("where_logonly", 0, int(e)) for e in M.rec["cal"]["capped_edits"]]
    fill = [r["text_id"] for r in sample(L.benign("heldout"), 7, 30)]
    res = {"n_cases": len(cases), "with_margin": {"tf_own_candidate": 0, "dec_own_candidate": 0}, "library_thresholds_only": {"tf_own_candidate": 0, "dec_own_candidate": 0},
           "cases": [list(c) for c in cases[:40]]}
    for lab, d, e in cases:
        t = L.rows["request"][e]["text_id"]
        bt = fill[:3] + [t] + fill[3:]
        for key, sg in (("with_margin", True), ("library_thresholds_only", False)):
            Mx = e10t.methods.build(lab, L, draw=d, served_guards=sg) if lab != "where_logonly" else e10t.methods.where(L, "logonly", served_guards=sg)
            with B.served(Mx.obj) as srv:
                B.tf([L.ids(x) for x in bt], [L.y0(x) for x in bt])
                f1 = srv.rows(own=[None] * 3 + [e] + [None] * 4)[3]
                B.decode([L.ids(x) for x in bt])
                f2 = srv.rows(own=[None] * 3 + [e] + [None] * 4)[3]
            res[key]["tf_own_candidate"] += int(f1["own_cand"])
            res[key]["dec_own_candidate"] += int(f2["own_cand"])
    ok = len(cases) > 0 and res["with_margin"]["tf_own_candidate"] == res["with_margin"]["dec_own_candidate"] == len(cases)
    return ok, res, "guards as served: every capped edit is a candidate on its own request in live padded passes (teacher forced and decode)"


def t_usage():
    """Execute the example blocks of USAGE.template.txt on the development log.  Substitutions (to keep the test short): rows = 3 of the
    eight rows for (i) - (iii); 2 calibration draws for (iv); 4 records for (v); prefixes 20 / 50 / 100 of the development log and its
    first 20 requests for (vi) (the examples name the reported stream there); scratch output and store directories."""
    import contextlib
    import io
    import re
    import shutil
    text = open(os.path.join(os.path.dirname(HERE), "USAGE.template.txt")).read().replace("{{E10T}}", e10t.E10T)
    blocks = dict(re.findall(r"#BEGIN (\w+)\n(.*?)#END", text, flags=re.S))
    scratch = os.path.join(HERE, "scratch", "usage_test")
    shutil.rmtree(scratch, ignore_errors=True)
    old_store, old_bb = e10t.STORE_ROOT, e10t.Backbone
    e10t.STORE_ROOT = os.path.join(scratch, "store")
    e10t.Backbone = lambda *a, **k: B                                    # the examples create the backbone; the test has one already
    subs = {"vi": [("(100, 500, 1000, 2000)", "(20, 50, 100)"), ('"TX-G2XL-rome-T2000"', f'"{e10t.DEV_LOG}"'), ("first=100", "first=20")],
            "v": [("L.records[:100]", "L.records[:4]")], "iv": [("range(e10t.N_DRAWS)", "range(2)")]}
    out, ns, ok = {}, {}, True
    try:
        for name in ("setup", "i", "ii", "iii", "iv", "v", "vi"):
            code = blocks[name]
            n_lines = len([ln for ln in code.strip().splitlines() if ln.strip()])
            for a, b in subs.get(name, []):
                assert a in code, (name, a)
                code = code.replace(a, b)
            buf = io.StringIO()
            t0 = time.time()
            with contextlib.redirect_stdout(buf):
                exec(compile(code, f"<USAGE example {name}>", "exec"), ns)       # noqa: S102
            if name == "setup":
                ns["out_dir"] = scratch
                ns["rows"] = {k: ns["rows"][k] for k in ("deployed", "preedit", "rescope")}
            lines = buf.getvalue().strip().splitlines()
            out[name] = {"n_lines": n_lines, "seconds": time.time() - t0, "n_printed_lines": len(lines), "first_printed_line": lines[0][:200] if lines else None}
            ok &= n_lines <= 10 and (name == "setup" or len(lines) > 0)
        out["hook_mismatch"] = ns["ses"].stats["hook_mismatch"] + ns["ses"].stats["hook_mismatch_dec"]
        ok &= out["hook_mismatch"] == 0 and sorted(blocks) == sorted(["setup", "i", "ii", "iii", "iv", "v", "vi"])
    finally:
        e10t.STORE_ROOT, e10t.Backbone = old_store, old_bb
    return ok, out, "the example of every metric in USAGE.txt runs as written (development log; substitutions in the test's docstring) and has at most ten lines"


TESTS = {"linear": t_linear, "silent": t_silent, "target": t_target, "null0": t_null0, "batch": t_batch, "hook": t_hook, "cache": t_cache, "public": t_public,
         "store": t_store, "guard": t_guard, "usage": t_usage}
if __name__ == "__main__":
    import fx
    names = sys.argv[1:] or list(TESTS)
    fails = 0
    for nm in names:
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        try:
            ok, detail, what = TESTS[nm]()
            err = None
        except Exception:  # noqa: BLE001
            import traceback
            ok, detail, what, err = False, {}, "", traceback.format_exc()
            if B._served is not None:                                   # leave the model at W_0 for the next test
                try:
                    B._served.__exit__(None, None, None)
                except Exception:  # noqa: BLE001
                    B.set_weight(B.W0)
                    B._served = None
        B.check_w0()
        res = {"name": f"g_{nm}", "pass": bool(ok), "what": what, "detail": detail, "error": err, "seconds": time.time() - t0, "code": e10t.code_fingerprint(),
               "log": e10t.DEV_LOG, "peak_alloc_MiB": torch.cuda.max_memory_allocated() / 2 ** 20, "peak_reserved_MiB": torch.cuda.max_memory_reserved() / 2 ** 20,
               "written": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}
        e10t.data.jdump(fx.io.clean(res), os.path.join(OUT, f"gpu_{nm}.json"))
        fails += int(not ok)
        print(f"{'PASS' if ok else 'FAIL'}  g_{nm}  {time.time() - t0:.1f}s  {what}" + (f"\n{err}" if err else ""), flush=True)
    sys.exit(1 if fails else 0)
