"""GPU tests of stage 2 of the fx driver (fx.review1), development logs only; one log per process.

  $GQ run --mem 10500 --prio 5 -- python t_r1_gpu.py TB-Q100-ae-l2_3 c q h        (17000 for TB-L100-ae-l2_3)
  -> tests/out/r1_gpu_<log>_<test>.json          (python render_r1.py renders tests/out/table_r1.txt)

  c   (3c) one behavioural pass per new SCOPED gate at calibration draw 0 on 100 sequences each of edits, heldout_disjoint and hard
      negatives, every sequence a real forward under the gate (ses.tf_hashed): a sequence on which nothing is released has the
      logits of the pre-edit model bit for bit; the hook's firing == the gate-level decision; the memo path gives the same rows
  q   (3c) the retrieval family and lookup on the same sequences: the driver's rows (memo path under the forced gates; silent
      sequences from the key cache) == the rows of the package's RequestGate through the library's Evaluator (every sequence a real
      forward); silent sequences == the pre-edit model bit for bit; decodes of the driver == decodes of the RequestGate
  h   the behavioural form of H3 for new gates (the raised gate through the memo path == the gate-level read-out)
"""
from __future__ import annotations

import hashlib
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import fx  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

R1, MT = fx.review1, fx.methods
OUT = os.path.join(HERE, "out")
FIELDS = ("exact", "token_acc", "n_tok", "finite", "first_mismatch", "D1", "tok_loc", "fired", "top_last")
SCOPED = [n for n in R1.NAMES if n not in R1.RETRIEVAL_FAMILY + ("lookup", "cosine_shared")]
REQUEST = list(R1.RETRIEVAL_FAMILY) + ["lookup"]


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def spread(v, n):
    return [v[int(i)] for i in np.linspace(0, len(v) - 1, n).round().astype(int)] if len(v) > n else list(v)


def sample_rows(L):
    ed = [L.edit_rows[m][i] for i in range(0, L.T, 3) for m in ("rel", "tgen", "igen")][:100]
    return ed + spread(L.pool("heldout_disjoint"), 100) + spread(L.rows("hn", valid=True), 100)


def n_fields_differ(a, b, fields=FIELDS) -> int:
    return sum(any(x.get(k) != y.get(k) for k in fields) for x, y in zip(a, b))


def finish(ses, res):
    res["hook_mismatch"] = int(ses.stats["hook_mismatch"])
    res["memo_conflicts"] = int(ses.memo.stats.get("conflicts", 0))
    res["forwards"], res["hashed"] = int(ses.stats["forwards"]), int(ses.stats.get("hashed", 0))
    res["ok"] = bool(res["ok"] and res["hook_mismatch"] == 0 and res["memo_conflicts"] == 0)
    return res


# ----------------------------------------------------------------------------- c: scoped gates
def t_c(L, H, ses, state):
    rows = sample_rows(L)
    pre = state.setdefault("pre", ses.tf_hashed(MT.preedit(L), rows))
    res, ok = {"n_sequences": len(rows), "gates": {}}, True
    for nm in SCOPED:
        t0 = time.time()
        M = MT.build(nm, L, alpha=0.05, draw=0)
        a = ses.tf_hashed(M, rows)
        d = fx.gate.decide(L, M, rows, "tf")
        sil = [i for i, r in enumerate(a) if not r["fired"]]
        b = ses.tf(M, rows)
        r = {"n_firing": len(rows) - len(sil), "n_silent": len(sil), "n_silent_not_bitwise_pre_edit": sum(a[i]["logit_sha1"] != pre[i]["logit_sha1"] for i in sil),
             "n_firing_with_pre_edit_logits": sum(a[i]["logit_sha1"] == pre[i]["logit_sha1"] for i in range(len(rows)) if a[i]["fired"]),
             "n_released_differs_from_gate_level": sum(bool(x["fired"]) != bool(f) or (bool(f) and int(x["top_last"]) != int(t)) for x, f, t in zip(a, d["fired"], d["top_edit"])),
             "n_fields_differ_from_memo_path": n_fields_differ(a, b), "seconds": time.time() - t0}
        r["ok"] = r["n_silent_not_bitwise_pre_edit"] == 0 and r["n_released_differs_from_gate_level"] == 0 and r["n_fields_differ_from_memo_path"] == 0
        ok &= r["ok"]
        res["gates"][nm] = r
        say(f"c {nm}: {r}")
    res.update(n_gates=len(SCOPED), n_ok=sum(v["ok"] for v in res["gates"].values()), ok=bool(ok))
    return res


# ----------------------------------------------------------------------------- q: the retrieval family and lookup
def hashed_request(ses, g, rows, lp="tf"):
    """Every sequence one forward under a RequestGate of the feature package (told the sequence before its forward)."""
    ev, out = ses.ev_plain, []
    with torch.no_grad(), ev.hz.HE.vision_cache(ses.H):
        for j in rows:
            ev.trim_vcache()
            it = ses.item(j)
            b = ev.batch(it)
            pre = ev.pre_ids(it, b)
            g.select(it["sid"])
            with ev.applied(g):
                r, lg = ev.tf(it, b, logits=True, lp=lp, pre_ids=pre)
            row = ses._from_lib(j, r, lp, src="hash")
            row["logit_sha1"] = hashlib.sha1(np.ascontiguousarray(lg.detach().float().cpu().numpy()).tobytes()).hexdigest()[:20]
            out.append(row)
    return out


def t_q(L, H, ses, state):
    R = R1.package("retr")
    rows = sample_rows(L)
    pre = state.setdefault("pre", ses.tf_hashed(MT.preedit(L), rows))
    S0 = MT.base_gate(L)[0]
    Sg = S0.replace(guard_key=S0.guard_key[:0], guard_edit=S0.guard_edit[:0], W0=H.W0).to(torch.device(H.dev))
    sids = [L.index[j]["lib_sid"] for j in rows]
    res, ok = {"n_sequences": len(rows), "gates": {}}, True
    for nm in REQUEST:
        t0 = time.time()
        M = MT.build(nm, L, alpha=0.05, draw=0)
        dec = M.obj.decide_rows(L, rows, "tf")
        g = R.RequestGate(Sg, dec, sids, name=nm)
        a = hashed_request(ses, g, rows)
        f0 = int(ses.stats["forwards"])
        b = ses.tf(M, rows)
        f1 = int(ses.stats["forwards"])
        b2 = ses.tf(M, rows)
        sil = [i for i in range(len(rows)) if not dec["fired"][i]]
        r = {"n_firing": len(rows) - len(sil), "n_silent": len(sil), "n_silent_not_bitwise_pre_edit": sum(a[i]["logit_sha1"] != pre[i]["logit_sha1"] for i in sil),
             "n_library_route_flags_differ_from_decision": sum(bool(x["fired"]) != bool(f) or (bool(f) and int(x["top_last"]) != int(t)) for x, f, t in zip(a, dec["fired"], dec["top_edit"])),
             "n_fields_differ_driver_vs_library_route": n_fields_differ(a, b), "driver_forwards_first_pass": f1 - f0, "driver_forwards_second_pass": int(ses.stats["forwards"]) - f1,
             "n_fields_differ_second_pass": n_fields_differ(b, b2), "driver_sources": {s: sum(x["src"] == s for x in b) for s in sorted({x["src"] for x in b})}}
        # decodes: up to 6 firing and 3 silent sequences
        fire = [i for i in range(len(rows)) if dec["fired"][i]]
        sub = spread(fire, 6) + sil[:3]
        gd = ses.generate(M, [rows[i] for i in sub])
        gl = R.gate_rows(ses.ev_plain, g, [ses.item(rows[i]) for i in sub], fn="generate")
        r["n_decodes"] = len(sub)
        r["n_decodes_differ"] = sum(x["y"] != y["y"] or x["long"] != y["long"] or bool(x["fired"]) != bool(y["fired"]) or (bool(y["fired"]) and int(x["top_last"]) != int(y["top_last"]))
                                    for x, y in zip(gd, gl))
        hn = [j for j in rows[200:] if True][:40]
        tfr, gen = ses.hard_negatives(M, rows=hn, generate=True)
        r["hard_negatives_generate"] = {"n": len(gen), "n_decoded": sum("line" in x for x in gen), "n_wrong_target": sum(bool(x.get("wrong_target")) for x in gen),
                                        "flags_consistent": all((x.get("wrong_target") is False) or ("line" in x) for x in gen) and len(tfr) == len(hn)}
        r["seconds"] = time.time() - t0
        r["ok"] = (r["n_silent_not_bitwise_pre_edit"] == 0 and r["n_library_route_flags_differ_from_decision"] == 0 and r["n_fields_differ_driver_vs_library_route"] == 0
                   and r["driver_forwards_second_pass"] == 0 and r["n_fields_differ_second_pass"] == 0 and r["n_decodes_differ"] == 0 and r["hard_negatives_generate"]["flags_consistent"]
                   and bool(raises_type(lambda: ses.tf_hashed(M, rows[:1]), TypeError)))
        ok &= r["ok"]
        res["gates"][nm] = r
        say(f"q {nm}: {r}")
    del Sg
    res.update(n_gates=len(REQUEST), n_ok=sum(v["ok"] for v in res["gates"].values()), ok=bool(ok))
    return res


def raises_type(fn, exc):
    try:
        fn()
    except exc:
        return True
    return False


# ----------------------------------------------------------------------------- h: the behavioural form of H3
def t_h(L, H, ses, state):
    res, ok = {"gates": {}}, True
    for nm in ("rescope_x", "whitecos_0.1", "whitecos_0.1_x", "cos_centred", "retr_sum", "retr_conj"):
        M = MT.build(nm, L, alpha=0.05, draw=0)
        hb = R1.h3_both(L, M) if nm.endswith("_x") else None
        h = hb["all"] if hb else fx.sweeps.h3(L, M)
        r = {"reachable": bool(h["reachable"]), "delta": h["delta"]}
        if h["reachable"]:
            e = ses.edits(h["method"], metrics=("tgen", "igen"), edits=h["odd_edits"])
            flags = np.array([bool(x["own_released_last"]) for x in e])
            want = np.concatenate([np.asarray(h["per_edit"]["tgen"], dtype=bool), np.asarray(h["per_edit"]["igen"], dtype=bool)])
            t = ses.hard_negatives(h["method"], rows=h["test_rows"]["rows"])
            r.update(n_edit_flags_differ=int((flags != want).sum()), n_test_rows=len(t),
                     n_test_flags_differ=sum(bool(x["fired"]) != bool(f) or (bool(f) and int(x["top_last"]) != int(tp)) for x, f, tp in zip(t, h["test_rows"]["fired"], h["test_rows"]["top_edit"])),
                     exact_odd_rephrases=100.0 * float(np.mean([bool(x["exact"]) for x in e])), coverage=h["rephrase"]["mean"])
            if hb:
                sub = hb["unseen_foreign"]["test_rows"]["rows"]
                t2 = ses.hard_negatives(h["method"], rows=sub)
                r["subset"] = {"n_test_rows": len(sub), "n_flags_differ": sum(bool(x["fired"]) != bool(f) for x, f in zip(t2, hb["unseen_foreign"]["test_rows"]["fired"]))}
            r["ok"] = r["n_edit_flags_differ"] == 0 and r["n_test_flags_differ"] == 0 and (not hb or r["subset"]["n_flags_differ"] == 0)
        else:
            r["ok"] = True
        ok &= r["ok"]
        res["gates"][nm] = r
        say(f"h {nm}: {r}")
    res["ok"] = bool(ok)
    return res


TESTS = {"c": ("c_scoped", t_c), "q": ("q_retrieval", t_q), "h": ("h_h3", t_h)}

if __name__ == "__main__":
    name, which = sys.argv[1], [a for a in sys.argv[2:] if not a.startswith("--")] or list(TESTS)
    if name not in fx.logs.names("development"):
        raise SystemExit("behavioural tests run on the development logs only")
    os.makedirs(OUT, exist_ok=True)
    L = fx.logs.open(name)
    H = fx.logs.load_model(name)
    ses = fx.behave.session(L, H)
    state = {}
    for w in which:
        tag, fn = TESTS[w]
        t0 = time.time()
        res = finish(ses, fn(L, H, ses, state))
        res.update(test=tag, log=name, seconds=time.time() - t0, code=fx.code_fingerprint(), time=time.strftime("%Y-%m-%d %H:%M:%S"), session=dict(ses.stats))
        fx.io.jdump(res, os.path.join(OUT, f"r1_gpu_{name}_{tag}.json"), indent=1)
        say(f"{w}: {'PASS' if res['ok'] else 'FAIL'} ({res['seconds']:.0f}s)")
    print("assert_clean:", ses.assert_clean(), flush=True)
