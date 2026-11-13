"""Check of e5c_validity.one_draw against the driver's calibrated gates: for calibration draws 0 - 2 of the driver (n = 200, alpha = 0.05),
the same rows and split seed are given to one_draw; false firing on heldout_disjoint (literal and with guards), the number of capped edits
and the coverage of text / image rephrases must equal fx.gate.decide of the driver's methods (rescope, rescope_shared; allow_silence=True
for the literal rule).     python check_c.py <log>"""
from e5common import *
import e5c_validity as C

def main(name):
    L = fx.logs.open(name)
    F = C.Feats(L)
    hd = L.pool("heldout_disjoint")
    Rhd = F.R(hd, "tf")
    E = C.prep_edits(F, "tf")
    out = {"log": name, "T": L.T, "checks": [], "n_disagreements": 0}
    for d in (0, 1, 2):
        dr = fx.draws.calibration(L, 200, d)
        Q = F.R(dr.rows, "tf")
        acc = C.Acc()
        rk = C.ranks(200, (0.05,))
        C.one_draw(Q, dr.seed, F.cap, {"hd": C.prep_test(F, Rhd, {"all": np.ones(len(hd), bool)})}, E, rk, acc, L.T)
        for rule, mname in (("two_block", "rescope"), ("shared", "rescope_shared")):
            M = fx.methods.build(mname, L, alpha=0.05, n=200, draw=d)
            Ml = fx.methods.build(mname, L, alpha=0.05, n=200, draw=d, allow_silence=True)
            ff = float(np.mean(fx.gate.decide(L, M, hd, "tf")["fired"]))
            ffl = float(np.mean(fx.gate.decide(L, Ml, hd, "tf")["fired"]))
            cov = {m: float(np.mean(fx.gate.decide(L, M, L.edit_rows[m], "tf")["own_top"])) for m in ("tgen", "igen")}
            mine = {"grd": acc.d[(rule, 0.05, "hd", "all", "grd")][0], "lit": acc.d[(rule, 0.05, "hd", "all", "lit")][0], "n_capped": acc.d[(rule, 0.05, "n_capped")][0],
                    "cov": {m: acc.d[(rule, 0.05, "cov", m, "grd")][0] for m in ("tgen", "igen")}}
            ok = abs(mine["grd"] - ff) < 1e-12 and abs(mine["lit"] - ffl) < 1e-12 and mine["n_capped"] == fx.methods.n_capped(M) and all(abs(mine["cov"][m] - cov[m]) < 1e-12 for m in cov)
            out["checks"].append({"draw": d, "rule": rule, "ok": bool(ok), "driver": {"grd": ff, "lit": ffl, "n_capped": fx.methods.n_capped(M), "cov": cov}, "mine": mine})
            out["n_disagreements"] += not ok
    jdump(out, f"{OUT}/c/check_{name}.json")
    print(name, "disagreements", out["n_disagreements"], "of", len(out["checks"]))
    for c in out["checks"]:
        if not c["ok"]:
            print(c)

if __name__ == "__main__":
    main(sys.argv[1])
