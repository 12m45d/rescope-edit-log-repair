"""E8 extended: the gate alone on the GPU for synthetic logs (no model in the process), fp64 as evaluated.
    gq run --mem 14500 --prio 2 -- python synth_gpu.py            (gq run --mem 3500 ... synth_gpu.py 10000: T <= 10000 only; finished T are skipped)
For both backbones and T in {1000, 10000, 100000}: the synthetic scoped log (synth.py), two-block thresholds from the 200 calibration
keys of draw 0 of the reference log, on the card in fp64; decision + response for one sequence with ONE eligible position (the
teacher-forced last-prompt key of 30 REPAIR queries from the key cache, given as a (1, 1, d) module input), NRUN = 50 runs per key,
CUDA-synchronised.  Reported: median [p10, p90] over all runs and the mean over keys of the per-key minimum (the library's convention for
the gate alone: the minimum is nearly free of contention by other processes on the card; a first run with 10 runs per key on a card at
97 % utilisation with two other processes gave minima that were still inflated: kept under out/latency/superseded_10runs/).  The work of a role-last gate does not depend on the other positions of the sequence, so this is the 'gate alone'
figure of lat.py without the row selection from a captured sequence."""
import sys, time
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from synth import *

NRUN = 50
TMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 100000        # optional: largest T of this call (a small reservation fits T <= 10000)


def main():
    cap = (int(os.environ["GQ_MEM"]) - 600) * 2 ** 20
    torch.cuda.set_per_process_memory_fraction(cap / torch.cuda.get_device_properties(0).total_memory, 0)
    dev = torch.device("cuda:0")
    os.makedirs(OUT + "/latency", exist_ok=True)
    for bb in ("qwen25vl", "llava"):
        Lr, _, _, _, dr = reference(bb)
        rep = Lr.pool("repair")
        rows = [rep[int(i)] for i in np.random.default_rng(20261111).choice(len(rep), 30, replace=False)]
        K = Lr.kc.keys("tf", rows).to(dev)
        for T in [t for t in (1000, 10000, 100000) if t <= TMAX]:
            path = f"{OUT}/latency/{bb}__synthetic-T{T}__gate_only_gpu.json"
            if os.path.exists(path):
                continue
            S, srec = synth_gate(bb, T)
            S2, crec = calibrate_slim(S, Lr.kc, dr.rows, dr.which)
            crec.pop("Q")
            m = S2.values.shape[1]
            g = S2.to(dev)
            del S, S2
            o = torch.zeros(1, 1, m, dtype=torch.bfloat16, device=dev)
            lp = torch.tensor([0], device=dev)
            g.firing(K[0][None, None], lp, 0, {})
            ts, nf = [], 0
            load = [gpu_load()]
            with torch.no_grad():
                for k in K:
                    x = k[None, None]
                    for _ in range(NRUN):
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        f = g.firing(x, lp, 0, {})
                        if f is not None:
                            g.respond(x, o, f)
                        torch.cuda.synchronize()
                        ts.append(time.perf_counter() - t0)
                    nf += f is not None
            load.append(gpu_load())
            A = np.array(ts).reshape(len(K), NRUN)
            res = {"backbone": bb, "T": T, "kind": "synthetic", "synthetic": srec, "calibration": crec, "gate_only_ms": q3(ts, 1e3), "gate_only_ms_mean_of_min": float(A.min(1).mean() * 1e3),
                   "n_keys": len(K), "runs_per_key": NRUN, "gate_only_ms_median_of_min": float(np.median(A.min(1)) * 1e3), "n_keys_firing": int(nf), "device": torch.cuda.get_device_name(0), "load_samples": load, "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()),
                   "note": __doc__}
            jdump(res, path)
            print(bb, T, res["gate_only_ms"], flush=True)
            del g
            torch.cuda.empty_cache()
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
