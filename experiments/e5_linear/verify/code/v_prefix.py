"""Verification, extra for task (3): how the prompt prefix of the edit sequences is shared.     usage: python v_prefix.py
                                                                                               -> verify/out/v_prefix.json
Thm. thm:capacity assumes a subspace Z, of dimension z, that lies in the span of EVERY protection group.  The report uses
z = the number of leading positions whose keys agree within 10 % (relative norm) in all T sequences.  Measured here on the
keys of the edit sequences (harness cache, module dtype): the leading positions are grouped into CLASSES of sequences whose
first z keys are bit for bit equal; per class its size and the range of its sequence lengths; whether the classes are
intervals of the sequence length (no two classes interleave); the relative difference between the prefix keys of two
classes; and the dimension of the span of ALL copies of the prefix keys (LAPACK singular values, cut max(shape) eps s_1).
Why the copies differ is NOT investigated here.
"""
from __future__ import annotations

import v_common as C                    # first: it fixes the number of BLAS threads before numpy is loaded

import hashlib
import os
import time

import numpy as np
import scipy.linalg as sla
import torch


def main():
    t0 = time.time()
    out = {"definition": __doc__, "logs": {}}
    for name, cfg in C.LOGS.items():
        ek = torch.load(os.path.join(C.HARNESS, "cache", cfg["editkeys"]), map_location="cpu")
        T = len(ek["K"])
        lens = [int(x) for x in ek["len"]]
        z = C.jload(os.path.join(C.OUT, f"v_capacity_{name}.json"))["edit_keys"]["z_near"]
        cls, of = {}, []
        for i in range(T):
            h = hashlib.blake2b((ek["K"][i][:z].float() + 0.0).numpy().tobytes(), digest_size=16).digest()
            of.append(cls.setdefault(h, len(cls)))
        info = []
        for c in range(len(cls)):
            mem = [i for i in range(T) if of[i] == c]
            info.append({"class": c, "n": len(mem), "len_min": min(lens[i] for i in mem), "len_max": max(lens[i] for i in mem), "first_edit": mem[0]})
        info.sort(key=lambda d: d["len_min"])
        intervals = all(info[k]["len_max"] < info[k + 1]["len_min"] for k in range(len(info) - 1))
        reps = [ek["K"][d["first_edit"]][:z].double() for d in info]
        rel = [float(((reps[a] - reps[b]).norm(dim=1) / reps[b].norm(dim=1)).max()) for a in range(len(reps)) for b in range(len(reps)) if a != b]
        allk = torch.cat(reps, 0).numpy()
        s = sla.svdvals(allk)
        order = sorted(range(T), key=lambda i: (lens[i], i))
        L = {"T": T, "z": int(z), "n_classes": len(cls), "classes_by_length": info, "classes_are_length_intervals": bool(intervals),
             "rel_diff_between_classes": {"min": min(rel) if rel else 0.0, "max": max(rel) if rel else 0.0},
             "n_prefix_keys_all_classes": int(allk.shape[0]),
             "rank_of_all_prefix_copies": int((s > max(allk.shape) * C.EPS * s[0]).sum()),
             "class_of_the_n_smallest": [of[i] for i in order],
             "first_n_with_a_second_class": next((n + 1 for n in range(T) if len({of[i] for i in order[: n + 1]}) >= 2), None)}
        out["logs"][name] = L
        print(name, {k: v for k, v in L.items() if k not in ("classes_by_length", "class_of_the_n_smallest")})
        print("   ", [(d["n"], d["len_min"], d["len_max"]) for d in info])
    out["sec"] = time.time() - t0
    C.jdump(out, os.path.join(C.OUT, "v_prefix.json"))


if __name__ == "__main__":
    main()
