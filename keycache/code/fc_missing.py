"""Number of capture parts of a backbone that are still missing (stdout: one integer).   python fc_missing.py <backbone> [-v]"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fc_common as C  # noqa: E402
import fc_capture as CAP  # noqa: E402


def missing(bb):
    cfg = C.BACKBONES[bb]
    man = C.jsonl_read(os.path.join(C.WORK, bb, "select", "manifest.jsonl"))
    out = []
    for fam in cfg["families"]:
        names = [p for p, _ in CAP.plan_parts(man, fam, 100, [g for g in CAP.GROUPS if g != "editpre"])]
        if fam == cfg["vlkeb"]:
            names += [f"{fam}_editpre_{j:04d}" for j in range(4)]
        out += [(fam, p) for p in names if not os.path.exists(os.path.join(C.WORK, bb, "parts", p + ".pt"))]
    return out


if __name__ == "__main__":
    m = missing(sys.argv[1])
    if "-v" in sys.argv:
        print(m, file=sys.stderr)
    print(len(m))
