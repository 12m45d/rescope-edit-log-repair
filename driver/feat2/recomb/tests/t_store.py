"""CPU tests of the capture bookkeeping and of the assembly, with FAKE keys (no model): rc_capture.py --fake, rc_assemble.py --tag fake.

    python tests/t_store.py            -> tests/out/t_store.json

s1  parts are resumable: a second run does nothing; a deleted part is recomputed bit for bit; a stale claim of a dead process is taken
    over; two workers (--shard 0/2, 1/2 --reverse) produce the parts of one worker
s2  assembling the same parts twice gives the same bytes of keys.pt (no time stamp inside); rows of keys.pt = rows of rows.jsonl in order
s3  the assembly refuses: parts of another rows.jsonl, a missing part, a test store without --out (the published directory is not touched)
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tcommon as tc  # noqa: E402
from tcommon import check, torch  # noqa: E402

BB, N, PART = "llava", 900, 200
PD = os.path.join(tc.WORK, BB, "parts_fake")
LOG = os.path.join(os.path.dirname(tc.WORK), "logs", f"capture_{BB}_fake.log")
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")


def run(script, *args):
    return subprocess.run([sys.executable, os.path.join(tc.FEAT, script), *args], env=ENV, capture_output=True, text=True)


def md5(p):
    with open(p, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def capture(*extra):
    p = run("rc_capture.py", BB, "--fake", "--limit", str(N), "--part", str(PART), *extra)
    check(p.returncode == 0, p.stdout[-500:] + p.stderr[-500:])
    return p.stdout


def parts():
    return sorted(f for f in os.listdir(PD) if f.endswith(".pt"))


def s1_resumable():
    shutil.rmtree(PD, ignore_errors=True)
    capture()
    names = parts()
    check(names == [f"part_{i:04d}.pt" for i in range(5)], str(names))
    ref = {f: torch.load(os.path.join(PD, f), weights_only=False) for f in names}
    check("nothing to do" in capture(), "a second run recomputed something")
    os.remove(os.path.join(PD, "part_0002.pt"))
    mt = {f: os.path.getmtime(os.path.join(PD, f)) for f in parts()}
    with open(os.path.join(PD, "part_0002.claim"), "w") as f:            # a stale claim of a dead process
        f.write("999999999")
    out = capture()
    check(parts() == names and "1/1 parts" in out, "the missing part was not recomputed alone")
    check(all(os.path.getmtime(os.path.join(PD, f)) == t for f, t in mt.items()), "a finished part was rewritten")
    b = torch.load(os.path.join(PD, "part_0002.pt"), weights_only=False)
    check(torch.equal(b["K"], ref["part_0002.pt"]["K"]) and b["rids"] == ref["part_0002.pt"]["rids"] and not os.path.exists(os.path.join(PD, "part_0002.claim")))
    # two workers
    shutil.rmtree(PD)
    capture("--shard", "0/2")
    check(parts() == ["part_0000.pt", "part_0002.pt", "part_0004.pt"], str(parts()))
    capture("--shard", "1/2", "--reverse")
    check(parts() == names)
    for f in names:
        b = torch.load(os.path.join(PD, f), weights_only=False)
        check(torch.equal(b["K"], ref[f]["K"]) and b["rids"] == ref[f]["rids"] and b["last_prompt_idx"] == ref[f]["last_prompt_idx"], f)
    return {"parts": names, "rows": sum(len(ref[f]["rids"]) for f in names)}


def s2_assemble_deterministic():
    tmp = tempfile.mkdtemp(dir=tc.WORK, prefix="t_store_")
    try:
        sums = []
        for k in range(2):
            d = os.path.join(tmp, str(k))
            p = run("rc_assemble.py", BB, "--tag", "fake", "--limit", str(N), "--out", d)
            check(p.returncode == 0, p.stdout[-500:] + p.stderr[-800:])
            sums.append(md5(os.path.join(d, "keys.pt")))
        check(sums[0] == sums[1], "keys.pt differs between two assemblies of the same parts")
        blob = torch.load(os.path.join(tmp, "0", "keys.pt"), map_location="cpu")
        import json
        with open(os.path.join(os.path.dirname(tc.WORK), BB, "rows.jsonl")) as f:
            rows = [json.loads(next(f)) for _ in range(N)]
        check(blob["rids"] == [r["rid"] for r in rows] and tuple(blob["K_last"].shape) == (N, 64) and blob["K_last"].dtype == torch.float32)
        check(len(blob["last_prompt_idx"]) == N and blob["backbone"] == BB and blob["layer"] == 16)
        a = 0
        for f in parts():                                                # the rows of keys.pt are the rows of the parts, in file order
            b = torch.load(os.path.join(PD, f), weights_only=False)
            check(torch.equal(blob["K_last"][a:a + len(b["rids"])], b["K"].float()) and blob["last_prompt_idx"][a:a + len(b["rids"])] == b["last_prompt_idx"])
            a += len(b["rids"])
        meta = json.load(open(os.path.join(tmp, "0", "capture_meta.json")))
        check(meta["keys_md5"] == sums[0] and meta["n_rows"] == N and meta["complete"] is False)
        return {"keys_md5": sums[0], "n_rows": N}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def s3_refusals():
    tmp = tempfile.mkdtemp(dir=tc.WORK, prefix="t_store_")
    pub = os.path.join(os.path.dirname(tc.WORK), BB)
    mine = ("keys.pt", "capture_meta.json", "README.txt")               # the files rc_assemble.py writes

    def state():
        return {f: os.path.getmtime(os.path.join(pub, f)) if os.path.exists(os.path.join(pub, f)) else None for f in mine}
    before = state()
    try:
        p = run("rc_assemble.py", BB, "--tag", "fake", "--limit", str(N))                       # no --out: must not write into the published directory
        check(p.returncode != 0 and "--out" in (p.stdout + p.stderr), "a test store was assembled without --out")
        f = os.path.join(PD, "part_0001.pt")
        b = torch.load(f, weights_only=False)
        torch.save(dict(b, rows_sha256="0" * 64), f)
        p = run("rc_assemble.py", BB, "--tag", "fake", "--limit", str(N), "--out", os.path.join(tmp, "a"))
        check(p.returncode != 0 and "another rows.jsonl" in (p.stdout + p.stderr), "a part of another rows.jsonl was accepted")
        os.remove(f)
        p = run("rc_assemble.py", BB, "--tag", "fake", "--limit", str(N), "--out", os.path.join(tmp, "b"))
        check(p.returncode != 0 and "without a captured part" in (p.stdout + p.stderr), "a missing part was accepted")
        check(state() == before, "the published directory changed")
        return {"refused": ["no --out", "part of another rows.jsonl", "missing part"], "published_directory_untouched": True}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    R = tc.Results("t_store")
    try:
        R.run("s1_parts_resumable", s1_resumable)
        R.run("s2_assembly_deterministic", s2_assemble_deterministic)
        R.run("s3_refusals", s3_refusals)
    finally:
        shutil.rmtree(PD, ignore_errors=True)
        if os.path.exists(LOG):
            os.remove(LOG)
    sys.exit(0 if R.save() else 1)
