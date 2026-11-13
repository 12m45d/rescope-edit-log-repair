"""python tests/extra/drop_stale.py [--dry]: remove the GPU / scale result files in tests/out that were produced by another
version of the package or of the test code (fingerprints of run_tests.py), so that the resumable commands
`run_tests.py gpu --log ...` / `run_tests.py scale` run exactly those tests again.  CPU results are not touched
(`run_tests.py cpu` always runs everything)."""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run_tests  # noqa: E402

dry = "--dry" in sys.argv
for p in sorted(glob.glob(os.path.join(run_tests.OUT, "gpu_*.json")) + glob.glob(os.path.join(run_tests.OUT, "scale_*.json"))):
    with open(p) as f:
        d = json.load(f)
    if d.get("code") != run_tests.CODE or d.get("tests") != run_tests.TESTS["gpu"]:
        print(("would remove " if dry else "removed ") + os.path.basename(p), d.get("code"), d.get("tests"))
        if not dry:
            os.remove(p)
print("current fingerprints", run_tests.CODE, run_tests.TESTS)
