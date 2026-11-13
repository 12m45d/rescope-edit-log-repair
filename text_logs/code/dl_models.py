"""Download the public checkpoints through the mirror (huggingface.co is not reachable from this machine).

usage: python dl_models.py gpt2-xl | gpt-j-6b
Resumable (huggingface_hub resumes partial files).  No GPU work.
"""
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP
import json
import os
import sys
import time

# ARTIFACT: the endpoint is the environment variable HF_ENDPOINT if it is set, otherwise the official hub.  The authors downloaded
# through a mirror (HF_ENDPOINT=https://hf-mirror.com; recorded in <model>/DOWNLOAD_RECORD.json and in out/provenance.json).
os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from huggingface_hub import HfApi, snapshot_download  # noqa: E402

MODELS = f"{_RP.ROOT}/02_research/ext/models"
SPEC = {
    # repo id, revision (pinned to what the mirror served on 2026-11-11), files
    "gpt2-xl": dict(
        repo="openai-community/gpt2-xl",
        revision="15ea56dee5df4983c59b2538573817e1667135e2",
        files=["config.json", "generation_config.json", "merges.txt", "vocab.json", "tokenizer.json",
               "tokenizer_config.json", "model.safetensors", "pytorch_model.bin", "README.md"],
    ),
    "gpt-j-6b": dict(
        repo="EleutherAI/gpt-j-6b",
        revision="47e169305d2e8376be1d31e765533382721b2cc1",  # main = float32
        files=["config.json", "added_tokens.json", "merges.txt", "vocab.json", "tokenizer.json",
               "tokenizer_config.json", "special_tokens_map.json", "pytorch_model.bin", "README.md"],
    ),
}


def main():
    name = sys.argv[1]
    spec = SPEC[name]
    dst = os.path.join(MODELS, name)
    os.makedirs(dst, exist_ok=True)
    t0 = time.time()
    for attempt in range(50):
        try:
            snapshot_download(spec["repo"], revision=spec["revision"], local_dir=dst,
                              allow_patterns=spec["files"], max_workers=4)
            break
        except Exception as e:  # network hiccups: resume
            print(f"[attempt {attempt}] {type(e).__name__}: {str(e)[:300]}", flush=True)
            time.sleep(20)
    else:
        raise SystemExit("download failed")
    api = HfApi(endpoint=os.environ["HF_ENDPOINT"])
    info = api.model_info(spec["repo"], revision=spec["revision"], files_metadata=True)
    want = {s.rfilename: (s.size, getattr(s, "lfs", None)) for s in info.siblings if s.rfilename in spec["files"]}
    rec = dict(name=name, repo=spec["repo"], revision=spec["revision"], endpoint=os.environ["HF_ENDPOINT"],
               seconds=round(time.time() - t0, 1), files={})
    ok = True
    for f, (size, lfs) in sorted(want.items()):
        p = os.path.join(dst, f)
        have = os.path.getsize(p) if os.path.exists(p) else None
        sha = None
        if lfs is not None:
            sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        rec["files"][f] = dict(size_expected=size, size_on_disk=have, lfs_sha256=sha)
        ok = ok and (have == size)
    rec["complete"] = ok
    with open(os.path.join(dst, "DOWNLOAD_RECORD.json"), "w") as fh:
        json.dump(rec, fh, indent=1)
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
