"""python tests/extra/snapshot_cache.py: a fixed state of the key caches for the tests, tests/out/keycache_snapshot/<backbone>/
(run the tests with RESCOPE_KEYCACHE_ROOT=tests/out/keycache_snapshot while the caches are still being built).
Per backbone: if the final directory 10_final/cache/<backbone>/ exists, a SYMBOLIC LINK to it (nothing is copied; KeyCache.is_final
is then True); otherwise a COPY of the newest partial assembly 10_final/cache/work/<backbone>/assembled_trial/cache.
Every snapshot is opened with KeyCache(verify=True) (index / key files / meta.json belong together, md5 of the key files).
Writes snapshot.json (source, kind, time, md5 of the index, counts)."""
import hashlib, json, os, shutil, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
from rescope import keycache as KC  # noqa: E402
ROOT = KC.DEFAULT_ROOT
DST = os.path.join(os.path.dirname(HERE), "out", "keycache_snapshot")
os.makedirs(DST, exist_ok=True)
out = {}
for bb in ("qwen25vl", "llava"):
    final, trial = os.path.join(ROOT, bb), os.path.join(ROOT, "work", bb, "assembled_trial", "cache")
    ok = lambda c: os.path.exists(os.path.join(c, "index.jsonl")) and os.path.exists(os.path.join(c, "keys_tf.pt"))      # noqa: E731
    d = os.path.join(DST, bb)
    if os.path.islink(d):
        os.remove(d)
    else:
        shutil.rmtree(d, ignore_errors=True)
    if ok(final):
        os.symlink(final, d)
        src, kind = final, "symbolic link to the final cache"
    elif ok(trial):
        tmp = d + ".tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        for f in ("index.jsonl", "keys_tf.pt", "keys_decode.pt", "keys_decode_steps.pt", "meta.json", "FORMAT.txt"):
            if os.path.exists(os.path.join(trial, f)):
                shutil.copy2(os.path.join(trial, f), os.path.join(tmp, f))
        os.replace(tmp, d)
        src, kind = trial, "copy of a partial assembly"
    else:
        out[bb] = {"source": None}
        continue
    kc = KC.KeyCache(d, verify=True)                                    # raises if the files do not belong together
    with open(os.path.join(d, "index.jsonl"), "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    out[bb] = {"source": src, "kind": kind, "is_final": kc.is_final, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "index_md5": md5,
               "complete": kc.meta.get("complete"), "n_sequences": kc.n, "by_family_kind": kc.summary()["by_family_kind"], "verified_md5": True}
with open(os.path.join(DST, "snapshot.json"), "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out, indent=1))
