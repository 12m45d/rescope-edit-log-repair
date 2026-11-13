"""rescope_paths.py -- the ONE configuration point for every location this repository uses.

The code of this repository was written and run in the authors' working tree and addresses files by the layout of that
tree (ROOT/02_research/experiments/10_final/e1_main/..., ROOT/02_research/code/rescope, ...).  Instead of rewriting
every composed path, this module rebuilds that layout as a small tree of symbolic links (ROOT, below) that points
into this repository, into the unpacked release archives and into your directory of public checkpoints and benchmarks.
Every path constant of the released code is `rescope_paths.ROOT + <path in the working tree>` (or PYTHON, EXT_VIEW).

CONFIGURATION (environment variables; only RESCOPE_HOME is required when the code is run from another directory)
  RESCOPE_HOME    the clone of this repository                                  default: the directory of this file
  RESCOPE_DATA    everything that is not in git                                 default: $RESCOPE_HOME/data
  RESCOPE_LOGS    the unpacked release archives (logs_vl/, logs_text/, ...)     default: $RESCOPE_DATA/logs
  RESCOPE_EXT     public checkpoints, benchmarks, third-party code (README.md, section "Checkpoints and benchmarks"):
                    models/Qwen2.5-VL-3B  models/llava-1.5-7b  models/gpt2-xl  datasets/VLKEB  datasets/MME
                    third_party/rome  data/rome_data  ...                       default: $RESCOPE_DATA/ext
  RESCOPE_WORK    caches (key caches, harness caches), memo stores, the link tree; safe to delete
                                                                                default: $RESCOPE_DATA/work
  RESCOPE_PYTHON  the interpreter that queued jobs are started with             default: sys.executable

THE TABLE `LAYOUT` below is the complete map  path in the working tree -> place in this repository / in your data.
Vision-language logs are released under the names of the paper (Q-A100-r3, ...); the code addresses them by the
directory names of the working tree (TB-...).  LOG_NAMES.json (next to this file) is the one map between the two; the
tree links each working-tree name to the released directory.

  python rescope_paths.py            prints the configuration and the tree, and says which data is present
  python rescope_paths.py --sh       prints `export RESCOPE_ROOT=... RESCOPE_PYTHON=... RESCOPE_EXT_VIEW=...` for shell scripts

A directory of the tree that has more than one source (repository + archive) is a real directory whose entries are links;
files that a script creates there stay under RESCOPE_WORK.  A directory with one source is a link: what a script
writes there (for instance experiments/<name>/out when an experiment is re-run) lands in that source.
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def _env(name: str, default: str) -> str:
    return os.path.abspath(os.path.expanduser(os.environ.get(name) or default))


HOME = _env("RESCOPE_HOME", _HERE)
DATA = _env("RESCOPE_DATA", os.path.join(HOME, "data"))
LOGS = _env("RESCOPE_LOGS", os.path.join(DATA, "logs"))
EXT = _env("RESCOPE_EXT", os.path.join(DATA, "ext"))
WORK = _env("RESCOPE_WORK", os.path.join(DATA, "work"))
PYTHON = os.environ.get("RESCOPE_PYTHON") or sys.executable
ROOT = os.path.join(WORK, "tree")            # the working-tree view (links); every path constant of the code starts here
EXT_VIEW = os.path.join(WORK, "ext_view")    # EXT_VIEW/data/models -> EXT/models, EXT_VIEW/data/datasets -> EXT/datasets

_X = "02_research/experiments/"
# path in the working tree            -> sources (first existing source wins for a name that several sources hold)
LAYOUT = [
    ("02_research/code/rescope",                            [HOME + "/rescope"]),                 # the repair library
    ("02_research/code/vendor_v7",                          [HOME + "/evalkit/vendor_v7"]),       # model / benchmark loaders
    (_X + "06_testbed_v3/code",                             [HOME + "/evalkit/testbed_code"]),    # decoders, registry of the logs
    (_X + "06_testbed_v3/_shared",                          [LOGS + "/logs_vl/_shared"]),
    (_X + "01_v1_attribution/w2/edit_batch.jsonl",          [LOGS + "/logs_vl/Q-A100-r10/edit_batch.jsonl"]),   # the stream files that the
    (_X + "02_complaint_debugging/p5_probe/edit_batch.jsonl", [LOGS + "/logs_vl/L-A100-r10/edit_batch.jsonl"]),  # registry of the 100-edit logs names
    (_X + "07_repair_v3/harness",                           [HOME + "/linear_baselines/harness"]),
    (_X + "08_rescope/p12_linear_streams",                  [HOME + "/linear_baselines/streams"]),
    (_X + "08_rescope/p11_retrofit/out/hard_negatives",     [LOGS + "/logs_vl/_hard_negatives"]),
    (_X + "09_testbed_scale",                               [HOME + "/evalkit/scale"]),           # registry of the long logs
    (_X + "09_testbed_scale/text_public",                   [HOME + "/text_logs", LOGS + "/logs_text"]),
    (_X + "10_final",                                       [HOME + "/experiments"]),             # e1_main ... x_release
    (_X + "10_final/common",                                [HOME + "/driver"]),                  # fx, tests, USAGE.txt
    (_X + "10_final/feat2",                                 [HOME + "/driver/feat2"]),
    (_X + "10_final/PROTOCOL.md",                           [HOME + "/protocol/PROTOCOL.md"]),
    (_X + "10_final/cache",                                 [HOME + "/keycache", LOGS + "/keycache_rows"]),   # + what the capture scripts write
    (_X + "10_final/cache2",                                [HOME + "/keycache/cache2"]),
    (_X + "10_final/memo",                                  [WORK + "/memo"]),                    # memo stores (created on use)
    (_X + "07_repair_v3/harness/cache",                     [WORK + "/harness_cache"]),           # caches of the harness (created on use)
    ("02_research/code/rescope/_hcache",                    [WORK + "/rescope_hcache"]),          # the library's private harness cache
    ("02_research/tools/gpuq/state",                        [WORK + "/gpuq_state"]),              # state of the GPU queue
    ("02_research/tools/gpuq",                              [HOME + "/tools/gpuq"]),
    ("02_research/ext",                                     [EXT]),
    ("01_paper/v4_rescope",                                 [HOME + "/paper"]),
    # three small inputs from outside the experiment directories, released under neutral paths (README, "What can be reproduced")
    ("03_analysis/theory/round4_review/P1b_capacity/predictions_P1b.json",      [HOME + "/experiments/e5_rescope/inputs/capacity_predictions.json"]),
    ("03_analysis/scope_lit/D_unlearning_editing_scale/records.json",           [HOME + "/experiments/e8_cost/inputs/published_figures.json"]),
    (_X + "07_repair_v3/rq1/rows",                          [HOME + "/linear_baselines/rollback_image_exploratory"]),
]
# directories of the tree that are always real directories (scripts create files in them)
_REAL = {_X + "10_final/cache", _X + "10_final/cache2", _X + "06_testbed_v3", "02_research/code", _X + "10_final/common/out/prefix_logs",
         _X + "10_final/common/out", _X + "10_final/cache/work", _X + "10_final/cache/work/qwen25vl", _X + "10_final/cache/work/llava", _X + "10_final/cache2/recomb", _X + "10_final/cache2/mme",
         _X + "10_final/cache2/retr", _X + "10_final/cache2/whiten", _X + "10_final/cache2/benign_parts"}


def log_names() -> dict:
    """LOG_NAMES.json: {"vl": {paper name: {"dir": working-tree name, "batch_deltas": directory, ...}}, "prefix": {...}}."""
    p = os.path.join(HOME, "LOG_NAMES.json")
    if not os.path.exists(p):
        return {"vl": {}, "prefix": {}}
    with open(p) as f:
        return json.load(f)


def _mounts() -> dict:
    m = {rel: list(src) for rel, src in LAYOUT}
    names = log_names()
    for paper, e in names.get("vl", {}).items():
        d = _X + "06_testbed_v3/" + e["dir"]
        if not os.path.isdir(LOGS + "/logs_vl/" + paper):
            continue
        m[d] = [LOGS + "/logs_vl/" + paper]
        m[d + "/batch_deltas"] = [LOGS + "/batch_deltas/" + e["batch_deltas"]]      # rebuilt by tools/logs/rebuild_batch_deltas.py
        _REAL.add(d)
    for paper, e in names.get("prefix", {}).items():
        d = _X + "10_final/common/out/prefix_logs/" + e["dir"]
        if not os.path.isdir(LOGS + "/logs_vl_prefix/" + paper):
            continue
        m[d] = [LOGS + "/logs_vl_prefix/" + paper]
        m[d + "/batch_deltas"] = [LOGS + "/batch_deltas/" + e["batch_deltas"]]
        _REAL.add(d)
    return m


def _link(src: str, dst: str):
    try:
        if os.path.islink(dst):
            if os.readlink(dst) == src:
                return
            os.unlink(dst)
        elif os.path.lexists(dst):
            return                                       # a real file or directory that a script created: keep it
        os.symlink(src, dst)
    except OSError:                                      # another process was faster, or a read-only tree
        pass


def _build(rel: str, inherited: list, mounts: dict):
    dst = os.path.join(ROOT, rel) if rel else ROOT
    sources = [s for s in mounts.get(rel, []) + inherited if os.path.lexists(s)]
    below = sorted({k[len(rel):].lstrip("/").split("/")[0] for k in mounts if k != rel and (not rel or k.startswith(rel + "/"))})
    if not below and rel not in _REAL:
        if len(sources) == 1 or (sources and not os.path.isdir(sources[0])):
            _link(sources[0], dst)
            return
        if not sources:
            return
    if os.path.islink(dst):
        os.unlink(dst)
    os.makedirs(dst, exist_ok=True)
    names = {}
    for s in sources:
        if os.path.isdir(s):
            for n in os.listdir(s):
                names.setdefault(n, []).append(os.path.join(s, n))
    for n in below:
        names.setdefault(n, [])
    for n, srcs in sorted(names.items()):
        sub = (rel + "/" + n) if rel else n
        if n in below or sub in _REAL or (len(srcs) > 1 and all(os.path.isdir(s) for s in srcs)):
            _build(sub, srcs, mounts)
        elif srcs:
            _link(srcs[0], os.path.join(dst, n))


def build_tree():
    """(Re)build the link tree; cheap and idempotent, called at import."""
    for d in [WORK, ROOT, os.path.join(EXT_VIEW, "data")] + [x for _, src in LAYOUT for x in src if x.startswith(WORK + os.sep)]:
        os.makedirs(d, exist_ok=True)
    _link(os.path.join(EXT, "models"), os.path.join(EXT_VIEW, "data", "models"))
    _link(os.path.join(EXT, "datasets"), os.path.join(EXT_VIEW, "data", "datasets"))
    _build("", [], _mounts())


def working_tree_name(paper_name: str) -> str:
    """Directory name that the code uses for a released vision-language log (LOG_NAMES.json)."""
    n = log_names()
    for group in ("vl", "prefix"):
        if paper_name in n.get(group, {}):
            return n[group][paper_name]["dir"]
    return paper_name


_RUN_TIME = None


def name_at_run_time(text: str) -> str:
    """The name(s) under which the released results were computed.  One tag in the working names of the logs of the first
    editor family was replaced in every released file; result files carry digests that were computed under the working names.
    LOG_NAMES.json keeps the working name of each affected log (field "name_at_run_time"); this function replaces, in a name or
    in any string that contains names, every released directory name ("dir") by it.  It is the one reader of that field: the
    digest functions (fx.methods.Method.digest, content_digest, the record fingerprint; fx.review1) pass names through it, so
    that a re-run reproduces the stored digests.  Nothing else uses it (seeded draws do not depend on a log's name)."""
    global _RUN_TIME
    if _RUN_TIME is None:
        import re
        n = log_names()
        m = {e["dir"]: e["name_at_run_time"] for g in ("vl", "prefix") for e in n.get(g, {}).values() if e.get("name_at_run_time")}
        rx = re.compile("|".join(re.escape(k) for k in sorted(m, key=len, reverse=True))) if m else None
        _RUN_TIME = (m, rx)
    m, rx = _RUN_TIME
    return rx.sub(lambda x: m[x.group(0)], text) if rx is not None else text


def log_dir(name: str) -> str:
    """Directory of a released log, by its paper name or by its working-tree name."""
    n = log_names()
    for group, base in (("vl", "02_research/experiments/06_testbed_v3/"), ("prefix", "02_research/experiments/10_final/common/out/prefix_logs/")):
        for paper, e in n.get(group, {}).items():
            if name in (paper, e["dir"]):
                return os.path.join(ROOT, base + e["dir"])
    return os.path.join(ROOT, "02_research/experiments/09_testbed_scale/text_public/logs_edit/library_format", name)


def missing_archives() -> list:
    """The release archives of logs and pools that are not unpacked under RESCOPE_LOGS: [(archive, a directory that is absent)]."""
    out = {}
    n = log_names()
    for group, base in (("vl", "logs_vl"), ("prefix", "logs_vl_prefix")):
        for paper, e in n.get(group, {}).items():
            if not os.path.isdir(os.path.join(LOGS, base, paper)):
                out.setdefault(e.get("archive", base + ".tar"), os.path.join(base, paper))
    if not os.path.isdir(os.path.join(LOGS, "keycache_rows")):
        out["pools_vl.tar"] = "keycache_rows"
    if not os.path.isdir(os.path.join(LOGS, "logs_text")):
        out["logs_text.tar"] = "logs_text"
    return sorted(out.items())


def _has_rows(d: str) -> bool:
    """True if the per-row directory d holds more than the note IN_RELEASE_ARCHIVE.txt that the repository ships in its place."""
    return os.path.isdir(d) and any(n != "IN_RELEASE_ARCHIVE.txt" for n in os.listdir(d))


def _summary_size(path: str):
    """(entries, values) of a JSON summary: entries = length of the top-level object (of its list "records" if it has one),
    values = number of non-null leaf values of the whole file.  None if the file cannot be read as JSON."""
    try:
        with open(path) as f:
            doc = [json.loads(x) for x in f if x.strip()] if path.endswith(".jsonl") else json.load(f)
    except (OSError, ValueError):
        return None
    n, stack = 0, [doc]
    while stack:
        x = stack.pop()
        if isinstance(x, dict):
            stack.extend(x.values())
        elif isinstance(x, list):
            stack.extend(x)
        elif x is not None:
            n += 1
    top = len(doc["records"]) if isinstance(doc, dict) and isinstance(doc.get("records"), list) else (len(doc) if isinstance(doc, (dict, list)) else 1)
    return top, n


def guard_outputs(script_file: str, module_name: str = "__main__", *more_dirs, needs=()):
    """Called in the first lines of every renderer / generator of the README's table: a run that fails, or that would lose
    numbers, leaves the stored outputs untouched and ends with a non-zero exit code.
    (1) `needs`: per-row input directories (relative to the repository) that the script aggregates.  If one of them is absent or
        holds only the note IN_RELEASE_ARCHIVE.txt, nothing is run: the message names the directory and the archive (exit code 2).
    (2) The script's work runs in a child process (the same command line).  Before it starts, the files of the script's output
        directories are copied aside ($RESCOPE_WORK/guard).  If the child ends with an error (a missing input, an exception, a
        kill), every such file is put back as it was, files that the run created there are removed, and the exit code is passed on.
    (3) Postcondition: if the child ends with exit code 0 but a JSON summary of the output directories is gone, is no longer
        readable, or holds fewer entries (top-level keys, or records) or fewer non-null values than before the run, the run has
        dropped numbers (an input was missing and the script went on): everything is put back as in (2), exit code 3.
        A deliberate shrink (for instance after re-running an experiment on fewer logs) is allowed with RESCOPE_ALLOW_SHRINK=1.
    Output directories: <directory above code/>/out with its sub-directories fixup, summary, records; for the paper's scripts
    paper/, paper/figs, paper/scripts/data; plus `more_dirs` (the per-log cache out/cache of experiments/e3_qwen/code/e3_analyze.py,
    so that a stopped run cannot leave a cache of reduced records behind).  Per-row directories are inputs and are not touched by
    these scripts."""
    if module_name != "__main__" or os.environ.get("RESCOPE_GUARDED") == "1":
        return
    import shutil
    import subprocess
    import tempfile
    absent = [n for n in needs if not _has_rows(os.path.join(HOME, n))]
    if absent:
        print(f"[rescope_paths.guard_outputs] {os.path.basename(script_file)}: INPUT NOT FOUND, nothing was run and nothing was changed.\n"
              "  Per-row result files that this script aggregates are absent: " + ", ".join(absent) + "\n"
              f"  They are in the release archive results_rows.tar.gz, which unpacks into the clone:   tar -xzf results_rows.tar.gz -C {HOME}\n"
              "  (or they are written by the gate-level and GPU runs of the experiment: README, table of experiments).  The shipped\n"
              "  summary files of this experiment (paper_numbers.json, results.json, table.txt) are untouched and remain valid.", file=sys.stderr, flush=True)
        sys.exit(2)
    here = os.path.dirname(os.path.abspath(script_file))
    dirs = []
    d = here
    while os.path.basename(d) in ("code", "scripts", "figs", "checks") or d == os.path.join(HOME, "paper"):
        d = os.path.dirname(d)                                         # <experiment>/code -> <experiment>;  paper/scripts[/figs] -> the repository root
    if os.path.abspath(here).startswith(os.path.join(HOME, "paper") + os.sep) or os.path.abspath(here) == os.path.join(HOME, "paper"):
        dirs += [os.path.join(HOME, "paper", s) for s in ("", "figs", "scripts/data")]
    else:
        dirs += [os.path.join(d, "out")] + [os.path.join(d, "out", s) for s in ("fixup", "summary", "records")]
    for x in more_dirs:                                                # a cache directory that was deleted is guarded too: what a stopped run writes there is removed
        os.makedirs(x, exist_ok=True)
    dirs += [os.path.abspath(x) for x in more_dirs]
    dirs = [os.path.normpath(x) for x in dirs if os.path.isdir(x)]
    os.makedirs(os.path.join(WORK, "guard"), exist_ok=True)
    keep = tempfile.mkdtemp(prefix="run_", dir=os.path.join(WORK, "guard"))
    before = {}
    for i, x in enumerate(dirs):
        os.makedirs(os.path.join(keep, str(i)))
        for n in os.listdir(x):
            p = os.path.join(x, n)
            if os.path.isfile(p) and not os.path.islink(p):
                shutil.copy2(p, os.path.join(keep, str(i), n))
                before[p] = os.path.join(keep, str(i), n)

    def same(p, c):
        return os.path.exists(p) and os.path.getsize(p) == os.path.getsize(c) and open(p, "rb").read() == open(c, "rb").read()

    rc = subprocess.call([sys.executable] + sys.argv, env=dict(os.environ, RESCOPE_GUARDED="1"))
    lost = []
    if rc == 0 and os.environ.get("RESCOPE_ALLOW_SHRINK") != "1":      # (3) the postcondition
        for p, c in sorted(before.items()):
            if not os.path.exists(p):
                lost.append(f"{os.path.relpath(p, HOME)}: the file is gone")
            elif p.endswith((".json", ".jsonl")) and not same(p, c):
                b = _summary_size(c)
                a = _summary_size(p) if b is not None else None
                if b is not None and a is None:
                    lost.append(f"{os.path.relpath(p, HOME)}: no longer readable as JSON")
                elif b is not None and (a[0] < b[0] or a[1] < b[1]):
                    lost.append(f"{os.path.relpath(p, HOME)}: {b[0]} -> {a[0]} entries, {b[1]} -> {a[1]} values")
        if lost:
            rc = 3
    if rc != 0:
        restored = removed = 0
        for i, x in enumerate(dirs):
            for n in os.listdir(x):
                p = os.path.join(x, n)
                if os.path.isfile(p) and not os.path.islink(p) and p not in before:
                    os.remove(p)
                    removed += 1
        for p, c in before.items():
            if not same(p, c):
                shutil.copy2(c, p)
                restored += 1
        if lost:
            print(f"\n[rescope_paths.guard_outputs] {os.path.basename(script_file)} ended without an error but would have DROPPED NUMBERS from "
                  f"{len(lost)} summary file(s):\n    " + "\n    ".join(lost[:12]) + (f"\n    ... and {len(lost) - 12} more" if len(lost) > 12 else "") + "\n"
                  "  This happens when an input of the script is missing: the per-row files of results_rows.tar.gz (unpack it into the clone:\n"
                  f"  tar -xzf results_rows.tar.gz -C {HOME}), a log archive under {LOGS} (python rescope_paths.py lists the logs that are\n"
                  "  present), or a summary file of another experiment.  The lines printed above by the script say what it did not find.\n"
                  "  After the input is back, run the command again; a script that caches per-log results (experiments/e3_qwen/code/\n"
                  "  e3_analyze.py, out/cache) recomputes them from the per-row files with the option --no-cache.\n"
                  "  To accept a smaller result on purpose, set RESCOPE_ALLOW_SHRINK=1.", file=sys.stderr)
        print(f"\n[rescope_paths.guard_outputs] {os.path.basename(script_file)} stopped with exit code {rc}: NOTHING WAS CHANGED. The output files under "
              f"{', '.join(os.path.relpath(x, HOME) for x in dirs)} are as they were before this run ({restored} put back, {removed} new files removed). "
              "If the message above names a missing input: README, section 'What can be reproduced', says for every experiment which archive "
              "holds its inputs (results_rows.tar.gz unpacked into the clone, the log archives under data/logs) and which step comes first "
              "(python tools/logs/rebuild_batch_deltas.py for E1, E2, E3).", file=sys.stderr, flush=True)
    shutil.rmtree(keep, ignore_errors=True)
    sys.exit(rc)


def expand(obj):
    """Replace the tokens ${RESCOPE_ROOT} and ${RESCOPE_EXT_VIEW} in the strings of a JSON object (registries and released records
    that hold locations) by ROOT and EXT_VIEW."""
    if isinstance(obj, str):
        return obj.replace("${RESCOPE_ROOT}", ROOT).replace("${RESCOPE_EXT_VIEW}", EXT_VIEW)
    if isinstance(obj, dict):
        return {k: expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand(v) for v in obj]
    return obj


def status() -> str:
    L = [f"RESCOPE_HOME   {HOME}", f"RESCOPE_DATA   {DATA}", f"RESCOPE_LOGS   {LOGS}   {'present' if os.path.isdir(LOGS) else 'MISSING (unpack the release archives here)'}",
         f"RESCOPE_EXT    {EXT}   {'present' if os.path.isdir(EXT) else 'missing (needed only for runs that load a model)'}",
         f"RESCOPE_WORK   {WORK}", f"RESCOPE_PYTHON {PYTHON}", f"ROOT (link tree) {ROOT}", ""]
    n = log_names()
    for group in ("vl", "prefix"):
        for paper, e in n.get(group, {}).items():
            d = log_dir(paper)
            have = os.path.exists(os.path.join(d, "edit_factors.pt"))
            bd = os.path.isdir(os.path.join(d, "batch_deltas"))
            L.append(f"  {paper:16s} {e['dir']:26s} {'present' if have else 'not unpacked'}{'' if not have else (', batch updates present' if bd else ', batch updates not rebuilt (tools/logs/rebuild_batch_deltas.py)')}")
    return "\n".join(L)


build_tree()

if __name__ == "__main__":
    if "--sh" in sys.argv:
        print(f'export RESCOPE_HOME="{HOME}" RESCOPE_ROOT="{ROOT}" RESCOPE_PYTHON="{PYTHON}" RESCOPE_EXT_VIEW="{EXT_VIEW}" RESCOPE_EXT="{EXT}"')
    else:
        print(status())
