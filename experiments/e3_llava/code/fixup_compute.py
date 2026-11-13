"""E3 fix-up (LLaVA-1.5-7B; amendment of 2026-11-12 06:16 UTC): additions to the finished and verified experiment, new files only.

    python fixup_compute.py        (CPU; reads out/rows, out/behave, out/gate; writes out/fixup/*.json)
    python fixup_render.py         (renders out/fixup/table_fixup.txt and out/fixup/FIXUP.txt from the JSON files)

Items 1 (paired differences H2 / H3: out/fixup/endpoints.json), 3 (exact McNemar tests: mcnemar.json) and 4 (reporting numbers next
to far locality: reporting.json); paper_numbers_fixup.json + paper_numbers_fixup_doc.json; checks.json.  The computation is shared by
both backbones and lives in x_release/code/xlib.py (class Backbone hides the two file layouts); nothing is typed by hand.
"""
import os as _go, sys as _gs  # ARTIFACT: a failed run leaves the stored output files untouched (rescope_paths.guard_outputs)
_gd = _go.environ.get("RESCOPE_HOME") or _go.path.dirname(_go.path.abspath(__file__))
while not _go.path.exists(_go.path.join(_gd, "rescope_paths.py")) and _go.path.dirname(_gd) != _gd: _gd = _go.path.dirname(_gd)
_gs.path.insert(0, _gd); import rescope_paths as _RPG; _RPG.guard_outputs(__file__, __name__)
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, f"{_RP.ROOT}/02_research/experiments/10_final/x_release/code")
import xlib  # noqa: E402

if __name__ == "__main__":
    xlib.run_backbone("L")
