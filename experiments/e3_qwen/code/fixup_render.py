"""E3 fix-up (Qwen2.5-VL-3B): renders out/fixup/table_fixup.txt (one table per editor family) and out/fixup/FIXUP.txt from the JSON
files written by fixup_compute.py.  The renderer is shared by both backbones (x_release/code/xrender.py); it formats stored numbers only.

    python fixup_render.py
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
import xrender  # noqa: E402

if __name__ == "__main__":
    xrender.render_backbone("Q")
