"""v7 core: VLM wrapper (vlm.py), VLKEB data (data.py), VLKEB-standard metrics (metrics.py).

Import with the project's ``code/`` directory on sys.path (``code`` itself clashes with
the stdlib module of the same name):

    import sys; sys.path.insert(0, "<project>/code")
    from v7 import VLM, load_vlkeb
"""
from .vlm import VLM, Capture, load_image, QA_TEMPLATE, DEFAULT_DTYPE, MODEL_DIRS  # noqa: F401

try:  # data/metrics import lazily-heavy deps only on use
    from .data import load_vlkeb  # noqa: F401
    from .metrics import run_probes, build_probes  # noqa: F401
    # [not released: code that no released code calls; it served the related project (README, "What can be reproduced")]
except ImportError:  # pragma: no cover - while modules are being written
    pass
