"""Shared repair harness for 07_repair_v3 (every RQ uses this package).

    from harness import load, sample_complaints, repair, evaluate
    H = load("TB-Q100-ae-l2_3")                       # model at W_E, exact W_0 / W_E
    cs = sample_complaints(H, n=10, seed=0)           # seeded, stratified image/text, repair pool
    corr, cost = repair("COMP", H, cs[:1], protect=("K_last",), query="answer")
    rec = evaluate(H, corr, cs[:1])                   # uniform metrics
"""
from .common import HARNESS_DIR  # noqa: F401
from .hlog import load, HLog, Correction  # noqa: F401
from .sampler import sample_complaints, split_pools  # noqa: F401
from .methods import repair, METHODS  # noqa: F401
from .evaluation import evaluate, baseline  # noqa: F401
