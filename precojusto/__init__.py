"""Preço Justo: fair-price estimation for Brazilian public procurement.

Pipeline: pull item prices (sources/) -> ask a typed LLM about each
description (questions.py, features.py) -> evaluate on an ablation ladder
with a locked, group-aware holdout (splits.py, ladder.py).
"""

__version__ = "0.1.0"
