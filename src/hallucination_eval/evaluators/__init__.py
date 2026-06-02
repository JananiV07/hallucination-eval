"""Evaluator classes for hallucination-eval.

Three complementary 0-1 metrics, all sharing the
:class:`~hallucination_eval.evaluators.base.BaseEvaluator` interface
(``evaluate`` / ``evaluate_batch``):

* :class:`FactScore`   - NLI factual consistency vs. a reference fact set.
* :class:`FaithScore`  - NLI faithfulness vs. a context passage.
* :class:`EntityScore` - named-entity grounding vs. the source.
"""
from __future__ import annotations

from .base import BaseEvaluator, summarize
from .entity_score import EntityScore
from .fact_score import FactScore
from .faith_score import FaithScore

__all__ = ["BaseEvaluator", "summarize", "FactScore", "FaithScore", "EntityScore"]
