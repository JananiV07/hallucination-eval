"""hallucination-eval: measure LLM hallucination with three complementary metrics.

- :class:`FactScore`   - NLI contradiction/entailment of answer claims vs. a
  reference fact set.
- :class:`FaithScore`  - NLI support of answer sentences against a context passage.
- :class:`EntityScore` - fraction of answer named-entities that appear in the source.

The heavy ML backends (sentence-transformers, spaCy, openai) are imported
lazily, so importing this package is cheap and does not download any weights.
"""
from __future__ import annotations

__version__ = "0.1.0"

from .evaluators import BaseEvaluator, EntityScore, FactScore, FaithScore
from .model_client import ModelClient
from .datasets import load_samples

__all__ = [
    "__version__",
    "BaseEvaluator",
    "FactScore",
    "FaithScore",
    "EntityScore",
    "ModelClient",
    "load_samples",
]
