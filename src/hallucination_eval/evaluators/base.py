"""Common base class for all evaluators.

Every evaluator exposes the same two-method contract:

* ``evaluate(question, context, answer) -> float`` - a single 0-1 score.
* ``evaluate_batch(samples) -> dict`` - scores for a list of sample dicts plus
  summary statistics and per-sample diagnostics.

Subclasses implement ``evaluate`` and, when they need extra per-sample fields
(e.g. FactScore needs the reference), override ``_score_sample``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


def summarize(scores: list[float]) -> dict[str, float]:
    """Return mean/min/max/std/count for a list of scores (population std)."""
    count = len(scores)
    if count == 0:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0, "count": 0}
    mean = sum(scores) / count
    variance = sum((x - mean) ** 2 for x in scores) / count
    return {
        "mean": mean,
        "min": min(scores),
        "max": max(scores),
        "std": variance**0.5,
        "count": count,
    }


class BaseEvaluator(ABC):
    """Abstract base for FactScore / FaithScore / EntityScore.

    Models are loaded lazily inside ``evaluate`` (never at import time), so
    constructing an evaluator is cheap and importing the package downloads
    nothing.
    """

    name: str = "base"

    @abstractmethod
    def evaluate(self, question: str, context: str, answer: str, **kwargs: Any) -> float:
        """Score a single (question, context, answer) triple in ``[0, 1]``."""
        raise NotImplementedError

    def _score_sample(self, sample: dict) -> tuple[float, dict]:
        """Score one sample dict, returning ``(score, diagnostics)``.

        Default implementation pulls ``question``/``context``/``answer`` and
        calls :meth:`evaluate`. Subclasses override this to surface richer
        diagnostics or to read extra fields.
        """
        score = self.evaluate(
            sample.get("question", "") or "",
            sample.get("context", "") or "",
            sample.get("answer", "") or "",
        )
        return score, {}

    def _batch_applicable(self, samples: list[dict]) -> bool:
        """Whether this metric is meaningful for ``samples``.

        Subclasses override this to mark a metric inapplicable (e.g. a
        context-based metric on a context-free dataset). Inapplicable metrics
        are still reported per-metric but are excluded from the report's
        combined score so they cannot distort it.
        """
        return True

    def evaluate_batch(self, samples: list[dict]) -> dict:
        """Evaluate a list of sample dicts.

        Each sample should contain at least ``answer`` (and ``context`` /
        ``reference`` depending on the evaluator). Returns::

            {
              "name": <evaluator name>,
              "mean", "min", "max", "std", "count": summary stats,
              "scores": [per-sample float, ...],
              "details": [{"id", "question", "answer", "score", ...diag}, ...],
            }
        """
        scores: list[float] = []
        details: list[dict] = []
        for i, sample in enumerate(samples):
            score, diagnostics = self._score_sample(sample)
            score = float(max(0.0, min(1.0, score)))
            scores.append(score)
            detail = {
                "id": sample.get("id", i),
                "question": sample.get("question", ""),
                "answer": sample.get("answer", ""),
                "score": score,
            }
            detail.update(diagnostics)
            details.append(detail)
        return {
            "name": self.name,
            "applicable": self._batch_applicable(samples),
            **summarize(scores),
            "scores": scores,
            "details": details,
        }
