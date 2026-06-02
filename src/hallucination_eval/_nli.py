"""Thin wrapper around a Natural Language Inference (NLI) cross-encoder.

Both FactScore and FaithScore reduce to the same primitive: given a *premise*
and a *hypothesis*, how strongly does the premise entail / contradict / stay
neutral toward the hypothesis? This module centralises:

* lazy loading of the cross-encoder (so importing the package downloads nothing),
* robust label-index resolution (we read the model config rather than assuming
  an order), and
* a version-tolerant ``predict`` that always returns softmax probabilities.

The default model is ``cross-encoder/nli-deberta-v3-small`` from
sentence-transformers, whose label order is
``[contradiction, entailment, neutral]`` - but we never hard-code that; we map
labels by name from ``model.config.id2label`` with a sane fallback.
"""
from __future__ import annotations

from typing import Sequence

DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-small"


def _resolve_label_indices(model) -> dict[str, int]:
    """Map ``{"contradiction": i, "entailment": j, "neutral": k}`` from a model.

    Reads ``model.config.id2label`` and matches by substring so it works across
    label spellings (``ENTAILMENT``, ``entail``, ...). Falls back to the
    conventional cross-encoder NLI order ``[contradiction, entailment, neutral]``
    for any label it cannot find.
    """
    conventional = {"contradiction": 0, "entailment": 1, "neutral": 2}
    id2label = getattr(getattr(model, "config", None), "id2label", None)
    if not id2label:
        return dict(conventional)

    mapping: dict[str, int] = {}
    for raw_idx, raw_label in id2label.items():
        label = str(raw_label).lower()
        idx = int(raw_idx)
        # Check the negated/compound labels carefully so that e.g.
        # "non_entailment" is not mistaken for "entailment".
        if "contradict" in label:
            mapping["contradiction"] = idx
        elif "neutral" in label:
            mapping["neutral"] = idx
        elif "entail" in label and "non" not in label and "not" not in label:
            mapping["entailment"] = idx

    # Only trust id2label if it resolved all three labels to distinct indices;
    # otherwise the conventional order is the safest guess.
    if set(mapping) == {"contradiction", "entailment", "neutral"} and len(set(mapping.values())) == 3:
        return mapping

    import warnings

    warnings.warn(
        f"Could not unambiguously resolve NLI label order from id2label={id2label!r}; "
        "falling back to the conventional [contradiction, entailment, neutral] order."
    )
    return dict(conventional)


def _softmax(matrix):
    import numpy as np

    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    shifted = matrix - matrix.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


class NLIScorer:
    """Lazy NLI cross-encoder returning per-pair label probabilities.

    A single instance can be shared between FactScore and FaithScore so the
    weights are only loaded once.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_NLI_MODEL,
        device: str | None = None,
        max_length: int | None = None,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self._model = None
        self._labels: dict[str, int] | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self):
        """Load and cache the cross-encoder. Safe to call repeatedly."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            kwargs = {}
            if self.max_length is not None:
                kwargs["max_length"] = self.max_length
            if self.device is not None:
                kwargs["device"] = self.device
            self._model = CrossEncoder(self.model_name, **kwargs)
            self._labels = _resolve_label_indices(self._model)
        return self._model

    def classify(self, pairs: Sequence[tuple[str, str]]) -> list[dict[str, float]]:
        """Return ``[{"contradiction":p, "entailment":p, "neutral":p}, ...]``.

        Each input pair is ``(premise, hypothesis)``. Probabilities sum to ~1.
        """
        if not pairs:
            return []
        import numpy as np

        model = self.load()
        assert self._labels is not None

        # sentence-transformers changed ``predict``'s signature across major
        # versions; ask for softmax but fall back to applying it ourselves.
        try:
            scores = model.predict(
                list(pairs),
                apply_softmax=True,
                convert_to_numpy=True,
                batch_size=self.batch_size,
            )
        except TypeError:
            # This signature has no apply_softmax, so predict returns raw logits;
            # apply softmax here rather than relying on the heuristic guard below.
            scores = model.predict(
                list(pairs), convert_to_numpy=True, batch_size=self.batch_size
            )
            scores = _softmax(np.asarray(scores, dtype=float))
        scores = np.asarray(scores, dtype=float)
        if scores.ndim == 1:
            scores = scores.reshape(1, -1)
        # Guarantee probabilities even if the model returned raw logits.
        row_sums = scores.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-2) or (scores < 0).any():
            scores = _softmax(scores)

        c = self._labels["contradiction"]
        e = self._labels["entailment"]
        n = self._labels["neutral"]
        return [
            {
                "contradiction": float(row[c]),
                "entailment": float(row[e]),
                "neutral": float(row[n]),
            }
            for row in scores
        ]
