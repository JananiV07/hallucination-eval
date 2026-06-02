"""Thin wrapper around a Natural Language Inference (NLI) cross-encoder.

Both FactScore and FaithScore reduce to the same primitive: given a *premise*
and a *hypothesis*, how strongly does the premise entail / contradict / stay
neutral toward the hypothesis? This module centralises:

* lazy loading of the cross-encoder (so importing the package downloads nothing),
* robust label-index resolution (we read the model config rather than assuming
  an order),
* a version-tolerant ``predict`` that always returns softmax probabilities, and
* a per-pair cache (in-memory, with optional on-disk persistence) so duplicate
  pairs and repeated runs are not recomputed.

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

    Reads ``model.config.id2label`` and matches by substring. Negated/compound
    labels (``non_entailment``) are not mistaken for ``entailment``. Falls back
    to the conventional ``[contradiction, entailment, neutral]`` order (with a
    warning) only when the config is absent or cannot be resolved to three
    distinct indices.
    """
    conventional = {"contradiction": 0, "entailment": 1, "neutral": 2}
    id2label = getattr(getattr(model, "config", None), "id2label", None)
    if not id2label:
        return dict(conventional)

    mapping: dict[str, int] = {}
    for raw_idx, raw_label in id2label.items():
        label = str(raw_label).lower()
        idx = int(raw_idx)
        if "contradict" in label:
            mapping["contradiction"] = idx
        elif "neutral" in label:
            mapping["neutral"] = idx
        elif "entail" in label and "non" not in label and "not" not in label:
            mapping["entailment"] = idx

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
    weights load only once. Results are cached per ``(premise, hypothesis)``.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_NLI_MODEL,
        device: str | None = None,
        max_length: int | None = None,
        batch_size: int = 32,
        cache_path: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.cache_path = cache_path
        self._model = None
        self._labels: dict[str, int] | None = None
        self._cache: dict[tuple[str, str], dict[str, float]] = {}
        self._cache_loaded = False
        self._cache_dirty = False

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

    def _predict(self, pairs) -> list[dict[str, float]]:
        """Run the model on ``pairs`` and return label-probability dicts (no cache)."""
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

    def classify(self, pairs: Sequence[tuple[str, str]]) -> list[dict[str, float]]:
        """Return ``[{"contradiction":p, "entailment":p, "neutral":p}, ...]``.

        Each input pair is ``(premise, hypothesis)``; probabilities sum to ~1.
        Results are memoised per pair so duplicates are computed once, and (when
        ``cache_path`` is set) persist across runs.
        """
        if not pairs:
            return []
        self._maybe_load_cache()
        results: list[dict[str, float] | None] = [None] * len(pairs)
        todo_idx: list[int] = []
        todo_pairs: list[tuple[str, str]] = []
        for i, pair in enumerate(pairs):
            key = (str(pair[0]), str(pair[1]))
            cached = self._cache.get(key)
            if cached is not None:
                results[i] = cached
            else:
                todo_idx.append(i)
                todo_pairs.append(key)
        if todo_pairs:
            for i, key, probs in zip(todo_idx, todo_pairs, self._predict(todo_pairs)):
                results[i] = probs
                self._cache[key] = probs
            self._cache_dirty = True
        return results  # type: ignore[return-value]

    # -- optional on-disk cache -------------------------------------------
    # Stored as JSON: {model_name: [[premise, hypothesis, {probs}], ...]} so no
    # key separator (and therefore no control-character) is ever needed.
    def _maybe_load_cache(self) -> None:
        if self._cache_loaded:
            return
        self._cache_loaded = True
        if not self.cache_path:
            return
        import json
        import os

        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, encoding="utf-8") as handle:
                    raw = json.load(handle)
                for entry in raw.get(self.model_name, []):
                    premise, hypothesis, probs = entry
                    self._cache[(premise, hypothesis)] = probs
            except (OSError, ValueError, TypeError):
                pass  # a corrupt/old cache is non-fatal; just recompute

    def save_cache(self) -> None:
        """Persist the score cache to ``cache_path`` (no-op if unset or unchanged)."""
        if not self.cache_path or not self._cache_dirty:
            return
        import json
        import os

        data: dict = {}
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, encoding="utf-8") as handle:
                    data = json.load(handle)
            except (OSError, ValueError):
                data = {}
        data[self.model_name] = [
            [premise, hypothesis, probs] for (premise, hypothesis), probs in self._cache.items()
        ]
        directory = os.path.dirname(self.cache_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as handle:
            json.dump(data, handle)
        self._cache_dirty = False
