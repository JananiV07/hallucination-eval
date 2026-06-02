"""EntityScore - grounding of named entities mentioned in an answer.

A cheap but surprisingly effective hallucination signal: if the answer names a
person, place, date, or organisation that never appears in the source context,
that entity was likely fabricated.

Scoring logic
-------------
1. Run spaCy NER (``en_core_web_sm`` by default) over the **source context**
   and over the **answer**.
2. Normalise every entity surface form (lowercase, collapse whitespace, strip
   surrounding punctuation).
3. ``EntityScore = |answer_entities n source_entities| / |answer_entities|`` -
   the fraction of the answer's entities that are grounded in the source.
4. If the answer contains **no** entities the score is ``1.0`` (there is nothing
   that could have been hallucinated).

Matching defaults to exact (normalised) string equality. ``match_mode="substring"``
also counts an answer entity grounded if it is a sub-/superstring of a source
entity (e.g. "Obama" vs "Barack Obama"), trading precision for recall.
``labels`` optionally restricts scoring to specific entity types
(e.g. ``{"PERSON", "GPE", "ORG"}``).
"""
from __future__ import annotations

import re

from .base import BaseEvaluator

DEFAULT_SPACY_MODEL = "en_core_web_sm"
_WS = re.compile(r"\s+")
_EDGE_PUNCT = ".,!?;:'\"()[]{}"


def _normalize_entity(text: str) -> str:
    return _WS.sub(" ", str(text).strip().lower()).strip(_EDGE_PUNCT).strip()


class EntityScore(BaseEvaluator):
    """Fraction of an answer's named entities that appear in the source."""

    name = "entity_score"

    def __init__(
        self,
        model_name: str = DEFAULT_SPACY_MODEL,
        labels=None,
        match_mode: str = "exact",
        device: str | None = None,
    ) -> None:
        if match_mode not in ("exact", "substring"):
            raise ValueError("match_mode must be 'exact' or 'substring'")
        self.model_name = model_name
        self.labels = set(labels) if labels else None
        self.match_mode = match_mode
        self.device = device
        self._nlp = None

    def _ensure_model(self):
        if self._nlp is None:
            try:
                import spacy
            except ImportError as exc:  # pragma: no cover - import guard
                raise ImportError(
                    "spaCy is required for EntityScore. Install it with `pip install spacy`."
                ) from exc
            if self.device and self.device.startswith("cuda"):
                try:
                    spacy.prefer_gpu()
                except Exception:  # pragma: no cover - GPU optional
                    pass
            try:
                self._nlp = spacy.load(self.model_name)
            except OSError as exc:
                raise OSError(
                    f"spaCy model '{self.model_name}' is not installed. "
                    f"Run: python -m spacy download {self.model_name}"
                ) from exc
        return self._nlp

    def _entities(self, text: str) -> set[str]:
        if not text or not str(text).strip():
            return set()
        nlp = self._ensure_model()
        doc = nlp(str(text))
        entities: set[str] = set()
        for ent in doc.ents:
            if self.labels is not None and ent.label_ not in self.labels:
                continue
            norm = _normalize_entity(ent.text)
            if norm:
                entities.add(norm)
        return entities

    def _matched(self, answer_entities: set[str], source_entities: set[str]) -> set[str]:
        if self.match_mode == "substring":
            return {
                a
                for a in answer_entities
                if any(a in s or s in a for s in source_entities)
            }
        return answer_entities & source_entities

    def evaluate(self, question: str, context: str, answer: str) -> float:
        source = self._entities(context)
        answer_entities = self._entities(answer)
        if not answer_entities:
            return 1.0
        matched = self._matched(answer_entities, source)
        return len(matched) / len(answer_entities)

    def _score_sample(self, sample: dict) -> tuple[float, dict]:
        context = sample.get("context", "") or ""
        answer = sample.get("answer", "") or ""
        source = self._entities(context)
        answer_entities = self._entities(answer)
        if not answer_entities:
            return 1.0, {
                "n_answer_entities": 0,
                "n_source_entities": len(source),
                "answer_entities": [],
                "matched": [],
                "missing": [],
            }
        matched = self._matched(answer_entities, source)
        missing = sorted(answer_entities - matched)
        score = len(matched) / len(answer_entities)
        return score, {
            "n_answer_entities": len(answer_entities),
            "n_source_entities": len(source),
            "answer_entities": sorted(answer_entities),
            "matched": sorted(matched),
            "missing": missing,
        }

    def _batch_applicable(self, samples: list[dict]) -> bool:
        """EntityScore needs a source context to ground answer entities against."""
        return any((s.get("context") or "").strip() for s in samples)
