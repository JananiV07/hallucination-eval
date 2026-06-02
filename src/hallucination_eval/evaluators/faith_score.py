"""FaithScore - faithfulness of an answer to a provided context passage.

Where FactScore checks an answer against gold facts, FaithScore checks it
against the *source context* the model was given. It answers: *does every
claim in the answer follow from the context, or did the model introduce
information the passage does not support?*

Scoring logic
-------------
1. The answer is split into sentences.
2. The context is chunked to fit the NLI cross-encoder's input window; each
   answer sentence is tested against every context chunk
   (premise = context chunk, hypothesis = answer sentence).
3. Per sentence we take the strongest entailment / contradiction across chunks
   and label it:

   * **supported**   -> ``1.0`` (the context entails the sentence);
   * **contradicted** -> ``0.0`` (the context contradicts the sentence);
   * **unsupported** -> ``neutral_weight`` (default ``0.5``) - the sentence is
     neither entailed nor contradicted, i.e. an unsupported addition.
4. ``FaithScore = mean(sentence support)`` in ``[0, 1]``. Higher is better
   (more faithful / less hallucinated).

Notes
-----
* FaithScore requires a context. With no context it is undefined; in batch mode
  it returns ``0.0`` and records a warning, since an ungrounded answer cannot be
  faithful to a passage that was not provided.
* Unsupported sentences are penalised (not free) because introducing claims the
  context does not back is the core failure mode FaithScore targets.
"""
from __future__ import annotations

from .._nli import DEFAULT_NLI_MODEL, NLIScorer
from .._text import chunk_text, split_sentences
from .base import BaseEvaluator


class FaithScore(BaseEvaluator):
    """NLI-based faithfulness score of an answer against a context passage."""

    name = "faith_score"

    def __init__(
        self,
        model_name: str = DEFAULT_NLI_MODEL,
        neutral_weight: float = 0.5,
        contradiction_threshold: float = 0.5,
        entailment_threshold: float = 0.5,
        context_chunk_chars: int = 1200,
        device: str | None = None,
        max_length: int | None = None,
        batch_size: int = 32,
        nli: NLIScorer | None = None,
    ) -> None:
        self.model_name = model_name
        self.neutral_weight = float(neutral_weight)
        self.contradiction_threshold = float(contradiction_threshold)
        self.entailment_threshold = float(entailment_threshold)
        self.context_chunk_chars = int(context_chunk_chars)
        self._nli = nli or NLIScorer(
            model_name, device=device, max_length=max_length, batch_size=batch_size
        )

    def _classify_sentences(self, chunks: list[str], sentences: list[str]) -> list[dict]:
        """Classify every answer sentence against all context chunks in one NLI batch."""
        n_chunk = len(chunks)
        pairs = [(chunk, sent) for sent in sentences for chunk in chunks]
        probs = self._nli.classify(pairs)
        details: list[dict] = []
        for si, sent in enumerate(sentences):
            rows = probs[si * n_chunk : (si + 1) * n_chunk]
            max_entailment = max(p["entailment"] for p in rows)
            max_contradiction = max(p["contradiction"] for p in rows)
            if (
                max_entailment >= self.entailment_threshold
                and max_entailment >= max_contradiction
            ):
                label, score = "supported", 1.0
            elif (
                max_contradiction >= self.contradiction_threshold
                and max_contradiction > max_entailment
            ):
                label, score = "contradicted", 0.0
            else:
                label, score = "unsupported", self.neutral_weight
            details.append(
                {
                    "sentence": sent,
                    "label": label,
                    "score": score,
                    "entailment": round(max_entailment, 4),
                    "contradiction": round(max_contradiction, 4),
                }
            )
        return details

    def evaluate(self, question: str, context: str, answer: str) -> float:
        if not context or not str(context).strip():
            return 0.0
        sentences = split_sentences(answer or "")
        if not sentences:
            return 0.0
        chunks = chunk_text(str(context), self.context_chunk_chars)
        if not chunks:
            return 0.0
        details = self._classify_sentences(chunks, sentences)
        return sum(d["score"] for d in details) / len(details)

    def _score_sample(self, sample: dict) -> tuple[float, dict]:
        context = sample.get("context", "") or ""
        answer = sample.get("answer", "") or ""
        if not str(context).strip():
            return 0.0, {
                "n_sentences": 0,
                "sentences": [],
                "warning": "no context provided; FaithScore is undefined and reported as 0.0",
            }
        sentences = split_sentences(answer)
        if not sentences:
            return 0.0, {"n_sentences": 0, "sentences": []}
        chunks = chunk_text(str(context), self.context_chunk_chars)
        details = self._classify_sentences(chunks, sentences)
        score = sum(d["score"] for d in details) / len(details)
        return score, {"n_sentences": len(details), "sentences": details}

    def _batch_applicable(self, samples: list[dict]) -> bool:
        """FaithScore needs a non-empty context for at least one sample."""
        return any((s.get("context") or "").strip() for s in samples)
