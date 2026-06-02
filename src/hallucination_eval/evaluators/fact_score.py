"""FactScore - factual consistency of an answer against a reference fact set.

Scoring logic
-------------
FactScore asks: *of the claims the model makes, how many are consistent with
what we know to be true?* "What we know to be true" is a **reference set** -
one or more gold fact strings (e.g. the dataset's correct answer, or a curated
list of fact-claim pairs).

1. The answer is split into atomic claims (sentence segmentation).
2. Each claim is tested against every reference fact with an NLI cross-encoder,
   producing ``P(contradiction)``, ``P(entailment)``, ``P(neutral)`` per pair.
3. For a claim we keep the strongest entailment and strongest contradiction
   signal across all reference facts, then label it:

   * **supported**  -> ``1.0`` if ``max P(entailment) >= entailment_threshold``
     and entailment beats contradiction;
   * **contradicted** -> ``0.0`` if ``max P(contradiction) >= contradiction_threshold``
     and contradiction beats entailment;
   * **neutral** -> ``neutral_weight`` (default ``0.5``) otherwise - the
     reference neither confirms nor denies the claim.
4. ``FactScore = mean(claim scores)`` in ``[0, 1]``. Higher is better (more
   factual). An empty answer scores ``0.0``; when no reference is available the
   claims cannot be checked and each is treated as ``neutral``.

The score deliberately rewards entailment and punishes contradiction while
treating unverifiable claims as partially credible, which matches how human
fact-checkers reason about a reference document.
"""
from __future__ import annotations

from .._nli import DEFAULT_NLI_MODEL, NLIScorer
from .._text import split_sentences
from .base import BaseEvaluator


def _shorten(text: str, limit: int = 160) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


class FactScore(BaseEvaluator):
    """NLI-based factual-consistency score against a reference fact set."""

    name = "fact_score"

    def __init__(
        self,
        model_name: str = DEFAULT_NLI_MODEL,
        neutral_weight: float = 0.5,
        contradiction_threshold: float = 0.5,
        entailment_threshold: float = 0.5,
        device: str | None = None,
        max_length: int | None = None,
        batch_size: int = 32,
        nli: NLIScorer | None = None,
    ) -> None:
        """
        Parameters
        ----------
        model_name:
            HuggingFace cross-encoder NLI model id.
        neutral_weight:
            Score given to a claim the reference neither entails nor
            contradicts (``0`` = treat unverifiable as wrong, ``1`` = treat as
            right). Default ``0.5``.
        contradiction_threshold / entailment_threshold:
            Minimum probability needed to call a claim contradicted / supported.
        nli:
            Optionally share a pre-built :class:`NLIScorer` (e.g. with
            FaithScore) so the weights load only once.
        """
        self.model_name = model_name
        self.neutral_weight = float(neutral_weight)
        self.contradiction_threshold = float(contradiction_threshold)
        self.entailment_threshold = float(entailment_threshold)
        self._nli = nli or NLIScorer(
            model_name, device=device, max_length=max_length, batch_size=batch_size
        )

    # -- internals ---------------------------------------------------------
    @staticmethod
    def _normalize_references(reference) -> list[str]:
        if reference is None:
            return []
        items = list(reference) if isinstance(reference, (list, tuple)) else [reference]
        out: list[str] = []
        for item in items:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                out.append(text)
        return out

    def _classify_claim(self, references: list[str], claim: str) -> dict:
        """Return the label + signals for a single claim vs. all references."""
        probs = self._nli.classify([(ref, claim) for ref in references])
        max_entailment = max(p["entailment"] for p in probs)
        max_contradiction = max(p["contradiction"] for p in probs)
        if (
            max_contradiction >= self.contradiction_threshold
            and max_contradiction > max_entailment
        ):
            label, score = "contradicted", 0.0
        elif (
            max_entailment >= self.entailment_threshold
            and max_entailment >= max_contradiction
        ):
            label, score = "supported", 1.0
        else:
            label, score = "neutral", self.neutral_weight
        return {
            "claim": claim,
            "label": label,
            "score": score,
            "entailment": round(max_entailment, 4),
            "contradiction": round(max_contradiction, 4),
        }

    # -- public API --------------------------------------------------------
    def evaluate(self, question: str, context: str, answer: str, reference=None) -> float:
        """Score ``answer`` against ``reference`` (falls back to ``context``).

        ``reference`` may be a string or a list of fact strings. If ``None``,
        ``context`` is used as the reference fact set.
        """
        references = self._normalize_references(reference if reference is not None else context)
        claims = split_sentences(answer or "")
        if not claims:
            return 0.0
        if not references:
            return self.neutral_weight
        scores = [self._classify_claim(references, c)["score"] for c in claims]
        return sum(scores) / len(scores)

    def _score_sample(self, sample: dict) -> tuple[float, dict]:
        reference = (
            sample.get("reference")
            or sample.get("gold_answer")
            or sample.get("context")
            or ""
        )
        references = self._normalize_references(reference)
        answer = sample.get("answer", "") or ""
        claims = split_sentences(answer)
        if not claims:
            return 0.0, {"n_claims": 0, "claims": [], "reference_used": _shorten(str(reference))}
        if not references:
            claim_details = [
                {"claim": c, "label": "no_reference", "score": self.neutral_weight}
                for c in claims
            ]
        else:
            claim_details = [self._classify_claim(references, c) for c in claims]
        score = sum(d["score"] for d in claim_details) / len(claim_details)
        diagnostics = {
            "n_claims": len(claim_details),
            "claims": claim_details,
            "reference_used": _shorten(str(reference)),
        }
        return score, diagnostics

    def _batch_applicable(self, samples: list[dict]) -> bool:
        """FactScore needs a reference fact set for at least one sample."""
        return any(
            self._normalize_references(
                s.get("reference") or s.get("gold_answer") or s.get("context") or ""
            )
            for s in samples
        )
