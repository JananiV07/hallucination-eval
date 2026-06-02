"""Unit tests for the three evaluators using injected fake backends.

These tests never download a model: FactScore/FaithScore accept a fake NLI
scorer via ``nli=``, and EntityScore's spaCy pipeline is replaced by setting
``_nlp`` directly. This keeps them fast and deterministic while still
exercising the real scoring logic.
"""
from types import SimpleNamespace

import pytest

from hallucination_eval import EntityScore, FactScore, FaithScore


class FakeNLI:
    """Returns confident label probabilities from a ``(premise, hypothesis)`` rule."""

    def __init__(self, rule):
        self.rule = rule

    def classify(self, pairs):
        out = []
        for premise, hypothesis in pairs:
            label = self.rule(premise, hypothesis)
            probs = {"contradiction": 0.05, "entailment": 0.05, "neutral": 0.05}
            probs[label] = 0.9
            out.append(probs)
        return out


def _capital_rule(premise, hypothesis):
    h = hypothesis.lower()
    if "paris" in h:
        return "entailment"
    if "london" in h:
        return "contradiction"
    return "neutral"


# --------------------------------------------------------------------------- #
# FactScore
# --------------------------------------------------------------------------- #
def test_factscore_supported_claim_scores_one():
    fs = FactScore(nli=FakeNLI(_capital_rule))
    score = fs.evaluate("Q", "The capital of France is Paris.", "Paris is the capital.")
    assert score == 1.0


def test_factscore_contradicted_claim_scores_zero():
    fs = FactScore(nli=FakeNLI(_capital_rule))
    score = fs.evaluate("Q", "The capital of France is Paris.", "London is the capital.")
    assert score == 0.0


def test_factscore_neutral_uses_weight():
    fs = FactScore(nli=FakeNLI(_capital_rule), neutral_weight=0.3)
    score = fs.evaluate("Q", "The capital of France is Paris.", "Cheese is tasty.")
    assert score == 0.3


def test_factscore_mean_over_multiple_claims():
    fs = FactScore(nli=FakeNLI(_capital_rule))
    # one supported (paris) + one contradicted (london) -> mean 0.5
    score = fs.evaluate("Q", "The capital of France is Paris.", "Paris is nice. London is the capital.")
    assert score == 0.5


def test_factscore_empty_answer_is_zero():
    fs = FactScore(nli=FakeNLI(_capital_rule))
    assert fs.evaluate("Q", "ref", "") == 0.0


def test_factscore_no_reference_returns_neutral_weight():
    fs = FactScore(nli=FakeNLI(_capital_rule), neutral_weight=0.5)
    assert fs.evaluate("Q", "", "Paris is the capital.") == 0.5


def test_factscore_batch_summary_and_details():
    fs = FactScore(nli=FakeNLI(_capital_rule))
    samples = [
        {"id": "a", "question": "Q", "reference": "The capital of France is Paris.", "answer": "Paris is the capital."},
        {"id": "b", "question": "Q", "reference": "The capital of France is Paris.", "answer": "London is the capital."},
    ]
    result = fs.evaluate_batch(samples)
    assert result["name"] == "fact_score"
    assert result["scores"] == [1.0, 0.0]
    assert result["mean"] == 0.5
    assert result["count"] == 2
    assert result["details"][0]["claims"][0]["label"] == "supported"
    assert result["details"][1]["claims"][0]["label"] == "contradicted"


# --------------------------------------------------------------------------- #
# FaithScore
# --------------------------------------------------------------------------- #
def _faith_rule(premise, hypothesis):
    # premise is the context chunk
    h = hypothesis.lower()
    if "blue" in h:
        return "entailment"
    if "green" in h:
        return "contradiction"
    return "neutral"


def test_faithscore_supported_sentence_scores_one():
    fe = FaithScore(nli=FakeNLI(_faith_rule))
    assert fe.evaluate("Q", "The sky is blue.", "The sky is blue.") == 1.0


def test_faithscore_contradicted_sentence_scores_zero():
    fe = FaithScore(nli=FakeNLI(_faith_rule))
    assert fe.evaluate("Q", "The sky is blue.", "The sky is green.") == 0.0


def test_faithscore_unsupported_sentence_uses_neutral_weight():
    fe = FaithScore(nli=FakeNLI(_faith_rule), neutral_weight=0.5)
    assert fe.evaluate("Q", "The sky is blue.", "Grass exists.") == 0.5


def test_faithscore_no_context_is_zero():
    fe = FaithScore(nli=FakeNLI(_faith_rule))
    assert fe.evaluate("Q", "", "The sky is blue.") == 0.0


def test_faithscore_batch_records_warning_without_context():
    fe = FaithScore(nli=FakeNLI(_faith_rule))
    result = fe.evaluate_batch([{"id": "x", "context": "", "answer": "The sky is blue."}])
    assert result["scores"] == [0.0]
    assert "warning" in result["details"][0]


# --------------------------------------------------------------------------- #
# EntityScore
# --------------------------------------------------------------------------- #
class _FakeDoc:
    def __init__(self, ents):
        self.ents = ents


class FakeNLP:
    """Replaces spaCy: extracts known entities by simple substring matching."""

    KNOWN = [("Paris", "GPE"), ("France", "GPE"), ("London", "GPE"), ("Einstein", "PERSON")]

    def __call__(self, text):
        ents = [
            SimpleNamespace(text=name, label_=label)
            for name, label in self.KNOWN
            if name.lower() in text.lower()
        ]
        return _FakeDoc(ents)


def _entity_scorer(**kwargs):
    es = EntityScore(**kwargs)
    es._nlp = FakeNLP()
    return es


def test_entityscore_all_grounded_scores_one():
    es = _entity_scorer()
    assert es.evaluate("Q", "Paris is in France.", "Paris is lovely.") == 1.0


def test_entityscore_half_grounded():
    es = _entity_scorer()
    # answer entities: paris (grounded), london (not) -> 0.5
    assert es.evaluate("Q", "Paris is in France.", "Paris and London.") == 0.5


def test_entityscore_no_answer_entities_scores_one():
    es = _entity_scorer()
    assert es.evaluate("Q", "Paris is in France.", "It is lovely here.") == 1.0


def test_entityscore_invalid_match_mode_raises():
    with pytest.raises(ValueError):
        EntityScore(match_mode="fuzzy")


def test_entityscore_batch_details_list_missing():
    es = _entity_scorer()
    result = es.evaluate_batch([{"id": "1", "context": "Paris is in France.", "answer": "Paris and London."}])
    assert result["scores"] == [0.5]
    assert result["details"][0]["missing"] == ["london"]
    assert "paris" in result["details"][0]["matched"]


# --------------------------------------------------------------------------- #
# Applicability (inapplicable metrics are excluded from the combined report score)
# --------------------------------------------------------------------------- #
def test_faithscore_applicable_only_with_context():
    fe = FaithScore(nli=FakeNLI(_faith_rule))
    assert fe.evaluate_batch([{"id": "1", "context": "", "answer": "The sky is blue."}])["applicable"] is False
    assert fe.evaluate_batch([{"id": "1", "context": "The sky is blue.", "answer": "The sky is blue."}])["applicable"] is True


def test_entityscore_applicable_only_with_context():
    es = _entity_scorer()
    assert es.evaluate_batch([{"id": "1", "context": "", "answer": "Paris."}])["applicable"] is False
    assert es.evaluate_batch([{"id": "1", "context": "Paris is in France.", "answer": "Paris."}])["applicable"] is True


def test_factscore_applicable_only_with_reference():
    fs = FactScore(nli=FakeNLI(_capital_rule))
    assert fs.evaluate_batch([{"id": "1", "reference": "Paris is the capital.", "answer": "Paris."}])["applicable"] is True
    assert fs.evaluate_batch([{"id": "1", "reference": "", "context": "", "answer": "Paris."}])["applicable"] is False


# --------------------------------------------------------------------------- #
# NLIScorer probability handling (no real model)
# --------------------------------------------------------------------------- #
def test_nli_applies_softmax_when_predict_rejects_apply_softmax():
    import numpy as np

    from hallucination_eval._nli import NLIScorer

    class FakeCrossEncoder:
        config = SimpleNamespace(id2label={0: "contradiction", 1: "entailment", 2: "neutral"})

        def predict(self, pairs, **kwargs):
            if "apply_softmax" in kwargs:
                raise TypeError("apply_softmax not supported on this version")
            return np.array([[2.0, 1.0, 0.5] for _ in pairs])  # raw logits

    scorer = NLIScorer()
    scorer._model = FakeCrossEncoder()
    scorer._labels = {"contradiction": 0, "entailment": 1, "neutral": 2}
    probs = scorer.classify([("a", "b")])[0]
    assert abs(sum(probs.values()) - 1.0) < 1e-6  # softmax was applied to the logits
    assert probs["contradiction"] > probs["entailment"] > probs["neutral"]


def test_resolve_label_indices_uses_config_order_and_falls_back():
    from hallucination_eval._nli import _resolve_label_indices

    swapped = SimpleNamespace(config=SimpleNamespace(id2label={0: "entailment", 1: "neutral", 2: "contradiction"}))
    assert _resolve_label_indices(swapped) == {"entailment": 0, "neutral": 1, "contradiction": 2}

    # Generic labels can't be resolved -> conventional fallback (with a warning).
    generic = SimpleNamespace(config=SimpleNamespace(id2label={0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}))
    assert _resolve_label_indices(generic) == {"contradiction": 0, "entailment": 1, "neutral": 2}


def test_nli_classify_caches_repeated_pairs():
    import numpy as np

    from hallucination_eval._nli import NLIScorer

    class CountingCE:
        config = SimpleNamespace(id2label={0: "contradiction", 1: "entailment", 2: "neutral"})

        def __init__(self):
            self.calls = 0
            self.rows_seen = 0

        def predict(self, pairs, **kwargs):
            self.calls += 1
            self.rows_seen += len(pairs)
            return np.array([[0.1, 0.8, 0.1] for _ in pairs])

    scorer = NLIScorer()
    ce = CountingCE()
    scorer._model = ce
    scorer._labels = {"contradiction": 0, "entailment": 1, "neutral": 2}

    scorer.classify([("a", "b"), ("c", "d")])  # 2 unique pairs -> 1 predict call
    assert ce.calls == 1 and ce.rows_seen == 2
    out = scorer.classify([("a", "b")])  # fully cached -> no new predict
    assert ce.calls == 1
    assert abs(out[0]["entailment"] - 0.8) < 1e-6
