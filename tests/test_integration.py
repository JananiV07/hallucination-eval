"""Integration test using the REAL NLI and spaCy models.

Run explicitly with::

    pytest -m integration

It downloads ``cross-encoder/nli-deberta-v3-small`` and requires
``en_core_web_sm``. No LLM API key is needed - it scores a known-correct answer
against a known-hallucinated one and asserts every metric ranks the correct
answer at least as high.
"""
import pytest

CONTEXT = (
    "The Eiffel Tower is a wrought-iron lattice tower located on the Champ de "
    "Mars in Paris, France. It was completed in 1889 and stands 330 metres tall. "
    "It was designed by the engineer Gustave Eiffel."
)
QUESTION = "Where is the Eiffel Tower and when was it built?"
CORRECT = "The Eiffel Tower is in Paris, France, and was completed in 1889."
HALLUCINATED = "The Eiffel Tower is in Rome, Italy, and was completed in 1750 by Leonardo da Vinci."


@pytest.mark.integration
def test_factscore_ranks_correct_above_hallucinated():
    from hallucination_eval import FactScore

    fs = FactScore()
    good = fs.evaluate(QUESTION, CONTEXT, CORRECT)
    bad = fs.evaluate(QUESTION, CONTEXT, HALLUCINATED)
    assert good > bad, f"FactScore should rank correct ({good}) above hallucinated ({bad})"


@pytest.mark.integration
def test_faithscore_ranks_correct_above_hallucinated():
    from hallucination_eval import FaithScore

    fe = FaithScore()
    good = fe.evaluate(QUESTION, CONTEXT, CORRECT)
    bad = fe.evaluate(QUESTION, CONTEXT, HALLUCINATED)
    assert good > bad, f"FaithScore should rank correct ({good}) above hallucinated ({bad})"


@pytest.mark.integration
def test_entityscore_ranks_correct_above_hallucinated():
    from hallucination_eval import EntityScore

    es = EntityScore()
    good = es.evaluate(QUESTION, CONTEXT, CORRECT)
    bad = es.evaluate(QUESTION, CONTEXT, HALLUCINATED)
    assert good >= bad, f"EntityScore should rank correct ({good}) >= hallucinated ({bad})"
    # The correct answer's entities are all grounded in the context.
    assert good == pytest.approx(1.0)
