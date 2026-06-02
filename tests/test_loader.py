"""Tests for dataset normalisation, with HuggingFace ``load_dataset`` mocked."""
import datasets
import pytest

from hallucination_eval.datasets.loader import load_samples


_HALUEVAL_ROWS = [
    {
        "knowledge": "The Eiffel Tower is in Paris.",
        "question": "Where is the Eiffel Tower?",
        "right_answer": "It is in Paris.",
        "hallucinated_answer": "It is in Rome.",
    },
    {
        "knowledge": "Water boils at 100 C at sea level.",
        "question": "What temperature does water boil at?",
        "right_answer": "100 degrees Celsius.",
        "hallucinated_answer": "50 degrees Celsius.",
    },
]

_TRUTHFULQA_ROWS = [
    {
        "question": "What happens if you swallow gum?",
        "best_answer": "It passes through your system.",
        "correct_answers": ["It is excreted normally."],
        "incorrect_answers": ["It stays for seven years."],
        "source": "https://example.com",
    }
]


def test_load_halueval_normalises_fields(monkeypatch):
    def fake_load(repo, config, split=None):
        assert repo == "pminervini/HaluEval"
        return _HALUEVAL_ROWS

    monkeypatch.setattr(datasets, "load_dataset", fake_load)
    samples = load_samples("halueval")
    assert len(samples) == 2
    first = samples[0]
    assert first["context"] == "The Eiffel Tower is in Paris."
    assert first["question"] == "Where is the Eiffel Tower?"
    assert first["reference"] == "It is in Paris."
    assert first["gold_answer"] == "It is in Paris."
    assert first["hallucinated_answer"] == "It is in Rome."
    assert first["answer"] is None


def test_load_halueval_respects_limit(monkeypatch):
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: _HALUEVAL_ROWS)
    samples = load_samples("halueval", limit=1)
    assert len(samples) == 1


def test_load_truthfulqa_builds_reference_list(monkeypatch):
    def fake_load(repo, config, split=None):
        assert repo == "truthful_qa"
        return _TRUTHFULQA_ROWS

    monkeypatch.setattr(datasets, "load_dataset", fake_load)
    samples = load_samples("truthful_qa")
    sample = samples[0]
    assert sample["context"] == ""
    assert sample["reference"] == ["It passes through your system.", "It is excreted normally."]
    assert sample["gold_answer"] == "It passes through your system."
    assert sample["hallucinated_answer"] == "It stays for seven years."


def test_unknown_dataset_raises():
    with pytest.raises(ValueError):
        load_samples("not-a-dataset")


def test_name_normalisation_accepts_separators(monkeypatch):
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: _HALUEVAL_ROWS)
    assert load_samples("Halu-Eval", limit=1)
    assert load_samples("HALU_EVAL", limit=1)
