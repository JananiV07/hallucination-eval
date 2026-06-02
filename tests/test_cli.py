"""CLI-level tests with the dataset loader mocked (no network, no models)."""
from hallucination_eval import cli

_SAMPLE = [
    {
        "id": "1",
        "question": "q",
        "context": "Paris is in France.",
        "reference": "Paris is in France.",
        "gold_answer": "Paris is in France.",
        "answer": None,
    }
]


def test_cli_rejects_unknown_evaluators(monkeypatch):
    monkeypatch.setattr(cli, "load_samples", lambda *a, **k: _SAMPLE)
    rc = cli.main(["--dataset", "halueval", "--no-generate", "--evaluators", "factt,bogus"])
    assert rc == 1


def test_cli_empty_evaluators_returns_1(monkeypatch):
    monkeypatch.setattr(cli, "load_samples", lambda *a, **k: _SAMPLE)
    rc = cli.main(["--dataset", "halueval", "--no-generate", "--evaluators", ", ,"])
    assert rc == 1


def test_cli_bad_dataset_returns_1(monkeypatch):
    def boom(*a, **k):
        raise ValueError("no such dataset")

    monkeypatch.setattr(cli, "load_samples", boom)
    rc = cli.main(["--dataset", "nope", "--no-generate"])
    assert rc == 1
