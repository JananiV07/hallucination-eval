"""Tests for report aggregation, JSON persistence and rich rendering."""
import io
import json

from rich.console import Console

from hallucination_eval.report import build_report, render_report, save_json

_RESULTS = [
    {"name": "fact_score", "mean": 0.8, "min": 0.5, "max": 1.0, "std": 0.2, "count": 3, "scores": [0.5, 0.9, 1.0], "details": [
        {"id": "a", "answer": "ans a", "score": 0.5},
        {"id": "b", "answer": "ans b", "score": 0.9},
        {"id": "c", "answer": "ans c", "score": 1.0},
    ]},
    {"name": "entity_score", "mean": 0.6, "min": 0.0, "max": 1.0, "std": 0.4, "count": 3, "scores": [0.0, 0.8, 1.0], "details": [
        {"id": "a", "answer": "ans a", "score": 0.0},
        {"id": "b", "answer": "ans b", "score": 0.8},
        {"id": "c", "answer": "ans c", "score": 1.0},
    ]},
]


def test_build_report_aggregates_metrics_and_combined():
    report = build_report(_RESULTS, meta={"model": "test-model", "dataset": "halueval", "n_samples": 3})
    assert set(report["metrics"]) == {"fact_score", "entity_score"}
    assert report["metrics"]["fact_score"]["mean"] == 0.8
    # combined faithfulness = mean of the two means
    assert abs(report["combined_faithfulness"] - 0.7) < 1e-9
    assert abs(report["hallucination_score"] - 0.3) < 1e-9


def test_save_json_roundtrip(tmp_path):
    report = build_report(_RESULTS, meta={"model": "m"})
    path = save_json(report, str(tmp_path / "out" / "report.json"))
    loaded = json.loads(open(path, encoding="utf-8").read())
    assert loaded["meta"]["model"] == "m"
    assert loaded["metrics"]["fact_score"]["count"] == 3


def test_render_report_runs_without_error():
    report = build_report(_RESULTS, meta={"model": "m", "dataset": "halueval", "n_samples": 3})
    console = Console(file=io.StringIO(), width=100)
    render_report(report, console=console, show_samples=True)
    output = console.file.getvalue()
    assert "fact_score" in output
    assert "entity_score" in output
    assert "Hallucination score" in output


def test_inapplicable_metric_excluded_from_combined():
    results = [
        {"name": "fact_score", "mean": 0.8, "min": 0.5, "max": 1.0, "std": 0.2, "count": 3, "applicable": True, "scores": [], "details": []},
        {"name": "faith_score", "mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0, "count": 3, "applicable": False, "scores": [], "details": []},
    ]
    report = build_report(results)
    # faith_score is inapplicable -> excluded; combined == fact_score mean only
    assert report["combined_faithfulness"] == 0.8
    assert report["metrics"]["faith_score"]["applicable"] is False
    assert report["metrics"]["fact_score"]["applicable"] is True


def test_inapplicable_metric_rendered_as_na():
    results = [
        {"name": "fact_score", "mean": 0.8, "min": 0.5, "max": 1.0, "std": 0.2, "count": 3, "applicable": True, "scores": [], "details": []},
        {"name": "entity_score", "mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0, "count": 3, "applicable": False, "scores": [], "details": []},
    ]
    report = build_report(results)
    console = Console(file=io.StringIO(), width=100)
    render_report(report, console=console)
    assert "n/a" in console.file.getvalue()
