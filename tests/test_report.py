"""Tests for report aggregation, JSON persistence and rich rendering."""
import io
import json

from rich.console import Console

from hallucination_eval.report import build_report, render_comparison, render_report, save_csv, save_json

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


def test_label_counts_in_build_report():
    results = [
        {
            "name": "fact_score", "mean": 0.5, "min": 0.0, "max": 1.0, "std": 0.5, "count": 1,
            "applicable": True, "scores": [0.5],
            "details": [{"id": "1", "answer": "a", "score": 0.5,
                         "claims": [{"label": "supported"}, {"label": "contradicted"}]}],
        }
    ]
    report = build_report(results)
    assert report["metrics"]["fact_score"]["labels"] == {"supported": 1, "contradicted": 1}


def test_save_csv_roundtrip(tmp_path):
    report = build_report(_RESULTS, meta={"model": "m"})
    path = save_csv(report, str(tmp_path / "out.csv"))
    lines = open(path, encoding="utf-8").read().splitlines()
    assert lines[0] == "id,fact_score,entity_score,answer"
    assert len(lines) == 1 + 3  # header + 3 samples


def test_render_comparison_runs():
    a = build_report(_RESULTS, meta={"model": "model-A"})
    b = build_report(_RESULTS, meta={"model": "model-B"})
    console = Console(file=io.StringIO(), width=120)
    render_comparison({"model-A": a, "model-B": b}, console=console)
    out = console.file.getvalue()
    assert "model-A" in out and "model-B" in out
    assert "combined_faithfulness" in out


def test_save_csv_sanitizes_formula_injection(tmp_path):
    results = [
        {
            "name": "fact_score", "mean": 1.0, "min": 1.0, "max": 1.0, "std": 0.0, "count": 1,
            "applicable": True, "scores": [1.0],
            "details": [{"id": "=danger", "answer": "=cmd|'/c calc'!A1", "score": 1.0}],
        }
    ]
    report = build_report(results)
    text = open(save_csv(report, str(tmp_path / "x.csv")), encoding="utf-8").read()
    assert "'=danger" in text  # leading '=' neutralised
    assert "'=cmd" in text
