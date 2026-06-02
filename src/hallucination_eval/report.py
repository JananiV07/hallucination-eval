"""Reporting: a rich terminal table plus a machine-readable JSON report.

``build_report`` aggregates the per-evaluator batch results into a single
report dict; ``render_report`` prints a coloured ``rich`` summary (and an
optional per-sample table); ``save_json`` persists the full report.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


def build_report(results: list[dict], meta: Optional[dict] = None) -> dict:
    """Aggregate evaluator batch results into one report.

    ``results`` is a list of dicts as returned by
    :meth:`BaseEvaluator.evaluate_batch`. The report records each metric's
    summary stats, a ``combined_faithfulness`` (mean of the metric means) and
    its complement ``hallucination_score = 1 - combined_faithfulness``.
    """
    metrics: dict[str, dict] = {}
    means: list[float] = []
    for result in results:
        applicable = bool(result.get("applicable", True)) and result.get("count", 0) > 0
        stats = {key: result.get(key) for key in ("mean", "min", "max", "std", "count")}
        stats["applicable"] = applicable
        metrics[result["name"]] = stats
        # Only applicable metrics feed the combined score, so a metric that is
        # structurally ~0 on this dataset (e.g. a context metric with no context)
        # cannot drag the headline number down.
        if applicable:
            means.append(float(result.get("mean", 0.0)))
    combined = sum(means) / len(means) if means else 0.0
    return {
        "meta": meta or {},
        "metrics": metrics,
        "combined_faithfulness": combined,
        "hallucination_score": 1.0 - combined,
        "evaluators": results,
    }


def _mean_color(mean: float) -> str:
    if mean >= 0.75:
        return "green"
    if mean >= 0.5:
        return "yellow"
    return "red"


def render_report(report: dict, console=None, show_samples: bool = False, max_sample_rows: int = 20) -> None:
    """Print the report to the terminal using ``rich``."""
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = console or Console()
    meta = report.get("meta", {})

    info_fields = [
        ("Model", "model"),
        ("Dataset", "dataset"),
        ("Split", "split"),
        ("Samples", "n_samples"),
        ("Answers", "answers"),
        ("NLI model", "nli_model"),
    ]
    info_lines = [
        f"[bold]{label}:[/bold] {meta[key]}"
        for label, key in info_fields
        if meta.get(key) not in (None, "")
    ]
    if info_lines:
        console.print(Panel("\n".join(info_lines), title="hallucination-eval", expand=False))

    table = Table(title="Evaluation summary", box=box.ROUNDED)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Mean", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("N", justify="right")
    has_inapplicable = False
    for name, stats in report.get("metrics", {}).items():
        if not stats.get("applicable", True):
            has_inapplicable = True
            table.add_row(name, "[dim]n/a[/dim]", "-", "-", "-", str(stats.get("count", 0) or 0))
            continue
        mean = float(stats.get("mean", 0.0) or 0.0)
        color = _mean_color(mean)
        table.add_row(
            name,
            f"[{color}]{mean:.3f}[/{color}]",
            f"{float(stats.get('min', 0.0) or 0.0):.3f}",
            f"{float(stats.get('max', 0.0) or 0.0):.3f}",
            f"{float(stats.get('std', 0.0) or 0.0):.3f}",
            str(stats.get("count", 0)),
        )
    console.print(table)
    if has_inapplicable:
        console.print(
            "[dim]n/a = metric not applicable to this dataset; excluded from the combined score.[/dim]"
        )

    combined = report.get("combined_faithfulness", 0.0)
    hallucination = report.get("hallucination_score", 0.0)
    console.print(
        f"[bold]Combined faithfulness:[/bold] [{_mean_color(combined)}]{combined:.3f}[/{_mean_color(combined)}]"
        f"    [bold]Hallucination score:[/bold] {hallucination:.3f}"
    )

    if show_samples:
        _render_samples(console, report, max_sample_rows)


def _render_samples(console, report: dict, max_sample_rows: int) -> None:
    from rich import box
    from rich.table import Table

    evaluators = report.get("evaluators", [])
    if not evaluators:
        return
    metric_names = [ev["name"] for ev in evaluators]
    n_rows = min(len(evaluators[0].get("details", [])), max_sample_rows)

    table = Table(title="Per-sample scores", box=box.SIMPLE)
    table.add_column("id", style="dim", no_wrap=True)
    for name in metric_names:
        table.add_column(name, justify="right")
    table.add_column("answer", overflow="fold", max_width=60)

    for i in range(n_rows):
        row = [str(evaluators[0]["details"][i].get("id", i))]
        answer = ""
        for ev in evaluators:
            detail = ev["details"][i]
            row.append(f"{detail.get('score', 0.0):.2f}")
            answer = detail.get("answer", "") or answer
        answer = " ".join(str(answer).split())
        row.append(answer[:120])
        table.add_row(*row)
    console.print(table)


def save_json(report: dict, path: str) -> str:
    """Write ``report`` to ``path`` as UTF-8 JSON. Returns the path written."""
    target = Path(path)
    if target.parent and not target.parent.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    return str(target)
