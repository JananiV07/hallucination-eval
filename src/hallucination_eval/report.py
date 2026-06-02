"""Reporting: a rich terminal table plus a machine-readable JSON report.

``build_report`` aggregates the per-evaluator batch results into a single
report dict; ``render_report`` prints a coloured ``rich`` summary (and an
optional per-sample table); ``save_json`` persists the full report.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


def _label_counts(result: dict) -> dict:
    """Tally per-claim / per-sentence / per-entity labels across a batch result."""
    counts: dict[str, int] = {}
    for detail in result.get("details", []):
        if "claims" in detail:
            for claim in detail["claims"]:
                counts[claim["label"]] = counts.get(claim["label"], 0) + 1
        elif "sentences" in detail:
            for sentence in detail["sentences"]:
                counts[sentence["label"]] = counts.get(sentence["label"], 0) + 1
        elif "matched" in detail or "missing" in detail:
            counts["grounded"] = counts.get("grounded", 0) + len(detail.get("matched", []))
            counts["ungrounded"] = counts.get("ungrounded", 0) + len(detail.get("missing", []))
    return counts


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
        stats["labels"] = _label_counts(result)
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

    breakdown = [
        f"{name}: " + ", ".join(f"{label} {n}" for label, n in stats["labels"].items())
        for name, stats in report.get("metrics", {}).items()
        if stats.get("labels")
    ]
    if breakdown:
        console.print("[dim]labels — " + "  |  ".join(breakdown) + "[/dim]")

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


_CSV_INJECTION_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def _csv_safe(value) -> str:
    """Neutralise spreadsheet formula injection by prefixing risky cells with ``'``."""
    text = str(value)
    if text[:1] in _CSV_INJECTION_PREFIXES:
        return "'" + text
    return text


def save_csv(report: dict, path: str) -> str:
    """Write per-sample scores to ``path`` as CSV: ``id, <metric scores...>, answer``.

    Text cells (``id``, ``answer``) are sanitised against CSV/formula injection;
    numeric score cells are written as-is.
    """
    import csv

    target = Path(path)
    if target.parent and not target.parent.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
    evaluators = report.get("evaluators", [])
    metric_names = [ev["name"] for ev in evaluators]
    n_rows = len(evaluators[0]["details"]) if evaluators else 0
    with open(target, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", *metric_names, "answer"])
        for i in range(n_rows):
            row = [_csv_safe(evaluators[0]["details"][i].get("id", i))]
            answer = ""
            for ev in evaluators:
                detail = ev["details"][i]
                row.append(round(float(detail.get("score", 0.0)), 4))
                answer = detail.get("answer", "") or answer
            row.append(_csv_safe(" ".join(str(answer).split())))
            writer.writerow(row)
    return str(target)


def render_comparison(named_reports: dict, console=None) -> None:
    """Render a side-by-side comparison of several reports (``name -> report``).

    Best mean per metric is highlighted; combined faithfulness and hallucination
    score are shown in a separate section.
    """
    from rich import box
    from rich.console import Console
    from rich.table import Table

    console = console or Console()
    names = list(named_reports)
    metric_names: list[str] = []
    for report in named_reports.values():
        for metric in report.get("metrics", {}):
            if metric not in metric_names:
                metric_names.append(metric)

    table = Table(title="Model comparison", box=box.ROUNDED)
    table.add_column("Metric", style="cyan", no_wrap=True)
    for name in names:
        table.add_column(name, justify="right")

    for metric in metric_names:
        values: dict[str, float] = {}
        for name in names:
            stats = named_reports[name].get("metrics", {}).get(metric)
            if stats and stats.get("applicable", True) and stats.get("mean") is not None:
                values[name] = float(stats["mean"])
        best = max(values.values(), default=None)
        row = [metric]
        for name in names:
            if name in values:
                text = f"{values[name]:.3f}"
                if best is not None and abs(values[name] - best) < 1e-9:
                    text = f"[green]{text}[/green]"
                row.append(text)
            else:
                row.append("[dim]n/a[/dim]")
        table.add_row(*row)

    table.add_section()
    for label, key in (("combined_faithfulness", "combined_faithfulness"), ("hallucination_score", "hallucination_score")):
        cells = []
        for name in names:
            value = named_reports[name].get(key)
            cells.append(f"{value:.3f}" if isinstance(value, (int, float)) else "n/a")
        table.add_row(label, *cells)
    console.print(table)
