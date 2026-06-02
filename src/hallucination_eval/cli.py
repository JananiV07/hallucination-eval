"""Command-line interface: ``hallucination-eval``.

Typical usage::

    hallucination-eval --model gpt-4o-mini --dataset halueval --limit 50
    hallucination-eval --model mistral --base-url http://localhost:11434/v1 \\
        --dataset truthfulqa --evaluators fact
    hallucination-eval --dataset halueval --no-generate          # score gold answers
    hallucination-eval --dataset halueval --use-hallucinated     # sanity check (low scores)

Flow: load samples -> obtain answers (generate, or use the dataset's gold /
hallucinated answers) -> run the selected evaluators -> render a rich summary ->
optionally save a JSON report.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from ._nli import DEFAULT_NLI_MODEL, NLIScorer
from .datasets.loader import SUPPORTED_DATASETS, load_samples
from .evaluators import EntityScore, FactScore, FaithScore
from .model_client import PRESETS, ModelClient
from .report import build_report, render_comparison, render_report, save_csv, save_json

# CLI evaluator keys -> factory callable(nli_model, device, shared_nli)
_EVALUATOR_KEYS = ("fact", "faith", "entity")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hallucination-eval",
        description="Measure LLM hallucination with FactScore, FaithScore and EntityScore.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help=f"Model preset ({', '.join(PRESETS)}) or any model id used with --base-url.",
    )
    parser.add_argument(
        "--dataset",
        default="halueval",
        help=f"Dataset to evaluate on ({', '.join(SUPPORTED_DATASETS)}).",
    )
    parser.add_argument("--split", default=None, help="Dataset split (default: per-dataset).")
    parser.add_argument("--config", default=None, help="Dataset config/subset (e.g. HaluEval 'qa').")
    parser.add_argument("--limit", type=int, default=20, help="Number of samples to evaluate.")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL override.")
    parser.add_argument("--api-key", default=None, help="API key (else taken from OPENAI_API_KEY).")
    parser.add_argument(
        "--evaluators",
        default="fact,faith,entity",
        help="Comma-separated subset of: fact, faith, entity.",
    )
    parser.add_argument("--reference", default=None, help="Optional JSON file overriding FactScore references.")
    parser.add_argument("--nli-model", default=DEFAULT_NLI_MODEL, help="NLI cross-encoder for Fact/Faith scores.")
    parser.add_argument("--device", default=None, help="Torch device for the NLI model (e.g. cpu, cuda).")
    parser.add_argument("--system-prompt", default=None, help="Override the generation system prompt.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature.")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max tokens to generate per answer.")
    parser.add_argument(
        "--no-generate",
        action="store_true",
        help="Skip generation; score the dataset's gold answers instead.",
    )
    parser.add_argument(
        "--use-hallucinated",
        action="store_true",
        help="Score the dataset's hallucinated_answer (sanity check; expect low scores).",
    )
    parser.add_argument("--output", "-o", default=None, help="Path to write the JSON report.")
    parser.add_argument("--csv", default=None, help="Also write per-sample scores to this CSV path.")
    parser.add_argument("--cache", default=None, help="JSON NLI score cache path (speeds up re-runs).")
    parser.add_argument(
        "--compare",
        nargs="+",
        default=None,
        metavar="REPORT.json",
        help="Compare two or more saved JSON reports side by side, then exit.",
    )
    parser.add_argument("--show-samples", action="store_true", help="Print a per-sample score table.")
    return parser


def _load_reference_overrides(path: str) -> dict:
    """Read a reference file mapping question -> reference fact(s).

    Accepts either a JSON object ``{question: reference}`` or a JSON list of
    ``{"question": ..., "reference": ...}`` objects.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        return {str(k): v for k, v in raw.items()}
    overrides: dict = {}
    for item in raw:
        question = item.get("question")
        if question is not None and "reference" in item:
            overrides[str(question)] = item["reference"]
    return overrides


def _select_answers(samples: list[dict], args, console) -> str:
    """Populate ``sample["answer"]`` and return a label describing the source."""
    if args.use_hallucinated:
        for sample in samples:
            sample["answer"] = sample.get("hallucinated_answer") or ""
        return "dataset hallucinated_answer"
    if args.no_generate:
        for sample in samples:
            sample["answer"] = sample.get("answer") or sample.get("gold_answer") or ""
        return "dataset gold_answer"

    client = ModelClient(
        args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        system_prompt=args.system_prompt,
    )
    from rich.progress import track

    failures = 0
    for sample in track(samples, description=f"Generating with {client.name}", console=console):
        try:
            sample["answer"] = client.generate(sample.get("question", ""), sample.get("context"))
        except Exception as exc:  # noqa: BLE001 - one bad sample shouldn't abort the whole run
            sample["answer"] = ""
            failures += 1
            if failures <= 3:
                console.print(f"[yellow]Generation failed for one sample:[/yellow] {exc}")
    if failures == len(samples):
        raise RuntimeError(f"All {len(samples)} generations failed for model '{client.name}'.")
    if failures:
        console.print(
            f"[yellow]{failures}/{len(samples)} generations failed; scored as empty answers.[/yellow]"
        )
    return f"generated by {client.name}"


def _build_evaluators(keys: list[str], args, console):
    needs_nli = any(key in ("fact", "faith") for key in keys)
    shared_nli = (
        NLIScorer(args.nli_model, device=args.device, cache_path=args.cache) if needs_nli else None
    )
    evaluators = []
    for key in keys:
        if key == "fact":
            evaluators.append(FactScore(model_name=args.nli_model, device=args.device, nli=shared_nli))
        elif key == "faith":
            evaluators.append(FaithScore(model_name=args.nli_model, device=args.device, nli=shared_nli))
        elif key == "entity":
            evaluators.append(EntityScore(device=args.device))
        else:
            console.print(f"[yellow]Warning:[/yellow] ignoring unknown evaluator '{key}'.")
    return evaluators, shared_nli


def _run_compare(paths: list[str], console) -> int:
    """Load saved JSON reports and print a side-by-side comparison."""
    named: dict[str, dict] = {}
    for raw_path in paths:
        try:
            report = json.loads(Path(raw_path).read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Failed to read report '{raw_path}':[/red] {exc}")
            return 1
        label = (report.get("meta") or {}).get("model") or Path(raw_path).stem
        base, k = label, 2
        while label in named:
            label = f"{base} ({k})"
            k += 1
        named[label] = report
    render_comparison(named, console=console)
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(sys.argv[1:] if argv is None else argv)

    from rich.console import Console

    console = Console()

    if args.compare:
        return _run_compare(args.compare, console)

    try:
        samples = load_samples(args.dataset, split=args.split, limit=args.limit, config=args.config)
    except Exception as exc:  # noqa: BLE001 - surface a clean CLI error
        console.print(f"[red]Failed to load dataset '{args.dataset}':[/red] {exc}")
        return 1
    if not samples:
        console.print(f"[red]No samples loaded for dataset '{args.dataset}'.[/red]")
        return 1

    if args.reference:
        try:
            overrides = _load_reference_overrides(args.reference)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Failed to read reference file '{args.reference}':[/red] {exc}")
            return 1
        for sample in samples:
            if sample.get("question") in overrides:
                sample["reference"] = overrides[sample["question"]]

    keys = [key.strip().lower() for key in args.evaluators.split(",") if key.strip()]
    if not keys:
        console.print("[red]No evaluators selected.[/red]")
        return 1
    unknown = [key for key in keys if key not in _EVALUATOR_KEYS]
    if unknown:
        console.print(
            f"[red]Unknown evaluator(s):[/red] {', '.join(unknown)}. "
            f"Choose from: {', '.join(_EVALUATOR_KEYS)}."
        )
        return 1

    # Warn when context-dependent metrics are requested for a dataset without context.
    has_context = any((s.get("context") or "").strip() for s in samples)
    if not has_context and ({"faith", "entity"} & set(keys)):
        console.print(
            "[yellow]Note:[/yellow] this dataset has no context passage, so FaithScore "
            "and EntityScore are not meaningful here (they will score ~0)."
        )

    try:
        answers_label = _select_answers(samples, args, console)
    except Exception as exc:  # noqa: BLE001 - generation/network failure
        console.print(f"[red]Answer generation failed:[/red] {exc}")
        return 1

    evaluators, shared_nli = _build_evaluators(keys, args, console)
    if not evaluators:
        console.print("[red]No valid evaluators to run.[/red]")
        return 1

    results = []
    with console.status("[bold]Scoring answers..."):
        for evaluator in evaluators:
            results.append(evaluator.evaluate_batch(samples))

    meta = {
        "model": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "config": args.config,
        "n_samples": len(samples),
        "answers": answers_label,
        "nli_model": args.nli_model if ({"fact", "faith"} & set(keys)) else None,
        "evaluators": [ev.name for ev in evaluators],
    }
    report = build_report(results, meta)
    render_report(report, console=console, show_samples=args.show_samples)

    if args.output:
        try:
            path = save_json(report, args.output)
        except OSError as exc:  # noqa: BLE001 - clean CLI error instead of a traceback
            console.print(f"[red]Failed to write report to '{args.output}':[/red] {exc}")
            return 1
        console.print(f"[green]Saved JSON report to[/green] {path}")
    if args.csv:
        try:
            csv_path = save_csv(report, args.csv)
        except OSError as exc:  # noqa: BLE001
            console.print(f"[red]Failed to write CSV to '{args.csv}':[/red] {exc}")
            return 1
        console.print(f"[green]Saved CSV to[/green] {csv_path}")
    if shared_nli is not None:
        shared_nli.save_cache()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
