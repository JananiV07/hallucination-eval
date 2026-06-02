# hallucination-eval — Design Spec

**Date:** 2026-06-02
**Status:** Approved (design), implementation in progress

## Purpose

A Python framework that measures LLM hallucination with three complementary
metrics, plugged into any OpenAI-compatible model. Given a dataset of
question/context/reference samples, it generates answers from the target model
and scores them.

## Decisions (locked)

1. **Verification:** Full install + real run on CPU. Download the real NLI and
   spaCy models and demonstrate scores separating correct vs hallucinated
   answers on real HaluEval samples.
2. **Pipeline:** Generate-then-score. The CLI calls the model to produce an
   answer per question, then runs the evaluators. An OpenAI-compatible
   `model_client` module is added.
3. **FactScore reference:** Derived from the dataset's gold answers
   (HaluEval `right_answer`, TruthfulQA correct answers). An optional
   `--reference` file may override.

## Package layout

Nested under a real package (`hallucination_eval`) to avoid the top-level
`datasets` name collision with the HuggingFace `datasets` library.

```
src/hallucination_eval/
  __init__.py
  evaluators/
    __init__.py        # exports FactScore, FaithScore, EntityScore, BaseEvaluator
    base.py            # BaseEvaluator ABC, lazy model loading, batch summary
    fact_score.py
    faith_score.py
    entity_score.py
  datasets/
    __init__.py
    loader.py          # load_samples(name, split, limit) -> list[Sample dict]
  model_client.py      # OpenAI-compatible client + preset registry
  report.py            # rich table + JSON
  cli.py               # argparse entrypoint -> main()
notebooks/walkthrough.ipynb
tests/
  test_evaluators.py   # mocked unit tests
  test_loader.py
  test_report.py
  test_integration.py  # @pytest.mark.integration, real models
pyproject.toml
README.md
```

CLI installs as `hallucination-eval` via
`[project.scripts] hallucination-eval = "hallucination_eval.cli:main"`.

## Unified sample schema

```python
{
  "id": str,
  "question": str,
  "context": str,          # passage / knowledge ("" if dataset has none)
  "reference": str,        # gold answer used as FactScore reference
  "gold_answer": str,      # same as reference (kept explicit)
  "hallucinated_answer": str | None,  # if dataset provides a negative example
  "answer": str | None,    # filled in by generation, or pre-supplied
}
```

## Scoring algorithms

### FactScore (`fact_score.py`)
NLI via `CrossEncoder("cross-encoder/nli-deberta-v3-small")` (labels:
contradiction / entailment / neutral; resolved from model config, not assumed).

1. Split `answer` into atomic claims (sentence segmentation).
2. For each claim, run NLI(premise=reference_fact, hypothesis=claim) over every
   reference fact; keep the strongest signal.
3. Label the claim: entailed -> 1.0, contradicted -> 0.0, neutral ->
   `neutral_weight` (default 0.5). A claim counts as contradicted when
   max P(contradiction) exceeds `contradiction_threshold` and beats its
   entailment probability.
4. `FactScore = mean(claim_scores)` in [0, 1]. Empty answer -> 0.0.

Constructor args: `model_name`, `neutral_weight`, `contradiction_threshold`,
`entailment_threshold`, `device`.

### FaithScore (`faith_score.py`)
Same NLI model; checks answer claims are grounded in the supplied context.

1. Split `answer` into sentences.
2. premise = context (chunked if longer than the model's max length; take the
   max entailment across chunks), hypothesis = each sentence.
3. Sentence support: entailed -> 1.0, neutral -> `neutral_weight`,
   contradicted -> 0.0.
4. `FaithScore = mean(sentence_support)`. No context -> returns 0.0 with a
   warning in batch mode. Empty answer -> 0.0.

### EntityScore (`entity_score.py`)
spaCy `en_core_web_sm` NER (lazy-loaded; clear error with the
`python -m spacy download en_core_web_sm` hint if missing).

1. Extract entities from `context` and from `answer`; normalize
   (lowercase, strip, collapse whitespace).
2. `EntityScore = |answer_entities ∩ source_entities| / |answer_entities|`.
3. No entities in the answer -> 1.0 (nothing to hallucinate).
   Optional `labels` filter and `match_mode` (exact|substring).

## BaseEvaluator interface

```python
class BaseEvaluator(ABC):
    name: str
    def evaluate(self, question, context, answer, **kwargs) -> float: ...
    def evaluate_batch(self, samples: list[dict]) -> dict:
        # -> {"name", "scores": [..], "mean", "min", "max", "std",
        #     "count", "details": [{sample fields + score + diagnostics}]}
```

Models load lazily on first `evaluate` and are cached on the instance, so
importing a module never triggers a multi-GB download.

## model_client.py

`ModelClient(model, base_url=None, api_key=None, ...)` wrapping the `openai`
SDK. Preset registry:

| name        | base_url                          | key env          |
|-------------|-----------------------------------|------------------|
| gpt-4o-mini | https://api.openai.com/v1         | OPENAI_API_KEY   |
| gemma2      | http://localhost:11434/v1 (Ollama)| (none, optional) |
| mistral     | http://localhost:11434/v1 (Ollama)| (none, optional) |
| custom      | user-provided `--base-url`        | `--api-key`/env  |

`generate(question, context=None, system_prompt=None) -> str` with retry/backoff
and a clear error if no key/endpoint is reachable.

## report.py
- `build_report(run_meta, per_evaluator_results) -> dict`
- `render_report(report)` -> `rich` summary table (mean/min/max/std/count per
  evaluator) + optional per-sample table; prints a combined hallucination score
  (`1 - mean(metric means)`).
- `save_json(report, path)`.

## cli.py
`argparse`. Flags: `--model --dataset --split --limit --base-url --api-key
--evaluators --reference --output --no-generate --system-prompt --device`.
Flow: load samples -> (generate answers | use gold) -> run evaluators ->
build report -> render -> save JSON.

## Testing
- Mocked unit tests: monkeypatch `CrossEncoder`, spaCy `nlp`, and the client so
  CI is fast and deterministic.
- `test_integration.py` (`@pytest.mark.integration`): loads the real NLI +
  spaCy models and asserts FactScore/FaithScore/EntityScore rank a correct
  HaluEval answer above its hallucinated counterpart. No LLM API key needed.

## Out of scope (YAGNI)
- Training/fine-tuning NLI models.
- A web UI or server.
- Caching model outputs to disk beyond HF's own cache.
- Non-OpenAI-compatible providers (Anthropic native, etc.) — reachable via any
  OpenAI-compatible proxy.
