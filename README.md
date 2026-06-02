# hallucination-eval

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JananiV07/hallucination-eval/blob/main/notebooks/walkthrough.ipynb)
[![tests](https://github.com/JananiV07/hallucination-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/JananiV07/hallucination-eval/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

Measure how much an LLM **hallucinates** with three complementary, reference-based
metrics. Plug in any OpenAI-compatible model (OpenAI, Ollama, vLLM, LM Studio,
Together, Groq, OpenRouter, …), pick a benchmark, and get a scored report.

| Metric | Question it answers | Backend |
| --- | --- | --- |
| **FactScore** | Are the answer's claims consistent with known **facts**? | NLI cross-encoder |
| **FaithScore** | Is every claim grounded in the supplied **context**? | NLI cross-encoder |
| **EntityScore** | Do the answer's named **entities** appear in the source? | spaCy NER |

All three return a score in `[0, 1]` where **higher = less hallucination**.

## Install

```bash
pip install hallucination-eval          # from PyPI (or `pip install -e .` from a clone)
python -m spacy download en_core_web_sm  # required for EntityScore
```

The first run downloads the NLI model `cross-encoder/nli-deberta-v3-small`
(~280 MB) from the HuggingFace Hub.

## Quickstart (CLI)

```bash
# Generate answers with gpt-4o-mini on HaluEval, then score them
export OPENAI_API_KEY=sk-...
hallucination-eval --model gpt-4o-mini --dataset halueval --limit 50

# Score a local open model served by Ollama (OpenAI-compatible)
hallucination-eval --model mistral --dataset halueval --limit 50
hallucination-eval --model gemma2  --dataset halueval --limit 50

# Any custom OpenAI-compatible endpoint + model id
hallucination-eval --model my-llama-3 --base-url http://localhost:8000/v1 --dataset truthfulqa

# Don't call a model — just score the dataset's gold answers (no API key needed)
hallucination-eval --dataset halueval --no-generate --limit 50

# Sanity check: score the dataset's *hallucinated* answers (expect low scores)
hallucination-eval --dataset halueval --use-hallucinated --limit 50 --show-samples

# Save a JSON report and choose a subset of metrics
hallucination-eval --dataset halueval --evaluators fact,entity -o reports/run.json
```

Useful flags: `--split`, `--config`, `--limit`, `--base-url`, `--api-key`,
`--evaluators fact,faith,entity`, `--reference file.json`, `--nli-model`,
`--device cpu|cuda`, `--system-prompt`, `--temperature`, `--max-tokens`,
`--output/-o`, `--show-samples`.

## Quickstart (Python API)

```python
from hallucination_eval import FactScore, FaithScore, EntityScore

context = "The Eiffel Tower is in Paris, France. It was completed in 1889."
question = "Where is the Eiffel Tower?"
answer = "The Eiffel Tower is in Paris and was completed in 1889."

print(FactScore().evaluate(question, context, answer))   # ~1.0  (consistent)
print(FaithScore().evaluate(question, context, answer))  # ~1.0  (grounded)
print(EntityScore().evaluate(question, context, answer)) # ~1.0  (entities grounded)

# Batch scoring with summary stats + per-sample diagnostics
samples = [{"id": "1", "question": question, "context": context,
            "reference": context, "answer": answer}]
print(FactScore().evaluate_batch(samples)["mean"])
```

Each evaluator implements the same interface:

- `evaluate(question, context, answer) -> float` — a single 0–1 score.
- `evaluate_batch(samples: list[dict]) -> dict` — `{mean, min, max, std, count, scores, details}`.

## How the scores work

- **FactScore** splits the answer into atomic claims, runs NLI against a
  reference fact set (the dataset's gold answer by default), and scores each
  claim `1.0` if **entailed**, `0.0` if **contradicted**, and `neutral_weight`
  (default `0.5`) if unverifiable. The score is the mean over claims.
- **FaithScore** runs NLI of each answer sentence against the **context**
  passage (chunked to fit the model). Sentences are `supported` (1.0),
  `contradicted` (0.0) or `unsupported` (neutral_weight). FaithScore needs a
  context — datasets without one (e.g. TruthfulQA) are not meaningful for it.
- **EntityScore** extracts named entities from the source and the answer with
  spaCy and returns the fraction of the answer's entities that appear in the
  source. An answer with no entities scores `1.0`.

The report also prints a **combined faithfulness** (mean of the metric means)
and its complement, the **hallucination score** (`1 − combined`).

## Datasets

- **HaluEval** (`--dataset halueval`, config `qa`) — `knowledge` passage,
  `question`, `right_answer`, `hallucinated_answer`. Great for all three
  metrics and for the `--use-hallucinated` sanity check.
- **TruthfulQA** (`--dataset truthfulqa`, config `generation`) — `question`,
  `best_answer`, `correct_answers`. No context passage, so use **FactScore**
  (`--evaluators fact`).

## Leaderboard

> **Placeholder** — numbers below are illustrative, not measured. Run the CLI
> and open a PR to populate this table.

| Model | Dataset | FactScore ↑ | FaithScore ↑ | EntityScore ↑ | Hallucination ↓ |
| --- | --- | --- | --- | --- | --- |
| gpt-4o-mini | HaluEval (qa) | – | – | – | – |
| gemma2 | HaluEval (qa) | – | – | – | – |
| mistral | HaluEval (qa) | – | – | – | – |
| _your model_ | _your dataset_ | – | – | – | – |

## Development

```bash
pip install -e ".[dev]"
python -m spacy download en_core_web_sm

pytest                 # fast mocked unit tests
pytest -m integration  # real NLI + spaCy models (downloads weights)
```

## License

[MIT](LICENSE)
