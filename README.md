<div align="center">

# 🧪 hallucination-eval

### Catch your LLM making things up — three reference-grounded metrics, any OpenAI-compatible model, one command.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JananiV07/hallucination-eval/blob/main/notebooks/walkthrough.ipynb)
[![tests](https://github.com/JananiV07/hallucination-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/JananiV07/hallucination-eval/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![NLI: DeBERTa](https://img.shields.io/badge/NLI-deberta--v3--small-8A2BE2.svg)](https://huggingface.co/cross-encoder/nli-deberta-v3-small)

</div>

---

**hallucination-eval** scores how much a language model *makes things up*, using three
complementary, **reference-grounded** metrics instead of an expensive, biased
LLM-as-a-judge. Point it at any OpenAI-compatible endpoint (OpenAI, **Google Gemini**,
Ollama, vLLM, Together, Groq, OpenRouter…), pick a benchmark, and get a scored report —
in the terminal, as JSON, and as CSV.

```bash
pip install hallucination-eval && python -m spacy download en_core_web_sm
hallucination-eval --model gemini-2.5-flash --dataset halueval --limit 50 --show-samples
```

## What it does

```mermaid
flowchart LR
    A[Dataset<br/>HaluEval · TruthfulQA · SQuAD] --> B{generate?}
    B -- "any OpenAI-compatible model" --> C[Model answer]
    B -- "--no-generate" --> C
    C --> D[FactScore<br/><i>NLI vs. facts</i>]
    C --> E[FaithScore<br/><i>NLI vs. context</i>]
    C --> F[EntityScore<br/><i>NER vs. source</i>]
    D --> G[Report<br/>rich table · JSON · CSV]
    E --> G
    F --> G
    G --> H[Combined faithfulness<br/>& hallucination score]
```

## The three lenses on hallucination

| Metric | The question it answers | Backend | Needs |
| --- | --- | --- | --- |
| **FactScore** | Are the answer's claims **consistent with known facts**? | NLI cross-encoder | a reference answer |
| **FaithScore** | Is every claim **grounded in the provided context**? | NLI cross-encoder | a context passage |
| **EntityScore** | Do the answer's named **entities appear in the source**? | spaCy NER | a source passage |

Every metric returns a score in `[0, 1]` where **higher = less hallucination**.

```mermaid
mindmap
  root((hallucination))
    FactScore
      contradicts known facts
      unverifiable claims
    FaithScore
      ungrounded additions
      contradicts the context
    EntityScore
      fabricated people
      fabricated places
      fabricated dates
```

## Why hallucination-eval?

- 🎯 **Reference-grounded, not LLM-as-judge.** Deterministic, cheap, reproducible, and free of judge-model bias — the verdict comes from a ~184 MB NLI model and spaCy NER, not another API call.
- 🧭 **Three signals, not one.** Factual consistency, contextual faithfulness, and entity grounding catch *different* failure modes.
- 🔌 **Plug in any model.** One `--base-url` away from OpenAI, Gemini, or a local Ollama/vLLM server. Gemini and Ollama presets are built in.
- 🛡️ **Applicability-aware scoring.** A metric that can't apply to a dataset (e.g. FaithScore on context-free TruthfulQA) is shown as `n/a` and **excluded from the combined score** — so it never silently drags the headline number down.
- ⚡ **Fast & honest.** Batched NLI inference + an on-disk score cache; ships **real measured numbers** and documents its **limitations** instead of hiding them.

### hallucination-eval vs. the alternatives

| Approach | Cost / sample | Deterministic | Bias | What it catches |
| --- | --- | --- | --- | --- |
| **hallucination-eval** | ~free (local NLI/NER) | ✅ yes | none | facts · faithfulness · entities |
| LLM-as-a-judge | 💸 an API call each | ❌ no | judge-model bias | broad, but opaque & pricey |
| Manual review | 🕒 human minutes | ✅ yes | human | everything, but doesn't scale |

## Install

```bash
pip install hallucination-eval          # or: pip install -e .  (from a clone)
python -m spacy download en_core_web_sm  # required for EntityScore
```

The first scoring run downloads the NLI model
[`cross-encoder/nli-deberta-v3-small`](https://huggingface.co/cross-encoder/nli-deberta-v3-small)
(~184 MB) from the HuggingFace Hub.

## 60-second quickstart

**CLI — generate answers with a model, then score them:**

```bash
# Google Gemini (uses GEMINI_API_KEY) — built-in preset
export GEMINI_API_KEY=...
hallucination-eval --model gemini-2.5-flash --dataset halueval --limit 50 -o run.json --csv run.csv

# OpenAI
export OPENAI_API_KEY=sk-...
hallucination-eval --model gpt-4o-mini --dataset halueval --limit 50

# A local open model via Ollama (OpenAI-compatible) — built-in presets
hallucination-eval --model mistral --dataset halueval
hallucination-eval --model gemma2  --dataset halueval

# Any custom endpoint + model id
hallucination-eval --model my-llama-3 --base-url http://localhost:8000/v1 --dataset squad

# No API key needed — score the dataset's own gold / hallucinated answers
hallucination-eval --dataset halueval --no-generate
hallucination-eval --dataset halueval --use-hallucinated --show-samples

# Compare two saved runs side by side
hallucination-eval --compare run_a.json run_b.json
```

**Python API:**

```python
from hallucination_eval import FactScore, FaithScore, EntityScore

context = "The Eiffel Tower is in Paris, France. It was completed in 1889."
question = "Where is the Eiffel Tower?"
answer = "The Eiffel Tower is in Paris and was completed in 1889."

FactScore().evaluate(question, context, answer)    # ~1.0  (consistent)
FaithScore().evaluate(question, context, answer)   # ~1.0  (grounded)
EntityScore().evaluate(question, context, answer)  # ~1.0  (entities grounded)

# Batch scoring -> {mean, min, max, std, count, scores, details, applicable}
FactScore().evaluate_batch([
    {"id": "1", "question": question, "context": context, "reference": context, "answer": answer}
])["mean"]
```

### Sample terminal output

```text
╭──────────────── hallucination-eval ────────────────╮
│ Model: gemini-flash-latest                          │
│ Dataset: halueval   Samples: 10                     │
│ Answers: generated by gemini-flash-latest           │
╰─────────────────────────────────────────────────────╯
            Evaluation summary
┏━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━┓
┃ Metric       ┃  Mean ┃   Min ┃   Max ┃   Std ┃ N ┃
┡━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━┩
│ fact_score   │ 0.450 │ 0.000 │ 1.000 │ 0.284 │ 10│
│ faith_score  │ 0.825 │ 0.500 │ 1.000 │ 0.214 │ 10│
│ entity_score │ 0.827 │ 0.000 │ 1.000 │ 0.281 │ 10│
└──────────────┴───────┴───────┴───────┴───────┴───┘
Combined faithfulness: 0.701    Hallucination score: 0.299
labels — fact_score: neutral 10, contradicted 1
```

## 📊 Leaderboard

> Models generated answers on **HaluEval (qa)**, which were then scored. `↑` higher is better; `↓` lower is better.
> Reproduce with `hallucination-eval --model <m> --dataset halueval`.
> A small **N=10** demo run measured by this repo — rerun with a larger `--limit` for stable numbers, and see [Caveats](#caveats-read-before-trusting-a-number) for how to read FaithScore.

| Model | N | FactScore ↑ | FaithScore ↑ | EntityScore ↑ | Hallucination ↓ |
| --- | --: | --: | --: | --: | --: |
| gemini-flash-latest | 10 | 0.45 | **0.82** | **0.83** | **0.30** |
| gemini-2.5-flash | 10 | **0.53** | 0.50 | 0.70 | 0.43 |
| gemini-flash-lite-latest | 10 | 0.38 | 0.68 | 0.65 | 0.43 |

```mermaid
xychart-beta
    title "Hallucination score by model — lower is better"
    x-axis ["flash-latest", "2.5-flash", "flash-lite"]
    y-axis "Hallucination" 0 --> 1
    bar [0.30, 0.43, 0.43]
```

<sub>Want your model here? Run the CLI and open a PR with the JSON report.</sub>

## Anatomy of a run

```mermaid
sequenceDiagram
    actor You
    participant CLI as cli.py
    participant Loader as datasets/loader
    participant Model as model_client
    participant Eval as evaluators
    participant Report as report

    You->>CLI: hallucination-eval --model … --dataset …
    CLI->>Loader: load_samples(dataset, limit)
    Loader-->>CLI: normalized samples
    loop per sample (generate mode)
        CLI->>Model: generate(question, context)
        Model-->>CLI: answer
    end
    CLI->>Eval: evaluate_batch(samples)
    Note over Eval: NLI + NER, batched & cached
    Eval-->>CLI: scores + per-claim diagnostics
    CLI->>Report: build_report → render
    Report-->>You: table · JSON · CSV
```

## How the scores work

All three split the answer into atomic claims/sentences and score each, then average.

### FactScore — factual consistency vs. a reference

For each claim, run NLI against every reference fact and keep the strongest signal:

```
score(claim) = 1.0   if entailed      (max P(entail) ≥ τ_e  and  ≥ max P(contra))
             = 0.0   if contradicted  (max P(contra) ≥ τ_c  and  >  max P(entail))
             = w     otherwise         (neutral / unverifiable, default w = 0.5)

FactScore = mean over claims
```

```mermaid
flowchart TD
    C[Answer claim] --> N[NLI vs. each reference fact]
    N --> D{strongest signal?}
    D -- entailed --> S1["1.0 ✅ supported"]
    D -- contradicted --> S0["0.0 ❌ contradicted"]
    D -- neither --> SW["w ⚠️ neutral / unverifiable"]
```

**Worked example** — `answer = "Paris is the capital. London is in France."`

```mermaid
flowchart LR
    A["answer"] --> B[split into claims]
    B --> C1["Paris is the capital"]
    B --> C2["London is in France"]
    C1 --> E1["entailed → 1.0 ✅"]
    C2 --> E2["contradicted → 0.0 ❌"]
    E1 --> M["FactScore = mean(1.0, 0.0) = 0.50"]
    E2 --> M
```

### FaithScore — faithfulness vs. the context

Same NLI engine, but the **premise is the context passage** (chunked to fit the model) and the hypothesis is each answer sentence. Sentences are `supported` (1.0), `contradicted` (0.0), or `unsupported` (w). Catches claims the model added that the passage never backs.

### EntityScore — entity grounding vs. the source

```
EntityScore = |entities(answer) ∩ entities(source)| / |entities(answer)|
```

spaCy NER on both texts; an answer with no entities scores `1.0` (nothing to fabricate).

The report also prints a **combined faithfulness** (mean of the *applicable* metric means) and its complement, the **hallucination score** (`1 − combined`).

## Datasets

| Dataset | `--dataset` | Context? | Best for |
| --- | --- | --- | --- |
| **HaluEval** (qa) | `halueval` | ✅ `knowledge` passage | all three metrics; `--use-hallucinated` sanity check |
| **TruthfulQA** (generation) | `truthfulqa` | ❌ none | **FactScore** (`--evaluators fact`) |
| **SQuAD** v1.1 | `squad` | ✅ Wikipedia passage | **FaithScore** & EntityScore on grounded answers |

## Features

- **Built-in model presets** — `gpt-4o-mini`, `gemini-2.5-flash`, `gemini-flash-latest`, `gemini-flash-lite-latest`, `gemma2`, `mistral`, plus any `--model <id> --base-url <url>`.
- **Batched NLI** — all of a sample's claim×reference (or sentence×chunk) pairs go through the cross-encoder in one call.
- **Score cache** — `--cache cache.json` memoises NLI results across runs.
- **Outputs** — coloured terminal table, `--output report.json`, `--csv scores.csv` (formula-injection safe), per-label breakdowns, `--show-samples`.
- **`--compare a.json b.json …`** — side-by-side model comparison with the best score per metric highlighted.
- **Applicability-aware** — context-dependent metrics are excluded from the combined score on context-free datasets.

## Architecture

```mermaid
flowchart TD
    CLI[cli.py] --> L[datasets/loader.py]
    CLI --> MC[model_client.py<br/>OpenAI-compatible]
    CLI --> EV[evaluators/]
    CLI --> R[report.py]
    EV --> FS[FactScore]
    EV --> FA[FaithScore]
    EV --> EN[EntityScore]
    FS --> NLI[_nli.py · NLIScorer<br/>lazy · cached · batched]
    FA --> NLI
    EN --> SP[spaCy en_core_web_sm]
    NLI --> ST[sentence-transformers<br/>CrossEncoder]
```

### Class design

```mermaid
classDiagram
    class BaseEvaluator {
      <<abstract>>
      +str name
      +evaluate(question, context, answer) float
      +evaluate_batch(samples) dict
      #_batch_applicable(samples) bool
    }
    class FactScore
    class FaithScore
    class EntityScore
    class NLIScorer {
      +classify(pairs) list
      +save_cache()
      -dict cache
    }
    class ModelClient {
      +generate(question, context) str
    }
    BaseEvaluator <|-- FactScore
    BaseEvaluator <|-- FaithScore
    BaseEvaluator <|-- EntityScore
    FactScore ..> NLIScorer : shares
    FaithScore ..> NLIScorer : shares
    EntityScore ..> spaCy : uses
```

```
src/hallucination_eval/
├── evaluators/{base,fact_score,faith_score,entity_score}.py
├── _nli.py          # shared NLI cross-encoder: lazy load, label resolution, cache
├── _text.py         # sentence splitting / context chunking
├── model_client.py  # OpenAI-compatible client + presets
├── datasets/loader.py
├── report.py        # rich table · JSON · CSV · comparison
└── cli.py
```

## CLI reference

| Flag | Description |
| --- | --- |
| `--model` | Preset name or model id (default `gpt-4o-mini`) |
| `--dataset` | `halueval` · `truthfulqa` · `squad` |
| `--split` / `--config` / `--limit` | Dataset split, config, and sample cap |
| `--base-url` / `--api-key` | Endpoint + key overrides (else from env) |
| `--evaluators` | Subset of `fact,faith,entity` |
| `--reference FILE` | Override FactScore references from a JSON file |
| `--nli-model` / `--device` | NLI cross-encoder id / torch device |
| `--system-prompt` / `--temperature` / `--max-tokens` | Generation controls |
| `--no-generate` / `--use-hallucinated` | Score gold / known-bad answers |
| `--output/-o` / `--csv` / `--cache` | JSON report / CSV / NLI cache paths |
| `--compare R1 R2 …` | Compare saved JSON reports and exit |
| `--show-samples` | Print a per-sample score table |

## FAQ

**Does it need a GPU?** No — it runs comfortably on CPU. Pass `--device cuda` if you have one.

**Which metric should I trust most?** It depends on the data: with a grounding passage, lean on FaithScore/EntityScore; for free-form factual QA, FactScore. The combined score blends only the *applicable* ones.

**Can I evaluate my own model?** Yes — any OpenAI-compatible endpoint via `--model <id> --base-url <url>`, or score answers you already have with `--no-generate`.

**Why is FaithScore sometimes low on HaluEval?** Its gold answers are terse entities the NLI model rates *neutral*. FaithScore is designed for propositional, RAG-style answers — see Caveats.

**Does it phone home / can it run offline?** After the one-time model download it runs fully offline (scoring needs no network; only generation calls your chosen endpoint).

## Development

```bash
pip install -e ".[dev]" && python -m spacy download en_core_web_sm
pytest                 # fast mocked unit tests (CI runs these on every push)
pytest -m integration  # real NLI + spaCy models (downloads weights; weekly CI)
```

CI: [`tests`](.github/workflows/ci.yml) on every push (Python 3.11 & 3.12),
[`integration`](.github/workflows/integration.yml) weekly, and
[`release`](.github/workflows/release.yml) publishes to PyPI on a `v*` tag via Trusted Publishing.

## Caveats (read before trusting a number)

- **FaithScore needs propositional answers.** On HaluEval-qa the gold answers are terse entities (e.g. `"Arthur's Magazine"`) that NLI rates *neutral*, so FaithScore is noisy there — it shines on RAG-style answers and SQuAD. EntityScore and FactScore separate good from hallucinated cleanly on HaluEval.
- **The NLI model is small** (`deberta-v3-small`) for speed; swap a larger one with `--nli-model` for more accuracy.
- **Scores are only as good as the reference.** Garbage references → garbage FactScore.
- These are **automatic proxies** for hallucination, not ground truth. Use them to compare models and catch regressions, not as an absolute verdict.

## License

[MIT](LICENSE)
