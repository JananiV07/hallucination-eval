"""Load hallucination benchmarks from the HuggingFace Hub into one schema.

Supported datasets
------------------
* **HaluEval** (``pminervini/HaluEval``, config ``qa`` by default) - each row
  has a ``knowledge`` passage, a ``question``, a ``right_answer`` and a
  ``hallucinated_answer``. The knowledge passage is a natural context for
  FaithScore/EntityScore and the right answer is the FactScore reference.
* **TruthfulQA** (``truthful_qa``, config ``generation``) - each row has a
  ``question``, a ``best_answer`` and lists of ``correct_answers`` /
  ``incorrect_answers``. TruthfulQA has *no* grounding passage, so ``context``
  is empty; FactScore still works (reference = the correct answers) but
  FaithScore/EntityScore are not meaningful for it.

Normalised sample schema
-------------------------
Every loader returns ``list[dict]`` with::

    {
      "id": str,
      "question": str,
      "context": str,                 # "" when the dataset has no passage
      "reference": str | list[str],   # gold fact(s) for FactScore
      "gold_answer": str,             # the canonical correct answer
      "hallucinated_answer": str|None,# a known-bad answer, when provided
      "answer": None,                 # filled in later (generated or gold)
    }

The HuggingFace ``datasets`` import is local to each loader so that importing
this module is cheap and does not require ``datasets`` to be installed unless a
load is actually requested.
"""
from __future__ import annotations

import warnings
from typing import Optional

SUPPORTED_DATASETS = ("halueval", "truthfulqa", "squad")


def _norm_name(name: str) -> str:
    return (name or "").strip().lower().replace("-", "").replace("_", "").replace(" ", "")


def _first_nonempty(row: dict, keys: list[str]) -> Optional[str]:
    for key in keys:
        value = row.get(key)
        if value:
            return value
    return None


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _resolve_split(load_dataset, repo: str, config: str, split: Optional[str], default_split: str):
    """Load a split, falling back to the first available one only if missing.

    Only a missing-split error (``ValueError``/``KeyError`` from ``datasets``)
    triggers the fallback; network/auth/build errors propagate so the caller
    sees the real cause. A warning is emitted if the split actually used differs
    from the one requested.
    """
    target = split or default_split
    try:
        return load_dataset(repo, config, split=target)
    except (ValueError, KeyError):
        dataset_dict = load_dataset(repo, config)
        if target in dataset_dict:
            return dataset_dict[target]
        chosen = next(iter(dataset_dict))
        warnings.warn(
            f"split '{target}' not found for {repo}/{config}; using '{chosen}' instead."
        )
        return dataset_dict[chosen]


def load_samples(
    name: str,
    split: Optional[str] = None,
    limit: Optional[int] = None,
    config: Optional[str] = None,
) -> list[dict]:
    """Load and normalise a supported dataset.

    Parameters
    ----------
    name: ``"halueval"`` or ``"truthfulqa"`` (case/separator insensitive).
    split: dataset split; sensible per-dataset default when ``None``.
    limit: cap on the number of samples returned.
    config: dataset config/subset (e.g. HaluEval ``"qa"``).
    """
    key = _norm_name(name)
    if key in ("halueval", "halu"):
        return _load_halueval(split, limit, config)
    if key in ("truthfulqa", "truthful"):
        return _load_truthfulqa(split, limit, config)
    if key in ("squad", "squadv1", "squad11"):
        return _load_squad(split, limit, config)
    raise ValueError(
        f"Unknown dataset '{name}'. Supported datasets: {', '.join(SUPPORTED_DATASETS)}."
    )


def _load_halueval(split, limit, config) -> list[dict]:
    from datasets import load_dataset

    config = config or "qa"
    dataset = _resolve_split(load_dataset, "pminervini/HaluEval", config, split, "data")

    samples: list[dict] = []
    for i, row in enumerate(dataset):
        if limit is not None and i >= limit:
            break
        question = _first_nonempty(row, ["question", "user_query", "query"]) or ""
        context = _first_nonempty(
            row, ["knowledge", "context", "document", "dialogue_history"]
        ) or ""
        right = _first_nonempty(row, ["right_answer", "answer", "ground_truth"]) or ""
        hallucinated = _first_nonempty(row, ["hallucinated_answer"])
        samples.append(
            {
                "id": str(row.get("id", f"halueval-{config}-{i}")),
                "question": str(question),
                "context": str(context),
                "reference": str(right),
                "gold_answer": str(right),
                "hallucinated_answer": str(hallucinated) if hallucinated else None,
                "answer": None,
            }
        )
    return samples


def _load_truthfulqa(split, limit, config) -> list[dict]:
    from datasets import load_dataset

    config = config or "generation"
    dataset = _resolve_split(load_dataset, "truthful_qa", config, split, "validation")

    samples: list[dict] = []
    for i, row in enumerate(dataset):
        if limit is not None and i >= limit:
            break
        question = row.get("question", "") or ""
        best = row.get("best_answer") or ""
        correct = list(row.get("correct_answers") or [])
        incorrect = list(row.get("incorrect_answers") or [])
        reference = _dedupe([best, *correct])
        samples.append(
            {
                "id": f"truthfulqa-{config}-{i}",
                "question": str(question),
                "context": "",  # TruthfulQA provides no grounding passage
                "reference": reference,
                "gold_answer": str(best),
                "hallucinated_answer": str(incorrect[0]) if incorrect else None,
                "answer": None,
                "source": row.get("source", ""),
            }
        )
    return samples


def _load_squad(split, limit, config) -> list[dict]:
    """SQuAD v1.1: a Wikipedia passage (context) + question + gold answer span(s).

    A context-rich benchmark, so all three metrics apply - especially FaithScore,
    which is most meaningful when there is a real passage to ground against.
    """
    from datasets import load_dataset

    dataset = _resolve_split(load_dataset, "rajpurkar/squad", config, split, "validation")
    samples: list[dict] = []
    for i, row in enumerate(dataset):
        if limit is not None and i >= limit:
            break
        answers = row.get("answers") or {}
        texts = list(answers.get("text") or [])
        gold = texts[0] if texts else ""
        samples.append(
            {
                "id": str(row.get("id", f"squad-{i}")),
                "question": str(row.get("question", "")),
                "context": str(row.get("context", "")),
                "reference": _dedupe(texts) or [gold],
                "gold_answer": str(gold),
                "hallucinated_answer": None,
                "answer": None,
            }
        )
    return samples
