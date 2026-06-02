"""Dependency-free text utilities shared by the NLI-based evaluators.

Sentence segmentation is intentionally regex-based (no spaCy) so that
:class:`~hallucination_eval.evaluators.fact_score.FactScore` and
:class:`~hallucination_eval.evaluators.faith_score.FaithScore` do not pull in a
parser. For splitting short model answers into atomic claims a sentence is a
reasonable proxy for a claim, which is all these evaluators need.
"""
from __future__ import annotations

import re

# Split after sentence-ending punctuation when followed by whitespace and a
# capital letter / digit / opening quote. This avoids splitting common
# abbreviations like "U.S." or decimals like "3.14" in most cases.
_SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[A-Z0-9])")
_WS = re.compile(r"\s+")


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into a list of sentence-like atomic claims.

    Returns an empty list for empty/blank input. Newlines and bullet markers
    are treated as additional separators so that list-style answers
    ("- foo\\n- bar") become separate claims. A single clause with no terminal
    punctuation is returned as one claim.
    """
    if not text:
        return []
    text = str(text).strip()
    if not text:
        return []

    sentences: list[str] = []
    for line in re.split(r"[\r\n]+", text):
        line = line.strip()
        if not line:
            continue
        # Strip leading bullet/list markers.
        line = re.sub(r"^[\-\*•‣◦⁃∙]+\s*", "", line)
        for chunk in _SENT_BOUNDARY.split(line):
            chunk = _WS.sub(" ", chunk).strip()
            if chunk:
                sentences.append(chunk)
    return sentences


def chunk_text(text: str, max_chars: int = 1200) -> list[str]:
    """Group ``text`` into chunks no longer than ``max_chars`` characters.

    Used to keep a long context within the NLI cross-encoder's input limit.
    Chunks are formed by accumulating whole sentences; a single sentence longer
    than ``max_chars`` is hard-split as a last resort. Returns ``[]`` for blank
    input.
    """
    if not text:
        return []
    text = _WS.sub(" ", str(text)).strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    current = ""
    for sentence in split_sentences(text) or [text]:
        if current and len(current) + len(sentence) + 1 > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)

    # Hard-split any oversized chunk (e.g. one very long sentence).
    final: list[str] = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            final.append(chunk)
        else:
            final.extend(chunk[i : i + max_chars] for i in range(0, len(chunk), max_chars))
    return final or [text[:max_chars]]
