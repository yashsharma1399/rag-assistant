
# answering.py
# Builds a "free" (no-LLM) answer by extracting the most relevant sentences
# from retrieved chunks.

"""
answering.py
Free (no-LLM) extractive answering.

Goal:
- Given a question + retrieved chunks, build a "final answer" as bullet points 
  by extracting the most relevant sentences from those chunks.

Update in this version (fragment repair):
- Sometimes chunks start mid-sentence, producing bullets like:
    "on, legal document analysis..."
- We fix that by "repairing" fragment sentences:
  - If a sentence looks like it started mid-thought, we merge it with the previous sentence from the SAME chunk:
        merged = prev + " " + fragment
  - If there's no previous sentence, we keep it but penalize it slightly (so it needs a stronger keyword match to win).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


# -----------------------------
# Data model for retrieved text
# -----------------------------
@dataclass
class RetrievedChunk:
    """
    Represents one retrieved chunk returned by Chroma.
    We keep only what we need for extractive answering.
    """
    text: str
    metadata: Dict[str, Any]
    distance: float


# -----------------------------
# 1) Sentence splitting
# -----------------------------
def _basic_sentence_split(text: str) -> List[str]:
    """
    Very simple sentence splitter (beginner-friendly).
    Splits on . ? ! followed by whitespace.

    NOTE: This isn't perfect for all PDFs, but good enough for a starter RAG project.
    """
    text = text.replace("\n", " ").strip()
    if not text:
        return []

    parts = re.split(r"(?<=[.!?])\s+", text)
    # keep non-empty
    return [p.strip() for p in parts if p.strip()]


# -----------------------------
# 2) Fragment detection + repair
# -----------------------------
_FRAGMENT_STARTERS = {
    "on", "of", "to", "for", "with", "by", "in", "at", "as", "from", "and", "or", "but"
}


def _starts_with_lowercase(s: str) -> bool:
    """True if the first alphabetic character is lowercase."""
    for ch in s.strip():
        if ch.isalpha():
            return ch.islower()
    return False


def _starts_with_fragment_starter(s: str) -> bool:
    """
    True if sentence begins with a common fragment starter.
    Examples:
      "on, ..." "of ..." "to ..." "and ..." etc.
    """
    s = s.strip().lower()

    # get first token (strip punctuation around it)
    tokens = re.findall(r"[a-zA-Z]+", s)
    if not tokens:
        return False

    first = tokens[0]
    return first in _FRAGMENT_STARTERS


def _is_fragment_sentence(sentence: str) -> bool:
    """
    A sentence is treated as a fragment if it looks like it started mid-thought.

    Rules (as per your request):
    - starts with lowercase letter
      OR
    - starts with common fragment starters like "on", "of", "to", "for", etc.
    """
    if not sentence or len(sentence.strip()) < 5:
        return False

    return _starts_with_lowercase(sentence) or _starts_with_fragment_starter(sentence)


def _repair_fragments(sentences: List[str]) -> List[Tuple[str, bool]]:
    """
    Repairs fragment sentences by merging them with the previous sentence
    from the SAME chunk.

    Returns a list of tuples: (sentence_text, is_fragment_unmerged)
      - sentence_text: repaired or original sentence
      - is_fragment_unmerged: True ONLY when it's a fragment but had no previous sentence
        (so we could not merge it).
    """
    repaired: List[Tuple[str, bool]] = []

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        # If it's a fragment AND we already have a previous sentence, merge into previous.
        if _is_fragment_sentence(sent) and repaired:
            prev_text, prev_flag = repaired[-1]

            # Merge fragment into previous sentence
            merged = f"{prev_text} {sent}".strip()

            # Replace the last entry with merged sentence.
            # Mark it as not "unmerged fragment" (False), because it's repaired now.
            repaired[-1] = (merged, False)
        else:
            # If it's a fragment but there is no previous sentence, keep it and flag it
            # so we can penalize it later.
            is_unmerged_fragment = _is_fragment_sentence(sent) and not repaired
            repaired.append((sent, is_unmerged_fragment))

    return repaired


def _split_into_sentences_with_repair(text: str) -> List[Tuple[str, bool]]:
    """
    Split chunk text into sentences, then repair fragments.

    Output:
      List[(sentence, is_unmerged_fragment)]
    """
    raw = _basic_sentence_split(text)

    # filter out very short "garbage" sentences early
    raw = [s for s in raw if len(s.strip()) >= 25]

    return _repair_fragments(raw)


# -----------------------------
# 3) Keyword extraction + scoring
# -----------------------------
def _keywords(question: str) -> List[str]:
    """
    Extract keywords from the question.

    We remove common stop words so scoring is more meaningful.
    """
    stop = {
        "the", "is", "are", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with",
        "what", "how", "why", "when", "where", "this", "that", "it", "about", "paper"
    }
    words = re.findall(r"[a-zA-Z0-9]+", question.lower())
    return [w for w in words if w not in stop and len(w) >= 3]


def _score_sentence(sentence: str, kws: List[str]) -> int:
    """
    Score a sentence by keyword overlap.
    Higher score => more relevant.
    """
    s = sentence.lower()
    score = 0
    for kw in kws:
        if kw in s:
            score += 1
    return score


# -----------------------------
# 4) Build the final extractive answer
# -----------------------------
def build_extractive_answer(question: str, chunks: List[RetrievedChunk], max_bullets: int = 5) -> List[str]:
    """
    Build bullet points from the retrieved chunks.

    Keeps overall logic the same:
    - keyword extraction
    - scoring by overlap
    - dedup
    - citations

    NEW:
    - sentence splitting now "repairs" fragments by merging them with previous sentence
    - unmerged fragments are penalized slightly unless they score strongly
    """
    kws = _keywords(question)

    # Each candidate: (score, sentence, citation)
    candidates: List[Tuple[int, str, str]] = []

    for ch in chunks:
        pdf_name = ch.metadata.get("pdf_name", "unknown.pdf")
        page_number = ch.metadata.get("page_number", "?")
        citation = f"[{pdf_name} p.{page_number}]"

        # Split into sentences + repair fragments
        repaired_sents = _split_into_sentences_with_repair(ch.text)

        for sent, is_unmerged_fragment in repaired_sents:
            score = _score_sentence(sent, kws)

            # If it’s an unmerged fragment (no previous sentence to attach),
            # apply a small penalty so it doesn't win unless it's clearly relevant.
            if is_unmerged_fragment:
                score -= 1

            # Keep only relevant sentences (score > 0) OR if question has no keywords (rare)
            if score > 0 or not kws:
                candidates.append((score, sent, citation))

    # Sort best first (score desc)
    candidates.sort(key=lambda x: x[0], reverse=True)

    # Deduplicate by exact sentence text (case-insensitive)
    bullets: List[str] = []
    seen = set()

    for score, sent, citation in candidates:
        key = sent.lower()
        if key in seen:
            continue
        seen.add(key)

        bullets.append(f"- {sent} {citation}")

        if len(bullets) >= max_bullets:
            break

    if not bullets:
        bullets.append(
            "- I couldn't extract a clear answer from the retrieved chunks. Try a more specific question."
        )

    return bullets
