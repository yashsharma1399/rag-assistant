
# answering.py
# Builds a "free" (no-LLM) answer by extracting the most relevant sentences
# from retrieved chunks.

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class RetrievedChunk:
    """
    Represents one retrieved chunk returned by Chroma.
    We keep only what we need for a free extractive answer.
    """
    text: str
    metadata: Dict[str, Any]
    distance: float


def _split_into_sentences(text: str) -> List[str]:
    """
    Very simple sentence splitter.
    Not perfect, but good enough for a beginner project.
    """
    text = text.replace("\n", " ").strip()
    if not text:
        return []

    # Split on . ? ! followed by whitespace
    parts = re.split(r"(?<=[.!?])\s+", text)
    # Remove super short garbage sentences
    return [s.strip() for s in parts if len(s.strip()) >= 25]


def _keywords(question: str) -> List[str]:
    """
    Extract keywords from the question.
    We remove very common stop words so scoring is better.
    """
    stop = {
        "the", "is", "are", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with",
        "what", "how", "why", "when", "where", "this", "that", "it", "about", "paper"
    }
    words = re.findall(r"[a-zA-Z0-9]+", question.lower())
    return [w for w in words if w not in stop and len(w) >= 3]


def _score_sentence(sentence: str, kws: List[str]) -> int:
    """
    Score a sentence by how many question keywords it contains.
    Higher score => more relevant.
    """
    s = sentence.lower()
    score = 0
    for kw in kws:
        if kw in s:
            score += 1
    return score


def build_extractive_answer(question: str, chunks: List[RetrievedChunk], max_bullets: int = 5) -> List[str]:
    """
    Build a list of bullet points as the final answer.
    Each bullet comes from a sentence in one of the retrieved chunks.
    """
    kws = _keywords(question)

    # Collect candidate (score, sentence, citation) tuples
    candidates: List[tuple[int, str, str]] = []

    for ch in chunks:
        # For each retrieved chunk, it builds a citation string.
        pdf_name = ch.metadata.get("pdf_name", "unknown.pdf")
        page_number = ch.metadata.get("page_number", "?")
        citation = f"[{pdf_name} p.{page_number}]"

        # Splits chunk into sentences.
        for sent in _split_into_sentences(ch.text):
            score = _score_sentence(sent, kws) #Scores each sentence.
            # Keep only sentences with at least 1 keyword match (or if no keywords, keep a few)
            if score > 0 or not kws:
                candidates.append((score, sent, citation))

    # Sort best sentences first (score desc)
    candidates.sort(key=lambda x: x[0], reverse=True)

    # Pick top unique-ish bullets (avoid repeating same sentence)
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

    # If we somehow got nothing, return a fallback message
    if not bullets:
        bullets.append("- I couldn't extract a clear answer from the retrieved chunks. Try a more specific question.")

    return bullets