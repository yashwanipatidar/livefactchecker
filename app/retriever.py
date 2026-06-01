import math
import re
from collections import Counter

from .models import EvidenceItem

TOKEN_PATTERN = re.compile(r"[a-z0-9']+")


def _tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def _cosine_similarity(left_text: str, right_text: str) -> float:
    left_tokens = Counter(_tokenize(left_text))
    right_tokens = Counter(_tokenize(right_text))

    if not left_tokens or not right_tokens:
        return 0.0

    shared_tokens = left_tokens.keys() & right_tokens.keys()
    dot_product = sum(left_tokens[token] * right_tokens[token] for token in shared_tokens)

    left_norm = math.sqrt(sum(value * value for value in left_tokens.values()))
    right_norm = math.sqrt(sum(value * value for value in right_tokens.values()))

    if not left_norm or not right_norm:
        return 0.0

    return dot_product / (left_norm * right_norm)

def chunk_text(text, chunk_size=500):

    chunks = []

    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])

    return chunks

def retrieve_relevant_context(claim: str, evidence_items: list[EvidenceItem]):

    if not evidence_items:
        return []

    similarities = [
        _cosine_similarity(claim, f"{item.title}\n{item.content}".strip())
        for item in evidence_items
    ]

    ranked_indices = sorted(range(len(similarities)), key=lambda index: similarities[index], reverse=True)
    ranked_items: list[EvidenceItem] = []

    for index in ranked_indices:
        ranked_items.append(
            evidence_items[index].model_copy(update={"relevance_score": float(similarities[index])})
        )

    return ranked_items