from __future__ import annotations

import logging
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from .models import EvidenceItem

logger = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"

_embedding_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """
    Load embedding model once.
    """
    global _embedding_model

    if _embedding_model is None:
        logger.info(
            "Loading embedding model: %s",
            EMBEDDING_MODEL_NAME,
        )

        _embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME
        )

    return _embedding_model


def chunk_text(
    text: str,
    chunk_size: int = 400,
    overlap: int = 100,
) -> List[str]:
    """
    Split text into overlapping chunks.

    Example:
        chunk_size = 400
        overlap = 100

        chunk1 = 0-400
        chunk2 = 300-700
        chunk3 = 600-1000
    """

    if not text:
        return []

    text = text.strip()

    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray,
) -> float:

    denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(a, b) / denominator
    )


def embed_texts(
    texts: List[str],
) -> np.ndarray:

    model = get_embedding_model()

    return model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def build_chunk_candidates(
    evidence_items: List[EvidenceItem],
    chunk_size: int = 400,
    overlap: int = 100,
) -> List[EvidenceItem]:
    """
    Convert evidence documents into chunk-level evidence.
    """

    candidates: List[EvidenceItem] = []

    for item in evidence_items:

        combined_text = (
            f"{item.title}\n\n{item.content}"
        ).strip()

        chunks = chunk_text(
            combined_text,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for chunk in chunks:

            candidates.append(
                EvidenceItem(
                    title=item.title,
                    content=chunk,
                    url=item.url,
                    domain=item.domain,
                )
            )

    return candidates


def deduplicate_chunks(
    chunks: List[EvidenceItem],
) -> List[EvidenceItem]:
    """
    Remove identical chunks.
    """

    seen = set()

    unique_chunks = []

    for chunk in chunks:

        key = (
            chunk.url,
            chunk.content[:200],
        )

        if key in seen:
            continue

        seen.add(key)

        unique_chunks.append(chunk)

    return unique_chunks


def retrieve_relevant_context(
    claim: str,
    evidence_items: List[EvidenceItem],
    top_k: int = 5,
    chunk_size: int = 400,
    overlap: int = 100,
) -> List[EvidenceItem]:
    """
    Semantic retrieval pipeline.

    Claim
      ↓
    Chunk Evidence
      ↓
    BGE Embeddings
      ↓
    Similarity Ranking
      ↓
    Top K Chunks
    """

    if not evidence_items:
        return []

    try:

        candidates = build_chunk_candidates(
            evidence_items,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        candidates = deduplicate_chunks(
            candidates
        )

        if not candidates:
            return []

        logger.info(
            "Created %d chunks",
            len(candidates),
        )

        claim_embedding = embed_texts(
            [claim]
        )[0]

        chunk_texts = [
            chunk.content
            for chunk in candidates
        ]

        chunk_embeddings = embed_texts(
            chunk_texts
        )

        scored_chunks: List[EvidenceItem] = []

        for item, embedding in zip(
            candidates,
            chunk_embeddings,
        ):

            similarity = cosine_similarity(
                claim_embedding,
                embedding,
            )

            scored_chunks.append(
                item.model_copy(
                    update={
                        "relevance_score": similarity
                    }
                )
            )

        scored_chunks.sort(
            key=lambda x: x.relevance_score,
            reverse=True,
        )

        selected = scored_chunks[:top_k]

        logger.info(
            "Selected top %d chunks",
            len(selected),
        )

        return selected

    except Exception:

        logger.exception(
            "Semantic retrieval failed"
        )

        return []