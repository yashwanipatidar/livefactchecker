import os
import math
import re
from collections import Counter
from urllib.parse import urlparse
import logging

from dotenv import load_dotenv

from .models import EvidenceItem, VerificationResult


logger = logging.getLogger(__name__)


# Load the project .env explicitly (project root is parent of the app folder)
project_root = os.path.dirname(os.path.dirname(__file__))
dotenv_path = os.path.join(project_root, ".env")
load_dotenv(dotenv_path)

SOURCE_SCORES = {
    "un.org": 0.98,
    "who.int": 0.98,
    "worldbank.org": 0.95,
    "nature.com": 0.95,
    "wikipedia.org": 0.75,
}

DEFAULT_NLI_MODEL = os.getenv("NLI_MODEL_NAME", "microsoft/deberta-v3-large-mnli")
NLI_LABELS = {
    0: "CONTRADICTION",
    1: "NEUTRAL",
    2: "ENTAILMENT",
}

NEGATION_MARKERS = (
    "false",
    "not true",
    "no evidence",
    "refute",
    "contradict",
    "dispute",
    "incorrect",
    "misleading",
    "debunk",
)

TOKEN_PATTERN = re.compile(r"[a-z0-9']+")
_nli_pipeline = None


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


def _extract_domain(url: str) -> str:
    if not url:
        return ""

    parsed = urlparse(url)
    return parsed.netloc.lower().removeprefix("www.")


def _score_credibility(domain: str) -> float:
    if not domain:
        return 0.5

    for trusted_domain, score in SOURCE_SCORES.items():
        if domain == trusted_domain or domain.endswith(f".{trusted_domain}"):
            return score

    return 0.6


def _get_nli_pipeline():
    global _nli_pipeline

    if _nli_pipeline is not None:
        return _nli_pipeline

    try:
        from transformers import pipeline
    except Exception:
        _nli_pipeline = None
        return None

    try:
        _nli_pipeline = pipeline(
            task="text-classification",
            model=DEFAULT_NLI_MODEL,
            tokenizer=DEFAULT_NLI_MODEL,
            top_k=None,
        )
    except Exception:
        _nli_pipeline = None

    return _nli_pipeline


def _normalize_scores(scores: list[dict]) -> dict[str, float]:
    normalized: dict[str, float] = {}

    for item in scores:
        label = str(item.get("label", "")).upper()
        score = float(item.get("score", 0.0))

        if "ENTAIL" in label:
            normalized["entailment"] = score
        elif "CONTRAD" in label:
            normalized["contradiction"] = score
        elif "NEUTRAL" in label:
            normalized["neutral"] = score

    return normalized


def _score_with_nli(claim: str, evidence: EvidenceItem) -> dict:
    # Try Hugging Face Inference API first (remote), then local transformers pipeline, then heuristic.
    evidence_text = f"{evidence.title}\n{evidence.content}".strip()
    logger.debug("Scoring evidence NLI for claim snippet: %s", (claim or '')[:120])

    def _nli_with_hf(claim_text: str, evidence_text: str, model: str | None = None) -> dict | None:
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            logger.debug("HF_TOKEN not set; skipping Hugging Face InferenceClient path")
            return None

        try:
            from huggingface_hub import InferenceClient
        except Exception:
            logger.exception("Failed to import huggingface_hub.InferenceClient; ensure huggingface-hub is installed")
            return None

        try:
            client = InferenceClient(provider="hf-inference", api_key=hf_token)
            # Prefer structured inputs for NLI if supported
            try:
                payload = {"inputs": {"premise": evidence_text, "hypothesis": claim_text}}
                resp = client.text_classification(payload, model=model or DEFAULT_NLI_MODEL)
            except Exception:
                # Fallback to simple concatenated text
                text = f"premise: {evidence_text}\nhypothesis: {claim_text}"
                resp = client.text_classification(text, model=model or DEFAULT_NLI_MODEL)
        except Exception:
            logger.exception("Hugging Face InferenceClient call failed")
            return None

        logger.debug("Hugging Face inference response type=%s", type(resp))
        # Normalize response
        scores_map: dict[str, float] = {}
        if isinstance(resp, list):
            for r in resp:
                if isinstance(r, dict) and "label" in r:
                    scores_map[str(r.get("label", "")).upper()] = float(r.get("score", 0.0))
        elif isinstance(resp, dict) and "label" in resp:
            scores_map[str(resp.get("label", "")).upper()] = float(resp.get("score", 0.0))
        else:
            return None

        entailment_score = scores_map.get("ENTAILMENT", scores_map.get("ENTAIL", 0.0))
        contradiction_score = scores_map.get("CONTRADICTION", scores_map.get("CONTRAD", 0.0))
        neutral_score = scores_map.get("NEUTRAL", 0.0)

        similarity = _cosine_similarity(claim_text, evidence_text)
        credibility_score = _score_credibility(evidence.domain or _extract_domain(evidence.url))

        label = max(scores_map, key=scores_map.get) if scores_map else ""

        return {
            "relevance_score": max(similarity, entailment_score),
            "similarity_score": similarity,
            "entailment_score": entailment_score,
            "contradiction_score": contradiction_score,
            "credibility_score": credibility_score,
            "verification_mode": "hf-inference",
            "nli_label": label,
            "nli_scores": {
                "entailment": entailment_score,
                "contradiction": contradiction_score,
                "neutral": neutral_score,
            },
        }

    # Try HF inference first
    try:
        hf_result = _nli_with_hf(claim, evidence_text)
        if hf_result is not None:
            logger.info("Using Hugging Face Inference API for NLI")
            return hf_result
    except Exception:
        logger.exception("Unexpected error in HF inference path; falling back")

    # Try local transformers pipeline
    nli = _get_nli_pipeline()
    if nli is None:
        logger.info("No local transformers pipeline available; using heuristic fallback")
        return _score_evidence_pair(claim, evidence) | {"verification_mode": "heuristic", "nli_label": "", "nli_scores": {}}

    try:
        raw_scores = nli({"text": claim, "text_pair": evidence_text})
    except Exception:
        logger.exception("Local transformers NLI pipeline failed; falling back to heuristic")
        return _score_evidence_pair(claim, evidence) | {"verification_mode": "heuristic", "nli_label": "", "nli_scores": {}}

    if raw_scores and isinstance(raw_scores, list) and raw_scores and isinstance(raw_scores[0], list):
        score_list = raw_scores[0]
    else:
        score_list = raw_scores if isinstance(raw_scores, list) else []

    normalized = _normalize_scores(score_list)
    entailment_score = normalized.get("entailment", 0.0)
    contradiction_score = normalized.get("contradiction", 0.0)
    neutral_score = normalized.get("neutral", max(0.0, 1.0 - entailment_score - contradiction_score))

    similarity = _cosine_similarity(claim, evidence_text)
    credibility_score = _score_credibility(evidence.domain or _extract_domain(evidence.url))

    return {
        "relevance_score": max(similarity, entailment_score),
        "similarity_score": similarity,
        "entailment_score": entailment_score,
        "contradiction_score": contradiction_score,
        "credibility_score": credibility_score,
        "verification_mode": "nli",
        "nli_label": max(normalized, key=normalized.get).upper() if normalized else "",
        "nli_scores": {
            "entailment": entailment_score,
            "contradiction": contradiction_score,
            "neutral": neutral_score,
        },
    }


def _score_evidence_pair(claim: str, evidence: EvidenceItem) -> dict:
    evidence_text = f"{evidence.title}\n{evidence.content}".strip()
    similarity = _cosine_similarity(claim, evidence_text)

    content = evidence_text.lower()
    contradiction_hint = any(marker in content for marker in NEGATION_MARKERS)
    contradiction_score = min(1.0, similarity * 0.45 + (0.45 if contradiction_hint else 0.0))

    credibility_score = _score_credibility(evidence.domain or _extract_domain(evidence.url))
    entailment_score = min(1.0, (similarity * 0.7) + (credibility_score * 0.3))

    return {
        "relevance_score": similarity,
        "similarity_score": similarity,
        "entailment_score": entailment_score,
        "contradiction_score": contradiction_score,
        "credibility_score": credibility_score,
        "verification_mode": "heuristic",
        "nli_label": "",
        "nli_scores": {},
    }


def _aggregate_scores(scored_evidence: list[EvidenceItem]) -> dict:
    if not scored_evidence:
        return {
            "similarity_score": 0.0,
            "entailment_score": 0.0,
            "contradiction_score": 0.0,
            "credibility_score": 0.0,
        }

    similarity_score = max(item.relevance_score for item in scored_evidence)
    entailment_score = max(item.entailment_score for item in scored_evidence)
    contradiction_score = max(
        min(1.0, item.relevance_score * 0.45 + (0.45 if "not" in f"{item.title} {item.content}".lower() else 0.0))
        for item in scored_evidence
    )
    credibility_score = sum(item.credibility_score for item in scored_evidence) / len(scored_evidence)

    return {
        "similarity_score": similarity_score,
        "entailment_score": entailment_score,
        "contradiction_score": contradiction_score,
        "credibility_score": credibility_score,
    }


def _build_explanation(verdict: str, scored_evidence: list[EvidenceItem], metrics: dict) -> str:
    if not scored_evidence:
        return "No evidence was available to support a decision."

    top_sources = [item.domain or _extract_domain(item.url) or item.url for item in scored_evidence[:3] if (item.url or item.domain)]
    source_text = ", ".join(filter(None, top_sources))

    if verdict == "TRUE":
        prefix = "The strongest evidence supports the claim"
    elif verdict == "FALSE":
        prefix = "The strongest evidence contradicts the claim"
    else:
        prefix = "The available evidence is mixed and does not support a confident binary verdict"

    return (
        f"{prefix}. "
        f"Top evidence sources: {source_text or 'n/a'}. "
        f"Similarity={metrics['similarity_score']:.2f}, "
        f"Credibility={metrics['credibility_score']:.2f}, "
        f"Contradiction={metrics['contradiction_score']:.2f}."
    )


def verify_claim(claim: str, evidence_items: list[EvidenceItem]):
    if not evidence_items:
        return VerificationResult(
            verdict="UNVERIFIED",
            confidence=0,
            explanation="No supporting evidence found.",
        ).model_dump()

    scored_evidence: list[EvidenceItem] = []
    for item in evidence_items:
        scores = _score_with_nli(claim, item)
        scored_evidence.append(
            item.model_copy(update=scores)
        )

    scored_evidence.sort(key=lambda item: item.relevance_score, reverse=True)
    metrics = _aggregate_scores(scored_evidence)

    best_entailment = max(item.entailment_score for item in scored_evidence)
    best_contradiction = max(item.contradiction_score for item in scored_evidence)
    best_similarity = max(item.relevance_score for item in scored_evidence)
    average_credibility = metrics["credibility_score"]

    if best_contradiction >= 0.80 and best_contradiction > best_entailment:
        verdict = "FALSE"
    elif best_entailment >= 0.72 and best_similarity >= 0.55:
        verdict = "TRUE"
    else:
        verdict = "MISLEADING"

    confidence = int(
        round(
            100
            * (
                best_entailment * 0.5
                + best_similarity * 0.3
                + average_credibility * 0.2
            )
        )
    )

    # Prefer explicit hf-inference if any evidence used it, then local nli, else heuristic
    if any(getattr(item, "verification_mode", "") == "hf-inference" for item in scored_evidence):
        verification_mode = "hf-inference"
    elif any(getattr(item, "nli_scores", {}) for item in scored_evidence):
        verification_mode = "nli"
    else:
        verification_mode = "heuristic"

    logger.info("Verification mode=%s entailment=%.3f contradiction=%.3f similarity=%.3f", verification_mode, best_entailment, best_contradiction, best_similarity)

    return VerificationResult(
        verdict=verdict,
        confidence=max(0, min(100, confidence)),
        explanation=_build_explanation(verdict, scored_evidence, metrics),
        verification_mode=verification_mode,
        similarity_score=best_similarity,
        entailment_score=best_entailment,
        contradiction_score=best_contradiction,
        credibility_score=average_credibility,
        nli_label=(scored_evidence[0].nli_label if scored_evidence else ""),
        nli_scores=(scored_evidence[0].nli_scores if scored_evidence else {}),
        evidence=scored_evidence,
    ).model_dump()
