from __future__ import annotations

import logging
import os
from typing import Any

from dotenv import load_dotenv

from .models import EvidenceItem, VerificationResult

logger = logging.getLogger(__name__)

# Load environment variables
project_root = os.path.dirname(os.path.dirname(__file__))
dotenv_path = os.path.join(project_root, ".env")
load_dotenv(dotenv_path)

DEFAULT_NLI_MODEL = os.getenv(
    "NLI_MODEL_NAME",
    "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
)

HF_TOKEN = os.getenv("HF_TOKEN")

_nli_pipeline = None


def get_nli_pipeline():
    global _nli_pipeline

    if _nli_pipeline is not None:
        return _nli_pipeline

    try:
        from transformers import pipeline

        logger.info(
            "Loading local NLI model: %s",
            DEFAULT_NLI_MODEL
        )

        _nli_pipeline = pipeline(
            task="text-classification",
            model=DEFAULT_NLI_MODEL,
            tokenizer=DEFAULT_NLI_MODEL,
            top_k=None,
        )

        return _nli_pipeline

    except Exception:
        logger.exception(
            "Failed to load local NLI model"
        )

        return None


def normalize_scores(
    scores: list[dict[str, Any]]
) -> dict[str, float]:

    normalized = {
        "entailment": 0.0,
        "contradiction": 0.0,
        "neutral": 0.0,
    }

    for item in scores:

        label = str(
            item.get("label", "")
        ).upper()

        score = float(
            item.get("score", 0.0)
        )

        if "ENTAIL" in label:
            normalized["entailment"] = score

        elif "CONTRAD" in label:
            normalized["contradiction"] = score

        elif "NEUTRAL" in label:
            normalized["neutral"] = score

    return normalized


def score_with_hf_api(
    premise: str,
    hypothesis: str,
) -> dict[str, float] | None:

    if not HF_TOKEN:
        return None

    try:

        from huggingface_hub import InferenceClient

        client = InferenceClient(
            api_key=HF_TOKEN
        )

        text = (
            f"premise: {premise}\n"
            f"hypothesis: {hypothesis}"
        )

        response = client.text_classification(
            text,
            model=DEFAULT_NLI_MODEL,
        )

        scores = {}

        if isinstance(response, list):

            for item in response:

                scores[
                    item["label"].upper()
                ] = float(
                    item["score"]
                )

        entailment = scores.get(
            "ENTAILMENT",
            0.0,
        )

        contradiction = scores.get(
            "CONTRADICTION",
            0.0,
        )

        neutral = scores.get(
            "NEUTRAL",
            0.0,
        )

        return {
            "entailment": entailment,
            "contradiction": contradiction,
            "neutral": neutral,
        }

    except Exception:

        logger.exception(
            "HF inference failed"
        )

        return None


def score_evidence(
    claim: str,
    evidence: EvidenceItem,
) -> dict[str, Any]:

    premise = evidence.content
    hypothesis = claim

    # HF API first
    scores = score_with_hf_api(
        premise,
        hypothesis,
    )

    if scores is not None:

        entailment = scores["entailment"]
        contradiction = scores["contradiction"]
        neutral = scores["neutral"]

        label = max(
            scores,
            key=scores.get,
        ).upper()

        return {
            "entailment_score": entailment,
            "contradiction_score": contradiction,
            "neutral_score": neutral,
            "nli_label": label,
            "nli_scores": scores,
            "verification_mode": "hf-inference",
        }

    # Local fallback
    pipeline = get_nli_pipeline()

    if pipeline is None:

        return {
            "entailment_score": 0.0,
            "contradiction_score": 0.0,
            "neutral_score": 0.0,
            "nli_label": "",
            "nli_scores": {},
            "verification_mode": "unavailable",
        }

    try:

        result = pipeline(
            {
                "text": premise,
                "text_pair": hypothesis,
            }
        )

        if (
            isinstance(result, list)
            and len(result) > 0
            and isinstance(result[0], list)
        ):
            result = result[0]

        normalized = normalize_scores(
            result
        )

        label = max(
            normalized,
            key=normalized.get,
        ).upper()

        return {
            "entailment_score":
                normalized["entailment"],

            "contradiction_score":
                normalized["contradiction"],

            "neutral_score":
                normalized["neutral"],

            "nli_label": label,

            "nli_scores": normalized,

            "verification_mode": "local-nli",
        }

    except Exception:

        logger.exception(
            "Local NLI scoring failed"
        )

        return {
            "entailment_score": 0.0,
            "contradiction_score": 0.0,
            "neutral_score": 0.0,
            "nli_label": "",
            "nli_scores": {},
            "verification_mode": "error",
        }


def verify_claim(
    claim: str,
    evidence_items: list[EvidenceItem],
) -> dict[str, Any]:

    if not evidence_items:

        return VerificationResult(
            verdict="UNVERIFIED",
            confidence=0,
            explanation="No evidence found.",
        ).model_dump()

    scored_evidence: list[
        EvidenceItem
    ] = []

    for item in evidence_items:

        scores = score_evidence(
            claim,
            item,
        )

        updated = item.model_copy(
            update={
                "entailment_score":
                    scores["entailment_score"],

                "contradiction_score":
                    scores["contradiction_score"],

                "neutral_score":
                    scores["neutral_score"],

                "nli_label":
                    scores["nli_label"],

                "nli_scores":
                    scores["nli_scores"],

                "verification_mode":
                    scores["verification_mode"],
            }
        )

        scored_evidence.append(
            updated
        )

    support_avg = sum(
        item.entailment_score
        for item in scored_evidence
    ) / len(scored_evidence)

    contradiction_avg = sum(
        item.contradiction_score
        for item in scored_evidence
    ) / len(scored_evidence)

    neutral_avg = sum(
        item.neutral_score
        for item in scored_evidence
    ) / len(scored_evidence)

    relevance_avg = sum(
        item.relevance_score
        for item in scored_evidence
    ) / len(scored_evidence)

    # Verdict Logic
    if (
        support_avg > 0.80
        and support_avg >
        contradiction_avg + 0.15
    ):
        verdict = "TRUE"

    elif (
        contradiction_avg > 0.80
        and contradiction_avg >
        support_avg + 0.15
    ):
        verdict = "FALSE"

    else:
        verdict = "MISLEADING"

    confidence = int(
        round(
            100 * max(
                0.0,
                (
                    support_avg
                    - contradiction_avg
                    + relevance_avg
                ) / 2
            )
        )
    )

    confidence = min(
        100,
        max(
            0,
            confidence,
        ),
    )

    return VerificationResult(
        verdict=verdict,
        confidence=confidence,
        explanation="",
        evidence_summary="",
        reasoning="",
        verification_mode=(
            scored_evidence[0]
            .verification_mode
            if scored_evidence
            else "nli"
        ),
        similarity_score=relevance_avg,
        entailment_score=support_avg,
        contradiction_score=contradiction_avg,
        neutral_score=neutral_avg,
        credibility_score=0.0,
        nli_label=(
            scored_evidence[0].nli_label
            if scored_evidence
            else ""
        ),
        nli_scores=(
            scored_evidence[0].nli_scores
            if scored_evidence
            else {}
        ),
        evidence=scored_evidence,
    ).model_dump()