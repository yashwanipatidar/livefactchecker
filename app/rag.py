from __future__ import annotations

import logging

from .retriever import retrieve_relevant_context
from .tavily_client import search_evidence
from .verifier import verify_claim
from .gemini_client import generate_explanation

logger = logging.getLogger(__name__)


def fact_check(claim: str) -> dict:
    """
    Complete fact-checking pipeline.

    Flow:
        Claim
          ↓
        Tavily Search
          ↓
        Semantic Retrieval
          ↓
        NLI Verification
          ↓
        Gemini Explanation
          ↓
        Response
    """

    if not claim or not claim.strip():
        return {
            "verdict": "UNVERIFIED",
            "confidence": 0,
            "explanation": "No claim provided.",
            "evidence_summary": "",
            "reasoning": "",
            "verification_mode": "none",
            "sources": [],
            "similarity_score": 0.0,
            "entailment_score": 0.0,
            "contradiction_score": 0.0,
            "neutral_score": 0.0,
            "credibility_score": 0.0,
            "nli_label": "",
            "nli_scores": {},
        }

    try:
        logger.info("Starting fact check")

        # Step 1: Retrieve evidence from Tavily
        evidence = search_evidence(claim)

        if not evidence:
            return {
                "verdict": "UNVERIFIED",
                "confidence": 0,
                "explanation": "No evidence could be retrieved.",
                "evidence_summary": "",
                "reasoning": "",
                "verification_mode": "retrieval_failed",
                "sources": [],
                "similarity_score": 0.0,
                "entailment_score": 0.0,
                "contradiction_score": 0.0,
                "neutral_score": 0.0,
                "credibility_score": 0.0,
                "nli_label": "",
                "nli_scores": {},
            }

        logger.info("Retrieved %d evidence documents", len(evidence))

        # Step 2: Semantic ranking + chunk retrieval
        ranked_evidence = retrieve_relevant_context(
            claim=claim,
            evidence_items=evidence,
            top_k=5,
        )

        if not ranked_evidence:
            return {
                "verdict": "UNVERIFIED",
                "confidence": 0,
                "explanation": "No relevant evidence found.",
                "evidence_summary": "",
                "reasoning": "",
                "verification_mode": "retrieval_failed",
                "sources": [],
                "similarity_score": 0.0,
                "entailment_score": 0.0,
                "contradiction_score": 0.0,
                "neutral_score": 0.0,
                "credibility_score": 0.0,
                "nli_label": "",
                "nli_scores": {},
            }

        logger.info(
            "Selected %d ranked evidence chunks",
            len(ranked_evidence),
        )

        # Step 3: NLI verification
        verification_result = verify_claim(
            claim=claim,
            evidence_items=ranked_evidence,
        )

        # Step 4: Gemini explanation generation
        try:
            gemini_result = generate_explanation(
                claim=claim,
                verdict=verification_result["verdict"],
                support_score=verification_result.get(
                    "entailment_score",
                    0.0,
                ),
                contradiction_score=verification_result.get(
                    "contradiction_score",
                    0.0,
                ),
                evidence=verification_result.get(
                    "evidence",
                    [],
                ),
            )

            verification_result["explanation"] = (
                gemini_result.get("explanation", "")
            )

            verification_result["evidence_summary"] = (
                gemini_result.get("evidence_summary", "")
            )

            verification_result["reasoning"] = (
                gemini_result.get("reasoning", "")
            )

        except Exception:
            logger.exception(
                "Gemini explanation generation failed"
            )

            verification_result.setdefault(
                "explanation",
                "Explanation unavailable.",
            )

            verification_result.setdefault(
                "evidence_summary",
                "",
            )

            verification_result.setdefault(
                "reasoning",
                "",
            )

        # Step 5: Add sources
        verification_result["sources"] = list(
            {
                item.url
                for item in ranked_evidence
                if item.url
            }
        )

        logger.info(
            "Fact check completed: %s",
            verification_result["verdict"],
        )

        return verification_result

    except Exception:
        logger.exception("Fact checking pipeline failed")

        return {
            "verdict": "ERROR",
            "confidence": 0,
            "explanation": "An internal error occurred.",
            "evidence_summary": "",
            "reasoning": "",
            "verification_mode": "error",
            "sources": [],
            "similarity_score": 0.0,
            "entailment_score": 0.0,
            "contradiction_score": 0.0,
            "neutral_score": 0.0,
            "credibility_score": 0.0,
            "nli_label": "",
            "nli_scores": {},
        }