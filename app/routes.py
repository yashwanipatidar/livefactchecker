from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from .models import ClaimRequest, ClaimResponse
from .rag import fact_check as run_fact_check

logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["Fact Checking"]
)


@router.get("/health")
async def health_check():
    """
    Health check endpoint.
    """

    return {
        "status": "healthy",
        "service": "livefactchecker"
    }


@router.post(
    "/fact-check",
    response_model=ClaimResponse,
    summary="Verify a factual claim",
    response_description="Fact-checking result"
)
async def fact_check(
    request: ClaimRequest
):
    """
    Verify a factual claim using:

    - Tavily evidence retrieval
    - Semantic retrieval
    - NLI verification
    - Gemini explanation generation
    """

    try:

        claim = request.claim.strip()

        if not claim:
            raise HTTPException(
                status_code=400,
                detail="Claim cannot be empty."
            )

        logger.info(
            "Received fact-check request: %s",
            claim[:200]
        )

        result = run_fact_check(claim)

        logger.info(
            "Fact-check completed: %s",
            result.get("verdict")
        )

        return ClaimResponse(**result)

    except HTTPException:
        raise

    except Exception as exc:

        logger.exception(
            "Fact-check endpoint failed"
        )

        return JSONResponse(
            status_code=500,
            content={
                "verdict": "ERROR",
                "confidence": 0,
                "explanation": (
                    "An internal server error occurred."
                ),
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
                "error": str(exc)
            }
        )