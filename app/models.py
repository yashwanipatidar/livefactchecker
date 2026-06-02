from __future__ import annotations

from pydantic import BaseModel, Field


class ClaimRequest(BaseModel):
    claim: str


class EvidenceItem(BaseModel):
    title: str = ""
    content: str = ""
    url: str = ""
    domain: str = ""

    # Retrieval
    relevance_score: float = 0.0

    # Source quality
    credibility_score: float = 0.0

    # NLI
    entailment_score: float = 0.0
    contradiction_score: float = 0.0
    neutral_score: float = 0.0

    nli_label: str = ""

    nli_scores: dict[str, float] = Field(
        default_factory=dict
    )

    verification_mode: str = "nli"


class VerificationResult(BaseModel):
    verdict: str
    confidence: int

    # Gemini-generated explanation
    explanation: str = ""

    # Gemini-generated summary
    evidence_summary: str = ""

    # Gemini-generated reasoning
    reasoning: str = ""

    verification_mode: str = "nli"

    similarity_score: float = 0.0

    entailment_score: float = 0.0
    contradiction_score: float = 0.0
    neutral_score: float = 0.0

    credibility_score: float = 0.0

    nli_label: str = ""

    nli_scores: dict[str, float] = Field(
        default_factory=dict
    )

    evidence: list[EvidenceItem] = Field(
        default_factory=list
    )


class ClaimResponse(BaseModel):
    verdict: str
    confidence: int

    explanation: str = ""

    evidence_summary: str = ""

    reasoning: str = ""

    verification_mode: str = "nli"

    sources: list[str] = Field(
        default_factory=list
    )

    similarity_score: float = 0.0

    entailment_score: float = 0.0
    contradiction_score: float = 0.0
    neutral_score: float = 0.0

    credibility_score: float = 0.0

    nli_label: str = ""

    nli_scores: dict[str, float] = Field(
        default_factory=dict
    )