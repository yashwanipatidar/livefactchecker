from pydantic import BaseModel, Field

class ClaimRequest(BaseModel):
    claim: str


class EvidenceItem(BaseModel):
    title: str = ""
    content: str = ""
    url: str = ""
    domain: str = ""
    relevance_score: float = 0.0
    credibility_score: float = 0.0
    # Per-evidence NLI/scoring fields (populated by verifier)
    similarity_score: float = 0.0
    entailment_score: float = 0.0
    contradiction_score: float = 0.0
    nli_label: str = ""
    nli_scores: dict[str, float] = Field(default_factory=dict)


class VerificationResult(BaseModel):
    verdict: str
    confidence: int
    explanation: str
    verification_mode: str = "heuristic"
    similarity_score: float = 0.0
    entailment_score: float = 0.0
    contradiction_score: float = 0.0
    credibility_score: float = 0.0
    nli_label: str = ""
    nli_scores: dict[str, float] = Field(default_factory=dict)
    evidence: list[EvidenceItem] = Field(default_factory=list)

class ClaimResponse(BaseModel):
    verdict: str
    explanation: str
    confidence: int
    verification_mode: str = "heuristic"
    sources: list[str] = Field(default_factory=list)
    similarity_score: float = 0.0
    entailment_score: float = 0.0
    contradiction_score: float = 0.0
    credibility_score: float = 0.0
    nli_label: str = ""
    nli_scores: dict[str, float] = Field(default_factory=dict)