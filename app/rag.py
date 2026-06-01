from .retriever import retrieve_relevant_context
from .tavily_client import search_evidence
from .verifier import verify_claim

def fact_check(claim: str):

    evidence = search_evidence(claim)
    ranked_evidence = retrieve_relevant_context(claim, evidence)

    result = verify_claim(
        claim,
        ranked_evidence
    )

    result["sources"] = [item.url for item in ranked_evidence if item.url]

    return result