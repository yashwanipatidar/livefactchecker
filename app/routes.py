from fastapi import APIRouter
from .models import ClaimRequest, ClaimResponse
from .rag import fact_check as run_fact_check

router = APIRouter()

@router.post("/fact-check", response_model=ClaimResponse)
def fact_check(request: ClaimRequest):
    return run_fact_check(request.claim)