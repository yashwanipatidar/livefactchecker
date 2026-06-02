from __future__ import annotations

import logging

from fastapi import FastAPI

from app.routes import router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Live Fact Checker",
    description=(
        "AI-powered fact-checking API using "
        "Tavily Search, Semantic Retrieval, "
        "DeBERTa NLI, and Gemini 2.5 Flash."
    ),
    version="2.0.0",
)

# Register routes
app.include_router(router)

logger.info("Live Fact Checker API initialized")


@app.get("/")
async def root():
    """
    Root endpoint.
    """

    return {
        "service": "Live Fact Checker",
        "version": "2.0.0",
        "status": "running",
    }