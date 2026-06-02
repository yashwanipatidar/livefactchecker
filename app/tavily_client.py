from __future__ import annotations

import logging
import os
from urllib.parse import urlparse

from dotenv import load_dotenv

from .models import EvidenceItem

logger = logging.getLogger(__name__)

# Load .env
project_root = os.path.dirname(os.path.dirname(__file__))
dotenv_path = os.path.join(project_root, ".env")
load_dotenv(dotenv_path)

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

try:
    from tavily import TavilyClient

    _client = (
        TavilyClient(api_key=TAVILY_API_KEY)
        if TAVILY_API_KEY
        else None
    )

except Exception:
    logger.exception("Failed to initialize Tavily client")
    _client = None


def extract_domain(url: str) -> str:
    """
    Extract domain from URL.
    """

    if not url:
        return ""

    try:
        parsed = urlparse(url)

        return (
            parsed.netloc
            .lower()
            .replace("www.", "")
        )

    except Exception:
        return ""


def search_evidence(
    claim: str,
    max_results: int = 10,
) -> list[EvidenceItem]:
    """
    Search Tavily for evidence related to a claim.

    Returns:
        List[EvidenceItem]
    """

    if not claim:
        return []

    if _client is None:

        logger.error(
            "Tavily client unavailable."
        )

        return []

    try:

        logger.info(
            "Searching Tavily for claim: %s",
            claim[:200],
        )

        response = _client.search(
            query=claim,
            search_depth="advanced",
            max_results=max_results,
            include_answer=False,
            include_raw_content=False,
        )

        results = response.get(
            "results",
            [],
        )

        evidence_items: list[
            EvidenceItem
        ] = []

        for result in results:

            url = result.get(
                "url",
                "",
            )

            content = result.get(
                "content",
                "",
            )

            title = result.get(
                "title",
                "",
            )

            if not content:
                continue

            evidence_items.append(
                EvidenceItem(
                    title=title,
                    content=content,
                    url=url,
                    domain=extract_domain(
                        url
                    ),
                )
            )

        logger.info(
            "Retrieved %d evidence documents",
            len(evidence_items),
        )

        return evidence_items

    except Exception:

        logger.exception(
            "Tavily search failed"
        )

        return []