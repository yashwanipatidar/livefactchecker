import os
from urllib.parse import urlparse

from dotenv import load_dotenv

try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None

from .models import EvidenceItem


# Load the project .env explicitly
project_root = os.path.dirname(os.path.dirname(__file__))
dotenv_path = os.path.join(project_root, ".env")
load_dotenv(dotenv_path)

api_key = os.getenv("TAVILY_API_KEY")
client = TavilyClient(api_key=api_key) if (api_key and TavilyClient is not None) else None


def _extract_domain(url: str) -> str:
    if not url:
        return ""

    parsed = urlparse(url)
    return parsed.netloc.lower().removeprefix("www.")

def search_evidence(claim: str):

    if client is None:
        return []

    try:
        response = client.search(
            query=claim,
            search_depth="advanced",
            max_results=5
        )
    except Exception:
        return []

    evidence: list[EvidenceItem] = []

    for result in response["results"]:
        url = result.get("url", "")
        evidence.append(
            EvidenceItem(
                title=result.get("title", ""),
                content=result.get("content", ""),
                url=url,
                domain=_extract_domain(url),
            )
        )

    return evidence