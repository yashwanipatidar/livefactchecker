from __future__ import annotations

import logging
import os
from typing import Any

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load .env
project_root = os.path.dirname(os.path.dirname(__file__))
dotenv_path = os.path.join(project_root, ".env")
load_dotenv(dotenv_path)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Modern Google GenAI SDK
try:
    from google import genai

    _client = genai.Client(
        api_key=GEMINI_API_KEY
    ) if GEMINI_API_KEY else None

except Exception:
    logger.exception("Failed to initialize Gemini client")
    _client = None


def generate_explanation(
    claim: str,
    verdict: str,
    support_score: float,
    contradiction_score: float,
    evidence: list[Any],
) -> dict[str, str]:
    """
    Gemini ONLY generates explanation text.

    It NEVER decides the verdict.
    """

    top_evidence = []

    for item in evidence[:3]:
        source = getattr(item, "url", "") or getattr(item, "domain", "")

        top_evidence.append(
            f"""
Source: {source}

Content:
{getattr(item, "content", "")[:1000]}
"""
        )

    evidence_text = "\n\n".join(top_evidence)

    prompt = f"""
You are a fact-checking explanation assistant.

IMPORTANT:
- Do NOT change the verdict.
- Do NOT re-evaluate the claim.
- Only explain the provided verdict.

Claim:
{claim}

Final Verdict:
{verdict}

Support Score:
{support_score:.3f}

Contradiction Score:
{contradiction_score:.3f}

Evidence:
{evidence_text}

Return:

1. explanation
2. evidence_summary
3. reasoning

Format:

EXPLANATION:
...

EVIDENCE_SUMMARY:
...

REASONING:
...
"""

    # Fallback if Gemini unavailable
    if _client is None:
        logger.warning("Gemini unavailable. Using fallback explanation.")

        return {
            "explanation": (
                f"The verdict is {verdict}. "
                f"Support score={support_score:.2f}, "
                f"Contradiction score={contradiction_score:.2f}."
            ),
            "evidence_summary": "\n".join(
                [
                    f"- {getattr(item, 'content', '')[:200]}"
                    for item in evidence[:3]
                ]
            ),
            "reasoning": (
                f"The verdict was determined using NLI aggregation. "
                f"Support={support_score:.2f}, "
                f"Contradiction={contradiction_score:.2f}."
            ),
        }

    try:

        response = _client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        text = response.text if response else ""

        explanation = ""
        evidence_summary = ""
        reasoning = ""

        current_section = None

        for line in text.splitlines():

            upper = line.strip().upper()

            if upper.startswith("EXPLANATION:"):
                current_section = "explanation"
                continue

            if upper.startswith("EVIDENCE_SUMMARY:"):
                current_section = "evidence_summary"
                continue

            if upper.startswith("REASONING:"):
                current_section = "reasoning"
                continue

            if current_section == "explanation":
                explanation += line + "\n"

            elif current_section == "evidence_summary":
                evidence_summary += line + "\n"

            elif current_section == "reasoning":
                reasoning += line + "\n"

        return {
            "explanation": explanation.strip(),
            "evidence_summary": evidence_summary.strip(),
            "reasoning": reasoning.strip(),
        }

    except Exception:
        logger.exception("Gemini generation failed")

        return {
            "explanation": (
                f"The verdict is {verdict}. "
                f"Support score={support_score:.2f}, "
                f"Contradiction score={contradiction_score:.2f}."
            ),
            "evidence_summary": "",
            "reasoning": (
                f"Support={support_score:.2f}, "
                f"Contradiction={contradiction_score:.2f}"
            ),
        }