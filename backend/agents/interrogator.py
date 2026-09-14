from __future__ import annotations

from typing import Any

from backend.agents.gemini_client import GeminiClient


class CandidateInterrogator:

    def __init__(
        self,
        retriever,
        gemini_client: GeminiClient,
    ):

        self.retriever = retriever
        self.gemini = gemini_client

    # ========================================================
    # RETRIEVER RESULT -> PROMPT
    # ========================================================

    def _format_evidence(
        self,
        results: list[dict[str, Any]],
    ) -> str:

        blocks = []

        for item in results:

            blocks.append(
                f"""
[EVIDENCE ID: {item.get("chunk_id", "")}]
Document: {item.get("document_id", "")}
Chapter: {item.get("chapter_number", "")}
Pages: {item.get("pdf_page_start", "")}-
{item.get("pdf_page_end", "")}

{item.get("text", "")}
"""
            )

        return "\n".join(blocks)

    # ========================================================
    # INTERROGATE
    # ========================================================

    def interrogate(
        self,
        candidate: str,
        question: str,
    ):

        search_query = (
            f"{candidate} {question}"
        )

        results = self.retriever.search(
            search_query,
            top_k=6,
        )

        evidence_text = self._format_evidence(
            results
        )

        prompt = f"""
You are a detective-style candidate interrogation agent.

The user is interrogating this candidate:

CANDIDATE:
{candidate}

QUESTION:
{question}

Use ONLY the retrieved corpus evidence below.

Do not invent dialogue, facts, motives, locations,
events or alibis.

If the evidence does not establish an answer,
say clearly that the corpus evidence is insufficient.

The response should sound like a concise answer
from an evidence-grounded investigation system,
not like invented roleplay.

RETRIEVED EVIDENCE:

{evidence_text}

Return a concise answer.
"""

        answer = self.gemini.generate(
            prompt=prompt
        )

        return answer, results