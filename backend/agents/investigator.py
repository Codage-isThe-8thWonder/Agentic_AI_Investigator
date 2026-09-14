from __future__ import annotations

from backend.agents.prompts import INVESTIGATOR_PROMPT
from backend.agents.schemas import (
    EvidenceItem,
    InvestigatorFinding,
    TraceEvent,
)


class InvestigatorAgent:

    def __init__(
        self,
        retriever,
        gemini_client,
        max_rounds: int = 2,
    ):
        self.retriever = retriever
        self.gemini_client = gemini_client
        self.max_rounds = max_rounds

    # ========================================================
    # CONVERT RETRIEVER RESULT
    # ========================================================

    def _to_evidence_item(
        self,
        item,
    ) -> EvidenceItem:

        if isinstance(item, EvidenceItem):
            return item

        if not isinstance(item, dict):
            raise TypeError(
                "Retriever result must be a dict "
                f"or EvidenceItem, got {type(item)}"
            )

        pages = []

        start = item.get("pdf_page_start")
        end = item.get("pdf_page_end")

        if start is not None:
            pages.append(int(start))

        if end is not None and end != start:
            pages.append(int(end))

        return EvidenceItem(
            chunk_id=str(
                item.get("chunk_id", "")
            ),

            document_id=str(
                item.get("document_id", "")
            ),

            chapter=item.get(
                "chapter_number"
            ),

            chapter_heading=item.get(
                "chapter_heading"
            ),

            pdf_pages=pages,

            text=str(
                item.get("text", "")
            ),

            retrieval_source="hybrid",

            relevance_score=float(
                item.get(
                    "hybrid_score",
                    0.0,
                )
            ),

            evidence_status="unverified",
        )

    # ========================================================
    # FORMAT EVIDENCE
    # ========================================================

    def _format_evidence(
        self,
        evidence: list[EvidenceItem],
    ) -> str:

        blocks = []

        for item in evidence:

            pages = ", ".join(
                str(page)
                for page in item.pdf_pages
            )

            blocks.append(
                f"""
[EVIDENCE ID: {item.chunk_id}]
Document: {item.document_id}
Chapter: {item.chapter}
Chapter Heading: {item.chapter_heading}
PDF Pages: {pages}

TEXT:
{item.text}
"""
            )

        return "\n".join(blocks)

    # ========================================================
    # INVESTIGATE
    # ========================================================

    def investigate(
        self,
        question: str,
    ) -> tuple[
        InvestigatorFinding,
        list[EvidenceItem],
        list[TraceEvent],
    ]:

        all_evidence = []
        trace = []

        current_query = question
        finding = None

        for round_number in range(
            1,
            self.max_rounds + 1,
        ):

            # ------------------------------------------------
            # RETRIEVAL
            # ------------------------------------------------

            results = self.retriever.search(
                current_query,
                top_k=6,
            )

            evidence = [
                self._to_evidence_item(item)
                for item in results
            ]

            existing_ids = {
                item.chunk_id
                for item in all_evidence
            }

            for item in evidence:

                if item.chunk_id not in existing_ids:
                    all_evidence.append(item)

            evidence_ids = [
                item.chunk_id
                for item in evidence
            ]

            trace.append(
                TraceEvent(
                    stage="investigator_retrieval",
                    message=(
                        f"Investigator retrieval "
                        f"round {round_number}"
                    ),
                    query=current_query,
                    evidence_ids=evidence_ids,
                )
            )

            # ------------------------------------------------
            # GEMINI
            # ------------------------------------------------

            prompt = INVESTIGATOR_PROMPT.format(
                question=question,
                query=current_query,
                evidence=self._format_evidence(
                    evidence
                ),
            )

            finding = (
                self.gemini_client
                .generate_structured(
                    prompt=prompt,
                    schema=InvestigatorFinding,
                )
            )

            finding.retrieval_rounds = round_number

            trace.append(
                TraceEvent(
                    stage="investigator_reasoning",
                    message=(
                        f"Investigator reasoning "
                        f"completed for round "
                        f"{round_number}"
                    ),
                    query=current_query,
                    evidence_ids=finding.evidence_ids,
                )
            )

            # ------------------------------------------------
            # COMPLETE
            # ------------------------------------------------

            if finding.answer_complete:
                break

            # ------------------------------------------------
            # SELF CORRECTION
            # ------------------------------------------------

            refined_query = (
                finding.refined_query
            )

            if not refined_query:
                break

            if (
                refined_query.strip().lower()
                == current_query.strip().lower()
            ):
                break

            current_query = refined_query

            trace.append(
                TraceEvent(
                    stage="investigator_self_correction",
                    message=(
                        "Investigator identified "
                        "missing information and "
                        "refined the query."
                    ),
                    query=current_query,
                    evidence_ids=[],
                )
            )

        # ====================================================
        # FALLBACK
        # ====================================================

        if finding is None:

            finding = InvestigatorFinding(
                hypothesis="",
                reasoning_summary=(
                    "No investigator finding "
                    "was produced."
                ),
                confidence="low",
                answer_complete=False,
                missing_information=[
                    "No finding was produced."
                ],
                retrieval_rounds=0,
            )

        return (
            finding,
            all_evidence,
            trace,
        )