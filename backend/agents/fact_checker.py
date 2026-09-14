from __future__ import annotations

from typing import Any

from backend.agents.gemini_client import GeminiClient
from backend.agents.prompts import FACT_CHECKER_PROMPT
from backend.agents.schemas import (
    EvidenceItem,
    FactCheckResult,
    InvestigatorFinding,
    TraceEvent,
)


class FactCheckerAgent:

    def __init__(
        self,
        retriever,
        gemini_client: GeminiClient,
    ):
        self.retriever = retriever
        self.gemini_client = gemini_client

    # ========================================================
    # CONVERT RETRIEVER RESULT -> EvidenceItem
    # ========================================================

    def _to_evidence_item(
        self,
        item: dict[str, Any],
    ) -> EvidenceItem:

        start = item.get(
            "pdf_page_start"
        )

        end = item.get(
            "pdf_page_end"
        )

        pages = []

        if start is not None:
            pages.append(int(start))

        if end is not None and end != start:
            pages.append(int(end))

        return EvidenceItem(
            chunk_id=str(
                item.get(
                    "chunk_id",
                    "",
                )
            ),

            document_id=str(
                item.get(
                    "document_id",
                    "",
                )
            ),

            chapter=item.get(
                "chapter_number"
            ),

            chapter_heading=item.get(
                "chapter_heading"
            ),

            pdf_pages=pages,

            text=str(
                item.get(
                    "text",
                    "",
                )
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

            blocks.append(
                f"""
[EVIDENCE ID: {item.chunk_id}]
Chapter: {item.chapter}
Chapter Heading: {item.chapter_heading}
Pages: {item.pdf_pages}

TEXT:
{item.text[:1800]}
"""
            )

        return "\n".join(blocks)

    # ========================================================
    # CHECK
    # ========================================================

    def check(
        self,
        question: str,
        investigator_finding: InvestigatorFinding,
        trace: list[TraceEvent],
    ) -> tuple[
        FactCheckResult,
        list[EvidenceItem],
        list[TraceEvent],
    ]:

        print()
        print("=" * 60)
        print("FACT CHECKER")
        print("=" * 60)

        hypothesis = (
            investigator_finding.hypothesis
        )

        # ----------------------------------------------------
        # Independent adversarial queries
        # ----------------------------------------------------

        queries = [
            hypothesis,

            (
                f"{question} "
                "direct evidence identities companions"
            ),

            (
                "Who is the associate or companion "
                "of Jonathan Small?"
            ),

            (
                f"{question} "
                "contradicting evidence alternative identity"
            ),
        ]

        evidence_by_id: dict[
            str,
            EvidenceItem,
        ] = {}

        # ----------------------------------------------------
        # RETRIEVE
        # ----------------------------------------------------

        for query in queries:

            print()
            print(
                f"Fact-check query: {query}"
            )

            results = self.retriever.search(
                query,
                top_k=5,
            )

            ids = []

            for result in results:

                evidence = (
                    self._to_evidence_item(
                        result
                    )
                )

                evidence_by_id[
                    evidence.chunk_id
                ] = evidence

                ids.append(
                    evidence.chunk_id
                )

            trace.append(
                TraceEvent(
                    stage="fact_checker_retrieval",
                    message=(
                        "Independent adversarial "
                        "retrieval"
                    ),
                    query=query,
                    evidence_ids=ids,
                )
            )

        evidence = list(
            evidence_by_id.values()
        )

        print(
            f"\nFact Checker collected "
            f"{len(evidence)} unique evidence chunks."
        )

        # ----------------------------------------------------
        # LIMIT PROMPT CONTEXT
        # ----------------------------------------------------

        evidence_for_prompt = evidence[:10]

        evidence_text = self._format_evidence(
            evidence_for_prompt
        )

        # ----------------------------------------------------
        # IMPORTANT
        #
        # We use FACT_CHECKER_PROMPT directly.
        # There is NO FACT_CHECKER_SYSTEM here.
        #
        # This eliminates the old
        # {investigator_evidence_ids} KeyError.
        # ----------------------------------------------------

        prompt = FACT_CHECKER_PROMPT.format(
            question=question,

            hypothesis=(
                investigator_finding.hypothesis
            ),

            reasoning=(
                investigator_finding
                .reasoning_summary
            ),

            missing_information=(
                investigator_finding
                .missing_information
            ),

            evidence=evidence_text,
        )

        # ----------------------------------------------------
        # GEMINI
        # ----------------------------------------------------

        result = (
            self.gemini_client
            .generate_structured(
                prompt=prompt,
                schema=FactCheckResult,
            )
        )

        # ----------------------------------------------------
        # MARK EVIDENCE STATUS
        # ----------------------------------------------------

        status_by_id = {}

        for claim in result.claims:

            for evidence_id in (
                claim.evidence_ids
            ):

                if (
                    claim.status
                    == "SUPPORTED"
                ):

                    status_by_id[
                        evidence_id
                    ] = "verified"

                elif (
                    claim.status
                    == "CONTRADICTED"
                ):

                    status_by_id[
                        evidence_id
                    ] = "misleading"

        for item in evidence:

            if item.chunk_id in status_by_id:

                item.evidence_status = (
                    status_by_id[
                        item.chunk_id
                    ]
                )

        # ----------------------------------------------------
        # TRACE
        # ----------------------------------------------------

        trace.append(
            TraceEvent(
                stage="fact_checker_reasoning",
                message=(
                    "Fact Checker completed: "
                    f"{result.overall_status}"
                ),
                query=None,
                evidence_ids=(
                    result.adversarial_evidence_ids
                ),
            )
        )

        print(
            "\nFact Checker completed:"
            f" {result.overall_status}"
        )

        return (
            result,
            evidence,
            trace,
        )