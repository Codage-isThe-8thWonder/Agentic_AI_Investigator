from __future__ import annotations

import uuid

from backend.agents.fact_checker import FactCheckerAgent
from backend.agents.investigator import InvestigatorAgent
from backend.agents.prompts import FINAL_SYNTHESIS_PROMPT
from backend.agents.schemas import (
    EvidenceItem,
    FinalVerdict,
    InvestigationResult,
    TraceEvent,
)


class InvestigationOrchestrator:

    def __init__(
        self,
        retriever,
        gemini_client,
    ):

        self.retriever = retriever
        self.gemini_client = gemini_client

        self.investigator = InvestigatorAgent(
            retriever=retriever,
            gemini_client=gemini_client,
            max_rounds=2,
        )

        self.fact_checker = FactCheckerAgent(
            retriever=retriever,
            gemini_client=gemini_client,
        )

    # ========================================================
    # MAIN PIPELINE
    # ========================================================

    def investigate(
        self,
        question: str,
    ) -> InvestigationResult:

        case_id = str(uuid.uuid4())

        trace = []

        # ====================================================
        # 1. INVESTIGATOR
        # ====================================================

        trace.append(
            TraceEvent(
                stage="orchestrator",
                message="Investigation started.",
                query=question,
                evidence_ids=[],
            )
        )

        (
            investigator_finding,
            investigator_evidence,
            investigator_trace,
        ) = self.investigator.investigate(
            question
        )

        trace.extend(
            investigator_trace
        )

        # ====================================================
        # 2. FACT CHECKER
        # ====================================================

        trace.append(
            TraceEvent(
                stage="orchestrator",
                message=(
                    "Starting independent "
                    "adversarial fact checking."
                ),
                query=question,
                evidence_ids=[
                    item.chunk_id
                    for item in investigator_evidence
                ],
            )
        )

        (
            fact_check,
            fact_check_evidence,
            fact_check_trace,
        ) = self.fact_checker.check(
            question=question,
            investigator_finding=investigator_finding,
            trace=trace,
        )

        # `trace` already contains fact-check events
        # because the method appends to the same list.
        #
        # Keep only additional returned events here.
        if fact_check_trace is not trace:
            trace = fact_check_trace

        # ====================================================
        # 3. MERGE EVIDENCE
        # ====================================================

        evidence_by_id = {}

        for item in investigator_evidence:
            evidence_by_id[
                item.chunk_id
            ] = item

        for item in fact_check_evidence:

            if item.chunk_id not in evidence_by_id:
                evidence_by_id[
                    item.chunk_id
                ] = item

        all_evidence = list(
            evidence_by_id.values()
        )

        # ====================================================
        # 4. FINAL SYNTHESIS
        # ====================================================

        trace.append(
            TraceEvent(
                stage="final_synthesis",
                message="Generating final verdict.",
                query=question,
                evidence_ids=[
                    item.chunk_id
                    for item in all_evidence
                ],
            )
        )

        evidence_blocks = []

        for item in all_evidence:

            evidence_blocks.append(
                f"""
[EVIDENCE ID: {item.chunk_id}]
STATUS: {item.evidence_status}
CHAPTER: {item.chapter}
PAGES: {item.pdf_pages}

{item.text[:1800]}
"""
            )

        evidence_text = "\n".join(
            evidence_blocks
        )

        prompt = FINAL_SYNTHESIS_PROMPT.format(
            question=question,

            investigator=(
                investigator_finding
                .model_dump_json(
                    indent=2
                )
            ),

            fact_check=(
                fact_check
                .model_dump_json(
                    indent=2
                )
            ),

            evidence=evidence_text,
        )

        verdict = (
            self.gemini_client
            .generate_structured(
                prompt=prompt,
                schema=FinalVerdict,
            )
        )

        # ====================================================
        # 5. CONFIDENCE SAFETY
        # ====================================================

        if (
            not investigator_finding.answer_complete
            and verdict.confidence == "high"
        ):
            verdict.confidence = "medium"

        if (
            fact_check.overall_status
            in {
                "MIXED",
                "CONTRADICTED",
            }
            and verdict.confidence == "high"
        ):
            verdict.confidence = "medium"

        # ====================================================
        # 6. EVIDENCE STATUS
        # ====================================================

        supporting = set(
            verdict.supporting_evidence_ids
        )

        contradicting = set(
            verdict.contradicting_evidence_ids
        )

        unverified = set(
            verdict.unverified_evidence_ids
        )

        for item in all_evidence:

            if item.chunk_id in supporting:

                item.evidence_status = "verified"

            elif item.chunk_id in contradicting:

                item.evidence_status = "misleading"

            elif item.chunk_id in unverified:

                item.evidence_status = "unverified"

        # ====================================================
        # 7. FINISH
        # ====================================================

        trace.append(
            TraceEvent(
                stage="orchestrator",
                message="Investigation completed.",
                query=question,
                evidence_ids=[
                    item.chunk_id
                    for item in all_evidence
                ],
            )
        )

        return InvestigationResult(
            case_id=case_id,
            question=question,
            investigator=investigator_finding,
            fact_check=fact_check,
            verdict=verdict,
            evidence=all_evidence,
            execution_trace=trace,
        )