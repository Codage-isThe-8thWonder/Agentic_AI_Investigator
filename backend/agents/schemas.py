from typing import Literal

from pydantic import BaseModel, Field


# ============================================================
# EVIDENCE
# ============================================================

class EvidenceItem(BaseModel):
    chunk_id: str
    document_id: str

    chapter: int | None = None
    chapter_heading: str | None = None

    pdf_pages: list[int] = Field(
        default_factory=list
    )

    text: str

    retrieval_source: str = "hybrid"

    relevance_score: float = 0.0

    evidence_status: Literal[
        "verified",
        "misleading",
        "unverified",
    ] = "unverified"


# ============================================================
# INVESTIGATOR
# ============================================================

class InvestigatorFinding(BaseModel):
    hypothesis: str

    reasoning_summary: str

    confidence: Literal[
        "low",
        "medium",
        "high",
    ]

    answer_complete: bool

    missing_information: list[str] = Field(
        default_factory=list
    )

    refined_query: str | None = None

    evidence_ids: list[str] = Field(
        default_factory=list
    )

    retrieval_rounds: int = 1


# ============================================================
# FACT CHECKER
# ============================================================

class ClaimCheck(BaseModel):
    claim: str

    status: Literal[
        "SUPPORTED",
        "CONTRADICTED",
        "UNVERIFIED",
    ]

    explanation: str

    evidence_ids: list[str] = Field(
        default_factory=list
    )


class FactCheckResult(BaseModel):
    overall_status: Literal[
        "SUPPORTED",
        "CONTRADICTED",
        "MIXED",
        "UNVERIFIED",
    ]

    claims: list[ClaimCheck] = Field(
        default_factory=list
    )

    adversarial_evidence_ids: list[str] = Field(
        default_factory=list
    )


# ============================================================
# FINAL VERDICT
# ============================================================

class FinalVerdict(BaseModel):
    verdict: str

    confidence: Literal[
        "low",
        "medium",
        "high",
    ]

    explanation: str

    supporting_evidence_ids: list[str] = Field(
        default_factory=list
    )

    contradicting_evidence_ids: list[str] = Field(
        default_factory=list
    )

    unverified_evidence_ids: list[str] = Field(
        default_factory=list
    )


# ============================================================
# EXECUTION TRACE
# ============================================================

class TraceEvent(BaseModel):
    stage: str

    message: str

    query: str | None = None

    evidence_ids: list[str] = Field(
        default_factory=list
    )


# ============================================================
# COMPLETE INVESTIGATION RESULT
# ============================================================

class InvestigationResult(BaseModel):
    case_id: str

    question: str

    investigator: InvestigatorFinding

    fact_check: FactCheckResult

    verdict: FinalVerdict

    evidence: list[EvidenceItem] = Field(
        default_factory=list
    )

    execution_trace: list[TraceEvent] = Field(
        default_factory=list
    )