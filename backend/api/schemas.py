from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ============================================================
# HEALTH
# ============================================================

class HealthResponse(BaseModel):
    status: str
    service: str
    retriever: str
    model: str


# ============================================================
# RETRIEVAL
# ============================================================

class RetrieveRequest(BaseModel):

    query: str = Field(
        ...,
        min_length=2,
        max_length=1000,
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
    )


class RetrievedEvidence(BaseModel):

    chunk_id: str
    document_id: str

    chapter: Optional[int] = None
    chapter_heading: Optional[str] = None

    pdf_page_start: Optional[int] = None
    pdf_page_end: Optional[int] = None

    text: str

    hybrid_score: Optional[float] = None
    bm25_score: Optional[float] = None
    semantic_score: Optional[float] = None


class RetrieveResponse(BaseModel):

    query: str
    results: List[RetrievedEvidence]
    count: int


# ============================================================
# INVESTIGATION
# ============================================================

class InvestigationRequest(BaseModel):

    question: str = Field(
        ...,
        min_length=5,
        max_length=1000,
    )


class InvestigationResponse(BaseModel):

    case_id: str
    question: str

    investigator: Dict[str, Any]
    fact_check: Dict[str, Any]
    verdict: Dict[str, Any]

    evidence: List[Dict[str, Any]]
    execution_trace: List[Dict[str, Any]]

    graph: Dict[str, Any]


# ============================================================
# CANDIDATES
# ============================================================

class Candidate(BaseModel):

    candidate_id: str

    name: str

    description: Optional[str] = None

    evidence_ids: List[str] = Field(
        default_factory=list
    )


class CandidateListResponse(BaseModel):

    candidates: List[Candidate]


# ============================================================
# INTERROGATION
# ============================================================

class InterrogateRequest(BaseModel):

    candidate: str = Field(
        ...,
        min_length=2,
        max_length=200,
    )

    question: str = Field(
        ...,
        min_length=2,
        max_length=1000,
    )


class InterrogateResponse(BaseModel):

    candidate: str

    question: str

    answer: str

    evidence: List[RetrievedEvidence]


# ============================================================
# FACT CHECK
# ============================================================

class FactCheckRequest(BaseModel):

    case_id: str

    question: str

    investigator: Dict[str, Any]


class FactCheckResponse(BaseModel):

    case_id: str

    fact_check: Dict[str, Any]

    evidence: List[Dict[str, Any]]


# ============================================================
# USER PREDICTION / VERDICT
# ============================================================

class SubmitVerdictRequest(BaseModel):

    case_id: str

    suspect: str = Field(
        ...,
        min_length=2,
        max_length=200,
    )

    reasoning: str = Field(
        ...,
        min_length=5,
        max_length=3000,
    )

    supporting_evidence_ids: List[str] = Field(
        default_factory=list
    )


class SubmitVerdictResponse(BaseModel):

    case_id: str

    user_suspect: str

    agent_verdict: str

    verdict_match: bool

    evaluation: str

    supporting_evidence_ids: List[str]

    agent_supporting_evidence_ids: List[str]


# ============================================================
# CORPUS
# ============================================================

class CorpusResponse(BaseModel):

    document_id: str

    title: str

    source: Optional[str] = None

    total_pages: Optional[int] = None
    novel_pages: Optional[int] = None

    total_chunks: int

    embedding_dimension: Optional[int] = None


# ============================================================
# ERROR
# ============================================================

class ErrorResponse(BaseModel):

    detail: str