from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from backend.api.schemas import (
    Candidate,
    CandidateListResponse,
    CorpusResponse,
    HealthResponse,
    InterrogateRequest,
    InterrogateResponse,
    InvestigationRequest,
    InvestigationResponse,
    RetrieveRequest,
    RetrieveResponse,
    RetrievedEvidence,
    SubmitVerdictRequest,
    SubmitVerdictResponse,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

METADATA_FILE = (
    PROCESSED_DIR
    / "corpus_metadata.json"
)

GRAPH_FILE = (
    PROCESSED_DIR
    / "evidence_graph.json"
)

CASES_DIR = (
    PROCESSED_DIR
    / "cases"
)

CASES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# ROUTER
# ============================================================

router = APIRouter()


# ============================================================
# LAZY SERVICES
# ============================================================

_retriever = None
_gemini_client = None
_orchestrator = None
_interrogator = None


# ============================================================
# RETRIEVER
# ============================================================

def get_retriever():

    global _retriever

    if _retriever is None:

        from backend.retrieval.hybrid_retriever import (
            HybridRetriever,
        )

        print()
        print("=" * 60)
        print("INITIALIZING API RETRIEVER")
        print("=" * 60)

        _retriever = HybridRetriever()

        print("API retriever ready ✅")

    return _retriever


# ============================================================
# GEMINI CLIENT
# ============================================================

def get_gemini_client():

    global _gemini_client

    if _gemini_client is None:

        from backend.agents.gemini_client import (
            GeminiClient,
        )

        print()
        print("=" * 60)
        print("INITIALIZING GEMINI CLIENT")
        print("=" * 60)

        _gemini_client = GeminiClient()

        print("Gemini client ready ✅")

    return _gemini_client


# ============================================================
# ORCHESTRATOR
# ============================================================

def get_orchestrator():

    global _orchestrator

    if _orchestrator is None:

        from backend.agents.orchestrator import (
            InvestigationOrchestrator,
        )

        print()
        print("=" * 60)
        print("INITIALIZING INVESTIGATION ORCHESTRATOR")
        print("=" * 60)

        _orchestrator = InvestigationOrchestrator(
            retriever=get_retriever(),
            gemini_client=get_gemini_client(),
        )

        print(
            "Investigation orchestrator ready ✅"
        )

    return _orchestrator


# ============================================================
# INTERROGATOR
# ============================================================

def get_interrogator():

    global _interrogator

    if _interrogator is None:

        from backend.agents.interrogator import (
            CandidateInterrogator,
        )

        print()
        print("=" * 60)
        print("INITIALIZING CANDIDATE INTERROGATOR")
        print("=" * 60)

        _interrogator = CandidateInterrogator(
            retriever=get_retriever(),
            gemini_client=get_gemini_client(),
        )

        print(
            "Candidate interrogator ready ✅"
        )

    return _interrogator


# ============================================================
# SERIALIZATION HELPER
# ============================================================

def convert_to_dict(
    value: Any,
):

    if hasattr(
        value,
        "model_dump",
    ):

        return value.model_dump()

    if isinstance(
        value,
        list,
    ):

        return [
            convert_to_dict(item)
            for item in value
        ]

    if isinstance(
        value,
        dict,
    ):

        return {
            key: convert_to_dict(val)
            for key, val in value.items()
        }

    return value


# ============================================================
# CASE STORAGE
# ============================================================

def case_file(
    case_id: str,
) -> Path:

    return (
        CASES_DIR
        / f"{case_id}.json"
    )


def save_case(
    result,
):

    path = case_file(
        result.case_id
    )

    payload = convert_to_dict(
        result
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            payload,
            f,
            indent=2,
            ensure_ascii=False,
        )


def load_case(
    case_id: str,
):

    path = case_file(
        case_id
    )

    if not path.exists():

        return None

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


# ============================================================
# EVIDENCE FORMATTER
# ============================================================

def format_retrieved_evidence(
    results,
):

    formatted = []

    for item in results:

        formatted.append(
            RetrievedEvidence(

                chunk_id=item.get(
                    "chunk_id",
                    "",
                ),

                document_id=item.get(
                    "document_id",
                    "",
                ),

                chapter_number=item.get(
                    "chapter_number",
                ),

                chapter_heading=item.get(
                    "chapter_heading",
                ),

                pdf_page_start=item.get(
                    "pdf_page_start",
                ),

                pdf_page_end=item.get(
                    "pdf_page_end",
                ),

                text=item.get(
                    "text",
                    "",
                ),

                hybrid_score=item.get(
                    "hybrid_score",
                ),

                bm25_score=item.get(
                    "bm25_score",
                ),

                semantic_score=item.get(
                    "semantic_score",
                ),
            )
        )

    return formatted


# ============================================================
# GEMINI ERROR HANDLER
# ============================================================

def raise_gemini_error(
    exc: Exception,
):

    message = str(exc)

    if (
        "GEMINI_QUOTA_EXCEEDED"
        in message
        or "429"
        in message
        or "quota"
        in message.lower()
        or "rate limit"
        in message.lower()
    ):

        raise HTTPException(
            status_code=429,
            detail=(
                "Gemini quota/rate limit "
                "is currently exhausted. "
                "Wait for quota reset or "
                "use an API plan with "
                "available quota."
            ),
        )

    raise HTTPException(
        status_code=500,
        detail=message,
    )


# ============================================================
# HEALTH
# ============================================================

@router.get(
    "/health",
    response_model=HealthResponse,
)
def health():

    return {
        "status": "ok",

        "service": (
            "Agentic RAG "
            "Investigation API"
        ),

        "retriever": (
            "hybrid-bm25-semantic"
        ),

        "model": (
            "gemini-3.6-flash"
        ),
    }


# ============================================================
# CORPUS
# ============================================================

@router.get(
    "/corpus",
    response_model=CorpusResponse,
)
def corpus():

    if not METADATA_FILE.exists():

        raise HTTPException(
            status_code=404,
            detail=(
                "Corpus metadata not found."
            ),
        )

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        metadata = json.load(f)

    return {

        "document_id": metadata.get(
            "document_id",
            "sof",
        ),

        "title": metadata.get(
            "title",
            "The Sign of the Four",
        ),

        "source": metadata.get(
            "source",
        ),

        "total_pages": metadata.get(
            "total_pages",
        ),

        "novel_pages": metadata.get(
            "novel_pages",
        ),

        "total_chunks": metadata.get(
            "total_chunks",
            181,
        ),

        "embedding_dimension": metadata.get(
            "embedding_dimension",
            384,
        ),
    }


# ============================================================
# RETRIEVE / EVIDENCE SEARCH
# ============================================================

@router.post(
    "/retrieve",
    response_model=RetrieveResponse,
)
def retrieve(
    request: RetrieveRequest,
):

    try:

        retriever = get_retriever()

        results = retriever.search(
            request.query,
            top_k=request.top_k,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Retrieval failed: {exc}"
            ),
        )

    formatted = (
        format_retrieved_evidence(
            results
        )
    )

    return {

        "query": request.query,

        "results": formatted,

        "count": len(formatted),
    }


# ============================================================
# CANDIDATES
# ============================================================

@router.get(
    "/candidates",
    response_model=CandidateListResponse,
)
def candidates():

    candidates_data = [

        Candidate(
            candidate_id=(
                "jonathan_small"
            ),

            name="Jonathan Small",

            description=(
                "Central figure connected "
                "to the treasure mystery "
                "and investigation."
            ),

            evidence_ids=[],
        ),

        Candidate(
            candidate_id="tonga",

            name="Tonga",

            description=(
                "Jonathan Small's associate."
            ),

            evidence_ids=[],
        ),

        Candidate(
            candidate_id=(
                "thaddeus_sholto"
            ),

            name="Thaddeus Sholto",

            description=(
                "Son of Major Sholto and "
                "participant in the "
                "investigation."
            ),

            evidence_ids=[],
        ),

        Candidate(
            candidate_id=(
                "mary_morstan"
            ),

            name="Mary Morstan",

            description=(
                "Central figure connected "
                "to the treasure mystery."
            ),

            evidence_ids=[],
        ),
    ]

    return {
        "candidates": candidates_data
    }


# ============================================================
# INVESTIGATION
# ============================================================

@router.post(
    "/investigate",
    response_model=InvestigationResponse,
)
def investigate(
    request: InvestigationRequest,
):

    question = (
        request.question.strip()
    )

    if not question:

        raise HTTPException(
            status_code=400,
            detail=(
                "Investigation question "
                "cannot be empty."
            ),
        )

    try:

        orchestrator = (
            get_orchestrator()
        )

        # IMPORTANT:
        # Current production orchestrator
        # uses .investigate()
        result = (
            orchestrator.investigate(
                question
            )
        )

    except RuntimeError as exc:

        raise_gemini_error(
            exc
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Investigation failed: "
                f"{exc}"
            ),
        )

    # ========================================================
    # SAVE CASE
    # ========================================================

    try:

        save_case(
            result
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Investigation completed, "
                "but case persistence failed: "
                f"{exc}"
            ),
        )

    # ========================================================
    # BUILD GRAPH
    # ========================================================

    try:

        from backend.graph.graph_builder import (
            EvidenceGraphBuilder,
        )

        graph_builder = (
            EvidenceGraphBuilder()
        )

        graph_builder.build_from_result(
            result
        )

        graph_builder.save_json()

        graph = (
            graph_builder.export_dict()
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Investigation succeeded, "
                "but graph generation failed: "
                f"{exc}"
            ),
        )

    # ========================================================
    # RESPONSE
    # ========================================================

    return {

        "case_id": result.case_id,

        "question": result.question,

        "investigator": (
            convert_to_dict(
                result.investigator
            )
        ),

        "fact_check": (
            convert_to_dict(
                result.fact_check
            )
        ),

        "verdict": (
            convert_to_dict(
                result.verdict
            )
        ),

        "evidence": (
            convert_to_dict(
                result.evidence
            )
        ),

        "execution_trace": (
            convert_to_dict(
                result.execution_trace
            )
        ),

        "graph": graph,
    }


# ============================================================
# CANDIDATE INTERROGATION
# ============================================================

@router.post(
    "/interrogate",
    response_model=InterrogateResponse,
)
def interrogate(
    request: InterrogateRequest,
):

    candidate = (
        request.candidate.strip()
    )

    question = (
        request.question.strip()
    )

    if not candidate:

        raise HTTPException(
            status_code=400,
            detail=(
                "Candidate cannot be empty."
            ),
        )

    if not question:

        raise HTTPException(
            status_code=400,
            detail=(
                "Interrogation question "
                "cannot be empty."
            ),
        )

    try:

        interrogator = (
            get_interrogator()
        )

        answer, results = (
            interrogator.interrogate(
                candidate=candidate,
                question=question,
            )
        )

    except RuntimeError as exc:

        raise_gemini_error(
            exc
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Interrogation failed: "
                f"{exc}"
            ),
        )

    formatted = (
        format_retrieved_evidence(
            results
        )
    )

    return {

        "candidate": candidate,

        "question": question,

        "answer": answer,

        "evidence": formatted,
    }


# ============================================================
# GET SAVED INVESTIGATION
# ============================================================

@router.get(
    "/investigate/{case_id}",
)
def get_investigation(
    case_id: str,
):

    result = load_case(
        case_id
    )

    if result is None:

        raise HTTPException(
            status_code=404,
            detail=(
                "Investigation case not found."
            ),
        )

    return result


# ============================================================
# FACT CHECK
#
# We intentionally expose the already-computed
# Fact Checker result instead of calling Gemini again.
#
# This avoids wasting quota and keeps the UI/API
# consistent with the main investigation.
# ============================================================

@router.get(
    "/fact-check/{case_id}",
)
def get_fact_check(
    case_id: str,
):

    result = load_case(
        case_id
    )

    if result is None:

        raise HTTPException(
            status_code=404,
            detail=(
                "Investigation case not found."
            ),
        )

    return {

        "case_id": case_id,

        "question": result.get(
            "question"
        ),

        "fact_check": result.get(
            "fact_check",
            {},
        ),

        "evidence": result.get(
            "evidence",
            [],
        ),
    }


# ============================================================
# SUBMIT FINAL VERDICT
#
# The user can submit their own suspect/reasoning
# after exploring the evidence.
# ============================================================

@router.post(
    "/submit-verdict",
    response_model=SubmitVerdictResponse,
)
def submit_verdict(
    request: SubmitVerdictRequest,
):

    case = load_case(
        request.case_id
    )

    if case is None:

        raise HTTPException(
            status_code=404,
            detail=(
                "Investigation case not found."
            ),
        )

    user_suspect = (
        request.suspect.strip()
    )

    user_suspect_lower = (
        user_suspect.lower()
    )

    agent_verdict = (
        case
        .get("verdict", {})
        .get("verdict", "")
    )

    agent_verdict_lower = (
        agent_verdict.lower()
    )

    # --------------------------------------------------------
    # Match candidate against final verdict text
    # --------------------------------------------------------

    verdict_match = (
        user_suspect_lower
        in agent_verdict_lower
    )

    if verdict_match:

        evaluation = (
            "Your prediction matches "
            "the candidate identified "
            "by the investigation."
        )

    else:

        evaluation = (
            "Your prediction does not "
            "match the final investigation "
            "verdict. Review the verified "
            "and contradicting evidence."
        )

    return {

        "case_id": request.case_id,

        "user_suspect": user_suspect,

        "agent_verdict": agent_verdict,

        "verdict_match": verdict_match,

        "evaluation": evaluation,

        "supporting_evidence_ids": (
            request
            .supporting_evidence_ids
        ),

        "agent_supporting_evidence_ids": (
            case
            .get("verdict", {})
            .get(
                "supporting_evidence_ids",
                [],
            )
        ),
    }


# ============================================================
# EVIDENCE GRAPH
# ============================================================

@router.get(
    "/graph",
)
def graph():

    if not GRAPH_FILE.exists():

        raise HTTPException(
            status_code=404,
            detail=(
                "Evidence graph not found. "
                "Run an investigation first."
            ),
        )

    with open(
        GRAPH_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)