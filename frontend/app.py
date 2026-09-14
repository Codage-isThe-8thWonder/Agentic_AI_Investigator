from __future__ import annotations

import os
from typing import Any

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


# ============================================================
# CONFIG
# ============================================================

API_URL = os.getenv(
    "BACKEND_URL",
    "http://127.0.0.1:8000",
).rstrip("/")


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Agentic Investigation",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1500px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 1.8rem;
        border-radius: 18px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 1.5rem;
    }

    .hero-title {
        font-size: 2.35rem;
        font-weight: 800;
        margin-bottom: 0.4rem;
    }

    .hero-subtitle {
        font-size: 1rem;
        opacity: 0.72;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        margin-top: 1rem;
    }

    .status-box {
        padding: 0.9rem;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 0.6rem;
    }

    .small {
        font-size: 0.8rem;
        opacity: 0.7;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "investigation_result": None,
    "last_question": "",
    "prediction_suspect": "",
    "prediction_reasoning": "",
    "prediction_evidence": [],
    "prediction_submitted": False,
    "verdict_submission": None,
    "search_results": None,
    "interrogation_result": None,
    "graph_data": None,
}


for key, value in DEFAULT_STATE.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# API HELPERS
# ============================================================

def api_get(
    endpoint: str,
    timeout: int = 30,
):
    response = requests.get(
        f"{API_URL}{endpoint}",
        timeout=timeout,
    )

    response.raise_for_status()

    return response.json()


def api_post(
    endpoint: str,
    payload: dict[str, Any],
    timeout: int = 120,
):
    response = requests.post(
        f"{API_URL}{endpoint}",
        json=payload,
        timeout=timeout,
    )

    response.raise_for_status()

    return response.json()


def get_error_message(
    exc: Exception,
) -> str:

    if isinstance(
        exc,
        requests.HTTPError,
    ):

        try:

            data = exc.response.json()

            return data.get(
                "detail",
                str(exc),
            )

        except Exception:

            return str(exc)

    return str(exc)


# ============================================================
# BACKEND STATUS
# ============================================================

def backend_status():

    try:

        data = api_get(
            "/api/health",
            timeout=5,
        )

        return True, data

    except Exception as exc:

        return False, str(exc)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-title">
            🕵️ Agentic RAG Investigation
        </div>

        <div class="hero-subtitle">
            Investigate • Interrogate • Search Evidence •
            Fact Check • Submit Verdict • Explore Graph
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ System")

    online, health_data = backend_status()

    if online:

        st.success("🟢 Backend Online")

        st.caption(
            f"Model: "
            f"{health_data.get('model', '-')}"
        )

        st.caption(
            f"Retriever: "
            f"{health_data.get('retriever', '-')}"
        )

    else:

        st.error("🔴 Backend Offline")

        st.code(
            "uvicorn backend.main:app --reload"
        )

    st.divider()

    st.header("📚 Corpus")

    try:

        corpus = api_get(
            "/api/corpus",
            timeout=5,
        )

        st.write(
            f"**{corpus.get('title', '-')}"
        )

        st.caption(
            f"Document: "
            f"{corpus.get('document_id', '-')}"
        )

        c1, c2 = st.columns(2)

        with c1:

            st.metric(
                "Chunks",
                corpus.get(
                    "total_chunks",
                    "-",
                ),
            )

        with c2:

            st.metric(
                "Dim",
                corpus.get(
                    "embedding_dimension",
                    "-",
                ),
            )

    except Exception:

        st.warning(
            "Corpus information unavailable."
        )

    st.divider()

    st.caption(
        "Hybrid RAG + Investigator + "
        "Fact Checker + Evidence Graph"
    )


# ============================================================
# MAIN TABS
# ============================================================

case_tab, search_tab, interrogation_tab, graph_tab = (
    st.tabs(
        [
            "🕵️ Case Room",
            "🔎 Evidence Search",
            "👤 Candidate Interrogation",
            "🕸️ Evidence Graph",
        ]
    )
)


# ============================================================
# CASE ROOM
# ============================================================

with case_tab:

    # ========================================================
    # STEP 1 — USER PREDICTION
    # ========================================================

    st.header("🎯 1. Make Your Prediction")

    st.write(
        "Before seeing the complete investigation result, "
        "make your own prediction and explain your reasoning."
    )

    try:

        candidate_response = api_get(
            "/api/candidates",
            timeout=10,
        )

        candidate_objects = (
            candidate_response.get(
                "candidates",
                [],
            )
        )

        candidate_names = [
            candidate.get(
                "name",
                "",
            )
            for candidate in candidate_objects
        ]

    except Exception:

        candidate_objects = []

        candidate_names = [
            "Jonathan Small",
            "Tonga",
            "Thaddeus Sholto",
            "Mary Morstan",
        ]

    prediction_options = [
        "— Select a candidate —"
    ] + candidate_names

    current_prediction = (
        st.session_state.prediction_suspect
    )

    if current_prediction in candidate_names:

        default_index = (
            prediction_options.index(
                current_prediction
            )
        )

    else:

        default_index = 0

    selected_candidate = st.selectbox(
        "Who do you think is responsible?",
        prediction_options,
        index=default_index,
        key="prediction_candidate_widget",
    )

    prediction_reasoning = st.text_area(
        "Why do you think so?",
        value=st.session_state.prediction_reasoning,
        placeholder=(
            "Explain your reasoning using evidence "
            "you have found so far..."
        ),
        height=120,
        key="prediction_reasoning_widget",
    )

    prediction_evidence_text = st.text_input(
        "Supporting evidence IDs (optional)",
        value=", ".join(
            st.session_state.prediction_evidence
        ),
        placeholder=(
            "Example: sof_0083, sof_0091"
        ),
    )

    if st.button(
        "💾 Save My Prediction",
        type="secondary",
        use_container_width=True,
    ):

        if (
            selected_candidate
            == "— Select a candidate —"
        ):

            st.warning(
                "Please select a candidate."
            )

        elif not prediction_reasoning.strip():

            st.warning(
                "Please provide your reasoning."
            )

        else:

            st.session_state.prediction_suspect = (
                selected_candidate
            )

            st.session_state.prediction_reasoning = (
                prediction_reasoning.strip()
            )

            st.session_state.prediction_evidence = [
                x.strip()
                for x in prediction_evidence_text.split(
                    ","
                )
                if x.strip()
            ]

            st.session_state.prediction_submitted = (
                True
            )

            st.success(
                "Prediction saved. Now investigate the case."
            )

    if st.session_state.prediction_submitted:

        st.info(
            f"Your prediction: "
            f"**{st.session_state.prediction_suspect}**"
        )

    st.divider()

    # ========================================================
    # STEP 2 — INVESTIGATION QUESTION
    # ========================================================

    st.header("🧠 2. Start Agentic Investigation")

    question = st.text_area(
        "Investigation question",
        value=st.session_state.last_question,
        placeholder=(
            "Example: Who killed Bartholomew Sholto?"
        ),
        height=100,
        key="investigation_question_widget",
    )

    col1, col2 = st.columns(2)

    with col1:

        investigate_clicked = st.button(
            "🚀 Start Investigation",
            type="primary",
            use_container_width=True,
        )

    with col2:

        clear_clicked = st.button(
            "🧹 Clear Case",
            use_container_width=True,
        )

    if clear_clicked:

        for key, value in DEFAULT_STATE.items():

            st.session_state[key] = value

        st.rerun()

    # ========================================================
    # RUN INVESTIGATION
    # ========================================================

    if investigate_clicked:

        if not question.strip():

            st.warning(
                "Enter an investigation question."
            )

        else:

            st.session_state.last_question = (
                question.strip()
            )

            with st.spinner(
                "🧠 Investigator → Fact Checker → "
                "Final Verdict..."
            ):

                try:

                    result = api_post(
                        "/api/investigate",
                        {
                            "question": (
                                question.strip()
                            )
                        },
                        timeout=240,
                    )

                    st.session_state.investigation_result = (
                        result
                    )

                    st.success(
                        "Investigation completed successfully."
                    )

                except Exception as exc:

                    st.error(
                        get_error_message(exc)
                    )

    result = (
        st.session_state.investigation_result
    )

    # ========================================================
    # RESULTS
    # ========================================================

    if result:

        verdict = result.get(
            "verdict",
            {},
        )

        investigator = result.get(
            "investigator",
            {},
        )

        fact_check = result.get(
            "fact_check",
            {},
        )

        evidence = result.get(
            "evidence",
            [],
        )

        trace = result.get(
            "execution_trace",
            [],
        )

        case_id = result.get(
            "case_id",
            "",
        )

        st.divider()

        # ====================================================
        # OVERVIEW
        # ====================================================

        st.header("📊 Investigation Overview")

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.metric(
                "Confidence",
                str(
                    verdict.get(
                        "confidence",
                        "UNKNOWN",
                    )
                ).upper(),
            )

        with c2:

            st.metric(
                "Evidence",
                len(evidence),
            )

        with c3:

            st.metric(
                "Fact Check",
                str(
                    fact_check.get(
                        "overall_status",
                        "UNKNOWN",
                    )
                ),
            )

        with c4:

            st.metric(
                "Retrieval Rounds",
                investigator.get(
                    "retrieval_rounds",
                    "-",
                ),
            )

        st.caption(
            f"Case ID: `{case_id}`"
        )

        # ====================================================
        # FINAL VERDICT
        # ====================================================

        st.header("⚖️ Final Verdict")

        agent_verdict = verdict.get(
            "verdict",
            "No verdict available.",
        )

        st.success(
            agent_verdict
        )

        explanation = verdict.get(
            "explanation",
            "",
        )

        if explanation:

            st.write(
                explanation
            )

        # ====================================================
        # USER VS AGENT
        # ====================================================

        if st.session_state.prediction_submitted:

            st.subheader(
                "🎯 Your Prediction vs Agent Verdict"
            )

            p1, p2 = st.columns(2)

            with p1:

                st.markdown(
                    "**Your Prediction**"
                )

                st.info(
                    st.session_state.prediction_suspect
                )

                st.caption(
                    st.session_state.prediction_reasoning
                )

            with p2:

                st.markdown(
                    "**Agent Verdict**"
                )

                st.success(
                    agent_verdict
                )

            if st.button(
                "📤 Submit Final Verdict",
                type="primary",
                use_container_width=True,
            ):

                try:

                    verdict_response = api_post(
                        "/api/submit-verdict",
                        {
                            "case_id": case_id,
                            "suspect": (
                                st.session_state
                                .prediction_suspect
                            ),
                            "reasoning": (
                                st.session_state
                                .prediction_reasoning
                            ),
                            "supporting_evidence_ids": (
                                st.session_state
                                .prediction_evidence
                            ),
                        },
                        timeout=30,
                    )

                    st.session_state.verdict_submission = (
                        verdict_response
                    )

                except Exception as exc:

                    st.error(
                        get_error_message(exc)
                    )

            submission = (
                st.session_state.verdict_submission
            )

            if submission:

                if submission.get(
                    "verdict_match",
                    False,
                ):

                    st.success(
                        "🎉 Your prediction matches "
                        "the agent verdict."
                    )

                else:

                    st.warning(
                        "Your prediction differs from "
                        "the agent verdict."
                    )

                st.write(
                    submission.get(
                        "evaluation",
                        "",
                    )
                )

        # ====================================================
        # INVESTIGATOR
        # ====================================================

        with st.expander(
            "🧠 Investigator Analysis",
            expanded=True,
        ):

            st.markdown(
                "**Initial / Refined Hypothesis**"
            )

            st.write(
                investigator.get(
                    "hypothesis",
                    "",
                )
            )

            st.markdown(
                "**Reasoning Summary**"
            )

            st.write(
                investigator.get(
                    "reasoning_summary",
                    "",
                )
            )

            st.markdown(
                "**Answer Complete**"
            )

            st.write(
                investigator.get(
                    "answer_complete",
                    False,
                )
            )

            missing = investigator.get(
                "missing_information",
                [],
            )

            if missing:

                st.markdown(
                    "**Missing Information**"
                )

                for item in missing:

                    st.write(
                        f"• {item}"
                    )

            refined_query = investigator.get(
                "refined_query"
            )

            if refined_query:

                st.markdown(
                    "**Refined Query**"
                )

                st.code(
                    refined_query
                )

        # ====================================================
        # FACT CHECKER
        # ====================================================

        with st.expander(
            "⚔️ Adversarial Fact Checker",
            expanded=True,
        ):

            status = str(
                fact_check.get(
                    "overall_status",
                    "UNVERIFIED",
                )
            ).upper()

            if status == "SUPPORTED":

                st.success(
                    f"Overall status: {status}"
                )

            elif status == "CONTRADICTED":

                st.error(
                    f"Overall status: {status}"
                )

            elif status == "MIXED":

                st.warning(
                    f"Overall status: {status}"
                )

            else:

                st.info(
                    f"Overall status: {status}"
                )

            claims = fact_check.get(
                "claims",
                [],
            )

            for i, claim in enumerate(
                claims,
                start=1,
            ):

                st.markdown(
                    f"### Claim {i}"
                )

                st.write(
                    claim.get(
                        "claim",
                        "",
                    )
                )

                claim_status = str(
                    claim.get(
                        "status",
                        "UNVERIFIED",
                    )
                ).upper()

                if claim_status == "SUPPORTED":

                    st.success(
                        claim_status
                    )

                elif claim_status == "CONTRADICTED":

                    st.error(
                        claim_status
                    )

                else:

                    st.warning(
                        claim_status
                    )

                st.caption(
                    claim.get(
                        "explanation",
                        "",
                    )
                )

                ids = claim.get(
                    "evidence_ids",
                    [],
                )

                if ids:

                    st.caption(
                        "Evidence IDs: "
                        + ", ".join(ids)
                    )

        # ====================================================
        # EVIDENCE
        # ====================================================

        st.header("📑 Evidence Board")

        if evidence:

            statuses = sorted(
                {
                    str(
                        item.get(
                            "evidence_status",
                            "unverified",
                        )
                    ).lower()
                    for item in evidence
                }
            )

            selected_statuses = st.multiselect(
                "Evidence status",
                statuses,
                default=statuses,
            )

            filtered = [
                item
                for item in evidence
                if str(
                    item.get(
                        "evidence_status",
                        "unverified",
                    )
                ).lower()
                in selected_statuses
            ]

            for item in filtered:

                evidence_id = item.get(
                    "chunk_id",
                    "-",
                )

                evidence_status = str(
                    item.get(
                        "evidence_status",
                        "unverified",
                    )
                ).upper()

                chapter = item.get(
                    "chapter",
                    item.get(
                        "chapter_number",
                        "-",
                    ),
                )

                pages = item.get(
                    "pdf_pages",
                    [],
                )

                with st.expander(
                    f"{evidence_id} • "
                    f"{evidence_status} • "
                    f"Chapter {chapter}"
                ):

                    c1, c2, c3 = st.columns(3)

                    with c1:

                        st.caption(
                            "Document"
                        )

                        st.write(
                            item.get(
                                "document_id",
                                "-",
                            )
                        )

                    with c2:

                        st.caption(
                            "Pages"
                        )

                        if pages:

                            st.write(
                                ", ".join(
                                    map(
                                        str,
                                        pages,
                                    )
                                )
                            )

                        else:

                            start = item.get(
                                "pdf_page_start",
                                "-",
                            )

                            end = item.get(
                                "pdf_page_end",
                                "-",
                            )

                            st.write(
                                f"{start}–{end}"
                            )

                    with c3:

                        st.caption(
                            "Retrieval"
                        )

                        st.write(
                            item.get(
                                "retrieval_source",
                                "-",
                            )
                        )

                    st.markdown(
                        "**Evidence Text**"
                    )

                    st.write(
                        item.get(
                            "text",
                            "",
                        )
                    )

        else:

            st.info(
                "No evidence available."
            )

        # ====================================================
        # EXECUTION TRACE
        # ====================================================

        st.header("🧭 Agent Execution Trace")

        if trace:

            for index, event in enumerate(
                trace,
                start=1,
            ):

                stage = event.get(
                    "stage",
                    "unknown",
                )

                message = event.get(
                    "message",
                    "",
                )

                query = event.get(
                    "query"
                )

                evidence_ids = event.get(
                    "evidence_ids",
                    [],
                )

                with st.expander(
                    f"{index}. {stage}"
                ):

                    st.write(
                        message
                    )

                    if query:

                        st.markdown(
                            "**Query**"
                        )

                        st.code(
                            query
                        )

                    if evidence_ids:

                        st.caption(
                            "Evidence: "
                            + ", ".join(
                                evidence_ids
                            )
                        )

        else:

            st.info(
                "No execution trace available."
            )


# ============================================================
# EVIDENCE SEARCH
# ============================================================

with search_tab:

    st.header("🔎 Hybrid Evidence Search")

    st.write(
        "Search the corpus directly using the "
        "BM25 + semantic hybrid retriever."
    )

    search_query = st.text_input(
        "Search query",
        placeholder=(
            "Example: Jonathan Small associate"
        ),
    )

    top_k = st.slider(
        "Results",
        min_value=1,
        max_value=20,
        value=5,
    )

    if st.button(
        "🔍 Search",
        type="primary",
        use_container_width=True,
    ):

        if not search_query.strip():

            st.warning(
                "Enter a search query."
            )

        else:

            with st.spinner(
                "Searching evidence..."
            ):

                try:

                    data = api_post(
                        "/api/retrieve",
                        {
                            "query": (
                                search_query.strip()
                            ),
                            "top_k": top_k,
                        },
                        timeout=60,
                    )

                    st.session_state.search_results = (
                        data
                    )

                except Exception as exc:

                    st.error(
                        get_error_message(exc)
                    )

    search_data = (
        st.session_state.search_results
    )

    if search_data:

        st.divider()

        st.write(
            f"**{search_data.get('count', 0)}** "
            "results found."
        )

        for i, item in enumerate(
            search_data.get(
                "results",
                [],
            ),
            start=1,
        ):

            with st.expander(
                f"{i}. "
                f"{item.get('chunk_id', '-')}"
            ):

                c1, c2, c3 = st.columns(3)

                with c1:

                    st.metric(
                        "Hybrid",
                        round(
                            item.get(
                                "hybrid_score",
                                0,
                            )
                            or 0,
                            4,
                        ),
                    )

                with c2:

                    st.metric(
                        "BM25",
                        round(
                            item.get(
                                "bm25_score",
                                0,
                            )
                            or 0,
                            4,
                        ),
                    )

                with c3:

                    st.metric(
                        "Semantic",
                        round(
                            item.get(
                                "semantic_score",
                                0,
                            )
                            or 0,
                            4,
                        ),
                    )

                st.caption(
                    f"Document: "
                    f"{item.get('document_id', '-')}"
                )

                st.caption(
                    f"Chapter: "
                    f"{item.get('chapter_number', '-')}"
                )

                st.caption(
                    f"Pages: "
                    f"{item.get('pdf_page_start', '-')}"
                    f"–"
                    f"{item.get('pdf_page_end', '-')}"
                )

                st.write(
                    item.get(
                        "text",
                        "",
                    )
                )


# ============================================================
# CANDIDATE INTERROGATION
# ============================================================

with interrogation_tab:

    st.header("👤 Candidate Interrogation")

    st.write(
        "Ask a candidate a question. The system retrieves "
        "corpus evidence and generates an evidence-grounded "
        "response."
    )

    try:

        candidate_response = api_get(
            "/api/candidates",
            timeout=10,
        )

        candidates = [
            candidate.get(
                "name",
                "",
            )
            for candidate in candidate_response.get(
                "candidates",
                [],
            )
        ]

    except Exception:

        candidates = [
            "Jonathan Small",
            "Tonga",
            "Thaddeus Sholto",
            "Mary Morstan",
        ]

    candidate = st.selectbox(
        "Candidate",
        candidates,
    )

    interrogation_question = st.text_area(
        "Question for the candidate",
        placeholder=(
            "Example: What was your relationship "
            "with Tonga?"
        ),
        height=100,
    )

    if st.button(
        "🎤 Interrogate Candidate",
        type="primary",
        use_container_width=True,
    ):

        if not interrogation_question.strip():

            st.warning(
                "Enter a question."
            )

        else:

            with st.spinner(
                "Retrieving evidence and interrogating..."
            ):

                try:

                    interrogation = api_post(
                        "/api/interrogate",
                        {
                            "candidate": candidate,
                            "question": (
                                interrogation_question.strip()
                            ),
                        },
                        timeout=180,
                    )

                    st.session_state.interrogation_result = (
                        interrogation
                    )

                except Exception as exc:

                    st.error(
                        get_error_message(exc)
                    )

    interrogation = (
        st.session_state.interrogation_result
    )

    if interrogation:

        st.divider()

        st.subheader(
            "💬 Grounded Response"
        )

        st.info(
            interrogation.get(
                "answer",
                "",
            )
        )

        st.subheader(
            "📑 Supporting Evidence"
        )

        for item in interrogation.get(
            "evidence",
            [],
        ):

            with st.expander(
                f"{item.get('chunk_id', '-')}"
            ):

                st.caption(
                    f"Document: "
                    f"{item.get('document_id', '-')}"
                )

                st.caption(
                    f"Chapter: "
                    f"{item.get('chapter_number', '-')}"
                )

                st.caption(
                    f"Pages: "
                    f"{item.get('pdf_page_start', '-')}"
                    f"–"
                    f"{item.get('pdf_page_end', '-')}"
                )

                st.write(
                    item.get(
                        "text",
                        "",
                    )
                )


# ============================================================
# GRAPH
# ============================================================

with graph_tab:

    st.header("🕸️ Evidence Graph")

    st.write(
        "Explore relationships between the investigation "
        "question, claims, evidence and entities."
    )

    if st.button(
        "🔄 Load Graph",
        type="primary",
        use_container_width=True,
    ):

        try:

            st.session_state.graph_data = (
                api_get(
                    "/api/graph",
                    timeout=30,
                )
            )

        except Exception as exc:

            st.error(
                get_error_message(exc)
            )

    graph_data = (
        st.session_state.graph_data
    )

    if graph_data:

        nodes = graph_data.get(
            "nodes",
            [],
        )

        edges = graph_data.get(
            "edges",
            [],
        )

        if nodes:

            G = nx.Graph()

            # ------------------------------------------------
            # Nodes
            # ------------------------------------------------

            for node in nodes:

                node_id = str(
                    node.get(
                        "id",
                        node.get(
                            "node_id",
                            "",
                        ),
                    )
                )

                attrs = {
                    key: value
                    for key, value in node.items()
                    if key not in {
                        "id",
                        "node_id",
                    }
                }

                G.add_node(
                    node_id,
                    **attrs,
                )

            # ------------------------------------------------
            # Edges
            # ------------------------------------------------

            for edge in edges:

                source = str(
                    edge.get(
                        "source",
                        "",
                    )
                )

                target = str(
                    edge.get(
                        "target",
                        "",
                    )
                )

                if source and target:

                    attrs = {
                        key: value
                        for key, value in edge.items()
                        if key not in {
                            "source",
                            "target",
                        }
                    }

                    G.add_edge(
                        source,
                        target,
                        **attrs,
                    )

            # ------------------------------------------------
            # Layout
            # ------------------------------------------------

            positions = nx.spring_layout(
                G,
                seed=42,
                k=1.4,
            )

            # ------------------------------------------------
            # Edges
            # ------------------------------------------------

            edge_x = []
            edge_y = []

            for source, target in G.edges():

                x0, y0 = positions[source]
                x1, y1 = positions[target]

                edge_x.extend(
                    [
                        x0,
                        x1,
                        None,
                    ]
                )

                edge_y.extend(
                    [
                        y0,
                        y1,
                        None,
                    ]
                )

            edge_trace = go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                hoverinfo="none",
                line={
                    "width": 1.2,
                },
            )

            # ------------------------------------------------
            # Nodes
            # ------------------------------------------------

            node_x = []
            node_y = []
            node_labels = []
            node_hover = []

            for node_id in G.nodes():

                x, y = positions[node_id]

                node_x.append(x)
                node_y.append(y)

                attrs = G.nodes[node_id]

                label = attrs.get(
                    "label",
                    node_id,
                )

                node_type = attrs.get(
                    "type",
                    attrs.get(
                        "node_type",
                        "unknown",
                    ),
                )

                node_labels.append(
                    str(label)[:35]
                )

                node_hover.append(
                    f"ID: {node_id}<br>"
                    f"Label: {label}<br>"
                    f"Type: {node_type}"
                )

            node_trace = go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                text=node_labels,
                textposition="top center",
                hovertext=node_hover,
                hoverinfo="text",
                marker={
                    "size": 18,
                },
            )

            # ------------------------------------------------
            # Figure
            # ------------------------------------------------

            fig = go.Figure(
                data=[
                    edge_trace,
                    node_trace,
                ]
            )

            fig.update_layout(
                height=700,
                margin=dict(
                    l=10,
                    r=10,
                    t=20,
                    b=10,
                ),
                showlegend=False,
                xaxis={
                    "showgrid": False,
                    "zeroline": False,
                    "showticklabels": False,
                },
                yaxis={
                    "showgrid": False,
                    "zeroline": False,
                    "showticklabels": False,
                },
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            # ------------------------------------------------
            # Stats
            # ------------------------------------------------

            c1, c2, c3 = st.columns(3)

            with c1:

                st.metric(
                    "Nodes",
                    len(G.nodes),
                )

            with c2:

                st.metric(
                    "Relationships",
                    len(G.edges),
                )

            with c3:

                st.metric(
                    "Components",
                    nx.number_connected_components(
                        G
                    ),
                )

            # ------------------------------------------------
            # Nodes table
            # ------------------------------------------------

            with st.expander(
                "📋 Graph Nodes"
            ):

                rows = []

                for node_id in G.nodes():

                    attrs = G.nodes[node_id]

                    rows.append(
                        {
                            "ID": node_id,
                            "Label": attrs.get(
                                "label",
                                "",
                            ),
                            "Type": attrs.get(
                                "type",
                                attrs.get(
                                    "node_type",
                                    "",
                                ),
                            ),
                        }
                    )

                st.dataframe(
                    pd.DataFrame(rows),
                    use_container_width=True,
                    hide_index=True,
                )

            # ------------------------------------------------
            # Edges table
            # ------------------------------------------------

            with st.expander(
                "🔗 Graph Relationships"
            ):

                rows = []

                for (
                    source,
                    target,
                    attrs,
                ) in G.edges(
                    data=True
                ):

                    rows.append(
                        {
                            "Source": source,
                            "Relationship": attrs.get(
                                "relation",
                                attrs.get(
                                    "type",
                                    "",
                                ),
                            ),
                            "Target": target,
                        }
                    )

                st.dataframe(
                    pd.DataFrame(rows),
                    use_container_width=True,
                    hide_index=True,
                )

        else:

            st.info(
                "Graph has no nodes."
            )

    else:

        st.info(
            "Run an investigation first, then load the graph."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Agentic RAG Investigation System • "
    "Hybrid Retrieval • Investigator • "
    "Adversarial Fact Checker • Evidence Graph"
)