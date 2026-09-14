import json

from backend.agents.gemini_client import GeminiClient
from backend.agents.orchestrator import (
    InvestigationOrchestrator,
)
from backend.retrieval.hybrid_retriever import (
    HybridRetriever,
)


def main():

    print()
    print("=" * 70)
    print("AGENTIC INVESTIGATION TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # SERVICES
    # --------------------------------------------------------

    retriever = HybridRetriever()

    gemini_client = GeminiClient()

    orchestrator = (
        InvestigationOrchestrator(
            retriever=retriever,
            gemini_client=gemini_client,
        )
    )

    # --------------------------------------------------------
    # QUESTION
    # --------------------------------------------------------

    question = (
        "Who killed Bartholomew Sholto?"
    )

    print()
    print("QUESTION:")
    print(question)

    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------

    result = orchestrator.investigate(
        question
    )

    # --------------------------------------------------------
    # INVESTIGATOR
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("INVESTIGATOR")
    print("=" * 70)

    print(
        result.investigator.model_dump_json(
            indent=2
        )
    )

    # --------------------------------------------------------
    # FACT CHECKER
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FACT CHECKER")
    print("=" * 70)

    print(
        result.fact_check.model_dump_json(
            indent=2
        )
    )

    # --------------------------------------------------------
    # FINAL VERDICT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    print(
        result.verdict.model_dump_json(
            indent=2
        )
    )

    # --------------------------------------------------------
    # TRACE
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("EXECUTION TRACE")
    print("=" * 70)

    for event in result.execution_trace:

        print(
            f"[{event.stage}] "
            f"{event.message}"
        )

        if event.query:

            print(
                f"  Query: {event.query}"
            )

        if event.evidence_ids:

            print(
                f"  Evidence: "
                f"{event.evidence_ids}"
            )

    # --------------------------------------------------------
    # EVIDENCE
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("EVIDENCE")
    print("=" * 70)

    for item in result.evidence:

        print(
            f"{item.chunk_id} | "
            f"Chapter {item.chapter} | "
            f"Pages {item.pdf_pages} | "
            f"Status: {item.evidence_status}"
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    with open(
        "agent_result.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result.model_dump(),
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        "Saved -> agent_result.json"
    )

    print()
    print("=" * 70)
    print(
        "AGENTIC INVESTIGATION TEST PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()