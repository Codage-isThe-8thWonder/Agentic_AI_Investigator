import json
from pathlib import Path

from backend.graph.graph_builder import (
    EvidenceGraphBuilder,
)


PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

RESULT_FILE = (
    PROJECT_ROOT
    / "agent_result.json"
)


def main():

    print("\n")
    print("=" * 70)
    print("EVIDENCE GRAPH TEST")
    print("=" * 70)

    if not RESULT_FILE.exists():

        raise FileNotFoundError(
            "agent_result.json not found. "
            "Run the agent test first."
        )

    with open(
        RESULT_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        result = json.load(f)

    print(
        "\nInvestigation result loaded ✅"
    )

    print(
        f"\nQuestion:\n"
        f"{result.get('question')}"
    )

    builder = (
        EvidenceGraphBuilder()
    )

    print(
        "\nBuilding evidence graph..."
    )

    builder.build_from_result(
        result
    )

    print(
        "Evidence graph built ✅"
    )

    stats = builder.statistics()

    print("\n")
    print("=" * 70)
    print("GRAPH STATISTICS")
    print("=" * 70)

    print(
        f"\nNodes: {stats['nodes']}"
    )

    print(
        f"Edges: {stats['edges']}"
    )

    print("\nNode types:")

    for key, value in (
        stats["node_types"].items()
    ):

        print(
            f"  {key}: {value}"
        )

    print("\nEdge types:")

    for key, value in (
        stats["edge_types"].items()
    ):

        print(
            f"  {key}: {value}"
        )

    print("\n")
    print("=" * 70)
    print("GRAPH EDGES")
    print("=" * 70)

    for (
        source,
        target,
        data,
    ) in builder.graph.edges(
        data=True
    ):

        print(
            f"{source} "
            f"--[{data.get('relation')}]--> "
            f"{target}"
        )

    output = (
        builder.save_json()
    )

    print(
        f"\nGraph saved → {output}"
    )

    print("\n")
    print("=" * 70)
    print(
        "EVIDENCE GRAPH TEST PASSED ✅"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()