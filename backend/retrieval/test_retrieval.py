from hybrid_retriever import HybridRetriever


# ============================================================
# PRINT RESULT
# ============================================================

def print_result(
    rank,
    result
):

    print()
    print("-" * 80)

    print(
        f"HYBRID RANK : {rank}"
    )

    print(
        f"Chunk       : "
        f"{result.get('chunk_id')}"
    )

    print(
        f"Chapter     : "
        f"{result.get('chapter_number')}"
    )

    print(
        f"Pages       : "
        f"{result.get('pdf_page_start')} - "
        f"{result.get('pdf_page_end')}"
    )

    print(
        f"Hybrid      : "
        f"{result.get('hybrid_score', 0):.6f}"
    )

    print(
        f"BM25        : "
        f"{result.get('bm25_score', 0):.4f}"
    )

    print(
        f"Semantic    : "
        f"{result.get('semantic_score', 0):.4f}"
    )

    print(
        f"BM25 Rank   : "
        f"{result.get('bm25_rank', '-')}"
    )

    print(
        f"Semantic Rank: "
        f"{result.get('semantic_rank', '-')}"
    )

    # --------------------------------------------------------
    # Text
    # --------------------------------------------------------

    text = result.get(
        "text",
        ""
    )

    if len(text) > 700:

        text = (
            text[:700]
            + "..."
        )

    print("\nEVIDENCE:")
    print(text)


# ============================================================
# TEST
# ============================================================

def main():

    print(
        "\n"
        + "=" * 80
    )

    print(
        "HYBRID RETRIEVAL TEST"
    )

    print(
        "=" * 80
    )

    # --------------------------------------------------------
    # Initialize
    # --------------------------------------------------------

    retriever = (
        HybridRetriever()
    )

    # --------------------------------------------------------
    # Test questions
    # --------------------------------------------------------

    queries = [

        "Who killed Bartholomew Sholto?",

        "How did the wooden-legged man enter the room?",

        "What happened to the treasure?",

        "Who is Mary Morstan?",

        "What happened on the steamship Aurora?"

    ]

    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------

    for query in queries:

        print(
            "\n\n"
            + "#" * 80
        )

        print(
            f"QUERY: {query}"
        )

        print(
            "#" * 80
        )

        results = (
            retriever.search(
                query=query,
                top_k=5
            )
        )

        for rank, result in enumerate(
            results,
            start=1
        ):

            print_result(
                rank,
                result
            )


if __name__ == "__main__":

    main()