from bm25_retriever import BM25Retriever
from semantic_retriever import SemanticRetriever


class HybridRetriever:

    def __init__(self):

        print("\n" + "=" * 60)
        print("INITIALIZING HYBRID RETRIEVER")
        print("=" * 60)

        self.bm25 = BM25Retriever()
        self.semantic = SemanticRetriever()

        print("\nHybrid retriever ready ✅")

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query: str,
        top_k: int = 8,
        candidate_k: int = 20,
        rrf_k: int = 60
    ):

        # ----------------------------------------------------
        # 1. BM25
        # ----------------------------------------------------

        bm25_results = self.bm25.search(
            query,
            top_k=candidate_k
        )

        # ----------------------------------------------------
        # 2. Semantic
        # ----------------------------------------------------

        semantic_results = self.semantic.search(
            query,
            top_k=candidate_k
        )

        # ----------------------------------------------------
        # 3. FUSION
        # ----------------------------------------------------

        fused = {}

        # ====================================================
        # BM25 RESULTS
        # ====================================================

        for result in bm25_results:

            chunk_id = result["chunk_id"]

            if chunk_id not in fused:

                fused[chunk_id] = {
                    "chunk": result.copy(),
                    "bm25_rank": None,
                    "semantic_rank": None,
                    "bm25_score": None,
                    "semantic_score": None,
                    "rrf_score": 0.0
                }

            fused[chunk_id]["bm25_rank"] = (
                result["bm25_rank"]
            )

            fused[chunk_id]["bm25_score"] = (
                result["bm25_score"]
            )

            fused[chunk_id]["rrf_score"] += (
                1.0 /
                (rrf_k + result["bm25_rank"])
            )

        # ====================================================
        # SEMANTIC RESULTS
        # ====================================================

        for result in semantic_results:

            chunk_id = result["chunk_id"]

            if chunk_id not in fused:

                fused[chunk_id] = {
                    "chunk": result.copy(),
                    "bm25_rank": None,
                    "semantic_rank": None,
                    "bm25_score": None,
                    "semantic_score": None,
                    "rrf_score": 0.0
                }

            fused[chunk_id]["semantic_rank"] = (
                result["semantic_rank"]
            )

            fused[chunk_id]["semantic_score"] = (
                result["semantic_score"]
            )

            fused[chunk_id]["rrf_score"] += (
                1.0 /
                (rrf_k + result["semantic_rank"])
            )

        # ====================================================
        # 4. SORT
        # ====================================================

        ranked = sorted(
            fused.values(),
            key=lambda x: x["rrf_score"],
            reverse=True
        )

        # ====================================================
        # 5. FINAL RESULTS
        # ====================================================

        final_results = []

        for final_rank, item in enumerate(
            ranked[:top_k],
            start=1
        ):

            result = item["chunk"].copy()

            result["hybrid_rank"] = final_rank
            result["hybrid_score"] = float(
                item["rrf_score"]
            )

            result["bm25_rank"] = (
                item["bm25_rank"]
            )

            result["semantic_rank"] = (
                item["semantic_rank"]
            )

            result["bm25_score"] = (
                item["bm25_score"]
            )

            result["semantic_score"] = (
                item["semantic_score"]
            )

            final_results.append(result)

        return final_results