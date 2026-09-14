import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHUNKS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chunks.jsonl"
)

EMBEDDINGS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "semantic_embeddings.npy"
)


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "all-MiniLM-L6-v2"


# ============================================================
# SEMANTIC RETRIEVER
# ============================================================

class SemanticRetriever:

    def __init__(
        self,
        chunks_file=CHUNKS_FILE,
        embeddings_file=EMBEDDINGS_FILE,
        model_name=MODEL_NAME
    ):

        self.chunks_file = Path(
            chunks_file
        )

        self.embeddings_file = Path(
            embeddings_file
        )

        # Load chunks
        self.chunks = (
            self._load_chunks()
        )

        # Check embeddings
        if not self.embeddings_file.exists():

            raise FileNotFoundError(
                f"\nSemantic embeddings not found:\n"
                f"{self.embeddings_file}\n\n"
                f"Run:\n"
                f"python backend\\retrieval\\build_index.py"
            )

        # Load embeddings
        self.embeddings = np.load(
            self.embeddings_file
        )

        # Validate count
        if len(self.chunks) != len(
            self.embeddings
        ):

            raise ValueError(
                f"Chunk count ({len(self.chunks)}) "
                f"does not match embedding count "
                f"({len(self.embeddings)})"
            )

        # Load model
        print(
            f"Loading semantic model: "
            f"{model_name}"
        )

        self.model = SentenceTransformer(
            model_name,
            device="cpu"
        )

        # Normalize embeddings
        norms = np.linalg.norm(
            self.embeddings,
            axis=1,
            keepdims=True
        )

        self.embeddings = (
            self.embeddings
            / np.maximum(norms, 1e-12)
        )

        print(
            f"Semantic retriever loaded: "
            f"{len(self.chunks)} chunks"
        )

    # --------------------------------------------------------
    # Load chunks
    # --------------------------------------------------------

    def _load_chunks(self):

        if not self.chunks_file.exists():

            raise FileNotFoundError(
                f"Chunks file not found:\n"
                f"{self.chunks_file}"
            )

        chunks = []

        with open(
            self.chunks_file,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                if line.strip():

                    chunks.append(
                        json.loads(line)
                    )

        return chunks

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 10
    ):

        # Query embedding
        query_embedding = (
            self.model.encode(
                [query],
                normalize_embeddings=True
            )[0]
        )

        # Cosine similarity
        scores = np.dot(
            self.embeddings,
            query_embedding
        )

        # Highest first
        ranked_indices = np.argsort(
            scores
        )[::-1][:top_k]

        results = []

        for rank, idx in enumerate(
            ranked_indices,
            start=1
        ):

            idx = int(idx)

            result = (
                self.chunks[idx].copy()
            )

            result[
                "semantic_score"
            ] = float(scores[idx])

            result[
                "semantic_rank"
            ] = rank

            results.append(result)

        return results