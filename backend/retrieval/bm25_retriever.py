import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


BASE_DIR = Path(__file__).resolve().parents[2]
CHUNKS_FILE = BASE_DIR / "data" / "processed" / "chunks.jsonl"


def tokenize(text: str):
    """
    Simple normalized tokenizer for BM25.
    """
    return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())


class BM25Retriever:

    def __init__(self, chunks_file=CHUNKS_FILE):
        self.chunks_file = Path(chunks_file)

        self.chunks = self._load_chunks()

        self.tokenized_corpus = [
            tokenize(chunk["text"])
            for chunk in self.chunks
        ]

        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _load_chunks(self):
        chunks = []

        with open(self.chunks_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))

        return chunks

    def search(self, query: str, top_k: int = 10):

        query_tokens = tokenize(query)

        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        results = []

        for rank, idx in enumerate(ranked_indices, start=1):

            chunk = self.chunks[idx].copy()

            chunk["bm25_score"] = float(scores[idx])
            chunk["bm25_rank"] = rank

            results.append(chunk)

        return results