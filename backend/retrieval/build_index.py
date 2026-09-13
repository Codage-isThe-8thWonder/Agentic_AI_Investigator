from pathlib import Path
import json

import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# PROJECT PATHS
# ============================================================

# build_index.py
#      ↓
# backend/retrieval/
#      ↓
# project root = agentic-investigation/

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHUNKS_FILE = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "semantic_embeddings.npy"

MODEL_NAME = "all-MiniLM-L6-v2"


# ============================================================
# LOAD CHUNKS
# ============================================================

def load_chunks():

    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"\nchunks.jsonl not found!\n"
            f"Expected location:\n{CHUNKS_FILE}\n"
        )

    chunks = []

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:

        for line in f:

            if line.strip():
                chunks.append(json.loads(line))

    return chunks


# ============================================================
# BUILD SEMANTIC INDEX
# ============================================================

def main():

    print("=" * 60)
    print("BUILDING SEMANTIC INDEX")
    print("=" * 60)

    print(f"\nProject root : {PROJECT_ROOT}")
    print(f"Chunks file  : {CHUNKS_FILE}")
    print(f"Output file  : {OUTPUT_FILE}")

    # --------------------------------------------------------
    # Load chunks
    # --------------------------------------------------------

    chunks = load_chunks()

    print(f"\nChunks loaded : {len(chunks)}")

    if len(chunks) == 0:
        raise ValueError("chunks.jsonl is empty!")

    # --------------------------------------------------------
    # Extract text
    # --------------------------------------------------------

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    # --------------------------------------------------------
    # Load embedding model
    # --------------------------------------------------------

    print(f"\nLoading model : {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    # --------------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------------

    print("\nGenerating embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        OUTPUT_FILE,
        embeddings
    )

    # --------------------------------------------------------
    # Verification
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("INDEX BUILD COMPLETE")
    print("=" * 60)

    print(f"Chunks           : {len(chunks)}")
    print(f"Embedding shape  : {embeddings.shape}")
    print(f"Saved to         : {OUTPUT_FILE}")

    print("\nDONE ✅")


if __name__ == "__main__":
    main()