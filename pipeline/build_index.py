"""Embed all chunks and save the vector store to Parquet."""
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CHUNKS_PATH, EMBEDDING_BATCH_SIZE, EMBEDDINGS_PATH
from core.embeddings import create_embeddings, load_chunks
from core.vector_store import VectorStore


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)

    if EMBEDDINGS_PATH.exists():
        logger.info("Index already exists at %s — delete it to rebuild.", EMBEDDINGS_PATH)
        return

    chunks = [c for c in load_chunks(str(CHUNKS_PATH)) if c.get("content", "").strip()]
    if not chunks:
        logger.error("No chunks found in %s — run pipeline/chunk_text.py first.", CHUNKS_PATH)
        return

    texts = [c["content"] for c in chunks]
    logger.info("Embedding %d chunks (model downloads on first run, ~420 MB)...", len(texts))
    start = time.time()

    all_embeddings: list = []
    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[i : i + EMBEDDING_BATCH_SIZE]
        all_embeddings.extend(create_embeddings(batch))
        logger.info(
            "Progress: %d/%d (%.1fs elapsed)",
            min(i + EMBEDDING_BATCH_SIZE, len(texts)),
            len(texts),
            time.time() - start,
        )

    store = VectorStore()
    for text, emb, meta in zip(texts, all_embeddings, chunks):
        store.add(text, emb, meta)

    store.save(str(EMBEDDINGS_PATH))
    logger.info("Done in %.1fs.", time.time() - start)


if __name__ == "__main__":
    main()
