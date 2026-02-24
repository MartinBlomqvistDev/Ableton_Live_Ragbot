import json
import logging
from typing import Dict, List

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Loaded once and kept in memory (~420 MB).
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-mpnet-base-v2")
    return _model


def create_embeddings(texts: List[str]) -> List[List[float]]:
    """Return unit-normalised embeddings for a list of texts."""
    return _get_model().encode(
        texts, normalize_embeddings=True, show_progress_bar=False
    ).tolist()


def load_chunks(path: str) -> List[Dict]:
    """Load chunks from a JSONL file."""
    chunks: List[Dict] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                chunks.append(json.loads(line))
    except FileNotFoundError:
        logger.error("Chunk file not found: %s", path)
    return chunks
