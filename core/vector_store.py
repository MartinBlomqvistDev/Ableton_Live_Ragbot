import logging
import os
from typing import Dict, List

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self) -> None:
        self.vectors: List[np.ndarray] = []
        self.texts: List[str] = []
        self.metadata: List[Dict] = []

    def add(self, text: str, embedding: List[float], metadata: Dict | None = None) -> None:
        """Append one item to the store."""
        self.vectors.append(np.array(embedding))
        self.texts.append(text)
        self.metadata.append(metadata or {})

    def search(self, query_embedding: List[float], k: int = 5) -> List[Dict]:
        """Return the k most similar items by cosine similarity."""
        if not self.vectors:
            return []
        qv = np.array(query_embedding)
        qn = np.linalg.norm(qv)
        scores = []
        for i, vec in enumerate(self.vectors):
            vn = np.linalg.norm(vec)
            sim = float(np.dot(qv, vec) / (qn * vn)) if qn and vn else 0.0
            scores.append((i, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [
            {"text": self.texts[i], "metadata": self.metadata[i], "similarity": s}
            for i, s in scores[:k]
        ]

    def save(self, path: str) -> None:
        """Write the store to a Parquet file."""
        pl.DataFrame(
            {"vectors": self.vectors, "texts": self.texts, "metadata": self.metadata}
        ).write_parquet(path)
        logger.info("Saved %d items to %s.", len(self.texts), path)

    def load(self, path: str) -> bool:
        """Load the store from a Parquet file. Return False if the file is missing."""
        if not os.path.exists(path):
            logger.error("Embeddings file not found: %s", path)
            return False
        df = pl.read_parquet(path)
        self.vectors = [np.array(v) for v in df["vectors"].to_list()]
        self.texts = df["texts"].to_list()
        self.metadata = df["metadata"].to_list()
        logger.info("Loaded %d items from %s.", len(self.texts), path)
        return True
