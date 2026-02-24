"""Split the extracted manual text into headed chunks and write to JSONL."""
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CHUNKS_PATH, MANUAL_TEXT_PATH


class Chunk:
    def __init__(
        self,
        chunk_id: str,
        title: str,
        content: str,
        level: str,
        parent_chain: List[Dict[str, str]],
    ) -> None:
        self.chunk_id = chunk_id
        self.title = title
        self.content = content
        self.level = level
        self.parent_chain = parent_chain

    def to_dict(self) -> Dict:
        return {
            "chunk_id": self.chunk_id,
            "title": self.title,
            "level": self.level,
            "content": self.content,
            "parent_chain": self.parent_chain,
        }


def _level(chunk_id: str) -> str:
    """Return a nesting-level label based on the number of dots in the ID."""
    return ("main", "sub", "subsub", "deep")[min(chunk_id.count("."), 3)]


def _build_chain(
    current_id: str, current_title: str, chain: List[Dict]
) -> List[Dict]:
    """Return a parent chain that ends with the current heading."""
    parts = current_id.split(".")
    new_chain = []
    for i in range(len(parts) - 1):
        pid = ".".join(parts[: i + 1])
        title = next(
            (p["title"] for p in chain if p["chunk_id"] == pid),
            f"Chapter {pid}",
        )
        new_chain.append({"chunk_id": pid, "title": title})
    new_chain.append({"chunk_id": current_id, "title": current_title})
    return new_chain


def chunk_text_from_file(input_path: str, output_path: str) -> int:
    """Split a numbered-heading text file into chunks. Write JSONL, return chunk count."""
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    chunks: List[Chunk] = []
    current_id: Optional[str] = None
    current_title = ""
    current_content: List[str] = []
    chain: List[Dict] = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        main_m = re.match(r"^(\d+)\.\s*(.*)$", line)
        sub_m = re.match(r"^(\d+(?:\.\d+)+)\s+(.+)", line)

        if main_m or sub_m:
            if current_id is not None:
                chunks.append(
                    Chunk(
                        chunk_id=current_id,
                        title=current_title,
                        content=" ".join(current_content).strip(),
                        level=_level(current_id),
                        parent_chain=chain[:-1].copy(),
                    )
                )
            if main_m:
                current_id = main_m.group(1)
                current_title = main_m.group(2) or f"Chapter {current_id}"
                chain = [{"chunk_id": current_id, "title": current_title}]
            else:
                current_id = sub_m.group(1)
                current_title = sub_m.group(2)
                chain = _build_chain(current_id, current_title, chain)
            current_content = []
        else:
            current_content.append(line)

    if current_id is not None:
        chunks.append(
            Chunk(
                chunk_id=current_id,
                title=current_title,
                content=" ".join(current_content).strip(),
                level=_level(current_id),
                parent_chain=chain[:-1].copy(),
            )
        )

    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            json.dump(chunk.to_dict(), f, ensure_ascii=False)
            f.write("\n")

    return len(chunks)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)
    if not MANUAL_TEXT_PATH.exists():
        logger.error("Input not found: %s — run pipeline/extract_text.py first.", MANUAL_TEXT_PATH)
        return
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    count = chunk_text_from_file(str(MANUAL_TEXT_PATH), str(CHUNKS_PATH))
    logger.info("Done — %d chunks written to %s.", count, CHUNKS_PATH)


if __name__ == "__main__":
    main()
