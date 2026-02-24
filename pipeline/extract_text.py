"""Extract and clean text from the Ableton Live 12 manual PDF."""
import logging
import re
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MANUAL_TEXT_PATH, PDF_PATH
from pypdf import PdfReader


def _clean_line(line: str) -> str:
    """Strip page numbers, artefacts, and broken heading numbers from a raw PDF line."""
    s = line.strip()
    s = re.sub(r'\s+(?:\d+\s?){1,3}$', '', s)
    if not s or re.fullmatch(r'^\s*$', s) or re.fullmatch(r'^-+$', s):
        return ''
    norm = re.sub(r'\s+', '', re.sub(r'\.+', '.', s))
    if re.fullmatch(r'^\d+(\.\d+)*\.?$', norm):
        return ''
    # Merge number fragments split by the PDF renderer ("1 7 . Routing" → "17. Routing").
    s = re.sub(r'^(\d+)\s+(\d+)(\s*\.)', r'\1\2\3', s)
    s = re.sub(r'(\d+)\s*\.\s*(\d+)', r'\1.\2', s)
    s = re.sub(r'(\d+\.\d+(?:\.\d+)*)\s+(\d+\.\d+(?:\.\d+)*)', r'\1.\2', s)
    s = re.sub(r'(\d+\.\d+)\s+(\d)(?!\.)', r'\1\2', s)
    s = re.sub(r'(\d+\.)\s+(\d+)', r'\1\2', s)
    s = re.sub(r'(\d+)\s*\.\s*(?=\S)', r'\1.', s)
    s = re.sub(r'\.\s*\.', '.', s)
    s = re.sub(r'\.\.+', '.', s)
    return re.sub(r'\s+', ' ', s).strip()


def _is_heading(line: str) -> bool:
    """Return True if the line looks like a numbered section heading."""
    return bool(re.match(r'^\s*\d{1,2}(?:\.\d+)*\.?\s+.+', line))


def _join_lines(lines: List[str]) -> str:
    """Merge a list of lines into a paragraph, handling hyphenation at line breaks."""
    parts: List[str] = []
    for i, line in enumerate(lines):
        if i > 0 and parts and (parts[-1].endswith('-') or parts[-1].endswith('—')):
            parts[-1] = parts[-1].rstrip('-').rstrip('—')
        elif parts:
            parts.append(' ')
        parts.append(line)
    return re.sub(r'\s+', ' ', ''.join(parts)).strip()


def extract_text(pdf_path: str) -> str:
    """Extract and clean all text from a PDF. Return a single string."""
    logger = logging.getLogger(__name__)
    reader = PdfReader(pdf_path)
    logger.info("PDF has %d pages.", len(reader.pages))
    sections: List[str] = []
    for i, page in enumerate(reader.pages):
        try:
            raw = page.extract_text() or ''
        except Exception as exc:
            logger.warning("Skipping page %d: %s", i + 1, exc)
            continue
        page_out: List[str] = []
        paragraph: List[str] = []
        for line in raw.split('\n'):
            cleaned = _clean_line(line)
            if not cleaned:
                continue
            if _is_heading(cleaned):
                if paragraph:
                    page_out.append(_join_lines(paragraph))
                    paragraph = []
                page_out.extend(['', cleaned, ''])
            else:
                paragraph.append(cleaned)
        if paragraph:
            page_out.append(_join_lines(paragraph))
        sections.append('\n'.join(page_out))
        if (i + 1) % 50 == 0:
            logger.info("Processed %d/%d pages.", i + 1, len(reader.pages))
    return '\n\n'.join(sections).strip()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)
    if not PDF_PATH.exists():
        logger.error("PDF not found: %s", PDF_PATH)
        return
    logger.info("Extracting text from %s...", PDF_PATH)
    text = extract_text(str(PDF_PATH))
    MANUAL_TEXT_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANUAL_TEXT_PATH.write_text(text, encoding="utf-8")
    logger.info("Saved to %s (%d chars).", MANUAL_TEXT_PATH, len(text))


if __name__ == "__main__":
    main()
