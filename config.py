from pathlib import Path

# --- Paths ---
DATA_DIR         = Path("data")
INDEX_DIR        = Path("index")
PDF_PATH         = DATA_DIR / "Ableton_12_manual.pdf"
MANUAL_TEXT_PATH = DATA_DIR / "full_manual_text.txt"
CHUNKS_PATH      = DATA_DIR / "chunks.jsonl"
EMBEDDINGS_PATH  = INDEX_DIR / "embeddings.parquet"

# --- Model config ---
EMBEDDING_MODEL   = "all-mpnet-base-v2"   # local sentence-transformers, ~420 MB
GENERATION_MODEL  = "gemini-2.0-flash"
MAX_OUTPUT_TOKENS = 1000

# --- Pipeline ---
EMBEDDING_BATCH_SIZE = 100

# --- Search ---
CHATBOT_TOP_K    = 5
EVALUATION_TOP_K = 15

# --- No-answer phrases (must match system prompt in core/llm.py exactly) ---
NO_ANSWER_EN = (
    "I found no relevant information in my sources. "
    "Try rephrasing your question or consult the Ableton Live 12 manual."
)
NO_ANSWER_SV = (
    "Jag hittade ingen relevant information i mina källor. "
    "Försök att omformulera din fråga eller konsultera Ableton Live 12 manualen."
)
