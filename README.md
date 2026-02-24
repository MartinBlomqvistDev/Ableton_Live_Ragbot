# Ableton Live 12 RAG-Bot

A chatbot that answers questions about the Ableton Live 12 manual using RAG (retrieval-augmented generation). Ask it anything — it finds the most relevant manual chunks and uses Gemini to write an answer based on what it found.

## How it works

1. The manual PDF is extracted and split into headed chunks.
2. Each chunk is embedded locally with `sentence-transformers` (`all-mpnet-base-v2`, ~420 MB, runs offline).
3. At query time, your question gets the same treatment and the closest chunks are retrieved by cosine similarity.
4. Gemini writes an answer grounded in those chunks.

Embeddings are fully local — no API key or rate limits needed for indexing. Only generation (one call per question) touches the Google API.

## Project structure

```
├── app.py                   # Streamlit frontend
├── config.py                # All paths, model names, and settings
├── requirements.txt
├── .env.example
│
├── core/                    # Reusable modules
│   ├── embeddings.py        # Local sentence-transformers embedding
│   ├── llm.py               # Gemini generation
│   └── vector_store.py      # Cosine-similarity store backed by Parquet
│
├── pipeline/                # One-time data preparation scripts
│   ├── extract_text.py      # Pull text from the PDF
│   ├── chunk_text.py        # Split into headed chunks
│   └── build_index.py       # Embed chunks and save to Parquet
│
├── data/                    # Gitignored — regenerate with pipeline scripts
│   ├── Ableton_12_manual.pdf
│   ├── full_manual_text.txt
│   └── chunks.jsonl
│
└── index/                   # Gitignored — regenerate with build_index.py
    └── embeddings.parquet
```

## Setup

Requires **Python 3.10+**.

```bash
git clone https://github.com/MartinBlomqvistDev/Ableton_Live_Ragbot.git
cd Ableton_Live_Ragbot

python -m venv chatbot
chatbot\Scripts\activate      # Windows
# source chatbot/bin/activate # macOS/Linux

pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your Gemini API key:

```
API_KEY=your_key_here
```

## Build the index

Drop `Ableton_12_manual.pdf` into `data/`, then run the pipeline scripts in order (only needed once):

```bash
python pipeline/extract_text.py   # extract text from PDF → data/full_manual_text.txt
python pipeline/chunk_text.py     # split into chunks    → data/chunks.jsonl
python pipeline/build_index.py    # embed and index      → index/embeddings.parquet
```

The first run of `build_index.py` downloads the embedding model (~420 MB). Subsequent runs skip the step since the index already exists.

## Run the app

```bash
streamlit run app.py
```

## Notes

- `data/` is gitignored (the PDF and raw text are too large). Run the pipeline locally if you need to rebuild the index.
- `index/embeddings.parquet` is committed to the repo (4 MB) so Streamlit Cloud can load it without running any pipeline steps.
- Add `API_KEY` to Streamlit Cloud's Secrets to enable generation. Embedding runs locally and needs no key.
- Built by Martin Blomqvist during the Data Scientist program at EC Utbildning 2025.
