# CiteMind

Basic RAG over a small arXiv library: PDF text → chunks → embeddings → cited answers.

```
extract.py   download papers + PDFs from arXiv
ingest.py    extract text from PDFs
chunks.py    split into overlapping passages (drops reference lists)
embed.py     MiniLM embeddings into Chroma
ask.py       CLI question
backend/     FastAPI + chat UI at http://localhost:8000
```

## 1. Setup

```bash
cd ~/Desktop/CiteMind
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then paste your Groq key
```

Get a free key at https://console.groq.com

## 2. Build the index (once)

You already have PDFs in `data/pdfs/`. From the project root:

```bash
python ingest.py
python chunks.py
python embed.py
```

To fetch a new batch instead:

```bash
python extract.py
```

then run ingest → chunks → embed again.

## 3. Chat

```bash
uvicorn backend.main:app --reload --port 8000
```

Open http://localhost:8000 and ask something like *How does retrieval-augmented generation work?*

CLI:

```bash
python ask.py "How does retrieval-augmented generation work?"
```

Answers are grounded in retrieved passages. Source links go to arXiv. Nothing is a full literature review — this is a local demo on ~10 papers.
