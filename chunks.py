import json
from pathlib import Path

from retrieval import is_bibliography

DATA_DIR = Path("data")
PAPERS_FILE = DATA_DIR / "papers.jsonl"
CHUNKS_FILE = DATA_DIR / "chunks.jsonl"

CHUNK_WORDS = 400
OVERLAP_WORDS = 50


def chunk_words(text, size=CHUNK_WORDS, overlap=OVERLAP_WORDS):
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    step = size - overlap
    while start < len(words):
        end = min(start + size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += step
    return chunks


def main():
    papers = []
    with PAPERS_FILE.open(encoding="utf-8") as f:
        for line in f:
            papers.append(json.loads(line))

    all_chunks = []
    skipped = 0
    for paper in papers:
        text_path = Path(paper["text_path"])
        text = text_path.read_text(encoding="utf-8")
        pieces = chunk_words(text)
        kept = 0

        for i, piece in enumerate(pieces):
            if is_bibliography(piece):
                skipped += 1
                continue
            all_chunks.append(
                {
                    "chunk_id": f"{paper['id']}_{i}",
                    "paper_id": paper["id"],
                    "title": paper["title"],
                    "chunk_index": i,
                    "kind": "body",
                    "text": piece,
                }
            )
            kept += 1

        print(f"{paper['id']}: {kept} chunks (dropped refs)")

    with CHUNKS_FILE.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + "\n")

    print(f"\nSaved {len(all_chunks)} chunks to {CHUNKS_FILE} (skipped {skipped} reference chunks)")


if __name__ == "__main__":
    main()
