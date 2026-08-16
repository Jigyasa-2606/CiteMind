import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from retrieval import is_bibliography

DATA_DIR = Path("data")
CHUNKS_FILE = DATA_DIR / "chunks.jsonl"
PAPERS_FILE = DATA_DIR / "papers.jsonl"
CHROMA_DIR = DATA_DIR / "chroma"
COLLECTION_NAME = "papers"
MODEL_NAME = "all-MiniLM-L6-v2"


def load_jsonl(path: Path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main():
    chunks = [c for c in load_jsonl(CHUNKS_FILE) if not is_bibliography(c["text"])]
    papers = load_jsonl(PAPERS_FILE)
    print(f"Loaded {len(chunks)} body chunks")

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    ids = [c["chunk_id"] for c in chunks]
    texts = [c["text"] for c in chunks]
    metadatas = [
        {
            "paper_id": c["paper_id"],
            "title": c["title"],
            "chunk_index": int(c["chunk_index"]),
            "kind": c.get("kind", "body"),
        }
        for c in chunks
    ]

    for paper in papers:
        abstract = (paper.get("abstract") or "").strip()
        title = (paper.get("title") or "").strip()
        if not title:
            continue
        ids.append(f"{paper['id']}_abstract")
        texts.append(f"{title}\n\n{abstract}".strip())
        metadatas.append(
            {
                "paper_id": paper["id"],
                "title": title,
                "chunk_index": -1,
                "kind": "abstract",
            }
        )

    print(f"Embedding {len(texts)} vectors (chunks + title/abstract)...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings.tolist(),
    )
    print(f"Stored {collection.count()} vectors in {CHROMA_DIR}")

    question = "How does retrieval-augmented generation work?"
    q_emb = model.encode([question]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=5)

    print(f"\nTest question: {question}")
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        kind = meta.get("kind", "body")
        print(f"\n- {meta['title']} ({kind}, chunk {meta['chunk_index']}, distance {dist:.3f})")
        print(f"  {doc[:240]}...")


if __name__ == "__main__":
    main()
