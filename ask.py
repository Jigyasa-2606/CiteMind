import os
import sys
from pathlib import Path

from dotenv import load_dotenv
import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer

from retrieval import FETCH_K, KEEP_K, filter_hits, strip_sources_section

load_dotenv()

CHROMA_DIR = Path("data/chroma")
COLLECTION_NAME = "papers"
MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.1-8b-instant"


def retrieve(model, collection, question):
    q_emb = model.encode([question]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=FETCH_K)
    return filter_hits(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
        keep_k=KEEP_K,
    )


def build_prompt(question, chunks):
    context = []
    for i, ch in enumerate(chunks, 1):
        context.append(
            f"[{i}] {ch['title']} (arXiv:{ch['paper_id']})\n{ch['text']}"
        )
    passages = "\n\n".join(context)
    return f"""You answer questions using ONLY the passages below.
If the passages are not enough, say you do not know from the indexed papers.
You may mention a paper inline as (title, arXiv id).
Do NOT add a Sources / References list at the end.
Do not invent papers.

Passages:
{passages}

Question: {question}
Answer:"""


def main():
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        raise SystemExit('Usage: python ask.py "your question"')

    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise SystemExit("Missing GROQ_API_KEY. Put it in a .env file in this folder.")

    if not CHROMA_DIR.exists():
        raise SystemExit("No index yet. Run: python ingest.py && python chunks.py && python embed.py")

    model = SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)
    chunks = retrieve(model, collection, question)
    if not chunks:
        raise SystemExit("No matching passages.")

    groq = Groq(api_key=key)
    resp = groq.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": build_prompt(question, chunks)}],
        temperature=0.2,
    )
    answer = strip_sources_section(resp.choices[0].message.content or "")
    print(answer)
    print("\nSources:")
    seen = set()
    for ch in chunks:
        if ch["paper_id"] in seen:
            continue
        seen.add(ch["paper_id"])
        print(f"- {ch['title']} ({ch['paper_id']})")


if __name__ == "__main__":
    main()
