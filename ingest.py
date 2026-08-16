import json
from pathlib import Path

import fitz  # pymupdf

DATA_DIR = Path("data")
TEXT_DIR = DATA_DIR / "text"
PAPERS_FILE = DATA_DIR / "papers.jsonl"


def extract_text(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    pages = [page.get_text("text") for page in doc]
    doc.close()
    return "\n".join(pages).strip()


def main():
    TEXT_DIR.mkdir(parents=True, exist_ok=True)

    papers = []
    with PAPERS_FILE.open(encoding="utf-8") as f:
        for line in f:
            papers.append(json.loads(line))

    updated = []
    for paper in papers:
        pdf_path = Path(paper["pdf_path"])
        text_path = TEXT_DIR / f"{paper['id']}.txt"

        if not pdf_path.exists():
            print(f"Missing PDF: {pdf_path}")
            continue

        text = extract_text(pdf_path)
        text_path.write_text(text, encoding="utf-8")

        paper["text_path"] = str(text_path)
        paper["char_count"] = len(text)
        updated.append(paper)
        print(f"{paper['id']}: {len(paper.get('abstract') or '')} abs chars -> {len(text)} paper chars")

    with PAPERS_FILE.open("w", encoding="utf-8") as f:
        for paper in updated:
            f.write(json.dumps(paper) + "\n")

    print(f"\nSaved full text to {TEXT_DIR}")


if __name__ == "__main__":
    main()
