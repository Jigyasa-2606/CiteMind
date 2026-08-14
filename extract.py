
import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

BASE = "https://export.arxiv.org/api/query"
DATA_DIR = Path("data")
PDF_DIR = DATA_DIR / "pdfs"
PAPERS_FILE = DATA_DIR / "papers.jsonl"
HEADERS = {"User-Agent": "rag-research-assistant/0.1"}
ATOM = "{http://www.w3.org/2005/Atom}"


def search_arxiv(query, start=0, max_results=10):
    params = {
        "search_query": query,
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    url = BASE + "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request) as resp:
        return resp.read()


def paper_id_from_url(entry_id):
    # http://arxiv.org/abs/2401.12345v1 -> 2401.12345v1
    return entry_id.rstrip("/").split("/")[-1]


def parse_papers(xml_bytes):

    root = ET.fromstring(xml_bytes)
    papers = []

    for entry in root.findall(f"{ATOM}entry"):
        pdf_url = None
        for link in entry.findall(f"{ATOM}link"):
            if link.attrib.get("type") == "application/pdf":
                pdf_url = link.attrib.get("href")
                break

        authors = [a.findtext(f"{ATOM}name") for a in entry.findall(f"{ATOM}author")]
        authors = [a for a in authors if a]

        title = (entry.findtext(f"{ATOM}title") or "").replace("\n", " ").strip()
        abstract = (entry.findtext(f"{ATOM}summary") or "").replace("\n", " ").strip()
        entry_id = entry.findtext(f"{ATOM}id") or ""

        papers.append(
            {
                "id": paper_id_from_url(entry_id),
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "published": entry.findtext(f"{ATOM}published"),
                "pdf_url": pdf_url,
            }
        )

    return papers


def download_pdf(pdf_url, dest_path):
    request = urllib.request.Request(pdf_url, headers=HEADERS)
    with urllib.request.urlopen(request) as resp:
        dest_path.write_bytes(resp.read())
    return dest_path


def main():
    DATA_DIR.mkdir(exist_ok=True)
    PDF_DIR.mkdir(exist_ok=True)

    query = "cat:cs.LG AND ti:retrieval"
    xml = search_arxiv(query, max_results=10)
    papers = parse_papers(xml)

    saved = []
    for paper in papers:
        pdf_url = paper["pdf_url"]
        pdf_path = PDF_DIR / f"{paper['id']}.pdf"
        paper["pdf_path"] = str(pdf_path) if pdf_url else None

        if pdf_url:
            print(f"Downloading {paper['id']}: {paper['title'][:80]}")
            download_pdf(pdf_url, pdf_path)
            time.sleep(3) 
        else:
            print(f"No PDF for {paper['id']}")

        saved.append(paper)
 
    with PAPERS_FILE.open("w", encoding="utf-8") as f:
        for paper in saved:
            f.write(json.dumps(paper) + "\n")

    print(f"\nSaved {len(saved)} papers to {PAPERS_FILE}")
    print(f"PDFs are in {PDF_DIR}")
    print("Next: run ingest.py to extract full text from the PDFs.")


if __name__ == "__main__":
    main()
