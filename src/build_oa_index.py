"""
build_oa_index.py
Builds a vector index from a Scopus export (NumPy store, no vector database needed).

  - Every paper: 1 abstract chunk   (skip non-OA ones with --oa-only)
  - Open Access papers with a DOI: full-text chunks via the Unpaywall API

Output (all git-ignored under data/):
  data/chunks.jsonl            every chunk with its metadata
  data/index/embeddings.npy    one normalised embedding per chunk
  data/index/records.json      chunk text + metadata, same order as embeddings.npy

Usage:
  set UNPAYWALL_EMAIL=you@college.edu        (Windows)
  export UNPAYWALL_EMAIL=you@college.edu     (Mac/Linux)
  python build_oa_index.py --input data/scopus.csv
  python build_oa_index.py --input data/scopus.csv --oa-only

Install:
  pip install pandas requests pymupdf sentence-transformers openpyxl python-dotenv
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

import fitz  # pymupdf
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

FULLTEXT_DIR = Path("data/fulltext")
CHUNKS_FILE = Path("data/chunks.jsonl")
INDEX_DIR = Path(os.getenv("INDEX_DIR", "data/index"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")  # must match rag.py; use paraphrase-multilingual-MiniLM-L12-v2 for German text


# ---------- 1. load + filter ----------
def load_data(path):
    df = pd.read_excel(path) if path.endswith((".xlsx", ".xls")) else pd.read_csv(path)
    df = df.fillna("")
    # drop duplicate abstracts (e.g. the 3 Hirche & Niethard papers)
    df = df[df["Abstract"].str.strip() != ""]
    df = df.drop_duplicates(subset="Abstract").reset_index(drop=True)
    df["is_oa"] = df["Open Access"].str.strip() != ""
    return df


# ---------- 2. full text via Unpaywall ----------
def find_pdf_url(doi, email):
    r = requests.get(f"https://api.unpaywall.org/v2/{doi}", params={"email": email}, timeout=20)
    if r.status_code != 200:
        return None
    data = r.json()
    locations = [data.get("best_oa_location")] + (data.get("oa_locations") or [])
    for loc in locations:
        if loc and loc.get("url_for_pdf"):
            return loc["url_for_pdf"]
    return None


def download_fulltext(eid, doi, email):
    """Returns full text string or None. Caches to data/fulltext/<eid>.txt"""
    cache = FULLTEXT_DIR / f"{eid}.txt"
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    try:
        url = find_pdf_url(doi, email)
        if not url:
            return None
        resp = requests.get(url, timeout=40, headers={"User-Agent": "Mozilla/5.0 (college-rag-project)"})
        if resp.status_code != 200 or not resp.content.startswith(b"%PDF"):
            return None  # got an HTML landing page, not a PDF
        with fitz.open(stream=resp.content, filetype="pdf") as doc:
            text = "\n".join(page.get_text() for page in doc)
        text = clean_text(text)
        if len(text) < 1500:  # scanned / broken PDF
            return None
        cache.write_text(text, encoding="utf-8")
        return text
    except Exception as e:
        print(f"   ! failed for {doi}: {e}")
        return None


def clean_text(text):
    text = re.sub(r"-\n(\w)", r"\1", text)      # fix hyphenated line breaks
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    # cut off the reference list, it pollutes retrieval
    m = re.search(r"\n\s*(References|REFERENCES|Bibliography)\s*\n", text)
    if m and m.start() > len(text) * 0.5:
        text = text[: m.start()]
    return text.strip()


# ---------- 3. chunking ----------
def chunk_text(text, size=900, overlap=150):
    """Sentence-aware-ish chunking: prefer to cut at a sentence end."""
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = text.rfind(". ", start + size // 2, end)
            if cut != -1:
                end = cut + 1
        chunk = text[start:end].strip()
        if len(chunk) > 100:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


# ---------- 4. build chunk records ----------
def to_int(x):
    try:
        return int(x)
    except (ValueError, TypeError):
        return 0


def build_records(df, email, oa_only):
    records = []
    for i, row in df.iterrows():
        meta = {
            "eid": row["EID"],
            "title": row["Title"],
            "year": to_int(row["Year"]),
            "source": row["Source title"],
            "doi": row["DOI"],
            "cited_by": to_int(row["Cited by"]),
            "doc_type": row["Document Type"],
            "open_access": bool(row["is_oa"]),
        }
        if row["is_oa"] or not oa_only:
            records.append({
                "id": f"{row['EID']}::abstract",
                "text": f"{row['Title']}. {row['Abstract']}",
                "metadata": {**meta, "chunk_type": "abstract", "chunk_no": 0},
            })

        if row["is_oa"] and row["DOI"]:
            print(f"[{i+1}/{len(df)}] OA fetch: {row['DOI']}")
            text = download_fulltext(row["EID"], row["DOI"], email)
            time.sleep(1)  # be polite to Unpaywall / publishers
            if text:
                for n, ch in enumerate(chunk_text(text)):
                    records.append({
                        "id": f"{row['EID']}::ft{n}",
                        "text": ch,
                        "metadata": {**meta, "chunk_type": "fulltext", "chunk_no": n + 1},
                    })
            else:
                print("   -> no downloadable PDF, abstract only")
    return records


# ---------- 5. embed + store ----------
def build_index(records):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL)
    embs = model.encode(
        [r["text"] for r in records],
        batch_size=64,
        normalize_embeddings=True,  # unit vectors -> dot product = cosine similarity
        show_progress_bar=False,
    )
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(INDEX_DIR / "embeddings.npy", np.asarray(embs, dtype="float32"))
    with open(INDEX_DIR / "records.json", "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)
    print(f"indexed {len(records)} chunks")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Scopus CSV or XLSX")
    ap.add_argument("--oa-only", action="store_true", help="skip non-OA papers entirely")
    args = ap.parse_args()

    email = os.environ.get("UNPAYWALL_EMAIL")
    if not email:
        raise SystemExit("Set UNPAYWALL_EMAIL environment variable (any real email).")

    FULLTEXT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data(args.input)
    print(f"{len(df)} unique papers, {df['is_oa'].sum()} open access")

    records = build_records(df, email, args.oa_only)
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_ft = sum(r["metadata"]["chunk_type"] == "fulltext" for r in records)
    print(f"{len(records)} chunks total ({n_ft} full-text, {len(records) - n_ft} abstract)")
    build_index(records)
    print("Done. Index saved to", INDEX_DIR)


if __name__ == "__main__":
    main()
