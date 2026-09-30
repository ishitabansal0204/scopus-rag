# Scopus RAG: Question Answering over Maternity-Law Research Papers

A Retrieval-Augmented Generation (RAG) system that answers questions about research papers
exported from Scopus, with inline citations. Abstracts are indexed for every paper; full text is
added for legally Open Access papers via the Unpaywall API.

## Architecture

```
Scopus export ──> filter/dedupe ──> abstract chunks ─┐
                       │                             ├─> embeddings ──> ChromaDB
                       └─ OA papers ─> Unpaywall ─> PDF ─> full-text chunks ─┘
                                                         │
Question ──> embed ──> vector search (+ filters) ──> top-k chunks ──> LLM ──> cited answer
```

**Indexing:** parent-child. Each paper (EID) has one abstract chunk plus N full-text chunks, all
carrying the same metadata (title, year, journal, DOI, citations, OA status).

## Setup

```bash
git clone https://github.com/<your-username>/scopus-rag.git
cd scopus-rag
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                # then edit it
```

**Data:** Scopus data is licensed, so it is not included. Export your own from Scopus
(Export -> CSV, include abstracts and all citation fields) and save it as `data/scopus.csv`.

**LLM:** either install [Ollama](https://ollama.com) and run `ollama pull llama3.2`, or set
`LLM_BACKEND=anthropic` and an API key in `.env`.

## Usage

```bash
export UNPAYWALL_EMAIL=you@college.edu              # Windows: set UNPAYWALL_EMAIL=...
python src/build_oa_index.py --input data/scopus.csv   # build the index (once)
python src/query.py "How does maternity leave affect breastfeeding in India?"
streamlit run app.py                                # web UI
python src/evaluate.py                              # retrieval evaluation
```

## Results

_Fill in after running `python src/evaluate.py`._

| Setup | Hit@5 | MRR |
|---|---|---|
| Abstract only | | |
| Abstract + OA full text | | |

Dataset: __ papers, __ Open Access, __ with retrievable full text, __ total chunks.

## Example queries

_Add 3 to 4 example questions with screenshots of the answers in `docs/`._

## Limitations

- Full text only for Open Access papers; others are abstract-only.
- Some papers in the export are off-topic keyword matches or short legal summaries.
- Some papers are in German; abstracts are English, full text is not.
- Small evaluation set (12 questions).

## Tech stack

Python, sentence-transformers, ChromaDB, PyMuPDF, Unpaywall API, Streamlit, Ollama / Claude API.
