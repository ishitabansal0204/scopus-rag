# Scopus RAG: Question Answering over Maternity-Law Research Papers

A Retrieval-Augmented Generation (RAG) system that answers questions about research papers exported from Scopus, with inline citations. Abstracts are indexed for every paper; full text is added for legally Open Access papers via the Unpaywall API.

## Architecture

```
Scopus export --> filter / dedupe --> abstract chunks --------------------+
                        |                                                 +--> embeddings --> NumPy vector store
                        +-- OA papers --> Unpaywall --> PDF --> full-text chunks --+
                                                                          |
Question --> embed --> cosine search (+ year / chunk-type filters) --> top-k chunks --> LLM --> cited answer
```

**Chunks.** Each paper has one abstract chunk (`Title. Abstract`). Open Access papers with a downloadable PDF also get full-text chunks of about 900 characters with 150 characters of overlap, cut at a sentence end where possible. The reference list is removed, and PDFs with under 1,500 characters of extractable text are rejected. Every chunk carries the paper's metadata (EID, title, year, journal, DOI, citations, OA status).

**Retrieval.** Chunks are embedded with `all-MiniLM-L6-v2` (normalised vectors, so a dot product is cosine similarity). A query returns the top-k chunks (default 6), with at most 2 chunks per paper so one long paper cannot fill every slot. Optional filters: minimum year and chunk type.

**Generation.** The LLM is told to use only the retrieved excerpts, cite them as [1], [2], and reply "I could not find this in the provided papers." when none is relevant.

**Why no vector database?** The index is small (342 chunks), so exact search with NumPy is instant. ChromaDB was the first choice, but in testing its native engine (version 1.5.9) crashed on Windows with an access violation, so it was replaced. The retrieval code is isolated in `src/rag.py`, so a vector database can be swapped back in for a larger corpus.

## Setup

```bash
git clone https://github.com/ishitabansal0204/scopus-rag.git
cd scopus-rag
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # Windows: copy .env.example .env, then edit it
```

**Data.** Scopus data is licensed, so it is not included. Export your own from Scopus (CSV, with abstracts and all citation fields) and save it as `data/scopus.csv`. Save it as UTF-8 (in Excel: "CSV UTF-8"), otherwise accented characters break the load. The script needs these columns: `Title`, `Abstract`, `Year`, `EID`, `DOI`, `Open Access`, `Source title`, `Cited by`, `Document Type`.

**Configuration (`.env`).** Set `UNPAYWALL_EMAIL` to any real email address (Unpaywall asks for one). Choose an LLM:
- Local and free: install [Ollama](https://ollama.com), run `ollama pull llama3.2`, and keep `LLM_BACKEND=ollama`.
- API: set `LLM_BACKEND=anthropic`, `LLM_MODEL=claude-sonnet-5-5` and `ANTHROPIC_API_KEY`.

**Windows note.** Tested on Windows with Python 3.12. On one machine the newest PyTorch builds failed to load (`WinError 1114`, `c10.dll`), and `torch==2.5.1` (CPU build) worked:

```
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```

## Usage

```bash
python src/build_oa_index.py --input data/scopus.csv   # build the index (once)
python src/query.py "How does maternity leave affect breastfeeding in India?"
streamlit run app.py                                   # web UI
python src/evaluate.py                                 # retrieval evaluation
```

The index is written to `data/index/` and downloaded PDF text is cached in `data/fulltext/`. Both are git-ignored.

## Dataset

45 records were exported from Scopus. One was excluded because its abstract was identical to another record's (two 2023 German papers on pregnant surgeons), leaving **44 unique papers**.

| | Count |
|---|---|
| Unique papers | 44 |
| Open Access | 12 |
| Open Access with a downloadable PDF (full text) | 4 |
| Chunks in total | 342 (44 abstract, 298 full text) |

Eight of the 12 Open Access papers had no downloadable PDF, so they are abstract-only. A DOI marked Open Access often points to an HTML page, not a PDF.

## Evaluation

`src/evaluate.py` compares abstract-only retrieval with abstracts plus full text, using questions in `eval/questions.json` with known correct paper IDs (EIDs). Paper-level metrics: **Hit@5** (is a correct paper in the top 5?) and **MRR** (how high is it ranked?).

| Setup | Hit@5 | MRR |
|---|---|---|
| Abstract only | 1.00 | 1.00 |
| Abstract + OA full text | 1.00 | 0.92 |

15 questions. The first 12 were written from abstracts, and 3 are detail questions about methods in one full-text paper. At paper level, abstract-only already finds the right paper, so this table does not show a benefit of full text. Adding full text slightly lowers MRR because some full-text chunks from other papers outrank the correct abstract.

To test what full text adds, detail questions also have an **Evidence@5** score: does a top-5 chunk from the correct paper contain the answer text? Abstracts rarely contain methods details, so this favours full text by design. It measures whether the system can surface evidence that only exists in the body of a paper.

| Setup | Evidence@5 |
|---|---|
| Abstract only | _fill in from evaluate.py_ |
| Abstract + OA full text | _fill in from evaluate.py_ |

## Example queries

_Add 3 to 4 example questions with screenshots of the answers in `docs/`, for example `![demo](docs/screenshot1.png)`._

## Limitations

- Full text exists for only 4 of 44 papers (the Open Access papers with a downloadable PDF). All others are abstract-only.
- The evaluation set is small (15 questions, 3 of them detail questions from a single paper), so the metrics are indicative, not conclusive. Hit@5 and MRR work at paper level.
- Some records are off-topic keyword matches (for example, a 1984 paper on the Adolescent Family Life Act is in the index).
- At least one German-language paper is indexed. The embedding model is English-focused, so retrieval on German text is probably weaker. This was not evaluated. A multilingual model can be set with `EMBED_MODEL` in `.env` and the index rebuilt.
- The default local LLM (`llama3.2`, about 3B parameters) can miss relevant excerpts or be overly cautious. A larger model or the Claude API gives better answers. Answers should be checked against the cited sources.

## Tech stack

Python, sentence-transformers (`all-MiniLM-L6-v2`), NumPy, PyMuPDF, Unpaywall API, Streamlit, Ollama / Claude API.

## Author

_Your name, course and year._

## License

_Add a LICENSE file (for example MIT)._
