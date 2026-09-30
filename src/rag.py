"""Core RAG logic: retrieve chunks from Chroma, then generate a cited answer."""
import os

import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")  # must match build_oa_index.py
CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_db")
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")            # "ollama" or "anthropic"
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")              # e.g. claude-sonnet-5-5 for anthropic

_model = None
_col = None


def _load():
    global _model, _col
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
        _col = chromadb.PersistentClient(path=CHROMA_PATH).get_collection("papers")
    return _model, _col


def _where(year_min=None, chunk_type=None):
    conds = []
    if year_min:
        conds.append({"year": {"$gte": int(year_min)}})
    if chunk_type:
        conds.append({"chunk_type": chunk_type})
    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"$and": conds}


def retrieve(question, k=6, year_min=None, chunk_type=None):
    """Return a list of dicts: {text, meta, distance}, best first."""
    model, col = _load()
    emb = model.encode([question]).tolist()
    res = col.query(query_embeddings=emb, n_results=k, where=_where(year_min, chunk_type))
    return [
        {"text": d, "meta": m, "distance": dist}
        for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]


def build_prompt(question, hits):
    blocks = []
    for i, h in enumerate(hits, 1):
        m = h["meta"]
        blocks.append(f"[{i}] {m['title']} ({m['year']}) - {m['chunk_type']}\n{h['text']}")
    context = "\n\n".join(blocks)
    return (
        "You are a research assistant. Answer the question using ONLY the excerpts below.\n"
        "Cite sources inline like [1], [2]. If the excerpts do not contain the answer, "
        "say you could not find it in the provided papers. Do not use outside knowledge.\n\n"
        f"Excerpts:\n{context}\n\nQuestion: {question}\nAnswer:"
    )


def generate(prompt):
    if LLM_BACKEND == "anthropic":
        import anthropic
        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        msg = client.messages.create(
            model=LLM_MODEL, max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text
    import ollama
    resp = ollama.chat(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}])
    return resp["message"]["content"]


def answer(question, k=6, year_min=None, chunk_type=None):
    hits = retrieve(question, k, year_min, chunk_type)
    if not hits:
        return "No matching papers found.", []
    return generate(build_prompt(question, hits)), hits
