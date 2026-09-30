"""Core RAG logic: retrieve chunks by cosine similarity (NumPy), then generate a cited answer."""
import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")  # must match build_oa_index.py
INDEX_DIR = Path(os.getenv("INDEX_DIR", "data/index"))       # written by build_oa_index.py
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")            # "ollama" or "anthropic"
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")              # e.g. claude-sonnet-5-5 for anthropic

_model = None
_emb = None
_recs = None


def _load():
    global _model, _emb, _recs
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
        _emb = np.load(INDEX_DIR / "embeddings.npy")
        with open(INDEX_DIR / "records.json", encoding="utf-8") as f:
            _recs = json.load(f)
    return _model, _emb, _recs


def _mask(recs, year_min=None, chunk_type=None):
    """Boolean mask of chunks that pass the optional year / chunk-type filters."""
    return np.array([
        (not year_min or r["metadata"]["year"] >= int(year_min))
        and (not chunk_type or r["metadata"]["chunk_type"] == chunk_type)
        for r in recs
    ])


def retrieve(question, k=6, year_min=None, chunk_type=None, max_per_paper=2):
    """Return a list of dicts: {text, meta, distance}, best first.
    distance = 1 - cosine similarity (smaller is closer).
    At most `max_per_paper` chunks per paper, so one long paper cannot fill every slot."""
    model, emb, recs = _load()
    q = model.encode([question], normalize_embeddings=True)[0]
    sims = emb @ q  # embeddings are unit vectors, so this is cosine similarity
    idx = np.where(_mask(recs, year_min, chunk_type))[0]
    if idx.size == 0:
        return []
    ranked = idx[np.argsort(-sims[idx])]
    hits, per_paper = [], {}
    for i in ranked:
        eid = recs[i]["metadata"]["eid"]
        if per_paper.get(eid, 0) >= max_per_paper:
            continue
        per_paper[eid] = per_paper.get(eid, 0) + 1
        hits.append({"text": recs[i]["text"], "meta": recs[i]["metadata"], "distance": float(1 - sims[i])})
        if len(hits) == k:
            break
    return hits


def build_prompt(question, hits):
    blocks = []
    for i, h in enumerate(hits, 1):
        m = h["meta"]
        blocks.append(f"[{i}] {m['title']} ({m['year']}) - {m['chunk_type']}\n{h['text']}")
    context = "\n\n".join(blocks)
    return (
        "You are a research assistant. Use ONLY the numbered excerpts below; "
        "do not use outside knowledge.\n"
        "If any excerpt contains information relevant to the question, summarise what the "
        "excerpts say and cite them inline like [1], [2]. Partial answers are fine; say "
        "what the excerpts do and do not cover.\n"
        "Only if NONE of the excerpts is relevant, reply exactly: "
        "\"I could not find this in the provided papers.\"\n\n"
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