"""
Retrieval evaluation:  python src/evaluate.py
Compares 'abstract only' vs 'all chunks (abstract + full text)'.

Paper-level metrics (all questions): Hit@k and MRR -- is a correct paper in the top-k?
Evidence@k (questions with "answer_contains"): does a top-k chunk FROM THE EXPECTED PAPER
contain the answer text? Full text can help here, because details such as sample sizes
or methods are usually not in the abstract.
"""
import json
import re

from rag import retrieve

K = 5
QUESTIONS = json.load(open("eval/questions.json", encoding="utf-8"))


def ranked_papers(question, chunk_type):
    hits = retrieve(question, k=K * 4, chunk_type=chunk_type)
    seen, order = set(), []
    for h in hits:  # collapse chunks to unique papers, keep rank order
        eid = h["meta"]["eid"]
        if eid not in seen:
            seen.add(eid)
            order.append(eid)
    return order[:K]


def score(chunk_type):
    hits, rr = 0, 0.0
    for item in QUESTIONS:
        order = ranked_papers(item["question"], chunk_type)
        ranks = [i + 1 for i, e in enumerate(order) if e in item["expected_eids"]]
        if ranks:
            hits += 1
            rr += 1 / min(ranks)
    n = len(QUESTIONS)
    return hits / n, rr / n


def norm(s):
    return re.sub(r"\s+", " ", s.lower())


def evidence_score(chunk_type):
    qs = [x for x in QUESTIONS if x.get("answer_contains")]
    found = 0
    for item in qs:
        hits = retrieve(item["question"], k=K, chunk_type=chunk_type)
        found += any(
            h["meta"]["eid"] in item["expected_eids"]
            and any(norm(s) in norm(h["text"]) for s in item["answer_contains"])
            for h in hits
        )
    return found, len(qs)


if __name__ == "__main__":
    setups = [("Abstract only", "abstract"), ("All chunks (abs+fulltext)", None)]
    print(f"{len(QUESTIONS)} questions, k={K}\n")
    print(f"{'Setup':<28}{'Hit@'+str(K):<10}{'MRR':<8}")
    for label, ct in setups:
        h, m = score(ct)
        print(f"{label:<28}{h:<10.2f}{m:<8.2f}")
    print("\nEvidence@" + str(K) + " (answer text found in a top-k chunk of the expected paper)")
    for label, ct in setups:
        f, n = evidence_score(ct)
        print(f"{label:<28}{f}/{n}")