"""
Retrieval evaluation:  python src/evaluate.py
Compares 'abstract only' vs 'all chunks (abstract + full text)'.
Metrics: Hit@k (is a correct paper in the top-k?) and MRR (how high is it ranked?).
"""
import json

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


if __name__ == "__main__":
    print(f"{len(QUESTIONS)} questions, k={K}\n")
    print(f"{'Setup':<28}{'Hit@'+str(K):<10}{'MRR':<8}")
    for label, ct in [("Abstract only", "abstract"), ("All chunks (abs+fulltext)", None)]:
        h, m = score(ct)
        print(f"{label:<28}{h:<10.2f}{m:<8.2f}")
