"""Command-line interface:  python src/query.py "your question" """
import sys

from rag import answer

if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or input("Question: ")
    text, hits = answer(q)
    print("\n" + text + "\n\nSources:")
    for i, h in enumerate(hits, 1):
        m = h["meta"]
        print(f"[{i}] {m['title']} ({m['year']}) doi:{m['doi']} [{m['chunk_type']}]")
