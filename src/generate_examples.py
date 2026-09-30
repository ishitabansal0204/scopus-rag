"""
Generate docs/example_answers.md: questions with the RAG system's answers and sources.
Usage:  python src/generate_examples.py           (extra questions + all eval questions)
        python src/generate_examples.py --quick   (extra questions + the detail questions only)
Needs Ollama running (or the Anthropic backend configured in .env).
"""
import argparse
import datetime
import json
from pathlib import Path

import rag

EXTRA = [
    "How does paid maternity leave affect breastfeeding in India?",
    "What is the capital of France?",  # out of scope: the system should refuse
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="only the detail questions from eval/questions.json")
    args = ap.parse_args()

    items = json.load(open("eval/questions.json", encoding="utf-8"))
    if args.quick:
        items = [x for x in items if x.get("answer_contains")]
    questions, seen = [], set()
    for q in EXTRA + [x["question"] for x in items]:
        if q not in seen:
            seen.add(q)
            questions.append(q)

    lines = [
        "# Example questions and answers",
        "",
        f"Generated on {datetime.date.today().isoformat()} with `{rag.LLM_BACKEND}` / `{rag.LLM_MODEL}` "
        "by `python src/generate_examples.py`. Answers use only the retrieved excerpts. "
        "A small local model can miss details, so check the cited sources.",
        "",
    ]
    for n, q in enumerate(questions, 1):
        print(f"[{n}/{len(questions)}] {q[:70]}")
        text, hits = rag.answer(q)
        quoted = "\n".join("> " + ln for ln in text.strip().splitlines())
        lines += [f"## {n}. {q}", "", "**Answer**", "", quoted, "", "**Sources**", ""]
        for i, h in enumerate(hits, 1):
            m = h["meta"]
            doi = f" - https://doi.org/{m['doi']}" if m["doi"] else ""
            lines.append(f"{i}. {m['title']} ({m['year']}) [{m['chunk_type']}]{doi}")
        lines.append("")

    Path("docs").mkdir(exist_ok=True)
    Path("docs/example_answers.md").write_text("\n".join(lines), encoding="utf-8")
    print("Saved docs/example_answers.md")


if __name__ == "__main__":
    main()