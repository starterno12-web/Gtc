#!/usr/bin/env python3
"""Create ten extractive QA prompts from the Arabic article's own sections."""
from __future__ import annotations
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/main/assets/arabic_language_training.txt"
OUTPUT = ROOT / "training/generated_article_qa.json"

def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!؟])\s+|[\r\n]+", text)
            if len(s.strip()) >= 35 and re.search(r"[\u0600-\u06ff]", s)]

def main() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    sections: list[tuple[str, str]] = []
    heading = ""
    body: list[str] = []
    def flush() -> None:
        if heading and body:
            for sentence in split_sentences(" ".join(body)):
                if not sentence.startswith(("س:", "ج:", "العنوان:", "الغرض:", "ملاحظة:")):
                    sections.append((heading, sentence))
                    break
    for raw in text.splitlines():
        line = raw.strip()
        match = re.match(r"^\d+\.\s+(.+)$", line)
        if match:
            flush()
            heading, body = match.group(1), []
        elif line and not re.fullmatch(r"[=\-]{3,}", line):
            body.append(line)
    flush()

    # One question per article section, with its answer copied verbatim from that section.
    items = []
    seen = set()
    for title, answer in sections:
        if title in seen:
            continue
        seen.add(title)
        items.append({
            "question": f"بحسب قسم «{title}»، ماذا يوضح المقال؟",
            "answer": answer,
            "source_section": title,
        })
        if len(items) == 10:
            break
    if len(items) != 10:
        raise SystemExit(f"Expected 10 distinct article sections; found {len(items)}")
    for i, item in enumerate(items, 1):
        if len(item["question"]) < 12 or len(item["answer"]) < 30:
            raise SystemExit(f"Generated QA item {i} is too short")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({
        "source": "src/main/assets/arabic_language_training.txt",
        "method": "extractive: questions from section headings; answers copied verbatim",
        "count": len(items),
        "items": items,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Generated and validated {len(items)} article-grounded QA pairs.")
    for i, item in enumerate(items, 1):
        print(f"{i}. Q: {item['question']}\n   A: {item['answer']}")

if __name__ == "__main__":
    main()
