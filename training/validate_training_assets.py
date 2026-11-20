#!/usr/bin/env python3
"""Preflight checks for the original Arabic corpus and its Android training wiring."""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "src/main/assets/arabic_language_training.txt"
ENGINE = ROOT / "النموذج_نسخة_محدثة__1_ (1).kt"

def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)

def read_utf8(path: Path) -> str:
    if not path.is_file():
        fail(f"missing required file: {path.relative_to(ROOT)}")
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeError as exc:
        fail(f"invalid UTF-8 in {path.relative_to(ROOT)}: {exc}")

corpus = read_utf8(CORPUS)
if len(corpus.strip()) < 1000 or not re.search(r"[\u0600-\u06FF]", corpus):
    fail("Arabic corpus is missing, too short, or contains no Arabic text")

for rel in ("assets/arabic_language_training.txt", "training/arabic_language_training.txt"):
    if read_utf8(ROOT / rel) != corpus:
        fail(f"{rel} differs from the packaged Android corpus")

lines = [line.strip() for line in corpus.splitlines()]
qa_pairs = [(line, lines[i + 1]) for i, line in enumerate(lines[:-1])
            if line.startswith("س:") and lines[i + 1].startswith("ج:")]
if len(qa_pairs) < 10:
    fail(f"expected at least 10 original adjacent س:/ج: pairs in corpus; found {len(qa_pairs)}")
for n, (question, answer) in enumerate(qa_pairs[:10], 1):
    if len(question[2:].strip()) < 5 or len(answer[2:].strip()) < 10:
        fail(f"source QA pair {n} is empty or too short")

engine = read_utf8(ENGINE)
required = (
    "class Trainer",
    'context.assets.open("arabic_language_training.txt")',
    "extractSourceQa(fullText)",
    "val selectedPairs = qaPairs.take(10)",
    "samples.shuffled().forEach { sample -> if (!stopFlag.get()) model.trainWindow(sample, 0.00012f) }",
    "for (round in 1..20)",
    "brain.selfAttempt(question)",
    "referenceAnswers[index]",
)
for marker in required:
    if marker not in engine:
        fail(f"Android engine is not wired to source-grounded training: missing {marker!r}")

sentences = [part.strip() for part in re.split(r"[\n\r]+|(?<=[.!؟])\s+", corpus)
             if len(part.strip()) >= 30 and re.search(r"\w", part, re.UNICODE)]
if len(sentences) < 10:
    fail(f"corpus produced only {len(sentences)} usable text segments")

print("Arabic source-training preflight: PASS")
print(f"Corpus: {len(corpus.encode('utf-8'))} UTF-8 bytes; {len(sentences)} usable segments")
print(f"Original Q/A pairs in corpus: {len(qa_pairs)}")
print("Engine wiring: 20 training rounds + 10 original-corpus QA prompts")
