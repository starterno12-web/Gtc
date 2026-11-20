#!/usr/bin/env python3
"""Validate the Arabic corpus and the Android NLM training wiring.

This is a repository-side preflight only. Actual gradient updates and the
10 model-generated answers run inside the Android app's Trainer workflow.
"""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "src/main/assets/arabic_language_training.txt"
QA = ROOT / "src/main/assets/train.txt"
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
if len(corpus.strip()) < 1000:
    fail("Arabic corpus is unexpectedly short")
if not re.search(r"[\u0600-\u06FF]", corpus):
    fail("Arabic characters were not found in the corpus")

# These copies are intentional mirrors; the Android packaged asset is src/main/assets.
for rel in ("assets/arabic_language_training.txt", "training/arabic_language_training.txt"):
    if read_utf8(ROOT / rel) != corpus:
        fail(f"{rel} differs from src/main/assets/arabic_language_training.txt")

qa_text = read_utf8(QA)
qa_lines = [line for line in qa_text.splitlines() if line.strip()]
if len(qa_lines) != 10:
    fail(f"train.txt must contain exactly 10 non-empty tab-separated pairs; found {len(qa_lines)}")

for number, line in enumerate(qa_lines, 1):
    if line.count("\t") != 1:
        fail(f"train.txt line {number} must contain exactly one tab")
    question, answer = line.split("\t")
    if not question.startswith("س:") or not answer.startswith("ج:"):
        fail(f"train.txt line {number} must use س:question<TAB>ج:answer")
    if len(question[2:].strip()) < 8 or len(answer[2:].strip()) < 15:
        fail(f"train.txt line {number} has an empty or implausibly short question/answer")

for rel in ("assets/train.txt", "training/train.txt"):
    if read_utf8(ROOT / rel) != qa_text:
        fail(f"{rel} differs from src/main/assets/train.txt")

engine = read_utf8(ENGINE)
required = (
    "class Trainer",
    'context.assets.open("arabic_language_training.txt")',
    'readAsset("train.txt")',
    "brain.trainPairs()",
    "samples.shuffled().forEach { sample -> if (!stopFlag.get()) model.trainWindow(sample, 0.00012f) }",
    "for (round in 1..20)",
    "brain.selfAttempt(question)",
    "if (qaPairs.size != 10)",
    "referenceAnswers[index]",
)
for marker in required:
    if marker not in engine:
        fail(f"Android engine is not wired to the training workflow: missing {marker!r}")

sentences = [
    part.strip()
    for part in re.split(r"[\n\r]+|(?<=[.!؟])\s+", corpus)
    if len(part.strip()) >= 30 and re.search(r"\w", part, re.UNICODE)
]
if len(sentences) < 10:
    fail(f"corpus produced only {len(sentences)} usable text segments")

print("Arabic training preflight: PASS")
print(f"Corpus: {len(corpus.encode('utf-8'))} UTF-8 bytes; {len(sentences)} usable segments")
print(f"Question/answer pairs: {len(qa_lines)}")
print("Engine wiring: 20 training rounds + exactly 10 QA evaluation prompts")
