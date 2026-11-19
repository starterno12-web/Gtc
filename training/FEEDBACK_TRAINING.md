# Feedback-loop training playground

## 1. Add your own text and questions
Create `training/feedback_pairs.jsonl`. Each line is a JSON record with the source text and question/answer pairs, for example:

```json
{"text":"Put your own source paragraph here.","questions":[{"q":"Ask a question about the paragraph","a":"Write the exact expected answer"}]}
```

Use your own text and facts; the trainer has no invented built-in quiz facts.

## 2. Run training
From the repository root, with the Python training dependencies installed:

```bash
python scripts/feedback_train.py --read-times 20 --max-retries 8
```

- `--read-times`: repeated supervised exposure (10–50; default 20).
- `--max-retries`: maximum correction attempts for each question (default 8).
- `--base`: source checkpoint (default `nlm2.bin`).
- `--out`: candidate checkpoint (default `nlm2_feedback.bin`).

For each question, it generates an answer, checks normalized exact-answer inclusion against your expected answer, trains again on the source text and correct answer if it fails, and asks the same question again up to the retry limit.

The candidate weights and JSON report are written to the output path and its `.feedback_report.json` sibling. The trainer does not overwrite the original checkpoint.
