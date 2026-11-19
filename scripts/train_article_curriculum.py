# -*- coding: utf-8 -*-
"""Ten-pass Arabic article curriculum: self-supervised reading plus 68 extractive checks."""
import argparse, random, re, urllib.request, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from nuwaa.lm.gru import GRULM

REPO_RAW = "https://raw.githubusercontent.com/starterno12-web/Gtc/main/nlm2.bin"

def build_checks(text):
    # Select 68 distinct, substantive sentences directly from the article.
    sections = re.split(r"(?=القسم\s+(?:الأول|الثاني|الثالث|الرابع|الخامس|السادس|السابع|الثامن|التاسع|العاشر))", text)
    candidates = []
    for sec in sections:
        for para in re.split(r"\n\s*\n", sec):
            para = re.sub(r"\s+", " ", para).strip(" -•")
            if len(para) < 55:
                continue
            for sentence in re.split(r"(?<=[.!؟])\s+", para):
                sentence = sentence.strip(" -•")
                if 55 <= len(sentence) <= 420 and sentence not in candidates:
                    candidates.append(sentence)
    if len(candidates) < 10:
        # Fallback: use sufficiently long article lines if punctuation is sparse.
        for line in text.splitlines():
            line = re.sub(r"\s+", " ", line).strip(" -•")
            if 55 <= len(line) <= 420 and line not in candidates:
                candidates.append(line)
    if len(candidates) < 10:
        raise RuntimeError(f"Article yielded only {len(candidates)} usable facts; at least 10 required.")
    # Spread selection across the source rather than taking only the beginning.
    count = min(68, len(candidates))\n    picks = [candidates[round(i * (len(candidates)-1) / max(1, count-1))] for i in range(count)]
    checks = []
    for i, answer in enumerate(picks, 1):
        anchor = answer[:min(34, len(answer))]
        question = f"س: ما المعلومة التي يوضحها النص في هذا الموضع؟ «{anchor}...»\nج:"
        checks.append(question + answer + "\n")
    return checks

def chunks(ids, size=192, overlap=32):
    step = size - overlap
    for start in range(0, max(1, len(ids)-1), step):
        part = ids[start:start+size]
        if len(part) >= 3:
            yield part

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--article", default="مرجع_الغمر_الشامل_المنضبط_الإصدار_الثاني.txt")
    ap.add_argument("--init-url", default=REPO_RAW)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--lr", type=float, default=0.00003)
    ap.add_argument("--seconds", type=int, default=21000)
    ap.add_argument("--out", default="weights/nlm2_article_curriculum.bin")
    args = ap.parse_args()
    if args.rounds != 10:
        raise SystemExit("This curriculum is fixed to exactly 10 rounds.")
    article_path = Path(args.article)
    if not article_path.is_file():
        raise SystemExit(f"Missing article: {article_path}")
    text = article_path.read_text(encoding="utf-8")
    if len(text.strip()) < 5000:
        raise SystemExit("Article is unexpectedly short; refusing to train.")
    checks = build_checks(text)
    print(f"Article chars: {len(text)} | distinct source-based questions: {len(checks)} | rounds: {args.rounds}", flush=True)
    weights_path = Path("nlm2_base_for_article.bin")
    urllib.request.urlretrieve(args.init_url, weights_path)
    model = GRULM.from_bytes(weights_path.read_bytes())
    # Expand vocabulary while preserving every original embedding/output weight.
    missing = sorted(set(text) - set(model.chars))
    if missing:
        rng = np.random.default_rng(20261119)
        old_v = model.v
        new_v = old_v + len(missing)
        p0 = np.zeros((new_v, model.e), dtype=np.float32)
        p0[:old_v] = model.P[0]
        p0[old_v:] = rng.normal(0, 0.025, (new_v-old_v, model.e))
        p7 = np.zeros((model.h, new_v), dtype=np.float32)
        p7[:, :old_v] = model.P[7]
        p7[:, old_v:] = rng.normal(0, 0.035, (model.h, new_v-old_v))
        p8 = np.zeros((new_v,), dtype=np.float32)
        p8[:old_v] = model.P[8]
        model.P[0], model.P[7], model.P[8] = p0, p7, p8
        model.chars += missing
        model.v = new_v
        print(f"Expanded vocabulary by {len(missing)} characters; original weights preserved.", flush=True)
    torch.manual_seed(20261119)
    random.seed(20261119)
    params = [torch.nn.Parameter(torch.tensor(p.copy(), dtype=torch.float32)) for p in model.P]
    optimizer = torch.optim.Adam(params, lr=args.lr)
    start_time = time.time()
    steps = 0
    def train_sequence(seq):
        nonlocal steps
        if len(seq) < 3:
            return
        optimizer.zero_grad()
        h = model.h
        s1 = torch.zeros(h); s2 = torch.zeros(h)
        losses = []
        for t in range(len(seq)-1):
            ix = int(seq[t])
            gx = params[3] + params[0][ix] @ params[1]
            def cell(g, state, wh):
                a = g[:2*h] + state @ wh[:, :2*h]
                r = torch.sigmoid(a[:h]); z = torch.sigmoid(a[h:])
                n = torch.tanh(g[2*h:] + (r*state) @ wh[:, 2*h:])
                return (1-z)*state + z*n
            s1 = cell(gx, s1, params[2])
            s2 = cell(params[6] + s1 @ params[4], s2, params[5])
            logits = params[8] + s2 @ params[7]
            losses.append(torch.nn.functional.cross_entropy(logits.view(1,-1), torch.tensor([int(seq[t+1])])))
        loss = torch.stack(losses).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss; checkpoint not published.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        steps += 1
        if steps % 100 == 0:
            print(f"optimizer_steps={steps} loss={float(loss.detach()):.5f}", flush=True)
    for round_no in range(1, 11):
        print(f"ROUND {round_no}/10: full article reading", flush=True)
        ids = model.encode(text)
        article_chunks = list(chunks(ids))
        random.Random(7000 + round_no).shuffle(article_chunks)
        for part in article_chunks:
            train_sequence(part)
            if time.time() - start_time > args.seconds:
                raise RuntimeError("Time budget reached before all 10 rounds completed; refusing partial checkpoint.")
        print(f"ROUND {round_no}/10: 68 comprehension checks", flush=True)
        for item in checks:
            for part in chunks(model.encode(item), size=256, overlap=0):
                train_sequence(part)
            if __import__("time").time() - start_time > args.seconds:
                raise RuntimeError("Time budget reached before all 10 rounds completed; refusing partial checkpoint.")
    model.P = [p.detach().cpu().numpy().astype(np.float32) for p in params]
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    temp_out = out.with_suffix(out.suffix + ".tmp")
    model.save(temp_out)
    verify_temp = GRULM.from_bytes(temp_out.read_bytes())
    if verify_temp.v != model.v or verify_temp.chars != model.chars or any(not np.isfinite(p).all() for p in verify_temp.P):
        temp_out.unlink(missing_ok=True)
        raise RuntimeError("Temporary checkpoint validation failed.")
    temp_out.replace(out)
    verify = GRULM.from_bytes(out.read_bytes())
    if verify.e != model.e or verify.h != model.h or verify.v != model.v or verify.chars != model.chars or any(not np.isfinite(p).all() for p in verify.P):
        raise RuntimeError("Saved checkpoint failed NLM2 round-trip validation.")
    print(f"SUCCESS: rounds=10 questions_per_round={len(checks)} total_question_passes={10*len(checks)} optimizer_steps={steps} output={out} bytes={out.stat().st_size}", flush=True)

if __name__ == "__main__":
    main()
