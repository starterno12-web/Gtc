# -*- coding: utf-8 -*-
"""Conservative clean-corpus fine-tuning plus exactly five memory questions for ten rounds."""
import argparse, json, math, random, re, time
from pathlib import Path
import numpy as np
import torch
from nuwaa.lm.gru import GRULM

MEMORY = [
    ("وين المفتاح؟", "فوق الدرج."),
    ("وين الحليب؟", "داخل الثلاجة."),
    ("وين الايس كريم؟", "مع أخوك."),
    ("وين راح الحمام؟", "هرب للسطح."),
    ("وين مفاتيح السيارة؟", "بالدرج الصغير."),
]
FILES = [
    "data/new_knowledge.jsonl",
    "data/اكواد_طويلة.jsonl",
    "training/curriculum_game_dev.jsonl",
    "training/curriculum_browser_audio.jsonl",
    "training/curriculum_game_engine_testing.jsonl",
    "training/curriculum_game_physics_ai.jsonl",
    "training/curriculum_geometry_vision_python_advanced.jsonl",
]
OLD_REPLY_MARKERS = (
    "تمام، حفظت", "حفظت إن", "أنا المحرك القديم", "زر التطوير",
    "اختبار التحقق", "لا أعدل نفسي", "بصراحة إني ما أعرف",
)
def fmt(q, a):
    return "س:" + q.replace("\n", " ¶ ") + "\nج:" + a.replace("\n", "¶") + "\n"

def clean_pair(q, a):
    q, a = str(q).strip(), str(a).strip()
    if not q or not a or len(q) > 700 or len(a) > 5000:
        return False
    if any(mark in a for mark in OLD_REPLY_MARKERS):
        return False
    text = q + a
    if any(ord(c) < 32 and c not in "\n\t\r" for c in text):
        return False
    if "\ufffd" in text or text.count("�") > 1:
        return False
    # Reject obvious repeated-character / duplicated-line corruption, not legitimate code.
    if re.search(r"(.)\1{14,}", a):
        return False
    lines = [x.strip() for x in a.splitlines() if x.strip()]
    if len(lines) >= 5 and len(set(lines)) <= max(1, len(lines)//4):
        return False
    return True

def load_clean_pairs():
    pairs, seen = [], set()
    for name in FILES:
        p = Path(name)
        if not p.is_file():
            raise SystemExit(f"Required historical training source missing: {name}")
        for no, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except ValueError as exc:
                raise SystemExit(f"Bad JSON in {name}:{no}: {exc}")
            q = item.get("user", "")
            a = item.get("bot", item.get("answer", ""))
            if clean_pair(q, a):
                value = (q.strip(), a.strip())
                if value not in seen:
                    seen.add(value); pairs.append(value)
    if len(pairs) < 50:
        raise SystemExit(f"Only {len(pairs)} clean replay examples; refusing to modify weights.")
    return pairs

def expand_vocab(model, texts):
    extra = sorted(set("".join(texts)) - set(model.chars))
    if not extra:
        return
    old_v, new_v = model.v, model.v + len(extra)
    rng = np.random.default_rng(20261119)
    p0 = np.zeros((new_v, model.e), np.float32); p0[:old_v] = model.P[0]
    p0[old_v:] = rng.normal(0, .015, (len(extra), model.e))
    p7 = np.zeros((model.h, new_v), np.float32); p7[:, :old_v] = model.P[7]
    p7[:, old_v:] = rng.normal(0, .015, (model.h, len(extra)))
    p8 = np.zeros((new_v,), np.float32); p8[:old_v] = model.P[8]
    model.P[0], model.P[7], model.P[8] = p0, p7, p8
    model.chars.extend(extra); model.v = new_v

def train_sequence(seq, params, optimizer, h):
    if len(seq) < 3: return None
    optimizer.zero_grad(set_to_none=True)
    s1 = torch.zeros(h); s2 = torch.zeros(h); losses = []
    for t in range(len(seq)-1):
        ix = int(seq[t])
        gx = params[3] + params[0][ix] @ params[1]
        def cell(g, s, wh):
            a = g[:2*h] + s @ wh[:, :2*h]
            r = torch.sigmoid(a[:h]); z = torch.sigmoid(a[h:])
            n = torch.tanh(g[2*h:] + (r*s) @ wh[:, 2*h:])
            return (1-z)*s + z*n
        s1 = cell(gx, s1, params[2])
        s2 = cell(params[6] + s1 @ params[4], s2, params[5])
        logits = params[8] + s2 @ params[7]
        losses.append(torch.nn.functional.cross_entropy(logits.view(1,-1), torch.tensor([int(seq[t+1])])))
    loss = torch.stack(losses).mean()
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite loss; original checkpoint remains untouched.")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, .5)
    optimizer.step()
    return float(loss.detach())

def generate(model, prompt, limit=100):
    h = model.h
    p = [torch.tensor(x.copy(), dtype=torch.float32) for x in model.P]
    s1 = torch.zeros(h); s2 = torch.zeros(h)
    def step(ix, s1, s2):
        gx = p[3] + p[0][ix] @ p[1]
        def cell(g, s, wh):
            a = g[:2*h] + s @ wh[:, :2*h]
            r = torch.sigmoid(a[:h]); z = torch.sigmoid(a[h:])
            n = torch.tanh(g[2*h:] + (r*s) @ wh[:, 2*h:])
            return (1-z)*s + z*n
        s1 = cell(gx, s1, p[2]); s2 = cell(p[6] + s1 @ p[4], s2, p[5])
        return p[8] + s2 @ p[7], s1, s2
    ids = model.encode(prompt)
    with torch.no_grad():
        logits = None
        for ix in ids: logits, s1, s2 = step(int(ix), s1, s2)
        out = []
        for _ in range(limit):
            ix = int(torch.argmax(logits).item())
            ch = model.chars[ix]
            if ch == "\n": break
            out.append(ch)
            logits, s1, s2 = step(ix, s1, s2)
    return "".join(out).strip()

def memory_test(model):
    rows=[]
    for round_no in range(1, 11):
        for q, expected in MEMORY:
            raw = generate(model, "س:" + q + "\nج:", 100)
            # Normalize light punctuation/spacing while preserving Arabic letters.
            norm = re.sub(r"[\s،.!؟]+", "", raw)
            exp = re.sub(r"[\s،.!؟]+", "", expected)
            rows.append({"round":round_no,"question":q,"expected":expected,"answer":raw,"pass":exp in norm})
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base",default="nlm2.bin")
    ap.add_argument("--out",default="weights/nlm2_clean_memory.bin")
    ap.add_argument("--report",default="memory_test_50.json")
    ap.add_argument("--lr",type=float,default=0.00001)
    ap.add_argument("--max-clean-examples",type=int,default=160)
    args=ap.parse_args()
    torch.set_num_threads(2); torch.manual_seed(20261119); np.random.seed(20261119)
    base=GRULM.from_bytes(Path(args.base).read_bytes())
    clean=load_clean_pairs()
    # Keep a balanced replay subset so useful old knowledge is reinforced without importing broad chat noise.
    if len(clean)>args.max_clean_examples:
        clean=sorted(clean,key=lambda x: (len(x[0])+len(x[1]), x[0]))[:args.max_clean_examples]
    texts=[fmt(q,a) for q,a in clean]+[fmt(q,a) for q,a in MEMORY]
    expand_vocab(base,texts)
    params=[torch.nn.Parameter(torch.tensor(x.copy(),dtype=torch.float32)) for x in base.P]
    opt=torch.optim.Adam(params,lr=args.lr)
    start=time.time(); losses=[]; steps=0
    # One conservative replay pass over vetted knowledge.
    random.Random(73).shuffle(clean)
    for q,a in clean:
        ids=base.encode(fmt(q,a)[:900])
        loss=train_sequence(ids,params,opt,base.h)
        if loss is not None: losses.append(loss); steps+=1
    # Exactly 10 rounds × 5 questions = 50 memory-question training passes.
    for round_no in range(1,11):
        for q,a in MEMORY:
            ids=base.encode(fmt(q,a))
            loss=train_sequence(ids,params,opt,base.h)
            if loss is not None: losses.append(loss); steps+=1
        print(f"memory_round={round_no}/10 questions=5",flush=True)
    base.P=[p.detach().cpu().numpy().astype(np.float32) for p in params]
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    tmp=out.with_suffix(out.suffix+".tmp"); base.save(tmp)
    check=GRULM.from_bytes(tmp.read_bytes())
    if check.chars!=base.chars or check.e!=base.e or check.h!=base.h or any(not np.isfinite(p).all() for p in check.P):
        raise RuntimeError("NLM2 round-trip validation failed; candidate not published.")
    tmp.replace(out)
    rows=memory_test(check)
    report={"rounds":10,"questions_per_round":5,"training_question_passes":50,
            "clean_replay_examples":len(clean),"optimizer_steps":steps,
            "mean_training_loss":round(float(np.mean(losses)),6) if losses else None,
            "memory_test_passes":sum(x["pass"] for x in rows),"memory_test_total":len(rows),
            "elapsed_seconds":round(time.time()-start,1),"candidate":str(out),"results":rows}
    Path(args.report).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    Path("memory_test_50.txt").write_text("\n".join(
        f"الجولة {r['round']} | {r['question']} | المتوقع: {r['expected']} | الناتج: {r['answer']} | {'نجح' if r['pass'] else 'لم يطابق'}"
        for r in rows)+"\n",encoding="utf-8")
    np.savez_compressed("weights/nlm2_clean_memory.npz",**{f"p{i}":p for i,p in enumerate(check.P)})
    print(json.dumps({k:v for k,v in report.items() if k!="results"},ensure_ascii=False),flush=True)

if __name__=="__main__": main()
