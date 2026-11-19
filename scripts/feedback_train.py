#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interactive correction-loop trainer for user-supplied text and QA pairs.

Dataset: training/feedback_pairs.jsonl
Each line: {"text":"...","questions":[{"q":"...","a":"..."}]}
No built-in/fabricated facts are used.
"""
import argparse, json, re, time
from pathlib import Path
import numpy as np
import torch
from nuwaa.lm.gru import GRULM

ROOT = Path(__file__).resolve().parents[1]

def normalize(s):
    s = str(s).lower()
    s = re.sub(r"[ًٌٍَُِّْـ]", "", s)
    s = re.sub(r"[\s،.!؟:؛,]+", "", s)
    return s.strip()

def fmt(text, q, a):
    return f"نص:{text}\nس:{q}\nج:{a}\n"

def train_seq(seq, params, opt, h):
    if len(seq) < 3: return
    opt.zero_grad(set_to_none=True)
    s1=torch.zeros(h); s2=torch.zeros(h); losses=[]
    for t in range(len(seq)-1):
        ix=int(seq[t]); gx=params[3]+params[0][ix]@params[1]
        def cell(g,s,wh):
            z=torch.sigmoid(g[:2*h]+s@wh[:,:2*h])
            r=z[:h]; gate=z[h:]
            n=torch.tanh(g[2*h:]+(r*s)@wh[:,2*h:])
            return (1-gate)*s+gate*n
        s1=cell(gx,s1,params[2]); s2=cell(params[6]+s1@params[4],s2,params[5])
        logits=params[8]+s2@params[7]
        losses.append(torch.nn.functional.cross_entropy(logits.view(1,-1),torch.tensor([int(seq[t+1])])))
    loss=torch.stack(losses).mean()
    if not torch.isfinite(loss): raise RuntimeError("Non-finite loss; candidate weights not saved.")
    loss.backward(); torch.nn.utils.clip_grad_norm_(params,.5); opt.step()

def generate(model, prompt, limit=160):
    p=[torch.tensor(x.copy(),dtype=torch.float32) for x in model.P]
    h=model.h; s1=torch.zeros(h); s2=torch.zeros(h)
    def step(ix,s1,s2):
        gx=p[3]+p[0][ix]@p[1]
        def cell(g,s,wh):
            z=torch.sigmoid(g[:2*h]+s@wh[:,:2*h]); r=z[:h]; gate=z[h:]
            n=torch.tanh(g[2*h:]+(r*s)@wh[:,2*h:])
            return (1-gate)*s+gate*n
        s1=cell(gx,s1,p[2]); s2=cell(p[6]+s1@p[4],s2,p[5])
        return p[8]+s2@p[7],s1,s2
    ids=model.encode(prompt)
    with torch.no_grad():
        logits=None
        for ix in ids: logits,s1,s2=step(int(ix),s1,s2)
        out=[]
        for _ in range(limit):
            ix=int(torch.argmax(logits).item()); ch=model.chars[ix]
            if ch=="\n": break
            out.append(ch); logits,s1,s2=step(ix,s1,s2)
    return "".join(out).strip()

def expand_vocab(model, texts):
    extra=sorted(set("".join(texts))-set(model.chars))
    if not extra: return
    old=model.v; new=old+len(extra); rng=np.random.default_rng(42)
    p0=np.zeros((new,model.e),np.float32); p0[:old]=model.P[0]; p0[old:]=rng.normal(0,.015,(len(extra),model.e))
    p7=np.zeros((model.h,new),np.float32); p7[:,:old]=model.P[7]; p7[:,old:]=rng.normal(0,.015,(model.h,len(extra)))
    p8=np.zeros((new,),np.float32); p8[:old]=model.P[8]
    model.P[0],model.P[7],model.P[8]=p0,p7,p8; model.chars.extend(extra); model.v=new

def main():
    ap=argparse.ArgumentParser(description="Train on your own text with answer/correct/retry feedback.")
    ap.add_argument("--dataset",default="training/feedback_pairs.jsonl")
    ap.add_argument("--base",default="nlm2.bin")
    ap.add_argument("--out",default="nlm2_feedback.bin")
    ap.add_argument("--read-times",type=int,default=20,choices=range(10,51))
    ap.add_argument("--max-retries",type=int,default=8)
    ap.add_argument("--lr",type=float,default=0.00002)
    args=ap.parse_args()
    torch.set_num_threads(2); torch.manual_seed(42); np.random.seed(42)
    dataset=Path(args.dataset)
    if not dataset.is_file(): raise SystemExit(f"Dataset not found: {dataset}. Create JSONL with text/questions.")
    items=[json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not items: raise SystemExit("Dataset is empty.")
    for item in items:
        if not item.get("text") or not item.get("questions"): raise SystemExit("Each record needs text and questions.")
        for qa in item["questions"]:
            if not qa.get("q") or not qa.get("a"): raise SystemExit("Every question needs q and a fields.")
    base=GRULM.from_bytes(Path(args.base).read_bytes())
    all_text=[item["text"]+ "".join(q["q"]+q["a"] for q in item["questions"]) for item in items]
    expand_vocab(base,all_text)
    params=[torch.nn.Parameter(torch.tensor(x.copy(),dtype=torch.float32)) for x in base.P]
    opt=torch.optim.Adam(params,lr=args.lr)
    total=0; passed=0; report=[]; start=time.time()
    # Reading/repetition: supervised text+QA examples, 10-50 passes.
    for epoch in range(args.read_times):
        for item in items:
            for qa in item["questions"]:
                ids=base.encode(fmt(item["text"],qa["q"],qa["a"]))
                train_seq(ids,params,opt,base.h); total+=1
    # Feedback loop: ask, compare to answer, train on correction, and re-ask same question.
    base.P=[p.detach().numpy().astype(np.float32) for p in params]
    for item_no,item in enumerate(items,1):
        for qa in item["questions"]:
            prompt=f"نص:{item['text']}\nس:{qa['q']}\nج:"
            expected=normalize(qa["a"]); success=False; attempts=[]
            for attempt in range(1,args.max_retries+1):
                answer=generate(base,prompt)
                ok=bool(expected) and expected in normalize(answer)
                attempts.append({"try":attempt,"answer":answer,"correct":ok})
                print(f"[item {item_no}] {qa['q']} | {answer} | {'PASS' if ok else 'FAIL'}",flush=True)
                if ok: success=True; passed+=1; break
                # Wrong attempt: return to the source answer, reinforce it, then retry same prompt.
                ids=base.encode(fmt(item["text"],qa["q"],qa["a"]))
                train_seq(ids,params,opt,base.h); total+=1
                base.P=[p.detach().numpy().astype(np.float32) for p in params]
            report.append({"question":qa["q"],"expected":qa["a"],"success":success,"attempts":attempts})
    base.P=[p.detach().cpu().numpy().astype(np.float32) for p in params]
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    tmp=out.with_suffix(out.suffix+".tmp"); base.save(tmp)
    check=GRULM.from_bytes(tmp.read_bytes())
    if check.chars!=base.chars or any(not np.isfinite(p).all() for p in check.P): raise RuntimeError("Candidate validation failed.")
    tmp.replace(out)
    report_path=out.with_suffix(".feedback_report.json")
    report_path.write_text(json.dumps({"read_times":args.read_times,"max_retries":args.max_retries,"questions_passed":passed,"questions_total":len(report),"training_steps":total,"seconds":round(time.time()-start,1),"results":report},ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"Saved candidate: {out}\nReport: {report_path}\nPassed: {passed}/{len(report)}")

if __name__=="__main__": main()
