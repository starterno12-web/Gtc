#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Automatic Arabic fact-question feedback-loop trainer. User supplies TEXT ONLY."""
import argparse, re, json, time
from pathlib import Path
import numpy as np
import torch
from nuwaa.lm.gru import GRULM

ROOT=Path(__file__).resolve().parents[1]
def norm(s):
    s=str(s).lower()
    s=re.sub(r"[ًٌٍَُِّْـ]","",s)
    return re.sub(r"[\s،.!؟:؛,]+","",s)
def auto_questions(text):
    # Extract simple Arabic location statements such as "المفتاح داخل الدرج" or "المفتاح جوة الدرج".
    pairs=[]
    patterns=[
      r"(.{1,45}?)\s+(?:داخل|جوة|جوا|في)\s+(.{1,45}?)(?:[.،؛!؟\n]|$)",
    ]
    for pat in patterns:
      for m in re.finditer(pat,text):
        subject=m.group(1).strip(" \t،.؛"); place=m.group(2).strip(" \t،.؛")
        if subject and place:
          q=f"وين {subject}؟"; a=f"{place}."
          pairs.append((q,a))
    # Deduplicate and ignore noisy overly broad matches.
    out=[]; seen=set()
    for q,a in pairs:
      key=(norm(q),norm(a))
      if key not in seen and len(q)<80 and len(a)<70:
        seen.add(key); out.append((q,a))
    return out
def fmt(source,q,a):
    return f"نص:{source}\nس:{q}\nج:{a}\n"
def train_seq(seq,params,opt,h):
    if len(seq)<3:return
    opt.zero_grad(set_to_none=True); s1=torch.zeros(h); s2=torch.zeros(h); losses=[]
    for t in range(len(seq)-1):
      ix=int(seq[t]); gx=params[3]+params[0][ix]@params[1]
      def cell(g,s,wh):
        gates=g[:2*h]+s@wh[:,:2*h]; r=torch.sigmoid(gates[:h]); z=torch.sigmoid(gates[h:])
        n=torch.tanh(g[2*h:]+(r*s)@wh[:,2*h:])
        return (1-z)*s+z*n
      s1=cell(gx,s1,params[2]); s2=cell(params[6]+s1@params[4],s2,params[5])
      logits=params[8]+s2@params[7]
      losses.append(torch.nn.functional.cross_entropy(logits.view(1,-1),torch.tensor([int(seq[t+1])])))
    loss=torch.stack(losses).mean()
    if not torch.isfinite(loss):raise RuntimeError("Invalid loss; candidate checkpoint not saved.")
    loss.backward();torch.nn.utils.clip_grad_norm_(params,.5);opt.step()
def generate(model,prompt,limit=100):
    p=[torch.tensor(x.copy(),dtype=torch.float32) for x in model.P];h=model.h;s1=torch.zeros(h);s2=torch.zeros(h)
    def step(ix,s1,s2):
      gx=p[3]+p[0][ix]@p[1]
      def cell(g,s,wh):
        gates=g[:2*h]+s@wh[:,:2*h];r=torch.sigmoid(gates[:h]);z=torch.sigmoid(gates[h:])
        n=torch.tanh(g[2*h:]+(r*s)@wh[:,2*h:]);return (1-z)*s+z*n
      s1=cell(gx,s1,p[2]);s2=cell(p[6]+s1@p[4],s2,p[5])
      return p[8]+s2@p[7],s1,s2
    ids=model.encode(prompt)
    with torch.no_grad():
      logits=None
      for ix in ids:logits,s1,s2=step(int(ix),s1,s2)
      out=[]
      for _ in range(limit):
        ix=int(torch.argmax(logits).item());ch=model.chars[ix]
        if ch=="\n":break
        out.append(ch);logits,s1,s2=step(ix,s1,s2)
    return "".join(out).strip()
def extend_vocab(model,texts):
    extra=sorted(set("".join(texts))-set(model.chars))
    if not extra:return
    old=model.v;new=old+len(extra);rng=np.random.default_rng(17)
    p0=np.zeros((new,model.e),np.float32);p0[:old]=model.P[0];p0[old:]=rng.normal(0,.015,(len(extra),model.e))
    p7=np.zeros((model.h,new),np.float32);p7[:,:old]=model.P[7];p7[:,old:]=rng.normal(0,.015,(model.h,len(extra)))
    p8=np.zeros(new,np.float32);p8[:old]=model.P[8]
    model.P[0],model.P[7],model.P[8]=p0,p7,p8;model.chars.extend(extra);model.v=new
def main():
    ap=argparse.ArgumentParser(description="Auto-generate simple questions from Arabic location sentences, train, retry after correction.")
    ap.add_argument("--text",default="training/feedback_text.txt")
    ap.add_argument("--base",default="nlm2.bin");ap.add_argument("--out",default="nlm2_feedback.bin")
    ap.add_argument("--read-times",type=int,default=20,choices=range(10,51));ap.add_argument("--max-retries",type=int,default=8)
    ap.add_argument("--lr",type=float,default=0.00002);args=ap.parse_args()
    torch.set_num_threads(2);torch.manual_seed(17);np.random.seed(17)
    source=Path(args.text).read_text(encoding="utf-8").strip()
    qa=auto_questions(source)
    if not qa:raise SystemExit("No simple location facts recognized. Use sentences like: المفتاح داخل الدرج.")
    print(f"Auto-created {len(qa)} questions:",flush=True)
    for q,a in qa:print(f"Q: {q} | expected: {a}",flush=True)
    model=GRULM.from_bytes(Path(args.base).read_bytes())
    training=[fmt(source,q,a) for q,a in qa]
    extend_vocab(model,training)
    params=[torch.nn.Parameter(torch.tensor(x.copy(),dtype=torch.float32)) for x in model.P]
    opt=torch.optim.Adam(params,lr=args.lr);steps=0;report=[];start=time.time()
    # Repeat reading/teacher-forced practice 10-50 passes.
    for epoch in range(args.read_times):
      for seq in training:
        train_seq(model.encode(seq),params,opt,model.h);steps+=1
    model.P=[p.detach().numpy().astype(np.float32) for p in params]
    # Error-feedback: ask same question, reinforce correct source pair, then retry.
    for q,a in qa:
      prompt=f"نص:{source}\nس:{q}\nج:";target=norm(a);attempts=[];passed=False
      for n in range(1,args.max_retries+1):
        ans=generate(model,prompt);ok=bool(target) and target in norm(ans)
        attempts.append({"attempt":n,"answer":ans,"correct":ok})
        print(f"{q} => {ans} {'PASS' if ok else 'MISS'}",flush=True)
        if ok:passed=True;break
        train_seq(model.encode(fmt(source,q,a)),params,opt,model.h);steps+=1
        model.P=[p.detach().numpy().astype(np.float32) for p in params]
      report.append({"q":q,"expected":a,"passed":passed,"attempts":attempts})
    model.P=[p.detach().cpu().numpy().astype(np.float32) for p in params]
    out=Path(args.out);tmp=out.with_suffix(out.suffix+".tmp");model.save(tmp)
    chk=GRULM.from_bytes(tmp.read_bytes())
    if chk.chars!=model.chars or any(not np.isfinite(p).all() for p in chk.P):raise RuntimeError("Candidate validation failed.")
    tmp.replace(out)
    out.with_suffix(".feedback_report.json").write_text(json.dumps({"source":str(args.text),"read_times":args.read_times,"max_retries":args.max_retries,"passes":sum(r["passed"] for r in report),"total_questions":len(report),"steps":steps,"duration_s":round(time.time()-start,1),"details":report},ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"Candidate checkpoint saved: {out}; report: {out.with_suffix('.feedback_report.json')}",flush=True)
if __name__=="__main__":main()
