#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Auto-question trainer with in-generation repetition blocking and strict correction."""
import argparse, re, json, time, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from nuwaa.lm.gru import GRULM

def norm(s):
    s=str(s).lower()
    s=re.sub(r"[ًٌٍَُِّْـ]","",s)
    return re.sub(r"[\s،.!؟:؛,]+","",s)

def auto_questions(text):
    pairs=[]
    pattern=r"(.{1,45}?)\s+(?:داخل|جوة|جوا|في)\s+(.{1,45}?)(?:[.،؛!؟\n]|$)"
    for m in re.finditer(pattern,text):
        subject=m.group(1).strip(" \t،.؛");place=m.group(2).strip(" \t،.؛")
        if subject and place:pairs.append((f"وين {subject}؟",f"{place}."))
    unique=[];seen=set()
    for q,a in pairs:
        key=(norm(q),norm(a))
        if key not in seen and len(q)<80 and len(a)<70:
            seen.add(key);unique.append((q,a))
    return unique

def fmt(source,q,a):
    return f"نص:{source}\nس:{q}\nج:{a}\n"

def train_seq(seq,params,opt,h):
    if len(seq)<3:return
    opt.zero_grad(set_to_none=True);s1=torch.zeros(h);s2=torch.zeros(h);losses=[]
    for t in range(len(seq)-1):
        ix=int(seq[t]);gx=params[3]+params[0][ix]@params[1]
        def cell(g,s,wh):
            gates=g[:2*h]+s@wh[:,:2*h];r=torch.sigmoid(gates[:h]);z=torch.sigmoid(gates[h:])
            n=torch.tanh(g[2*h:]+(r*s)@wh[:,2*h:]);return (1-z)*s+z*n
        s1=cell(gx,s1,params[2]);s2=cell(params[6]+s1@params[4],s2,params[5])
        logits=params[8]+s2@params[7]
        losses.append(torch.nn.functional.cross_entropy(logits.view(1,-1),torch.tensor([int(seq[t+1])])))
    loss=torch.stack(losses).mean()
    if not torch.isfinite(loss):raise RuntimeError("Non-finite loss; training aborted.")
    loss.backward();torch.nn.utils.clip_grad_norm_(params,.5);opt.step()

def generate(model,prompt,limit=120,temperature=.85):
    p=[torch.tensor(x.copy(),dtype=torch.float32) for x in model.P]
    h=model.h;s1=torch.zeros(h);s2=torch.zeros(h)
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
        out=[];ngram=4
        for _ in range(limit):
            probs=torch.softmax(logits/max(temperature,.05),dim=-1)
            candidates=torch.argsort(probs,descending=True).tolist()
            current="".join(out);selected=None
            for candidate in candidates:
                ch=model.chars[int(candidate)]
                if ch=="\n":selected=int(candidate);break
                trial=current+ch
                if len(trial)>=ngram:
                    newgram=trial[-ngram:]
                    if newgram in [trial[i:i+ngram] for i in range(max(0,len(trial)-ngram))]:
                        continue
                selected=int(candidate);break
            if selected is None:
                # If every next character would repeat, end this attempt; trainer rejects it.
                break
            ch=model.chars[selected]
            if ch=="\n":break
            out.append(ch);logits,s1,s2=step(selected,s1,s2)
    answer="".join(out).strip()
    compact=norm(answer)
    repeated=any(compact[i:i+5] in compact[:i] for i in range(2,max(2,len(compact)-4)))
    return answer,repeated

def extend_vocab(model,texts):
    extra=sorted(set("".join(texts))-set(model.chars))
    if not extra:return
    old=model.v;new=old+len(extra);rng=np.random.default_rng(17)
    p0=np.zeros((new,model.e),np.float32);p0[:old]=model.P[0];p0[old:]=rng.normal(0,.015,(len(extra),model.e))
    p7=np.zeros((model.h,new),np.float32);p7[:,:old]=model.P[7];p7[:,old:]=rng.normal(0,.015,(model.h,len(extra)))
    p8=np.zeros(new,np.float32);p8[:old]=model.P[8]
    model.P[0],model.P[7],model.P[8]=p0,p7,p8;model.chars.extend(extra);model.v=new

def main():
    ap=argparse.ArgumentParser(description="Train from text facts and retry each generated question until correct.")
    ap.add_argument("--text",default="training/feedback_text.txt")
    ap.add_argument("--behavior",default="training_behavior.txt")
    ap.add_argument("--base",default="nlm2.bin");ap.add_argument("--out",default="nlm2_feedback.bin")
    ap.add_argument("--read-times",type=int,default=20)
    ap.add_argument("--max-retries",type=int,default=0,help="0 means unlimited retries; nonzero aborts, never skips.")
    ap.add_argument("--lr",type=float,default=.00002);args=ap.parse_args()
    torch.set_num_threads(2);torch.manual_seed(17);np.random.seed(17)
    source=Path(args.text).read_text(encoding="utf-8").strip()
    qa=auto_questions(source)
    if not qa:raise SystemExit("No supported location facts found in text.")
    print(f"Generated {len(qa)} questions:",flush=True)
    for q,a in qa:print(f"Q: {q} | target: {a}",flush=True)
    model=GRULM.from_bytes(Path(args.base).read_bytes())
    behavior=Path(args.behavior).read_text(encoding="utf-8").strip()
    phase1=behavior+"\n"
    training=[fmt(source,q,a) for q,a in qa];extend_vocab(model,[phase1]+training)
    params=[torch.nn.Parameter(torch.tensor(x.copy(),dtype=torch.float32)) for x in model.P]
    opt=torch.optim.Adam(params,lr=args.lr);steps=0;report=[];start=time.time();out=Path(args.out)
    def save_progress():
        model.P=[p.detach().cpu().numpy().astype(np.float32) for p in params]
        model.save(out)
        out.with_suffix(".feedback_report.json").write_text(json.dumps({
          "source":args.text,"read_times":args.read_times,"max_retries":args.max_retries,
          "passed":sum(bool(r["passed"]) for r in report),"total_questions":len(qa),
          "steps":steps,"elapsed_seconds":round(time.time()-start,1),"details":report
        },ensure_ascii=False,indent=2),encoding="utf-8")
    for epoch in range(args.read_times):
        # Phase 1: manifesto-only training, no generation/questions/tests.
        train_seq(model.encode(phase1),params,opt,model.h);steps+=1
        if (epoch+1) % 50 == 0: print('Manifesto pass '+str(epoch+1)+'/'+str(args.read_times),flush=True)
    print('Phase 1 complete; starting fact/question phase.',flush=True)
    for q,a in qa:
        prompt=f"نص:{source}\nس:{q}\nج:";target=norm(a);attempts=[];n=0
        while True:
            n+=1
            time.sleep(5)
            ans,repeated=generate(model,prompt);normalized=norm(ans)
            ok=bool(target) and normalized==target and not repeated
            attempts.append({"attempt":n,"answer":ans,"correct":ok,"repetition_blocked":repeated})
            print(f"Q: {q}\nA: {ans}\n{'صحيح' if ok else 'خطأ - تدريب وإعادة'}",flush=True)
            if ok:
                report.append({"q":q,"expected":a,"passed":True,"attempts":attempts})
                save_progress();break
            train_seq(model.encode(fmt(source,q,a)),params,opt,model.h);steps+=1
            save_progress()
            if args.max_retries and n>=args.max_retries:
                raise RuntimeError(f"Retry limit reached at {q}; aborting without skipping.")
    model.P=[p.detach().cpu().numpy().astype(np.float32) for p in params]
    tmp=out.with_suffix(out.suffix+".tmp");model.save(tmp)
    chk=GRULM.from_bytes(tmp.read_bytes())
    if chk.chars!=model.chars or any(not np.isfinite(p).all() for p in chk.P):
        raise RuntimeError("Checkpoint validation failed.")
    tmp.replace(out)
    out.with_suffix(".feedback_report.json").write_text(json.dumps({
      "source":args.text,"read_times":args.read_times,"max_retries":args.max_retries,
      "passed":sum(bool(r["passed"]) for r in report),"total_questions":len(report),
      "steps":steps,"elapsed_seconds":round(time.time()-start,1),"details":report
    },ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"Checkpoint saved: {out}; report: {out.with_suffix('.feedback_report.json')}",flush=True)
if __name__=="__main__":main()
