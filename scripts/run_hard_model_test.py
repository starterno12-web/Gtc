# -*- coding: utf-8 -*-
"""Generate a hard held-out coding-test answer from an NLM2 GRU checkpoint."""
import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from nuwaa.lm.gru import GRULM

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--weights",default="weights/nlm2_article_curriculum.bin")
    ap.add_argument("--prompt",default="tests/hard_coding_prompt.txt")
    ap.add_argument("--out",default="hard_test_answer.txt")
    ap.add_argument("--tokens",type=int,default=1800)
    ap.add_argument("--temperature",type=float,default=0.85)
    ap.add_argument("--top-k",type=int,default=12)
    args=ap.parse_args()
    model=GRULM.from_bytes(Path(args.weights).read_bytes())
    prompt=Path(args.prompt).read_text(encoding="utf-8")
    ids=model.encode(prompt)
    p=[torch.tensor(x.copy(),dtype=torch.float32) for x in model.P]
    h=model.h
    s1=torch.zeros(h); s2=torch.zeros(h)
    def step(ix,s1,s2):
        gx=p[3]+p[0][ix]@p[1]
        def cell(g,s,wh):
            a=g[:2*h]+s@wh[:,:2*h]
            r=torch.sigmoid(a[:h]); z=torch.sigmoid(a[h:])
            n=torch.tanh(g[2*h:]+(r*s)@wh[:,2*h:])
            return (1-z)*s+z*n
        s1=cell(gx,s1,p[2])
        s2=cell(p[6]+s1@p[4],s2,p[5])
        logits=p[8]+s2@p[7]
        return logits,s1,s2
    torch.manual_seed(20261120); random.seed(20261120)
    with torch.no_grad():
        logits=None
        for ix in ids:
            logits,s1,s2=step(int(ix),s1,s2)
        generated=[]
        for _ in range(args.tokens):
            scores=logits/ max(args.temperature,0.05)
            k=min(max(args.top_k,1),scores.numel())
            vals,inds=torch.topk(scores,k)
            probs=torch.softmax(vals,dim=0)
            chosen=int(inds[torch.multinomial(probs,1)].item())
            generated.append(chosen)
            logits,s1,s2=step(chosen,s1,s2)
    answer="".join(model.chars[i] for i in generated)
    result=("PROMPT:\n"+prompt+"\n\nMODEL OUTPUT (RAW):\n"+answer+"\n")
    Path(args.out).write_text(result,encoding="utf-8")
    print(json.dumps({"status":"success","checkpoint":args.weights,"vocab":model.v,"prompt_chars":len(prompt),"generated_chars":len(answer),"output":args.out},ensure_ascii=False))
if __name__=="__main__": main()
