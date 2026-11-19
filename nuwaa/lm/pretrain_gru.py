# -*- coding: utf-8 -*-
"""Train Android-compatible NLM2 weights from the repository's game-development corpus."""
import argparse, json, random, time
from pathlib import Path
import numpy as np
import torch
from nuwaa.lm.gru import GRULM

def load_pairs(root):
    """Load only the curated game-development/browser/code curriculum.
    Personal-memory and unrelated conversation JSONL files are deliberately excluded.
    No profanity/offensive-language filtering is applied to included curriculum text.
    """
    base = Path(root)
    files = [
        base / "data" / "new_knowledge.jsonl",
        base / "data" / "اكواد_طويلة.jsonl",
        base / "training" / "curriculum_game_dev.jsonl",
        base / "training" / "curriculum_browser_audio.jsonl",
        base / "training" / "curriculum_game_engine_testing.jsonl",
        base / "training" / "curriculum_game_physics_ai.jsonl",
    ]
    out = []
    for path in files:
        if not path.is_file():
            raise SystemExit(f"Required curated training file is missing: {path}")
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except ValueError as exc:
                raise SystemExit(f"Invalid JSONL at {path}:{line_no}: {exc}")
            user = item.get("user", "").strip()
            answer = item.get("bot", item.get("answer", "")).strip()
            if user and answer:
                out.append("س:" + user.replace("\n", " ¶ ") + "\nج:" + answer.replace("\n", "¶") + "\n")
    unique = list(dict.fromkeys(out))
    if len(unique) < 100:
        raise SystemExit(f"Curated game-development corpus too small: {len(unique)} unique examples")
    return unique

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",default="data"); ap.add_argument("--out",default="weights/nlm2.npz")
    ap.add_argument("--bin",default="nlm2.bin"); ap.add_argument("--init",default="")
    ap.add_argument("--epochs",type=int,default=12); ap.add_argument("--lr",type=float,default=.0003)
    ap.add_argument("--seconds",type=int,default=16000); ap.add_argument("--embed",type=int,default=64); ap.add_argument("--hidden",type=int,default=384)
    a=ap.parse_args()
    if a.epochs < 1 or a.lr <= 0 or a.seconds < 1:
        raise SystemExit("epochs, learning rate, and seconds must all be positive")
    torch.manual_seed(73); np.random.seed(73)
    texts=load_pairs(a.data)
    if len(texts) < 20: raise SystemExit(f"Insufficient training corpus: only {len(texts)} unique examples; refusing to train")
    # Keep complete Arabic/English/Python/HTML/CSS/JS character coverage.
    chars=["\n","�"]+sorted(set("".join(texts)) - {"\n","�"})
    if a.init and Path(a.init).exists():
        model=GRULM.from_bytes(Path(a.init).read_bytes())
        chars=list(dict.fromkeys(model.chars+chars)); e,h=model.e,model.h
        # Existing vocab is preserved; new characters are added only at the end.
        if chars!=model.chars:
            raise SystemExit("Vocabulary expansion requires a compatible widened checkpoint; refusing lossy resume")
    else:
        e,h=a.embed,a.hidden; model=GRULM(e,h,chars)
    # Adam + exact Nlm gate equations (reset applied before recurrent candidate matrix).
    params=[torch.nn.Parameter(torch.tensor(x.copy())) for x in model.P]
    opt=torch.optim.Adam(params,lr=a.lr)
    encoded=[torch.tensor(model.encode(s),dtype=torch.long) for s in texts]
    start=time.time(); best=float("inf"); step=0
    for epoch in range(a.epochs):
        random.Random(1000+epoch).shuffle(encoded)
        total=0.; count=0
        for seq in encoded:
            if len(seq)<3: continue
            opt.zero_grad(); s1=torch.zeros(h); s2=torch.zeros(h); losses=[]
            for t in range(len(seq)-1):
                ix=seq[t]
                gx=params[3]+params[0][ix]@params[1]
                def cell(gx,s,wh):
                    a0=gx[:2*h]+s@wh[:,:2*h]; r=torch.sigmoid(a0[:h]); z=torch.sigmoid(a0[h:])
                    n=torch.tanh(gx[2*h:]+(r*s)@wh[:,2*h:])
                    return (1-z)*s+z*n
                s1=cell(gx,s1,params[2]); s2=cell(params[6]+s1@params[4],s2,params[5])
                logits=params[8]+s2@params[7]
                losses.append(torch.nn.functional.cross_entropy(logits.view(1,-1),seq[t+1].view(1)))
            loss=torch.stack(losses).mean()
            if not torch.isfinite(loss): raise SystemExit("Non-finite loss; aborting without saving")
            loss.backward(); torch.nn.utils.clip_grad_norm_(params,1.0); opt.step()
            total+=float(loss.detach()); count+=1; step+=1
            if time.time()-start>a.seconds: break
        avg=total/max(count,1); print(f"epoch={epoch+1} loss={avg:.5f} sequences={count}",flush=True)
        if avg<best: best=avg
        if time.time()-start>a.seconds: break
    if step < 1:
        raise SystemExit("Training performed zero optimizer steps; refusing to save")
    model.P=[p.detach().cpu().numpy().astype(np.float32) for p in params]
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(a.out,**{f"p{i}":p for i,p in enumerate(model.P)})
    Path(a.bin).parent.mkdir(parents=True,exist_ok=True); model.save(a.bin)
    # Round-trip check before any workflow can publish this checkpoint.
    check=GRULM.from_bytes(Path(a.bin).read_bytes())
    if check.e!=e or check.h!=h or check.chars!=model.chars or any(not np.isfinite(p).all() for p in check.P):
        raise SystemExit("NLM2 round-trip validation failed")
    print(json.dumps({"examples":len(texts),"epochs_requested":a.epochs,"steps":step,"loss":best,"bytes":Path(a.bin).stat().st_size,"optimizer":"Adam","bptt":"full-character-sequence"},ensure_ascii=False),flush=True)
if __name__=="__main__": main()
