#!/usr/bin/env python3
"""Train the consolidated, deduplicated master corpus and resume a compatible GRU checkpoint."""
import argparse, json, pathlib, random, torch
from torch import nn

class CharGRU(nn.Module):
    def __init__(self, n, emb=96, hidden=256):
        super().__init__()
        self.emb=nn.Embedding(n,emb)
        self.gru=nn.GRU(emb,hidden,batch_first=True)
        self.head=nn.Linear(hidden,n)
    def forward(self,x):
        z,_=self.gru(self.emb(x)); return self.head(z)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--epochs",type=int,default=40)
    ap.add_argument("--seq-len",type=int,default=160)
    ap.add_argument("--batch-size",type=int,default=32)
    ap.add_argument("--checkpoint",default="training/checkpoints/game_domain_master.pt")
    args=ap.parse_args()
    random.seed(42); torch.manual_seed(42)
    path=pathlib.Path("training/master_training_corpus.jsonl")
    records=[json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    text="".join("س: "+r["user"]+"\nج: "+r["bot"]+"\n" for r in records)
    vocab=sorted(set(text)); stoi={c:i for i,c in enumerate(vocab)}
    ids=torch.tensor([stoi[c] for c in text],dtype=torch.long)
    seq=args.seq_len
    if len(ids)<seq*2: seq=max(8,len(ids)//4)
    split=max(seq+1,int(len(ids)*0.92)); split=min(split,len(ids)-seq-1)
    train,valid=ids[:split],ids[split:]
    if len(valid)<seq+1: valid=ids[-(seq+1):]
    model=CharGRU(len(vocab)); checkpoint=pathlib.Path(args.checkpoint)
    prior_records=0; rounds=0; sources=[]
    if checkpoint.exists():
        ck=torch.load(checkpoint,map_location="cpu",weights_only=False); oldv=ck.get("vocab",[]); old=ck.get("model_state",{})
        with torch.no_grad():
            for oi,c in enumerate(oldv):
                if c not in stoi: continue
                ni=stoi[c]
                if "emb.weight" in old: model.emb.weight[ni].copy_(old["emb.weight"][oi])
                if "head.weight" in old: model.head.weight[ni].copy_(old["head.weight"][oi])
                if "head.bias" in old: model.head.bias[ni].copy_(old["head.bias"][oi])
            for key in ("gru.weight_ih_l0","gru.weight_hh_l0","gru.bias_ih_l0","gru.bias_hh_l0"):
                if key in old: model.state_dict()[key].copy_(old[key])
        prior_records=int(ck.get("records",0)); rounds=int(ck.get("training_rounds",0))
        sources=ck.get("checkpoint_sources",[])+ck.get("consolidation_manifest",{}).get("source_files",[])
        print(f"Resumed from master checkpoint; previous records={prior_records}, rounds={rounds}")
    opt=torch.optim.AdamW(model.parameters(),lr=2e-4); lossfn=nn.CrossEntropyLoss()
    def batches(src,steps=80):
        n=max(1,len(src)-seq-1)
        for _ in range(steps):
            ix=torch.randint(0,n,(args.batch_size,))
            x=torch.stack([src[i:i+seq] for i in ix])
            y=torch.stack([src[i+1:i+seq+1] for i in ix])
            yield x,y
    best=float("inf")
    for ep in range(1,args.epochs+1):
        model.train(); ls=[]
        for x,y in batches(train):
            opt.zero_grad(); out=model(x); loss=lossfn(out.reshape(-1,len(vocab)),y.reshape(-1))
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); ls.append(loss.item())
        model.eval()
        with torch.no_grad():
            x=valid[:-1].unsqueeze(0); y=valid[1:].unsqueeze(0)
            val=lossfn(model(x).reshape(-1,len(vocab)),y.reshape(-1)).item()
        print(f"epoch {ep:03d} train_loss={sum(ls)/len(ls):.4f} valid_loss={val:.4f}")
        if val<best:
            best=val; checkpoint.parent.mkdir(parents=True,exist_ok=True)
            torch.save({"model_state":model.state_dict(),"vocab":vocab,"config":{"embedding":96,"hidden":256,"char_level":True},"best_valid_loss":best,"records":prior_records+len(records),"training_rounds":rounds+1,"corpus_file":path.name,"corpus_records":len(records),"checkpoint_sources":sources},checkpoint)
    print(f"Master checkpoint saved: {checkpoint}; best validation loss={best:.4f}; total records={prior_records+len(records)}")
if __name__=="__main__": main()
