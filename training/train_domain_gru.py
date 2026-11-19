#!/usr/bin/env python3
"""Cumulative domain GRU trainer. Resume compatible weights and expand vocabulary safely."""
import argparse, json, pathlib, random
import torch
from torch import nn
def load_data(root):
    records=[]
    for path in sorted(pathlib.Path(root).glob("curriculum_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            row=json.loads(line)
            if row.get("user") and row.get("bot"): records.append("س: "+row["user"]+"\nج: "+row["bot"]+"\n")
    if not records: raise RuntimeError("No curriculum_*.jsonl data found")
    return records
class CharGRU(nn.Module):
    def __init__(self,vocab_size,emb=96,hidden=256):
        super().__init__(); self.emb=nn.Embedding(vocab_size,emb); self.gru=nn.GRU(emb,hidden,batch_first=True); self.head=nn.Linear(hidden,vocab_size)
    def forward(self,x):
        z,_=self.gru(self.emb(x)); return self.head(z)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--epochs",type=int,default=40); ap.add_argument("--seq-len",type=int,default=160); ap.add_argument("--batch-size",type=int,default=32); ap.add_argument("--checkpoint",default="training/checkpoints/game_domain_master.pt"); args=ap.parse_args()
    random.seed(42); torch.manual_seed(42)
    records=load_data("training"); corpus="".join(records); chars=sorted(set(corpus)); stoi={c:i for i,c in enumerate(chars)}; ids=torch.tensor([stoi[c] for c in corpus],dtype=torch.long)
    split=max(args.seq_len+1,int(len(ids)*0.92)); train=ids[:split]; valid=ids[split:]
    if len(valid)<args.seq_len+1: valid=ids[-(args.seq_len+1):]
    model=CharGRU(len(chars)); previous_records=0; old_rounds=0
    cp=pathlib.Path(args.checkpoint)
    if cp.exists():
        ck=torch.load(cp,map_location="cpu",weights_only=False); old_chars=ck.get("vocab",[]); old_state=ck.get("model_state",{})
        with torch.no_grad():
            for oi,c in enumerate(old_chars):
                if c not in stoi: continue
                ni=stoi[c]
                if old_state.get("emb.weight") is not None: model.emb.weight[ni].copy_(old_state["emb.weight"][oi])
                if old_state.get("head.weight") is not None:
                    model.head.weight[ni].copy_(old_state["head.weight"][oi]); model.head.bias[ni].copy_(old_state["head.bias"][oi])
            for key in ("gru.weight_ih_l0","gru.weight_hh_l0","gru.bias_ih_l0","gru.bias_hh_l0"):
                if key in old_state: model.state_dict()[key].copy_(old_state[key])
        previous_records=ck.get("records",0); old_rounds=ck.get("training_rounds",0); print(f"Resumed prior checkpoint (records={previous_records}, rounds={old_rounds})")
    opt=torch.optim.AdamW(model.parameters(),lr=2e-4); loss_fn=nn.CrossEntropyLoss()
    def batches(src,steps=80):
        n=max(1,len(src)-args.seq_len-1)
        for _ in range(steps):
            ix=torch.randint(0,n,(args.batch_size,)); x=torch.stack([src[i:i+args.seq_len] for i in ix]); y=torch.stack([src[i+1:i+args.seq_len+1] for i in ix]); yield x,y
    best=float("inf")
    for ep in range(1,args.epochs+1):
        model.train(); losses=[]
        for x,y in batches(train):
            opt.zero_grad(); logits=model(x); loss=loss_fn(logits.reshape(-1,len(chars)),y.reshape(-1)); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            x=valid[:-1].unsqueeze(0); y=valid[1:].unsqueeze(0); val=loss_fn(model(x).reshape(-1,len(chars)),y.reshape(-1)).item()
        print(f"epoch {ep:03d} train_loss={sum(losses)/len(losses):.4f} valid_loss={val:.4f}")
        if val<best:
            best=val; cp.parent.mkdir(parents=True,exist_ok=True)
            torch.save({"model_state":model.state_dict(),"vocab":chars,"config":{"embedding":96,"hidden":256,"char_level":True},"best_valid_loss":best,"records":previous_records+len(records),"training_rounds":old_rounds+1,"data_files":[p.name for p in sorted(pathlib.Path("training").glob("curriculum_*.jsonl"))]},cp)
    print(f"Saved cumulative checkpoint: {cp}; validation loss={best:.4f}; records total={previous_records+len(records)}")
if __name__=="__main__": main()
