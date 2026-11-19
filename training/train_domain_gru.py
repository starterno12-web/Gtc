#!/usr/bin/env python3
"""Train a small domain language model from the curated game-dev curriculum.
This is a research checkpoint; it is NOT compatible with Android's nlm2.bin format.
"""
import argparse, json, random, pathlib, re, sys
try:
    import torch
    from torch import nn
except ImportError:
    print("PyTorch is required: pip install torch", file=sys.stderr); sys.exit(2)

def load_lines(path):
    rows=[]
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        x=json.loads(line)
        if x.get("user") and x.get("bot"): rows.append(f"س: {x['user']}\nج: {x['bot']}\n")
    return rows

class CharGRU(nn.Module):
    def __init__(self, vocab_size, emb=96, hidden=256):
        super().__init__()
        self.emb=nn.Embedding(vocab_size,emb)
        self.gru=nn.GRU(emb,hidden,batch_first=True)
        self.head=nn.Linear(hidden,vocab_size)
    def forward(self,x):
        z,_=self.gru(self.emb(x))
        return self.head(z)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",default="training/curriculum_game_dev.jsonl")
    p.add_argument("--epochs",type=int,default=30)
    p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--seq-len",type=int,default=160)
    p.add_argument("--out",default="training/checkpoints/game_domain_gru.pt")
    p.add_argument("--seed",type=int,default=42)
    args=p.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    rows=load_lines(args.data)
    if len(rows)<10: raise SystemExit("Dataset too small; expected at least 10 records.")
    # Character-level tokenization preserves Arabic, English, punctuation, and code syntax.
    text="".join(rows); chars=sorted(set(text)); stoi={c:i for i,c in enumerate(chars)}
    ids=torch.tensor([stoi[c] for c in text],dtype=torch.long)
    split=max(args.seq_len+1,int(len(ids)*0.95)); train=ids[:split]; valid=ids[split:]
    if len(valid)<args.seq_len+1: valid=ids[-max(args.seq_len+1, len(ids)//10):]
    model=CharGRU(len(chars)); opt=torch.optim.AdamW(model.parameters(),lr=3e-4)
    loss_fn=nn.CrossEntropyLoss()
    def batches(src):
        n=max(1,(len(src)-1)//args.seq_len)
        for _ in range(args.batch_size):
            starts=torch.randint(0,n,(args.batch_size,))
            ix=starts*args.seq_len
            x=torch.stack([src[i:i+args.seq_len] for i in ix])
            y=torch.stack([src[i+1:i+args.seq_len+1] for i in ix])
            yield x,y
    best=float("inf")
    for epoch in range(1,args.epochs+1):
        model.train(); losses=[]
        for x,y in batches(train):
            opt.zero_grad(); logits=model(x)
            loss=loss_fn(logits.reshape(-1,len(chars)),y.reshape(-1))
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            x=valid[:-1].unsqueeze(0); y=valid[1:].unsqueeze(0)
            val=loss_fn(model(x).reshape(-1,len(chars)),y.reshape(-1)).item()
        print(f"epoch {epoch:03d} train_loss={sum(losses)/len(losses):.4f} valid_loss={val:.4f}")
        if val<best:
            best=val
            out=pathlib.Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
            torch.save({"model_state":model.state_dict(),"vocab":chars,"config":{"embedding":96,"hidden":256,"char_level":True},"best_valid_loss":best,"records":len(rows)},out)
    print(f"Saved best checkpoint: {args.out} (validation loss {best:.4f})")
    print("Note: checkpoint is NOT a replacement for Android nlm2.bin; conversion/integration needs the mobile model implementation.")

if __name__=="__main__": main()
