#!/usr/bin/env python3
"""Cumulative master GRU trainer with durable per-epoch checkpoints and optimizer-state resume."""
import argparse, json, os, pathlib, random, shutil, subprocess, tempfile
import torch
from torch import nn

class CharGRU(nn.Module):
    def __init__(self, n, emb=96, hidden=256):
        super().__init__()
        self.emb=nn.Embedding(n,emb)
        self.gru=nn.GRU(emb,hidden,batch_first=True)
        self.head=nn.Linear(hidden,n)
    def forward(self,x):
        z,_=self.gru(self.emb(x)); return self.head(z)

def atomic_save(payload, path):
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name+".",suffix=".tmp",dir=str(path.parent)); os.close(fd)
    try:
        torch.save(payload,tmp); os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def persist_git(paths, epoch):
    subprocess.run(["git","config","user.name","github-actions[bot]"],check=True)
    subprocess.run(["git","config","user.email","41898282+github-actions[bot]@users.noreply.github.com"],check=True)
    subprocess.run(["git","add",*paths],check=True)
    changed=subprocess.run(["git","diff","--cached","--quiet"]).returncode != 0
    if changed:
        subprocess.run(["git","commit","-m",f"[training-checkpoint] durable epoch {epoch}"],check=True)
        subprocess.run(["git","push"],check=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--epochs",type=int,default=40,help="additional epochs to train")
    ap.add_argument("--seq-len",type=int,default=160)
    ap.add_argument("--batch-size",type=int,default=32)
    ap.add_argument("--checkpoint",default="training/checkpoints/game_domain_master.pt")
    ap.add_argument("--persist-each-epoch",action="store_true")
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
    prior_records=0; rounds=0; completed=0; sources=[]; best=float("inf"); best_state=None
    opt=torch.optim.AdamW(model.parameters(),lr=2e-4)
    if checkpoint.exists():
        ck=torch.load(checkpoint,map_location="cpu",weights_only=False)
        oldv=ck.get("vocab",[]); old=ck.get("model_state",ck.get("latest_model_state",{}))
        with torch.no_grad():
            for oi,c in enumerate(oldv):
                if c not in stoi: continue
                ni=stoi[c]
                if "emb.weight" in old: model.emb.weight[ni].copy_(old["emb.weight"][oi])
                if "head.weight" in old: model.head.weight[ni].copy_(old["head.weight"][oi])
                if "head.bias" in old: model.head.bias[ni].copy_(old["head.bias"][oi])
            for key in ("gru.weight_ih_l0","gru.weight_hh_l0","gru.bias_ih_l0","gru.bias_hh_l0"):
                if key in old and old[key].shape == model.state_dict()[key].shape:
                    model.state_dict()[key].copy_(old[key])
        # Adam moments are only valid when parameter shapes and vocabulary are identical.
        if ck.get("optimizer_state") and oldv == vocab:
            try: opt.load_state_dict(ck["optimizer_state"])
            except (ValueError,KeyError,RuntimeError): print("Optimizer state incompatible; continuing with fresh optimizer.")
        elif ck.get("optimizer_state"):
            print("Vocabulary changed; mapped model weights retained and optimizer moments reset safely.")
        prior_records=int(ck.get("records",0)); rounds=int(ck.get("training_rounds",0))
        completed=int(ck.get("completed_epochs",0)); best=float(ck.get("best_valid_loss",float("inf")))
        # Rebase the best snapshot to the current vocabulary; never keep stale-shaped tensors.
        best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        sources=ck.get("checkpoint_sources",[])+ck.get("consolidation_manifest",{}).get("source_files",[])
        print(f"Resumed durable checkpoint: epoch={completed}, records={prior_records}, rounds={rounds}")
    elif not checkpoint.exists():
        initial=pathlib.Path("training/checkpoints/game_domain_master_initial.pt")
        if not initial.exists():
            atomic_save({"model_state":model.state_dict(),"vocab":vocab,"note":"initial weights before cumulative master training"},initial)
    def batches(src,steps=80):
        n=max(1,len(src)-seq-1)
        for _ in range(steps):
            ix=torch.randint(0,n,(args.batch_size,))
            x=torch.stack([src[i:i+seq] for i in ix])
            y=torch.stack([src[i+1:i+seq+1] for i in ix])
            yield x,y
    checkpoint_paths=[str(checkpoint),"training/checkpoints/game_domain_master_candidate.pt"]
    for _ in range(args.epochs):
        model.train(); ls=[]
        for x,y in batches(train):
            opt.zero_grad(); out=model(x); loss=nn.CrossEntropyLoss()(out.reshape(-1,len(vocab)),y.reshape(-1))
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); ls.append(loss.item())
        model.eval()
        with torch.no_grad():
            x=valid[:-1].unsqueeze(0); y=valid[1:].unsqueeze(0)
            val=nn.CrossEntropyLoss()(model(x).reshape(-1,len(vocab)),y.reshape(-1)).item()
        completed+=1
        if val<best:
            best=val; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        payload={"model_state":model.state_dict(),"latest_model_state":model.state_dict(),
          "best_model_state":best_state,"optimizer_state":opt.state_dict(),"vocab":vocab,
          "config":{"embedding":96,"hidden":256,"char_level":True},"best_valid_loss":best,
          "last_valid_loss":val,"completed_epochs":completed,"records":prior_records+len(records),
          "training_rounds":rounds+1,"corpus_file":path.name,"corpus_records":len(records),
          "checkpoint_sources":sources}
        atomic_save(payload,checkpoint)
        candidate=dict(payload); candidate["model_state"]=best_state or model.state_dict()
        candidate["optimizer_state"]=None; candidate["note"]="best validation model snapshot; not the Android nlm2.bin format"
        atomic_save(candidate,pathlib.Path("training/checkpoints/game_domain_master_candidate.pt"))
        print(f"epoch {completed:03d} train_loss={sum(ls)/len(ls):.4f} valid_loss={val:.4f} best={best:.4f}",flush=True)
        if args.persist_each_epoch:
            persist_git(checkpoint_paths,completed)
    print(f"Durable checkpoint: {checkpoint}; candidate: training/checkpoints/game_domain_master_candidate.pt; completed_epochs={completed}; best_valid_loss={best:.4f}")
if __name__=="__main__": main()
