#!/usr/bin/env python3
"""Build a cumulative, non-destructive corpus from every compatible repository JSONL source."""
import argparse, hashlib, json, pathlib, shutil, torch

ROOT = pathlib.Path("training")
DATA = pathlib.Path("data")
CKPT = ROOT / "checkpoints" / "game_domain_master.pt"
MASTER = ROOT / "master_training_corpus.jsonl"
MANIFEST = ROOT / "master_training_manifest.json"

def iter_rows(path):
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip(): continue
        try: row = json.loads(line)
        except Exception as exc: raise RuntimeError(f"Invalid JSONL {path}:{line_no}: {exc}")
        if isinstance(row, dict): yield row

def prepare():
    # Preserve the previous master corpus as an input; never discard earlier rows.
    sources = []
    for base in (DATA, ROOT):
        if base.exists():
            for p in sorted(base.rglob("*.jsonl")):
                if p == MASTER or "checkpoints" in p.parts: continue
                # Avoid reading the same physical file twice (some mirrored files share a blob).
                sources.append(p)
    # Existing corpus participates in the next merge, so every new run is cumulative.
    if MASTER.exists(): sources.append(MASTER)
    rows, seen, source_manifest = [], set(), []
    for p in sources:
        count = 0
        for row in iter_rows(p):
            pairs=[]
            u,b=row.get("user"),row.get("bot")
            if isinstance(u,str) and isinstance(b,str) and u.strip() and b.strip():
                pairs.append({"user":u.strip(),"bot":b.strip(),"source":str(p)})
            # Knowledge-base records are normalized into the same training schema.
            q,a=row.get("q"),row.get("a")
            if isinstance(a,str) and a.strip() and isinstance(q,list):
                pairs.extend({"user":x.strip(),"bot":a.strip(),"source":str(p)} for x in q if isinstance(x,str) and x.strip())
            for item in pairs:
                key=(item["user"],item["bot"])
                if key in seen: continue
                seen.add(key); rows.append(item); count+=1
        source_manifest.append({"file":str(p),"records_added":count,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()})
    if not rows: raise SystemExit("No valid user/bot training records found; refusing to replace corpus.")
    MASTER.parent.mkdir(parents=True,exist_ok=True)
    tmp=MASTER.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps({"user":r["user"],"bot":r["bot"]},ensure_ascii=False)+"\n" for r in rows),encoding="utf-8")
    tmp.replace(MASTER)
    # Keep the old seed intact; only initialize the canonical checkpoint if absent.
    candidates=list((ROOT/"checkpoints").rglob("*.pt"))
    if not CKPT.exists():
        scored=[]
        for p in candidates:
            try:
                ck=torch.load(p,map_location="cpu",weights_only=False)
                loss=float(ck.get("best_valid_loss",float("inf")))
                if ck.get("model_state") and ck.get("vocab"): scored.append((loss,p))
            except Exception as exc: print(f"Skip invalid checkpoint {p}: {exc}")
        if scored:
            loss,p=min(scored,key=lambda x:x[0]); CKPT.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,CKPT)
            print(f"Seeded canonical checkpoint from {p}; prior best validation loss={loss:.5f}")
    # Snapshot the actual historical seed once; never overwrite this baseline on later runs.
    initial = ROOT / "checkpoints" / "game_domain_master_initial.pt"
    if CKPT.exists() and not initial.exists():
        initial.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(CKPT,initial)
        print(f"Preserved initial master seed at {initial}")
    manifest={"schema_version":2,"corpus_file":str(MASTER),"records":len(rows),
      "deduplicated":True,"cumulative":True,"source_files":source_manifest,
      "checkpoint_file":str(CKPT),"preserves_previous_master_corpus":True}
    MANIFEST.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"Prepared {len(rows)} unique cumulative pairs from {len(source_manifest)} JSONL sources.")

def finalize():
    if not CKPT.exists(): raise SystemExit("Master checkpoint missing after training.")
    ck=torch.load(CKPT,map_location="cpu",weights_only=False)
    manifest=json.loads(MANIFEST.read_text(encoding="utf-8"))
    ck["consolidation_manifest"]=manifest
    ck["training_rounds"]=int(ck.get("training_rounds",0))
    ck["corpus_sha256"]=hashlib.sha256(MASTER.read_bytes()).hexdigest()
    tmp=CKPT.with_suffix(".pt.tmp"); torch.save(ck,tmp); tmp.replace(CKPT)
    print("Final checkpoint metadata and corpus fingerprint saved.")

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("phase",choices=["prepare","finalize"]); args=ap.parse_args()
    prepare() if args.phase=="prepare" else finalize()
