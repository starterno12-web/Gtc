#!/usr/bin/env python3
"""Prepare a non-destructive master dataset and seed cumulative GRU from the best prior checkpoint."""
import argparse, json, pathlib, shutil, torch

ROOT = pathlib.Path("training")
CKPT = ROOT / "checkpoints" / "game_domain_master.pt"
MASTER = ROOT / "master_training_corpus.jsonl"
SOURCES = sorted(p for p in ROOT.glob("curriculum_*.jsonl") if p.name != MASTER.name)

def prepare():
    rows, seen = [], set()
    for p in SOURCES:
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            row = json.loads(line)
            key = (row.get("user", ""), row.get("bot", ""))
            if not all(key) or key in seen: continue
            seen.add(key); rows.append(row)
    MASTER.write_text("".join(json.dumps(r, ensure_ascii=False) + "\\n" for r in rows), encoding="utf-8")
    candidates = list(pathlib.Path("training/checkpoints").glob("*/game_domain_gru.pt")) + list(pathlib.Path("training/checkpoints").glob("*/game_domain_master.pt"))
    # Bootstrap folders may contain a direct file; also scan all checkpoints recursively.
    candidates += list(pathlib.Path("training/checkpoints").rglob("*.pt"))
    unique = []
    for p in candidates:
        if p.exists() and p.resolve() != CKPT.resolve() and p not in unique: unique.append(p)
    if CKPT.exists():
        print("Existing master checkpoint preserved as seed.")
    elif unique:
        scored=[]
        for p in unique:
            try:
                ck=torch.load(p,map_location="cpu",weights_only=False)
                loss=float(ck.get("best_valid_loss", float("inf")))
                if ck.get("model_state") and ck.get("vocab"): scored.append((loss,p,ck))
            except Exception as e: print(f"Skip invalid checkpoint {p}: {e}")
        if scored:
            loss,p,ck=min(scored,key=lambda x:x[0])
            CKPT.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(p,CKPT)
            print(f"Seeded master from {p} (best prior validation loss={loss:.4f})")
        else: print("No valid prior checkpoint found; trainer will initialize from scratch.")
    else: print("No previous checkpoint candidates; trainer will initialize from scratch.")
    manifest={"corpus_file":str(MASTER),"records":len(rows),"source_files":[p.name for p in SOURCES],"deduplicated":True}
    pathlib.Path("training/master_training_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"Master dataset prepared: {len(rows)} unique records from {len(SOURCES)} curricula")

def finalize():
    if not CKPT.exists(): raise SystemExit("Master checkpoint missing after training.")
    ck=torch.load(CKPT,map_location="cpu",weights_only=False)
    manifest=json.loads(pathlib.Path("training/master_training_manifest.json").read_text(encoding="utf-8"))
    ck["consolidation_manifest"]=manifest
    ck["training_rounds"]=int(ck.get("training_rounds",0))
    torch.save(ck,CKPT)
    print("Master checkpoint metadata finalized.")

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("phase",choices=["prepare","finalize"]); args=ap.parse_args()
    prepare() if args.phase=="prepare" else finalize()
