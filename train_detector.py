"""Train the bidirectional ConvLSTM variable-cardinality tracklet detector."""
import argparse
import csv
import matplotlib.pyplot as plt
import random
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import TRACKLETS_ROOT, CHECKPOINT_DIR,OUTPUT_DIR, get_config, seed_everything
from synthetic_dataset import SyntheticTrackletDataset, group_key
from model import VariableTrackletConvLSTM


def collate_one(items):
    # Sequence lengths vary; this project intentionally trains batch-size one.
    return items[0]


def split_usable_items(items, seed=42):
    groups = {}
    for i, item in enumerate(items):
        try:
            key = group_key(item["folder"])
        except Exception:
            key = str(item["folder"])
        groups.setdefault(key, []).append(i)
    rng = random.Random(seed)
    keys = sorted(groups)
    rng.shuffle(keys)
    if len(keys) >= 3:
        ntr = max(1, int(len(keys) * .70))
        nv = max(1, int(len(keys) * .15))
        if ntr + nv >= len(keys):
            ntr, nv = len(keys)-2, 1
        tr, va, te = keys[:ntr], keys[ntr:ntr+nv], keys[ntr+nv:]
        return ([i for k in tr for i in groups[k]],
                [i for k in va for i in groups[k]],
                [i for k in te for i in groups[k]], "field/night groups")
    order = list(range(len(items)))
    rng.shuffle(order)
    if len(order) < 3:
        raise RuntimeError(f"Need at least 3 usable tracklets/groups, found {len(order)}")
    ntr = max(1, int(len(order)*.70))
    nv = max(1, int(len(order)*.15))
    if ntr+nv >= len(order):
        ntr, nv = len(order)-2, 1
    return order[:ntr], order[ntr:ntr+nv], order[ntr+nv:], "individual tracklets (weaker independence)"


def make_loader(ds, training, workers=0, prefetch_factor=2):
    kwargs = dict(batch_size=1, shuffle=training, num_workers=workers,
                  collate_fn=collate_one, pin_memory=torch.cuda.is_available())
    if workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=prefetch_factor)
    return DataLoader(ds, **kwargs)


from losses import compute_loss


def run_epoch(model, loader, optimizer, scaler, cfg, device, training):
    model.train(training)
    sums={"total":0.0,"center":0.0,"velocity":0.0,"offset":0.0,"objects":0.0}
    steps=0
    for sample in tqdm(loader, desc="train" if training else "val", leave=False):
        frames=sample["frames"].unsqueeze(0).to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        grad_context = nullcontext() if training else torch.no_grad()
        with grad_context:
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=bool(cfg.get("amp",True) and device.type=="cuda")):
                output=model(frames)
                loss,stats=compute_loss(output,sample)
        if training:
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
                optimizer.step()
        for k in sums: sums[k]+=stats[k]
        steps+=1
    return {k:v/max(steps,1) for k,v in sums.items()}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--preset",choices=["test","full"],default="full")
    ap.add_argument("--resume",action="store_true")
    ap.add_argument("--workers",type=int,default=None)
    args=ap.parse_args()
    cfg=get_config(args.preset); seed_everything(cfg["seed"])
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device={device}"+(f" | GPU={torch.cuda.get_device_name(0)}" if device.type=="cuda" else ""))
    folders=sorted(p for p in TRACKLETS_ROOT.iterdir() if p.is_dir() and (p/"metadata.csv").exists()) if TRACKLETS_ROOT.exists() else []
    if not folders: raise RuntimeError(f"No tracklets with metadata.csv under {TRACKLETS_ROOT}")
    print("BIDIRECTIONAL ConvLSTM | one sequence-level center heatmap | variable peak-decoded track count")
    all_ds=SyntheticTrackletDataset(folders,cfg,training=True)
    tr,va,te,split=split_usable_items(all_ds.items,cfg["seed"])
    print(f"Split={split}; usable tracklets={len(all_ds.items)}; train={len(tr)} val={len(va)} test={len(te)}")
    train_ds=all_ds.subset(tr,True); val_ds=all_ds.subset(va,False)
    workers=int(cfg.get("num_workers",0) if args.workers is None else max(0,args.workers))
    train_loader=make_loader(train_ds,True,workers,int(cfg.get("prefetch_factor",2)))
    val_loader=make_loader(val_ds,False,workers,int(cfg.get("prefetch_factor",2)))
    model=VariableTrackletConvLSTM().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg["lr"],weight_decay=cfg["weight_decay"])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg["epochs"],eta_min=1e-6)
    scaler=torch.amp.GradScaler("cuda",enabled=bool(cfg.get("amp",True) and device.type=="cuda"))
    latest=CHECKPOINT_DIR/f"biconvlstm_tracklet_{args.preset}_latest.pth"
    best=CHECKPOINT_DIR/f"biconvlstm_tracklet_{args.preset}_best.pth"
    start=1; best_val=float("inf")
    if args.resume:
        if not latest.exists(): raise FileNotFoundError(f"No checkpoint to resume: {latest}")
        state=torch.load(latest,map_location=device,weights_only=False)
        if state.get("architecture")!=model.architecture:
            raise RuntimeError("Checkpoint architecture mismatch; do not resume from the residual-CNN checkpoint.")
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        if "scheduler" in state: scheduler.load_state_dict(state["scheduler"])
        if "scaler" in state: scaler.load_state_dict(state["scaler"])
        start=int(state["epoch"])+1; best_val=float(state.get("best_val",best_val))
    history=[]
    # Recall curves are evaluated on the same held-out validation tracklets, with a
    # fixed synthetic cardinality per curve (1, 2, 3, 4, or 5 injections/sample).
    recall_indices=va
    recall_samples=max(1,int(cfg.get("validation_per_tracklet",1)))
    recall_negatives=max(0,int(cfg.get("validation_negatives_per_tracklet",1)))
    for epoch in range(start,int(cfg["epochs"])+1):
        train_stats=run_epoch(model,train_loader,optimizer,scaler,cfg,device,True)
        val_stats=run_epoch(model,val_loader,optimizer,scaler,cfg,device,False)
        scheduler.step()
        from evaluate_sequences import score_scenario
        model.eval()
        epoch_recalls={}
        for n_inj in range(1,6):
            ev=score_scenario(model,all_ds,recall_indices,f"epoch{epoch}_n{n_inj}",n_inj,
                recall_samples,recall_negatives,cfg["seed"]+epoch*1009,device,0.5,10.0,0.5,12.0)
            epoch_recalls[n_inj]=float(ev["recall"])
        history.append({"epoch":epoch,"train_loss":train_stats["total"],"val_loss":val_stats["total"],
                        **{f"recall_{k}_injection":v for k,v in epoch_recalls.items()}})
        print(f"Epoch {epoch:03d}/{cfg['epochs']} | train={train_stats['total']:.4f} center={train_stats['center']:.4f} vel={train_stats['velocity']:.4f} off={train_stats['offset']:.4f} objs={train_stats['objects']:.2f} | val={val_stats['total']:.4f} center={val_stats['center']:.4f} vel={val_stats['velocity']:.4f} off={val_stats['offset']:.4f} objs={val_stats['objects']:.2f}")
        state={"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),
               "scaler":scaler.state_dict(),"epoch":epoch,"best_val":min(best_val,val_stats["total"]),
               "preset":args.preset,"architecture":model.architecture,"velocity_scale":model.velocity_scale,
               "input_size":1024,"output_stride":4}
        torch.save(state,latest)
        if val_stats["total"]<best_val:
            best_val=val_stats["total"]; state["best_val"]=best_val; torch.save(state,best)
            print(f"[BEST] {best} (validation loss={best_val:.5f})")
    # Persist numeric history and produce requested plots.
    hist_path=OUTPUT_DIR/f"training_history_{args.preset}.csv"
    hist_path.parent.mkdir(parents=True,exist_ok=True)
    columns=["epoch","train_loss","val_loss"]+[f"recall_{k}_injection" for k in range(1,6)]
    with hist_path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=columns); w.writeheader(); w.writerows(history)
    fig,ax=plt.subplots(figsize=(8,5)); ax.plot([h["epoch"] for h in history],[h["train_loss"] for h in history],label="Training loss"); ax.plot([h["epoch"] for h in history],[h["val_loss"] for h in history],label="Validation loss"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Training and validation loss"); ax.grid(True,alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig(OUTPUT_DIR/f"loss_vs_epoch_{args.preset}.png",dpi=200); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    for k in range(1,6): ax.plot([h["epoch"] for h in history],[100*h[f"recall_{k}_injection"] for h in history],marker=".",label=f"{k} injection(s)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation recall (%)"); ax.set_ylim(0,100); ax.set_title("Validation recall by injection count"); ax.grid(True,alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig(OUTPUT_DIR/f"recall_vs_epoch_1to5_{args.preset}.png",dpi=200); plt.close(fig)
    print(f"[DONE] history={hist_path} | plots saved under {OUTPUT_DIR}")
    print(f"[DONE] latest={latest}\n[DONE] best={best}")


if __name__=="__main__":
    main()
