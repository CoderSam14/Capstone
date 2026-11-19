"""Track-level held-out evaluation for BiConvLSTM variable-cardinality outputs.

Reports single-injection, two-injection, and micro-aggregated combined scores.
Candidate count is peak-decoded from the sequence heatmap; there is no top-K cap.
"""
import argparse
import csv
import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from config import TRACKLETS_ROOT, CHECKPOINT_DIR, OUTPUT_DIR, get_config, seed_everything
from synthetic_dataset import SyntheticTrackletDataset
from train_detector import split_usable_items, collate_one
from model import VariableTrackletConvLSTM


def greedy_associate(edges):
    # edges: (-coverage, median error, pred index, GT index)
    used_p, used_g, matches = set(), set(), []
    for negcov, mederr, p, g in sorted(edges):
        if p in used_p or g in used_g:
            continue
        used_p.add(p); used_g.add(g)
        matches.append((p,g,-negcov,mederr))
    return matches, used_p, used_g


def score_scenario(model, base_ds, test_indices, scenario, injection_count, samples_per_tracklet,
                   negatives_per_tracklet, seed, device, threshold, match_px, min_coverage, nms_px):
    ds=base_ds.subset(test_indices,False)
    ds.cfg["injection_count_range"]=(injection_count,injection_count)
    ds.cfg["validation_per_tracklet"]=samples_per_tracklet
    ds.cfg["validation_negatives_per_tracklet"]=negatives_per_tracklet
    ds.synthetic_per_tracklet=int(samples_per_tracklet)
    ds.negative_per_tracklet=int(negatives_per_tracklet)
    ds.seed=int(seed)+10007*injection_count
    ds._rebuild_samples()
    loader=DataLoader(ds,batch_size=1,shuffle=False,num_workers=0,collate_fn=collate_one)
    totals=dict(tp=0,fp=0,fn=0,tn=0)
    sample_correct = 0
    matched_cover=[]; matched_err=[]; vel_err=[]; pred_rms=[]; snr_outcomes=[]
    rows=[]
    with torch.inference_mode():
        for sample_i,sample in enumerate(loader):
            frames=sample["frames"].unsqueeze(0).to(device)
            out=model(frames)
            candidates=model.decode_candidates(out,confidence_threshold=threshold,nms_radius_px=nms_px)[0]
            gt=sample["coords"].numpy() # [N,T,2]
            n_gt=gt.shape[0]
            n_pred=len(candidates)
            t=gt.shape[1] if n_gt else sample["frames"].shape[0]
            ref=t//2
            edges=[]; stats={}
            for pi,cand in enumerate(candidates):
                pred_path=np.asarray(cand["trajectory"],dtype=float)
                for gi in range(n_gt):
                    gt_path=gt[gi]
                    valid=np.isfinite(gt_path).all(axis=1)&(gt_path>=0).all(axis=1)
                    if not valid.any(): continue
                    errs=np.linalg.norm(pred_path[valid]-gt_path[valid],axis=1)
                    coverage=float((errs<=match_px).sum()/valid.sum())
                    med=float(np.median(errs))
                    stats[(pi,gi)]=(coverage,med,errs,valid)
                    if coverage>=min_coverage:
                        edges.append((-coverage,med,pi,gi))
            matches,usedp,usedg=greedy_associate(edges)
            amps=sample.get("injection_amplitudes",torch.empty(0)).numpy()
            sigmas=sample.get("injection_sigmas",torch.empty(0)).numpy()
            gt_snr=[]
            for gi in range(n_gt):
                # Idealized Gaussian-PSF matched-filter SNR in robust-normalized
                # unit-noise pixels: amp * sqrt(sum(PSF**2)) ~= amp*sigma*sqrt(pi).
                # This is a proxy; reprojection/local background changes realized SNR.
                a=float(amps[gi]) if gi<len(amps) else float("nan")
                sig=float(sigmas[gi]) if gi<len(sigmas) else float("nan")
                gt_snr.append(a*sig*math.sqrt(math.pi))
                snr_outcomes.append((gt_snr[-1],gi in usedg))
            totals["tp"]+=len(matches)
            totals["fp"]+=n_pred-len(usedp)
            totals["fn"]+=n_gt-len(usedg)
            if n_gt==0 and n_pred==0: totals["tn"]+=1
            # Sample-level exact-set accuracy: all GTs matched and no extra
            # predictions. A negative sample is correct only with zero candidates.
            if len(matches) == n_gt and len(usedg) == n_gt and len(usedp) == n_pred:
                sample_correct += 1
            tracklet=Path(sample["folder"]).name
            for pi,gi,cov,med in matches:
                coverage,median,errs,valid=stats[(pi,gi)]
                candidate=candidates[pi]
                gt_vel=sample["velocities"][gi].numpy()
                pv=np.asarray([candidate["vx"],candidate["vy"]])
                verr=float(np.linalg.norm(pv-gt_vel))
                # RMS deviation of predicted candidate path from its own straight-line parameterization.
                ppath=np.asarray(candidate["trajectory"])
                linear=ppath[ref]+(np.arange(t)[:,None]-ref)*pv[None,:]
                rms=float(np.sqrt(np.mean(np.sum((ppath-linear)**2,axis=1))))
                matched_cover.append(coverage); matched_err.append(median); vel_err.append(verr); pred_rms.append(rms)
                rows.append({"scenario":scenario,"sample":sample_i,"tracklet":tracklet,"status":"TP",
                    "pred_id":pi,"gt_id":gi,"confidence":candidate["confidence"],"coverage":coverage,
                    "median_error_px":median,"velocity_error_px_frame":verr,"pred_vx":pv[0],"pred_vy":pv[1],
                    "gt_vx":gt_vel[0],"gt_vy":gt_vel[1],"pred_path_linear_rms_px":rms,
                    "approx_matched_filter_snr":gt_snr[gi]})
            for pi,c in enumerate(candidates):
                if pi not in usedp:
                    rows.append({"scenario":scenario,"sample":sample_i,"tracklet":tracklet,"status":"FP",
                        "pred_id":pi,"gt_id":"","confidence":c["confidence"],"coverage":"","median_error_px":"",
                        "velocity_error_px_frame":"","pred_vx":c["vx"],"pred_vy":c["vy"],
                        "gt_vx":"","gt_vy":"","pred_path_linear_rms_px":"","approx_matched_filter_snr":""})
            for gi in range(n_gt):
                if gi not in usedg:
                    rows.append({"scenario":scenario,"sample":sample_i,"tracklet":tracklet,"status":"FN",
                        "pred_id":"","gt_id":gi,"confidence":"","coverage":"","median_error_px":"",
                        "velocity_error_px_frame":"","pred_vx":"","pred_vy":"",
                        "gt_vx":sample["velocities"][gi,0].item(),"gt_vy":sample["velocities"][gi,1].item(),
                        "pred_path_linear_rms_px":"","approx_matched_filter_snr":gt_snr[gi]})
    p=totals["tp"]/max(totals["tp"]+totals["fp"],1)
    r=totals["tp"]/max(totals["tp"]+totals["fn"],1)
    f1=2*p*r/max(p+r,1e-12)
    sample_accuracy = sample_correct / max(len(ds), 1)
    return {"scenario":scenario,"samples":len(ds),**totals,"precision":p,"recall":r,"f1":f1,
            "sample_accuracy":sample_accuracy,"sample_correct":sample_correct,
            "coverage":matched_cover,"locerr":matched_err,"velerr":vel_err,"rms":pred_rms,"rows":rows,
            "snr_outcomes":snr_outcomes}


def report(r):
    print(f"\n[{r['scenario']}] samples={r['samples']} | track TP={r['tp']} FP={r['fp']} FN={r['fn']} TN(samples)={r['tn']}")
    print(f"precision={r['precision']:.4f} recall={r['recall']:.4f} F1={r['f1']:.4f}")
    if "sample_accuracy" in r:
        print(f"sample-level exact-set accuracy={r['sample_accuracy']:.4f} ({r.get('sample_correct', 0)}/{r['samples']})")
    if r["coverage"]:
        print(f"matched={len(r['coverage'])} | median coverage={np.median(r['coverage']):.3f} | "
              f"median localization error={np.median(r['locerr']):.2f}px | "
              f"median velocity error={np.median(r['velerr']):.3f}px/frame")
    # Recall by idealized matched-filter SNR proxy, grouped over GT injections.
    bins=[(0.0,3.0),(3.0,5.0),(5.0,8.0),(8.0,float("inf"))]
    for lo,hi in bins:
        subset=[hit for snr,hit in r.get("snr_outcomes",[]) if lo<=snr<hi]
        if subset:
            label=f"{lo:g}–{hi:g}" if np.isfinite(hi) else f"{lo:g}+"
            print(f"approx MF-SNR {label}: recall={sum(subset)/len(subset):.3f} ({sum(subset)}/{len(subset)})")



def save_evaluation_plots(results, output_dir):
    """Save graphs 1, 2, and 4: recall vs N, P/R/F1 vs N, and recall vs SNR proxy."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    counts = [int(r["scenario"].split("_")[0]) for r in results]
    recalls = [r["recall"] for r in results]
    precisions = [r["precision"] for r in results]
    f1s = [r["f1"] for r in results]

    # Graph 1: Recall vs injection count.
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(counts, recalls, marker="o", linewidth=2)
    ax.set(xlabel="Number of injected objects", ylabel="Recall",
           title="Recall vs. Injection Count", xticks=counts, ylim=(0, 1.05))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "recall_vs_injection_count.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Graph 2: Precision, recall, and F1 vs injection count.
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(counts, precisions, marker="o", linewidth=2, label="Precision")
    ax.plot(counts, recalls, marker="s", linewidth=2, label="Recall")
    ax.plot(counts, f1s, marker="^", linewidth=2, label="F1-score")
    ax.set(xlabel="Number of injected objects", ylabel="Score",
           title="Precision, Recall, and F1 vs. Injection Count",
           xticks=counts, ylim=(0, 1.05))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "precision_recall_f1_vs_injection_count.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Additional graph: sample-level exact-set accuracy vs injection count.
    sample_accuracies = [r.get("sample_accuracy", np.nan) for r in results]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(counts, sample_accuracies, marker="D", linewidth=2, label="Exact-set accuracy")
    ax.set(xlabel="Number of injected objects", ylabel="Sample-level exact-set accuracy",
           title="Sample-level Accuracy vs. Injection Count",
           xticks=counts, ylim=(0, 1.05))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "sample_accuracy_vs_injection_count.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Graph 4: pooled recall by approximate matched-filter SNR proxy.
    bins = [(0.0, 3.0), (3.0, 5.0), (5.0, 8.0), (8.0, float("inf"))]
    labels = ["0–3", "3–5", "5–8", "8+"]
    outcomes = [(float(snr), bool(hit))
                for result in results
                for snr, hit in result.get("snr_outcomes", [])
                if np.isfinite(snr)]
    recalls_by_bin, counts_by_bin = [], []
    for lo, hi in bins:
        hits = [hit for snr, hit in outcomes if lo <= snr < hi]
        counts_by_bin.append(len(hits))
        recalls_by_bin.append(sum(hits) / len(hits) if hits else np.nan)

    valid = [i for i, n in enumerate(counts_by_bin) if n > 0]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(valid, [recalls_by_bin[i] for i in valid], marker="o", linewidth=2)
    ax.set_xticks(valid, [f"{labels[i]}\n(n={counts_by_bin[i]})" for i in valid])
    ax.set(xlabel="Approximate matched-filter SNR proxy bin", ylabel="Recall",
           title="Recall vs. Approximate Injected-Source SNR", ylim=(0, 1.05))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "recall_vs_approx_snr.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved graphs to: {output_dir}")
    print("  recall_vs_injection_count.png")
    print("  precision_recall_f1_vs_injection_count.png")
    print("  recall_vs_approx_snr.png")
    print("  sample_accuracy_vs_injection_count.png")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--preset",choices=["test","full"],default="full")
    ap.add_argument("--checkpoint",default=None)
    ap.add_argument("--split",choices=["val","test"],default="test",
                    help="Use validation for threshold tuning; reserve test for final reporting")
    ap.add_argument("--confidence-threshold",type=float,default=0.85)
    ap.add_argument("--nms-radius-px",type=float,default=12.0)
    ap.add_argument("--match-px",type=float,default=10.0)
    ap.add_argument("--min-coverage",type=float,default=0.5)
    ap.add_argument("--samples-per-tracklet",type=int,default=20)
    ap.add_argument("--negatives-per-tracklet",type=int,default=20)
    ap.add_argument("--output",default=None)
    ap.add_argument("--seed",type=int,default=None)
    a=ap.parse_args()
    if not 0<=a.confidence_threshold<=1: ap.error("confidence threshold must be in [0,1]")
    if a.samples_per_tracklet<1 or a.negatives_per_tracklet<0: ap.error("invalid sample counts")
    cfg=get_config(a.preset); seed=cfg["seed"] if a.seed is None else a.seed
    seed_everything(seed); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folders=sorted(p for p in TRACKLETS_ROOT.iterdir() if p.is_dir() and (p/"metadata.csv").exists()) if TRACKLETS_ROOT.exists() else []
    if not folders: raise RuntimeError(f"No tracklet folders with metadata.csv under {TRACKLETS_ROOT}")
    base=SyntheticTrackletDataset(folders,cfg,training=False)
    _,val_indices,test_indices,split=split_usable_items(base.items,seed)
    eval_indices=val_indices if a.split=="val" else test_indices
    checkpoint=Path(a.checkpoint) if a.checkpoint else CHECKPOINT_DIR/f"biconvlstm_tracklet_{a.preset}_best.pth"
    state=torch.load(checkpoint,map_location=device,weights_only=False)
    model=VariableTrackletConvLSTM().to(device)
    if state.get("architecture")!=model.architecture: raise RuntimeError(f"Expected {model.architecture}, got {state.get('architecture')}")
    model.load_state_dict(state["model"],strict=True); model.eval()
    print(f"Checkpoint={checkpoint} epoch={state.get('epoch','?')} | group split={split} | evaluation={a.split} tracklets={len(eval_indices)}")
    print(f"confidence={a.confidence_threshold:.3f}; NMS radius={a.nms_radius_px:g}px; track match <= {a.match_px:g}px on >= {a.min_coverage:.0%} frames")
    results=[]
    for count in range(1,6):
        scenario=f"{count}_injections"
        r=score_scenario(model,base,eval_indices,scenario,count,a.samples_per_tracklet,
            a.negatives_per_tracklet,seed,device,a.confidence_threshold,a.match_px,a.min_coverage,a.nms_radius_px)
        results.append(r); report(r)
        # Per-injection-count detailed CSV
        per_out=Path(a.output).with_name(f"{Path(a.output).stem}_{count}injections.csv") if a.output else OUTPUT_DIR/f"track_eval_{a.preset}_{count}injections.csv"
        per_out.parent.mkdir(parents=True,exist_ok=True)
        fields=["scenario","sample","tracklet","status","pred_id","gt_id","confidence","coverage","median_error_px",
                "velocity_error_px_frame","pred_vx","pred_vy","gt_vx","gt_vy","pred_path_linear_rms_px","approx_matched_filter_snr"]
        with per_out.open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(r["rows"])
        print(f"Detailed {count}-injection CSV={per_out}")
    plot_dir = Path(a.output).parent / "evaluation_graphs" if a.output else OUTPUT_DIR / "evaluation_graphs"
    save_evaluation_plots(results, plot_dir)

    combined={"scenario":"combined_1_to_5","samples":sum(r["samples"] for r in results)}
    for k in ("tp","fp","fn","tn"): combined[k]=sum(r[k] for r in results)
    combined["precision"]=combined["tp"]/max(combined["tp"]+combined["fp"],1)
    combined["recall"]=combined["tp"]/max(combined["tp"]+combined["fn"],1)
    combined["f1"]=2*combined["precision"]*combined["recall"]/max(combined["precision"]+combined["recall"],1e-12)
    combined["sample_correct"]=sum(r.get("sample_correct", 0) for r in results)
    combined["sample_accuracy"]=combined["sample_correct"]/max(combined["samples"],1)
    for k in ("coverage","locerr","velerr","rms","rows","snr_outcomes"): combined[k]=[x for r in results for x in r[k]]
    report(combined)
    out=Path(a.output) if a.output else OUTPUT_DIR/f"biconvlstm_track_eval_{a.preset}_combined_1to5.csv"
    out.parent.mkdir(parents=True,exist_ok=True)
    fields=["scenario","sample","tracklet","status","pred_id","gt_id","confidence","coverage","median_error_px",
            "velocity_error_px_frame","pred_vx","pred_vy","gt_vx","gt_vy","pred_path_linear_rms_px","approx_matched_filter_snr"]
    with out.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(combined["rows"])
    print(f"CSV={out}")
    # Compact scenario-level summary, including each cardinality and micro-combined result.
    summary=out.with_name(out.stem+"_summary.csv")
    with summary.open("w",newline="",encoding="utf-8") as f:
        cols=["scenario","samples","tp","fp","fn","tn","precision","recall","f1","sample_correct","sample_accuracy"]
        w=csv.DictWriter(f,fieldnames=cols); w.writeheader()
        for item in results+[combined]: w.writerow({k:item.get(k) for k in cols})
    print(f"Scenario summary CSV={summary}")
    print("Combined is micro-aggregated across fixed injection counts 1 through 5.")
    print("Caveat: synthetic injection evaluation on held-out FITS backgrounds; crops from a tracklet are correlated.")
    print("Caveat: velocity is px/frame (frame-index time), not physical sky motion or irregular cadence.")
if __name__=="__main__": main()
