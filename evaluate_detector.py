"""
V1-evaluation.

Evaluates a trained multi-target ConvLSTM on a fresh deterministic
synthetic validation set.

Metrics:
- Slot-level presence TP/FP/TN/FN after confidence thresholding
- Object-level matched precision / recall / F1
- Start localization error
- Per-frame trajectory error
- Velocity error
- Recall broken down by number of true targets
- Exact-count accuracy
- Duplicate/collapse rate

Important:
The model's output slots are NOT assumed to correspond to target indices.
Predictions are matched to ground truth by minimum trajectory cost.
"""

import itertools
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import *
from env_gen import MultiTargetTrackletDataset
from model import V5MultiTargetDetector



#Settings
CHECKPOINT = BASE_DIR / "v5m_overnight.pth"

CONFIDENCE_THRESHOLD = 0.50
MATCH_DISTANCE_PX = 30.0

EVAL_SEQUENCES = VAL_SEQUENCES


def pair_cost(pred_start, pred_velocity, gt_start, gt_velocity):
    """
    Normalized trajectory similarity cost.
    """
    start = torch.linalg.vector_norm(
        pred_start - gt_start
    ).item() / IMAGE_SIZE

    velocity = torch.linalg.vector_norm(
        pred_velocity - gt_velocity
    ).item() / IMAGE_SIZE

    return 2.0 * start + velocity


def match_scene(pred_start, pred_velocity, pred_score,
                gt_coords, gt_velocity, gt_mask):
    pred_indices = [
        i for i in range(len(pred_score))
        if pred_score[i] >= CONFIDENCE_THRESHOLD
    ]

    gt_indices = [
        i for i in range(len(gt_mask))
        if gt_mask[i] > 0.5
    ]

    if not pred_indices or not gt_indices:
        return [], pred_indices, gt_indices

    best_pairs = []
    best_cost = float("inf")

    n = min(len(pred_indices), len(gt_indices))

    for chosen_preds in itertools.permutations(pred_indices, n):
        # If more GT than predictions, this evaluates all predictions.
        for chosen_gts in itertools.permutations(gt_indices, n):
            pairs = list(zip(chosen_preds, chosen_gts))

            cost = sum(
                pair_cost(
                    pred_start[p],
                    pred_velocity[p],
                    gt_coords[g, 0],
                    gt_velocity[g],
                )
                for p, g in pairs
            )

            if cost < best_cost:
                best_cost = cost
                best_pairs = pairs

    return best_pairs, pred_indices, gt_indices


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

if not CHECKPOINT.exists():
    raise FileNotFoundError(
        f"Checkpoint not found:\n{CHECKPOINT}\n"
        f"Train first or change CHECKPOINT at the top of this file."
    )

dataset = MultiTargetTrackletDataset(
    EVAL_SEQUENCES,
    seed=SEED + 100000,
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda"),
)

model = V5MultiTargetDetector().to(device)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device,
    weights_only=False,
)

model.load_state_dict(
    checkpoint["model"]
)

model.eval()


#Calculations
tp = fp = fn = 0

start_errors = []
velocity_errors = []
trajectory_errors = []

by_count = {
    n: {
        "scenes": 0,
        "gt": 0,
        "matched": 0,
        "fp": 0,
        "fn": 0,
        "start": [],
        "trajectory": [],
        "velocity": [],
    }
    for n in range(MAX_TARGETS + 1)
}

exact_count_correct = 0
total_scenes = 0
matched_within_threshold = 0

with torch.no_grad():
    for batch in tqdm(
        loader,
        desc="Evaluating V5-M",
    ):
        frames = batch["frames"].to(
            device,
            non_blocking=True,
        )

        output = model(
            frames,
            use_hard_start=False,
        )

        hard_output = model(
            frames,
            use_hard_start=True,
        )

        pred_start = output["start"]
        pred_velocity = output["velocity"]
        pred_coords = output["coords"]

        pred_score = torch.sigmoid(
            output["presence_logits"]
        )

        hard_coords = hard_output["coords"]

        batch_size = frames.shape[0]

        for b in range(batch_size):
            gt_coords = batch["coords"][b].to(device)
            gt_velocity = batch["velocities"][b].to(device)
            gt_mask = batch["target_mask"][b].to(device)

            gt_count = int(
                gt_mask.sum().item()
            )

            scene = by_count[gt_count]
            scene["scenes"] += 1
            scene["gt"] += gt_count

            pairs, pred_indices, gt_indices = match_scene(
                pred_start[b],
                pred_velocity[b],
                pred_score[b],
                gt_coords,
                gt_velocity,
                gt_mask,
            )

            matched = 0

            for p, g in pairs:
                err = torch.linalg.vector_norm(
                    pred_coords[b, p] - gt_coords[g],
                    dim=-1,
                )

                hard_err = torch.linalg.vector_norm(
                    hard_coords[b, p] - gt_coords[g],
                    dim=-1,
                )

                start_err = torch.linalg.vector_norm(
                    pred_start[b, p] - gt_coords[g, 0]
                ).item()

                velocity_err = torch.linalg.vector_norm(
                    pred_velocity[b, p] - gt_velocity[g]
                ).item()

                mean_err = err.mean().item()

                if mean_err > MATCH_DISTANCE_PX:
                    continue

                matched += 1

                trajectory_errors.extend(
                    err.cpu().numpy().tolist()
                )

                start_errors.append(
                    start_err
                )

                velocity_errors.append(
                    velocity_err
                )

                scene["matched"] += 1
                scene["start"].append(
                    start_err
                )
                scene["trajectory"].extend(
                    err.cpu().numpy().tolist()
                )
                scene["velocity"].append(
                    velocity_err
                )

                matched_within_threshold += 1

            scene["fp"] += max(
                0,
                len(pred_indices) - matched,
            )

            scene["fn"] += max(
                0,
                gt_count - matched,
            )

            tp += matched
            fp += max(
                0,
                len(pred_indices) - matched,
            )
            fn += max(
                0,
                gt_count - matched,
            )

            if len(pred_indices) == gt_count:
                exact_count_correct += 1

            total_scenes += 1



#Final report printed here!
precision = tp / max(tp + fp, 1)
recall = tp / max(tp + fn, 1)

f1 = (
    2 * precision * recall
    / max(precision + recall, 1e-9)
)

exact_count_accuracy = (
    exact_count_correct
    / max(total_scenes, 1)
)

trajectory_errors = np.asarray(
    trajectory_errors,
    dtype=np.float32,
)

start_errors = np.asarray(
    start_errors,
    dtype=np.float32,
)

velocity_errors = np.asarray(
    velocity_errors,
    dtype=np.float32,
)


print()
print("=" * 72)
print("V1- MULTI-TARGET CONVLSTM REPORT")
print("=" * 72)

print(
    f"Checkpoint       : {CHECKPOINT.name}"
)
print(
    f"Evaluation scenes: {total_scenes}"
)
print(
    f"Confidence       : {CONFIDENCE_THRESHOLD:.2f}"
)
print(
    f"Match threshold  : {MATCH_DISTANCE_PX:.1f}px"
)

print()
print(
    f"TP={tp} FP={fp} FN={fn}"
)
print(
    f"Precision={precision:.4f} "
    f"Recall={recall:.4f} "
    f"F1={f1:.4f}"
)

print(
    f"Exact target-count accuracy="
    f"{exact_count_accuracy:.4f}"
)

if len(start_errors):
    print()
    print(
        f"Start mean={start_errors.mean():.2f}px | "
        f"median={np.median(start_errors):.2f}px"
    )

if len(velocity_errors):
    print(
        f"Velocity mean={velocity_errors.mean():.2f}px/frame | "
        f"median={np.median(velocity_errors):.2f}px/frame"
    )

if len(trajectory_errors):
    print(
        f"Trajectory mean={trajectory_errors.mean():.2f}px | "
        f"median={np.median(trajectory_errors):.2f}px"
    )

    print(
        f"Within 5px="
        f"{100*np.mean(trajectory_errors <= 5):.1f}% | "
        f"10px="
        f"{100*np.mean(trajectory_errors <= 10):.1f}% | "
        f"30px="
        f"{100*np.mean(trajectory_errors <= 30):.1f}%"
    )

print()
print("Performance by number of true moving objects")
print("-" * 72)

for n in range(MAX_TARGETS + 1):
    s = by_count[n]

    if s["scenes"] == 0:
        continue

    p = s["matched"] / max(
        s["matched"] + s["fp"],
        1,
    )

    r = s["matched"] / max(
        s["gt"],
        1,
    )

    print(
        f"{n} target(s): "
        f"scenes={s['scenes']} "
        f"GT={s['gt']} "
        f"matched={s['matched']} "
        f"FP={s['fp']} "
        f"FN={s['fn']} "
        f"P={p:.3f} "
        f"R={r:.3f}"
    )

    if s["start"]:
        print(
            f"    start={np.mean(s['start']):.2f}px "
            f"median={np.median(s['start']):.2f}px"
        )

    if s["velocity"]:
        print(
            f"    velocity="
            f"{np.mean(s['velocity']):.2f}px/frame"
        )

    if s["trajectory"]:
        print(
            f"    trajectory="
            f"{np.mean(s['trajectory']):.2f}px "
            f"median="
            f"{np.median(s['trajectory']):.2f}px"
        )

print("=" * 72)
