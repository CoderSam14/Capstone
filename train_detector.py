"""
V1-MULTI quick/overnight training.

Change ONLY the SETTINGS block below to switch runs.


"""

import itertools
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import *
from env_gen import MultiTargetTrackletDataset
from model import V5MultiTargetDetector


#Cahnge epoch and sequences according to your hardware

RUN_NAME = "overnight"

PRESETS = {
    "quick": {
        "train_sequences": 2000,
        "epochs": 10,
        "batch_size": 8,
    },
    "overnight": {
        "train_sequences": 50000,
        "epochs": 10,
        "batch_size": 8,
    },
    "research": {
        "train_sequences": 100000,
        "epochs": 60,
        "batch_size": 8,
    },
}

CFG = PRESETS[RUN_NAME]

TRAIN_N = CFG["train_sequences"]
EPOCHS_RUN = CFG["epochs"]
BATCH_RUN = CFG["batch_size"]

VAL_N = VAL_SEQUENCES
LR = LEARNING_RATE

# Loss weights.
W_HEATMAP = 2.0
W_START = 3.0
W_VELOCITY = 2.0
W_PRESENCE = 1.0
W_DIVERSITY = 0.15

MODEL_PATH = BASE_DIR / f"v5m_{RUN_NAME}.pth"



random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)




def assignment_for_sample(pred_start, pred_vel, pred_presence,
                          gt_start, gt_vel, gt_mask):
    
    k = pred_start.shape[0]
    gt_indices = [
        i for i in range(gt_mask.shape[0])
        if gt_mask[i] > 0.5
    ]

    n = len(gt_indices)

    if n == 0:
        return []

    best = None
    best_cost = None

    for pred_slots in itertools.permutations(range(k), n):
        cost = 0.0

        for pred_slot, gt_slot in zip(
            pred_slots,
            gt_indices,
        ):
            position_cost = torch.mean(torch.abs(
                pred_start[pred_slot]
                - gt_start[gt_slot]
            )) / IMAGE_SIZE

            velocity_cost = torch.mean(torch.abs(
                pred_vel[pred_slot]
                - gt_vel[gt_slot]
            )) / IMAGE_SIZE

            presence_cost = -torch.log(
                torch.sigmoid(
                    pred_presence[pred_slot]
                ) + 1e-7
            )

            cost += (
                3.0 * position_cost.item()
                + 1.5 * velocity_cost.item()
                + 0.5 * presence_cost.item()
            )

        if best_cost is None or cost < best_cost:
            best_cost = cost
            best = list(zip(pred_slots, gt_indices))

    return best


def gaussian_heatmap(h, w, x, y, sigma=1.5, device=None):
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij",
    )

    sx = x * (w - 1) / (IMAGE_SIZE - 1)
    sy = y * (h - 1) / (IMAGE_SIZE - 1)

    g = torch.exp(
        -((xx-sx)**2 + (yy-sy)**2)
        / (2*sigma*sigma)
    )

    return g / (g.sum() + 1e-8)


def diversity_loss(logits):
    
    b, k, h, w = logits.shape

    p = F.softmax(
        logits.flatten(2),
        dim=-1,
    )

    loss = torch.zeros(
        (),
        device=logits.device,
    )

    pairs = 0

    for i in range(k):
        for j in range(i + 1, k):
            overlap = (p[:, i] * p[:, j]).sum(dim=-1)
            loss = loss + overlap.mean()
            pairs += 1

    if pairs:
        loss = loss / pairs

    return loss


def compute_loss(output, batch):
    
    b, k, h, w = output["start_logits"].shape

    total_heatmap = torch.zeros((), device=output["start_logits"].device)
    total_start = torch.zeros((), device=output["start_logits"].device)
    total_velocity = torch.zeros((), device=output["start_logits"].device)
    total_presence = torch.zeros((), device=output["start_logits"].device)

    for sample in range(b):
        pred_start = output["start"][sample]
        pred_vel = output["velocity"][sample]
        pred_presence = output["presence_logits"][sample]

        gt_start = batch["coords"][sample, :, 0]
        gt_vel = batch["velocities"][sample]
        gt_mask = batch["target_mask"][sample]

        assignment = assignment_for_sample(
            pred_start.detach(),
            pred_vel.detach(),
            pred_presence.detach(),
            gt_start,
            gt_vel,
            gt_mask,
        )

        matched_pred = set()

        for pred_slot, gt_slot in assignment:
            matched_pred.add(pred_slot)

            target_map = gaussian_heatmap(
                h, w,
                gt_start[gt_slot, 0],
                gt_start[gt_slot, 1],
                sigma=1.5,
                device=output["start_logits"].device,
            )

            heatmap_logits = output["start_logits"][sample, pred_slot]

            log_probs = F.log_softmax(
                heatmap_logits.flatten(),
                dim=0,
            ).reshape_as(heatmap_logits)

            total_heatmap = total_heatmap + (
                -(target_map * log_probs).sum()
            )

            total_start = total_start + F.smooth_l1_loss(
                pred_start[pred_slot] / IMAGE_SIZE,
                gt_start[gt_slot] / IMAGE_SIZE,
            )

            total_velocity = total_velocity + F.smooth_l1_loss(
                pred_vel[pred_slot] / IMAGE_SIZE,
                gt_vel[gt_slot] / IMAGE_SIZE,
            )

            total_presence = total_presence + F.binary_cross_entropy_with_logits(
                pred_presence[pred_slot],
                torch.ones((), device=pred_presence.device),
            )

        
        for pred_slot in range(k):
            if pred_slot not in matched_pred:
                total_presence = total_presence + F.binary_cross_entropy_with_logits(
                    pred_presence[pred_slot],
                    torch.zeros((), device=pred_presence.device),
                )

    normalizer = max(b, 1)

    return (
        W_HEATMAP * total_heatmap / normalizer
        + W_START * total_start / normalizer
        + W_VELOCITY * total_velocity / normalizer
        + W_PRESENCE * total_presence / normalizer
        + W_DIVERSITY * diversity_loss(output["start_logits"])
    )




train_ds = MultiTargetTrackletDataset(
    TRAIN_N,
    seed=SEED,
)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_RUN,
    shuffle=True,
    num_workers=0,
    pin_memory=(device.type == "cuda"),
)



model = V5MultiTargetDetector().to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4,
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS_RUN,
    eta_min=1e-6,
)

amp = device.type == "cuda"
scaler = torch.amp.GradScaler(
    "cuda",
    enabled=amp,
)

print("=" * 72)
print("V1- MULTI-TARGET CONVLSTM")
print("=" * 72)
print(f"Device          : {device}")
print(f"Run             : {RUN_NAME}")
print(f"Training samples: {TRAIN_N:,}")
print(f"Epochs          : {EPOCHS_RUN}")
print(f"Batch           : {BATCH_RUN}")
print(f"Max targets     : {MAX_TARGETS}")
print(f"AMP             : {amp}")
print(f"LR              : {LR}")
print()
print("Input: 4 complete BLIP frames")
print("Output: up to 3 localized moving trajectories")
print("=" * 72)



#Main training loop
for epoch in range(EPOCHS_RUN):
    model.train()

    total_loss = 0.0
    steps = 0

    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{EPOCHS_RUN}",
    )

    for batch in progress:

    
        batch = {
            k: v.to(device, non_blocking=True)
            if torch.is_tensor(v)
            else v
            for k, v in batch.items()
        }

        frames = batch["frames"]

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type=device.type,
            enabled=amp,
        ):
            output = model(frames)
            loss = compute_loss(
                output,
                batch,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            5.0,
        )

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        steps += 1

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    scheduler.step()

    mean_loss = total_loss / max(steps, 1)

    print(
        f"Epoch {epoch+1:03d}/{EPOCHS_RUN} | "
        f"loss={mean_loss:.5f} | "
        f"lr={scheduler.get_last_lr()[0]:.8f}"
    )

    
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch + 1,
            "run_name": RUN_NAME,
        },
        MODEL_PATH,
    )

print()
print(f"[SUCCESS] Saved -> {MODEL_PATH}")
