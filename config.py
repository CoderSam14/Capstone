from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
_PARENT_TRACKLETS = PROJECT_ROOT.parent / "Data" / "tracklets"
_LOCAL_TRACKLETS = PROJECT_ROOT / "Data" / "tracklets"
TRACKLETS_ROOT = _PARENT_TRACKLETS if _PARENT_TRACKLETS.exists() else _LOCAL_TRACKLETS
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
IMAGE_SIZE = 1024
MIN_SEQUENCE_FRAMES = 3
INJECTION_SPEED_RANGE = (50, 75)

PRESETS = {
    "test": dict(
        epochs=1, lr=1e-4, weight_decay=1e-4, seed=42, amp=True,
        crop_size=IMAGE_SIZE, min_frames=MIN_SEQUENCE_FRAMES,
        synthetic_per_tracklet=1, synthetic_negatives_per_tracklet=1,
        validation_per_tracklet=1, validation_negatives_per_tracklet=1,
        injection_amplitude_range=(40.0, 50.0), injection_sigma_range=(1.2, 2.2),
        synthetic_speed_px_frame_range=INJECTION_SPEED_RANGE,
        max_velocity_px_frame=INJECTION_SPEED_RANGE[1], motion_jitter_std=0.10,
        injection_count_range=(1, 4), num_workers=0, prefetch_factor=2,
    ),
    "full": dict(
        epochs=60, lr=1e-4, weight_decay=1e-4, seed=42, amp=True,
        crop_size=IMAGE_SIZE, min_frames=MIN_SEQUENCE_FRAMES,
        synthetic_per_tracklet=8, synthetic_negatives_per_tracklet=8,
        validation_per_tracklet=2, validation_negatives_per_tracklet=2,
        injection_amplitude_range=(40.0, 50.0), injection_sigma_range=(1.2, 2.2),
        synthetic_speed_px_frame_range=INJECTION_SPEED_RANGE,
        max_velocity_px_frame=INJECTION_SPEED_RANGE[1], motion_jitter_std=0.10,
        injection_count_range=(1, 4), num_workers=4, prefetch_factor=2,
    ),
}

def get_config(preset="full"):
    if preset not in PRESETS:
        raise ValueError("preset must be 'test' or 'full'")
    cfg = dict(PRESETS[preset])
    cfg["preset"] = preset
    cfg["output_stride"] = 4
    cfg["min_registered_fraction"] = 0.90
    return cfg

def seed_everything(seed=SEED):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
