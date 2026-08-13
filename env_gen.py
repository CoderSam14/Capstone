"""Synthetic 4-frame tracklet generator."""

import math
import numpy as np
import torch
from torch.utils.data import Dataset
from config import *


def stamp(image, x, y, amplitude, sigma):
    r = max(3, int(math.ceil(4 * sigma)))
    h, w = image.shape
    x0, x1 = max(0, int(x)-r), min(w, int(x)+r+2)
    y0, y1 = max(0, int(y)-r), min(h, int(y)+r+2)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    image[y0:y1, x0:x1] += amplitude * np.exp(
        -((xx-x)**2 + (yy-y)**2) / (2*sigma*sigma)
    )


def generate_linear_track(rng):
    for _ in range(500):
        speed = rng.uniform(*VELOCITY_RANGE)
        angle = rng.uniform(0, 2*np.pi)
        velocity = np.array(
            [speed*np.cos(angle), speed*np.sin(angle)],
            dtype=np.float32
        )
        start = rng.uniform(
            30, IMAGE_SIZE-30, size=2
        ).astype(np.float32)

        acceleration = np.zeros(2, dtype=np.float32)
        if rng.random() < ACCELERATION_PROBABILITY:
            acceleration = rng.uniform(
                *ACCELERATION_RANGE, size=2
            ).astype(np.float32)

        coords = np.asarray([
            start + t*velocity + 0.5*(t**2)*acceleration
            for t in range(SEQ_LEN)
        ], dtype=np.float32)

        if ((coords > 12).all() and
            (coords < IMAGE_SIZE-12).all()):
            return coords, velocity

    raise RuntimeError("Could not generate valid track")


def sufficiently_separate(candidate, existing, minimum_distance=12.0):
    for other in existing:
        if np.linalg.norm(candidate-other, axis=1).min() < minimum_distance:
            return False
    return True


def generate_targets(rng):
    n = int(rng.integers(MIN_TARGETS, MAX_TARGETS+1))
    tracks, velocities = [], []

    for _ in range(n):
        for _attempt in range(500):
            coords, velocity = generate_linear_track(rng)
            if sufficiently_separate(coords, tracks):
                tracks.append(coords)
                velocities.append(velocity)
                break

    if tracks:
        return (
            np.stack(tracks).astype(np.float32),
            np.stack(velocities).astype(np.float32)
        )

    return (
        np.empty((0, SEQ_LEN, 2), np.float32),
        np.empty((0, 2), np.float32)
    )


def generate_raw_sequence(rng):
    h = w = IMAGE_SIZE
    base = rng.normal(
        0, rng.uniform(*BACKGROUND_NOISE_STD_RANGE), (h, w)
    ).astype(np.float32)

    gx = np.linspace(-1, 1, w, dtype=np.float32)
    gy = np.linspace(-1, 1, h, dtype=np.float32)
    base += 0.004*gx[None, :] + 0.004*gy[:, None]

    for _ in range(int(rng.integers(MIN_STARS, MAX_STARS+1))):
        stamp(
            base, rng.uniform(5,w-5), rng.uniform(5,h-5),
            rng.uniform(*STAR_AMP_RANGE),
            rng.uniform(*STAR_SIGMA_RANGE)
        )

    return np.stack([
        base + rng.normal(
            0, rng.uniform(*BACKGROUND_NOISE_STD_RANGE), (h,w)
        ).astype(np.float32)
        for _ in range(SEQ_LEN)
    ]).astype(np.float32)


def add_distractors(raw, rng):
    for t in range(SEQ_LEN):
        for _ in range(int(rng.integers(
            MIN_DISTRACTORS, MAX_DISTRACTORS+1
        ))):
            stamp(
                raw[t],
                rng.uniform(10, IMAGE_SIZE-10),
                rng.uniform(10, IMAGE_SIZE-10),
                rng.uniform(*DISTRACTOR_AMP_RANGE),
                rng.uniform(*DISTRACTOR_SIGMA_RANGE)
            )


def render_targets(raw, coords, rng):
    for track in coords:
        amplitude = rng.uniform(*TARGET_AMP_RANGE)
        sigma = rng.uniform(*TARGET_SIGMA_RANGE)
        rendered = track + rng.normal(
            0, MOTION_JITTER_STD, track.shape
        ).astype(np.float32)

        for t in range(SEQ_LEN):
            stamp(
                raw[t], rendered[t,0], rendered[t,1],
                amplitude, sigma
            )


def make_blips(raw):
    reference = np.median(raw, axis=0)
    residual = raw - reference[None]
    med = np.median(residual)
    mad = np.median(np.abs(residual-med))
    scale = max(1.4826*mad*6.0, 0.04)
    return np.clip(
        (residual-med)/scale, -1, 1
    ).astype(np.float32)


class MultiTargetTrackletDataset(Dataset):
    """Each sample contains 0-3 true moving objects."""

    def __init__(self, n, seed=SEED):
        self.n = int(n)
        self.seed = int(seed)

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        rng = np.random.default_rng(self.seed + int(index))
        raw = generate_raw_sequence(rng)
        coords, velocities = generate_targets(rng)

        add_distractors(raw, rng)
        render_targets(raw, coords, rng)

        blips = make_blips(raw)
        n = len(coords)

        padded_coords = np.full(
            (MAX_TARGETS, SEQ_LEN, 2), -1, np.float32
        )
        padded_velocities = np.zeros(
            (MAX_TARGETS, 2), np.float32
        )
        mask = np.zeros(MAX_TARGETS, np.float32)

        if n:
            padded_coords[:n] = coords
            padded_velocities[:n] = velocities
            mask[:n] = 1

        return {
            "frames": torch.from_numpy(blips[:, None]),
            "raw_frames": torch.from_numpy(raw),
            "coords": torch.from_numpy(padded_coords),
            "velocities": torch.from_numpy(padded_velocities),
            "target_count": torch.tensor(n, dtype=torch.long),
            "target_mask": torch.from_numpy(mask),
        }
