# V1- Multi-Target ConvLSTM Asteroid Tracklet Detector

A prototype deep-learning pipeline that automates the manual "blink" step
analysts currently do by eye in tools like Astrometrica: given 4 chronological
frames of the same patch of sky, find point sources that move in a straight
line across the sequence, and reject everything that doesn't (stars,
cosmic rays, hot pixels, one-off artifacts).

> **Status: early prototype, synthetic data only.**
> The model has never been trained or evaluated on real telescope data
> (ATLAS or otherwise). All numbers in this README come from a synthetic
> data generator built to mimic the general structure of the problem
> (point sources, linear motion, background noise), not real sky images.
> Treat this as a proof of concept that the architecture *can learn the
> task*, not as a validated detector.

---

## Why this exists

Blinking 4-5 images by eye to spot a moving asteroid among thousands of
static stars is slow and doesn't scale. This project explores whether a
small neural network can learn to do that same visual comparison
automatically: watch a short image sequence, tell moving point sources
apart from noise, and report where each one is and how fast it's moving.

## How it works

```
4 chronological frames
        |
        v
 per-frame CNN encoder (shared weights)      "shrink each frame, keep blob-like features"
        |
        v
 bidirectional ConvLSTM                      "watch the sequence forwards AND backwards"
        |
        v
 temporal fusion -> one 64-channel map        "one summary of everything observed"
        |
        v
 up to 3 heatmap slots                        "where might a moving object be?"
        |
        v
 soft-argmax -> (x, y) per slot                "turn each glow into a coordinate"
        |
        v
 per-slot velocity + presence heads            "how fast is it moving? is it real?"
        |
        v
 coords = start + t * velocity                 "draw the straight line through all 4 frames"
```

The straight-line assumption is baked directly into the model's output
layer (not just the loss) — the network can only ever predict a start
point and a constant velocity per slot, never a curved or erratic path.
This mirrors the real physics of near-Earth-object tracklets over a short
observation window.

## Repository structure

| File | Purpose |
|---|---|
| `config.py` | All tunable constants: image size, synthetic-data ranges, training presets |
| `env_gen.py` | Synthetic 4-frame tracklet generator (stars, distractors, moving targets, BLIP differencing) |
| `verify_genData.py` / `generate_examples.py` | Renders sample synthetic scenes to PNG for visual sanity-checking the generator |
| `model.py` | The `V5MultiTargetDetector` architecture (bidirectional ConvLSTM + multi-slot heatmap head) |
| `train_detector.py` | Training loop with `quick` / `overnight` / `research` presets |
| `evaluate_detector.py` | Runs a trained checkpoint against a held-out synthetic validation set and reports precision/recall/F1/localization error |
| `app.py` | Flask web app: upload 4 frames, get back detected trajectories. Automatically switches to an overlapping sliding-window scan for frames larger than the model's native 256×256 input |
| `v5m_quick.pth` | Checkpoint from the fastest training preset (see Results) |

## Setup

```bash
pip install torch numpy tqdm matplotlib flask astropy pillow --break-system-packages
```


## Usage

**1. Sanity-check the synthetic data generator**
```bash
python verify_genData.py
```
Saves example scenes (raw frames, BLIP frames, and ground-truth tracks) to
`generated_examples/` so you can visually confirm the synthetic tracklets
look reasonable before training on them.

**2. Train**

Edit the `RUN_NAME` variable at the top of `train_detector.py` to one of:

| Preset | Sequences | Epochs | Rough scale |
|---|---|---|---|
| `quick` | 2,000 | 10 | fast smoke test |
| `overnight` | 50,000 | 40 (or fewer — see Results) | main working checkpoint |
| `research` | 100,000 | 60 | not yet run |

```bash
python train_detector.py
```
Checkpoints are saved every epoch, so an interrupted run isn't lost.

**3. Evaluate**
```bash
python evaluate_detector.py
```
Runs the checkpoint against 500 held-out synthetic scenes and reports
precision/recall/F1, exact target-count accuracy, and localization error,
broken down by how many true moving objects were in each scene.

**4. Try it interactively**
```bash
python app.py
```
Open `http://127.0.0.1:5000`, upload 4 chronological frames (image or FITS),
and get back detected trajectories overlaid on the BLIP-differenced images.
Frames ≤256×256 are used as-is; larger frames are automatically scanned with
overlapping windows, with duplicate detections from overlapping windows
merged automatically.

## Results so far

Evaluated on 500 synthetic held-out scenes, match threshold 30px, confidence
threshold 0.50.

| | `quick` (2k seqs / 10 epochs) | 50k seqs / 10 epochs |
|---|---|---|
| Precision | 0.49 | 0.75 |
| Recall | 0.53 | 0.76 |
| F1 | 0.51 | 0.76 |
| Median trajectory error | 13.2 px | 6.48 px |
| Detections within 5 px | 16.5% | 41.6% |
| False positives on empty scenes | ~10.4% | ~3.0% |

Broken down by scene difficulty (50k/10-epoch checkpoint):

| True targets in scene | Precision | Recall |
|---|---|---|
| 1 | 0.984 | 0.976 |
| 2 | 0.732 | 0.783 |
| 3 | 0.694 | 0.664 |

**Reading these numbers:** single-target detection is essentially solved on
synthetic data. Performance degrades as more true targets are packed into
one scene — the leading hypothesis is competition between the model's 3
heatmap slots when multiple real objects are present, rather than a
fundamental architectural limitation, since more training data produced a
large jump in exactly this regime. This is the main open problem, not
overall detection viability.

## Known limitations

- **No real astronomical data yet.** All results are on synthetic scenes.
  Real ATLAS crops will have artifacts the generator doesn't yet simulate
  well: satellite trails, subtraction dipoles near bright stars, PSF
  elongation for fast movers, cosmic rays with non-Gaussian profiles.
- **Multi-target scenes (2-3 objects) are the weakest case**, both in
  precision and recall.
- **Only 3 target slots.** Denser fields with more than 3 simultaneous
  movers aren't supported by the current architecture.
- Assumes frames are already spatially registered/aligned (no internal
  handling of frame-to-frame drift beyond what's baked into the BLIP step).

## Roadmap

- [ ] Run the full `overnight` (40-epoch) and `research` presets
- [ ] Add more realistic distractor types to the synthetic generator
      (streaks, dipole residuals, non-Gaussian cosmic-ray profiles)
- [ ] Validate against real ATLAS tracklets
- [ ] Investigate slot-competition in dense (2-3 target) scenes
- [ ] Add `requirements.txt` / environment file
- [ ] Add a license

## Disclaimer

This is a research prototype, not a production detection pipeline. It has
not been validated against real survey data and should not be relied on
for actual asteroid discovery or follow-up decisions.
