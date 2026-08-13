"""
app.py
------
Streamlit inference app for the V1 multi-target ConvLSTM tracklet detector.

Drop in SEQ_LEN (4) frames as .fits/.fit or .png, in time order. The app:
  1. Loads and resizes each frame to IMAGE_SIZE x IMAGE_SIZE.
  2. Runs the EXACT same BLIP preprocessing used in training
     (env_gen.make_blips: median-frame differencing + robust MAD
     normalization), so inference sees the representation the model
     was actually trained on.
  3. Runs the trained V5MultiTargetDetector checkpoint.
  4. Shows per-slot detections (position / velocity / confidence),
     an annotated frame grid, and an animated GIF with predicted
     tracks overlaid.

Place this file alongside config.py / env_gen.py / model.py, e.g.:
    C:\\#Self Study\\Projects\\Casptone Prototype\\src\\app.py

Install (once):
    pip install streamlit torch numpy pillow astropy

Run:
    streamlit run app.py
"""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image, ImageDraw

try:
    from astropy.io import fits
    HAVE_ASTROPY = True
except ImportError:
    HAVE_ASTROPY = False

from config import *
from env_gen import make_blips
from model import V5MultiTargetDetector


# ---------------------------------------------------------------- #
# Page setup
# ---------------------------------------------------------------- #
st.set_page_config(
    page_title="Asteroid Tracklet Detector (V1 ConvLSTM)",
    layout="wide",
)

st.title("Asteroid Tracklet Detector — ConvLSTM Prototype")
st.caption(
    "Upload 4 sequential frames (FITS or PNG). The app rebuilds the "
    "exact BLIP preprocessing used in training, runs the trained "
    "ConvLSTM detector, and returns detections plus an annotated GIF. "
    "Note: this checkpoint was trained entirely on synthetic data — "
    "treat outputs on real frames as a feasibility check, not a "
    "calibrated result."
)


# ---------------------------------------------------------------- #
# Model loading
# ---------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading model...")
def load_model(checkpoint_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = V5MultiTargetDetector().to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    return model, device, ckpt.get("epoch", "?"), ckpt.get("run_name", "?")


# ---------------------------------------------------------------- #
# Sidebar - checkpoint + thresholds
# ---------------------------------------------------------------- #
st.sidebar.header("Model")

default_ckpt = str(BASE_DIR / "v5m_overnight.pth")
checkpoint_path = st.sidebar.text_input("Checkpoint path (.pth)", value=default_ckpt)

confidence_threshold = st.sidebar.slider(
    "Presence confidence threshold", 0.0, 1.0, 0.50, 0.01
)

use_hard_start = st.sidebar.checkbox(
    "Use hard-argmax localization (instead of soft-argmax)", value=False
)

flip_fits_vertically = st.sidebar.checkbox(
    "Flip FITS frames vertically (astronomical origin)", value=True
)

gif_fps = st.sidebar.slider("Output GIF speed (fps)", 1, 10, 2)

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Model expects {SEQ_LEN} frames of size {IMAGE_SIZE}x{IMAGE_SIZE}. "
    f"Uploaded frames are resized to fit — if your real frames have a "
    f"different field of view / plate scale than the synthetic training "
    f"data, predicted velocities are only relative, not physically "
    f"calibrated to arcsec/frame."
)

if not HAVE_ASTROPY:
    st.sidebar.warning("astropy is not installed — FITS uploads will fail. `pip install astropy`.")


# ---------------------------------------------------------------- #
# File loading helpers
# ---------------------------------------------------------------- #
def load_frame_array(uploaded_file) -> np.ndarray:
    """Load a single uploaded FITS or PNG file as a float32 2D array."""
    name = uploaded_file.name.lower()
    data = uploaded_file.read()

    if name.endswith((".fits", ".fit")):
        if not HAVE_ASTROPY:
            raise RuntimeError("astropy is required to read FITS files (pip install astropy).")

        with fits.open(io.BytesIO(data)) as hdul:
            arr = None
            for hdu in hdul:
                if hdu.data is not None:
                    arr = hdu.data
                    break
            if arr is None:
                raise ValueError(f"No image data found in FITS file: {uploaded_file.name}")

        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[0]
        if flip_fits_vertically:
            arr = np.flipud(arr)

    else:
        img = Image.open(io.BytesIO(data)).convert("L")
        arr = np.asarray(img, dtype=np.float32)

    return arr


def resize_to_model_input(arr: np.ndarray) -> np.ndarray:
    """Resize a 2D array to IMAGE_SIZE x IMAGE_SIZE via PIL (bilinear)."""
    img = Image.fromarray(arr)
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)


def normalize_for_display(arr: np.ndarray) -> np.ndarray:
    """Percentile stretch of a raw frame, for viewing only."""
    lo, hi = np.percentile(arr, [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    out = np.clip((arr - lo) / (hi - lo), 0, 1)
    return (out * 255).astype(np.uint8)


def blip_to_display(blip: np.ndarray) -> np.ndarray:
    """BLIP frame in [-1, 1] -> uint8 grayscale for display."""
    out = (blip + 1.0) / 2.0
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


# ---------------------------------------------------------------- #
# Upload
# ---------------------------------------------------------------- #
st.subheader("1. Upload 4 sequential frames")

uploaded_files = st.file_uploader(
    f"Drop exactly {SEQ_LEN} frames, in time order (t=0 first)",
    type=["fits", "fit", "png"],
    accept_multiple_files=True,
)

ready = uploaded_files is not None and len(uploaded_files) == SEQ_LEN

if uploaded_files and not ready:
    st.warning(
        f"Model expects exactly {SEQ_LEN} frames — you uploaded "
        f"{len(uploaded_files)}. Add or remove files to continue."
    )

raw_stack = None

if ready:
    st.caption(
        "Order below is upload order and is treated as t=0..t=3. "
        "Re-upload in the correct order if this looks wrong."
    )

    try:
        raw_arrays = [resize_to_model_input(load_frame_array(f)) for f in uploaded_files]
    except Exception as e:
        st.error(f"Failed to load one or more frames: {e}")
        raw_arrays = None

    if raw_arrays is not None:
        cols = st.columns(SEQ_LEN)
        for i, (f, arr) in enumerate(zip(uploaded_files, raw_arrays)):
            with cols[i]:
                st.image(
                    normalize_for_display(arr),
                    caption=f"t={i}  ({f.name})",
                    use_container_width=True,
                )
        raw_stack = np.stack(raw_arrays).astype(np.float32)  # [T, H, W]


# ---------------------------------------------------------------- #
# Run inference
# ---------------------------------------------------------------- #
st.subheader("2. Run detection")

run_clicked = st.button("Run detection", type="primary", disabled=raw_stack is None)

if run_clicked and raw_stack is not None:
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        st.error(f"Checkpoint not found:\n{checkpoint_path}")
        st.stop()

    try:
        with st.spinner("Preprocessing (median-frame BLIP transform)..."):
            blips = make_blips(raw_stack)  # [T, H, W] — identical to training

        with st.spinner("Loading model and running inference..."):
            model, device, epoch, run_name = load_model(str(ckpt_file))

            frames_tensor = (
                torch.from_numpy(blips[:, None]).unsqueeze(0).to(device)
            )  # [1, T, 1, H, W]

            with torch.no_grad():
                output = model(frames_tensor, use_hard_start=use_hard_start)

            pred_start = output["start"][0].cpu().numpy()                        # [K,2]
            pred_velocity = output["velocity"][0].cpu().numpy()                  # [K,2]
            pred_coords = output["coords"][0].cpu().numpy()                      # [K,T,2]
            pred_score = torch.sigmoid(output["presence_logits"][0]).cpu().numpy()  # [K]

    except Exception as e:
        st.error(f"Inference failed: {e}")
        st.stop()

    st.success(f"Inference complete — checkpoint epoch {epoch} (run: {run_name})")

    detections = [
        {
            "slot": k,
            "confidence": float(pred_score[k]),
            "start_x": float(pred_start[k, 0]),
            "start_y": float(pred_start[k, 1]),
            "velocity_x": float(pred_velocity[k, 0]),
            "velocity_y": float(pred_velocity[k, 1]),
            "detected": bool(pred_score[k] >= confidence_threshold),
        }
        for k in range(pred_start.shape[0])
    ]
    det_df = pd.DataFrame(detections)

    st.subheader("3. Detections")
    st.dataframe(det_df, use_container_width=True)
    st.download_button(
        "Download detections (CSV)",
        data=det_df.to_csv(index=False).encode("utf-8"),
        file_name="detections.csv",
        mime="text/csv",
    )

    if not det_df["detected"].any():
        st.info("No slots passed the confidence threshold — try lowering it in the sidebar.")

    # ------------------------------------------------------------ #
    # Annotated frames + GIF
    # ------------------------------------------------------------ #
    st.subheader("4. Annotated tracklet")

    slot_colors = ["#ff3b30", "#34c759", "#007aff"]  # up to MAX_TARGETS slots
    annotated_frames = []

    for t in range(SEQ_LEN):
        base = Image.fromarray(blip_to_display(blips[t])).convert("RGB")
        draw = ImageDraw.Draw(base)

        for d in detections:
            if not d["detected"]:
                continue
            k = d["slot"]
            color = slot_colors[k % len(slot_colors)]

            pts = [(pred_coords[k, i, 0], pred_coords[k, i, 1]) for i in range(SEQ_LEN)]
            draw.line(pts, fill=color, width=2)

            x, y = pred_coords[k, t]
            r = 6
            draw.ellipse([x - r, y - r, x + r, y + r], outline=color, width=2)
            draw.text((x + 8, y - 8), f"T{k} {d['confidence']:.2f}", fill=color)

        annotated_frames.append(base)

    frame_cols = st.columns(SEQ_LEN)
    for i, img in enumerate(annotated_frames):
        with frame_cols[i]:
            st.image(img, caption=f"t={i}", use_container_width=True)

    gif_buffer = io.BytesIO()
    duration_ms = int(1000 / gif_fps)
    annotated_frames[0].save(
        gif_buffer,
        format="GIF",
        save_all=True,
        append_images=annotated_frames[1:],
        duration=duration_ms,
        loop=0,
    )
    gif_buffer.seek(0)

    st.image(gif_buffer, caption="Predicted tracklet (looping)")
    st.download_button(
        "Download annotated GIF",
        data=gif_buffer,
        file_name="tracklet_detection.gif",
        mime="image/gif",
    )

elif raw_stack is None:
    st.info(f"Upload {SEQ_LEN} frames above to enable detection.")
