"""
V5-MULTI web app.

Upload exactly four chronological images/FITS files.

Pipeline:
    4 images
      -> load/normalize
      -> complete four-frame temporal median BLIP
      -> V5-Multi ConvLSTM
      -> up to MAX_TARGETS trajectory slots
      -> confidence filtering
      -> colored trajectory visualization

This is an inference/visualization app only. It does not retrain the model.
"""

import io
import base64
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from flask import Flask, request, render_template_string
from astropy.io import fits

from config import *
from model import V5MultiTargetDetector


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = BASE_DIR / "v5m_quick.pth"

DEFAULT_THRESHOLD = 0.50


# ============================================================
# APP / MODEL
# ============================================================

app = Flask(__name__)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = V5MultiTargetDetector().to(device)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model checkpoint not found:\n{MODEL_PATH}\n"
        "Train V5-M first or change MODEL_PATH in app.py."
    )

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False,
)

model.load_state_dict(
    checkpoint["model"]
)

model.eval()

print(f"[INFO] Device: {device}")
print(f"[INFO] Model: {MODEL_PATH}")


# ============================================================
# HTML
# ============================================================

HTML = r"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>V5-M Multi-Target ConvLSTM Detector</title>

<style>
body {
    background:#09111b;
    color:#eeeeee;
    font-family:Segoe UI,Arial,sans-serif;
    padding:28px;
    max-width:1450px;
    margin:auto;
}

.card {
    background:#101c2b;
    border:1px solid #26384d;
    border-radius:12px;
    padding:20px;
    margin-bottom:18px;
}

input, button {
    margin:7px;
    padding:9px;
}

button {
    cursor:pointer;
    font-weight:600;
}

img {
    max-width:100%;
    border-radius:8px;
}

.err {
    color:#ff9b9b;
}

.good {
    color:#9ff0b0;
}

table {
    border-collapse:collapse;
    width:100%;
}

td, th {
    border-bottom:1px solid #26384d;
    padding:8px;
    text-align:left;
}

code {
    color:#a9d7ff;
}

.small {
    color:#aab7c5;
    font-size:0.92em;
}
</style>
</head>

<body>

<h2>V5-M Multi-Target ConvLSTM Detector</h2>

<div class="card">

<p>
Upload <b>exactly four chronological frames</b>.
PNG/JPG/TIFF/BMP and astronomical FITS files are supported.
</p>

<p class="small">
All four frames are loaded before the temporal BLIP reference is
constructed. The model then makes one retrospective inference over
the complete four-frame sequence.
</p>

<form method="post" enctype="multipart/form-data">

<label>
Frame 0:
<input type="file" name="f0"
accept="image/*,.fits,.fit,.fts" required>
</label>
<br>

<label>
Frame 1:
<input type="file" name="f1"
accept="image/*,.fits,.fit,.fts" required>
</label>
<br>

<label>
Frame 2:
<input type="file" name="f2"
accept="image/*,.fits,.fit,.fts" required>
</label>
<br>

<label>
Frame 3:
<input type="file" name="f3"
accept="image/*,.fits,.fit,.fts" required>
</label>
<br>

<label>
Confidence threshold:
<input
type="number"
name="threshold"
min="0"
max="1"
step="0.01"
value="{{threshold if threshold else 0.50}}"
>
</label>

<br>

<button type="submit">
BLIP all 4 + detect trajectories
</button>

</form>
</div>


{% if error %}
<div class="card err">
<b>Error:</b> {{error}}
</div>
{% endif %}


{% if results %}

<div class="card">

<h3>Detection summary</h3>

<p class="good">
{{summary}}
</p>

<table>
<tr>
<th>Track</th>
<th>Confidence</th>
<th>Start (x,y)</th>
<th>Velocity (vx,vy)</th>
<th>Frame 0</th>
<th>Frame 1</th>
<th>Frame 2</th>
<th>Frame 3</th>
</tr>

{% for r in results %}
<tr>
<td>Track {{r.id}}</td>
<td>{{"%.4f"|format(r.confidence)}}</td>
<td>({{"%.1f"|format(r.start_x)}}, {{"%.1f"|format(r.start_y)}})</td>
<td>({{"%.2f"|format(r.vx)}}, {{"%.2f"|format(r.vy)}})</td>

{% for p in r.points %}
<td>({{"%.1f"|format(p[0])}}, {{"%.1f"|format(p[1])}})</td>
{% endfor %}

</tr>
{% endfor %}

</table>

</div>


<div class="card">
<h3>1. Uploaded / normalized frames</h3>
<img src="data:image/png;base64,{{raw_img}}">
</div>


<div class="card">
<h3>2. Signed BLIP frames actually fed to V5-M</h3>
<img src="data:image/png;base64,{{blip_img}}">
</div>


<div class="card">
<h3>3. Multiple inferred trajectories</h3>
<img src="data:image/png;base64,{{infer_img}}">
</div>


<div class="card">
<h3>4. Final spatial evidence for all target slots</h3>
<img src="data:image/png;base64,{{heat_img}}">
</div>

{% endif %}

</body>
</html>
"""


# ============================================================
# IMAGE LOADING
# ============================================================

def normalize_fits(data):
    data = np.asarray(
        data,
        dtype=np.float32,
    )

    data = np.squeeze(data)

    while data.ndim > 2:
        data = data[0]

    if data.ndim != 2:
        raise ValueError(
            f"Unsupported FITS dimensions: {data.shape}"
        )

    data = np.nan_to_num(
        data,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    lo = np.percentile(data, 1.0)
    hi = np.percentile(data, 99.5)

    if hi <= lo:
        hi = lo + 1.0

    data = np.clip(
        (data - lo) / (hi - lo),
        0.0,
        1.0,
    )

    return data


def load_uploaded_image(file_storage):
    """
    Supports common image formats and FITS.

    FITS normalization is for visualization/prototyping and should
    not be interpreted as calibrated astronomical photometry.
    """

    name = (
        file_storage.filename or ""
    ).lower()

    if name.endswith(
        (".fits", ".fit", ".fts")
    ):
        with fits.open(
            file_storage.stream,
            memmap=False,
        ) as hdul:

            image = None

            for hdu in hdul:
                if getattr(hdu, "data", None) is not None:
                    image = hdu.data
                    break

            if image is None:
                raise ValueError(
                    "No image data found in FITS file."
                )

        image = normalize_fits(image)

        image = Image.fromarray(
            (image * 255).astype(np.uint8)
        )

    else:
        image = Image.open(
            file_storage.stream
        ).convert("L")

    image = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.LANCZOS,
    )

    return np.asarray(
        image,
        dtype=np.float32,
    ) / 255.0


# ============================================================
# BLIP
# ============================================================

def make_blips(raw):
    """
    Build signed difference/BLIP frames after ALL four inputs
    have been loaded.
    """

    reference = np.median(
        raw,
        axis=0,
    )

    residual = (
        raw
        - reference[None]
    )

    med = np.median(
        residual
    )

    mad = np.median(
        np.abs(
            residual - med
        )
    )

    scale = max(
        1.4826 * mad * 6.0,
        0.04,
    )

    return np.clip(
        (residual - med) / scale,
        -1.0,
        1.0,
    ).astype(np.float32)


# ============================================================
# VISUALIZATION
# ============================================================

def fig_to_base64(fig):
    buffer = io.BytesIO()

    fig.savefig(
        buffer,
        format="png",
        dpi=130,
        bbox_inches="tight",
    )

    plt.close(fig)

    return base64.b64encode(
        buffer.getvalue()
    ).decode()


def plot_four_frames(
    data,
    titles,
    cmap="gray",
    vmin=None,
    vmax=None,
):
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(14, 3.5),
    )

    for t, ax in enumerate(axes):
        ax.imshow(
            data[t],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_title(
            titles[t]
        )

        ax.axis("off")

    fig.tight_layout()

    return fig_to_base64(fig)


def plot_tracks(
    blips,
    tracks,
):
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(14, 3.5),
    )

    for t, ax in enumerate(axes):

        ax.imshow(
            blips[t],
            cmap="gray",
            vmin=-1,
            vmax=1,
        )

        for track in tracks:

            points = np.asarray(
                track["points"]
            )

            ax.plot(
                points[:, 0],
                points[:, 1],
                linewidth=2.0,
                marker="o",
                markersize=3,
                label=(
                    f"T{track['id']} "
                    f"{track['confidence']:.2f}"
                ),
            )

            ax.scatter(
                points[t, 0],
                points[t, 1],
                marker="+",
                s=130,
                linewidths=2,
            )

        ax.set_title(
            f"BLIP F{t}"
        )

        ax.axis("off")

    if tracks:
        axes[0].legend(
            loc="upper right",
            fontsize=8,
        )

    fig.tight_layout()

    return fig_to_base64(fig)


def plot_heatmaps(start_logits):
    """
    Show each prediction slot's spatial evidence independently.
    """

    k = start_logits.shape[0]

    fig, axes = plt.subplots(
        1,
        k,
        figsize=(4.0 * k, 4.0),
    )

    if k == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        im = ax.imshow(
            start_logits[i],
            cmap="magma",
        )

        ax.set_title(
            f"Slot {i}"
        )

        ax.axis("off")

        fig.colorbar(
            im,
            ax=ax,
            fraction=0.046,
            pad=0.04,
        )

    fig.tight_layout()

    return fig_to_base64(fig)


# ============================================================
# INFERENCE
# ============================================================

def infer(blips, threshold):
    tensor = torch.from_numpy(
        blips[:, None]
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(
            tensor,
            use_hard_start=False,
        )

        hard_output = model(
            tensor,
            use_hard_start=True,
        )

    probabilities = torch.sigmoid(
        output["presence_logits"][0]
    ).cpu().numpy()

    soft_start = (
        output["start"][0]
        .cpu()
        .numpy()
    )

    velocity = (
        output["velocity"][0]
        .cpu()
        .numpy()
    )

    soft_coords = (
        output["coords"][0]
        .cpu()
        .numpy()
    )

    hard_coords = (
        hard_output["coords"][0]
        .cpu()
        .numpy()
    )

    start_logits = (
        output["start_logits"][0]
        .cpu()
        .numpy()
    )

    tracks = []

    for k in range(
        len(probabilities)
    ):

        if probabilities[k] < threshold:
            continue

        # Use the soft trajectory for the main visualization.
        # Hard coordinates are retained for diagnostics.
        points = soft_coords[k]

        tracks.append(
            {
                "id": k,
                "confidence": float(
                    probabilities[k]
                ),
                "start_x": float(
                    soft_start[k, 0]
                ),
                "start_y": float(
                    soft_start[k, 1]
                ),
                "vx": float(
                    velocity[k, 0]
                ),
                "vy": float(
                    velocity[k, 1]
                ),
                "points": [
                    [
                        float(p[0]),
                        float(p[1]),
                    ]
                    for p in points
                ],
                "hard_points": [
                    [
                        float(p[0]),
                        float(p[1]),
                    ]
                    for p in hard_coords[k]
                ],
            }
        )

    tracks.sort(
        key=lambda x: x["confidence"],
        reverse=True,
    )

    return (
        tracks,
        start_logits,
    )


# ============================================================
# ROUTE
# ============================================================

@app.route(
    "/",
    methods=["GET", "POST"],
)
def home():

    if request.method == "GET":
        return render_template_string(
            HTML,
            error=None,
            results=None,
            threshold=DEFAULT_THRESHOLD,
        )

    try:
        threshold = float(
            request.form.get(
                "threshold",
                DEFAULT_THRESHOLD,
            )
        )

        threshold = float(
            np.clip(
                threshold,
                0.0,
                1.0,
            )
        )

        uploads = [
            request.files.get(
                f"f{i}"
            )
            for i in range(4)
        ]

        if any(
            f is None or not f.filename
            for f in uploads
        ):
            raise ValueError(
                "Please provide all four chronological frames."
            )

        # IMPORTANT:
        # Load ALL FOUR before making any BLIP.
        raw = np.stack(
            [
                load_uploaded_image(f)
                for f in uploads
            ],
            axis=0,
        )

        blips = make_blips(
            raw
        )

        tracks, start_logits = infer(
            blips,
            threshold,
        )

        raw_img = plot_four_frames(
            raw,
            [
                "Input F0",
                "Input F1",
                "Input F2",
                "Input F3",
            ],
            vmin=0,
            vmax=1,
        )

        blip_img = plot_four_frames(
            blips,
            [
                "BLIP F0",
                "BLIP F1",
                "BLIP F2",
                "BLIP F3",
            ],
            vmin=-1,
            vmax=1,
        )

        infer_img = plot_tracks(
            blips,
            tracks,
        )

        heat_img = plot_heatmaps(
            start_logits
        )

        if tracks:
            summary = (
                f"Detected {len(tracks)} "
                f"trajectory(s) above threshold "
                f"{threshold:.2f}."
            )
        else:
            summary = (
                f"No trajectory exceeded "
                f"threshold {threshold:.2f}."
            )

        return render_template_string(
            HTML,
            error=None,
            results=tracks,
            summary=summary,
            threshold=threshold,
            raw_img=raw_img,
            blip_img=blip_img,
            infer_img=infer_img,
            heat_img=heat_img,
        )

    except Exception as exc:

        return render_template_string(
            HTML,
            error=str(exc),
            results=None,
            threshold=DEFAULT_THRESHOLD,
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "[INFO] V5-M multi-target app"
    )

    print(
        f"[INFO] Device: {device}"
    )

    print(
        f"[INFO] Max target slots: {MAX_TARGETS}"
    )

    print(
        f"[INFO] Model: {MODEL_PATH}"
    )

    print(
        "[INFO] Open http://127.0.0.1:5000"
    )

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
    )
