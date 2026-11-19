"""Bidirectional ConvLSTM asteroid tracklet detector.

One sequence-level center heatmap is produced for the reference (middle) frame.
Each heatmap peak represents a mover's reference-frame position; a dense motion
head at that peak regresses (vx, vy) in input pixels/frame. Candidate count is
obtained from peaks/NMS, not from a fixed number of learned slots.
"""
from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(channels: int) -> int:
    for g in (8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class ConvNormAct(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(_groups(out_ch), out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_ch), out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class ConvLSTMCell(nn.Module):
    """Spatial ConvLSTM cell; gates operate on concatenated input/hidden maps."""
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        pad = kernel_size // 2
        self.gates = nn.Conv2d(input_channels + hidden_channels, 4 * hidden_channels,
                               kernel_size, padding=pad)
        self.norm = nn.GroupNorm(_groups(4 * hidden_channels), 4 * hidden_channels)

    def forward(self, x, state):
        h, c = state
        gates = self.norm(self.gates(torch.cat((x, h), dim=1)))
        i, f, o, g = gates.chunk(4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_state(self, x):
        shape = (x.shape[0], self.hidden_channels, x.shape[-2], x.shape[-1])
        return (x.new_zeros(shape), x.new_zeros(shape))


class ConvLSTMDirection(nn.Module):
    def __init__(self, input_channels, hidden_channels, reverse=False):
        super().__init__()
        self.cell = ConvLSTMCell(input_channels, hidden_channels)
        self.reverse = bool(reverse)

    def forward(self, sequence):
        # sequence [B,T,C,H,W]
        b, t, _, h, w = sequence.shape
        state = self.cell.init_state(sequence[:, 0])
        outputs = [None] * t
        indices = range(t - 1, -1, -1) if self.reverse else range(t)
        for ti in indices:
            state = self.cell(sequence[:, ti], state)
            outputs[ti] = state[0]
        return torch.stack(outputs, dim=1)


class BidirectionalConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super().__init__()
        self.forward_lstm = ConvLSTMDirection(input_channels, hidden_channels, reverse=False)
        self.backward_lstm = ConvLSTMDirection(input_channels, hidden_channels, reverse=True)

    def forward(self, sequence):
        return torch.cat((self.forward_lstm(sequence), self.backward_lstm(sequence)), dim=2)


class VariableTrackletConvLSTM(nn.Module):
    """BiConvLSTM encoder -> one reference-frame heatmap + per-pixel motion field."""
    architecture = "BiConvLSTM_VariableTrackletHeatmap_v2_SkipsRef"
    velocity_scale = 100.0  # head predicts velocity / this scale for stable optimization

    def __init__(self, input_channels=1, hidden_channels=16):
        super().__init__()
        # Spatial pyramid: retain stride-4 and stride-8 features for decoder skips.
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 12, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_groups(12), 12), nn.GELU(),              # stride 2
        )
        self.encoder_stride4 = nn.Sequential(
            nn.Conv2d(12, 20, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_groups(20), 20), nn.GELU(),
            ConvNormAct(20, 20),
        )
        self.encoder_stride8 = nn.Sequential(
            nn.Conv2d(20, 28, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_groups(28), 28), nn.GELU(),
            ConvNormAct(28, 28),
        )
        self.encoder_stride16 = nn.Sequential(
            nn.Conv2d(28, 32, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_groups(32), 32), nn.GELU(),
            ConvNormAct(32, 32),
        )
        self.temporal = BidirectionalConvLSTM(32, hidden_channels)
        # Mean + max summarize the sequence; explicit reference state anchors
        # the sequence-level heatmap to frame T//2 (including even-length T).
        self.temporal_fusion = nn.Sequential(
            nn.Conv2d(6 * hidden_channels, 48, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(48), 48), nn.GELU(),
            ConvNormAct(48, 32),
        )
        self.decoder_stride8 = ConvNormAct(32 + 28, 24)
        self.decoder_stride4 = ConvNormAct(24 + 20, 16)
        self.center_head = nn.Conv2d(16, 1, 1)
        nn.init.constant_(self.center_head.bias, -2.19)
        self.motion_head = nn.Conv2d(16, 2, 1)
        self.offset_head = nn.Conv2d(16, 2, 1)

    def forward(self, frames):
        if frames.ndim != 5:
            raise ValueError(f"Expected [B,T,C,H,W], got {tuple(frames.shape)}")
        b, t, c, h, w = frames.shape
        if t < 3:
            raise ValueError(f"Need T>=3 frames; got T={t}")
        if h % 8 or w % 8:
            raise ValueError(f"Spatial dimensions must be divisible by 8; got {h}x{w}")
        flat = frames.reshape(b*t, c, h, w)
        feat2 = self.stem(flat)
        feat4 = self.encoder_stride4(feat2)
        feat8 = self.encoder_stride8(feat4)
        feat16 = self.encoder_stride16(feat8)
        _, ch, eh, ew = feat16.shape
        encoded = feat16.reshape(b, t, ch, eh, ew)
        bi = self.temporal(encoded)  # [B,T,2H,eh,ew]
        ref_idx = t // 2
        temporal_mean = bi.mean(dim=1)
        temporal_max = bi.max(dim=1).values
        temporal_ref = bi[:, ref_idx]
        fused = self.temporal_fusion(torch.cat((temporal_mean, temporal_max, temporal_ref), dim=1))
        fused8 = F.interpolate(fused, size=feat8.shape[-2:], mode="bilinear", align_corners=False)
        skip8 = feat8.reshape(b, t, 28, *feat8.shape[-2:])[:, ref_idx]
        decoded = self.decoder_stride8(torch.cat((fused8, skip8), dim=1))
        decoded4 = F.interpolate(decoded, size=feat4.shape[-2:], mode="bilinear", align_corners=False)
        skip4 = feat4.reshape(b, t, 20, *feat4.shape[-2:])[:, ref_idx]
        decoded = self.decoder_stride4(torch.cat((decoded4, skip4), dim=1))
        center_logits = self.center_head(decoded)
        motion_normalized = self.motion_head(decoded)
        offsets = torch.tanh(self.offset_head(decoded)) * 0.5
        return {
            "center_logits": center_logits,
            "center_probability": torch.sigmoid(center_logits),
            "motion_normalized": motion_normalized,
            "motion_px_per_frame": motion_normalized * self.velocity_scale,
            "offsets": offsets,
            "reference_frame": ref_idx,
            "sequence_length": t,
            "input_size": (h, w),
        }

    @torch.no_grad()
    def decode_candidates(self, output: Dict, confidence_threshold=0.5, nms_radius_px=12.0):
        """Return an unbounded list of candidate tracks for each batch element.

        Each item has reference-frame (x,y), velocity (vx,vy), score, and the
        reconstructed [T,2] path. No top-K/max-target cap is applied.
        """
        probs = output["center_probability"][:, 0].float()
        motion = output["motion_px_per_frame"].float()
        b, mh, mw = probs.shape
        h, w = output["input_size"]
        # Sequence length is retained in forward output for trajectory reconstruction.
        ref = int(output["reference_frame"])
        results = []
        for bi in range(b):
            p = probs[bi]
            # First find local maxima; exact-radius greedy suppression below removes plateaus.
            radius_cells = max(1, int(round(nms_radius_px * max(mh/h, mw/w))))
            kernel = 2 * radius_cells + 1
            local = p == F.max_pool2d(p[None, None], kernel, stride=1, padding=radius_cells)[0, 0]
            ys, xs = torch.where(local & (p >= float(confidence_threshold)))
            if xs.numel():
                scores = p[ys, xs]
                order = torch.argsort(scores, descending=True)
                chosen = []
                radius2 = float(nms_radius_px) ** 2
                for oi in order.tolist():
                    xg, yg = int(xs[oi]), int(ys[oi])
                    dx = float(output["offsets"][bi, 0, yg, xg])
                    dy = float(output["offsets"][bi, 1, yg, xg])
                    x = (xg + dx) * (w - 1) / max(mw - 1, 1)
                    y = (yg + dy) * (h - 1) / max(mh - 1, 1)
                    if any((x-q["x_ref"])**2 + (y-q["y_ref"])**2 < radius2 for q in chosen):
                        continue
                    vx = float(motion[bi, 0, yg, xg])
                    vy = float(motion[bi, 1, yg, xg])
                    seq_len = int(output["sequence_length"])
                    path = [[x + vx*(frame-ref), y + vy*(frame-ref)] for frame in range(seq_len)]
                    chosen.append({
                        "x_ref": float(x), "y_ref": float(y), "vx": vx, "vy": vy,
                        "confidence": float(scores[oi]), "reference_frame": ref, "trajectory": path,
                    })
                results.append(chosen)
            else:
                results.append([])
        return results


# Friendly alias for scripts and downstream dashboard integration.
AsteroidTrackletDetector = VariableTrackletConvLSTM
