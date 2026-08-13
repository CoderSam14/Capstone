"""
V1-prototype model.

Input:
    [B, 4, 1, 256, 256] signed BLIP frames

Output:
    K spatial heatmaps
    K start coordinates
    K velocity vectors
    K presence logits
    K reconstructed trajectories

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import *


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x, state=None):
        b, _, h, w = x.shape

        if state is None:
            hidden = torch.zeros(
                b, self.hidden_channels, h, w,
                device=x.device, dtype=x.dtype
            )
            cell = torch.zeros_like(hidden)
        else:
            hidden, cell = state

        i, f, o, g = torch.chunk(
            self.gates(torch.cat([x, hidden], dim=1)),
            4,
            dim=1,
        )

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        cell = f * cell + i * g
        hidden = o * torch.tanh(cell)

        return hidden, cell


def heatmap_to_xy(logits):
    
    b, k, h, w = logits.shape

    probs = F.softmax(
        logits.flatten(2),
        dim=-1,
    ).reshape(b, k, h, w)

    xs = torch.linspace(
        0, IMAGE_SIZE - 1, w,
        device=logits.device,
        dtype=logits.dtype,
    )
    ys = torch.linspace(
        0, IMAGE_SIZE - 1, h,
        device=logits.device,
        dtype=logits.dtype,
    )

    px = probs.sum(dim=2)
    py = probs.sum(dim=3)

    x = (px * xs).sum(dim=-1)
    y = (py * ys).sum(dim=-1)

    return torch.stack([x, y], dim=-1)


def heatmap_to_xy_hard(logits):
    
    b, k, h, w = logits.shape

    q = logits.flatten(2).argmax(dim=-1)
    y = q // w
    x = q % w

    return torch.stack([
        x.float() * (IMAGE_SIZE - 1) / max(w - 1, 1),
        y.float() * (IMAGE_SIZE - 1) / max(h - 1, 1),
    ], dim=-1)


class V5MultiTargetDetector(nn.Module):
    def __init__(self, max_targets=MAX_TARGETS):
        super().__init__()

        self.max_targets = max_targets

        # 256 -> 128 -> 64 spatial representation.
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 24, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(24, 48, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(48, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.forward_lstm = ConvLSTMCell(64, 64)
        self.backward_lstm = ConvLSTMCell(64, 64)

        self.temporal_fusion = nn.Sequential(
            nn.Conv2d(128, 96, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )

       
        self.fusion = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        
        self.start_head = nn.Conv2d(
            64,
            max_targets,
            kernel_size=1,
        )

        
        self.presence_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

        
        self.velocity_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )

    def spatial_slot_pool(self, feature_map, heatmaps):
        
        b, c, h, w = feature_map.shape
        k = heatmaps.shape[1]

        attention = F.softmax(
            heatmaps.flatten(2),
            dim=-1,
        )

        features = feature_map.flatten(2)

        return torch.einsum(
            "bkn,bcn->bkc",
            attention,
            features,
        )

    def forward(self, x, use_hard_start=False):
        # x = [B,T,1,H,W]
        t = x.shape[1]

        encoded = [
            self.encoder(x[:, i])
            for i in range(t)
        ]

        forward = []
        state = None

        for i in range(t):
            state = self.forward_lstm(
                encoded[i],
                state,
            )
            forward.append(state[0])

        backward = [None] * t
        state = None

        for i in reversed(range(t)):
            state = self.backward_lstm(
                encoded[i],
                state,
            )
            backward[i] = state[0]

        temporal = [
            self.temporal_fusion(
                torch.cat([forward[i], backward[i]], dim=1)
            )
            for i in range(t)
        ]

        fused = self.fusion(
            torch.cat(temporal, dim=1)
        )

        start_logits = self.start_head(fused)

        if use_hard_start:
            start = heatmap_to_xy_hard(start_logits)
        else:
            start = heatmap_to_xy(start_logits)

        slot_features = self.spatial_slot_pool(
            fused,
            start_logits,
        )

        velocity = torch.tanh(
            self.velocity_head(slot_features)
        ) * (VELOCITY_RANGE[1] * 1.25)

        presence_logits = self.presence_head(
            slot_features
        ).squeeze(-1)

        times = torch.arange(
            t,
            device=x.device,
            dtype=x.dtype,
        )[None, None, :, None]

        coords = (
            start[:, :, None, :]
            + times * velocity[:, :, None, :]
        )

        return {
            "start_logits": start_logits,
            "start": start,
            "velocity": velocity,
            "coords": coords,
            "presence_logits": presence_logits,
            "slot_features": slot_features,
        }
