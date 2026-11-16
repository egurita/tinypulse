"""Neural-network HR estimators (PyTorch) and their input preparation.

Both models output 113 logits on the DSP's HR grid (bin b = (16 + b) * 1.875 BPM), so
they can be decoded by argmax or fed (as log-probabilities) into the HMM tracker.

SpecCNN ("hybrid"): input = DSP front-end spectra. Fully convolutional over frequency:
    the same small detector looks at every candidate HR bin, seeing PPG and ACC energy
    at that bin and at its 2nd harmonic, plus a learned per-bin prior.
TimeCNN: input = 8 s of bandpassed PPG + ACC at 32 Hz, 4 conv blocks, global pooling.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .features import HR_BINS, hr_to_bin

HARM = 16 + 2 * np.arange(HR_BINS)        # index of the 2nd harmonic of HR bin b in spec[...,0:241]


# ---------------------------------------------------------------- inputs
def spec_input(spec: np.ndarray) -> np.ndarray:
    """(n, 2, 241) magnitudes -> (n, 4, 113): [ppg(b), ppg(2b), acc(b), acc(2b)],
    each source normalised by its own per-window max (scale-free, like the firmware's
    block floating point)."""
    ppg = spec[:, 0] / (spec[:, 0].max(axis=1, keepdims=True) + 1e-9)
    acc = spec[:, 1] / (spec[:, 1].max(axis=1, keepdims=True) + 1e-9)
    return np.stack([ppg[:, :HR_BINS], ppg[:, HARM], acc[:, :HR_BINS], acc[:, HARM]], 1).astype(np.float32)


def time_input(time: np.ndarray) -> np.ndarray:
    """Per-window, per-channel z-score: no statistics shared across windows or subjects."""
    x = time.astype(np.float32)
    x = x - x.mean(axis=2, keepdims=True)
    return x / (x.std(axis=2, keepdims=True) + 1e-6)


def soft_target(hr_bpm: np.ndarray, sigma_bins: float = 1.5) -> np.ndarray:
    b = hr_to_bin(hr_bpm)[:, None]
    t = np.exp(-0.5 * ((np.arange(HR_BINS)[None] - b) / sigma_bins) ** 2)
    return (t / t.sum(axis=1, keepdims=True)).astype(np.float32)


# ---------------------------------------------------------------- models
def conv_bn(cin, cout, k, dilation=1):
    return nn.Sequential(nn.Conv1d(cin, cout, k, padding=dilation * (k - 1) // 2, dilation=dilation, bias=False),
                         nn.BatchNorm1d(cout), nn.ReLU())


class SpecCNN(nn.Module):
    def __init__(self, ch=16):
        super().__init__()
        self.body = nn.Sequential(conv_bn(4, ch, 5, 1), conv_bn(ch, ch, 5, 2), conv_bn(ch, ch, 5, 4))
        self.head = nn.Conv1d(ch, 1, 1)
        self.prior = nn.Parameter(torch.zeros(HR_BINS))

    def forward(self, x):
        return self.head(self.body(x))[:, 0] + self.prior


class TimeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            conv_bn(4, 16, 7), nn.MaxPool1d(2),
            conv_bn(16, 32, 7), nn.MaxPool1d(2),
            conv_bn(32, 32, 7), nn.MaxPool1d(2),
            conv_bn(32, 64, 7), nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.head = nn.Linear(64, HR_BINS)

    def forward(self, x):
        return self.head(self.body(x))


ARCHS = {"spec": SpecCNN, "time": TimeCNN}


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def macs(model: nn.Module, x: torch.Tensor) -> int:
    """Multiply-accumulates per inference (convolutions and linear layers)."""
    total = 0

    def hook(m, inp, out):
        nonlocal total
        if isinstance(m, nn.Conv1d):
            total += out.shape[-1] * m.out_channels * m.in_channels * m.kernel_size[0]
        elif isinstance(m, nn.Linear):
            total += m.in_features * m.out_features
    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv1d, nn.Linear))]
    model.eval()
    with torch.no_grad():
        model(x[:1])
    for h in hs:
        h.remove()
    return total
