"""INT8 post-training quantization of SpecCNN + an integer-exact NumPy reference.

Scheme (same arithmetic as firmware/src/nn.c, so the two match bit for bit):
  input      int8, scale 1/127 (spectra normalised to [0, 1] per window)
  weights    int8, symmetric, one scale per output channel (BatchNorm folded in first)
  bias       int32, scale s_in * s_w[c]
  conv       int32 accumulator = sum(w_q * x_q) + b_q
  requant    y_q = clamp((acc * M0[c] + 2^(S-1)) >> S, 0, 127)   with M0/2^S ~ s_in*s_w[c]/s_out
  head       int32 logits, scale s_in * s_w; prior folded into the head bias
  tracker    log-prob (log2, Q16) = (logit * L0 + 2^(LS-1)) >> LS; the per-window
             log-sum-exp is a constant across bins, so the HMM argmax doesn't need it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .features import HR_BINS
from .nn import SpecCNN

REQ_SHIFT = 30          # requantisation multipliers are M0 / 2^30 (M < 2 always here)


@dataclass
class QLayer:
    w: np.ndarray            # int8 (cout, cin, k)
    b: np.ndarray            # int32 (cout,) or (cout, L) for the head (bias + prior)
    dilation: int
    m0: np.ndarray           # int64 (cout,) requant multiplier (relu layers)
    relu: bool


@dataclass
class QModel:
    layers: list[QLayer] = field(default_factory=list)
    logit_scale: float = 0.0          # float value of one logit LSB
    l0: int = 0                       # logit -> log2 Q16 multiplier
    ls: int = 0                       # ... and shift
    act_scales: list[float] = field(default_factory=list)


def fold(model: SpecCNN) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """[(W, b, dilation)] for the 3 conv-BN-ReLU blocks, BN folded into W, b."""
    out = []
    for blk in model.body:
        conv, bn = blk[0], blk[1]
        g = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).detach().numpy()
        w = conv.weight.detach().numpy() * g[:, None, None]
        b = (bn.bias - bn.running_mean * bn.weight / torch.sqrt(bn.running_var + bn.eps)).detach().numpy()
        out.append((w.astype(np.float64), b.astype(np.float64), conv.dilation[0]))
    return out


def conv_same(x: np.ndarray, w: np.ndarray, d: int) -> np.ndarray:
    """1D 'same' convolution (cross-correlation, like PyTorch). x (n,cin,L), w (cout,cin,k)."""
    k = w.shape[2]
    pad = d * (k - 1) // 2
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad)))
    L = x.shape[2]
    out = np.zeros((x.shape[0], w.shape[0], L), dtype=np.result_type(x, w, np.int64) if x.dtype.kind == "i" else np.float64)
    for j in range(k):
        out += np.einsum("oc,ncl->nol", w[:, :, j], xp[:, :, j * d: j * d + L])
    return out


def float_forward(model: SpecCNN, x: np.ndarray, collect=None) -> np.ndarray:
    h = x.astype(np.float64)
    for w, b, d in fold(model):
        h = np.maximum(conv_same(h, w, d) + b[None, :, None], 0)
        if collect is not None:
            collect.append(h)
    hw = model.head.weight.detach().numpy()[:, :, 0].astype(np.float64)       # (1, ch)
    return np.einsum("oc,ncl->nl", hw, h) + model.head.bias.item() + model.prior.detach().numpy()


def quantize(model: SpecCNN, x_calib: np.ndarray, pct: float = 99.99) -> QModel:
    model.eval()
    acts: list[np.ndarray] = []
    float_forward(model, x_calib, acts)
    q = QModel()
    s_in = 1 / 127
    for (w, b, d), a in zip(fold(model), acts):
        s_out = max(np.percentile(a, pct), 1e-6) / 127
        s_w = np.maximum(np.abs(w).max(axis=(1, 2)), 1e-9) / 127
        wq = np.clip(np.round(w / s_w[:, None, None]), -127, 127).astype(np.int8)
        bq = np.round(b / (s_in * s_w)).astype(np.int32)
        m0 = np.round(s_in * s_w / s_out * 2**REQ_SHIFT).astype(np.int64)
        assert m0.max() < 2**31
        q.layers.append(QLayer(wq, bq, d, m0, True))
        q.act_scales.append(float(s_out))
        s_in = s_out
    hw = model.head.weight.detach().numpy()[:, :, 0:1].astype(np.float64)        # (1, ch, 1)
    s_w = np.abs(hw).max() / 127
    wq = np.clip(np.round(hw / s_w), -127, 127).astype(np.int8)
    bias = model.head.bias.item() + model.prior.detach().numpy().astype(np.float64)
    bq = np.round(bias / (s_in * s_w)).astype(np.int32)
    q.layers.append(QLayer(wq, bq, 1, np.zeros(1, np.int64), False))
    q.logit_scale = float(s_in * s_w)
    # logit LSB (nats) -> log2 Q16:  * logit_scale / ln 2 * 2^16
    mult = q.logit_scale / np.log(2) * 65536
    q.ls = 20
    q.l0 = int(round(mult * 2**q.ls))
    assert 0 < q.l0 < 2**31
    return q


def quantize_input(spec_mag: np.ndarray) -> np.ndarray:
    """Integer version of nn.spec_input: (n, 2, 241) non-negative ints -> int8 (n, 4, 113).
    q = (v * 127 + max/2) // max per source, exactly as the firmware computes it."""
    from .nn import HARM
    v = spec_mag.astype(np.int64)
    out = np.zeros((len(v), 4, HR_BINS), np.int8)
    for src in range(2):
        mx = v[:, src].max(axis=1, keepdims=True)
        mx = np.maximum(mx, 1)
        qv = (v[:, src] * 127 + mx // 2) // mx
        out[:, 2 * src] = qv[:, :HR_BINS]
        out[:, 2 * src + 1] = qv[:, HARM]
    return out


def int_forward(q: QModel, xq: np.ndarray) -> np.ndarray:
    """Integer inference, bit-exact with firmware/src/nn.c. Returns int32 logits (n, 113)."""
    h = xq.astype(np.int64)
    for layer in q.layers[:-1]:
        acc = conv_same(h, layer.w.astype(np.int64), layer.dilation) + layer.b.astype(np.int64)[None, :, None]
        h = np.clip((acc * layer.m0[None, :, None] + (1 << (REQ_SHIFT - 1))) >> REQ_SHIFT, 0, 127)
    head = q.layers[-1]
    logits = np.einsum("oc,ncl->nl", head.w[:, :, 0].astype(np.int64), h) + head.b.astype(np.int64)[None, :]
    return logits.astype(np.int32)


def logits_to_logp_q16(q: QModel, logits: np.ndarray) -> np.ndarray:
    return ((logits.astype(np.int64) * q.l0 + (1 << (q.ls - 1))) >> q.ls).astype(np.int64)


def hmm_track_int(logp_q16: np.ndarray, max_jump=2, penalty_q16=47274, score_min=-(1 << 30)) -> np.ndarray:
    """Integer HMM tracker on log2-Q16 scores: same recursion as firmware/src/track.c."""
    out = np.empty(len(logp_q16), np.int64)
    score = None
    for k, lp in enumerate(logp_q16):
        if score is None:
            score = lp.copy()
        else:
            best = np.full_like(score, np.iinfo(np.int64).min // 4)
            for d in range(-max_jump, max_jump + 1):
                sh = np.roll(score, d) - penalty_q16 * abs(d)
                if d > 0:
                    sh[:d] = np.iinfo(np.int64).min // 4
                elif d < 0:
                    sh[d:] = np.iinfo(np.int64).min // 4
                best = np.maximum(best, sh)
            score = lp + best
        b = int(np.argmax(score))
        score = np.maximum(score - score[b], score_min)
        out[k] = b
    return out
