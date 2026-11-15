"""Stage 1 heart-rate pipeline: PPG + ACC -> one HR estimate per 8 s window (2 s shift).

    ppg, acc --[bandpass]--> [NLMS motion cancel] --> 8 s windows --> spectrum
             --[ACC spectral mask]--> [HMM tracker | tallest peak] --> BPM

Every block is switchable through Config, so each improvement is measured alone.
Parameters were tuned on development subjects S1-S3 only (see DEV_SUBJECTS).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import FS_PPG, HR_SHIFT_S, HR_WINDOW_S, Subject
from .dsp import acc_mask, bandpass, hmm_track_step, nlms, spectrum

NFFT = 2048            # zero-padded FFT: 64 Hz / 2048 = 0.03125 Hz = ~1.9 BPM per bin
F_LO, F_HI = 0.5, 4.0  # 30-240 BPM
DEV_SUBJECTS = ("S1", "S2", "S3")

# Sensor integer formats shared with the firmware (see firmware/include/hr_config.h).
PPG_COUNTS_PER_UNIT = 16      # dataset PPG unit -> MAX30102-style 18-bit ADC counts
ADC_MID = 1 << 17             # mid-scale of the 18-bit ADC (dataset PPG is zero-centred)
ADC_MAX = (1 << 18) - 1
ACC_COUNTS_PER_G = 64         # accelerometer LSB = 1/64 g (exact for PPG-DaLiA's E4 data)


def ppg_counts(subj: Subject) -> np.ndarray:
    """PPG as the 18-bit unsigned integers a MAX30102 would deliver."""
    c = np.round(subj.ppg.astype(np.float64) * PPG_COUNTS_PER_UNIT) + ADC_MID
    return np.clip(c, 0, ADC_MAX).astype(np.int32)


def acc_counts(subj: Subject) -> np.ndarray:
    """Accelerometer as int16 counts of 1/64 g, (N_acc, 3) at 32 Hz."""
    return np.round(subj.acc.astype(np.float64) * ACC_COUNTS_PER_G).astype(np.int16)


@dataclass(frozen=True)
class Config:
    name: str
    bandpass: bool = False
    lms: bool = False
    lms_taps: int = 16
    lms_mu: float = 0.02
    lms_eps: float = 0.1
    acc_mask: float = 0.0          # 0 = off
    hmm: bool = False
    hmm_max_jump: int = 2          # bins per 2 s step (~3.8 BPM)
    hmm_penalty: float = 0.5


# Ablation ladder: each row adds one block to the previous one.
NAIVE = Config("1 naive (raw FFT peak)")
BANDPASS = Config("2 + bandpass", bandpass=True)
LMS = Config("3 + NLMS motion cancel", bandpass=True, lms=True)
HMM = Config("4 + HMM tracker", bandpass=True, lms=True, hmm=True)
FINAL = Config("5 + ACC spectral mask (final)", bandpass=True, lms=True, hmm=True, acc_mask=0.7)
FINAL_NO_LMS = Config("final without NLMS", bandpass=True, hmm=True, acc_mask=0.7)
CONFIGS = [NAIVE, BANDPASS, LMS, HMM, FINAL, FINAL_NO_LMS]


def upsample_acc(acc: np.ndarray) -> np.ndarray:
    """32 -> 64 Hz, CAUSAL linear interpolation (what the firmware does sample by sample).

    A new ACC sample a[k] arrives with PPG sample 2k. Output 2k is the midpoint of
    a[k-1] and a[k], output 2k+1 repeats a[k]: linear interpolation delayed by one
    64 Hz sample, so it never needs a future sample. (Stage 1 originally used
    scipy's resample_poly, which is non-causal; switching changed test MAE 9.79 -> 9.73.)
    """
    a = np.asarray(acc, dtype=np.float64)
    prev = np.vstack([a[:1], a[:-1]])
    up = np.empty((2 * len(a), a.shape[1]))
    up[0::2] = (prev + a) / 2
    up[1::2] = a
    return up


def acc_at_ppg_rate(subj: Subject) -> np.ndarray:
    """Bandpassed accelerometer in g, at 64 Hz, aligned with the PPG."""
    acc = upsample_acc(acc_counts(subj) / ACC_COUNTS_PER_G)[: len(subj.ppg)]
    return bandpass(acc, FS_PPG, axis=0)


def preprocess(subj: Subject, cfg: Config, acc: np.ndarray | None = None) -> np.ndarray:
    """Causal, streaming-equivalent preprocessing of the whole recording -> PPG at 64 Hz."""
    ppg = (ppg_counts(subj) - ADC_MID) / PPG_COUNTS_PER_UNIT
    if cfg.bandpass:
        ppg = bandpass(ppg, FS_PPG)
    if cfg.lms:
        acc = acc_at_ppg_rate(subj) if acc is None else acc
        ppg = nlms(ppg, acc, cfg.lms_taps, cfg.lms_mu, cfg.lms_eps)
    return ppg


def estimate_hr(subj: Subject, cfg: Config) -> np.ndarray:
    """One BPM estimate per reference window (same length as subj.hr_ref)."""
    acc = acc_at_ppg_rate(subj) if (cfg.lms or cfg.acc_mask) else None
    ppg = preprocess(subj, cfg, acc)
    freqs = np.fft.rfftfreq(NFFT, 1 / FS_PPG)
    band = (freqs >= F_LO) & (freqs <= F_HI)
    win, hop = HR_WINDOW_S * FS_PPG, HR_SHIFT_S * FS_PPG

    out = np.empty(len(subj.hr_ref))
    score = None
    for k in range(len(out)):
        s = slice(k * hop, k * hop + win)
        mag = spectrum(ppg[s], FS_PPG, NFFT)[1][band]
        if cfg.acc_mask:
            acc_mag = sum(spectrum(acc[s, i], FS_PPG, NFFT)[1][band] for i in range(3))
            mag = acc_mask(mag, acc_mag, cfg.acc_mask)
        if cfg.hmm:
            score = hmm_track_step(score, mag, cfg.hmm_max_jump, cfg.hmm_penalty)
            out[k] = 60 * freqs[band][np.argmax(score)]
        else:
            out[k] = 60 * freqs[band][np.argmax(mag)]
    return out
