"""Per-window model inputs, aligned 1:1 with the reference HR windows.

time  (n_win, 4, 256) float16: bandpassed PPG + ACC x/y/z, 8 s at 32 Hz (for the time CNN)
spec  (n_win, 2, 241) float32: |FFT| bins 16..256 (0.5-8 Hz) of the NLMS-cleaned PPG and
                               of the summed ACC axes (for the hybrid spectral model).
                               Bins 129..256 hold the 2nd harmonic of every HR bin.

Both are computed with the causal Stage 1/2 front-end, so the firmware can produce the
same inputs. Cached in data/processed/features/.
"""

from __future__ import annotations

import numpy as np

from .data import FS_PPG, HR_SHIFT_S, HR_WINDOW_S, PROCESSED_DIR, Subject, load_subject
from .pipeline import FINAL, NFFT, acc_at_ppg_rate, preprocess
from .dsp import bandpass, spectrum
from .pipeline import ADC_MID, PPG_COUNTS_PER_UNIT, ppg_counts

CACHE = PROCESSED_DIR / "features"
SPEC_LO, SPEC_HI = 16, 256          # FFT bins (1.875 BPM each): 30..480 BPM
HR_BINS = 113                       # 16..128 = the HR search band, same grid as the DSP
VERSION = 1


def compute(subj: Subject) -> dict[str, np.ndarray]:
    win, hop = HR_WINDOW_S * FS_PPG, HR_SHIFT_S * FS_PPG
    acc = acc_at_ppg_rate(subj)                                   # bandpassed, 64 Hz, g
    clean = preprocess(subj, FINAL, acc)                          # bandpass + NLMS
    ppg_bp = bandpass((ppg_counts(subj) - ADC_MID) / PPG_COUNTS_PER_UNIT, FS_PPG)
    n = len(subj.hr_ref)
    time = np.empty((n, 4, win // 2), np.float16)
    spec = np.empty((n, 2, SPEC_HI - SPEC_LO + 1), np.float32)
    for k in range(n):
        s = slice(k * hop, k * hop + win)
        time[k, 0] = ppg_bp[s][::2]
        time[k, 1:] = acc[s][::2].T
        spec[k, 0] = spectrum(clean[s], FS_PPG, NFFT)[1][SPEC_LO:SPEC_HI + 1]
        spec[k, 1] = sum(spectrum(acc[s, i], FS_PPG, NFFT)[1][SPEC_LO:SPEC_HI + 1] for i in range(3))
    return {"time": time, "spec": spec}


def load(sid: str) -> dict[str, np.ndarray]:
    path = CACHE / f"{sid}_v{VERSION}.npz"
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez(path, **compute(load_subject(sid)))
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def hr_to_bin(hr_bpm: np.ndarray) -> np.ndarray:
    """Continuous bin coordinate on the 0..112 HR grid (bin b = (16 + b) * 1.875 BPM)."""
    return hr_bpm / (60 * FS_PPG / NFFT) - SPEC_LO


def bin_to_hr(b: np.ndarray) -> np.ndarray:
    return (np.asarray(b) + SPEC_LO) * (60 * FS_PPG / NFFT)
