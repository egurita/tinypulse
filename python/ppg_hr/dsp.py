"""Classical DSP building blocks for heart-rate estimation."""

from __future__ import annotations

import numpy as np


def spectrum(x: np.ndarray, fs: float, nfft: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Magnitude spectrum of a real signal segment.

    Returns (freqs_hz, magnitude). Three practical details on top of a bare FFT:

    1. Remove the mean. A constant offset (the sensor's baseline, gravity on the
       accelerometer) is a "0 Hz rhythm". It's usually far larger than the pulse and
       would produce a giant peak at 0 Hz that dwarfs everything we care about.

    2. Multiply by a Hann window (a smooth hump that goes to zero at both ends).
       The FFT implicitly assumes the segment repeats forever. If the segment's end
       doesn't line up with its start, that "wrap-around" is a sudden jump, and a jump
       contains every frequency, so energy smears into neighbouring bins
       ("spectral leakage"). Tapering both ends to zero removes the jump.
       Cost: peaks get slightly wider. Almost always worth it.

    3. Zero-padding (nfft > len(x)). Appending zeros gives MORE POINTS on the frequency
       axis, so the plot is smoother and the peak location less coarse. It does NOT
       add information: the true resolution is still 1 / segment_duration Hz.
       Two rhythms closer than that can't be separated, no matter how much you pad.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    x = x * np.hanning(len(x))
    nfft = nfft or len(x)
    # rfft: the input is real, so the spectrum is mirror-symmetric; only keep the
    # non-negative half (0 .. fs/2). fs/2 is the Nyquist frequency: the highest
    # frequency a signal sampled at fs can represent.
    mag = np.abs(np.fft.rfft(x, n=nfft))
    freqs = np.fft.rfftfreq(nfft, d=1 / fs)
    return freqs, mag


def dominant_frequency(x: np.ndarray, fs: float, fmin: float = 0.5, fmax: float = 4.0,
                       nfft: int = 4096) -> float:
    """Frequency (Hz) of the tallest spectral peak inside [fmin, fmax].

    The simplest possible HR estimator: HR_bpm = 60 * dominant_frequency(ppg).
    Known weakness (we'll see it on the stairs data): during motion, the tallest
    peak may be the step rate, not the heart.
    """
    freqs, mag = spectrum(x, fs, nfft)
    band = (freqs >= fmin) & (freqs <= fmax)
    return float(freqs[band][np.argmax(mag[band])])


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def bandpass_sos(fs: float, lo: float = 0.5, hi: float = 4.0, order: int = 4) -> np.ndarray:
    """Butterworth bandpass as second-order sections (numerically safer than (b, a)).

    The SOS coefficients are what the firmware will implement as a cascade of biquads.
    """
    from scipy.signal import butter
    return butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")


def bandpass(x: np.ndarray, fs: float, lo: float = 0.5, hi: float = 4.0, order: int = 4,
             axis: int = 0) -> np.ndarray:
    """CAUSAL bandpass (sosfilt, not filtfilt): exactly what a streaming MCU can do.

    Causal filtering delays the signal (phase delay), but the HR estimate only uses
    the spectrum magnitude, which a delay doesn't change.
    """
    from scipy.signal import sosfilt
    return sosfilt(bandpass_sos(fs, lo, hi, order), np.asarray(x, dtype=np.float64), axis=axis)


def nlms(desired: np.ndarray, reference: np.ndarray, taps: int = 16, mu: float = 0.02,
         eps: float = 0.1) -> np.ndarray:
    """Normalized LMS adaptive noise canceller, sample by sample.

    desired:   (N,)   PPG = heart + motion artifact
    reference: (N, C) accelerometer channels = motion only
    Learns FIR weights w (taps per channel) so that w . recent_reference ~= the motion
    part of `desired`; returns the error e = desired - estimate, i.e. the cleaned PPG.
    Written as a plain per-sample loop on purpose: it mirrors the firmware version.

    eps matters: at rest the bandpassed ACC is ~0, so u.u ~ 0. With a tiny eps the
    normalized step mu/(eps+u.u) explodes and the filter "adapts" to sensor noise,
    distorting the PPG (measured: eps=1e-6 made MAE worse than no LMS at all).
    eps=0.1 (in g^2) keeps adaptation off when there's no real motion.
    """
    d = np.asarray(desired, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    if r.ndim == 1:
        r = r[:, None]
    n, c = r.shape
    # padded[t : t+taps] holds the last `taps` samples of every channel, newest first.
    padded = np.vstack([np.zeros((taps - 1, c)), r])
    w = np.zeros(taps * c)
    out = np.empty(n)
    for t in range(n):
        u = padded[t: t + taps][::-1].ravel()
        e = d[t] - w @ u
        w += (mu / (eps + u @ u)) * e * u
        out[t] = e
    return out


# ---------------------------------------------------------------------------
# Peak picking
# ---------------------------------------------------------------------------

def acc_mask(ppg_mag: np.ndarray, acc_mag: np.ndarray, strength: float) -> np.ndarray:
    """Attenuate PPG spectrum bins where the accelerometer has energy.

    acc_mag = summed |spectrum| of the ACC axes for the same window. A bin where ACC
    is at its maximum is scaled by (1 - strength); bins with no motion are untouched.
    """
    return ppg_mag * np.clip(1 - strength * acc_mag / (acc_mag.max() + 1e-12), 0, 1)

# Note: a simpler tracker ("tallest peak within +-N BPM of the previous estimate")
# was tried first and made results WORSE (dev MAE 15.0 -> 16.5-21 BPM): one bad
# window locks it onto a motion peak forever. hmm_track_step below fixes that.


def hmm_track_step(score: np.ndarray | None, mag: np.ndarray, max_jump: int,
                   penalty: float) -> np.ndarray:
    """One causal step of a Viterbi-style HR tracker over spectrum bins.

    score[j] = best accumulated log-evidence of any HR path ending in bin j.
      new[j] = log p[j] + max_{|i-j| <= max_jump} (score[i] - penalty * |i-j|)
    where p = this window's normalized spectrum. Unlike "search near the previous
    estimate", this keeps every candidate path alive, so one bad window (a motion
    peak) can't permanently lock the tracker onto the wrong frequency.
    O(bins * max_jump) per window -> cheap enough for an MCU.
    """
    logp = np.log(mag / (mag.sum() + 1e-12) + 1e-12)
    if score is None:
        return logp
    best = np.full_like(score, -np.inf)
    for d in range(-max_jump, max_jump + 1):
        shifted = np.roll(score, d)
        if d > 0:
            shifted[:d] = -np.inf
        elif d < 0:
            shifted[d:] = -np.inf
        best = np.maximum(best, shifted - penalty * abs(d))
    new = logp + best
    return new - new.max()   # renormalize so scores don't drift to -inf over hours
