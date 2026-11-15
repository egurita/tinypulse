"""Unit tests on synthetic signals: we construct the input, so we know the right answer."""

import numpy as np
import pytest

from ppg_hr.dsp import bandpass, dominant_frequency, hmm_track_step, nlms, spectrum

FS = 64
rng = np.random.default_rng(0)  # fixed seed -> the test is deterministic, never flaky


def sine(bpm, seconds=8, amp=1.0):
    t = np.arange(int(seconds * FS)) / FS
    return amp * np.sin(2 * np.pi * (bpm / 60) * t)


@pytest.mark.parametrize("bpm", [45, 72, 120, 180])
def test_clean_sine_recovers_bpm(bpm):
    assert 60 * dominant_frequency(sine(bpm), FS) == pytest.approx(bpm, abs=1)


def test_72bpm_with_noise_and_offset():
    # Big DC offset + noise, like a real sensor. The mean removal and band limits must cope.
    x = 500 + sine(72) + 0.5 * rng.standard_normal(8 * FS)
    assert 60 * dominant_frequency(x, FS) == pytest.approx(72, abs=1)


def test_tallest_peak_wins_even_if_it_is_motion():
    # Documents the WEAKNESS of the naive estimator: a stronger 132 "steps/min"
    # component beats a 105 BPM heartbeat. Motion compensation must fix this later.
    x = sine(105, amp=1.0) + sine(132, amp=2.0)
    assert 60 * dominant_frequency(x, FS) == pytest.approx(132, abs=1)


def test_spectrum_axis():
    freqs, mag = spectrum(np.zeros(8 * FS), FS)
    assert freqs[0] == 0 and freqs[-1] == FS / 2   # 0 .. Nyquist
    assert len(freqs) == len(mag) == 8 * FS // 2 + 1


# ---- bandpass -----------------------------------------------------------------


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


@pytest.mark.parametrize("f_hz, passes", [(0.1, False), (1.2, True), (3.0, True), (10.0, False)])
def test_bandpass_passes_hr_band_only(f_hz, passes):
    t = np.arange(60 * FS) / FS
    y = bandpass(np.sin(2 * np.pi * f_hz * t), FS)[10 * FS:]   # skip filter start-up transient
    gain = rms(y) / rms(np.sin(2 * np.pi * f_hz * t))
    assert (gain > 0.9) if passes else (gain < 0.1)


def test_bandpass_removes_dc_offset():
    y = bandpass(1000 + sine(72, seconds=60), FS)[10 * FS:]
    assert abs(y.mean()) < 0.05


# ---- NLMS -----------------------------------------------------------------------
def test_nlms_cancels_motion_correlated_with_reference():
    n = 120 * FS
    motion = bandpass(rng.standard_normal(n), FS)          # broadband in-band "wrist motion"
    heart = 0.5 * sine(75, seconds=120)
    artifact = np.convolve(motion, [0.0, 3.0, -1.5, 0.5])[:n]  # unknown linear path to the PPG
    out = nlms(heart + artifact, motion, taps=8)
    tail = slice(60 * FS, None)                             # after convergence
    assert rms(out[tail] - heart[tail]) < 0.2 * rms(artifact[tail])


def test_nlms_leaves_ppg_alone_without_motion():
    heart = sine(75, seconds=30)
    out = nlms(heart, np.zeros((len(heart), 3)))
    np.testing.assert_allclose(out, heart)


# ---- HMM tracker --------------------------------------------------------------
def test_hmm_ignores_a_single_outlier_window():
    bins = 100
    peak = lambda j: np.exp(-0.5 * ((np.arange(bins) - j) / 1.5) ** 2) + 0.01
    track = [40] * 10 + [80] + [40] * 3          # one window where motion wins
    score, picks = None, []
    for j in track:
        score = hmm_track_step(score, peak(j), max_jump=2, penalty=0.5)
        picks.append(int(np.argmax(score)))
    assert all(p == 40 for p in picks)


def test_hmm_follows_a_gradual_change():
    bins = 100
    peak = lambda j: np.exp(-0.5 * ((np.arange(bins) - j) / 1.5) ** 2) + 0.01
    score = None
    for j in range(30, 60):                       # +1 bin per window: a real HR ramp
        score = hmm_track_step(score, peak(j), max_jump=2, penalty=0.5)
    assert abs(int(np.argmax(score)) - 59) <= 1
