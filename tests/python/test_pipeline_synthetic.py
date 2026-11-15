"""End-to-end tests on synthetic recordings with a known heart rate, for both the
float pipeline and the C firmware. No dataset needed: these run everywhere."""

import numpy as np
import pytest

from ppg_hr import firmware as fw
from ppg_hr.data import FS_ACC, FS_PPG, Subject
from ppg_hr.pipeline import FINAL, NAIVE, estimate_hr

BIN_BPM = fw.BPM_PER_BIN       # 1.875 BPM: the FFT bin width, the best possible resolution
rng = np.random.default_rng(7)


def synthetic(hr_bpm, seconds=120, motion_bpm=None, motion_amp=0.0, noise=2.0):
    """PPG = pulse wave at hr_bpm (+ harmonic) + optional motion artifact that is a
    linear function of the wrist acceleration + white noise."""
    hr_bpm = np.broadcast_to(np.asarray(hr_bpm, float), (seconds * FS_PPG,))
    t = np.arange(seconds * FS_PPG) / FS_PPG
    phase = 2 * np.pi * np.cumsum(hr_bpm / 60) / FS_PPG
    ppg = 20 * np.sin(phase) + 6 * np.sin(2 * phase + 0.7) + noise * rng.standard_normal(len(t))
    ta = np.arange(seconds * FS_ACC) / FS_ACC
    acc = np.zeros((len(ta), 3))
    acc[:, 2] = 1.0                                           # gravity
    if motion_bpm:
        swing = 0.5 * np.sin(2 * np.pi * motion_bpm / 60 * ta)
        acc[:, 0] += swing
        acc[:, 1] += 0.3 * np.sin(2 * np.pi * motion_bpm / 60 * ta + 1.0)
        ppg += motion_amp * np.interp(t, ta, swing)           # artifact follows the arm
    n_win = (seconds - 8) // 2 + 1
    hr_ref = np.array([hr_bpm[k * 128: k * 128 + 512].mean() for k in range(n_win)])
    return Subject("synthetic", ppg.astype(np.float32), acc.astype(np.float32), np.zeros(1, np.float32),
                   np.zeros(1, np.int64), hr_ref.astype(np.float32), np.ones(seconds * 4, np.int8))


ESTIMATORS = {"python": lambda s: estimate_hr(s, FINAL), "c_firmware": fw.run_subject}


@pytest.mark.parametrize("impl", ESTIMATORS)
@pytest.mark.parametrize("bpm", [45, 72, 75, 120, 180])
def test_constant_hr_with_noise(impl, bpm):
    s = synthetic(bpm)
    est = ESTIMATORS[impl](s)[5:]                 # skip the first 10 s (filters settling)
    assert np.all(np.abs(est - bpm) <= BIN_BPM), (bpm, np.unique(est))


@pytest.mark.parametrize("impl", ESTIMATORS)
def test_motion_artifact_is_rejected(impl):
    # Heart 90 BPM, arm swinging at 130/min with an artifact 3x stronger than the pulse:
    # the naive "tallest peak" reports the arm; the full pipeline must report the heart.
    s = synthetic(90, motion_bpm=130, motion_amp=60)
    naive = estimate_hr(s, NAIVE)[5:]
    assert np.median(np.abs(naive - 130)) <= BIN_BPM, "test setup: the artifact should dominate"
    est = ESTIMATORS[impl](s)[5:]
    assert np.mean(np.abs(est - 90) <= BIN_BPM) > 0.95


@pytest.mark.parametrize("impl", ESTIMATORS)
def test_tracks_a_gradual_hr_increase(impl):
    seconds = 180
    hr = np.interp(np.arange(seconds * FS_PPG), [0, 60 * FS_PPG, 120 * FS_PPG, seconds * FS_PPG],
                   [70, 70, 110, 110])        # +40 BPM over one minute, like starting to climb stairs
    s = synthetic(hr, seconds=seconds)
    est = ESTIMATORS[impl](s)
    err = np.abs(est - s.hr_ref)
    centre = s.hr_ref_times()
    steady = ((centre > 10) & (centre < 56)) | (centre > 136)
    ramp = (centre >= 56) & (centre <= 136)
    assert err[steady].max() <= BIN_BPM
    # The tracker trades lag for robustness (it moves at most ~3.75 BPM per 2 s step):
    # measured worst lag on this +1.33 BPM/s ramp is ~6 BPM.
    assert err[ramp].max() <= 4 * BIN_BPM
