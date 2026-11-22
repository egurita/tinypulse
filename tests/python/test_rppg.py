"""Remote-PPG tests on synthetic data (no camera needed): POS vs green under lighting
changes, irregular frame timing, ROI geometry and skin masking, detector model file."""

import hashlib

import numpy as np
import pytest

from ppg_hr import rppg
from ppg_hr.pipeline import FINAL, estimate_signal

BIN = 1.875
rng = np.random.default_rng(3)


def synthetic_face_rgb(hr_bpm, seconds=60, fps=30.0, flicker=0.0, jitter=0.0):
    """Mean skin RGB over time: skin tone x (1 + pulse * per-channel strength) x lighting.
    Pulse strengths follow the known blood absorption pattern (G strongest, then B, R)."""
    t = np.arange(0, seconds, 1 / fps)
    t = np.sort(t + rng.uniform(-jitter, jitter, len(t))) if jitter else t
    pulse = np.sin(2 * np.pi * hr_bpm / 60 * t) + 0.3 * np.sin(4 * np.pi * hr_bpm / 60 * t + 0.6)
    tone = np.array([180.0, 120.0, 95.0])
    strength = np.array([0.33, 0.77, 0.53]) * 2e-3
    light = 1 + flicker * (np.sin(2 * np.pi * 1.1 * t) + 0.5 * np.sin(2 * np.pi * 0.37 * t))
    rgb = tone * (1 + pulse[:, None] * strength) * light[:, None]
    rgb += rng.normal(0, 0.05, rgb.shape)                  # sensor noise (8-bit average over ROI)
    return t, rgb


def hr_of(t, rgb, method):
    tu, x = rppg.resample(t, rgb)
    sig = rppg.pos(x) if method == "pos" else rppg.green(x)
    return estimate_signal(sig, None, FINAL)


@pytest.mark.parametrize("bpm", [55, 72, 96, 130])
def test_pos_recovers_hr(bpm):
    t, rgb = synthetic_face_rgb(bpm)
    est = hr_of(t, rgb, "pos")[3:]
    assert np.all(np.abs(est - bpm) <= BIN), np.unique(est)


def test_pos_cancels_lighting_that_fools_the_green_channel():
    # Brightness flicker at 1.1 Hz (66/min), 30x stronger than the pulse, hits all
    # channels equally. Green alone locks onto it; POS projects it out.
    t, rgb = synthetic_face_rgb(90, flicker=0.05)
    assert np.median(np.abs(hr_of(t, rgb, "green")[3:] - 66)) <= 2 * BIN
    assert np.mean(np.abs(hr_of(t, rgb, "pos")[3:] - 90) <= BIN) > 0.9


def test_irregular_frame_timing():
    t, rgb = synthetic_face_rgb(80, fps=29.0, jitter=0.012)   # +-12 ms jitter, like real webcams
    est = hr_of(t, rgb, "pos")[3:]
    assert np.all(np.abs(est - 80) <= BIN)


def test_resample_bridges_frames_without_face():
    t = np.arange(0, 10, 1 / 30)
    x = np.sin(t)[:, None].repeat(3, 1)
    x[100:120] = np.nan                                         # face lost for 0.66 s
    tu, xu = rppg.resample(t, x)
    assert np.all(np.isfinite(xu)) and abs(np.diff(tu).mean() - 1 / 64) < 1e-9


def test_rois_sit_on_forehead_and_cheeks():
    face = rppg.Face(box=np.array([100, 80, 200, 240.0]),
                     landmarks=np.array([[160, 170], [240, 170], [200, 220], [170, 260], [230, 260.0]]), score=0.99)
    (fx0, fy0, fx1, fy1), c1, c2 = rppg.skin_rois(face)
    assert fy1 < 170 and fy0 > 80                   # forehead: above the eyes, inside the box
    assert c1[1] > 170 and c2[1] > 170              # cheeks: below the eyes
    assert c1[2] <= 200 <= c2[0] + 30               # one cheek each side of the nose


def test_skin_mask_ignores_background():
    frame = np.zeros((100, 100, 3), np.uint8)
    frame[:, :50] = (95, 120, 180)                  # BGR skin tone
    frame[:, 50:] = (200, 60, 30)                   # blue background
    mean, n = rppg.roi_mean_rgb(frame, [(0, 0, 100, 100)])
    assert n == 100 * 50
    np.testing.assert_allclose(mean, [180, 120, 95])


def test_detector_model_file_is_the_pinned_one():
    assert hashlib.sha256(rppg.DETECTOR_PATH.read_bytes()).hexdigest() == rppg.DETECTOR_SHA256


def test_detector_finds_no_face_in_a_blank_frame():
    assert rppg.FaceTracker().detect(np.zeros((480, 640, 3), np.uint8)) is None
