"""Sanity tests on the dataset itself.

These encode our ASSUMPTIONS about the data (sampling rates, units, alignment).
If an assumption is wrong, every later result is silently wrong, so we check
them first. Skipped automatically when the data hasn't been downloaded (e.g. in CI
before we add a data-caching step).
"""

import numpy as np
import pytest

from ppg_hr.data import (FS_ACC, FS_ACT, FS_ECG, FS_PPG, HR_SHIFT_S, HR_WINDOW_S,
                         available_subjects, load_subject)

SUBJECTS = available_subjects()
pytestmark = pytest.mark.skipif(not SUBJECTS, reason="PPG-DaLiA not prepared")


@pytest.fixture(scope="module", params=SUBJECTS)
def subj(request):
    return load_subject(request.param)


def test_streams_cover_same_duration(subj):
    # Different rates, same recording: all streams must span the same wall-clock time.
    durations = {
        "ppg": len(subj.ppg) / FS_PPG,
        "acc": len(subj.acc) / FS_ACC,
        "ecg": len(subj.ecg) / FS_ECG,
        "act": len(subj.activity) / FS_ACT,
    }
    ref = durations["ppg"]
    for name, d in durations.items():
        assert abs(d - ref) < 2.0, f"{name} lasts {d:.1f}s vs ppg {ref:.1f}s"


def test_label_count_matches_windowing(subj):
    # 8 s windows every 2 s -> floor((T - 8) / 2) + 1 windows (allow off-by-one at the end).
    expected = (subj.duration_s - HR_WINDOW_S) // HR_SHIFT_S + 1
    assert abs(len(subj.hr_ref) - expected) <= 1


def test_reference_hr_is_physiological(subj):
    assert 30 < subj.hr_ref.min() and subj.hr_ref.max() < 220


def test_accelerometer_is_in_g(subj):
    # Sitting still, the only acceleration is gravity -> |acc| ~ 1 g.
    # This checks the "divide raw E4 units by 64" assumption.
    sitting = subj.activity_at(np.arange(len(subj.acc)) / FS_ACC) == 1
    mag = np.linalg.norm(subj.acc[sitting], axis=1)
    assert 0.9 < np.median(mag) < 1.1


def test_rpeaks_agree_with_hr_label(subj):
    # Independent check that we understand the ground truth: the mean HR computed from
    # R-peaks (beats/minute = 60 / seconds between beats) should match the label's mean.
    rr = np.diff(subj.rpeaks) / FS_ECG
    rr = rr[(rr > 0.25) & (rr < 2.0)]  # drop obvious detection glitches
    assert abs(60 / rr.mean() - subj.hr_ref.mean()) < 5
