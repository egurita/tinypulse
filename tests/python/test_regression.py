"""Dataset regression tests: the final pipeline must stay under per-activity error limits
on the held-out subjects S4-S15. Limits = Stage 1 result (2026-11-15) + ~10% margin, so any
change that makes the algorithm noticeably worse fails CI.
"""

import pytest

from ppg_hr.data import SUBJECTS, available_subjects, load_subject
from ppg_hr.evaluate import mae_by_activity, mae_subject_mean, run
from ppg_hr.pipeline import DEV_SUBJECTS, FINAL, FINAL_NO_LMS, NAIVE

pytestmark = pytest.mark.skipif(len(available_subjects()) < 15, reason="PPG-DaLiA not prepared")
TEST = [s for s in SUBJECTS if s not in DEV_SUBJECTS]

LIMITS_BPM = {  # measured -> limit
    "sitting": 3.0,        # 2.4
    "stairs": 30.0,        # 27.2
    "table soccer": 27.0,  # 24.2
    "cycling": 11.5,       # 10.2
    "driving": 6.0,        # 5.3
    "lunch": 5.5,          # 5.0
    "walking": 22.0,       # 19.9
    "working": 4.2,        # 3.7
    "all windows": 10.8,   # 9.8
}


@pytest.fixture(scope="module")
def results():
    subjects = [load_subject(s) for s in TEST]
    return {cfg.name: run(subjects, cfg) for cfg in (NAIVE, FINAL, FINAL_NO_LMS)}


@pytest.mark.parametrize("activity", LIMITS_BPM)
def test_final_pipeline_within_limit(results, activity):
    mae = mae_by_activity(results[FINAL.name], TEST)[activity]
    assert mae <= LIMITS_BPM[activity], f"{activity}: {mae:.2f} BPM > {LIMITS_BPM[activity]}"


def test_final_beats_naive_on_every_activity(results):
    naive = mae_by_activity(results[NAIVE.name], TEST)
    final = mae_by_activity(results[FINAL.name], TEST)
    worse = {a: (naive[a], final[a]) for a in naive if final[a] >= naive[a]}
    assert not worse, worse


def test_motion_compensation_helps(results):
    # The NLMS block must earn its place: removing it should make the test error worse.
    assert mae_subject_mean(results[FINAL.name], TEST) < mae_subject_mean(results[FINAL_NO_LMS.name], TEST)
