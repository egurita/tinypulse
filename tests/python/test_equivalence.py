"""C (fixed point, via ctypes) vs Python (float) equivalence.

Block level on synthetic signals (no dataset needed), then end to end on every subject.
Tolerances are relative errors: what fixed-point rounding is allowed to cost.
"""

import importlib.util
import tempfile
from pathlib import Path

import numpy as np
import pytest

from conftest import needs_dataset
from ppg_hr import firmware as fw
from ppg_hr.data import FS_PPG, REPO_ROOT, SUBJECTS, load_subject
from ppg_hr.dsp import bandpass, nlms, spectrum
from ppg_hr.pipeline import FINAL, NFFT, estimate_hr

pytestmark = pytest.mark.firmware
rng = np.random.default_rng(42)
BAND = slice(16, 129)


def rel_rms(err, ref):
    return float(np.std(err) / np.std(ref))


def test_generated_tables_match_python_reference():
    """firmware/gen/hr_tables.c must be what the generator produces from the current
    Python code: catches someone retuning the filter in Python but not in C."""
    spec = importlib.util.spec_from_file_location(
        "gen", REPO_ROOT / "python" / "scripts" / "gen_firmware_tables.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    with tempfile.TemporaryDirectory() as tmp:
        gen.main(Path(tmp))
        for name in ("hr_tables.c", "hr_tables.h"):
            assert (Path(tmp) / name).read_text() == (REPO_ROOT / "firmware" / "gen" / name).read_text(), name


def test_bandpass_matches_float():
    x = np.round(rng.standard_normal(64 * 120) * 50 * fw.Q12).astype(np.int64)
    yq = fw.bandpass_q(x) / fw.Q12
    yf = bandpass(x / fw.Q12, FS_PPG)
    assert rel_rms(yq - yf, yf) < 1e-3


def test_nlms_matches_float():
    n = 64 * 120
    u = bandpass(rng.standard_normal((n, 3)) * 0.3, FS_PPG, axis=0)
    d = bandpass(rng.standard_normal(n) * 20, FS_PPG) + 40 * u[:, 0]
    dq, uq = np.round(d * fw.Q12).astype(np.int32), np.round(u * fw.Q22).astype(np.int32)
    eq = fw.nlms_q(dq, uq) / fw.Q12
    ef = nlms(dq / fw.Q12, uq / fw.Q22)
    assert rel_rms(eq - ef, ef) < 1e-3


@pytest.mark.parametrize("amp", [0.01, 1.0, 100.0, 3000.0])   # block floating point: any level
def test_spectrum_matches_float_at_any_amplitude(amp):
    for _ in range(20):
        x = rng.standard_normal(512) * amp + np.sin(np.arange(512) * rng.uniform(0.05, 0.4)) * amp
        xq = np.round(x * fw.Q12)
        mf = spectrum(xq / fw.Q12, FS_PPG, NFFT)[1][BAND]
        mq = fw.spectrum_q(xq).astype(float)
        assert np.abs(mq / mq.max() - mf / mf.max()).max() < 1e-3
        assert np.argmax(mq) == np.argmax(mf)


def test_log2_q16():
    for x in [1, 2, 3, 7, 1000, 65535, 123456789, 2**33 + 1, 2**60]:
        assert abs(fw.lib().hr_log2_q16(x) / 65536 - np.log2(float(x))) < 2e-5


@needs_dataset
@pytest.mark.dataset
def test_end_to_end_fixed_equals_float_on_every_subject(metrics):
    identical, total, mae_gap = 0, 0, []
    for sid in SUBJECTS:
        s = load_subject(sid)
        f, q = estimate_hr(s, FINAL), fw.run_subject(s)
        same = np.mean(f == q)
        assert same > 0.995, f"{sid}: only {same:.2%} identical windows"
        identical += int(np.sum(f == q)); total += len(f)
        mae_gap.append(abs(np.abs(q - s.hr_ref).mean() - np.abs(f - s.hr_ref).mean()))
    metrics.record("equivalence_identical_windows", identical)
    metrics.record("equivalence_total_windows", total)
    metrics.record("equivalence_max_subject_mae_gap_bpm", max(mae_gap))
    assert max(mae_gap) < 0.05
