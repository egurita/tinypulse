"""Loading PPG-DaLiA.

Why this module converts the data instead of just reading the original files:

1. The dataset ships as Python *pickle* files (S1.pkl ... S15.pkl). Unpickling can
   run arbitrary code, so you should only unpickle files you trust. We do it ONCE,
   for a file downloaded from the official UCI source, and save the result as .npz
   (plain numeric arrays, loaded with allow_pickle=False, so it can't run code).
2. Each pickle is ~1 GB because it holds every chest sensor at 700 Hz (EMG, EDA,
   respiration...). We only keep what this project uses: wrist PPG, wrist
   accelerometer, chest ECG and the labels. That brings each subject down to a
   few tens of MB, which also makes every later experiment load in well under a second.

The streams have DIFFERENT sampling rates. That's normal for embedded systems
(each sensor runs at its own rate), and it's the first trap: you can never compare
two arrays index-by-index; you always go through time = index / fs.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

# Sampling rates (Hz), from the PPG-DaLiA readme.
FS_PPG = 64     # Empatica E4 wrist "BVP" channel = the PPG signal
FS_ACC = 32     # E4 wrist accelerometer
FS_ECG = 700    # RespiBAN chest ECG (the "truth" the HR labels come from)
FS_ACT = 4      # activity label stream

# Ground-truth heart rate: one value per 8 s window, windows shifted by 2 s.
# We'll use exactly this windowing for our own estimates so they line up 1:1.
HR_WINDOW_S = 8
HR_SHIFT_S = 2

ACTIVITY_NAMES = {
    0: "transient",     # moving between activities, not a real activity
    1: "sitting",
    2: "stairs",
    3: "table soccer",
    4: "cycling",
    5: "driving",
    6: "lunch",
    7: "walking",
    8: "working",
}

SUBJECTS = [f"S{i}" for i in range(1, 16)]


@dataclass
class Subject:
    sid: str
    ppg: np.ndarray        # (N_ppg,)     raw PPG, arbitrary units, 64 Hz
    acc: np.ndarray        # (N_acc, 3)   wrist acceleration in g (x, y, z), 32 Hz
    ecg: np.ndarray        # (N_ecg,)     chest ECG, mV, 700 Hz
    rpeaks: np.ndarray     # sample indices (in the 700 Hz ECG) of each heartbeat
    hr_ref: np.ndarray     # (N_win,)     reference HR in BPM, one per 8 s window / 2 s shift
    activity: np.ndarray   # (N_act,)     activity id, 4 Hz

    @property
    def duration_s(self) -> float:
        return len(self.ppg) / FS_PPG

    def activity_at(self, t_s: np.ndarray | float) -> np.ndarray:
        """Activity id at time(s) t, looked up in the 4 Hz activity stream."""
        idx = np.clip((np.asarray(t_s) * FS_ACT).astype(int), 0, len(self.activity) - 1)
        return self.activity[idx]

    def hr_ref_times(self) -> np.ndarray:
        """Center time (s) of each reference-HR window: window k covers [2k, 2k+8) s."""
        return np.arange(len(self.hr_ref)) * HR_SHIFT_S + HR_WINDOW_S / 2


def convert_pickle(f: BinaryIO, out_path: Path) -> None:
    """One-time conversion of a trusted S*.pkl (open binary stream) into a compact .npz."""
    # The pickles were written by Python 2; latin1 decodes their byte strings correctly.
    d = pickle.load(f, encoding="latin1")

    wrist, chest = d["signal"]["wrist"], d["signal"]["chest"]
    np.savez(
        out_path,
        ppg=wrist["BVP"].ravel().astype(np.float32),
        # Already in g in the pickle. (The raw E4 CSVs use 1/64 g; dividing again was a bug
        # caught by tests/python/test_data.py::test_accelerometer_is_in_g: at rest |acc| = 1 g.)
        acc=wrist["ACC"].astype(np.float32),
        ecg=chest["ECG"].ravel().astype(np.float32),
        rpeaks=np.asarray(d["rpeaks"], dtype=np.int64),
        hr_ref=np.asarray(d["label"], dtype=np.float32),
        activity=d["activity"].ravel().astype(np.int8),
    )


def load_subject(sid: str, root: Path = PROCESSED_DIR) -> Subject:
    path = root / f"{sid}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run: uv run python python/scripts/prepare_data.py")
    with np.load(path, allow_pickle=False) as z:
        return Subject(sid=sid, **{k: z[k] for k in z.files})


def available_subjects(root: Path = PROCESSED_DIR) -> list[str]:
    return [s for s in SUBJECTS if (root / f"{s}.npz").exists()]


def activity_runs(activity: np.ndarray):
    """Contiguous blocks of the same activity -> list of (activity_id, start_s, end_s)."""
    change = np.flatnonzero(np.diff(activity)) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(activity)]
    return [(int(activity[s]), s / FS_ACT, e / FS_ACT) for s, e in zip(starts, ends)]


def middle_of(runs, act_id: int, length_s: float) -> float:
    """Start time of a `length_s` segment in the middle of the longest block of `act_id`."""
    act, s, e = max((r for r in runs if r[0] == act_id), key=lambda r: r[2] - r[1])
    return (s + e) / 2 - length_s / 2


