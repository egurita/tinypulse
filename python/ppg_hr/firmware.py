"""ctypes bindings to the firmware C code compiled for the host (firmware/build/native).

Used to run the exact fixed-point firmware arithmetic on the whole dataset from
Python, and for C-vs-Python equivalence tests.
"""

from __future__ import annotations

import ctypes as C
import platform
import subprocess

import numpy as np

from .data import REPO_ROOT, Subject
from .pipeline import ACC_COUNTS_PER_G, F_LO, NFFT, acc_counts, ppg_counts
from .data import FS_PPG

FW_DIR = REPO_ROOT / "firmware"
LIB_PATH = FW_DIR / "build" / "native" / f"libhr.{'dylib' if platform.system() == 'Darwin' else 'so'}"
BIN_LO = int(round(F_LO * NFFT / FS_PPG))   # 16
BPM_PER_BIN = 60 * FS_PPG / NFFT            # 1.875

Q12, Q22 = 1 << 12, 1 << 22

_lib = None


def build() -> None:
    subprocess.run(["make", "-C", str(FW_DIR), "-s", str(LIB_PATH.relative_to(FW_DIR))], check=True)


def lib() -> C.CDLL:
    global _lib
    if _lib is None:
        build()
        L = C.CDLL(str(LIB_PATH))
        P = np.ctypeslib.ndpointer
        L.hr_run.restype = C.c_size_t
        L.hr_run.argtypes = [P(np.uint32), C.c_size_t, P(np.int16), C.c_size_t, P(np.uint16), C.c_size_t]
        L.hr_bandpass_run.argtypes = [P(np.int32), C.c_size_t, P(np.int32)]
        L.hr_nlms_run.argtypes = [P(np.int32), P(np.int32), C.c_size_t, P(np.int32)]
        L.hr_spectrum_run.argtypes = [P(np.int32), P(np.uint32)]
        L.hr_log2_q16.restype = C.c_int32
        L.hr_log2_q16.argtypes = [C.c_uint64]
        _lib = L
    return _lib


def run_subject(subj: Subject) -> np.ndarray:
    """Firmware HR estimates (BPM) for every reference window of a subject."""
    ppg = np.ascontiguousarray(ppg_counts(subj).astype(np.uint32))
    acc = np.ascontiguousarray(acc_counts(subj))
    out = np.zeros(len(subj.hr_ref) + 8, dtype=np.uint16)
    n = lib().hr_run(ppg, len(ppg), acc, len(acc), out, len(out))
    assert n >= len(subj.hr_ref), (n, len(subj.hr_ref))
    return (BIN_LO + out[: len(subj.hr_ref)].astype(float)) * BPM_PER_BIN


def bandpass_q(x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.int32)
    y = np.empty_like(x)
    lib().hr_bandpass_run(x, len(x), y)
    return y


def nlms_q(d: np.ndarray, u: np.ndarray) -> np.ndarray:
    d = np.ascontiguousarray(d, dtype=np.int32)
    u = np.ascontiguousarray(u, dtype=np.int32)
    e = np.empty_like(d)
    lib().hr_nlms_run(d, u, len(d), e)
    return e


def spectrum_q(x512: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x512, dtype=np.int32)
    mag = np.empty(113, dtype=np.uint32)
    lib().hr_spectrum_run(x, mag)
    return mag


__all__ = ["run_subject", "bandpass_q", "nlms_q", "spectrum_q", "lib", "Q12", "Q22", "ACC_COUNTS_PER_G"]
