"""Software-in-the-loop: the firmware receives real PPG-DaLiA samples over a serial
port at the sensor's real rate (64 Hz) and must reply with the same heart rates as
the host batch run, on time, with no protocol errors.

Targets: native build (fast, every machine) and the emulated MAX32652 (Renode).
"""

import numpy as np
import pytest

from conftest import needs_dataset, needs_renode
from ppg_hr import firmware as fw
from ppg_hr import sil
from ppg_hr.data import FS_PPG, load_subject
from ppg_hr.pipeline import acc_counts, ppg_counts

pytestmark = [pytest.mark.sil, pytest.mark.firmware, pytest.mark.dataset, needs_dataset]

SID, START_S = "S5", 828            # held-out subject, just before the stairs


def segment(seconds):
    s = load_subject(SID)
    a = START_S * FS_PPG
    a -= a % 128
    ppg = ppg_counts(s)[a:a + seconds * FS_PPG].astype(np.uint32)
    acc = acc_counts(s)[a // 2:(a + seconds * FS_PPG) // 2]
    return ppg, acc


def expected_bins(ppg, acc):
    out = np.zeros(len(ppg) // 128 + 2, dtype=np.uint16)
    n = fw.lib().hr_run(np.ascontiguousarray(ppg), len(ppg), np.ascontiguousarray(acc), len(acc), out, len(out))
    return out[:n].tolist()


def check_run(run, ppg, acc, n_frames):
    assert run.errors == []
    assert run.done == {"rx": n_frames + 1, "chk": 0, "seq": 0, "lost": 0, "est": len(run.estimates)}
    assert [b for _, b, _ in run.estimates] == expected_bins(ppg, acc)
    assert [k for k, _, _ in run.estimates] == list(range(len(run.estimates)))


def test_native_realtime_64hz(metrics):
    ppg, acc = segment(30)
    run = sil.run_native(sil.encode(ppg, acc), rate_hz=FS_PPG)
    check_run(run, ppg, acc, len(ppg))
    lat = run.latencies_ms()
    metrics.record("sil_native_latency_ms", lat.tolist())
    assert run.wall_s == pytest.approx(30, abs=1.5)       # really streamed at 64 Hz
    assert lat.max() < 50


def test_native_fast_ten_minutes():
    ppg, acc = segment(600)
    run = sil.run_native(sil.encode(ppg, acc), rate_hz=None)
    check_run(run, ppg, acc, len(ppg))


def test_native_corrupted_and_dropped_frames_are_reported():
    ppg, acc = segment(30)
    frames = sil.encode(ppg, acc)
    bad = bytearray(frames[100]); bad[4] ^= 0x10            # bit flip in the PPG payload
    frames[100] = bytes(bad)
    del frames[200]                                         # a frame lost on the line
    run = sil.run_native(frames, rate_hz=None)
    # A frame failing its checksum is also missing, so the next one is a sequence gap.
    assert run.errors == ["ERR CHK 100", "ERR SEQ 100 101", "ERR SEQ 200 201"]
    assert run.done["chk"] == 1 and run.done["seq"] == 2
    # The device keeps going: 2 fewer samples -> estimates for the samples it got.
    assert len(run.estimates) == (len(ppg) - 2 - 512) // 128 + 1


def test_native_garbage_between_frames_resyncs():
    ppg, acc = segment(20)
    frames = sil.encode(ppg, acc)
    frames.insert(50, bytes([0x00, 0xA5, 0x13, 0x37]))      # line noise incl. a fake sync byte
    run = sil.run_native(frames, rate_hz=None)
    assert len(run.estimates) == (len(ppg) - 512) // 128 + 1
    assert [b for _, b, _ in run.estimates] == expected_bins(ppg, acc)


@needs_renode
@pytest.mark.renode
def test_emulated_mcu_realtime_64hz(metrics):
    ppg, acc = segment(30)
    run = sil.run_renode(sil.encode(ppg, acc), rate_hz=FS_PPG)
    check_run(run, ppg, acc, len(ppg))
    lat = run.latencies_ms()
    metrics.record("sil_renode_latency_ms", lat.tolist())
    assert lat.max() < 500           # host wall clock, includes emulator overhead
