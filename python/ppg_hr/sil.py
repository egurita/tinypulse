"""Software-in-the-loop harness: stream sensor frames over a serial port to the firmware.

Two targets, same protocol (firmware/include/sil_app.h):
  native  - firmware/build/native/hr_sil, attached to a pseudo-terminal (pty)
  renode  - firmware/build/arm/fw_sil.elf in the emulated MAX32652, UART0 bridged to a pty

The harness paces frames at the sensor rate (64 Hz, or faster for quick runs),
timestamps every sent frame and every received line, so tests can check both the
values and the latency of each heart-rate estimate.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import threading
import time
import tty
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .data import REPO_ROOT
from .emulator import renode_binary

FW = REPO_ROOT / "firmware"
NATIVE_BIN = FW / "build" / "native" / "hr_sil"
SIL_ELF = FW / "build" / "arm" / "fw_sil.elf"
SYNC, FLAG_ACC, FLAG_END = 0xA5, 0x01, 0x80


def frame(seq: int, ppg: int, acc=None, flags: int = 0) -> bytes:
    if acc is not None:
        flags |= FLAG_ACC
    ax, ay, az = (int(v) for v in (acc if acc is not None else (0, 0, 0)))
    body = bytes([seq & 0xFF, flags]) + int(ppg).to_bytes(3, "little") + b"".join(
        int(v).to_bytes(2, "little", signed=True) for v in (ax, ay, az))
    chk = 0
    for b in body:
        chk ^= b
    return bytes([SYNC]) + body + bytes([chk])


def encode(ppg: np.ndarray, acc: np.ndarray) -> list[bytes]:
    """One frame per PPG sample; the ACC sample k rides with PPG sample 2k (32 Hz)."""
    return [frame(i, int(p), acc[i // 2] if i % 2 == 0 and i // 2 < len(acc) else None)
            for i, p in enumerate(ppg)]


def end_frame(seq: int) -> bytes:
    return frame(seq, 0, None, FLAG_END)


@dataclass
class SilRun:
    lines: list[tuple[float, str]] = field(default_factory=list)   # (host time, line)
    sent: list[float] = field(default_factory=list)                 # host time per frame
    wall_s: float = 0.0

    @property
    def estimates(self) -> list[tuple[int, int, int]]:
        return [tuple(map(int, l.split()[1:])) for _, l in self.lines if l.startswith("HR ")]

    @property
    def errors(self) -> list[str]:
        return [l for _, l in self.lines if l.startswith("ERR")]

    @property
    def done(self) -> dict[str, int]:
        for _, l in self.lines:
            if l.startswith("DONE"):
                return {k: int(v) for k, v in re.findall(r"(\w+)=(\d+)", l)}
        return {}

    def latencies_ms(self, win: int = 512, hop: int = 128) -> np.ndarray:
        """Time from sending the last sample of window k to receiving HR k."""
        out = []
        for t, l in self.lines:
            if l.startswith("HR "):
                k = int(l.split()[1])
                idx = win - 1 + hop * k
                if idx < len(self.sent):
                    out.append((t - self.sent[idx]) * 1e3)
        return np.array(out)


class _Link:
    """Raw serial link over a pty master fd, with a background line reader."""

    def __init__(self, fd: int):
        self.fd, self.run, self.buf = fd, SilRun(), b""
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._reader, daemon=True)
        self._t.start()

    def _reader(self):
        while not self._stop.is_set():
            try:
                chunk = os.read(self.fd, 4096)
            except OSError:
                return
            if not chunk:
                return
            now = time.perf_counter()
            self.buf += chunk
            while b"\n" in self.buf:
                line, self.buf = self.buf.split(b"\n", 1)
                self.run.lines.append((now, line.decode(errors="replace").strip()))

    def wait_for(self, prefix: str, timeout: float) -> bool:
        end = time.perf_counter() + timeout
        while time.perf_counter() < end:
            if any(l.startswith(prefix) for _, l in self.run.lines):
                return True
            time.sleep(0.01)
        return False

    def stream(self, frames: list[bytes], rate_hz: float | None):
        t0 = time.perf_counter()
        for i, f in enumerate(frames):
            if rate_hz:
                delay = t0 + i / rate_hz - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            os.write(self.fd, f)
            self.run.sent.append(time.perf_counter())
        os.write(self.fd, end_frame((frames[-1][1] + 1) if frames else 0))   # next seq after the last frame sent

    def close(self):
        self._stop.set()


def _raw_pty() -> tuple[int, int]:
    master, slave = os.openpty()
    tty.setraw(master)
    tty.setraw(slave)
    return master, slave


def run_native(frames: list[bytes], rate_hz: float | None = 64.0, timeout: float = 120) -> SilRun:
    subprocess.run(["make", "-C", str(FW), "-s", "build/native/hr_sil"], check=True)
    master, slave = _raw_pty()
    proc = subprocess.Popen([str(NATIVE_BIN), os.ttyname(slave)])
    link = _Link(master)
    try:
        assert link.wait_for("READY", 10), "native SIL firmware did not boot"
        t = time.perf_counter()
        link.stream(frames, rate_hz)
        link.wait_for("DONE", timeout)
        link.run.wall_s = time.perf_counter() - t
    finally:
        proc.terminate()
        proc.wait(5)
        link.close()
        os.close(slave)
        os.close(master)
    return link.run


def run_renode(frames: list[bytes], rate_hz: float | None = 64.0, timeout: float = 300) -> SilRun:
    subprocess.run(["make", "-C", str(FW), "-s", "build/arm/fw_sil.elf"], check=True)
    with tempfile.TemporaryDirectory() as tmp:
        pty_link, script = Path(tmp) / "uart", Path(tmp) / "sil.resc"
        script.write_text(f"""mach create
machine LoadPlatformDescription @platforms/cpus/max32652.repl
sysbus LoadELF @{SIL_ELF}
emulation CreateUartPtyTerminal "term" "{pty_link}" true
connector Connect sysbus.uart0 term
start
""")
        proc = subprocess.Popen([renode_binary(), "--disable-gui", "--plain", "-P", "-1", str(script)],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        link = None
        try:
            end = time.time() + 60
            while not pty_link.exists():
                assert time.time() < end and proc.poll() is None, "Renode did not create the UART pty"
                time.sleep(0.1)
            fd = os.open(os.path.realpath(pty_link), os.O_RDWR | os.O_NOCTTY)
            tty.setraw(fd)
            link = _Link(fd)
            # the firmware repeats READY until it hears from us, so attaching late is fine
            assert link.wait_for("READY", 30), "emulated firmware did not boot"
            # Renode is still warming up (JIT) right after boot: on slow machines (CI) the
            # guest briefly stalls while host bytes keep arriving, and the emulated UART's
            # small RX FIFO overflows. A real UART at a fixed baud rate has no such stall.
            time.sleep(2.0)
            t = time.perf_counter()
            link.stream(frames, rate_hz)
            link.wait_for("DONE", timeout)
            link.run.wall_s = time.perf_counter() - t
        finally:
            proc.terminate()
            try:
                proc.wait(10)
            except subprocess.TimeoutExpired:
                proc.kill()
            if link:
                link.close()
                os.close(link.fd)
    return link.run
