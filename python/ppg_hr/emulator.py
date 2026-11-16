"""Run the firmware image in Renode (emulated MAX32652) and collect its output.

Returns the UART text and the benchmark markers: every call to bench_mark(id) is
hooked by Renode, which logs the CPU's executed-instruction counter.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .data import REPO_ROOT

FW_DIR = REPO_ROOT / "firmware"
ELF = FW_DIR / "build" / "arm" / "fw.elf"
MARK_RE = re.compile(r"MARK (\d+) (\d+)")


def renode_binary() -> str:
    for cand in (os.environ.get("RENODE"), shutil.which("renode"),
                 str(Path.home() / "Applications/Renode.app/Contents/MacOS/renode")):
        if cand and Path(cand).exists():
            return cand
    raise FileNotFoundError("Renode not found: set RENODE=/path/to/renode")


@dataclass
class EmulatorRun:
    uart: str
    marks: list[tuple[int, int]]        # (marker id, executed instructions)


def build_arm() -> None:
    subprocess.run(["make", "-C", str(FW_DIR), "-s", "arm"], check=True)


def run(elf: Path = ELF, virtual_seconds: float = 5.0, timeout_s: float = 1200,
        count_instructions: bool = True) -> EmulatorRun:
    """count_instructions=False skips the per-marker hook (each costs a Python call in
    Renode): ~10x faster when only the UART output is needed."""
    hook = ("cpu AddHook `sysbus GetSymbolAddress \"bench_mark\"` \"self.Log(LogLevel.Warning, "
            "'MARK %d %d' % (self.GetRegister(0).RawValue, self.ExecutedInstructions))\"\n"
            if count_instructions else "")
    with tempfile.TemporaryDirectory() as tmp:
        uart_file, script = Path(tmp) / "uart.txt", Path(tmp) / "run.resc"
        script.write_text(f"""mach create
machine LoadPlatformDescription @platforms/cpus/max32652.repl
sysbus LoadELF @{elf}
sysbus.uart0 CreateFileBackend @{uart_file} true
{hook}emulation RunFor "{virtual_seconds}"
quit
""")
        log = subprocess.run([renode_binary(), "--disable-gui", "--plain", "-P", "-1", str(script)],
                             capture_output=True, text=True, timeout=timeout_s).stdout
        uart = uart_file.read_text() if uart_file.exists() else ""
    marks = [(int(a), int(b)) for a, b in MARK_RE.findall(log)]
    return EmulatorRun(uart=uart, marks=marks)


def mark_stats(marks: list[tuple[int, int]], base: int = 0) -> dict[str, "np.ndarray"]:
    """Instruction counts between benchmark markers of one run (base 0 = DSP, 10 = NN):
    per hr_push() call without / with a window, and per driver FIFO read."""
    import numpy as np
    push, window, fifo = [], [], []
    for (i1, c1), (i2, c2) in zip(marks, marks[1:]):
        if i1 == base + 1 and i2 == base + 2: push.append(c2 - c1)
        if i1 == base + 1 and i2 == base + 3: window.append(c2 - c1)
        if i1 == base + 4 and i2 == base + 5: fifo.append(c2 - c1)
    return {"push": np.array(push), "window": np.array(window), "fifo": np.array(fifo)}


def uart_results(uart: str, prefix: str) -> tuple[list[int], dict[str, str]]:
    """HR bins printed with `prefix` ("HR" DSP, "HN" NN) and that run's DONE fields."""
    bins = [int(l.split()[2]) for l in uart.splitlines() if l.startswith(prefix + " ")]
    mode = "nn" if prefix == "HN" else "dsp"
    line = next(l for l in uart.splitlines() if l.startswith(f"DONE mode={mode}"))
    return bins, dict(kv.split("=") for kv in line.split()[1:])
