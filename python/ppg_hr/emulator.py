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


def run(elf: Path = ELF, virtual_seconds: float = 5.0, timeout_s: float = 600) -> EmulatorRun:
    with tempfile.TemporaryDirectory() as tmp:
        uart_file, script = Path(tmp) / "uart.txt", Path(tmp) / "run.resc"
        script.write_text(f"""mach create
machine LoadPlatformDescription @platforms/cpus/max32652.repl
sysbus LoadELF @{elf}
sysbus.uart0 CreateFileBackend @{uart_file} true
cpu AddHook `sysbus GetSymbolAddress "bench_mark"` "self.Log(LogLevel.Warning, 'MARK %d %d' % (self.GetRegister(0).RawValue, self.ExecutedInstructions))"
emulation RunFor "{virtual_seconds}"
quit
""")
        log = subprocess.run([renode_binary(), "--disable-gui", "--plain", "-P", "-1", str(script)],
                             capture_output=True, text=True, timeout=timeout_s).stdout
        uart = uart_file.read_text() if uart_file.exists() else ""
    marks = [(int(a), int(b)) for a, b in MARK_RE.findall(log)]
    return EmulatorRun(uart=uart, marks=marks)
