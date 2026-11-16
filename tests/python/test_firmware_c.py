"""Run the C unit tests (DSP vs libm, driver vs mock) from pytest, so one runner covers all layers."""

import re
import subprocess

import pytest

from ppg_hr.data import REPO_ROOT

pytestmark = pytest.mark.firmware


def test_c_unit_tests_pass(metrics):
    r = subprocess.run(["make", "-C", str(REPO_ROOT / "firmware"), "-s", "test"],
                       capture_output=True, text=True, timeout=600)
    print(r.stdout[-3000:])
    m = re.search(r"(\d+) checks, (\d+) failures", r.stdout)
    assert m, r.stdout + r.stderr
    checks, failures = map(int, m.groups())
    metrics.record("c_unit_checks", checks)
    metrics.record("c_unit_failures", failures)
    assert r.returncode == 0 and failures == 0


@pytest.mark.renode
def test_emulated_mcu_both_modes_bit_exact_and_stack_within_budget(metrics):
    """The benchmark image (120 s of held-out S5, DSP pass then INT8-NN pass) on the
    emulated MAX32652 must reproduce the host build exactly, lose no samples and stay
    well inside its 4 KB stack. This catches target-only bugs such as the Stage 4 stack
    overflow (5 KB of per-window locals silently corrupting .bss)."""
    from conftest import have_renode
    if not have_renode():
        pytest.skip("Renode or arm-none-eabi-gcc missing")
    from ppg_hr import emulator
    emulator.build_arm()
    run = emulator.run(virtual_seconds=6, count_instructions=False)
    for prefix in ("HR", "HN"):
        bins, done = emulator.uart_results(run.uart, prefix)
        assert len(bins) == 57 and done["mismatches_vs_host"] == "0" and done["samples_lost"] == "0", done
        metrics.record(f"mcu_stack_used_bytes_{done['mode']}", int(done["stack_used"]))
        assert int(done["stack_used"]) < 3 * 1024


def test_firmware_builds_for_cortex_m4_without_warnings():
    pytest.importorskip("shutil").which("arm-none-eabi-gcc") or pytest.skip("no ARM toolchain")
    subprocess.run(["make", "-C", str(REPO_ROOT / "firmware"), "-s", "clean"], check=True)
    r = subprocess.run(["make", "-C", str(REPO_ROOT / "firmware"), "-s", "arm"],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr
    assert "warning" not in (r.stdout + r.stderr).lower(), r.stderr
