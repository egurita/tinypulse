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


def test_firmware_builds_for_cortex_m4_without_warnings():
    pytest.importorskip("shutil").which("arm-none-eabi-gcc") or pytest.skip("no ARM toolchain")
    subprocess.run(["make", "-C", str(REPO_ROOT / "firmware"), "-s", "clean"], check=True)
    r = subprocess.run(["make", "-C", str(REPO_ROOT / "firmware"), "-s", "arm"],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr
    assert "warning" not in (r.stdout + r.stderr).lower(), r.stderr
