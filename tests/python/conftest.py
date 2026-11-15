"""Shared fixtures: dataset availability, Renode availability, and a metrics recorder.

Tests call `metrics.record("name", value)`; at the end of the session everything is
written to reports/ci/metrics.json, which python/scripts/ci_report.py turns into the
CI report (tables + plots).
"""

import json
import shutil
from pathlib import Path

import pytest

from ppg_hr.data import REPO_ROOT, available_subjects

METRICS_FILE = REPO_ROOT / "reports" / "ci" / "metrics.json"
_metrics: dict = {}


class Metrics:
    def record(self, name: str, value):
        _metrics[name] = value


@pytest.fixture(scope="session")
def metrics() -> Metrics:
    return Metrics()


def pytest_sessionfinish(session, exitstatus):
    if _metrics:
        METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
        old = json.loads(METRICS_FILE.read_text()) if METRICS_FILE.exists() else {}
        old.update(_metrics)
        METRICS_FILE.write_text(json.dumps(old, indent=2, default=float))


def have_dataset() -> bool:
    return len(available_subjects()) == 15


def have_renode() -> bool:
    try:
        from ppg_hr.emulator import renode_binary
        renode_binary()
        return shutil.which("arm-none-eabi-gcc") is not None
    except FileNotFoundError:
        return False


needs_dataset = pytest.mark.skipif(not have_dataset(), reason="PPG-DaLiA not prepared")
needs_renode = pytest.mark.skipif(not have_renode(), reason="Renode or arm-none-eabi-gcc missing")
