"""Build the CI test report from pytest's JUnit XML and the metrics the tests recorded.

Inputs:  reports/ci/junit.xml, reports/ci/metrics.json
Outputs: reports/ci/report.md (+ PNG plots), reports/ci/summary.md (for $GITHUB_STEP_SUMMARY)

Usage:  uv run python python/scripts/ci_report.py
"""

import json
import os
import xml.etree.ElementTree as ET
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from ppg_hr.data import REPO_ROOT

CI = REPO_ROOT / "reports" / "ci"
LAYERS = {   # test file -> layer of the test pyramid, bottom to top
    "test_dsp": "1 unit: Python DSP blocks (synthetic signals)",
    "test_firmware_c": "1 unit: C firmware + driver vs mock, ARM build",
    "test_data": "2 data sanity: dataset assumptions",
    "test_pipeline_synthetic": "3 end-to-end synthetic: Python and C",
    "test_equivalence": "4 equivalence: C fixed point vs Python float",
    "test_regression": "5 regression: per-activity error limits",
    "test_sil": "6 software-in-the-loop: serial, 64 Hz, native + emulated MCU",
    "test_nn": "7 neural network: LOSO regression, INT8 C = NumPy, quantization loss",
}
INK, MUTED, BLUE, RED, GREY = "#1f2328", "#8c959f", "#2a78d6", "#d1242f", "#c9ced6"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                     "grid.color": "#eaeef2", "font.size": 9})


def junit_summary():
    stats = defaultdict(lambda: {"passed": 0, "failed": 0, "skipped": 0, "time": 0.0})
    failures = []
    for case in ET.parse(CI / "junit.xml").getroot().iter("testcase"):
        mod = case.get("classname", "").split(".")[-1]
        s = stats[mod]
        s["time"] += float(case.get("time", 0))
        if case.find("failure") is not None or case.find("error") is not None:
            s["failed"] += 1
            failures.append(f"{mod}::{case.get('name')}")
        elif case.find("skipped") is not None:
            s["skipped"] += 1
        else:
            s["passed"] += 1
    return stats, failures


def plots(m):
    out = []
    if "regression_mae_by_activity" in m:
        mae, naive, lim = m["regression_mae_by_activity"], m["regression_naive_mae_by_activity"], m["regression_limits"]
        acts = list(lim)
        y = np.arange(len(acts))
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(y - 0.2, [naive[a] for a in acts], 0.38, color=GREY, label="naive baseline")
        ax.barh(y + 0.2, [mae[a] for a in acts], 0.38, color=BLUE, label="current pipeline")
        ax.plot([lim[a] for a in acts], y + 0.2, "|", ms=14, mew=2, color=RED, label="regression limit")
        ax.set(yticks=y, yticklabels=acts, xlabel="MAE (BPM), held-out subjects S4–S15")
        ax.invert_yaxis(); ax.grid(axis="y", visible=False)
        ax.legend(frameon=False, loc="lower right")
        fig.tight_layout(); fig.savefig(CI / "regression.png", dpi=130); plt.close(fig)
        out.append(("Dataset regression", "regression.png"))
    lat = {k: m[f"sil_{k}_latency_ms"] for k in ("native", "renode") if f"sil_{k}_latency_ms" in m}
    if lat:
        fig, ax = plt.subplots(figsize=(8, 2.6))
        for i, (k, v) in enumerate(lat.items()):
            ax.plot(v, np.full(len(v), i) + np.random.default_rng(0).uniform(-0.15, 0.15, len(v)),
                    "o", ms=4, color=BLUE if k == "native" else INK, alpha=0.8)
        ax.set(yticks=range(len(lat)), yticklabels=[f"{k} firmware" for k in lat], xscale="log",
               xlabel="ms from last sample of a window sent to its HR received (host clock)")
        ax.set_ylim(-0.6, len(lat) - 0.4)
        fig.tight_layout(); fig.savefig(CI / "sil_latency.png", dpi=130); plt.close(fig)
        out.append(("SIL latency at 64 Hz", "sil_latency.png"))
    return out


def main():
    stats, failures = junit_summary()
    m = json.loads((CI / "metrics.json").read_text()) if (CI / "metrics.json").exists() else {}
    tot = {k: sum(s[k] for s in stats.values()) for k in ("passed", "failed", "skipped")}
    status = "PASS" if tot["failed"] == 0 else "FAIL"
    commit = os.environ.get("GITHUB_SHA", "local")[:8]

    rows = ["| layer | passed | failed | skipped | time |", "|---|---|---|---|---|"]
    for mod in sorted(stats, key=lambda k: LAYERS.get(k, "9")):
        s = stats[mod]
        rows.append(f"| {LAYERS.get(mod, mod)} | {s['passed']} | {s['failed']} | {s['skipped']} | {s['time']:.1f} s |")

    key = ["| metric | value |", "|---|---|"]
    if "regression_test_mae" in m:
        key.append(f"| held-out MAE (S4–S15) | {m['regression_test_mae']:.2f} BPM |")
    if "equivalence_total_windows" in m:
        key.append(f"| C fixed point = Python float | {m['equivalence_identical_windows']}/{m['equivalence_total_windows']} windows "
                   f"({m['equivalence_identical_windows'] / m['equivalence_total_windows'] * 100:.2f}%) |")
    for k in ("native", "renode"):
        if f"sil_{k}_latency_ms" in m:
            v = np.array(m[f"sil_{k}_latency_ms"])
            key.append(f"| SIL {k}: HR latency (median / max) | {np.median(v):.1f} / {v.max():.1f} ms |")
    if "nn_loso_mae" in m:
        key.append(f"| NN (spectral CNN + tracker), LOSO all 15 | {m['nn_loso_mae']:.2f} BPM (DSP {m['dsp_mae_all15']:.2f}) |")
    if "int8_vs_float_mae_gap_S5" in m:
        key.append(f"| INT8 vs float (S5) | +{m['int8_vs_float_mae_gap_S5']:.2f} BPM, "
                   f"{m['int8_vs_float_identical_S5'] * 100:.0f}% identical windows |")
    if "mcu_stack_used_bytes_nn" in m:
        key.append(f"| emulated MCU stack high-water | {m['mcu_stack_used_bytes_nn']} / 4096 B |")
    if "c_unit_checks" in m:
        key.append(f"| C unit checks | {m['c_unit_checks']} ({m['c_unit_failures']} failed) |")

    body = (f"## Test report: {status} ({tot['passed']} passed, {tot['failed']} failed, {tot['skipped']} skipped), commit `{commit}`\n\n"
            + "\n".join(rows) + "\n\n" + "\n".join(key) + "\n")
    if failures:
        body += "\n**Failures:**\n" + "\n".join(f"- `{f}`" for f in failures) + "\n"
    (CI / "summary.md").write_text(body)
    figs = plots(m)
    (CI / "report.md").write_text(body + "".join(f"\n### {t}\n![{t}]({f})\n" for t, f in figs))
    print(body)


if __name__ == "__main__":
    main()
