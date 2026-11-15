"""Concept figure: a signal is a list of numbers measured at regular time steps.

3 seconds of S1 while sitting, every individual sample drawn as a dot, so the
different sampling rates (64 / 32 / 700 per second) are visible.

Usage:  uv run python python/scripts/plot_samples.py
"""

import matplotlib.pyplot as plt
import numpy as np

from ppg_hr.data import FS_ACC, FS_ECG, FS_PPG, REPO_ROOT, activity_runs, load_subject, middle_of

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#eaeef2", "font.size": 9})
s = load_subject("S1")
t0, L = middle_of(activity_runs(s.activity), 1, 3.0), 3.0

fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
rows = [
    (s.ppg, FS_PPG, "PPG (watch light sensor)", "#2a6fdb", 4),
    (s.acc[:, 2], FS_ACC, "accelerometer z-axis (g)", "#3a9e6f", 5),
    (s.ecg, FS_ECG, "ECG (chest, mV)", "#1f2328", 1.5),
]
for ax, (x, fs, name, color, ms) in zip(axes, rows):
    i0, i1 = int(t0 * fs), int((t0 + L) * fs)
    t = np.arange(i0, i1) / fs - t0
    ax.plot(t, x[i0:i1], color=color, lw=0.6, alpha=0.4)
    ax.plot(t, x[i0:i1], "o", color=color, ms=ms)
    ax.set_title(f"{name}: {fs} measurements per second  →  {i1 - i0} dots in {L:.0f} s",
                 loc="left")
axes[1].set_ylim(s.acc[int(t0 * FS_ACC), 2] - 0.05, s.acc[int(t0 * FS_ACC), 2] + 0.05)
axes[-1].set_xlabel("seconds")
fig.suptitle("Each dot = one number in the data file. Sitting still, person S1.", x=0.01,
             ha="left", fontsize=11)
fig.tight_layout()
out = REPO_ROOT / "reports" / "figures" / "concept_samples.png"
fig.savefig(out, dpi=150)
print("wrote", out.relative_to(REPO_ROOT))
