"""Concept figure: why a periodic signal can have peaks at f, 2f, 3f.

Left: a pure sine at 0.91 Hz. Right: a PPG-like shape at the SAME rate 0.91 Hz
(fast rise, slow fall). Same period, different shape -> different spectrum.

Usage:  uv run python python/scripts/concept_harmonics.py
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import sawtooth

from ppg_hr.data import REPO_ROOT
from ppg_hr.dsp import spectrum

FS, F, SECONDS = 64, 0.91, 20
t = np.arange(SECONDS * FS) / FS
signals = {
    "pure sine": np.sin(2 * np.pi * F * t),
    # width=0.2: rises during the first 20% of each period, falls for the remaining 80%,
    # roughly like a pulse wave. Then negated so it rises fast and decays slowly.
    "PPG-like shape (fast rise, slow fall)": -sawtooth(2 * np.pi * F * t, width=0.8),
}

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#eaeef2", "font.size": 9})
fig, axes = plt.subplots(2, 2, figsize=(11, 5.5), sharey="row")
for c, (name, x) in enumerate(signals.items()):
    axes[0, c].plot(t[: 5 * FS], x[: 5 * FS], color="#2a6fdb")
    axes[0, c].set(title=f"{name}, repeats every {1/F:.2f} s", xlabel="seconds")
    freqs, mag = spectrum(x, FS, 4096)
    axes[1, c].plot(freqs, mag, color="#1f2328")
    for k in (1, 2, 3, 4):
        axes[1, c].axvline(k * F, color="#d1242f", ls="--", lw=0.8, alpha=0.5)
        axes[1, c].text(k * F, 1.0, f"{k}f", color="#d1242f", ha="center", va="bottom",
                        transform=axes[1, c].get_xaxis_transform())
    axes[1, c].set(xlim=(0, 4.5), xlabel="frequency (Hz)")
axes[0, 0].set_ylabel("signal")
axes[1, 0].set_ylabel("spectrum magnitude")
fig.tight_layout()
out = REPO_ROOT / "reports" / "figures" / "concept_harmonics.png"
fig.savefig(out, dpi=150)
print("wrote", out.relative_to(REPO_ROOT))
