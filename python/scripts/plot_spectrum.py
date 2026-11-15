"""Spectra of the sitting vs stairs segments, with the predictions drawn on.

Row 1: PPG spectrum. Row 2: accelerometer-magnitude spectrum (motion only, no heart).
Dashed lines: reference heart rate f and its harmonic 2f (from the ECG).
Dotted line: the step rate you estimated by counting peaks (2.2 Hz).

Usage:  uv run python python/scripts/plot_spectrum.py [S1]
"""

import sys

import matplotlib.pyplot as plt
import numpy as np

from ppg_hr.data import FS_ACC, FS_PPG, REPO_ROOT, activity_runs, load_subject, middle_of
from ppg_hr.dsp import dominant_frequency, spectrum

INK, MUTED, ACCENT, RED = "#1f2328", "#8c959f", "#2a6fdb", "#d1242f"
plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": "#eaeef2",
    "font.size": 9,
})
LENGTH_S = 20.0
NFFT = 4096


def main():
    sid = sys.argv[1] if len(sys.argv) > 1 else "S1"
    subj = load_subject(sid)
    runs = activity_runs(subj.activity)
    hr_t = subj.hr_ref_times()

    fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex=True, sharey="row")
    for c, (name, act_id, step_guess) in enumerate([("sitting", 1, None), ("stairs", 2, 2.2)]):
        t0 = middle_of(runs, act_id, LENGTH_S)
        ppg = subj.ppg[int(t0 * FS_PPG): int((t0 + LENGTH_S) * FS_PPG)]
        acc = subj.acc[int(t0 * FS_ACC): int((t0 + LENGTH_S) * FS_ACC)]
        # Magnitude |acc| = how hard the wrist is being shaken, independent of how the
        # watch is rotated (the 3 axes change with orientation, the magnitude doesn't).
        acc_mag = np.linalg.norm(acc, axis=1)
        f_hr = subj.hr_ref[(hr_t >= t0) & (hr_t <= t0 + LENGTH_S)].mean() / 60

        for r, (sig, fs, label) in enumerate([(ppg, FS_PPG, "PPG"), (acc_mag, FS_ACC, "|acc|")]):
            ax = axes[r, c]
            freqs, mag = spectrum(sig, fs, NFFT)
            ax.axvspan(0.5, 4.0, color="#f3f4f6", lw=0, zorder=0)
            ax.plot(freqs, mag, color=ACCENT if r == 0 else INK, lw=1.2)
            for k, ls in [(1, "--"), (2, (0, (4, 4)))]:
                ax.axvline(k * f_hr, color=RED, ls=ls, lw=1, alpha=0.8 if k == 1 else 0.45)
            if step_guess:
                ax.axvline(step_guess, color="#e8833a", ls=":", lw=1.6)
            peak = dominant_frequency(sig, fs, nfft=NFFT)
            ax.set_title(f"{name.upper()} {label}: tallest peak in band = {peak:.2f} Hz "
                         f"({60 * peak:.0f} /min)", loc="left")
            print(f"{name:8s} {label:6s} tallest peak {peak:.2f} Hz = {60*peak:.0f}/min   "
                  f"(reference HR {f_hr:.2f} Hz = {60*f_hr:.0f} BPM)")
        axes[1, c].set_xlabel("frequency (Hz)")

    axes[0, 0].set_ylabel("PPG magnitude")
    axes[1, 0].set_ylabel("|acc| magnitude")
    axes[0, 0].set_xlim(0, 5)
    fig.suptitle(f"{sid}: spectra of 20 s segments.  Red dashed = reference HR f (darker) and 2f; "
                 "orange dotted = your step estimate; grey band = 0.5–4 Hz", x=0.01, ha="left")
    fig.tight_layout()
    out = REPO_ROOT / "reports" / "figures" / f"spectrum_{sid}.png"
    fig.savefig(out, dpi=150)
    print("wrote", out.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
