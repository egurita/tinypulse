"""Look at the raw data before touching it.

Figure 1 (overview): reference HR over the whole ~2.5 h session, with the activity
protocol marked. Shows the "story" of the recording.

Figure 2 (zoom): the same 3 signals for 20 s of SITTING vs 20 s of STAIRS, on the
same y-scales so you can compare them directly. This one figure motivates the whole
project: when the wrist moves, the PPG is dominated by motion, not by the heartbeat.

Usage:  uv run python python/scripts/plot_raw.py [S1]
"""

import sys

import matplotlib.pyplot as plt
import numpy as np

from ppg_hr.data import (ACTIVITY_NAMES, FS_ACC, FS_ECG, FS_PPG, REPO_ROOT, activity_runs,
                         load_subject, middle_of)

FIG_DIR = REPO_ROOT / "reports" / "figures"
INK, MUTED, ACCENT = "#1f2328", "#8c959f", "#2a6fdb"
plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": "#eaeef2", "grid.linewidth": 0.8,
    "lines.linewidth": 1.0, "font.size": 9,
})


def plot_overview(subj, runs):
    fig, ax = plt.subplots(figsize=(12, 3.2))
    t = subj.hr_ref_times() / 60
    for i, (act, s, e) in enumerate(r for r in runs if r[0] != 0):
        ax.axvspan(s / 60, e / 60, color="#f3f4f6" if i % 2 else "#e5e7eb", lw=0)
        ax.text((s + e) / 120, 1.01, ACTIVITY_NAMES[act], transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=7.5, color=INK, rotation=30)
    ax.plot(t, subj.hr_ref, color=ACCENT, lw=1.2)
    ax.set(xlabel="time (min)", ylabel="reference HR (BPM)", xlim=(0, t[-1]))
    ax.set_title(f"{subj.sid}: ECG-derived heart rate across the protocol "
                 "(grey bands = activities, white gaps = transitions)", loc="left", pad=28)
    fig.tight_layout()
    return fig


def plot_zoom(subj, runs, length_s=20.0):
    cols = [("sitting", 1), ("stairs", 2)]
    fig, axes = plt.subplots(3, 2, figsize=(12, 7), sharex="col", sharey="row")
    hr_t = subj.hr_ref_times()

    for c, (name, act_id) in enumerate(cols):
        t0 = middle_of(runs, act_id, length_s)

        def window(fs, n):  # sample indices + time axis (seconds from t0) for this stream
            i0, i1 = int(t0 * fs), int((t0 + length_s) * fs)
            return slice(i0, min(i1, n)), np.arange(i0, min(i1, n)) / fs - t0

        # Row 0: PPG. Light absorption by blood in the wrist. Each heartbeat pushes a pulse
        # of blood -> one bump per beat. Units are arbitrary (raw sensor counts).
        sl, t = window(FS_PPG, len(subj.ppg))
        axes[0, c].plot(t, subj.ppg[sl], color=ACCENT)

        # Row 1: wrist accelerometer, 3 axes in g. Sitting: flat lines (just gravity split
        # across the axes). Stairs: a periodic pattern at the step rate.
        sl, t = window(FS_ACC, len(subj.acc))
        for k, (axis_name, col) in enumerate(zip("xyz", ["#2a6fdb", "#e8833a", "#3a9e6f"])):
            axes[1, c].plot(t, subj.acc[sl, k], color=col, label=axis_name)

        # Row 2: chest ECG with the R-peaks (the sharp spike of each beat) the dataset
        # authors detected. The time between R-peaks is the ground truth for heart rate.
        sl, t = window(FS_ECG, len(subj.ecg))
        axes[2, c].plot(t, subj.ecg[sl], color=INK, lw=0.7)
        r = subj.rpeaks[(subj.rpeaks >= sl.start) & (subj.rpeaks < sl.stop)]
        axes[2, c].plot(r / FS_ECG - t0, subj.ecg[r], "o", ms=4, mfc="none", color="#d1242f",
                        label="R-peak")

        hr = subj.hr_ref[(hr_t >= t0) & (hr_t <= t0 + length_s)].mean()
        axes[0, c].set_title(f"{name.upper()}  (t = {t0/60:.1f} min, reference HR ≈ {hr:.0f} BPM, "
                             f"{len(r)} beats in {length_s:.0f} s)", loc="left")
        axes[2, c].set_xlabel("seconds")

    axes[0, 0].set_ylabel("PPG (raw units)")
    axes[1, 0].set_ylabel("wrist acc (g)")
    axes[2, 0].set_ylabel("ECG (mV), ○ = R-peak")
    axes[1, 1].legend(loc="upper right", ncol=3, frameon=False)
    fig.suptitle(f"{subj.sid}: raw signals, same y-scales left vs right", x=0.01, ha="left",
                 fontsize=11)
    fig.tight_layout()
    return fig


def main():
    sid = sys.argv[1] if len(sys.argv) > 1 else "S1"
    subj = load_subject(sid)
    runs = activity_runs(subj.activity)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for name, fig in [("overview", plot_overview(subj, runs)), ("zoom", plot_zoom(subj, runs))]:
        out = FIG_DIR / f"raw_{name}_{sid}.png"
        fig.savefig(out, dpi=150)
        print("wrote", out.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
