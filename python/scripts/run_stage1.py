"""Stage 1 evaluation: every pipeline variant on all 15 subjects.

Writes reports/stage1_results.md (tables) and reports/figures/stage1_*.png.
Tuning used S1-S3 only; headline numbers are on held-out S4-S15.

Usage:  uv run python python/scripts/run_stage1.py
"""

import json
import time

import matplotlib.pyplot as plt
import numpy as np

from ppg_hr.data import FS_PPG, REPO_ROOT, SUBJECTS, activity_runs, load_subject, middle_of
from ppg_hr.dsp import bandpass, nlms, spectrum
from ppg_hr.evaluate import mae_by_activity, mae_subject_mean, run
from ppg_hr.pipeline import CONFIGS, DEV_SUBJECTS, FINAL, NAIVE, NFFT, acc_at_ppg_rate

REPORTS = REPO_ROOT / "reports"
FIG = REPORTS / "figures"
TEST = [s for s in SUBJECTS if s not in DEV_SUBJECTS]
SHORT = dict(zip([c.name for c in CONFIGS],
                 ["naive", "+BP", "+NLMS", "+HMM", "final", "final w/o NLMS"]))
INK, MUTED, GRID, BLUE, ORANGE = "#1f2328", "#8c959f", "#eaeef2", "#2a78d6", "#eb6834"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                     "grid.color": GRID, "font.size": 9})


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def main():
    subjects = [load_subject(s) for s in SUBJECTS]
    results, timing = {}, {}
    for cfg in CONFIGS:
        t = time.perf_counter()
        results[cfg.name] = run(subjects, cfg)
        timing[cfg.name] = time.perf_counter() - t
        print(f"{cfg.name:32s} test MAE {mae_subject_mean(results[cfg.name], TEST):5.2f} BPM "
              f"({timing[cfg.name]:.0f} s)")

    # ---- tables -----------------------------------------------------------------
    acts = list(mae_by_activity(results[NAIVE.name], TEST))
    ladder = md_table(
        ["variant", "dev S1-S3", "test S4-S15", "all 15"],
        [[c.name, f"{mae_subject_mean(results[c.name], list(DEV_SUBJECTS)):.2f}",
          f"**{mae_subject_mean(results[c.name], TEST):.2f}**",
          f"{mae_subject_mean(results[c.name], SUBJECTS):.2f}"] for c in CONFIGS])
    per_act = {c.name: mae_by_activity(results[c.name], TEST) for c in CONFIGS}
    act_table = md_table(["activity"] + [SHORT[c.name] for c in CONFIGS],
                         [[a] + [f"{per_act[c.name][a]:.1f}" for c in CONFIGS] for a in acts])
    per_subj = md_table(["subject"] + SUBJECTS,
                        [[SHORT[c.name]] + [f"{results[c.name][s]['err'].mean():.1f}"
                                                      for s in SUBJECTS] for c in (NAIVE, FINAL)])
    (REPORTS / "stage1_results.md").write_text(f"""# Stage 1 results: classical DSP

MAE in BPM vs the ECG-derived reference (8 s windows, 2 s shift).
Parameters tuned on S1-S3 only; **test = held-out S4-S15**.
Overall MAE = mean over subjects of each subject's MAE.

## Ablation (each row adds one block)
{ladder}

## Per activity, test subjects (pooled windows, MAE in BPM)
{act_table}

## Per subject (naive vs final)
{per_subj}
""")
    summary = {c.name: {"test": mae_subject_mean(results[c.name], TEST),
                        "per_activity_test": per_act[c.name]} for c in CONFIGS}
    (REPORTS / "stage1_results.json").write_text(json.dumps(summary, indent=2))

    # ---- figure 1: naive vs final per activity ------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.2))
    y = np.arange(len(acts))
    for off, cfg, col, lab in [(-0.2, NAIVE, "#c9ced6", "naive FFT peak"),
                               (0.2, FINAL, BLUE, "final DSP pipeline")]:
        vals = [per_act[cfg.name][a] for a in acts]
        ax.barh(y + off, vals, height=0.38, color=col, label=lab)
        for yi, v in zip(y + off, vals):
            ax.text(v + 0.3, yi, f"{v:.1f}", va="center", fontsize=8, color=INK)
    ax.set(yticks=y, yticklabels=acts, xlabel="MAE (BPM), lower is better")
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Held-out subjects S4–S15: error per activity", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "stage1_per_activity.png", dpi=150)

    # ---- figure 2: HR trace for a held-out subject -------------------------------
    sid = "S5"
    s = next(x for x in subjects if x.sid == sid)
    t = s.hr_ref_times() / 60
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.plot(t, results[NAIVE.name][sid]["est"], ".", ms=2, color="#b0b7c3", label="naive FFT peak")
    ax.plot(t, results[FINAL.name][sid]["est"], color=BLUE, lw=1, label="final DSP pipeline")
    ax.plot(t, s.hr_ref, color=INK, lw=1.2, label="ECG reference")
    for act, a, b in activity_runs(s.activity):
        if act in (2, 4, 7):
            ax.axvspan(a / 60, b / 60, color="#f3f4f6", lw=0, zorder=0)
            ax.text((a + b) / 120, 0.98, {2: "stairs", 4: "cycling", 7: "walking"}[act],
                    transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8)
    ax.set(xlabel="time (min)", ylabel="HR (BPM)", ylim=(30, 200), xlim=(0, t[-1]))
    ax.legend(frameon=False, ncol=3, loc="upper right", bbox_to_anchor=(1, 1.12))
    ax.set_title(f"{sid} (held out): naive MAE {results[NAIVE.name][sid]['err'].mean():.1f} → "
                 f"final {results[FINAL.name][sid]['err'].mean():.1f} BPM", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / f"stage1_trace_{sid}.png", dpi=150)

    # ---- figure 3: what each block does to one stairs window ---------------------
    s = subjects[0]
    t0 = middle_of(activity_runs(s.activity), 2, 8)
    sl = slice(int(t0 * FS_PPG), int((t0 + 8) * FS_PPG))
    acc = acc_at_ppg_rate(s)
    bp = bandpass(s.ppg, FS_PPG)
    stages = [("raw PPG", s.ppg.astype(float)), ("bandpassed", bp), ("bandpassed + NLMS", nlms(bp, acc))]
    hr = s.hr_ref[int(t0 // 2)] / 60
    fig, axes = plt.subplots(2, 4, figsize=(14, 5))
    tt = np.arange(sl.stop - sl.start) / FS_PPG
    for c, (name, sig) in enumerate(stages):
        axes[0, c].plot(tt, sig[sl], color=BLUE, lw=0.9)
        axes[0, c].set_title(name, loc="left")
        f, m = spectrum(sig[sl], FS_PPG, NFFT)
        axes[1, c].plot(f, m / m[(f > 0.5) & (f < 4)].max(), color=BLUE)
    axes[0, 3].plot(tt, acc[sl], lw=0.8)
    axes[0, 3].set_title("bandpassed ACC (x, y, z)", loc="left")
    am = sum(spectrum(acc[sl, i], FS_PPG, NFFT)[1] for i in range(3))
    f = np.fft.rfftfreq(NFFT, 1 / FS_PPG)
    axes[1, 3].plot(f, am / am.max(), color=INK)
    for ax in axes[1]:
        ax.axvline(hr, color=ORANGE, ls="--", lw=1)
        ax.set(xlim=(0, 5), xlabel="Hz")
    axes[1, 0].set_ylabel("normalized spectrum")
    axes[0, 0].set_ylabel("signal")
    fig.suptitle(f"S1, one 8 s stairs window. Orange dashed = true HR ({hr*60:.0f} BPM)",
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "stage1_blocks_stairs.png", dpi=150)
    print("wrote reports/stage1_results.md and figures")


if __name__ == "__main__":
    main()
