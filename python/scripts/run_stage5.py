"""Stage 5 evaluation: webcam recordings (saved by webcam_demo.py --save ... --ref ...)
against their reference heart rate.

Methods, all on the same per-frame skin-colour traces:
  green          green channel -> DSP pipeline (no motion reference)
  POS            POS pulse     -> DSP pipeline (no motion reference)
  POS + motion   POS pulse     -> DSP pipeline with face motion as the "accelerometer"
  firmware DSP   POS pulse fed as sensor counts into the C firmware (same code as the MCU)
  firmware NN    same, INT8 network mode (trained on wrist PPG: tests transfer, not tuned)

Writes reports/stage5_results.md/.json and reports/figures/stage5_recordings.png.
Usage:  uv run python python/scripts/run_stage5.py [recordings_dir]
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ppg_hr import firmware as fw
from ppg_hr import rppg
from ppg_hr.data import FS_PPG, REPO_ROOT
from ppg_hr.pipeline import ADC_MID, FINAL, Config, estimate_signal

NO_MOTION = Config("no motion", bandpass=True, hmm=True)
SKIP_S = 10           # ignore the first windows while the tracker settles


def to_counts(sig, motion):
    """POS signal -> 18-bit 'ADC counts' (64 Hz) and motion -> int16 'ACC counts' (32 Hz)."""
    ppg = np.clip(ADC_MID + np.round(sig / (sig.std() + 1e-12) * 2000), 0, 2**18 - 1).astype(np.uint32)
    # motion_reference() is in "g-like" units, the firmware ACC LSB is 1/64 g
    acc = np.clip(np.round((motion - motion.mean(0)) * 64), -32768, 32767).astype(np.int16)[::2]
    return ppg, acc


def evaluate(path):
    z = np.load(path)
    t, rgb, box, ref = z["t"], z["rgb"], z["box"], float(z["ref"])
    tu, x = rppg.resample(t, np.hstack([rgb, box]))
    p, g = rppg.pos(x[:, :3]), rppg.green(x[:, :3])
    motion = rppg.motion_reference(x[:, 3:])
    ppg_c, acc_c = to_counts(p, motion)
    n = (len(p) - 512) // 128 + 1
    est = {
        "green": estimate_signal(g, None, NO_MOTION),
        "POS": estimate_signal(p, None, NO_MOTION),
        "POS + motion": estimate_signal(p, motion, FINAL),
        "firmware DSP (C)": (16 + fw.run_counts(ppg_c, acc_c, n, fw.MODE_DSP)) * 1.875,
        "firmware INT8 NN (C)": (16 + fw.run_counts(ppg_c, acc_c, n, fw.MODE_NN)) * 1.875,
    }
    k0 = SKIP_S // 2
    out = {"file": path.name, "label": str(z["label"]), "ref": ref, "seconds": float(t[-1] - t[0]),
           "fps": float(len(t) / (t[-1] - t[0])), "face_found_pct": float(np.isfinite(rgb[:, 0]).mean() * 100)}
    for m, e in est.items():
        e = np.asarray(e, float)[k0:]
        out[m] = {"median_bpm": float(np.median(e)), "abs_err_median": float(abs(np.median(e) - ref)),
                  "window_mae": float(np.abs(e - ref).mean())}
    return out, est


def main():
    rec_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "data" / "webcam"
    files = sorted(p for p in rec_dir.glob("*.npz")
                   if np.isfinite(float(np.load(p)["ref"])) and np.ptp(np.load(p)["t"]) >= 30)
    if not files:
        print(f"no recordings with a reference in {rec_dir}")
        return
    results, series = [], []
    for f in files:
        r, est = evaluate(f)
        results.append(r)
        series.append(est)
    methods = [m for m in results[0] if isinstance(results[0][m], dict)]

    rows = "\n".join(
        f"| {r['label']} | {r['seconds']:.0f} s, {r['fps']:.0f} fps | {r['ref']:.0f} | "
        + " | ".join(f"{r[m]['median_bpm']:.0f} ({r[m]['window_mae']:.1f})" for m in methods) + " |"
        for r in results)
    mean_row = "| **mean window MAE** | | | " + " | ".join(
        f"**{np.mean([r[m]['window_mae'] for r in results]):.1f}**" for m in methods) + " |"
    md = f"""# Stage 5 results: heart rate from the laptop webcam

{len(results)} recordings of the author with a reference heart rate. Cell = median HR over the
recording (after the first {SKIP_S} s), in brackets the MAE of the 8 s-window estimates vs the
reference, in BPM. Only per-frame skin-colour averages are stored, never video.

| recording | length | ref | """ + " | ".join(methods) + " |\n|---|---|---|" + "---|" * len(methods) + f"""
{rows}
{mean_row}
"""
    (REPO_ROOT / "reports" / "stage5_results.md").write_text(md)
    (REPO_ROOT / "reports" / "stage5_results.json").write_text(json.dumps(results, indent=2))

    fig, axes = plt.subplots(len(results), 1, figsize=(10, 2.4 * len(results)), squeeze=False)
    cols = {"green": "#b0b7c3", "POS": "#2a78d6", "POS + motion": "#1baf7a", "firmware INT8 NN (C)": "#1f2328"}
    for ax, r, est in zip(axes[:, 0], results, series):
        for m, c in cols.items():
            e = np.asarray(est[m])
            ax.plot(np.arange(len(e)) * 2 + 4, e, color=c, lw=1.2, label=m)
        ax.axhline(r["ref"], color="#eb6834", ls="--", lw=1.2, label="reference")
        ax.set(ylabel="BPM", ylim=(40, 160), title=f"{r['label']}")
        ax.title.set_ha("left"); ax.title.set_x(0)
    axes[0, 0].legend(frameon=False, ncol=5, fontsize=8, loc="upper right")
    axes[-1, 0].set_xlabel("seconds")
    fig.tight_layout()
    fig.savefig(REPO_ROOT / "reports" / "figures" / "stage5_recordings.png", dpi=150)
    print(md)


if __name__ == "__main__":
    main()
