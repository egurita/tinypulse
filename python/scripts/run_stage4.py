"""Stage 4 report: DSP vs neural networks, leave-one-subject-out, INT8 quantization.

Needs the trained folds (train_stage4.py spec/time) and gen_nn_model.py for the INT8
per-fold models. Writes reports/stage4_results.md/.json and reports/figures/stage4_*.png.

Usage:  uv run python python/scripts/run_stage4.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from ppg_hr import features
from ppg_hr.data import ACTIVITY_NAMES, REPO_ROOT, SUBJECTS, activity_runs, load_subject
from ppg_hr.dsp import acc_mask, hmm_track_step
from ppg_hr.evaluate import window_activity
from ppg_hr.nn import SpecCNN, TimeCNN, macs, n_params
from ppg_hr.pipeline import DEV_SUBJECTS, FINAL, estimate_hr
from ppg_hr.quant import hmm_track_int, int_forward, logits_to_logp_q16

sys.path.insert(0, str(REPO_ROOT / "python" / "scripts"))
from gen_nn_model import quantized_fold  # noqa: E402

M = REPO_ROOT / "models" / "stage4"
FIG, REPORTS = REPO_ROOT / "reports" / "figures", REPO_ROOT / "reports"
TEST = [s for s in SUBJECTS if s not in DEV_SUBJECTS]
INK, MUTED, BLUE, ORANGE, AQUA = "#1f2328", "#8c959f", "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                     "grid.color": "#eaeef2", "font.size": 9})


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def track(evidence, start=0, n=None):
    """Float HMM tracker over per-window evidence (probabilities or magnitudes)."""
    ev = evidence[start:start + n if n else None]
    out, score = np.empty(len(ev)), None
    for k, e in enumerate(ev):
        score = hmm_track_step(score, e, 2, 0.5)
        out[k] = np.argmax(score)
    return out


def dsp_evidence(spec):
    """The DSP's per-window evidence (ACC-masked PPG band spectrum) from the features."""
    return np.stack([acc_mask(s[0, :113], s[1, :113], 0.7) for s in spec])


def emulated():
    """Both modes on the emulated MAX32652: bit-exactness and instruction counts."""
    import subprocess
    from ppg_hr import emulator
    emulator.build_arm()
    meta = json.loads((REPO_ROOT / "firmware" / "gen" / "test_vector.json").read_text())
    run = emulator.run(virtual_seconds=6)
    out = {}
    for prefix, base, key in (("HR", 0, "dsp"), ("HN", 10, "nn")):
        bins, done = emulator.uart_results(run.uart, prefix)
        st = emulator.mark_stats(run.marks, base)
        expected = meta["expected_bins" if key == "dsp" else "expected_bins_nn"]
        out[key] = {"bit_exact_vs_host": bins == expected, "estimates": len(bins), "done": done,
                    "instr_per_sample": float(st["push"].mean()),
                    "instr_per_window": float(st["window"].mean()),
                    "cpu_load_pct_120MHz": float((64 * st["push"].mean() + 0.5 * st["window"].mean()) / 120e6 * 100)}
    # flash / RAM of the network code + weights (Cortex-M4, -O2)
    fw = REPO_ROOT / "firmware"
    flags = ["arm-none-eabi-gcc", "-mcpu=cortex-m4", "-mthumb", "-mfloat-abi=soft", "-O2", "-ffreestanding",
             "-Iinclude", "-Igen", "-c", "-o", "/dev/stdout"]
    sizes = {}
    for name, src in (("nn.c (int8 kernels)", "src/nn.c"), ("nn_model.c (weights)", "gen/nn_model.c")):
        obj = fw / "build" / "arm" / (Path(src).stem + "_s4.o")
        subprocess.run(flags[:-2] + ["-o", str(obj), src], cwd=fw, check=True)
        t, d, b = map(int, subprocess.run(["arm-none-eabi-size", str(obj)], capture_output=True, text=True,
                                          check=True).stdout.splitlines()[1].split()[:3])
        sizes[name] = {"flash": t + d, "ram": d + b}
    out["memory"] = sizes
    return out


def main():
    hr = {s: load_subject(s).hr_ref.astype(float) for s in SUBJECTS}
    act = {s: window_activity(load_subject(s)) for s in SUBJECTS}
    feats = {s: features.load(s) for s in SUBJECTS}
    est, info = {}, {}

    est["DSP (Stage 1/2)"] = {s: estimate_hr(load_subject(s), FINAL) for s in SUBJECTS}
    for arch, label in (("time", "Time CNN"), ("spec", "Spectral CNN (hybrid)")):
        logits = {s: np.load(M / arch / f"pred_{s}.npy") for s in SUBJECTS}
        est[f"{label}, argmax"] = {s: features.bin_to_hr(logits[s].argmax(1)) for s in SUBJECTS}
        est[f"{label} + HMM tracker"] = {s: features.bin_to_hr(track(softmax(logits[s]))) for s in SUBJECTS}
        info[arch] = logits

    # INT8: each fold quantized with its own training-subject calibration, integer inference,
    # integer tracker (the arithmetic the firmware runs).
    from ppg_hr.quant import quantize_input
    int8, ident = {}, []
    for s in SUBJECTS:
        q = quantized_fold(s)
        spec = feats[s]["spec"]
        xq = quantize_input(np.round(spec * 2**20).astype(np.int64))    # same normalisation as C
        lg = int_forward(q, xq)
        int8[s] = features.bin_to_hr(hmm_track_int(logits_to_logp_q16(q, lg)))
        ident.append(np.mean(int8[s] == est["Spectral CNN (hybrid) + HMM tracker"][s]))
    est["Spectral CNN INT8 + HMM (firmware arithmetic)"] = int8

    # ---- metrics
    def subj_mae(e, sids):
        return float(np.mean([np.abs(e[s] - hr[s]).mean() for s in sids]))

    def by_act(e):
        err = np.concatenate([np.abs(e[s] - hr[s]) for s in SUBJECTS])
        a = np.concatenate([act[s] for s in SUBJECTS])
        return {ACTIVITY_NAMES[k]: float(err[a == k].mean()) for k in ACTIVITY_NAMES if k}

    table = {m: {"all15": subj_mae(e, SUBJECTS), "test": subj_mae(e, TEST), "per_activity": by_act(e),
                 "per_subject": {s: float(np.abs(e[s] - hr[s]).mean()) for s in SUBJECTS}} for m, e in est.items()}
    size = {"Time CNN": (n_params(TimeCNN()), macs(TimeCNN(), torch.randn(1, 4, 256))),
            "Spectral CNN": (n_params(SpecCNN()), macs(SpecCNN(), torch.randn(1, 4, 113)))}

    # Cold start: tracker started fresh at the beginning of each activity block (first 2 min).
    cold = {"DSP": [], "Spectral CNN": []}
    for s in TEST:
        ev_dsp, ev_nn = dsp_evidence(feats[s]["spec"]), softmax(info["spec"][s])
        for a_id, a, b in activity_runs(load_subject(s).activity):
            if a_id == 0 or b - a < 120:
                continue
            k0 = int(a // 2)
            n = min(60, len(hr[s]) - k0)
            ref = hr[s][k0:k0 + n]
            cold["DSP"].append(np.abs(features.bin_to_hr(track(ev_dsp, k0, n)) - ref).mean())
            cold["Spectral CNN"].append(np.abs(features.bin_to_hr(track(ev_nn, k0, n)) - ref).mean())
    cold = {k: float(np.mean(v)) for k, v in cold.items()}

    emu = emulated()
    res = {"table": table, "size": size, "cold_start_tracker_only": cold, "emulator": emu,
           "int8_identical_to_float_pct": float(np.mean(ident) * 100),
           "train_log": {a: json.loads((M / a / "train_log.json").read_text()) for a in ("spec", "time")}}
    (REPORTS / "stage4_results.json").write_text(json.dumps(res, indent=2))

    # ---- figure: per activity
    acts = list(table["DSP (Stage 1/2)"]["per_activity"])
    show = [("DSP (Stage 1/2)", "#c9ced6"), ("Time CNN + HMM tracker", AQUA),
            ("Spectral CNN (hybrid) + HMM tracker", BLUE), ("Spectral CNN INT8 + HMM (firmware arithmetic)", INK)]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    y = np.arange(len(acts))
    h = 0.8 / len(show)
    for i, (m, col) in enumerate(show):
        ax.barh(y - 0.4 + h * (i + 0.5), [table[m]["per_activity"][a] for a in acts], h * 0.92,
                color=col, label=f"{m}: {table[m]['all15']:.2f} BPM")
    ax.set(yticks=y, yticklabels=acts, xlabel="MAE (BPM), leave-one-subject-out, all 15 subjects")
    ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="lower right", title="mean over subjects", title_fontsize=8, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "stage4_per_activity.png", dpi=150)

    # ---- figure: S5 trace
    s = "S5"
    t = np.arange(len(hr[s])) * 2 / 60 + 4 / 60
    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.plot(t, est["DSP (Stage 1/2)"][s], color="#b0b7c3", lw=0.9, label=f"DSP ({table['DSP (Stage 1/2)']['per_subject'][s]:.1f})")
    ax.plot(t, est["Spectral CNN INT8 + HMM (firmware arithmetic)"][s], color=BLUE, lw=0.9,
            label=f"Spectral CNN INT8 + HMM ({table['Spectral CNN INT8 + HMM (firmware arithmetic)']['per_subject'][s]:.1f})")
    ax.plot(t, hr[s], color=INK, lw=1.2, label="ECG reference")
    ax.set(xlabel="time (min)", ylabel="HR (BPM)", ylim=(30, 200), xlim=(0, t[-1]),
           title="S5 (held out from every model shown): MAE in brackets")
    ax.title.set_ha("left"); ax.title.set_x(0)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    fig.tight_layout(); fig.savefig(FIG / "stage4_trace_S5.png", dpi=150)

    # ---- markdown
    rows = "\n".join(f"| {m} | {v['all15']:.2f} | {v['test']:.2f} |" for m, v in table.items())
    act_rows = "\n".join(f"| {a} | " + " | ".join(f"{table[m]['per_activity'][a]:.1f}" for m, _ in show) + " |" for a in acts)
    md = f"""# Stage 4 results: DSP vs neural networks

Leave-one-subject-out (LOSO): 15 folds; for held-out subject T, the next subject is the
validation set (epoch selection only), the other 13 train. Inputs normalised per window;
no statistics shared across subjects. MAE in BPM, mean over subjects.
The DSP has no training, but its ~6 parameters were tuned on S1-S3, so "S4-S15" is the
fairest column for it.

| method | all 15 (LOSO) | S4-S15 |
|---|---|---|
{rows}

| model | parameters | MACs / inference (every 2 s) |
|---|---|---|
| Time CNN (PPG + ACC, 8 s @ 32 Hz) | {size['Time CNN'][0]:,} | {size['Time CNN'][1]:,} |
| Spectral CNN (DSP front-end spectra) | {size['Spectral CNN'][0]:,} | {size['Spectral CNN'][1]:,} |

## Per activity (all 15 subjects, LOSO)
| activity | """ + " | ".join(m for m, _ in show) + " |\n|---|" + "---|" * len(show) + f"""
{act_rows}

## INT8 quantization (spectral CNN)
Per-channel int8 weights, int8 activations (calibrated on each fold's training subjects),
int32 accumulators, integer tracker. Windows with the same HR as the float model:
**{res['int8_identical_to_float_pct']:.1f}%**; MAE float {table['Spectral CNN (hybrid) + HMM tracker']['all15']:.2f} ->
int8 {table['Spectral CNN INT8 + HMM (firmware arithmetic)']['all15']:.2f} BPM.

## On the emulated MCU (MAX32652, Renode, 120 s of held-out S5)
Deployed model: the LOSO fold that never saw S5. Both modes are in one firmware image
(runtime switch); instruction counts are exact, time assumes 1 instruction/cycle.

| | DSP mode | INT8 NN mode |
|---|---|---|
| bit-exact vs host build | {emu['dsp']['bit_exact_vs_host']} | {emu['nn']['bit_exact_vs_host']} |
| instructions per sample | {emu['dsp']['instr_per_sample']:.0f} | {emu['nn']['instr_per_sample']:.0f} |
| instructions per 2 s window | {emu['dsp']['instr_per_window']/1e6:.2f} M | {emu['nn']['instr_per_window']/1e6:.2f} M |
| CPU load @ 120 MHz | {emu['dsp']['cpu_load_pct_120MHz']:.2f}% | {emu['nn']['cpu_load_pct_120MHz']:.2f}% |

Network footprint: """ + ", ".join(f"{k}: {v['flash']/1024:.1f} KB flash, {v['ram']/1024:.1f} KB RAM" for k, v in emu["memory"].items()) + f""".

## Cold start (tracker started at the beginning of each activity, first 2 min, S4-S15)
| | MAE |
|---|---|
| DSP evidence | {cold['DSP']:.1f} BPM |
| Spectral CNN evidence | {cold['Spectral CNN']:.1f} BPM |
"""
    (REPORTS / "stage4_results.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
