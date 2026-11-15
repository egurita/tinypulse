"""Stage 2 report: fixed-point C firmware vs float Python, emulated MCU benchmark.

  1. C unit tests (make test)
  2. Fixed (C, ctypes) vs float (Python) on all 15 subjects
  3. Cold-start behaviour of the firmware (estimator started mid-session)
  4. Laptop latency: native C vs Python
  5. Renode (MAX32652, Cortex-M4): bit-exactness vs host + instruction counts
  6. Flash / RAM from the ARM build

Writes reports/stage2_results.md, reports/stage2_results.json, reports/figures/stage2_*.png.
Usage:  uv run python python/scripts/run_stage2.py
"""

import json
import re
import subprocess
import time

import matplotlib.pyplot as plt
import numpy as np

from ppg_hr import emulator, firmware as fw
from ppg_hr.data import ACTIVITY_NAMES, REPO_ROOT, SUBJECTS, activity_runs, load_subject
from ppg_hr.evaluate import window_activity
from ppg_hr.pipeline import DEV_SUBJECTS, FINAL, acc_counts, estimate_hr, ppg_counts

REPORTS, FIG, FW = REPO_ROOT / "reports", REPO_ROOT / "reports" / "figures", REPO_ROOT / "firmware"
TEST = [s for s in SUBJECTS if s not in DEV_SUBJECTS]
MCU_HZ = 120e6               # MAX32652 core clock (Renode PerformanceInMips = 120)
INK, MUTED, BLUE, ORANGE = "#1f2328", "#8c959f", "#2a78d6", "#eb6834"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                     "grid.color": "#eaeef2", "font.size": 9})


def sh(*cmd, cwd=FW):
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True).stdout


def c_unit_tests():
    sh("make", "-s", "native")
    out = sh("./build/native/test_firmware")
    checks, fails = map(int, re.search(r"(\d+) checks, (\d+) failures", out).groups())
    return {"checks": checks, "failures": fails, "tests": out.count(" ok")}


def equivalence():
    rows, per_act = {}, {}
    errs = {"float": {}, "fixed": {}}
    t_c = t_py = 0.0
    n_samples = 0
    for sid in SUBJECTS:
        s = load_subject(sid)
        t = time.perf_counter(); f = estimate_hr(s, FINAL); t_py += time.perf_counter() - t
        t = time.perf_counter(); q = fw.run_subject(s); t_c += time.perf_counter() - t
        n_samples += len(s.ppg)
        rows[sid] = {"identical_pct": float(np.mean(f == q) * 100), "max_diff_bpm": float(np.abs(f - q).max()),
                     "mae_float": float(np.abs(f - s.hr_ref).mean()), "mae_fixed": float(np.abs(q - s.hr_ref).mean())}
        act = window_activity(s)
        for name, est in (("float", f), ("fixed", q)):
            errs[name].setdefault("act", []).append(act)
            errs[name].setdefault("err", []).append(np.abs(est - s.hr_ref))
        per_act.setdefault("sid", []).append(sid)
    for name in ("float", "fixed"):
        e = np.concatenate([x for x, sid in zip(errs[name]["err"], SUBJECTS) if sid in TEST])
        a = np.concatenate([x for x, sid in zip(errs[name]["act"], SUBJECTS) if sid in TEST])
        per_act[name] = {ACTIVITY_NAMES[k]: float(e[a == k].mean()) for k in ACTIVITY_NAMES if k}
    total = sum(len(load_subject(s).hr_ref) for s in SUBJECTS)
    ident = sum(rows[s]["identical_pct"] / 100 * len(load_subject(s).hr_ref) for s in SUBJECTS)
    return {
        "per_subject": rows,
        "identical_windows": int(round(ident)), "total_windows": total,
        "test_mae_float": float(np.mean([rows[s]["mae_float"] for s in TEST])),
        "test_mae_fixed": float(np.mean([rows[s]["mae_fixed"] for s in TEST])),
        "per_activity_test": {k: (per_act["float"][k], per_act["fixed"][k]) for k in per_act["float"]},
        "ns_per_sample_c": t_c / n_samples * 1e9, "ns_per_sample_python": t_py / n_samples * 1e9,
    }


def cold_start():
    """Start the firmware fresh at the beginning of each activity block (as if the
    watch were put on then); MAE of the first 2 min vs the same windows when the
    estimator has been running since the start of the recording."""
    cold, warm = [], []
    for sid in TEST:
        s = load_subject(sid)
        ppg, acc = ppg_counts(s).astype(np.uint32), acc_counts(s)
        full = fw.run_subject(s)
        for act, a, b in activity_runs(s.activity):
            if act == 0 or b - a < 120:
                continue
            k0 = int(a // 2)
            n = 60 * 128 + 512
            seg_p = np.ascontiguousarray(ppg[k0 * 128: k0 * 128 + n])
            seg_a = np.ascontiguousarray(acc[k0 * 64: (k0 * 128 + n) // 2])
            bins = np.zeros(70, dtype=np.uint16)
            k = fw.lib().hr_run(seg_p, len(seg_p), seg_a, len(seg_a), bins, len(bins))
            k = min(k, 60, len(s.hr_ref) - k0)
            est = (fw.BIN_LO + bins[:k]) * fw.BPM_PER_BIN
            cold.append(np.abs(est - s.hr_ref[k0:k0 + k]).mean())
            warm.append(np.abs(full[k0:k0 + k] - s.hr_ref[k0:k0 + k]).mean())
    return {"segments": len(cold), "mae_cold": float(np.mean(cold)), "mae_warm": float(np.mean(warm))}


def emulated():
    emulator.build_arm()
    meta = json.loads((FW / "gen" / "test_vector.json").read_text())
    t = time.perf_counter()
    run = emulator.run(virtual_seconds=3)
    wall = time.perf_counter() - t
    hr = [tuple(map(int, l.split()[1:])) for l in run.uart.splitlines() if l.startswith("HR ")]
    done = dict(kv.split("=") for kv in run.uart.strip().splitlines()[-1].split()[1:])
    bins = [b for _, b, _ in hr]
    marks = run.marks
    push, window, fifo = [], [], []
    for (i1, c1), (i2, c2) in zip(marks, marks[1:]):
        if i1 == 1 and i2 == 2: push.append(c2 - c1)
        if i1 == 1 and i2 == 3: window.append(c2 - c1)
        if i1 == 4 and i2 == 5: fifo.append(c2 - c1)
    total = marks[-1][1]
    seconds_of_data = meta["n_samples"] / 64
    push, window, fifo = map(np.array, (push, window, fifo))
    per_second = 64 * push.mean() + 0.5 * window.mean() + 4 * fifo.mean()
    return {
        "uart_done": done, "bit_exact_vs_host": bins == meta["expected_bins"],
        "estimates": len(bins), "wall_s": wall,
        "instr_per_sample_mean": float(push.mean()), "instr_per_sample_max": int(push.max()),
        "instr_per_window_mean": float(window.mean()), "instr_per_window_max": int(window.max()),
        "instr_per_fifo_read_mean": float(fifo.mean()),
        "instr_total_benchmark": int(total), "seconds_of_data": seconds_of_data,
        "instr_per_second_of_data": float(per_second),
        "cpu_load_pct_at_120MHz_cpi1": float(per_second / MCU_HZ * 100),
        "window_latency_ms_at_120MHz_cpi1": float(window.max() / MCU_HZ * 1e3),
        "_push_series": push.tolist(), "_window_series": window.tolist(),
    }


def memory():
    """Per-module flash/RAM: each source compiled alone for the Cortex-M4."""
    flags = ["-mcpu=cortex-m4", "-mthumb", "-mfloat-abi=soft", "-O2", "-ffreestanding",
             "-ffunction-sections", "-fdata-sections", "-Iinclude", "-Igen", "-Imock", "-Iport/max32652", "-c"]
    modules = {"bandpass": "src/bandpass.c", "nlms": "src/nlms.c", "spectrum + FFT": "src/spectrum.c",
               "tracker (mask, HMM, log2)": "src/track.c", "estimator glue": "src/hr_estimator.c",
               "MAX30102 driver": "src/max30102.c", "tables (coefs, Hann, twiddles, log2)": "gen/hr_tables.c",
               "startup + UART + libc": None}
    out = {}
    tmp = FW / "build" / "arm" / "mod"
    tmp.mkdir(parents=True, exist_ok=True)
    for name, src in modules.items():
        srcs = [src] if src else ["port/max32652/startup.c", "port/max32652/uart.c", "port/max32652/mini_libc.c"]
        text = data = bss = 0
        for s in srcs:
            obj = tmp / (s.replace("/", "_") + ".o")
            sh("arm-none-eabi-gcc", *flags, s, "-o", str(obj))
            t, d, b = map(int, sh("arm-none-eabi-size", str(obj)).splitlines()[1].split()[:3])
            text, data, bss = text + t, data + d, bss + b
        out[name] = {"flash": text + data, "ram": data + bss}
    # Linked image totals, and what is benchmark-only (test vector, mock, main).
    syms = sh("arm-none-eabi-nm", "-S", "--size-sort", "build/arm/fw.elf")
    size = {l.split()[3]: int(l.split()[1], 16) for l in syms.splitlines() if len(l.split()) == 4}
    image = sh("arm-none-eabi-size", "build/arm/fw.elf").splitlines()[1].split()
    text, data, bss = map(int, image[:3])
    bench_only_flash = sum(v for k, v in size.items() if k.startswith("tv_"))
    out["estimator state (hr_estimator_t)"] = {"flash": 0, "ram": size["est"]}
    return {"modules": out, "image_flash": text + data, "image_ram_incl_stack": data + bss,
            "test_vector_flash": bench_only_flash,
            "largest_ram": sorted(((k, v) for k, v in size.items() if k in ("fft_re", "fft_im", "est", "sensor")),
                                  key=lambda kv: -kv[1])}


def figures(emu, mem):
    fig, ax = plt.subplots(figsize=(11, 3.4))
    push = np.array(emu["_push_series"])
    win = np.array(emu["_window_series"])
    is_win = np.zeros(len(push) + len(win), bool)
    is_win[511::128] = True                       # sample index of each window's last sample
    t = np.arange(len(is_win)) / 64
    ax.plot(t[~is_win], push, color=BLUE, lw=0.6, label=f"per sample: {push.mean():.0f} (4 bandpasses + NLMS)")
    ax.plot(t[is_win], win, "o", ms=3.5, color=ORANGE,
            label=f"window, every 2 s: {win.mean()/1e6:.2f} M (4 FFTs + mask + HMM)")
    ax.set(yscale="log", xlabel="seconds of data", ylabel="instructions",
           title="Emulated MAX32652 (Cortex-M4): instructions per hr_push() call")
    ax.title.set_ha("left"); ax.title.set_x(0)
    ax.set_ylim(1e3, 1e7)
    ax.legend(frameon=False, loc="center right")
    fig.tight_layout(); fig.savefig(FIG / "stage2_instructions.png", dpi=150)

    names = list(mem["modules"])
    fl = [mem["modules"][n]["flash"] / 1024 for n in names]
    ra = [mem["modules"][n]["ram"] / 1024 for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, vals, title, col in ((axes[0], fl, "Flash (KB)", BLUE), (axes[1], ra, "Static RAM (KB)", INK)):
        y = np.arange(len(names))
        ax.barh(y, vals, color=col, height=0.6)
        for yi, v in zip(y, vals):
            ax.text(v + 0.2, yi, f"{v:.1f}", va="center", fontsize=8)
        ax.set(yticks=y, yticklabels=names, title=title); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    fig.suptitle("Firmware footprint per module (Cortex-M4, -O2; benchmark data excluded)", x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(FIG / "stage2_memory.png", dpi=150)


def main():
    res = {"c_unit_tests": c_unit_tests()}
    print("C unit tests:", res["c_unit_tests"])
    res["equivalence"] = eq = equivalence()
    print(f"fixed vs float: {eq['identical_windows']}/{eq['total_windows']} identical")
    res["cold_start"] = cs = cold_start()
    print("cold start:", cs)
    res["emulator"] = emu = emulated()
    print({k: v for k, v in emu.items() if not k.startswith("_")})
    res["memory"] = mem = memory()
    print({k: v for k, v in mem.items() if k != "modules"})
    figures(emu, mem)

    core = [n for n in mem["modules"] if n not in ("MAX30102 driver", "startup + UART + libc")]
    core_flash = sum(mem["modules"][n]["flash"] for n in core)
    core_ram = sum(mem["modules"][n]["ram"] for n in core)
    pa = eq["per_activity_test"]
    md = f"""# Stage 2 results: fixed-point C firmware

Target: **MAX32652** (ADI, Cortex-M4F @ 120 MHz) emulated in Renode 1.17; firmware is
integer-only C99 (no FPU use, no malloc, no libc), sample-by-sample streaming.

## Fixed point vs float
| | value |
|---|---|
| windows with identical HR (all 15 subjects) | **{eq['identical_windows']} / {eq['total_windows']} ({eq['identical_windows']/eq['total_windows']*100:.2f}%)** |
| test MAE, float Python | {eq['test_mae_float']:.3f} BPM |
| test MAE, fixed-point C | {eq['test_mae_fixed']:.3f} BPM |

Per activity (test S4-S15, float / fixed): """ + ", ".join(f"{k} {v[0]:.1f}/{v[1]:.1f}" for k, v in pa.items()) + f"""

## Emulated MCU (Renode, MAX32652)
120 s of held-out S5 (stairs) streamed through mock MAX30102 -> I2C HAL -> driver -> estimator.

| | value |
|---|---|
| HR outputs bit-exact vs host build | **{emu['bit_exact_vs_host']}** ({emu['estimates']} estimates, {emu['uart_done']['samples_lost']} samples lost) |
| instructions per sample (bandpass x4 + NLMS) | mean {emu['instr_per_sample_mean']:.0f}, max {emu['instr_per_sample_max']} |
| instructions per window (4 FFTs + mask + HMM) | mean {emu['instr_per_window_mean']/1e6:.2f} M, max {emu['instr_per_window_max']/1e6:.2f} M |
| instructions per driver FIFO read (16 samples) | {emu['instr_per_fifo_read_mean']:.0f} |
| instructions per second of data | {emu['instr_per_second_of_data']/1e6:.2f} M |
| CPU load at 120 MHz, assuming 1 instr/cycle | **{emu['cpu_load_pct_at_120MHz_cpi1']:.1f}%** |
| worst-case window latency at 120 MHz, 1 instr/cycle | {emu['window_latency_ms_at_120MHz_cpi1']:.1f} ms |

Honesty note: Renode counts **instructions exactly** but does not model cycle timing
(pipeline stalls, flash wait states, 64-bit multiply/divide latency). Real Cortex-M4
CPI is typically 1.2-2 for this kind of code, so real load would be ~1.2-2x the figure
above. Time numbers are estimates; instruction counts are measured.

## Memory (Cortex-M4, -O2)
| | flash | static RAM |
|---|---|---|
""" + "\n".join(f"| {n} | {v['flash']/1024:.1f} KB | {v['ram']/1024:.1f} KB |" for n, v in mem['modules'].items()) + f"""
| **HR pipeline total** | **{core_flash/1024:.1f} KB** | **{core_ram/1024:.1f} KB** |

Linked benchmark image: {mem['image_flash']/1024:.1f} KB flash (of which {mem['test_vector_flash']/1024:.1f} KB is
the embedded test recording), {mem['image_ram_incl_stack']/1024:.1f} KB RAM including a 4 KB stack.
The 16 KB FFT scratch buffer (2 x 2048 x int32) is the largest item; a real-input FFT
would halve it.

## Laptop (Apple M4) latency
| | per sample |
|---|---|
| native C (fixed point) | {eq['ns_per_sample_c']:.0f} ns |
| Python float reference | {eq['ns_per_sample_python']:.0f} ns |

## Finding: cold start
Started fresh at the beginning of an activity (watch put on mid-session), the first 2
minutes have MAE **{cs['mae_cold']:.1f} BPM** vs **{cs['mae_warm']:.1f} BPM** on the same windows
when the estimator has been running since the start ({cs['segments']} segments, test subjects).
The tracker can lock onto a motion or sub-harmonic peak when it starts during motion
and only moves ~3.75 BPM per 2 s. A "teleport" transition was tried and did not help
(dev cold-start MAE unchanged), so it is left as a known limitation for Stage 4.

## C unit tests
{res['c_unit_tests']['tests']} tests, {res['c_unit_tests']['checks']} checks, {res['c_unit_tests']['failures']} failures
(UBSan on macOS; ASan hangs at startup on this macOS release, CI on Linux will add it).
"""
    (REPORTS / "stage2_results.md").write_text(md)
    clean = {k: ({kk: vv for kk, vv in v.items() if not kk.startswith("_")} if isinstance(v, dict) else v)
             for k, v in res.items()}
    (REPORTS / "stage2_results.json").write_text(json.dumps(clean, indent=2, default=str))
    print("wrote reports/stage2_results.md")


if __name__ == "__main__":
    main()
