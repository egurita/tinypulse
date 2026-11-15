# PPG heart-rate pipeline: DSP vs NN, embedded C, multi-level tests

Wrist-PPG heart-rate estimation on PPG-DaLiA, compared across a classical DSP pipeline
and a neural network, ported to fixed-point embedded C (emulated Cortex-M), tested at
the unit / regression / equivalence / software-in-the-loop levels, plus a live webcam demo.

> Work in progress: Stage 1 (classical DSP) done.

## Stage 1 results: classical DSP (Python)

```
PPG 64 Hz ─► bandpass 0.5–4 Hz ─► NLMS motion cancel (ACC ref) ─► 8 s FFT ─► ACC spectral mask ─► HMM tracker ─► BPM
ACC 32 Hz ─► resample 64 Hz, bandpass ─┘──────────────────────────────────────┘
```

Mean absolute error vs ECG reference, parameters tuned on S1–S3, **evaluated on held-out S4–S15**:

| variant | MAE (BPM) |
|---|---|
| naive FFT peak | 20.8 |
| + bandpass | 19.5 |
| + NLMS motion cancellation | 18.6 |
| + HMM tracker | 13.4 |
| + ACC spectral mask (final) | **9.8** |
| final without NLMS | 11.4 |

Per activity, naive → final: sitting 3.1 → 2.4, cycling 30.5 → 10.2, walking 30.6 → 19.9, stairs 39.2 → 27.2.
Full tables: [`reports/stage1_results.md`](reports/stage1_results.md).

![per activity](reports/figures/stage1_per_activity.png)
![trace](reports/figures/stage1_trace_S5.png)

## Layout
```
python/ppg_hr/     library code (data loading, DSP, models)
python/scripts/    runnable scripts (data prep, plots, evaluation)
firmware/          C firmware (Stage 2)
tests/python/      pytest suites
reports/figures/   generated plots
data/              dataset (git-ignored)
```

## Setup
```bash
uv sync
# download PPG-DaLiA (~2.7 GB) from UCI into data/raw/ppg_dalia.zip, then:
uv run python python/scripts/prepare_data.py
uv run pytest
uv run python python/scripts/plot_raw.py S1
uv run python python/scripts/run_stage1.py   # all variants, all subjects -> reports/
```
