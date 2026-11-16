# Stage 2 results: fixed-point C firmware

Target: **MAX32652** (ADI, Cortex-M4F @ 120 MHz) emulated in Renode 1.17; firmware is
integer-only C99 (no FPU use, no malloc, no libc), sample-by-sample streaming.

## Fixed point vs float
| | value |
|---|---|
| windows with identical HR (all 15 subjects) | **64652 / 64697 (99.93%)** |
| test MAE, float Python | 9.725 BPM |
| test MAE, fixed-point C | 9.724 BPM |

Per activity (test S4-S15, float / fixed): sitting 2.4/2.4, stairs 27.1/27.1, table soccer 23.7/23.7, cycling 10.0/10.0, driving 5.3/5.3, lunch 5.0/5.0, walking 20.2/20.2, working 3.7/3.7

## Emulated MCU (Renode, MAX32652)
120 s of held-out S5 (stairs) streamed through mock MAX30102 -> I2C HAL -> driver -> estimator.

| | value |
|---|---|
| HR outputs bit-exact vs host build | **True** (57 estimates, 0 samples lost) |
| instructions per sample (bandpass x4 + NLMS) | mean 2475, max 2502 |
| instructions per window (4 FFTs + mask + HMM) | mean 3.58 M, max 3.59 M |
| instructions per driver FIFO read (16 samples) | 2803 |
| instructions per second of data | 1.96 M |
| CPU load at 120 MHz, assuming 1 instr/cycle | **1.6%** |
| worst-case window latency at 120 MHz, 1 instr/cycle | 29.9 ms |

Honesty note: Renode counts **instructions exactly** but does not model cycle timing
(pipeline stalls, flash wait states, 64-bit multiply/divide latency). Real Cortex-M4
CPI is typically 1.2-2 for this kind of code, so real load would be ~1.2-2x the figure
above. Time numbers are estimates; instruction counts are measured.

## Memory (Cortex-M4, -O2)
| | flash | static RAM |
|---|---|---|
| bandpass | 0.2 KB | 0.0 KB |
| nlms | 0.4 KB | 0.0 KB |
| spectrum + FFT | 1.0 KB | 16.0 KB |
| tracker (mask, HMM, log2) | 0.7 KB | 0.0 KB |
| estimator glue | 0.7 KB | 5.1 KB |
| MAX30102 driver | 0.8 KB | 0.0 KB |
| tables (coefs, Hann, twiddles, log2) | 10.1 KB | 0.0 KB |
| startup + UART + libc | 0.5 KB | 0.0 KB |
| estimator state (hr_estimator_t) | 0.0 KB | 9.1 KB |
| **HR pipeline total** | **13.0 KB** | **30.2 KB** |

Linked benchmark image: 73.4 KB flash (of which 52.7 KB is
the embedded test recording), 38.4 KB RAM including a 4 KB stack.
The 16 KB FFT scratch buffer (2 x 2048 x int32) is the largest item; a real-input FFT
would halve it.

## Laptop (Apple M4) latency
| | per sample |
|---|---|
| native C (fixed point) | 981 ns |
| Python float reference | 2527 ns |

## Finding: cold start
Started fresh at the beginning of an activity (watch put on mid-session), the first 2
minutes have MAE **17.1 BPM** vs **12.6 BPM** on the same windows
when the estimator has been running since the start (93 segments, test subjects).
The tracker can lock onto a motion or sub-harmonic peak when it starts during motion
and only moves ~3.75 BPM per 2 s. A "teleport" transition was tried and did not help
(dev cold-start MAE unchanged), so it is left as a known limitation for Stage 4.

## C unit tests
21 tests, 89 checks, 0 failures
(UBSan on macOS; ASan hangs at startup on this macOS release, CI on Linux will add it).
