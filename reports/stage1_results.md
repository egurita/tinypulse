# Stage 1 results: classical DSP

MAE in BPM vs the ECG-derived reference (8 s windows, 2 s shift).
Parameters tuned on S1-S3 only; **test = held-out S4-S15**.
Overall MAE = mean over subjects of each subject's MAE.

## Ablation (each row adds one block)
| variant | dev S1-S3 | test S4-S15 | all 15 |
|---|---|---|---|
| 1 naive (raw FFT peak) | 16.02 | **20.77** | 19.82 |
| 2 + bandpass | 15.03 | **19.54** | 18.64 |
| 3 + NLMS motion cancel | 13.88 | **18.56** | 17.63 |
| 4 + HMM tracker | 8.54 | **13.36** | 12.40 |
| 5 + ACC spectral mask (final) | 7.61 | **9.79** | 9.35 |
| final without NLMS | 7.98 | **11.36** | 10.69 |

## Per activity, test subjects (pooled windows, MAE in BPM)
| activity | naive | +BP | +NLMS | +HMM | final | final w/o NLMS |
|---|---|---|---|---|---|---|
| sitting | 3.1 | 3.0 | 3.1 | 2.4 | 2.4 | 2.4 |
| stairs | 39.2 | 38.3 | 34.9 | 32.9 | 27.2 | 27.1 |
| table soccer | 34.7 | 32.2 | 30.5 | 25.9 | 24.2 | 26.1 |
| cycling | 30.5 | 29.5 | 24.1 | 23.5 | 10.2 | 22.5 |
| driving | 15.4 | 14.1 | 14.1 | 6.2 | 5.3 | 5.1 |
| lunch | 16.1 | 14.8 | 14.5 | 7.7 | 5.0 | 5.0 |
| walking | 30.6 | 29.4 | 28.3 | 27.1 | 19.9 | 23.2 |
| working | 9.9 | 9.3 | 9.1 | 4.2 | 3.7 | 3.9 |
| all windows | 20.6 | 19.4 | 18.4 | 13.5 | 9.8 | 11.4 |

## Per subject (naive vs final)
| subject | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 | S11 | S12 | S13 | S14 | S15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| naive | 13.2 | 15.3 | 19.5 | 21.8 | 45.9 | 27.8 | 9.6 | 17.4 | 26.4 | 15.5 | 26.9 | 15.6 | 10.7 | 16.2 | 15.5 |
| final | 7.5 | 10.1 | 5.2 | 13.4 | 14.7 | 8.3 | 4.5 | 9.1 | 15.4 | 4.1 | 16.6 | 8.4 | 6.8 | 7.2 | 9.1 |
