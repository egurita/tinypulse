# Stage 1 results: classical DSP

MAE in BPM vs the ECG-derived reference (8 s windows, 2 s shift).
Parameters tuned on S1-S3 only; **test = held-out S4-S15**.
Overall MAE = mean over subjects of each subject's MAE.

## Ablation (each row adds one block)
| variant | dev S1-S3 | test S4-S15 | all 15 |
|---|---|---|---|
| 1 naive (raw FFT peak) | 16.02 | **20.77** | 19.82 |
| 2 + bandpass | 15.02 | **19.54** | 18.64 |
| 3 + NLMS motion cancel | 13.86 | **18.54** | 17.61 |
| 4 + HMM tracker | 8.56 | **13.32** | 12.37 |
| 5 + ACC spectral mask (final) | 7.55 | **9.72** | 9.29 |
| final without NLMS | 7.95 | **11.29** | 10.62 |

## Per activity, test subjects (pooled windows, MAE in BPM)
| activity | naive | +BP | +NLMS | +HMM | final | final w/o NLMS |
|---|---|---|---|---|---|---|
| sitting | 3.1 | 3.0 | 3.1 | 2.3 | 2.4 | 2.4 |
| stairs | 39.2 | 38.3 | 34.8 | 32.5 | 27.1 | 27.0 |
| table soccer | 34.7 | 32.2 | 30.3 | 25.7 | 23.7 | 26.0 |
| cycling | 30.5 | 29.5 | 24.3 | 23.5 | 10.0 | 22.3 |
| driving | 15.4 | 14.1 | 14.1 | 6.2 | 5.3 | 5.1 |
| lunch | 16.1 | 14.8 | 14.5 | 7.8 | 5.0 | 5.0 |
| walking | 30.6 | 29.4 | 28.2 | 27.3 | 20.2 | 22.9 |
| working | 9.9 | 9.3 | 9.2 | 4.2 | 3.7 | 3.9 |
| all windows | 20.6 | 19.4 | 18.4 | 13.4 | 9.7 | 11.3 |

## Per subject (naive vs final)
| subject | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 | S11 | S12 | S13 | S14 | S15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| naive | 13.2 | 15.3 | 19.5 | 21.8 | 45.9 | 27.8 | 9.6 | 17.4 | 26.4 | 15.5 | 26.9 | 15.6 | 10.7 | 16.2 | 15.5 |
| final | 7.5 | 10.0 | 5.2 | 13.3 | 14.5 | 8.3 | 4.5 | 9.0 | 15.4 | 4.0 | 16.3 | 8.4 | 6.6 | 7.5 | 9.0 |
