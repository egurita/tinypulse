# Stage 4 results: DSP vs neural networks

Leave-one-subject-out (LOSO): 15 folds; for held-out subject T, the next subject is the
validation set (epoch selection only), the other 13 train. Inputs normalised per window;
no statistics shared across subjects. MAE in BPM, mean over subjects.
The DSP has no training, but its ~6 parameters were tuned on S1-S3, so "S4-S15" is the
fairest column for it.

| method | all 15 (LOSO) | S4-S15 |
|---|---|---|
| DSP (Stage 1/2) | 9.29 | 9.72 |
| Time CNN, argmax | 6.98 | 7.62 |
| Time CNN + HMM tracker | 5.89 | 6.25 |
| Spectral CNN (hybrid), argmax | 8.05 | 8.49 |
| Spectral CNN (hybrid) + HMM tracker | 5.71 | 5.99 |
| Spectral CNN INT8 + HMM (firmware arithmetic) | 5.73 | 6.02 |

| model | parameters | MACs / inference (every 2 s) |
|---|---|---|
| Time CNN (PPG + ACC, 8 s @ 32 Hz) | 33,169 | 1,498,176 |
| Spectral CNN (DSP front-end spectra) | 3,106 | 327,248 |

## Per activity (all 15 subjects, LOSO)
| activity | DSP (Stage 1/2) | Time CNN + HMM tracker | Spectral CNN (hybrid) + HMM tracker | Spectral CNN INT8 + HMM (firmware arithmetic) |
|---|---|---|---|---|
| sitting | 2.6 | 3.0 | 3.1 | 3.1 |
| stairs | 27.9 | 15.3 | 13.9 | 13.9 |
| table soccer | 23.3 | 8.5 | 12.0 | 12.0 |
| cycling | 11.3 | 7.7 | 10.0 | 10.0 |
| driving | 5.1 | 3.8 | 3.9 | 3.9 |
| lunch | 4.9 | 4.0 | 3.9 | 3.9 |
| walking | 16.4 | 9.0 | 7.7 | 7.6 |
| working | 3.8 | 2.9 | 3.0 | 3.0 |

## INT8 quantization (spectral CNN)
Per-channel int8 weights, int8 activations (calibrated on each fold's training subjects),
int32 accumulators, integer tracker. Windows with the same HR as the float model:
**85.7%**; MAE float 5.71 ->
int8 5.73 BPM.

## On the emulated MCU (MAX32652, Renode, 120 s of held-out S5)
Deployed model: the LOSO fold that never saw S5. Both modes are in one firmware image
(runtime switch); instruction counts are exact, time assumes 1 instruction/cycle.

| | DSP mode | INT8 NN mode |
|---|---|---|
| bit-exact vs host build | True | True |
| instructions per sample | 2475 | 2475 |
| instructions per 2 s window | 3.58 M | 7.09 M |
| CPU load @ 120 MHz | 1.62% | 3.09% |

Network footprint: nn.c (int8 kernels): 0.8 KB flash, 3.5 KB RAM, nn_model.c (weights): 3.6 KB flash, 0.0 KB RAM.

## Cold start (tracker started at the beginning of each activity, first 2 min, S4-S15)
| | MAE |
|---|---|
| DSP evidence | 17.3 BPM |
| Spectral CNN evidence | 8.8 BPM |
