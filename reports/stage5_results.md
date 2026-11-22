# Stage 5 results: heart rate from the laptop webcam

2 recordings of the author with a reference heart rate. Cell = median HR over the
recording (after the first 10 s), in brackets the MAE of the 8 s-window estimates vs the
reference, in BPM. Only per-frame skin-colour averages are stored, never video.

| recording | length | ref | green | POS | POS + motion | firmware DSP (C) | firmware INT8 NN (C) |
|---|---|---|---|---|---|---|---|
| still | 60 s, 29 fps | 80 | 38 (41.3) | 36 (44.6) | 103 (18.7) | 84 (5.5) | 86 (6.2) |
| talking | 60 s, 29 fps | 91 | 39 (52.4) | 45 (49.6) | 120 (28.6) | 120 (28.0) | 84 (10.2) |
| **mean window MAE** | | | **46.9** | **47.1** | **23.7** | **16.8** | **8.2** |
