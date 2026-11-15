# Firmware (Stage 2)

Fixed-point, streaming C99 implementation of the heart-rate pipeline for the
**MAX32652** (ADI Cortex-M4F), emulated in Renode. No malloc, no libc, no floats.

```
include/        public headers: hr_config.h (Q formats, parameters), hr_dsp.h, hr_estimator.h,
                hal_i2c.h (I2C HAL), max30102.h (register map + driver API), fixed.h
src/            bandpass (DF-I biquads, Q29), nlms, spectrum (Hann + Q31 FFT, block floating
                point), track (ACC mask, log2, HMM), hr_estimator (streaming glue), max30102 driver
gen/            GENERATED from Python: hr_tables.[ch] (coefficients, window, twiddles, log2),
                test_vector.[ch] (120 s of real data + expected host output)
mock/           register-level MAX30102 model behind the same HAL
host/           batch wrappers exported to Python (ctypes)
tests/          C unit tests (DSP vs libm, driver vs mock)
port/max32652/  startup, linker script, UART, benchmark main
```

```bash
make test      # C unit tests (host, UBSan)
make native    # build/native/libhr.{dylib,so} for Python
make arm       # build/arm/fw.elf for the MAX32652 (needs arm-none-eabi-gcc)
make tables    # regenerate gen/hr_tables.[ch]
```

Run in the emulator / full report: `uv run python python/scripts/run_stage2.py`
(needs Renode; set `RENODE=/path/to/renode` if it is not on PATH).
