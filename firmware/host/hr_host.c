/* Host-only batch wrappers around the firmware code, exported from libhr for
 * Python (ctypes): whole-recording runs and per-block runs for equivalence tests.
 * Never linked into the MCU image.
 */
#include <stddef.h>
#include "hr_estimator.h"
#include "nn.h"

size_t hr_state_size(void) { return sizeof(hr_estimator_t); }

/* acc sample k (3 int16) is delivered together with ppg sample 2k.
 * mode: 0 = DSP, 1 = INT8 NN. If spec_out != NULL, each window's spectra are also
 * written there (HR_NSPEC PPG values, then HR_NSPEC summed-ACC values, as uint64).
 * Returns the number of estimates written to bins_out. */
size_t hr_run_mode(int mode, const uint32_t *ppg, size_t n, const int16_t *acc, size_t n_acc,
                   uint16_t *bins_out, size_t max_out, uint64_t *spec_out) {
    static hr_estimator_t st;
    hr_init(&st);
    hr_set_mode(&st, mode ? HR_MODE_NN : HR_MODE_DSP);
    size_t n_out = 0;
    for (size_t i = 0; i < n; i++) {
        const int16_t *a = (i % 2 == 0 && i / 2 < n_acc) ? &acc[3 * (i / 2)] : NULL;
        hr_result_t r;
        if (hr_push(&st, ppg[i], a, &r) && n_out < max_out) {
            if (spec_out) {
                uint32_t pm[HR_NSPEC];
                uint64_t *dst = &spec_out[n_out * 2 * HR_NSPEC];
                hr_window_spectra(&st, pm, &dst[HR_NSPEC]);
                for (int k = 0; k < HR_NSPEC; k++) dst[k] = pm[k];
            }
            bins_out[n_out++] = r.bin;
        }
    }
    return n_out;
}

size_t hr_run(const uint32_t *ppg, size_t n, const int16_t *acc, size_t n_acc,
              uint16_t *bins_out, size_t max_out) {
    return hr_run_mode(0, ppg, n, acc, n_acc, bins_out, max_out, NULL);
}

void hr_nn_forward(const int8_t *in, int32_t *logits) {
    nn_forward(in, logits);
}

void hr_bandpass_run(const int32_t *x, size_t n, int32_t *y) {
    bandpass_t f;
    bandpass_init(&f);
    for (size_t i = 0; i < n; i++) y[i] = bandpass_step(&f, x[i]);
}

void hr_nlms_run(const int32_t *d, const int32_t *u, size_t n, int32_t *e) {
    static nlms_t f;
    nlms_init(&f);
    for (size_t i = 0; i < n; i++) e[i] = nlms_step(&f, d[i], &u[3 * i]);
}

/* Spectrum of one chronological window x[HR_WIN] with its own block shift. */
void hr_spectrum_run(const int32_t *x, uint32_t *mag) {
    int32_t mean;
    uint64_t mx = spectrum_windowed_max(x, 0, &mean);
    spectrum_band(x, 0, mean, spectrum_block_shift(mx), mag, HR_NBINS);
}

int32_t hr_log2_q16(uint64_t x) { return log2_q16(x); }
