/* Streaming pipeline: per sample bandpass + NLMS; every 128 samples one estimate. */
#include "hr_estimator.h"
#include "fixed.h"
#include "nn.h"

void hr_init(hr_estimator_t *st) {
    st->mode = HR_MODE_DSP;
    bandpass_init(&st->ppg_bp);
    for (int c = 0; c < HR_NLMS_CH; c++) {
        bandpass_init(&st->acc_bp[c]);
        st->acc_prev[c] = st->acc_cur[c] = 0;
    }
    nlms_init(&st->nlms);
    hmm_init(&st->hmm);
    st->have_acc = 0;
    st->n = 0;
    st->pos = 0;
    for (int i = 0; i < HR_WIN; i++) {
        st->ppg_ring[i] = 0;
        for (int c = 0; c < HR_NLMS_CH; c++) st->acc_ring[c][i] = 0;
    }
}

void hr_set_mode(hr_estimator_t *st, hr_mode_t mode) { st->mode = mode; }

void hr_window_spectra(const hr_estimator_t *st, uint32_t ppg_mag[HR_NSPEC], uint64_t acc_mag[HR_NSPEC]) {
    uint32_t oldest = st->pos;          /* ring is full: next write slot = oldest sample */
    int32_t mean;

    uint64_t mx = spectrum_windowed_max(st->ppg_ring, oldest, &mean);
    spectrum_band(st->ppg_ring, oldest, mean, spectrum_block_shift(mx), ppg_mag, HR_NSPEC);

    /* ACC: one common block shift for all three axes, because their magnitudes are
     * summed (a per-axis scale would change the relative weights). */
    int32_t acc_mean[HR_NLMS_CH];
    uint64_t acc_mx = 0;
    for (int c = 0; c < HR_NLMS_CH; c++) {
        uint64_t m = spectrum_windowed_max(st->acc_ring[c], oldest, &acc_mean[c]);
        if (m > acc_mx) acc_mx = m;
    }
    int acc_shift = spectrum_block_shift(acc_mx);
    for (int k = 0; k < HR_NSPEC; k++) acc_mag[k] = 0;
    for (int c = 0; c < HR_NLMS_CH; c++) {
        static uint32_t m[HR_NSPEC];
        spectrum_band(st->acc_ring[c], oldest, acc_mean[c], acc_shift, m, HR_NSPEC);
        for (int k = 0; k < HR_NSPEC; k++) acc_mag[k] += m[k];
    }
}

/* Per-window scratch (~5 KB). Static, not on the stack: as locals these overflowed the
 * 4 KB MCU stack into .bss and corrupted the estimator state (found in Stage 4 when the
 * emulated firmware stopped matching the host build). Like the FFT buffer, this
 * assumes one estimator instance computes a window at a time. */
static uint32_t ppg_mag[HR_NSPEC];          /* 0.5-8 Hz: HR band + 2nd harmonics */
static uint64_t acc_mag[HR_NSPEC];
static int8_t   nn_in[NN_IN_CH][HR_NBINS];
static int32_t  nn_logits[HR_NBINS], nn_logp[HR_NBINS];

static int estimate_window(hr_estimator_t *st, hr_result_t *out) {
    hr_window_spectra(st, ppg_mag, acc_mag);

    int bin;
    if (st->mode == HR_MODE_NN) {
        int8_t (*in)[HR_NBINS] = nn_in;
        int32_t *logits = nn_logits, *logp = nn_logp;
        nn_make_input(ppg_mag, acc_mag, in);
        nn_forward(&in[0][0], logits);
        nn_logits_to_logp(logits, logp);
        bin = hmm_step_logp(&st->hmm, logp);
    } else {
        acc_mask_apply(ppg_mag, acc_mag);          /* uses the HR band (first HR_NBINS) */
        bin = hmm_step(&st->hmm, ppg_mag);
    }

    out->window_index = (st->n - HR_WIN) / HR_HOP;
    out->bin = (uint16_t)bin;
    out->bpm_x1000 = (uint32_t)(HR_BIN_LO + bin) * HR_BPM_X1000_PER_BIN;
    return 1;
}

int hr_push(hr_estimator_t *st, uint32_t ppg_counts, const int16_t *acc, hr_result_t *out) {
    /* PPG: remove ADC mid-scale, go to Q12, bandpass. */
    int32_t x = ((int32_t)ppg_counts - (int32_t)HR_ADC_MID) * (1 << HR_PPG_SHIFT);
    int32_t ppg = bandpass_step(&st->ppg_bp, x);

    /* ACC 32 -> 64 Hz, causal linear interpolation: on a new sample output the
     * midpoint of previous and new, on the in-between tick repeat the new one. */
    int32_t up[HR_NLMS_CH];
    for (int c = 0; c < HR_NLMS_CH; c++) {
        if (acc) {
            st->acc_cur[c] = (int32_t)acc[c] * (1 << HR_ACC_SHIFT);
            if (!st->have_acc) st->acc_prev[c] = st->acc_cur[c];
            up[c] = (st->acc_prev[c] + st->acc_cur[c]) / 2;   /* exact: both even */
            st->acc_prev[c] = st->acc_cur[c];
        } else {
            up[c] = st->acc_cur[c];
        }
    }
    if (acc) st->have_acc = 1;

    int32_t accf[HR_NLMS_CH];
    for (int c = 0; c < HR_NLMS_CH; c++) accf[c] = bandpass_step(&st->acc_bp[c], up[c]);
    int32_t clean = nlms_step(&st->nlms, ppg, accf);

    st->ppg_ring[st->pos] = clean;
    for (int c = 0; c < HR_NLMS_CH; c++) st->acc_ring[c][st->pos] = accf[c];
    st->pos = (st->pos + 1) % HR_WIN;
    st->n++;

    if (st->n >= HR_WIN && (st->n - HR_WIN) % HR_HOP == 0)
        return estimate_window(st, out);
    return 0;
}
