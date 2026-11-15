/* Streaming pipeline: per sample bandpass + NLMS; every 128 samples one estimate. */
#include "hr_estimator.h"
#include "fixed.h"

void hr_init(hr_estimator_t *st) {
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

static int estimate_window(hr_estimator_t *st, hr_result_t *out) {
    uint32_t oldest = st->pos;          /* ring is full: next write slot = oldest sample */
    uint32_t ppg_mag[HR_NBINS];
    uint64_t acc_mag[HR_NBINS];
    int32_t mean;

    uint64_t mx = spectrum_windowed_max(st->ppg_ring, oldest, &mean);
    spectrum_band(st->ppg_ring, oldest, mean, spectrum_block_shift(mx), ppg_mag);

    /* ACC: one common block shift for all three axes, because their magnitudes are
     * summed (a per-axis scale would change the relative weights). */
    int32_t acc_mean[HR_NLMS_CH];
    uint64_t acc_mx = 0;
    for (int c = 0; c < HR_NLMS_CH; c++) {
        uint64_t m = spectrum_windowed_max(st->acc_ring[c], oldest, &acc_mean[c]);
        if (m > acc_mx) acc_mx = m;
    }
    int acc_shift = spectrum_block_shift(acc_mx);
    for (int k = 0; k < HR_NBINS; k++) acc_mag[k] = 0;
    for (int c = 0; c < HR_NLMS_CH; c++) {
        uint32_t m[HR_NBINS];
        spectrum_band(st->acc_ring[c], oldest, acc_mean[c], acc_shift, m);
        for (int k = 0; k < HR_NBINS; k++) acc_mag[k] += m[k];
    }

    acc_mask_apply(ppg_mag, acc_mag);
    int bin = hmm_step(&st->hmm, ppg_mag);

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
