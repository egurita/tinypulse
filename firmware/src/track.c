/* Peak selection: ACC spectral mask + Viterbi-style HMM tracker, integer-only.
 *
 * The float reference works in natural log; here scores are in log2 (Q16) and the
 * penalty is converted (0.5 nats = 0.7213 bits), which gives the same argmax.
 */
#include "hr_dsp.h"
#include "fixed.h"

/* log2(x) in Q16 for x > 0: integer part from the leading-one position (CLZ),
 * fractional part from a 257-entry table of log2(1 + i/256), linearly interpolated.
 * Max error ~2e-6 (checked against libm in the unit tests). */
int32_t log2_q16(uint64_t x) {
    int n = 63 - __builtin_clzll(x);
    uint64_t norm = x << (63 - n);                       /* 1.xxx with the 1 at bit 63 */
    uint32_t idx  = (uint32_t)(norm >> (63 - HR_LOG2_TBL_BITS)) & ((1u << HR_LOG2_TBL_BITS) - 1);
    uint32_t frac = (uint32_t)(norm >> (63 - HR_LOG2_TBL_BITS - 16)) & 0xFFFF;
    int32_t lo = hr_log2_tbl_q16[idx], hi = hr_log2_tbl_q16[idx + 1];
    return (n << 16) + lo + (int32_t)(((int64_t)(hi - lo) * frac) >> 16);
}

void acc_mask_apply(uint32_t ppg_mag[HR_NBINS], const uint64_t acc_mag[HR_NBINS]) {
    uint64_t acc_max = 0;
    for (int k = 0; k < HR_NBINS; k++) if (acc_mag[k] > acc_max) acc_max = acc_mag[k];
    if (acc_max == 0) return;
    for (int k = 0; k < HR_NBINS; k++) {
        /* mask = 1 - 0.7 * acc/acc_max, in Q15 (always > 0.3, so no clipping needed) */
        uint32_t cut  = (uint32_t)((acc_mag[k] * HR_MASK_Q15) / acc_max);
        uint32_t mask = 32768u - cut;
        ppg_mag[k] = (uint32_t)(((uint64_t)ppg_mag[k] * mask) >> 15);
    }
}

void hmm_init(hmm_t *h) { h->started = 0; }

int hmm_step(hmm_t *h, const uint32_t mag[HR_NBINS]) {
    uint64_t sum = 0;
    for (int k = 0; k < HR_NBINS; k++) sum += mag[k];

    int32_t logp[HR_NBINS];
    int32_t log_sum = sum ? log2_q16(sum) : 0;
    for (int k = 0; k < HR_NBINS; k++) {
        int32_t lp = mag[k] ? log2_q16(mag[k]) - log_sum : HR_HMM_LOG_FLOOR_Q16;
        logp[k] = lp < HR_HMM_LOG_FLOOR_Q16 ? HR_HMM_LOG_FLOOR_Q16 : lp;
    }

    return hmm_step_logp(h, logp);
}

int hmm_step_logp(hmm_t *h, const int32_t logp[HR_NBINS]) {
    int32_t best_score = INT32_MIN;
    int best = 0;
    if (!h->started) {
        for (int k = 0; k < HR_NBINS; k++) h->score[k] = logp[k];
        h->started = 1;
    } else {
        int32_t next[HR_NBINS];
        for (int j = 0; j < HR_NBINS; j++) {
            int32_t m = INT32_MIN;
            for (int d = -HR_HMM_MAX_JUMP; d <= HR_HMM_MAX_JUMP; d++) {
                int i = j - d;                     /* path comes from bin i into bin j */
                if (i < 0 || i >= HR_NBINS) continue;
                int32_t c = h->score[i] - HR_HMM_PENALTY_Q16 * (d < 0 ? -d : d);
                if (c > m) m = c;
            }
            next[j] = logp[j] + m;
        }
        for (int k = 0; k < HR_NBINS; k++) h->score[k] = next[k];
    }
    /* Renormalise (max -> 0) so scores stay bounded over hours of streaming. */
    for (int k = 0; k < HR_NBINS; k++)
        if (h->score[k] > best_score) { best_score = h->score[k]; best = k; }
    for (int k = 0; k < HR_NBINS; k++) {
        int32_t s = h->score[k] - best_score;
        h->score[k] = s < HR_HMM_SCORE_MIN ? HR_HMM_SCORE_MIN : s;
    }
    return best;
}
