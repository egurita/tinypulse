/* Normalized LMS motion canceller in fixed point.
 *
 * Float reference (python/ppg_hr/dsp.py::nlms):
 *     e  = d - w.u
 *     w += mu * e * u / (eps + u.u)
 *
 * Formats: d, e Q12 (PPG units); u Q22 (g); w Q16 (PPG units per g); u.u Q44.
 *   y   = sum(w*u) >> 26                       (Q16 * Q22 = Q38 -> Q12)
 *   dw  = mu*e*u/(eps+u.u) * 2^16  in Q16
 *       = (mu_q16*e) * u * 2^10 / den          (derivation in the Stage 2 report)
 * Computed as k = (mu_q16*e << 28) / den, then dw_i = (k * u_i) >> 18, so the one
 * 64-bit division per sample happens once, not once per tap.
 */
#include "hr_dsp.h"
#include "fixed.h"

#define ME_LIMIT (((int64_t)1) << 34)   /* keeps (me << 28) inside int64 */

void nlms_init(nlms_t *f) {
    for (int i = 0; i < HR_NLMS_N; i++) { f->w[i] = 0; f->hist[i] = 0; }
}

int32_t nlms_step(nlms_t *f, int32_t d, const int32_t u[HR_NLMS_CH]) {
    /* Shift the reference history by one tap (newest first), insert new sample. */
    for (int i = HR_NLMS_N - 1; i >= HR_NLMS_CH; i--) f->hist[i] = f->hist[i - HR_NLMS_CH];
    for (int c = 0; c < HR_NLMS_CH; c++) f->hist[c] = u[c];

    int64_t y = 0, uu = 0;
    for (int i = 0; i < HR_NLMS_N; i++) {
        y  += (int64_t)f->w[i] * f->hist[i];
        uu += (int64_t)f->hist[i] * f->hist[i];
    }
    int32_t e = sat32((int64_t)d - rshr64(y, 26));

    int64_t den = HR_NLMS_EPS_Q44 + uu;
    int64_t me  = clamp64((int64_t)HR_NLMS_MU_Q16 * e, -ME_LIMIT, ME_LIMIT);
    int64_t k   = (me * ((int64_t)1 << 28)) / den;
    for (int i = 0; i < HR_NLMS_N; i++)
        f->w[i] = sat32((int64_t)f->w[i] + rshr64(k * f->hist[i], 18));
    return e;
}
