/* 0.5-4 Hz Butterworth bandpass: 4 biquads, Direct Form I, Q29 coefficients.
 *
 * Why Direct Form I with a 64-bit accumulator: the low cutoff (0.5 Hz at 64 Hz
 * sampling) puts poles very close to the unit circle (a2 ~ 0.97). In DF-II the
 * internal state can grow far beyond the input and overflow 32 bits; DF-I only
 * stores real inputs/outputs, and summing five 32x32-bit products in 64 bits
 * cannot overflow (|x| < 2^26, |coef| < 2^31 -> each product < 2^57).
 */
#include "hr_dsp.h"
#include "fixed.h"

void bandpass_init(bandpass_t *f) {
    for (int i = 0; i < HR_BP_SECTIONS; i++)
        f->s[i] = (biquad_state_t){0, 0, 0, 0};
}

int32_t bandpass_step(bandpass_t *f, int32_t x) {
    const int32_t *c = hr_bp_coef;
    for (int i = 0; i < HR_BP_SECTIONS; i++, c += 5) {
        biquad_state_t *s = &f->s[i];
        int64_t acc = (int64_t)c[0] * x + (int64_t)c[1] * s->x1 + (int64_t)c[2] * s->x2
                    - (int64_t)c[3] * s->y1 - (int64_t)c[4] * s->y2;
        int32_t y = sat32(rshr64(acc, HR_BP_COEF_Q));
        s->x2 = s->x1; s->x1 = x;
        s->y2 = s->y1; s->y1 = y;
        x = y;                       /* output of this section feeds the next */
    }
    return x;
}
