/* Streaming heart-rate estimator: push one PPG sample at 64 Hz (plus an ACC sample
 * every other call, i.e. 32 Hz); every 2 s it produces one HR estimate.
 *
 * All state lives in hr_estimator_t (caller-owned, typically a static global):
 * no malloc, no hidden globals except the shared FFT scratch buffer.
 */
#ifndef HR_ESTIMATOR_H
#define HR_ESTIMATOR_H

#include <stdint.h>
#include "hr_dsp.h"

typedef struct {
    uint32_t window_index;   /* k: estimate for samples [128k, 128k + 512) */
    uint16_t bin;            /* 0..HR_NBINS-1 */
    uint32_t bpm_x1000;      /* HR in milli-BPM */
} hr_result_t;

typedef struct {
    bandpass_t ppg_bp;
    bandpass_t acc_bp[HR_NLMS_CH];
    nlms_t     nlms;
    hmm_t      hmm;
    int32_t    acc_prev[HR_NLMS_CH], acc_cur[HR_NLMS_CH];   /* Q22, for 32 -> 64 Hz */
    uint8_t    have_acc;
    uint32_t   n;                                         /* samples pushed */
    uint32_t   pos;                                       /* next ring write index */
    int32_t    ppg_ring[HR_WIN];                          /* cleaned PPG, Q12 */
    int32_t    acc_ring[HR_NLMS_CH][HR_WIN];              /* bandpassed ACC, Q22 */
} hr_estimator_t;

void hr_init(hr_estimator_t *st);

/* ppg_counts: 18-bit ADC value. acc: 3 axes in 1/64 g counts, or NULL when no new
 * ACC sample arrived this tick (32 Hz ACC = every other call).
 * Returns 1 and fills *out when a new estimate is ready, else 0. */
int hr_push(hr_estimator_t *st, uint32_t ppg_counts, const int16_t *acc, hr_result_t *out);

#endif
