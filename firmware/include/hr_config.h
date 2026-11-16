/* Heart-rate pipeline configuration: integer formats and tuned parameters.
 *
 * Mirrors python/ppg_hr/pipeline.py (FINAL config). Every real-valued parameter is
 * stored as an integer in a fixed-point "Q format": Qn means value = int / 2^n.
 *
 * Signal formats inside the pipeline:
 *   PPG   int32, Q12 in dataset PPG units   ((adc_counts - ADC_MID) << 8; 16 counts/unit)
 *   ACC   int32, Q22 in g                   (counts of 1/64 g << 16)
 */
#ifndef HR_CONFIG_H
#define HR_CONFIG_H

#include <stdint.h>
#include "hr_tables.h"

#define HR_FS_HZ            64
#define HR_HOP              128          /* 2 s shift between windows */
/* HR_WIN (512 = 8 s) and HR_NFFT (2048) come from hr_tables.h */

/* Sensor input formats */
#define HR_ADC_MID          (1L << 17)   /* 18-bit ADC mid-scale */
#define HR_PPG_SHIFT        8            /* counts -> Q12 units */
#define HR_ACC_SHIFT        16           /* 1/64 g counts -> Q22 g */

/* Search band 0.5-4 Hz = FFT bins 16..128 (bin width 64/2048 Hz = 1.875 BPM) */
#define HR_BIN_LO           16
#define HR_BIN_HI           128
#define HR_NBINS            (HR_BIN_HI - HR_BIN_LO + 1)
/* Spectrum computed up to 8 Hz (bin 256) so the NN also sees each HR bin's 2nd harmonic. */
#define HR_SPEC_HI          256
#define HR_NSPEC            (HR_SPEC_HI - HR_BIN_LO + 1)
#define HR_BPM_X1000_PER_BIN 1875

/* NLMS motion canceller: 16 taps x 3 axes, mu = 0.02, eps = 0.1 g^2 */
#define HR_NLMS_TAPS        16
#define HR_NLMS_CH          3
#define HR_NLMS_N           (HR_NLMS_TAPS * HR_NLMS_CH)
#define HR_NLMS_MU_Q16      1311                     /* round(0.02 * 2^16)      */
#define HR_NLMS_EPS_Q44     1759218604442LL          /* round(0.1  * 2^44)      */

/* ACC spectral mask strength 0.7 */
#define HR_MASK_Q15         22938                    /* round(0.7 * 2^15)       */

/* HMM tracker: max jump 2 bins/window, penalty 0.5 nats/bin = 0.7213 bits/bin */
#define HR_HMM_MAX_JUMP     2
#define HR_HMM_PENALTY_Q16  47274                    /* round(0.5/ln2 * 2^16)   */
#define HR_HMM_LOG_FLOOR_Q16 (-2612471L)             /* log2(1e-12) * 2^16      */
#define HR_HMM_SCORE_MIN    (-(1L << 30))

#endif
