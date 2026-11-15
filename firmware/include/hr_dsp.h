/* Fixed-point DSP blocks of the heart-rate pipeline. */
#ifndef HR_DSP_H
#define HR_DSP_H

#include <stdint.h>
#include "hr_config.h"

/* ---- Bandpass: cascade of Direct-Form-I biquads, Q29 coefs, 64-bit accumulator ---- */
typedef struct { int32_t x1, x2, y1, y2; } biquad_state_t;
typedef struct { biquad_state_t s[HR_BP_SECTIONS]; } bandpass_t;

void    bandpass_init(bandpass_t *f);
int32_t bandpass_step(bandpass_t *f, int32_t x);

/* ---- NLMS adaptive motion canceller ---- */
typedef struct {
    int32_t w[HR_NLMS_N];      /* weights, Q16 (PPG units per g)                    */
    int32_t hist[HR_NLMS_N];   /* reference history, tap-major, newest first, Q22 g */
} nlms_t;

void    nlms_init(nlms_t *f);
/* d: PPG sample (Q12). u: one ACC sample per axis (Q22). Returns cleaned PPG (Q12). */
int32_t nlms_step(nlms_t *f, int32_t d, const int32_t u[HR_NLMS_CH]);

/* ---- Spectrum of one 8 s window, band bins only ----
 * ring: circular buffer of HR_WIN samples, oldest at index `oldest`.
 * Removes the mean, applies the Hann window, scales the block to use the
 * available range (block floating point; `shift` from spectrum_block_shift), runs a
 * fixed-point FFT and writes |X[k]| for k = HR_BIN_LO..HR_BIN_HI to mag[0..HR_NBINS-1]. */
uint64_t spectrum_windowed_max(const int32_t *ring, uint32_t oldest, int32_t *mean_out);
int      spectrum_block_shift(uint64_t max_abs);
void     spectrum_band(const int32_t *ring, uint32_t oldest, int32_t mean, int shift,
                       uint32_t mag[HR_NBINS]);

/* In-place radix-2 FFT, HR_NFFT points, Q31 twiddles, scaled by 1/2 per stage. */
void fft_q31(int32_t *re, int32_t *im);

/* ---- Peak selection ---- */
int32_t log2_q16(uint64_t x);                     /* x > 0 */
void    acc_mask_apply(uint32_t ppg_mag[HR_NBINS], const uint64_t acc_mag[HR_NBINS]);

typedef struct { int32_t score[HR_NBINS]; uint8_t started; } hmm_t;
void hmm_init(hmm_t *h);
/* Feed one window's (masked) spectrum; returns the tracked bin index 0..HR_NBINS-1. */
int  hmm_step(hmm_t *h, const uint32_t mag[HR_NBINS]);

#endif
