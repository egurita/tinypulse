/* Windowed, zero-padded fixed-point FFT of one 8 s window -> band magnitudes.
 *
 * Block floating point: every later use of a spectrum is scale-invariant (argmax,
 * normalisation to probabilities, acc/acc_max), so each window is shifted to fill
 * ~28 bits before the FFT. That keeps full precision for quiet signals and no
 * overflow for loud ones, without per-sample saturation logic.
 */
#include "hr_dsp.h"
#include "fixed.h"

#define FFT_LOG2N 11
#if (1 << FFT_LOG2N) != HR_NFFT
#error "FFT_LOG2N does not match HR_NFFT"
#endif

/* Scratch buffer shared by all spectra (they are computed one after another).
 * 2 x 2048 x 4 B = 16 KB: the largest RAM item in the firmware. */
static int32_t fft_re[HR_NFFT], fft_im[HR_NFFT];

/* Windowed sample with the Hann window's 15 fractional bits KEPT (int64). Dropping
 * them here (>> 15) before the block scaling below lost precision on quiet signals;
 * caught by test_spectrum_matches_float_at_any_amplitude[0.01]. */
static inline int64_t windowed(const int32_t *ring, uint32_t oldest, int32_t mean, int i) {
    int32_t x = ring[(oldest + (uint32_t)i) % HR_WIN] - mean;
    return (int64_t)x * hr_hann_q15[i];
}

uint64_t spectrum_windowed_max(const int32_t *ring, uint32_t oldest, int32_t *mean_out) {
    int64_t sum = 0;
    for (int i = 0; i < HR_WIN; i++) sum += ring[i];
    int32_t mean = (int32_t)(sum >= 0 ? (sum + HR_WIN / 2) / HR_WIN : (sum - HR_WIN / 2) / HR_WIN);
    uint64_t mx = 0;
    for (int i = 0; i < HR_WIN; i++) {
        int64_t w = windowed(ring, oldest, mean, i);
        uint64_t a = (uint64_t)(w < 0 ? -w : w);
        if (a > mx) mx = a;
    }
    *mean_out = mean;
    return mx;
}

/* Left shift (negative = right shift) that brings max_abs into [2^27, 2^28). */
int spectrum_block_shift(uint64_t max_abs) {
    if (max_abs == 0) return 0;
    int msb = 63 - __builtin_clzll(max_abs);
    return 27 - msb;
}

void fft_q31(int32_t *re, int32_t *im) {
    /* Bit-reversal permutation. */
    for (uint32_t i = 1, j = 0; i < HR_NFFT; i++) {
        uint32_t bit = HR_NFFT >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            int32_t t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }
    /* Radix-2 decimation-in-time butterflies, each stage scaled by 1/2. */
    for (uint32_t size = 2; size <= HR_NFFT; size <<= 1) {
        uint32_t half = size >> 1, step = HR_NFFT / size;
        for (uint32_t start = 0; start < HR_NFFT; start += size) {
            for (uint32_t j = 0; j < half; j++) {
                int64_t wr = hr_twiddle_cos_q31[j * step], wi = hr_twiddle_sin_q31[j * step];
                uint32_t a = start + j, b = a + half;
                int64_t tr = rshr64(re[b] * wr - im[b] * wi, 31);
                int64_t ti = rshr64(re[b] * wi + im[b] * wr, 31);
                int64_t ar = re[a], ai = im[a];
                re[b] = (int32_t)rshr64(ar - tr, 1);  im[b] = (int32_t)rshr64(ai - ti, 1);
                re[a] = (int32_t)rshr64(ar + tr, 1);  im[a] = (int32_t)rshr64(ai + ti, 1);
            }
        }
    }
}

void spectrum_band(const int32_t *ring, uint32_t oldest, int32_t mean, int shift,
                   uint32_t *mag, int nbins) {
    for (int i = 0; i < HR_NFFT; i++) { fft_re[i] = 0; fft_im[i] = 0; }
    for (int i = 0; i < HR_WIN; i++) {
        int64_t x = windowed(ring, oldest, mean, i);
        fft_re[i] = (int32_t)(shift >= 0 ? x * ((int64_t)1 << shift) : rshr64(x, (unsigned)-shift));
    }
    fft_q31(fft_re, fft_im);
    for (int k = 0; k < nbins; k++) {
        int64_t r = fft_re[HR_BIN_LO + k], m = fft_im[HR_BIN_LO + k];
        mag[k] = isqrt64((uint64_t)(r * r) + (uint64_t)(m * m));
    }
}
