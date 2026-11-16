/* INT8 inference of the spectral HR network (SpecCNN), integer-only.
 *
 * Input: 4 x 113 int8 (PPG and ACC spectra at each HR bin and at its 2nd harmonic,
 * normalised to 0..127 per window). Three dilated conv-ReLU layers (int8 weights,
 * per-channel requantisation), a 1x1 head and a per-bin prior -> 113 int32 logits.
 * Bit-exact with python/ppg_hr/quant.py::int_forward.
 */
#ifndef NN_H
#define NN_H

#include <stdint.h>
#include "hr_config.h"

#define NN_IN_CH 4

/* Build the int8 input from the two band spectra (bins HR_BIN_LO..HR_SPEC_HI). */
void nn_make_input(const uint32_t ppg_mag[HR_NSPEC], const uint64_t acc_mag[HR_NSPEC],
                   int8_t in[NN_IN_CH][HR_NBINS]);
/* in: NN_IN_CH x HR_NBINS int8, channel-major (flat, so callers needn't match qualifiers). */
void nn_forward(const int8_t *in, int32_t logits[HR_NBINS]);
/* Logits -> log2 likelihoods in Q16 for the HMM tracker. */
void nn_logits_to_logp(const int32_t logits[HR_NBINS], int32_t logp[HR_NBINS]);

#endif
