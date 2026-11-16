/* INT8 SpecCNN inference (see nn.h). Weights in gen/nn_model.c (generated). */
#include "nn.h"
#include "fixed.h"
#include "nn_model.h"

#define L HR_NBINS

/* Two ping-pong activation buffers: 2 x 16 x 113 B = 3.6 KB. */
static int8_t act_a[NN_CH * L], act_b[NN_CH * L];

void nn_make_input(const uint32_t ppg_mag[HR_NSPEC], const uint64_t acc_mag[HR_NSPEC],
                   int8_t in[NN_IN_CH][HR_NBINS]) {
    uint64_t mx_p = 1, mx_a = 1;
    for (int k = 0; k < HR_NSPEC; k++) {
        if (ppg_mag[k] > mx_p) mx_p = ppg_mag[k];
        if (acc_mag[k] > mx_a) mx_a = acc_mag[k];
    }
    for (int b = 0; b < L; b++) {
        int h = 16 + 2 * b;                     /* 2nd harmonic of HR bin b, same indexing */
        in[0][b] = (int8_t)(((uint64_t)ppg_mag[b] * 127 + mx_p / 2) / mx_p);
        in[1][b] = (int8_t)(((uint64_t)ppg_mag[h] * 127 + mx_p / 2) / mx_p);
        in[2][b] = (int8_t)((acc_mag[b] * 127 + mx_a / 2) / mx_a);
        in[3][b] = (int8_t)((acc_mag[h] * 127 + mx_a / 2) / mx_a);
    }
}

/* 'same' 1D convolution + bias + per-channel requantisation + ReLU, int8 -> int8.
 * Layout [channel][bin]. int32 accumulation cannot overflow: 16*5*127*127 < 2^21. */
static void conv_relu(const int8_t *in, int cin, int8_t *out, int cout, const int8_t *w,
                      const int32_t *bias, const int32_t *m0, int k, int dil) {
    int pad = dil * (k - 1) / 2;
    for (int co = 0; co < cout; co++) {
        for (int t = 0; t < L; t++) {
            int32_t acc = bias[co];
            for (int ci = 0; ci < cin; ci++) {
                const int8_t *wr = &w[(co * cin + ci) * k];
                const int8_t *xr = &in[ci * L];
                for (int j = 0; j < k; j++) {
                    int ti = t - pad + j * dil;
                    if (ti >= 0 && ti < L) acc += wr[j] * xr[ti];
                }
            }
            int64_t v = rshr64((int64_t)acc * m0[co], NN_REQ_SHIFT);
            out[co * L + t] = (int8_t)(v < 0 ? 0 : (v > 127 ? 127 : v));
        }
    }
}

void nn_forward(const int8_t *in, int32_t logits[HR_NBINS]) {
    conv_relu(in, NN_IN_CH, act_a, NN_CH, nn_w1, nn_b1, nn_m1, NN_K, NN_D1);
    conv_relu(act_a, NN_CH, act_b, NN_CH, nn_w2, nn_b2, nn_m2, NN_K, NN_D2);
    conv_relu(act_b, NN_CH, act_a, NN_CH, nn_w3, nn_b3, nn_m3, NN_K, NN_D3);
    for (int t = 0; t < L; t++) {
        int32_t acc = nn_bh[t];                 /* head bias + learned per-bin prior */
        for (int c = 0; c < NN_CH; c++) acc += nn_wh[c] * act_a[c * L + t];
        logits[t] = acc;
    }
}

void nn_logits_to_logp(const int32_t logits[HR_NBINS], int32_t logp[HR_NBINS]) {
    for (int t = 0; t < L; t++) {
        int64_t v = rshr64((int64_t)logits[t] * NN_L0, NN_LS);
        logp[t] = (int32_t)clamp64(v, -(1L << 28), 1L << 28);
    }
}
