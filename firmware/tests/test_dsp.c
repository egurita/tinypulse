/* Unit tests of the fixed-point blocks against double-precision math (host only). */
#include <math.h>
#include <stdlib.h>
#include "ctest.h"
#include "fixed.h"
#include "hr_estimator.h"

static const double PI = 3.14159265358979323846;

static void test_arithmetic_shift_assumption(void) {
    CHECK_EQ(-5 >> 1, -3);                 /* the code relies on arithmetic >> */
    CHECK_EQ(rshr64(-5, 1), -2);           /* round half up: -2.5 -> -2 */
    CHECK_EQ(rshr64(5, 1), 3);
}

static void test_isqrt_exact(void) {
    uint64_t v[] = {0, 1, 2, 3, 4, 15, 16, 17, 1000000, 0xFFFFFFFFull, 1ull << 59, (1ull << 62) - 1};
    for (unsigned i = 0; i < sizeof v / sizeof v[0]; i++) {
        uint64_t r = isqrt64(v[i]);
        CHECK(r * r <= v[i] && (r + 1) * (r + 1) > v[i]);
    }
    srand(1);
    for (int i = 0; i < 10000; i++) {
        uint64_t x = ((uint64_t)rand() << 31 ^ (uint64_t)rand()) & ((1ull << 59) - 1);
        uint64_t r = isqrt64(x);
        if (!(r * r <= x && (r + 1) * (r + 1) > x)) { CHECK(0); break; }
    }
}

static void test_log2_accuracy(void) {
    double worst = 0;
    for (uint64_t x = 1; x < (1ull << 50); x = x * 3 + 7) {
        double err = fabs(log2_q16(x) / 65536.0 - log2((double)x));
        if (err > worst) worst = err;
    }
    CHECK(worst < 2e-5);
}

static double run_bandpass_gain(double f_hz) {
    bandpass_t f;
    bandpass_init(&f);
    double in2 = 0, out2 = 0;
    for (int n = 0; n < 64 * 60; n++) {
        double x = sin(2 * PI * f_hz * n / HR_FS_HZ) * (1 << 20);
        int32_t y = bandpass_step(&f, (int32_t)lround(x));
        if (n >= 64 * 20) { in2 += x * x; out2 += (double)y * y; }   /* after the transient */
    }
    return sqrt(out2 / in2);
}

static void test_bandpass_passband_and_stopband(void) {
    CHECK(fabs(run_bandpass_gain(1.2) - 1.0) < 0.05);
    CHECK(fabs(run_bandpass_gain(2.5) - 1.0) < 0.05);
    CHECK(run_bandpass_gain(0.1) < 0.05);
    CHECK(run_bandpass_gain(10.0) < 0.05);
}

static void test_bandpass_full_scale_no_overflow(void) {
    /* Worst case input: full-scale square wave at the passband centre. */
    bandpass_t f;
    bandpass_init(&f);
    int32_t big = 1 << 25, mx = 0;
    for (int n = 0; n < 64 * 30; n++) {
        int32_t y = bandpass_step(&f, ((n / 16) % 2) ? big : -big);
        if (abs(y) > mx) mx = abs(y);
    }
    CHECK(mx < INT32_MAX / 4);
}

static void test_fft_single_tone(void) {
    static int32_t re[HR_NFFT], im[HR_NFFT];
    for (int i = 0; i < HR_NFFT; i++) {
        re[i] = (int32_t)lround(cos(2 * PI * 40 * i / HR_NFFT) * (1 << 27));
        im[i] = 0;
    }
    fft_q31(re, im);
    /* cos at bin 40: X[40] = N/2 * A, scaled by 1/N -> A/2 = 2^26 */
    CHECK(llabs((long long)re[40] - (1 << 26)) < 64);
    CHECK(abs(re[41]) < 64 && abs(im[40]) < 64);
}

static void test_nlms_without_motion_is_identity(void) {
    nlms_t f;
    nlms_init(&f);
    int32_t zero[3] = {0, 0, 0};
    int ok = 1;
    for (int n = 0; n < 1000; n++) {
        int32_t d = (int32_t)lround(sin(n * 0.1) * 100000);
        if (nlms_step(&f, d, zero) != d) ok = 0;
    }
    CHECK(ok);
}

static void test_estimator_window_timing_and_72bpm(void) {
    static hr_estimator_t st;
    hr_init(&st);
    hr_result_t r;
    int16_t acc[3] = {0, 0, 64};                 /* still wrist: 1 g on z */
    int first = -1, count = 0;
    uint32_t last_bpm = 0;
    for (int n = 0; n < 64 * 60; n++) {
        uint32_t ppg = (uint32_t)(HR_ADC_MID + lround(800 * sin(2 * PI * 1.2 * n / HR_FS_HZ)));
        if (hr_push(&st, ppg, (n % 2 == 0) ? acc : NULL, &r)) {
            if (first < 0) first = n;
            CHECK_EQ(r.window_index, (uint32_t)count);
            count++;
            last_bpm = r.bpm_x1000;
        }
    }
    CHECK_EQ(first, HR_WIN - 1);                 /* first estimate after 512 samples */
    CHECK_EQ(count, (64 * 60 - HR_WIN) / HR_HOP + 1);
    CHECK(labs((long)last_bpm - 72000) <= 1875); /* within one FFT bin of 72 BPM */
}

void run_dsp_tests(void) {
    RUN(test_arithmetic_shift_assumption);
    RUN(test_isqrt_exact);
    RUN(test_log2_accuracy);
    RUN(test_bandpass_passband_and_stopband);
    RUN(test_bandpass_full_scale_no_overflow);
    RUN(test_fft_single_tone);
    RUN(test_nlms_without_motion_is_identity);
    RUN(test_estimator_window_timing_and_72bpm);
}
