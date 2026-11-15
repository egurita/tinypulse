/* Fixed-point helpers. All header-only and inlined.
 *
 * Assumes >> on negative signed integers is an arithmetic shift (true for GCC and
 * Clang on every target we build for; checked by a unit test).
 */
#ifndef FIXED_H
#define FIXED_H

#include <stdint.h>

static inline int32_t sat32(int64_t x) {
    if (x > INT32_MAX) return INT32_MAX;
    if (x < INT32_MIN) return INT32_MIN;
    return (int32_t)x;
}

/* Round-to-nearest right shift (plain >> rounds toward -inf, a systematic bias
 * that accumulates in recursive filters). s must be >= 1. */
static inline int64_t rshr64(int64_t x, unsigned s) {
    return (x + ((int64_t)1 << (s - 1))) >> s;
}

static inline int64_t clamp64(int64_t x, int64_t lo, int64_t hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

static inline uint32_t abs32(int32_t x) {
    return x < 0 ? (uint32_t)(-(int64_t)x) : (uint32_t)x;
}

/* floor(sqrt(x)) for 64-bit x: bit-by-bit, no division, no floats. */
static inline uint32_t isqrt64(uint64_t x) {
    uint64_t res = 0, bit = (uint64_t)1 << 62;
    while (bit > x) bit >>= 2;
    while (bit) {
        if (x >= res + bit) { x -= res + bit; res = (res >> 1) + bit; }
        else res >>= 1;
        bit >>= 2;
    }
    return (uint32_t)res;
}

#endif
