/* Minimal unit-test macros: no dependencies, works on host and (with a putchar) on target. */
#ifndef CTEST_H
#define CTEST_H

#include <stdio.h>

extern int ctest_checks, ctest_failures;

#define CHECK(cond) do { ctest_checks++; if (!(cond)) { ctest_failures++; \
    printf("    FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); } } while (0)

#define CHECK_EQ(a, b) do { long long _a = (long long)(a), _b = (long long)(b); ctest_checks++; \
    if (_a != _b) { ctest_failures++; \
    printf("    FAIL %s:%d: %s == %s (%lld != %lld)\n", __FILE__, __LINE__, #a, #b, _a, _b); } } while (0)

#define RUN(test) do { int _f = ctest_failures; test(); \
    printf("  %-48s %s\n", #test, ctest_failures == _f ? "ok" : "FAILED"); } while (0)

#endif
