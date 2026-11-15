#include "ctest.h"

int ctest_checks = 0, ctest_failures = 0;

void run_dsp_tests(void);
void run_max30102_tests(void);

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("DSP (fixed point)\n");
    run_dsp_tests();
    printf("MAX30102 driver vs mock\n");
    run_max30102_tests();
    printf("%d checks, %d failures\n", ctest_checks, ctest_failures);
    return ctest_failures ? 1 : 0;
}
