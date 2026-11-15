/* Minimal Cortex-M startup: vector table + Reset_Handler. No libc. */
#include <stdint.h>

extern uint32_t _sidata, _sdata, _edata, _sbss, _ebss, _estack;
int main(void);

void Reset_Handler(void);
void Default_Handler(void);

void Default_Handler(void) {
    for (;;) { }
}

void Reset_Handler(void) {
    uint32_t *src = &_sidata, *dst = &_sdata;
    while (dst < &_edata) *dst++ = *src++;          /* .data: flash -> RAM */
    for (dst = &_sbss; dst < &_ebss;) *dst++ = 0;   /* .bss: zero */
    main();
    for (;;) { }
}

/* Cortex-M core exceptions only; the firmware polls, so no peripheral IRQs. */
/* Entry 0 is the initial stack pointer (a data address), the rest are handlers,
 * so the table is declared as plain words. */
__attribute__((section(".isr_vector"), used))
const uintptr_t vector_table[16] = {
    (uintptr_t)&_estack,
    (uintptr_t)Reset_Handler,
    (uintptr_t)Default_Handler,   /* NMI */
    (uintptr_t)Default_Handler,   /* HardFault */
    (uintptr_t)Default_Handler,   /* MemManage */
    (uintptr_t)Default_Handler,   /* BusFault */
    (uintptr_t)Default_Handler,   /* UsageFault */
    0, 0, 0, 0,
    (uintptr_t)Default_Handler,   /* SVCall */
    (uintptr_t)Default_Handler,   /* DebugMon */
    0,
    (uintptr_t)Default_Handler,   /* PendSV */
    (uintptr_t)Default_Handler,   /* SysTick */
};
