/* Minimal Cortex-M startup: vector table + Reset_Handler. No libc. */
#include <stdint.h>

extern uint32_t _sidata, _sdata, _edata, _sbss, _ebss, _sstack, _estack;
int main(void);

void Reset_Handler(void);
void Default_Handler(void);
uint32_t stack_high_water(void);

#define STACK_PAINT 0xC0FFEE11u

void Default_Handler(void) {
    for (;;) { }
}

/* Bytes of stack ever used: the stack is painted at reset; count from the bottom
 * until the first overwritten word. */
uint32_t stack_high_water(void) {
    const uint32_t *p = &_sstack;
    while (p < &_estack && *p == STACK_PAINT) p++;
    return (uint32_t)((uintptr_t)&_estack - (uintptr_t)p);
}

void Reset_Handler(void) {
    uint32_t *src = &_sidata, *dst = &_sdata;
    while (dst < &_edata) *dst++ = *src++;          /* .data: flash -> RAM */
    for (dst = &_sbss; dst < &_ebss;) *dst++ = 0;   /* .bss: zero */
    /* Paint the stack below the current frame (leave 64 B for this function). */
    uint32_t *sp_now;
    __asm volatile("mov %0, sp" : "=r"(sp_now));
    for (dst = &_sstack; dst < sp_now - 16;) *dst++ = STACK_PAINT;
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
