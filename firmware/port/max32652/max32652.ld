/* Linker script for the MAX32652 (Cortex-M4F): 3 MB flash @ 0x10000000, 1 MB SRAM. */
ENTRY(Reset_Handler)

MEMORY
{
    FLASH (rx)  : ORIGIN = 0x10000000, LENGTH = 3M
    RAM   (rwx) : ORIGIN = 0x20000000, LENGTH = 1M
}

STACK_SIZE = 4K;

SECTIONS
{
    .isr_vector : { KEEP(*(.isr_vector)) } > FLASH

    .text : {
        *(.text*)
        *(.rodata*)
        . = ALIGN(4);
        _etext = .;
    } > FLASH

    /* Initialised data: stored in flash, copied to RAM by Reset_Handler. */
    .data : {
        _sdata = .;
        *(.data*)
        . = ALIGN(4);
        _edata = .;
    } > RAM AT > FLASH
    _sidata = LOADADDR(.data);

    /* Zero-initialised data (all our static state lives here). */
    .bss (NOLOAD) : {
        _sbss = .;
        *(.bss*) *(COMMON)
        . = ALIGN(4);
        _ebss = .;
    } > RAM

    /* Reserve the stack explicitly so --print-memory-usage counts it. */
    .stack (NOLOAD) : {
        . = ALIGN(8);
        . += STACK_SIZE;
        _estack = .;
    } > RAM

    /DISCARD/ : { *(.ARM.exidx*) *(.comment) }
}
