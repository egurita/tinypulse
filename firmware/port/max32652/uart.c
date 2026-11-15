/* Polled UART0 output and the benchmark marker.
 * Register offsets/bits from the MAX32650 SVD (UART0 @ 0x40042000). */
#include "board.h"

#define UART0_BASE   0x40042000u
#define UART_CTRL    (*(volatile uint32_t *)(UART0_BASE + 0x00))
#define UART_STAT    (*(volatile uint32_t *)(UART0_BASE + 0x08))
#define UART_BAUD0   (*(volatile uint32_t *)(UART0_BASE + 0x14))
#define UART_FIFO    (*(volatile uint32_t *)(UART0_BASE + 0x1C))
#define UART_CTRL_ENABLE   (1u << 0)
#define UART_CTRL_8BIT     (3u << 8)
#define UART_STAT_TX_FULL  (1u << 7)
#define UART_STAT_RX_EMPTY (1u << 4)

void uart_init(void) {
    UART_BAUD0 = 65;                       /* ~115200 baud from the peripheral clock */
    UART_CTRL = UART_CTRL_ENABLE | UART_CTRL_8BIT;
}

void uart_putc(char c) {
    while (UART_STAT & UART_STAT_TX_FULL) { }
    UART_FIFO = (uint8_t)c;
}

int uart_getc_nonblock(void) {
    if (UART_STAT & UART_STAT_RX_EMPTY) return -1;
    return (int)(UART_FIFO & 0xFF);
}

void uart_puts(const char *s) {
    while (*s) uart_putc(*s++);
}

void uart_putu(uint32_t v) {
    char buf[11];
    int i = 0;
    do { buf[i++] = (char)('0' + v % 10); v /= 10; } while (v);
    while (i) uart_putc(buf[--i]);
}

void uart_puti(int32_t v) {
    if (v < 0) { uart_putc('-'); uart_putu((uint32_t)(-(int64_t)v)); }
    else uart_putu((uint32_t)v);
}

/* Benchmark marker: Renode hooks this address and logs the CPU's executed-
 * instruction counter together with `id`. noinline so the address exists. */
__attribute__((noinline)) void bench_mark(uint32_t id) {
    __asm volatile("" : : "r"(id) : "memory");
}
