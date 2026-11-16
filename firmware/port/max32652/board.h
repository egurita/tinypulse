#ifndef BOARD_H
#define BOARD_H

#include <stdint.h>

void uart_init(void);
void uart_putc(char c);
int  uart_getc_nonblock(void);
void uart_puts(const char *s);
void uart_putu(uint32_t v);
void uart_puti(int32_t v);
void bench_mark(uint32_t id);
uint32_t stack_high_water(void);   /* startup.c: bytes of stack used so far */

#endif
