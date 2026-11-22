/* MAX32652 SIL image: serial frames arrive on UART0, results go back on UART0.
 *
 * Handshake: until the first byte arrives, READY is repeated periodically, so a host
 * that attaches after boot still sees it (a single READY at power-up can be missed). */
#include "board.h"
#include "sil_app.h"

#define READY_EVERY 1000000u    /* idle polls between READY repeats (~0.1 s at 120 MHz) */

int main(void) {
    uart_init();
    sil_init(uart_puts);
    uint32_t idle = 0;
    int heard = 0;
    for (;;) {
        int c = uart_getc_nonblock();
        if (c >= 0) {
            heard = 1;
            sil_rx_byte((uint8_t)c);
        } else if (!heard && ++idle == READY_EVERY) {
            idle = 0;
            uart_puts("READY\n");
        }
    }
}
