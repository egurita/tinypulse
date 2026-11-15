/* MAX32652 SIL image: serial frames arrive on UART0, results go back on UART0. */
#include "board.h"
#include "sil_app.h"

int main(void) {
    uart_init();
    sil_init(uart_puts);
    for (;;) {
        int c = uart_getc_nonblock();
        if (c >= 0) sil_rx_byte((uint8_t)c);
    }
}
