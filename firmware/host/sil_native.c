/* Native SIL target: the firmware application on the host, talking over a serial
 * device (a pty created by the test harness) or stdin/stdout. */
#include <fcntl.h>
#include <string.h>
#include <unistd.h>
#include "sil_app.h"

static int fd_out = 1;

static void write_line(const char *s) {
    size_t n = strlen(s);
    while (n) {
        ssize_t w = write(fd_out, s, n);
        if (w <= 0) return;
        s += w; n -= (size_t)w;
    }
}

int main(int argc, char **argv) {
    int fd_in = 0;
    if (argc > 1) {                       /* serial device path */
        fd_in = fd_out = open(argv[1], O_RDWR | O_NOCTTY);
        if (fd_in < 0) return 2;
    }
    sil_init(write_line);
    uint8_t buf[256];
    while (!sil_done()) {
        ssize_t n = read(fd_in, buf, sizeof buf);
        if (n <= 0) break;
        for (ssize_t i = 0; i < n && !sil_done(); i++) sil_rx_byte(buf[i]);
    }
    return 0;
}
