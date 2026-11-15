/* Software-in-the-loop application: the host plays the sensor over a serial link.
 *
 * Host -> device frame (13 bytes, one per 64 Hz sample):
 *   [0]     0xA5 sync
 *   [1]     seq (u8, +1 per frame)
 *   [2]     flags: bit0 = ACC sample present, bit7 = end of stream
 *   [3..5]  PPG, u24 little endian (18-bit ADC counts)
 *   [6..11] ACC x, y, z: int16 little endian, 1/64 g
 *   [12]    XOR of bytes 1..11
 * Device -> host: text lines
 *   READY | HR <window> <bin> <milli-BPM> | ERR CHK <seq> | ERR SEQ <expected> <got>
 *   DONE rx=<frames> chk=<errors> seq=<errors> lost=<samples> est=<estimates>
 *
 * The received PPG goes into the mock MAX30102's FIFO and is read back through the
 * real driver, so the sensor driver is part of the loop.
 */
#ifndef SIL_APP_H
#define SIL_APP_H

#include <stdint.h>

#define SIL_SYNC        0xA5
#define SIL_FRAME_LEN   13
#define SIL_FLAG_ACC    0x01
#define SIL_FLAG_END    0x80

typedef void (*sil_write_fn)(const char *s);

void sil_init(sil_write_fn write_line);
void sil_rx_byte(uint8_t b);
int  sil_done(void);

#endif
