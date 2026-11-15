/* Register-level model of a MAX30102, behind the same hal_i2c_t the driver uses.
 *
 * Models the datasheet behaviour the driver depends on: 32-sample FIFO with
 * 5-bit write/read pointers, overflow counter (saturating, cleared when a sample
 * is popped), FIFO_DATA burst reads that do not auto-increment, clear-on-read
 * interrupt status, A_FULL threshold, soft reset, shutdown, ADC resolution set by
 * the LED pulse width. Samples are pushed by the test or by the dataset player.
 */
#ifndef MAX30102_MOCK_H
#define MAX30102_MOCK_H

#include <stdint.h>
#include "hal_i2c.h"

typedef struct {
    uint8_t  regs[256];
    uint8_t  fifo[32][6];
    uint8_t  fifo_byte;          /* byte offset inside the sample being burst-read */
    uint8_t  count;              /* samples stored: internal to the chip. With 5-bit
                                    pointers a full FIFO (32) has WR_PTR == RD_PTR,
                                    exactly like an empty one. */
    /* fault injection */
    uint8_t  nack;               /* 1: every transaction is NACKed */
    uint8_t  stuck_reset;        /* 1: RESET bit never clears */
    uint8_t  part_id;            /* value returned for PART_ID */
    /* statistics */
    uint32_t transactions;
    uint32_t bytes_read;
} max30102_mock_t;

void mock_max30102_init(max30102_mock_t *m);
/* Sensor produces one sample (called at the sample rate). Values are 18-bit. */
void mock_max30102_push(max30102_mock_t *m, uint32_t red, uint32_t ir);
int  mock_max30102_unread(const max30102_mock_t *m);
/* A hal_i2c_t whose ctx is the mock. */
hal_i2c_t mock_max30102_hal(max30102_mock_t *m);

#endif
