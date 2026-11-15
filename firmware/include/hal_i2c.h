/* Hardware abstraction layer for an I2C bus.
 *
 * The driver only talks to the bus through these function pointers, so the same
 * driver code runs against the real MCU peripheral, the mock sensor in unit tests,
 * or the mock inside the emulated firmware.
 */
#ifndef HAL_I2C_H
#define HAL_I2C_H

#include <stddef.h>
#include <stdint.h>

typedef enum { HAL_OK = 0, HAL_NACK = -1, HAL_BUS_ERROR = -2 } hal_status_t;

typedef struct {
    /* Write `len` bytes to device `addr7`; data[0] is the register address. */
    hal_status_t (*write)(void *ctx, uint8_t addr7, const uint8_t *data, size_t len);
    /* Write register address `reg`, repeated START, read `len` bytes into buf. */
    hal_status_t (*read)(void *ctx, uint8_t addr7, uint8_t reg, uint8_t *buf, size_t len);
    void (*delay_ms)(void *ctx, uint32_t ms);
    void *ctx;
} hal_i2c_t;

#endif
