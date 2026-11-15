/* MAX30102 pulse-oximetry / heart-rate sensor driver (register map from the
 * Maxim/ADI MAX30102 datasheet). Talks to the chip only through hal_i2c_t.
 */
#ifndef MAX30102_H
#define MAX30102_H

#include <stdint.h>
#include "hal_i2c.h"

#define MAX30102_I2C_ADDR       0x57
#define MAX30102_PART_ID_VALUE  0x15
#define MAX30102_FIFO_DEPTH     32

/* Register map */
enum {
    MAX30102_REG_INTR_STATUS_1 = 0x00,  /* A_FULL[7] PPG_RDY[6] ALC_OVF[5] PWR_RDY[0], clear on read */
    MAX30102_REG_INTR_STATUS_2 = 0x01,  /* DIE_TEMP_RDY[1] */
    MAX30102_REG_INTR_ENABLE_1 = 0x02,
    MAX30102_REG_INTR_ENABLE_2 = 0x03,
    MAX30102_REG_FIFO_WR_PTR   = 0x04,  /* [4:0] */
    MAX30102_REG_OVF_COUNTER   = 0x05,  /* [4:0], saturates at 31 */
    MAX30102_REG_FIFO_RD_PTR   = 0x06,  /* [4:0] */
    MAX30102_REG_FIFO_DATA     = 0x07,  /* burst reads stay on this register */
    MAX30102_REG_FIFO_CONFIG   = 0x08,  /* SMP_AVE[7:5] ROLLOVER_EN[4] A_FULL[3:0] */
    MAX30102_REG_MODE_CONFIG   = 0x09,  /* SHDN[7] RESET[6] MODE[2:0] */
    MAX30102_REG_SPO2_CONFIG   = 0x0A,  /* ADC_RGE[6:5] SR[4:2] LED_PW[1:0] */
    MAX30102_REG_LED1_PA       = 0x0C,  /* red LED current, 0.2 mA/LSB */
    MAX30102_REG_LED2_PA       = 0x0D,  /* IR  LED current */
    MAX30102_REG_MULTI_LED_1   = 0x11,
    MAX30102_REG_MULTI_LED_2   = 0x12,
    MAX30102_REG_TEMP_INT      = 0x1F,
    MAX30102_REG_TEMP_FRAC     = 0x20,
    MAX30102_REG_TEMP_CONFIG   = 0x21,
    MAX30102_REG_REV_ID        = 0xFE,
    MAX30102_REG_PART_ID       = 0xFF,
};

#define MAX30102_INT_A_FULL     0x80
#define MAX30102_INT_PPG_RDY    0x40
#define MAX30102_INT_PWR_RDY    0x01
#define MAX30102_MODE_SHDN      0x80
#define MAX30102_MODE_RESET     0x40
#define MAX30102_MODE_HR        0x02     /* red only          */
#define MAX30102_MODE_SPO2      0x03     /* red + IR          */
#define MAX30102_FIFO_ROLLOVER  0x10

typedef enum { MAX30102_SR_50, MAX30102_SR_100, MAX30102_SR_200, MAX30102_SR_400,
               MAX30102_SR_800, MAX30102_SR_1000, MAX30102_SR_1600, MAX30102_SR_3200 } max30102_sr_t;
typedef enum { MAX30102_PW_69US_15BIT, MAX30102_PW_118US_16BIT,
               MAX30102_PW_215US_17BIT, MAX30102_PW_411US_18BIT } max30102_pw_t;
typedef enum { MAX30102_RANGE_2048NA, MAX30102_RANGE_4096NA,
               MAX30102_RANGE_8192NA, MAX30102_RANGE_16384NA } max30102_range_t;

typedef struct {
    max30102_sr_t    sample_rate;
    max30102_pw_t    pulse_width;
    max30102_range_t adc_range;
    uint8_t          sample_avg_log2;   /* 0..5: average 1..32 samples per FIFO entry */
    uint8_t          led_red_pa;        /* LED current, 0.2 mA per LSB */
    uint8_t          led_ir_pa;
    uint8_t          fifo_a_full;       /* 0..15: interrupt when this many slots remain free */
} max30102_config_t;

typedef enum {
    MAX30102_OK = 0,
    MAX30102_ERR_BUS = -1,       /* NACK / bus error */
    MAX30102_ERR_ID = -2,        /* wrong PART_ID: not a MAX30102 (or wrong address) */
    MAX30102_ERR_CONFIG = -3,    /* invalid combination in max30102_config_t */
    MAX30102_ERR_TIMEOUT = -4,   /* RESET bit never cleared */
    MAX30102_ERR_VERIFY = -5,    /* register read-back differs from what was written */
} max30102_status_t;

typedef struct { uint32_t red, ir; } max30102_sample_t;   /* 18-bit ADC counts */

typedef struct {
    const hal_i2c_t *hal;
    uint32_t samples_lost;       /* total samples dropped by FIFO overflow */
} max30102_t;

max30102_status_t max30102_init(max30102_t *dev, const hal_i2c_t *hal, const max30102_config_t *cfg);
/* Validate a config against the datasheet's sample-rate / pulse-width limits. */
max30102_status_t max30102_check_config(const max30102_config_t *cfg);
/* Read up to `max` samples from the FIFO. Returns the count (>= 0) or a negative error. */
int max30102_read_fifo(max30102_t *dev, max30102_sample_t *out, int max);
max30102_status_t max30102_set_shutdown(max30102_t *dev, int shutdown);
/* Reads and clears INTR_STATUS_1. */
int max30102_read_int_status(max30102_t *dev);

#endif
