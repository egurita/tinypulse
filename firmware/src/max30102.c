/* MAX30102 driver: configuration, FIFO reading, overflow accounting. */
#include "max30102.h"

#define RESET_POLL_MAX 10

static max30102_status_t wr(max30102_t *dev, uint8_t reg, uint8_t val) {
    uint8_t buf[2] = {reg, val};
    return dev->hal->write(dev->hal->ctx, MAX30102_I2C_ADDR, buf, 2) == HAL_OK ? MAX30102_OK
                                                                             : MAX30102_ERR_BUS;
}

static max30102_status_t rd(max30102_t *dev, uint8_t reg, uint8_t *buf, size_t len) {
    return dev->hal->read(dev->hal->ctx, MAX30102_I2C_ADDR, reg, buf, len) == HAL_OK ? MAX30102_OK
                                                                                    : MAX30102_ERR_BUS;
}

/* Write and read back: catches a write that the bus acknowledged but did not land. */
static max30102_status_t wr_verify(max30102_t *dev, uint8_t reg, uint8_t val) {
    uint8_t back;
    max30102_status_t s = wr(dev, reg, val);
    if (s == MAX30102_OK) s = rd(dev, reg, &back, 1);
    if (s == MAX30102_OK && back != val) s = MAX30102_ERR_VERIFY;
    return s;
}

max30102_status_t max30102_check_config(const max30102_config_t *cfg) {
    /* SpO2 mode: a longer LED pulse leaves less time per sample, so it caps the
     * sample rate (datasheet, SpO2 mode allowed-settings table). */
    static const uint8_t max_sr_for_pw[4] = {
        MAX30102_SR_1600, MAX30102_SR_1000, MAX30102_SR_800, MAX30102_SR_400};
    if (cfg->pulse_width > MAX30102_PW_411US_18BIT || cfg->sample_rate > MAX30102_SR_3200 ||
        cfg->adc_range > MAX30102_RANGE_16384NA || cfg->sample_avg_log2 > 5 || cfg->fifo_a_full > 15)
        return MAX30102_ERR_CONFIG;
    if (cfg->sample_rate > max_sr_for_pw[cfg->pulse_width]) return MAX30102_ERR_CONFIG;
    return MAX30102_OK;
}

max30102_status_t max30102_init(max30102_t *dev, const hal_i2c_t *hal, const max30102_config_t *cfg) {
    dev->hal = hal;
    dev->samples_lost = 0;
    max30102_status_t s = max30102_check_config(cfg);
    if (s != MAX30102_OK) return s;

    uint8_t id;
    if ((s = rd(dev, MAX30102_REG_PART_ID, &id, 1)) != MAX30102_OK) return s;
    if (id != MAX30102_PART_ID_VALUE) return MAX30102_ERR_ID;

    /* Soft reset, then wait for the RESET bit to self-clear. */
    if ((s = wr(dev, MAX30102_REG_MODE_CONFIG, MAX30102_MODE_RESET)) != MAX30102_OK) return s;
    uint8_t mode = MAX30102_MODE_RESET;
    for (int i = 0; i < RESET_POLL_MAX && (mode & MAX30102_MODE_RESET); i++) {
        hal->delay_ms(hal->ctx, 1);
        if ((s = rd(dev, MAX30102_REG_MODE_CONFIG, &mode, 1)) != MAX30102_OK) return s;
    }
    if (mode & MAX30102_MODE_RESET) return MAX30102_ERR_TIMEOUT;

    /* Empty the FIFO: write pointers, overflow counter and read pointer to 0. */
    uint8_t clr[4] = {MAX30102_REG_FIFO_WR_PTR, 0, 0, 0};
    if (hal->write(hal->ctx, MAX30102_I2C_ADDR, clr, sizeof clr) != HAL_OK) return MAX30102_ERR_BUS;

    const uint8_t fifo_cfg = (uint8_t)((cfg->sample_avg_log2 << 5) | (cfg->fifo_a_full & 0x0F));
    const uint8_t spo2_cfg = (uint8_t)((cfg->adc_range << 5) | (cfg->sample_rate << 2) | cfg->pulse_width);
    if ((s = wr_verify(dev, MAX30102_REG_FIFO_CONFIG, fifo_cfg)) != MAX30102_OK) return s;
    if ((s = wr_verify(dev, MAX30102_REG_SPO2_CONFIG, spo2_cfg)) != MAX30102_OK) return s;
    if ((s = wr_verify(dev, MAX30102_REG_LED1_PA, cfg->led_red_pa)) != MAX30102_OK) return s;
    if ((s = wr_verify(dev, MAX30102_REG_LED2_PA, cfg->led_ir_pa)) != MAX30102_OK) return s;
    if ((s = wr(dev, MAX30102_REG_INTR_ENABLE_1, MAX30102_INT_A_FULL)) != MAX30102_OK) return s;
    return wr_verify(dev, MAX30102_REG_MODE_CONFIG, MAX30102_MODE_SPO2);
}

int max30102_read_fifo(max30102_t *dev, max30102_sample_t *out, int max) {
    /* WR_PTR, OVF_COUNTER, RD_PTR are consecutive registers: one 3-byte burst. */
    uint8_t p[3];
    if (rd(dev, MAX30102_REG_FIFO_WR_PTR, p, 3) != MAX30102_OK) return MAX30102_ERR_BUS;
    uint8_t wr_ptr = p[0] & 0x1F, ovf = p[1] & 0x1F, rd_ptr = p[2] & 0x1F;

    int n = (wr_ptr - rd_ptr) & 0x1F;
    if (n == 0 && ovf) n = MAX30102_FIFO_DEPTH;      /* full FIFO: pointers are equal too */
    dev->samples_lost += ovf;                        /* counter clears when we read data */
    if (n > max) n = max;
    if (n == 0) return 0;

    /* SpO2 mode: 6 bytes per sample (red then IR), each 3 bytes MSB first, 18 valid bits. */
    uint8_t buf[MAX30102_FIFO_DEPTH * 6];
    if (rd(dev, MAX30102_REG_FIFO_DATA, buf, (size_t)n * 6) != MAX30102_OK) return MAX30102_ERR_BUS;
    for (int i = 0; i < n; i++) {
        const uint8_t *b = &buf[6 * i];
        out[i].red = (((uint32_t)b[0] << 16) | ((uint32_t)b[1] << 8) | b[2]) & 0x3FFFF;
        out[i].ir  = (((uint32_t)b[3] << 16) | ((uint32_t)b[4] << 8) | b[5]) & 0x3FFFF;
    }
    return n;
}

max30102_status_t max30102_set_shutdown(max30102_t *dev, int shutdown) {
    uint8_t mode;
    max30102_status_t s = rd(dev, MAX30102_REG_MODE_CONFIG, &mode, 1);
    if (s != MAX30102_OK) return s;
    mode = shutdown ? (uint8_t)(mode | MAX30102_MODE_SHDN) : (uint8_t)(mode & ~MAX30102_MODE_SHDN);
    return wr(dev, MAX30102_REG_MODE_CONFIG, mode);
}

int max30102_read_int_status(max30102_t *dev) {
    uint8_t v;
    return rd(dev, MAX30102_REG_INTR_STATUS_1, &v, 1) == MAX30102_OK ? v : MAX30102_ERR_BUS;
}
