#include "max30102_mock.h"
#include "max30102.h"

#define R(m, reg) ((m)->regs[MAX30102_REG_##reg])

static void por(max30102_mock_t *m) {
    for (int i = 0; i < 256; i++) m->regs[i] = 0;
    R(m, INTR_STATUS_1) = MAX30102_INT_PWR_RDY;
    R(m, REV_ID) = 0x03;
    R(m, PART_ID) = m->part_id;
    m->fifo_byte = 0;
    m->count = 0;
}

void mock_max30102_init(max30102_mock_t *m) {
    m->nack = 0;
    m->stuck_reset = 0;
    m->part_id = MAX30102_PART_ID_VALUE;
    m->transactions = 0;
    m->bytes_read = 0;
    por(m);
}

int mock_max30102_unread(const max30102_mock_t *m) { return m->count; }

void mock_max30102_push(max30102_mock_t *m, uint32_t red, uint32_t ir) {
    uint8_t mode = R(m, MODE_CONFIG);
    if ((mode & MAX30102_MODE_SHDN) || (mode & 0x07) == 0) return;   /* not sampling */

    /* Resolution from LED pulse width: 15..18 bits, left-justified in the 18-bit field. */
    unsigned bits = 15u + (R(m, SPO2_CONFIG) & 0x03);
    uint32_t keep = (0x3FFFFu >> (18 - bits)) << (18 - bits);
    red &= keep; ir &= keep;

    if (m->count == 32) {
        if (R(m, OVF_COUNTER) < 0x1F) R(m, OVF_COUNTER)++;
        if (!(R(m, FIFO_CONFIG) & MAX30102_FIFO_ROLLOVER)) return;   /* new sample lost */
        R(m, FIFO_RD_PTR) = (R(m, FIFO_RD_PTR) + 1) & 0x1F;           /* oldest overwritten */
        m->count--;
    }
    uint8_t *s = m->fifo[R(m, FIFO_WR_PTR)];
    uint32_t v[2] = {red, ir};
    for (int c = 0; c < 2; c++) {
        s[3 * c] = (uint8_t)(v[c] >> 16); s[3 * c + 1] = (uint8_t)(v[c] >> 8); s[3 * c + 2] = (uint8_t)v[c];
    }
    R(m, FIFO_WR_PTR) = (R(m, FIFO_WR_PTR) + 1) & 0x1F;
    m->count++;

    R(m, INTR_STATUS_1) |= MAX30102_INT_PPG_RDY;
    int free_slots = 32 - mock_max30102_unread(m);
    if (free_slots <= (R(m, FIFO_CONFIG) & 0x0F)) R(m, INTR_STATUS_1) |= MAX30102_INT_A_FULL;
}

static hal_status_t mock_write(void *ctx, uint8_t addr7, const uint8_t *data, size_t len) {
    max30102_mock_t *m = ctx;
    m->transactions++;
    if (m->nack || addr7 != MAX30102_I2C_ADDR || len == 0) return HAL_NACK;
    uint8_t reg = data[0];
    for (size_t i = 1; i < len; i++, reg++) {
        uint8_t v = data[i];
        switch (reg) {
        case MAX30102_REG_INTR_STATUS_1: case MAX30102_REG_INTR_STATUS_2:
        case MAX30102_REG_FIFO_DATA: case MAX30102_REG_REV_ID: case MAX30102_REG_PART_ID:
            break;                                              /* read-only */
        case MAX30102_REG_FIFO_WR_PTR: case MAX30102_REG_OVF_COUNTER: case MAX30102_REG_FIFO_RD_PTR:
            m->regs[reg] = v & 0x1F;
            m->fifo_byte = 0;
            m->count = (uint8_t)((R(m, FIFO_WR_PTR) - R(m, FIFO_RD_PTR)) & 0x1F);
            break;
        case MAX30102_REG_MODE_CONFIG:
            if (v & MAX30102_MODE_RESET) {
                por(m);
                if (m->stuck_reset) R(m, MODE_CONFIG) = MAX30102_MODE_RESET;
            } else {
                m->regs[reg] = v & 0x87;
            }
            break;
        default:
            m->regs[reg] = v;
        }
    }
    return HAL_OK;
}

static hal_status_t mock_read(void *ctx, uint8_t addr7, uint8_t reg, uint8_t *buf, size_t len) {
    max30102_mock_t *m = ctx;
    m->transactions++;
    if (m->nack || addr7 != MAX30102_I2C_ADDR) return HAL_NACK;
    m->bytes_read += (uint32_t)len;
    for (size_t i = 0; i < len; i++) {
        if (reg == MAX30102_REG_FIFO_DATA) {
            /* Burst stays on FIFO_DATA; the read pointer advances after each full sample. */
            int bytes_per_sample = (R(m, MODE_CONFIG) & 0x07) == MAX30102_MODE_HR ? 3 : 6;
            if (m->count == 0) { buf[i] = 0; continue; }
            buf[i] = m->fifo[R(m, FIFO_RD_PTR)][m->fifo_byte++];
            if (m->fifo_byte == bytes_per_sample) {
                m->fifo_byte = 0;
                R(m, FIFO_RD_PTR) = (R(m, FIFO_RD_PTR) + 1) & 0x1F;
                m->count--;
                R(m, OVF_COUNTER) = 0;          /* datasheet: cleared when a sample is popped */
            }
            continue;
        }
        buf[i] = m->regs[reg];
        if (reg == MAX30102_REG_INTR_STATUS_1 || reg == MAX30102_REG_INTR_STATUS_2)
            m->regs[reg] = 0;                   /* clear on read */
        reg++;
    }
    return HAL_OK;
}

static void mock_delay(void *ctx, uint32_t ms) { (void)ctx; (void)ms; }

hal_i2c_t mock_max30102_hal(max30102_mock_t *m) {
    hal_i2c_t h = {mock_write, mock_read, mock_delay, m};
    return h;
}
